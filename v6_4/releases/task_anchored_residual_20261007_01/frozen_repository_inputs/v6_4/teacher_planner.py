"""Finite multi-start teacher proposals, never cached-trace success labels.

The caller must execute and independently evaluate every selected reference
before admitting it as a demonstration. Failed planning attempts are returned
alongside accepted screening candidates and are never silently discarded.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from model_test.whole_body_verifier_v5 import WholeBodyCollisionVerifier, WholeBodyVerificationConfig
from v6_lite.hierarchical_qp import CONTINUUM_EE_OFFSET_M, joint_addresses, free_joint_slices
from v6_lite.visualization.generate_visualizations import _scenario_obstacles
from v6_4.contracts import TaskSpec, TrajectoryProposal
from v6_4.reference_adapter import SplineReferenceProvider, scenario_from_task
from v6_4.trajectory_codec import CubicBSplineCodec


@dataclass
class TeacherAttempt:
    attempt_index: int
    proposal: TrajectoryProposal | None
    controls: np.ndarray | None
    provider: SplineReferenceProvider | None
    screening_passed: bool
    metadata: dict

    def to_dict(self) -> dict:
        return {"attempt_index": self.attempt_index,
                "proposal": self.proposal.to_dict() if self.proposal is not None else None,
                "screening_passed": self.screening_passed, "metadata": self.metadata,
                "closed_loop_success_established": False, "demonstration_eligible": False}


def load_bootstrap(path: Path, *, source_task: TaskSpec | None = None) -> tuple[np.ndarray, np.ndarray, dict]:
    """An explicitly supplied old nominal trace supplies only a planning prior."""
    path = Path(path)
    repository=Path(__file__).resolve().parents[1]
    baseline=repository/"v6_lite/output/runs/research_acceptance_01"
    known=tuple((baseline/"simulation/traces"/f"v6_lite_scenario_{index:02d}.npz").resolve() for index in range(5))
    if path.resolve() in known:
        import json
        manifest_path=baseline/"manifest.json"
        if hashlib.sha256(manifest_path.read_bytes()).hexdigest() != "379b9977a441c540a20cfc50a9e90614a1e0985fa2a5a73bbb7c9690f823eec0":
            raise ValueError("frozen five-scene bootstrap manifest changed")
        manifest={key.replace('\\','/'):value for key,value in json.loads(manifest_path.read_text()).items()}
        if hashlib.sha256(path.read_bytes()).hexdigest()!=manifest[path.resolve().relative_to(baseline).as_posix()]:
            raise ValueError("frozen bootstrap trace changed")
        source_split,source_group="bootstrap",f"research_acceptance_01:{path.stem}"
    else:
        if source_task is None or source_task.split != "train":
            raise ValueError("new bootstrap source requires an explicit training TaskSpec; no val/test future trace")
        source_split,source_group="train",source_task.group_id
    with np.load(path, allow_pickle=False) as source:
        q = np.asarray(source["planner_q"], dtype=float).copy()
        times = np.asarray(source["time"], dtype=float).copy()
        if q.shape != (13500,17) or times.shape != (13500,):
            raise ValueError("bootstrap must be one complete 27s nominal trace, not a partial failure")
        # The initial 17-vector is supplied separately by the actual new task;
        # copied old post-integration states are never used as actual test state.
    return times, q, {"source":path.as_posix(),"sha256":hashlib.sha256(path.read_bytes()).hexdigest(),
                      "time_array_sha256":hashlib.sha256(times.tobytes()).hexdigest(),
                      "planner_q_array_sha256":hashlib.sha256(q.tobytes()).hexdigest(),
                      "source_split":source_split,"source_group_id":source_group,
                      "scope":"old nominal source used only as bootstrap trajectory prior",
                      "new_reference_closed_loop_success":False}


class TeacherPlanner:
    def __init__(self, spec, *, seed: int = 0, max_starts: int = 3,
                 optimization_rounds: int = 2, ik_max_evaluations: int = 35,
                 geometry_stride: int = 10, geometry_optimization_rounds: int = 8,
                 geometry_line_search_steps: int = 4) -> None:
        if (isinstance(seed,bool) or not isinstance(seed,int) or seed < 0
                or any(isinstance(x,bool) or not isinstance(x,int) or x < 1
                       for x in (max_starts,optimization_rounds,ik_max_evaluations,geometry_stride,geometry_line_search_steps))
                or isinstance(geometry_optimization_rounds,bool) or not isinstance(geometry_optimization_rounds,int)
                or not 0<=geometry_optimization_rounds<=32 or geometry_line_search_steps>4):
            raise ValueError("finite positive teacher budgets and a nonnegative integer seed required")
        self.spec, self.seed, self.max_starts = spec, seed, max_starts
        self.optimization_rounds, self.ik_max_evaluations, self.geometry_stride = optimization_rounds, ik_max_evaluations, geometry_stride
        self.geometry_optimization_rounds,self.geometry_line_search_steps=geometry_optimization_rounds,geometry_line_search_steps

    def _native_geometry(self, task, prediction, model, pairs):
        if not pairs:raise ValueError("teacher geometry screening requires the original nonempty pair policy")
        selected=np.unique(np.r_[np.arange(0,1351,self.geometry_stride),1350,
            [int(round(point.time_s/.02)) for point in task.requirements]]).astype(int)
        data=mujoco.MjData(model);witness=np.zeros(6);minimum=float('inf');queries=0;worst=None;state_witnesses=[]
        for index in selected:
            data.qpos[:],data.qvel[:],data.time=prediction['full_qpos'][index],prediction['full_qvel'][index],float(prediction['time'][index])
            mujoco.mj_forward(model,data)
            state_minimum=float('inf');state_worst=None
            for pair_index,pair in enumerate(pairs):
                distance=float(mujoco.mj_geomDistance(model,data,pair.geom_a,pair.geom_b,2.5,witness));queries+=1
                if not np.isfinite(distance):raise ValueError('teacher native geometry returned nonfinite distance')
                if distance<state_minimum:
                    state_minimum=distance;state_worst={'state_index':int(index),'time_s':float(prediction['time'][index]),
                        'pair_index':pair_index,'pair_class':pair.pair_class,'geom_a':pair.geom_a_name,'geom_b':pair.geom_b_name,
                        'body_a':pair.body_a_name,'body_b':pair.body_b_name,'raw_fromto_m':witness.tolist(),
                        'minimum_m':distance,'minimum_is_censored':distance>=2.5-1e-12}
                if distance<minimum:minimum=distance;worst=state_worst
            state_witnesses.append(state_worst)
        return {'status':'COMPLETED','minimum_m':minimum,'query_count':queries,'worst':worst,
                'state_witnesses':state_witnesses,
                'state_count':len(selected),'pair_count':len(pairs),'state_indices':selected.tolist(),'distance_max_m':2.5,
                'scope':'finite coarse private predicted configurations, original native pair policy; not execution or a continuous certificate'}

    def _geometry_gradient(self, model, predicted_qpos, pair):
        """Central native distance derivative in the local coupled 17D map.

        Only private qpos perturbations are integrated; full path/base holonomy
        is re-evaluated by the subsequent prediction/line search.
        """
        data=mujoco.MjData(model);data.qpos[:]=predicted_qpos;mujoco.mj_forward(model,data)
        _,dofs=joint_addresses(model,self.spec);_,base=free_joint_slices(model,self.spec.base_joint_name)
        bid=np.arange(base.start,base.stop);mass=np.empty((model.nv,model.nv));mujoco.mj_fullM(model,mass,data.qM)
        mapping=np.zeros((model.nv,17));mapping[dofs]=self.spec.planner_to_low_level
        mapping[bid]=-np.linalg.solve(mass[np.ix_(bid,bid)],mass[np.ix_(bid,dofs)]@self.spec.planner_to_low_level)
        epsilon=1e-5;gradient=np.zeros(17);witness=np.zeros(6)
        for coordinate in range(17):
            values=[]
            for sign in (-1.,1.):
                data.qpos[:]=predicted_qpos
                mujoco.mj_integratePos(model,data.qpos,mapping[:,coordinate],sign*epsilon)
                mujoco.mj_forward(model,data)
                values.append(float(mujoco.mj_geomDistance(model,data,pair.geom_a,pair.geom_b,2.5,witness)))
            gradient[coordinate]=(values[1]-values[0])/(2.*epsilon)
        if not np.all(np.isfinite(gradient)):raise ValueError('native geometry gradient became nonfinite')
        return gradient,{'native_distance_queries':34,'private_configuration_perturbations':34,'central_step_rad':epsilon,
                         'scope':'local zero-momentum coupled finite native distance derivative; not full-path total derivative'}

    @staticmethod
    def _geometry_projected_basis(codec,coordinate_anchor_matrices,time_s):
        """Project each joint only against that arm's required q anchors.

        A continuum task point does not require the rigid arm's redundant
        posture there. Both initial controls remain fixed by the codec.
        """
        free_basis=codec.basis(np.array([time_s]))[0,2:]
        projected=np.repeat(free_basis[:,None],17,axis=1)
        for coordinate,anchors in enumerate(coordinate_anchor_matrices):
            if len(anchors):
                projected[:,coordinate]-=anchors.T@np.linalg.solve(anchors@anchors.T,anchors@free_basis)
        return projected

    @staticmethod
    def _geometry_free_direction(codec,coordinate_anchor_matrices,time_s,gradient,deficit_m):
        return TeacherPlanner._geometry_joint_free_direction(codec,coordinate_anchor_matrices,
            np.array([time_s]),np.asarray(gradient)[None,:],np.array([deficit_m]))

    @staticmethod
    def _geometry_joint_free_direction(codec,coordinate_anchor_matrices,times,gradients,deficits):
        rows=np.array([(TeacherPlanner._geometry_projected_basis(codec,coordinate_anchor_matrices,time_s)
                        *gradient[None,:]).reshape(-1) for time_s,gradient in zip(times,gradients)])
        if np.max(np.sum(rows**2,axis=1))<1e-16:return None
        # One finite minimum-norm native linearization over at most eight
        # temporally distributed witnesses. Rows annihilate required-arm
        # anchors; this is an intrinsic planner update, not output repair.
        flat,_,_,_=np.linalg.lstsq(rows,deficits,rcond=1e-10)
        direction=flat.reshape(30,17)
        maximum=float(np.max(abs(direction)))
        # A trust-region bound on the optimizer step, never clipping a proposal.
        if maximum>.25:direction*=.25/maximum
        return direction

    @staticmethod
    def _geometry_active_witnesses(geometry):
        # Eight fixed equal time bins retain the worst active native pair in
        # each bin, addressing a long collision interval within one round.
        bins={}
        for witness in geometry['state_witnesses']:
            if witness['minimum_m']>=.007:continue
            bin_index=min(7,int(witness['time_s']/27.*8))
            if bin_index not in bins or witness['minimum_m']<bins[bin_index]['minimum_m']:
                bins[bin_index]=witness
        return sorted(bins.values(),key=lambda witness:(witness['minimum_m'],witness['state_index']))

    def _geometry_domains(self,codec,controls):
        if np.any(controls<self.spec.planner_lower) or np.any(controls>self.spec.planner_upper):return False
        sampled=codec.sample(controls,np.linspace(0.,27.,1351))
        return bool(np.all(sampled['q']>=self.spec.planner_lower)&np.all(sampled['q']<=self.spec.planner_upper)
            and np.all(abs(sampled['dq'])<=self.spec.planner_velocity_limits+1e-12)
            and np.all(abs(sampled['ddq'])<=self.spec.planner_acceleration_limits+1e-12))

    def _optimize_geometry(self,task,codec,controls,model,initial_data,scenario,pairs):
        anchor_times=np.array(sorted({point.time_s for point in task.requirements}))
        arm_anchor_times={arm:np.array(sorted({point.time_s for point in task.requirements if point.arm==arm}))
                          for arm in ('continuum','rigid')}
        arm_anchors={arm:codec.basis(times)[:,2:] for arm,times in arm_anchor_times.items()}
        for arm,anchors in arm_anchors.items():
            if len(anchors) and np.linalg.matrix_rank(anchors)!=len(anchors):
                raise ValueError(f'geometry {arm} anchor basis is rank deficient')
        coordinate_anchors=[arm_anchors['continuum']]*10+[arm_anchors['rigid']]*7
        logs=[];accepted=0;native_queries=0;predictions=0;perturbations=0
        provider=SplineReferenceProvider(task,controls).prepare(self.spec,model,initial_data,scenario);predictions+=1
        geometry=self._native_geometry(task,provider.prediction,model,pairs);native_queries+=geometry['query_count']
        for round_index in range(self.geometry_optimization_rounds):
            if geometry['minimum_m']>=.005:break
            active=self._geometry_active_witnesses(geometry);gradients=[];derivatives=[]
            for witness in active:
                gradient,derivative=self._geometry_gradient(model,provider.prediction['full_qpos'][witness['state_index']],
                                                            pairs[witness['pair_index']])
                gradients.append(gradient);derivatives.append(derivative)
                native_queries+=derivative['native_distance_queries'];perturbations+=derivative['private_configuration_perturbations']
            direction=self._geometry_joint_free_direction(codec,coordinate_anchors,
                np.array([witness['time_s'] for witness in active]),np.array(gradients),
                np.array([.007-witness['minimum_m'] for witness in active]))
            log={'round':round_index,'before':geometry,'active_witnesses':active,'native_gradients_17d':np.array(gradients).tolist(),
                 'derivatives':derivatives,'line_search':[],
                 'accepted':False,'target_optimizer_clearance_m':.007,'unchanged_acceptance_clearance_m':.005}
            if direction is None:
                log['termination']='DEGENERATE_PROJECTED_NATIVE_GRADIENT';logs.append(log);break
            free=codec.encode_free(controls)
            for line_index in range(self.geometry_line_search_steps):
                fraction=2.**(-line_index);candidate=codec.decode_free(free+fraction*direction)
                trial={'fraction':fraction,'domains_valid':self._geometry_domains(codec,candidate)}
                if not trial['domains_valid']:
                    trial['rejection']='DOMAIN_OR_RATE_LIMIT';log['line_search'].append(trial);continue
                next_provider=SplineReferenceProvider(task,candidate).prepare(self.spec,model,initial_data,scenario);predictions+=1
                next_geometry=self._native_geometry(task,next_provider.prediction,model,pairs);native_queries+=next_geometry['query_count']
                trial['geometry']=next_geometry
                if next_geometry['minimum_m']>geometry['minimum_m']+1e-9:
                    trial['accepted']=True;log['accepted']=True
                    delta=candidate-controls
                    log['required_arm_anchor_q_change_max_rad']=max(
                        [float(np.max(abs(codec.basis(arm_anchor_times[arm])@delta[:,coordinates])))
                         for arm,coordinates in [('continuum',slice(0,10)),('rigid',slice(10,17))]
                         if len(arm_anchor_times[arm])]+[0.])
                    log['all_arm_q_change_at_task_times_max_rad']=float(np.max(abs(codec.basis(anchor_times)@delta)))
                    controls,provider,geometry=candidate,next_provider,next_geometry;accepted+=1
                    log['line_search'].append(trial);break
                trial['accepted']=False;trial['rejection']='NO_GLOBAL_COARSE_MINIMUM_IMPROVEMENT';log['line_search'].append(trial)
            logs.append(log)
            if not log['accepted']:break
        return controls,{'rounds':logs,'accepted_steps':accepted,'native_distance_queries':native_queries,
            'private_prediction_count':predictions,'private_configuration_perturbations':perturbations,
            'before_task_anchor_refinement':geometry,'private_physics_steps':0,
            'required_arm_task_anchor_q_preserved_during_geometry':True,
            'active_witnesses_per_round_budget':8,'native_gradient_queries_per_round_budget':272,
            'active_witness_selection':'worst pair below 7mm in each of eight fixed equal 27s time bins',
            'required_arm_anchor_times_s':{arm:times.tolist() for arm,times in arm_anchor_times.items()},
            'unrequired_arm_q_free_at_other_arm_task_times':True,
            'world_task_pose_still_requires_coupled_anchor_refinement':True}

    def _environment(self, task, model, initial_data, pairs):
        scenario = scenario_from_task(task)
        if model is None:
            verifier = WholeBodyCollisionVerifier(self.spec,_scenario_obstacles(task.scenario),WholeBodyVerificationConfig(
                minimum_clearance=.005,query_distance_max=2.5,adaptive_subdivisions=4,
                self_collision_ancestor_exclusion_depth=3,include_target_satellite_pairs=True))
            model, pairs = verifier.model, verifier.pairs
            model.geom_contype[:], model.geom_conaffinity[:] = 0, 0
        if initial_data is None:
            initial_data = mujoco.MjData(model)
            initial_data.qpos[:], initial_data.qvel[:] = task.initial_qpos, task.initial_qvel
            initial_data.ctrl[:], initial_data.time = 0., 0.
            mujoco.mj_forward(model,initial_data)
        return scenario,model,initial_data,tuple(pairs or ())

    def _anchor_configuration(self, model, predicted_qpos, guess, points):
        """Finite bounded IK on a fixed private predicted base/target pose.

        The outer prediction updates the base after fitting these anchors.
        This local optimization is not a physical-feasibility certificate.
        """
        data = mujoco.MjData(model); data.qpos[:] = predicted_qpos; data.qvel[:] = 0.
        ids,_ = joint_addresses(model,self.spec)
        body_ids = {branch:mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,name)
                    for branch,name in (("rigid",self.spec.rigid_tip_body_name),("continuum",self.spec.continuum_tip_body_name))}
        target_body = mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,"target_satellite")
        if min([target_body,*body_ids.values()]) < 0:
            raise ValueError("teacher IK body mapping missing")
        guess = np.asarray(guess,dtype=float)
        if np.any(guess < self.spec.planner_lower) or np.any(guess > self.spec.planner_upper):
            raise ValueError("teacher anchor starts outside the unchanged planner domain")
        def residual(q):
            data.qpos[ids] = self.spec.encode_position(q)
            mujoco.mj_forward(model,data)
            target_r = data.xmat[target_body].reshape(3,3)
            target_p = data.xpos[target_body]
            values = []
            for point in points:
                body = body_ids[point.arm]
                actual_r = data.xmat[body].reshape(3,3)
                actual_p = data.xpos[body].copy()
                if point.arm == "continuum":
                    actual_p += actual_r @ CONTINUUM_EE_OFFSET_M
                desired_p,desired_r = np.asarray(point.position_m),np.asarray(point.rotation).reshape(3,3)
                if point.frame == "target":
                    desired_p,desired_r = target_p+target_r@desired_p,target_r@desired_r
                values.extend((actual_p-desired_p)/point.position_tolerance_m)
                values.extend(Rotation.from_matrix(desired_r@actual_r.T).as_rotvec()/point.orientation_tolerance_rad)
            # Preserve the prior's unrequired redundant shape with a small,
            # explicitly teacher-only regularizer, not a controller weight.
            values.extend(1e-4*(q-guess))
            return np.asarray(values)
        result = least_squares(residual,guess,bounds=(self.spec.planner_lower,self.spec.planner_upper),
                               max_nfev=self.ik_max_evaluations,ftol=1e-9,xtol=1e-9,gtol=1e-9)
        return result.x,{"optimizer_success":bool(result.success),"status":int(result.status),
                         "nfev":int(result.nfev),"normalized_task_residual_norm":float(np.linalg.norm(residual(result.x)[:-17]))}

    def _optimize(self, task, codec, controls, prior_times, prior_q, model, initial_data, scenario):
        groups = {}
        for point in task.requirements:
            groups.setdefault(point.time_s,[]).append(point)
        anchor_times = np.asarray(sorted(groups))
        anchor_basis = codec.basis(anchor_times)
        matrix = anchor_basis[:,2:]
        if np.linalg.matrix_rank(matrix) != len(anchor_times):
            raise ValueError("required task times exceed identifiable free anchor constraints")
        logs=[]
        for outer in range(self.optimization_rounds):
            provider=SplineReferenceProvider(task,controls).prepare(self.spec,model,initial_data,scenario)
            prediction=provider.prediction
            anchored_q=[]; ik=[]
            for time_s in anchor_times:
                index=int(round(time_s/.02))
                # Exact q at the requirement time; predicted base/target uses
                # its nearest declared 20ms forecast, with error retained.
                guess=codec.sample(controls,np.array([time_s]))["q"][0]
                value,report=self._anchor_configuration(model,prediction["full_qpos"][index],guess,groups[float(time_s)])
                anchored_q.append(value);ik.append({"time_s":float(time_s),"predicted_grid_time_s":float(prediction["time"][index]),**report})
            free=codec.encode_free(controls)
            required=np.asarray(anchored_q)-anchor_basis[:,:2]@codec.fixed_controls
            correction=matrix.T@np.linalg.solve(matrix@matrix.T,required-matrix@free)
            controls=codec.decode_free(free+correction)
            logs.append({"round":outer,"ik":ik,"free_control_correction_norm":float(np.linalg.norm(correction)),
                         "linear_anchor_residual_max":float(np.max(np.abs(anchor_basis@controls-np.asarray(anchored_q))))})
        return controls,logs

    def _screen(self, task, provider, model, pairs):
        from v6_4.proposal_gate import requirement_results
        prediction=provider.prediction
        q,dq,ddq=prediction["q"],prediction["dq"],prediction["ddq"]
        checks={"position_domain":bool(np.all(q>=self.spec.planner_lower)&np.all(q<=self.spec.planner_upper)),
                "velocity_domain":bool(np.all(np.abs(dq)<=self.spec.planner_velocity_limits+1e-12)),
                "acceleration_domain":bool(np.all(np.abs(ddq)<=self.spec.planner_acceleration_limits+1e-12))}
        requirements=requirement_results(task,prediction)
        checks["independent_task_requirements"]=requirements["passed"]
        geometry={"status":"NOT_RUN","minimum_m":None,"query_count":0,
                  "scope":"finite coarse private predicted configurations, original native pair policy; not execution or a continuous certificate"}
        if all(checks.values()):
            geometry=self._native_geometry(task,prediction,model,pairs)
            checks["coarse_native_geometry"]=geometry['minimum_m']>=.005
        else:
            checks["coarse_native_geometry"]=False
        return {"passed":all(checks.values()),"checks":checks,"requirements":requirements,"geometry":geometry,
                "closed_loop_success_established":False}

    def propose(self, task: TaskSpec, bootstrap_times: np.ndarray, bootstrap_q: np.ndarray, *,
                model=None, initial_data=None, pairs=None, bootstrap_metadata: dict | None = None) -> list[TeacherAttempt]:
        if task.split in ("val","test") and (bootstrap_metadata is None
                or bootstrap_metadata.get("source_split") not in ("train","bootstrap")
                or not bootstrap_metadata.get("source_group_id")
                or bootstrap_metadata.get("source_group_id")==task.group_id):
            raise ValueError("validation/test teacher needs a declared disjoint training/bootstrap prior, never its future actual trace")
        times=np.asarray(bootstrap_times,dtype=float);q=np.asarray(bootstrap_q,dtype=float)
        if (times.ndim!=1 or q.shape!=(len(times),17) or len(times)<32 or not np.all(np.isfinite(q))
                or not np.all(np.isfinite(times)) or np.any(np.diff(times)<=0)
                or times[0]<0 or abs(times[-1]-27.)>1e-8):
            raise ValueError("teacher prior must be a complete finite ordered 27s trajectory")
        if bootstrap_metadata is not None and (
                hashlib.sha256(times.tobytes()).hexdigest()!=bootstrap_metadata.get("time_array_sha256")
                or hashlib.sha256(q.tobytes()).hexdigest()!=bootstrap_metadata.get("planner_q_array_sha256")):
            raise ValueError("teacher prior arrays differ from the declared bootstrap source")
        codec=CubicBSplineCodec(task.initial_planner_q,task.initial_planner_dq)
        if times[0]>0:
            times=np.r_[0.,times];q=np.vstack([task.initial_planner_q,q])
        scenario,model,initial_data,pairs=self._environment(task,model,initial_data,pairs)
        rng=np.random.default_rng(self.seed);attempts=[]
        for attempt_index in range(self.max_starts):
            started=time.perf_counter();proposal=controls=provider=None
            metadata={"seed":self.seed,"method":"finite_multistart_shape_offset_spline_and_private_coupled_anchor_IK",
                      "bootstrap_source":dict(bootstrap_metadata) if bootstrap_metadata is not None else None,
                      "bootstrap_is_success_label":False,"optimizer_round_budget":self.optimization_rounds,
                      "ik_nfev_per_anchor_budget":self.ik_max_evaluations,"geometry_stride":self.geometry_stride,
                      "geometry_round_budget":self.geometry_optimization_rounds,"geometry_line_search_budget":self.geometry_line_search_steps,
                      "geometry_active_witnesses_per_round_budget":8,"geometry_native_gradient_queries_per_round_budget":272,
                      "geometry_control_step_max_abs_rad":.25,"geometry_target_clearance_m":.007,
                      "physics_steps_executed":0,"failure":None}
            try:
                # The zero-offset start is retained as the bootstrap comparator.
                offset=np.zeros(6) if attempt_index==0 else rng.uniform(-.018,.018,6)
                envelope=np.sin(np.pi*times/27.)**2
                for point in task.requirements:
                    envelope*=1.-np.exp(-.5*((times-point.time_s)/.40)**2)
                prior=q.copy();coordinates=np.array([0,2,4,6,8,11])
                prior[:,coordinates]+=envelope[:,None]*offset[None,:]
                controls=codec.fit(times,prior)
                controls,optimization=self._optimize(task,codec,controls,times,prior,model,initial_data,scenario)
                provider=SplineReferenceProvider(task,controls).prepare(self.spec,model,initial_data,scenario)
                screening=self._screen(task,provider,model,pairs)
                geometry_optimization=None
                if (self.geometry_optimization_rounds and screening['geometry']['status']=='COMPLETED'
                        and not screening['checks']['coarse_native_geometry']):
                    controls,geometry_optimization=self._optimize_geometry(task,codec,controls,model,initial_data,scenario,pairs)
                    if geometry_optimization['accepted_steps']:
                        # Geometry preserves required-arm q anchors; base holonomy can still
                        # move their world poses, so refine the unchanged tasks.
                        controls,refinement=self._optimize(task,codec,controls,times,prior,model,initial_data,scenario)
                        geometry_optimization['post_geometry_task_anchor_refinement']=refinement
                    provider=SplineReferenceProvider(task,controls).prepare(self.spec,model,initial_data,scenario)
                    screening=self._screen(task,provider,model,pairs)
                metadata.update({"path_offset_rad":offset.tolist(),"optimization":optimization,"screening":screening,
                                 "geometry_optimization":geometry_optimization,
                                 "intrinsic_teacher_operations":["constrained_spline_fit","private_coupled_task_anchor_optimization"],
                                 "prior_fit_rmse_rad":float(np.sqrt(np.mean((codec.sample(controls,times)["q"]-q)**2)))})
                proposal=TrajectoryProposal.from_controls(task,codec.encode_free(controls),origin="teacher",seed=self.seed+attempt_index,
                    postprocessing=(),metadata={"teacher_attempt":attempt_index,
                    "intrinsic_algorithm":"multistart_shape_offset_constrained_spline_fit_private_task_anchor_and_native_geometry_optimization",
                    "geometry_round_budget":self.geometry_optimization_rounds,"geometry_line_search_budget":self.geometry_line_search_steps,
                    "geometry_active_witnesses_per_round_budget":8,"geometry_native_gradient_queries_per_round_budget":272})
                if geometry_optimization is not None:metadata['intrinsic_teacher_operations'].append('finite_native_geometry_anchor_nullspace_optimization')
                passed=screening["passed"]
            except Exception as exc:
                passed=False;metadata["failure"]={"type":type(exc).__name__,"message":str(exc)}
            metadata["planning_wall_time_s"]=time.perf_counter()-started
            attempts.append(TeacherAttempt(attempt_index,proposal,controls,provider,bool(passed),metadata))
        return attempts
