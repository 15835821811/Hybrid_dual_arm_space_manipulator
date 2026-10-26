"""Initialize exactly once from saved terminal INTEGRATION; never alter actual state in loop."""
from types import SimpleNamespace
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation
from tracking_diffusion.environment import free_slices,Task
from ff_prior.initial_state import InitialState
from ff_prior.geometry import CollisionBackend
from ff_prior_dynamic_handoff.execution import Environment as Controller
from ff_prior_dynamic_reference.execution import integration,RecordedCollision
from v6_lite.irregular_waypoints import IrregularWaypointTarget
from .protocol import P

class Environment(Controller):
    def __init__(self,robot,terminal,contract,budget):
        self.robot=robot;self.model=robot.model;self.spec=robot.spec;self.budget=budget
        self.initial=InitialState(np.asarray(terminal['integration']).copy(),contract['scene_model_hash'])
        self.data=self.initial.reset(robot)
        if not np.array_equal(integration(robot,self.data),terminal['integration']):
            raise ValueError('TERMINAL_FULL_STATE_MISMATCH')
        self.origin_time=float(self.data.time);self.contract=contract
        self.task=Task(seed=P['seed'],obstacle_count=0)
        self.qids,self.vids=robot.qpos_ids,robot.dof_ids
        self.bp,self.bv=free_slices(self.model,self.spec.base_joint_name)
        self.tp,self.tv=free_slices(self.model,self.spec.target_free_joint_name)
        self.base_ids=robot.base_dofs
        self.body_ids=[robot.tip_c,robot.tip_r,robot.id(mujoco.mjtObj.mjOBJ_BODY,'target_satellite')]
        w=contract['waypoints'];goals=np.asarray(contract['static_goals'])
        self.path=IrregularWaypointTarget(np.asarray(w['initial_position_w']),
            np.asarray(w['waypoint_points_m']),np.asarray(w['target_rotation_world']),
            w['transition_duration_s'],w['path_duration_s'],np.asarray(w['segment_durations_s']),w)
        self.start_rotation=goals[0,:3,:3]
        self.rotation_bridge=Rotation.from_matrix(self.path.target_rotation_world@self.start_rotation.T).as_rotvec()
        self.grasp_local=np.asarray(contract['grasp_local'])
        self.grasp_rotation_local=np.asarray(contract['grasp_rotation_local'])
        self.mass=np.zeros((self.model.nv,self.model.nv))
        self.reference_q=np.asarray(terminal['reference_q']).copy()
        self.command=np.asarray(terminal['reference_qd']).copy()
        cc=CollisionBackend(robot);self.collision=RecordedCollision(cc)
        self.pairs=[SimpleNamespace(geom_a=int(a),geom_b=int(b),pair_class='V7_no_exempt',
            geom_a_name=cc.describe(a,b)['geom_a'],geom_b_name=cc.describe(a,b)['geom_b']) for a,b in cc.pairs]
        self.pair_a,self.pair_b=cc.pairs.T
        self.safe_distances=np.full(len(self.pairs),.001)
        self.minimum_clearance=self.minimum_margin=float('inf');self.minimum_pair=None
        self.qp_failures=0;self.last_filter_delta=0.;self.scratch=robot.new_data()
        self.history={k:[] for k in ('time','qpos','qvel','ee','desired','base','q17','qref','clearance','momentum','exchange','skeleton_c','skeleton_r')}
        self.controls=[];self.raw=[];self.command_qd=[];self.command_qdd=[]
        self.integration_history=[];self.filter_history=[]
        self.max_geom_step=0.;self.interpolation_states=0
        self.previous_qpos=self.previous_geom=self.previous_geom_r=None
        self.record()

    def reference(self,t):
        local=float(t-self.origin_time);pc,vc=self.path.sample(local)
        s=np.clip(local/4.5,0.,1.);alpha=s**3*(10-15*s+6*s*s)
        rate=30*s*s*(1-s)**2/4.5 if 0<=local<4.5 else 0.
        rc=Rotation.from_rotvec(alpha*self.rotation_bridge).as_matrix()@self.start_rotation
        wc=rate*self.rotation_bridge
        pt,rt=self.pose(2);lead=t-self.data.time
        v=self.data.qvel[self.tv.start:self.tv.start+3];w=self.data.qvel[self.tv.start+3:self.tv.stop]
        future=rt@Rotation.from_rotvec(w*lead).as_matrix()
        pr=pt+v*lead+future@self.grasp_local
        vr=v+np.cross(future@w,future@self.grasp_local)
        return pc,rc,vc,wc,pr,future@self.grasp_rotation_local,vr,future@w

    def advance_native(self,velocity):
        r,d=self.robot,self.data;old=self.command.copy();acceleration=(velocity-old)/.02
        for substep in range(10):
            v=old+(substep+1)/10*(velocity-old);self.reference_q+=.002*v
            r.forward(d);mass=r.mass(d);w=np.r_[np.full(60,42.),np.full(7,34.)]
            aref=r.B@acceleration+w*w*(r.B@self.reference_q-d.qpos[r.qpos_ids])+2*w*(r.B@v-d.qvel[r.dof_ids])
            cap=np.r_[np.full(60,45.),np.full(7,70.)];aref=np.clip(aref,-cap,cap)
            b,a=r.base_dofs,r.dof_ids;force=d.qfrc_bias-d.qfrc_passive
            ba=-np.linalg.solve(mass[np.ix_(b,b)],mass[np.ix_(b,a)]@aref+force[b])
            raw=mass[np.ix_(a,b)]@ba+mass[np.ix_(a,a)]@aref+force[a]
            if not np.isfinite(raw).all():raise RuntimeError('NONFINITE_TORQUE')
            d.ctrl[r.actuator_ids]=np.clip(raw,-r.spec.torque_limits,r.spec.torque_limits)
            self.controls.append(d.ctrl.copy());self.raw.append(raw.copy())
            self.command_qd.append(v.copy());self.command_qdd.append(acceleration.copy())
            self.budget.step('native')
            mujoco.mj_step(r.model,d)
            self.record()
            if np.any(d.warning.number):raise RuntimeError('MUJOCO_WARNING')
        self.command=velocity.copy()
