"""Generate tasks without observing any competing planner's results."""
from dataclasses import asdict
import time
import mujoco
import numpy as np
from ff_prior.storage import digest
from ff_prior.scenes import SceneSpec, Obstacle
from ff_prior.robot_adapter import RobotAdapter
from ff_prior.initial_state import InitialState
from ff_prior.geometry import CollisionBackend
from ff_prior.spline import Reference
from ff_prior.contracts import frozen_task
from ff_prior.coupled_predictor import predict_coupled
from ff_prior.model_gates import evaluate_reference
from ff_prior.guidance import public_goal_initializer
from ff_prior.proposal_generator import quick_gates

CONTRACT = {'parents_per_split': 3, 'splits': ['TRAIN', 'VAL'],
    'kinds': ['larger_motion', 'static_detour'], 'scale': 1.5,
    'bow_amplitude_rad': .12, 'duration_s': 15., 'obstacle_radius_m': .012,
    'screen_candidates': 24, 'screen_stride': 8, 'screen_clearance_m': .003,
    'screen_phases': [.25, .375, .5, .625, .75], 'seed': 20261041,
    'selection': 'hash order; one parent per existing split family; no replacement',
    'planner_results_used': False, 'goal_relabel_after_freeze': False}


def select_parents(rows):
    selected = []
    for split in CONTRACT['splits']:
        pool = sorted((r for r,s in rows if s == split),
                      key=lambda r: digest({'complex_v1': r['sample_id']}))
        families = set()
        for row in pool:
            if row['family_id'] in families: continue
            selected.append((row, split)); families.add(row['family_id'])
            if len(families) == CONTRACT['parents_per_split']: break
        if len(families) != CONTRACT['parents_per_split']: raise ValueError('insufficient split families')
    return selected


def task_for(robot, initial, collision, profile, goals, duration):
    return frozen_task({'initial_features': robot.spec.planner_zero, 'goals': goals,
        'duration': duration, 'pose_mask': np.ones((2,6))},
        {'initial_state_id': initial.state_id, 'robot_hash': initial.model_hash,
         'pair_policy_hash': collision.policy_hash, 'acceptance': asdict(profile)})


def scene_runtime(scene):
    robot = RobotAdapter(scene=scene)
    initial = InitialState.capture(robot, robot.new_data())
    return robot, initial, CollisionBackend(robot)


def transfer_audit(parent, parent_initial, child, child_initial):
    """Explicitly permit only added static world geometry, never altered dynamics."""
    a,b = parent.model, child.model
    attrs = ('body_parentid','body_mass','body_inertia','body_ipos','body_iquat',
        'body_pos','body_quat','jnt_type','jnt_bodyid','jnt_pos','jnt_axis','jnt_range',
        'dof_armature','dof_damping','dof_frictionloss','actuator_trnid','actuator_gear',
        'actuator_gainprm','actuator_biasprm','actuator_ctrlrange','actuator_forcerange',
        'exclude_signature','qpos0','mesh_vert','mesh_face')
    checks = {n: np.array_equal(getattr(a,n),getattr(b,n)) for n in attrs}
    checks.update(dimensions=(a.nq,a.nv,a.nu,a.nbody,a.njnt)==(b.nq,b.nv,b.nu,b.nbody,b.njnt),
        integration_state=np.array_equal(parent_initial.values,child_initial.values),
        mapping=np.array_equal(parent.B,child.B),
        domain=np.array_equal(parent.lower,child.lower) and np.array_equal(parent.upper,child.upper),
        gravity=np.array_equal(a.opt.gravity,b.opt.gravity), timestep=a.opt.timestep==b.opt.timestep,
        integrator=a.opt.integrator==b.opt.integrator)
    geom_attrs = ('geom_type','geom_bodyid','geom_size','geom_pos','geom_quat',
        'geom_contype','geom_conaffinity','geom_friction','geom_solref','geom_solimp','geom_margin','geom_gap','geom_dataid')
    names = set()
    for i in range(a.ngeom):
        name = mujoco.mj_id2name(a,mujoco.mjtObj.mjOBJ_GEOM,i); names.add(name)
        j = mujoco.mj_name2id(b,mujoco.mjtObj.mjOBJ_GEOM,name)
        checks['geom:'+name] = j >= 0 and all(np.array_equal(getattr(a,n)[i],getattr(b,n)[j]) for n in geom_attrs)
    added = [i for i in range(b.ngeom) if mujoco.mj_id2name(b,mujoco.mjtObj.mjOBJ_GEOM,i) not in names]
    checks['added_static_world_only'] = all(b.geom_bodyid[i]==0 and b.geom_contype[i] and b.geom_conaffinity[i] for i in added)
    if not all(checks.values()): raise ValueError('scene transfer changed robot: '+str([k for k,v in checks.items() if not v]))
    return {'status':'PASS','checks':checks,'checkpoint_robot_hash':parent_initial.model_hash,
        'checkpoint_initial_state_id':parent_initial.state_id,'scene_robot_hash':child_initial.model_hash,
        'scene_initial_state_id':child_initial.state_id,'added_world_geoms':len(added),
        'network_obstacle_conditioned':False,'scope':'static-scene transfer of unchanged robot prior; full new-scene validation required'}


def witness_z(original, q0, kind, seed):
    z = q0 + CONTRACT['scale']*(np.asarray(original)-q0)
    if kind == 'static_detour':
        s = np.r_[np.linspace(0,1,20)[1:-1],1.]
        direction = np.random.default_rng(seed).normal(size=17)
        direction /= max(abs(direction)); direction[:10] *= .5
        z += CONTRACT['bow_amplitude_rad']*np.sin(np.pi*s)[:,None]*direction
    return z


def choose_obstacle(robot, initial, collision, profile, witness, prediction, family, deadline):
    """Scratch scene screening only. Final immutable scene gets independent full gates."""
    task = task_for(robot,initial,collision,profile,prediction['ee'][-1],witness.duration)
    base = public_goal_initializer(task,robot,initial)
    direct = predict_coupled(initial,Reference(base,witness.q0,witness.duration),robot,128,deadline)
    scratch = RobotAdapter(scene=SceneSpec((Obstacle('screen','sphere',(10.,10.,10.),(.012,0.,0.)),),'screen'))
    backend = CollisionBackend(scratch); data = scratch.new_data()
    oid = mujoco.mj_name2id(scratch.model,mujoco.mjtObj.mjOBJ_GEOM,'v7_screen')
    candidates=[]
    for phase in CONTRACT['screen_phases']:
        ti=int(phase*128)
        for gi in collision.geoms:
            if robot.model.body_weldid[robot.model.geom_bodyid[gi]] in (0,robot.base_body): continue
            center=direct['geom'][ti,gi]
            points=[center]
            mesh=int(robot.model.geom_dataid[gi])
            if robot.model.geom_type[gi]==mujoco.mjtGeom.mjGEOM_MESH:
                start=int(robot.model.mesh_vertadr[mesh]); count=int(robot.model.mesh_vertnum[mesh])
                vertices=robot.model.mesh_vert[start:start+count]
                extremes=np.r_[vertices.argmax(axis=0),vertices.argmin(axis=0)]
                rotation=direct['geom_rotation'][ti,gi].reshape(3,3)
                points=[center+rotation@vertices[j] for j in extremes]
            for point in points:
                # Conservative ranking only: no acceptance claim from bounding spheres.
                lower=np.linalg.norm(prediction['geom'][::8]-point,axis=-1)-robot.model.geom_rbound-.012
                candidates.append((float(lower.min()),ti,gi,point.copy()))
    candidates.sort(key=lambda x:(-x[0],x[1],x[2]))
    logs=[]; selected=None
    for rank,(_,ti,gi,center) in enumerate(candidates[:CONTRACT['screen_candidates']]):
        if time.time()>=deadline: raise RuntimeError('BUDGET_EXHAUSTED')
        scratch.model.geom_pos[oid]=center
        data.qpos[:]=direct['qpos'][ti]; scratch.kinematics(data)
        blocked=backend.scan(data,0.)
        minimum=2.5; checked=0
        if blocked['minimum_clearance_m'] < -.0001:
            for state in prediction['qpos'][::CONTRACT['screen_stride']]:
                data.qpos[:]=state; scratch.kinematics(data); r=backend.scan(data,CONTRACT['screen_clearance_m'])
                minimum=min(minimum,r['minimum_clearance_m']); checked+=1
                if not r['passed']: break
        passed=blocked['minimum_clearance_m'] < -.0001 and checked==len(prediction['qpos'][::8]) and minimum>=.003
        logs.append({'rank':rank,'phase':ti/128,'geom':mujoco.mj_id2name(robot.model,mujoco.mjtObj.mjOBJ_GEOM,gi),
            'position':center,'direct_clearance_m':blocked['minimum_clearance_m'],
            'witness_screen_clearance_m':minimum,'screen_states':checked,'passed':passed})
        if passed:
            selected=SceneSpec((Obstacle('detour','sphere',tuple(center),(.012,0.,0.)),),family); break
    return selected, {'attempts':logs,'scratch_calls':scratch.calls,'public_reference_hash':digest(base),
        'MODEL_VALID_claimed':False,'method_results_used':False,'screen_only':True}


def generate(row,split,kind,index,source,robot,initial,collision,profile,deadline):
    sid=f'complex_{split.lower()}_{index:02d}_{kind}'
    with np.load(source/row['arrays']) as a: original=a['Z'].copy()
    z=witness_z(original,robot.spec.planner_zero,kind,CONTRACT['seed']+index)
    ref=Reference(z,robot.spec.planner_zero,CONTRACT['duration_s'])
    result={'sample_id':sid,'split':split,'kind':kind,'parent_sample_id':row['sample_id'],
        'parent_family_id':row['family_id'],'family_id':row['family_id'],
        'training_eligible':False,'status':'BLOCKED','Z':z,'reference_hash':ref.reference_hash,
        'duration':ref.duration,'method_results_used':False,'goal_relabelled':False}
    prediction,validation=evaluate_reference(robot,initial,ref,collision,profile,deadline)
    result['base_witness_validation']=validation
    if validation['validation_level']!='MODEL_VALID':
        result['reason']='INDEPENDENT_WITNESS_FAILED_BASE_MODEL_GATES'; return result
    scene=SceneSpec()
    if kind=='static_detour':
        scene,screen=choose_obstacle(robot,initial,collision,profile,ref,prediction,sid,deadline)
        result['screen']=screen
        if scene is None:
            result['reason']='NO_VALID_SCENE_IN_FROZEN_24_ATTEMPTS'; return result
    # Immutable world target is derived exactly once, before all competing candidates.
    result.update(status='GENERATED',scene=scene.to_dict(),scene_hash=scene.scene_hash,
        goals=prediction['ee'][-1],initial_features=ref.q0,
        maximum_control_motion_rad=float(np.max(abs(z-ref.q0))),
        parent_maximum_control_motion_rad=float(np.max(abs(original-ref.q0))),
        base_translation_m=float(np.max(np.linalg.norm(prediction['base'][:,:3]-prediction['base'][0,:3],axis=1))))
    return result
