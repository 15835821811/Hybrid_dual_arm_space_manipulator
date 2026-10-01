"""Independent scalar verification of three frozen linear infeasibilities.

No producer/LP diagnostic function, optimizer, geometry query, or physics step
is called. Only this new ignored directory receives verification products.
"""
from pathlib import Path
from fractions import Fraction
from datetime import datetime, timezone
import hashlib, json, math, subprocess, sys, traceback
import numpy as np
ROOT=Path(__file__).resolve().parents[4]
OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
import mujoco
from model_test.robot_model_spec_v5 import RobotModelSpecV5, default_robot_model_spec_v5
from model_test.whole_body_verifier_v5 import WholeBodyCollisionVerifier, WholeBodyVerificationConfig, WorkspaceSphere
TRIAL=ROOT/'v6_lite/output/runs/research_velocity_stress_01'
LP=ROOT/'v6_lite/output/runs/research_velocity_failure_feasibility_02'
COMMIT='9c7120e8ca8711cd58dc65471e58763c965148ff'
CASES=['v6_lite_scenario_00','v6_lite_scenario_03','v6_lite_scenario_04']
CHECKS=[]
INPUTS={}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    path=Path(path)
    INPUTS[str(path.resolve())]={'sha256':sha(path),'bytes':path.stat().st_size}
    return json.loads(path.read_text(encoding='utf-8'),parse_constant=lambda s:(_ for _ in ()).throw(ValueError(s)))


def check(name,value):
    CHECKS.append({'name':name,'passed':bool(value)})
    if not value:
        raise ValueError('independent scalar check failed: '+name)


def write(path,value):
    with Path(path).open('x',encoding='utf-8',newline='\n') as f:
        json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')


def run():
    check('source_head',subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()==COMMIT)
    plan=load(TRIAL/'plan.json'); metadata=load(TRIAL/'simulation/run_metadata.json')
    lp_report=load(LP/'report.json'); lp_manifest=load(LP/'manifest.json')
    check('formal_lp_02_valid_not_failed_01',lp_report['evidence_valid'] is True
        and lp_report['diagnostic_execution_completed'] is True and lp_report['trial_produced_commit']==COMMIT)
    actual={p.name for p in LP.iterdir() if p.is_file() and p.name!='manifest.json'}
    check('lp_manifest_file_set',actual==set(lp_manifest))
    for name,record in lp_manifest.items():
        path=LP/name
        check('lp_manifest:'+name,sha(path)==record['sha256'] and path.stat().st_size==record['size_bytes'])
        INPUTS[str(path.resolve())]={'sha256':sha(path),'bytes':path.stat().st_size}
    for relative in ('model_test/robot_model_spec_v5.py','model_test/angle_convention.py',
        'v6_lite/continuum_model_spec.py','v6_lite/execution_ramp.py','v6_lite/b2_interval_online.py',
        'v6_lite/b2_interval_online_optimized.py','model_test/whole_body_verifier_v5.py'):
        path=ROOT/relative;record=plan['source']['files'][relative]
        check('frozen_source:'+relative,sha(path)==record['sha256_raw'] and path.stat().st_size==record['size_bytes'])
        INPUTS[str(path.resolve())]={'sha256':sha(path),'bytes':path.stat().st_size}
    config,qp=plan['run_config'],plan['qp_config']
    check('original_contract',config['task_period_s']==.020 and config['physics_period_s']==.002
        and qp['feasibility_tolerance']==qp['velocity_tolerance_rad_s']==1e-4
        and qp['velocity_limit_scale']==.7 and qp['joint_barrier_gain']==4. and qp['joint_position_margin_rad']==.04)
    weights=np.asarray([(1.-s/10.,s/10.) for s in range(1,11)],dtype=np.float64)
    old_weight,new_weight=float(np.mean(weights[:,0])),float(np.mean(weights[:,1]))
    check('original_column_mean_ramp_weights',old_weight==.45 and new_weight==.55)
    # Hand-constructed named 67x17 configuration map, independent of either
    # diagnosis assembly. Each segment repeats the physical MuJoCo y/z order.
    mapping=np.zeros((67,17))
    pattern=(1,0,0,1,1,0,0,1,1,0,0,1)
    for segment in range(5):
        for row,column in enumerate(pattern):
            mapping[12*segment+row,2*segment+column]=1./6.
    mapping[60:,10:]=np.eye(7)
    decoder=np.linalg.pinv(mapping)
    spec=RobotModelSpecV5.from_urdf(default_robot_model_spec_v5().source_urdf,
        continuum_kp=100.,continuum_kd=5.,continuum_torque_limit=4.,rigid_kp=200.,rigid_kd=20.,rigid_torque_limit=80.)
    check('manual_17_planner_map_matches_named_original',np.array_equal(mapping,spec.planner_to_low_level)
        and np.array_equal(decoder,spec.low_level_to_planner)
        and list(spec.low_level_joint_names)==metadata['model_identity']['low_level_joint_names'])
    check('model_contract',spec.runtime_contract_sha256()==metadata['model_identity']['runtime_contract_sha256'])
    scenes={s['scenario_id']:s for s in load(TRIAL/'scenario_definitions.json')['scale_two']}
    producer_manifest=load(TRIAL/'manifest.json')
    certificates=[]
    for sid in CASES:
        snapshot_path=TRIAL/'simulation/failures'/(sid+'_counterexample.json')
        snap=load(snapshot_path); state=snap['planning_snapshots'][-1]
        diag=load(LP/(sid+'_diagnosis.json'))
        check(sid+':original_snapshot_hash',sha(snapshot_path)==diag['snapshot_sha256']
            ==producer_manifest[snapshot_path.relative_to(TRIAL).as_posix()])
        check(sid+':original_failed_frozen_state',state['selected_command'] is None and state['failure_reason']=='iteration_limit'
            and state['solver_status']=='maximum_iterations' and snap['next_servo_step_executed'] is False)
        obstacles=[WorkspaceSphere(o['name'],np.asarray(o['center_w']),o['radius_m']) for o in scenes[sid]['workspace_obstacles']]
        model=WholeBodyCollisionVerifier(spec,obstacles,WholeBodyVerificationConfig(minimum_clearance=.005,
            query_distance_max=2.5,adaptive_subdivisions=4,self_collision_ancestor_exclusion_depth=3,include_target_satellite_pairs=True)).model
        addresses=[]
        for name in spec.low_level_joint_names:
            jid=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,name)
            check(sid+':named_hinge:'+name,jid>=0 and int(model.jnt_type[jid])==int(mujoco.mjtJoint.mjJNT_HINGE))
            addresses.append(int(model.jnt_qposadr[jid]))
        check(sid+':actual_named_addresses',addresses==diag['joint_qpos_addresses_in_low_level_name_order'])
        qpos=np.asarray(state['qpos']);old=np.asarray(state['old_command'])
        q=decoder@qpos[addresses]
        check(sid+':finite_exact_planner_state',qpos.shape==(model.nq,) and old.shape==(17,)
            and np.all(np.isfinite(qpos)) and np.all(np.isfinite(old)) and np.array_equal(q,np.asarray(diag['measured_planner_q'])))
        j=8
        # Preserve original floating-point operation order: offset first.
        offset=q[j]+config['task_period_s']*old_weight*old[j]
        upper=(1.-offset)/(config['task_period_s']*new_weight)
        lower=max(-.8*qp['velocity_limit_scale'],old[j]-2.5*config['task_period_s'],
            -qp['joint_barrier_gain']*(q[j]-(-math.pi+qp['joint_position_margin_rad'])))
        check(sid+':original_velocity_lower_reconstructed',lower==state['velocity_lower'][j])
        check(sid+':original_domain_upper_reconstructed',upper==diag['domain_endpoint_upper_rad_s'][j]
            and diag['work_domain_upper_rad'][j]==1. and diag['work_domain_lower_rad'][j]==-1.)
        lower_tolerance=Fraction.from_float(qp['velocity_tolerance_rad_s'])
        upper_tolerance=Fraction.from_float(qp['feasibility_tolerance'])
        adjusted_lower=Fraction.from_float(lower)-lower_tolerance
        adjusted_upper=Fraction.from_float(float(upper))+upper_tolerance
        gap=adjusted_lower-adjusted_upper
        check(sid+':both_original_tolerances_still_empty',gap>0)
        lp_cert=diag['exact_scalar_contradiction_certificate']
        check(sid+':matches_lp_scalar_certificate',lp_cert['planner_index']==j
            and lower==lp_cert['velocity_lower_rad_s'] and upper==lp_cert['domain_endpoint_upper_rad_s']
            and float(gap)==lp_cert['tolerance_adjusted_gap_rad_s']
            and str(gap.numerator)==lp_cert['exact_binary_rational_adjusted_gap']['numerator']
            and str(gap.denominator)==lp_cert['exact_binary_rational_adjusted_gap']['denominator'])
        selected_low_indices=[i for i in range(67) if mapping[i,j]!=0.]
        certificates.append({'scenario_id':sid,'failure_time_s':state['time_s'],'planner_index':j,'planner_name':spec.planner_coordinate_names[j],
            'qpos_named_addresses':addresses,'planner_coordinate_low_level_joint_names':[spec.low_level_joint_names[i] for i in selected_low_indices],
            'measured_planner_q8':float(q[j]),'old_command8':float(old[j]),'old_weight':old_weight,'new_weight':new_weight,
            'work_domain':[-1.,1.],'velocity_lower':float(lower),'domain_endpoint_upper':float(upper),
            'lower_row_tolerance':float(lower_tolerance),'upper_row_tolerance':float(upper_tolerance),
            'tolerance_adjusted_lower':float(adjusted_lower),'tolerance_adjusted_upper':float(adjusted_upper),
            'positive_empty_intersection_gap':float(gap),'exact_gap_numerator':str(gap.numerator),'exact_gap_denominator':str(gap.denominator),
            'snapshot_sha256':sha(snapshot_path),'lp_diagnosis_sha256':sha(LP/(sid+'_diagnosis.json')),
            'proof':'The same endpoint u8 must be >= velocity_lower - 1e-4 and <= domain_endpoint_upper + 1e-4. Exact binary-rational adjusted lower exceeds adjusted upper.'})
    check('all_inputs_unchanged',all(sha(p)==v['sha256'] and Path(p).stat().st_size==v['bytes'] for p,v in INPUTS.items()))
    return {'schema':'research_velocity_failure_scalar_independent_verification_v1','evidence_valid':True,
        'frozen_linear_infeasibilities_confirmed':3,'research_acceptance_passed':False,'source_trial_commit':COMMIT,
        'script_sha256':sha(__file__),'completed_utc':datetime.now(timezone.utc).isoformat(),'checks':CHECKS,'certificates':certificates,'inputs':INPUTS,
        'scope':'Three frozen original 17-dimensional linear constraint sets have an empty scalar intersection even after both original 1e-4 row tolerances. No optimizer, geometry query, physics replay, or diagnostic function was called.',
        'limits':'Does not prove the entire 2x task intrinsically impossible, identify a safe alternative trajectory, validate nonlinear execution, delay/model robustness, continuous-time safety, hardware deployment or wall C11.'}


if __name__=='__main__':
    if (OUT/'verification.json').exists():
        raise FileExistsError('preserve previous scalar verification')
    try:
        result=run()
    except Exception as e:
        result={'evidence_valid':False,'research_acceptance_passed':False,'script_sha256':sha(__file__),
            'checks':CHECKS,'error':{'type':type(e).__name__,'message':str(e),'traceback':traceback.format_exc()}}
    write(OUT/'verification.json',result)
    write(OUT/'manifest.json',{p.name:{'sha256':sha(p),'bytes':p.stat().st_size} for p in OUT.iterdir() if p.is_file() and p.name!='manifest.json'})
    print(json.dumps({'evidence_valid':result['evidence_valid'],'checks':len(CHECKS),
        'gaps':[c['positive_empty_intersection_gap'] for c in result.get('certificates',[])],
        'error':result.get('error'),'verification':str(OUT/'verification.json')},ensure_ascii=False,indent=2))
    raise SystemExit(0 if result['evidence_valid'] else 1)
