"""Software and frozen-resource checks; no native physics or optimizer steps."""
from pathlib import Path
import os
import sys
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='2'
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT));os.chdir(ROOT)
import json
from types import SimpleNamespace
import numpy as np
from ff_prior_fullchain_acceptance.inputs import identity
from ff_prior_fullchain_acceptance.protocol import PACKAGE,P
from ff_prior_fullchain_acceptance.handoff import check_boundary,PhaseSession
from ff_prior_fullchain_acceptance.experiment import exact_reuse,summarize
from ff_prior_static_goal_seed.physical_audit import replay_agrees
from ff_prior_static_goal_seed.tracker import integration_state
from ff_prior.robot_adapter import RobotAdapter
from ff_prior.storage import digest,read_json
from ff_prior.spline import Reference
from ff_prior.pipeline import acceptance
from ff_prior_pilot_train.storage import atomic_json

checks={}
def ok(name,value):
    checks[name]=bool(value)
    if not value:raise AssertionError(name)
def rejects(name,fn,reason):
    try:fn()
    except ValueError as e:ok(name,str(e)==reason)
    else:ok(name,False)

ident=identity();sel=ident['selection'];units=sel['units']
ok('thirty_denominators',len(units)==30 and len({u['id'] for u in units})==30)
ok('five_scenes',set(u['scene'] for u in units)==set(P['scenes']))
ok('six_new_static',sum(u['action']=='NEW_STATIC_EXECUTION' for u in units)==6)
ok('twelve_missing_kept',sum(u['action']=='BLOCKED' for u in units)==12)
ok('twelve_historic_eight_unique',sum(u['action']=='EXACT_HISTORICAL_REUSE' for u in units)==12 and
   len({u['execution_source'] for u in units if u['action']=='EXACT_HISTORICAL_REUSE'})==8)
ok('combo_not_falsely_frozen',all(u['task_hash'] is None for u in units if u['scene']=='five_obstacle_to_drifting'))
profile=acceptance(read_json(ROOT/'artifacts/v7_1/smoke_002/run.json')['config'])
steps=0
for u in units:
    if u['action']=='NEW_STATIC_EXECUTION':
        entry=next(e for e in sel['entries'] if e['sample_id']==u['scene'])
        v=read_json(ROOT/u['reference']);ref=Reference(v['Z'],entry['task']['condition']['initial_features'],entry['task']['condition']['duration'])
        ok('reference_hash_'+u['id'],ref.reference_hash==v['reference_hash'])
        steps+=round((ref.duration+profile.hold_s)/.002)*2
    elif u['action']=='EXACT_HISTORICAL_REUSE':
        r=exact_reuse(u);ok('identity_reuse_'+u['id'],r['new_physics_steps']==0 and r['new_execution'] is False)
ok('full_step_budget',steps==165000 and steps==P['new_native_replay_steps'])
ok('no_test_training_expansion',P['TEST'] is False and P['TRAIN_added']==P['new_training_updates']==P['new_proposals']==0)
robot=RobotAdapter();data=robot.new_data();robot.forward(data)
state=integration_state(robot,data);pose=robot.ee(data).copy();model=digest(robot.identity())
check_boundary(state,state.copy(),model,model,'EXECUTION_VERIFIED',pose,pose)
ok('boundary_full_vector_equality',True)
changed=state.copy();changed[-1]+=1e-3
rejects('extra_state_tamper_rejected',lambda:check_boundary(state,changed,model,model,'EXECUTION_VERIFIED',pose,pose),'PHYSICAL_STATE_CHANGED_AT_HANDOFF')
rejects('wrong_model_rejected',lambda:check_boundary(state,state,model,'wrong','EXECUTION_VERIFIED',pose,pose),'MODEL_OR_SCENE_CHANGED')
rejects('unqualified_parent_rejected',lambda:check_boundary(state,state,model,model,'MODEL_VALID',pose,pose),'PREVIOUS_PHASE_NOT_VERIFIED')
bad=pose.copy();bad[0,:3,3]+=.1
rejects('jump_target_rejected',lambda:check_boundary(state,state,model,model,'EXECUTION_VERIFIED',pose,bad),'NEXT_TARGET_DISCONTINUITY')
receipt=dict(level='EXECUTION_VERIFIED',model_hash=model,end_state_hash=digest(state))
session=PhaseSession(robot,data,'software_fixture',float(data.time))
policy,transition=session.switch('software_next',receipt,pose,lambda r,d,t:SimpleNamespace(data=d))
ok('same_live_data_clock_no_writes',policy.data is data and np.array_equal(state,integration_state(robot,data)) and session.local_time()==0)
rejects('new_env_rejected',lambda:session.switch('bad',receipt,pose,lambda r,d,t:SimpleNamespace(data=object())),'NEW_ENVIRONMENT_OR_DATA_AT_HANDOFF')
ok('full_state_replay_required',not replay_agrees({'qpos':0.,'qvel':0.},1e-10))
ok('nan_full_replay_rejected',not replay_agrees({'integration':float('nan')},1e-10))
empty=summarize([]);ok('missing_rows_cannot_pass',not empty['engineering_complete'] and all(v['missing']==3 for d in empty['scene_counts'].values() for v in d.values()))
ok('software_did_not_step_physics',robot.calls['physics_steps']==0)
result=dict(status='PASS',checks=checks,source_hashes=ident['source_hashes'],source_files=len(ident['source_hashes']),
    bound_inputs=len(ident['binding']['files']),new_physics_steps=0,new_training_updates=0,
    scope='Software fixtures and identity/resource checks; no successful physical handoff claimed')
atomic_json(PACKAGE/'evidence/software_tests.json',result)
print(json.dumps({k:v for k,v in result.items() if k not in ('checks','source_hashes')},ensure_ascii=False))
print('checks',len(checks))
