from dataclasses import asdict
import os
import time
import numpy as np
import pytest
from ff_prior.contracts import AcceptanceConfig,frozen_task
from ff_prior.storage import digest,atomic_json
from ff_prior.spline import Reference
from ff_prior.torque_tracker import track
from ff_prior_fixed.correction import choose_targets,correct,is_eligible
from ff_prior_fixed.consistency import analyze,metrics,base_poses
from ff_prior_fixed.experiment import verified_training_rows
from ff_prior_fixed.job import Monitor,status,launch


def task_at_home(physical):
    robot,initial,collision=physical; profile=AcceptanceConfig()
    condition={'initial_features':robot.spec.planner_zero,'goals':robot.ee(initial.reset(robot)),
        'duration':10.,'pose_mask':np.ones((2,6))}
    task=frozen_task(condition,{'initial_state_id':initial.state_id,'robot_hash':initial.model_hash,
        'pair_policy_hash':collision.policy_hash,'acceptance':asdict(profile)})
    return task,profile


def test_train_selection_is_fixed_family_unique_and_excludes_val():
    rows=[({'sample_id':str(i),'family_id':str(i//2)},'TRAIN') for i in range(20)]
    rows += [({'sample_id':'private_dev','family_id':'other'},'DEV')]
    a=choose_targets(rows); b=choose_targets(list(reversed(rows)))
    assert a==b and len(a)==8 and len({r['family_id'] for r in a})==8
    assert all(r['sample_id']!='private_dev' for r in a)


def test_train_correction_preserves_goal_and_obeys_real_model_gates(physical):
    robot,initial,collision=physical; task,profile=task_at_home(physical); before=digest(task)
    z,r=correct(task,'software_regression','TRAIN',0,robot,initial,collision,profile,time.time()+60)
    assert digest(task)==before and not r['goal_relabelled'] and r['task_hash']==task['task_hash']
    assert r['training_eligible'] and r['frozen_goal_pass'] and r['validation_level']=='MODEL_VALID'
    assert r['optimization_coupled_calls']<=1+8*(34+4) and r['iterations']<=8
    assert r['execution_status']=='NOT_RUN'
    for split in ('VAL','DEV','TEST'):
        with pytest.raises(ValueError,match='TRAIN'): correct(task,'x',split,0,robot,initial,collision,profile,time.time()+1)


def test_failed_fixed_goal_is_not_relabelled_or_admitted(physical):
    robot,initial,collision=physical; task,profile=task_at_home(physical)
    condition=dict(task['condition']); goals=np.array(condition['goals']).copy(); goals[:,:3,3]+=10
    condition['goals']=goals; task=frozen_task(condition,task['contract']); before=digest(task)
    _,r=correct(task,'unreachable_fixture','TRAIN',1,robot,initial,collision,profile,time.time()+60,max_iterations=0)
    assert digest(task)==before and not r['frozen_goal_pass'] and not r['training_eligible']
    assert r['validation_level']=='REJECTED'
    assert not is_eligible({'validation_level':'MODEL_VALID'},False)


def test_known_execution_failures_are_excluded_without_touching_sources():
    rows=[({'sample_id':'old_train'},'TRAIN'),({'sample_id':'old_val'},'VAL')]
    new={'sample_id':'new','split':'TRAIN','training_eligible':True,'frozen_goal_pass':True,
        'validation_level':'MODEL_VALID','execution_status':'NOT_RUN'}
    chosen=verified_training_rows(rows,[new],{'old_train'})
    assert [r['sample_id'] for r,s in chosen]==['old_val','new']
    assert rows[0][0]['sample_id']=='old_train' and new['execution_status']=='NOT_RUN'
    with pytest.raises(ValueError,match='unqualified'): verified_training_rows(rows,[dict(new,split='DEV')],set())


def test_native_conditioned_reconstruction_on_real_stationary_torque_trace(physical,tmp_path):
    robot,initial,_=physical
    z=np.repeat(robot.spec.planner_zero[None],19,axis=0); reference=Reference(z,robot.spec.planner_zero,.04)
    arrays,metadata=track(robot,initial,reference,.5,300)
    assert metadata['robot_state_writes_during_step_loop']==0
    r=analyze(robot,initial,reference,arrays,tmp_path/'comparison.npz',time.time()+60)
    assert r['all_67_native_joint_rates_used'] and not r['Bplus_used_for_reconstruction']
    assert not r['per_frame_base_state_copied'] and all(r['diagnostic_flags'].values())
    assert r['actual_native_path_reconstruction_vs_execution']['end_effectors']['max_position_m']<1e-8
    # A visible discrepancy must not be hidden by passing model gates.
    changed=dict(arrays); changed['ee']=arrays['ee'].copy(); changed['ee'][:,:,:3,3]+=.02
    q=analyze(robot,initial,reference,changed,tmp_path/'mismatch.npz',time.time()+60)
    assert not q['diagnostic_flags']['reference_prediction_within_5mm_1deg']


def test_new_job_refuses_live_duplicate_and_preserves_lock(tmp_path):
    atomic_json(tmp_path/'run.json',{'software_fixture':True})
    with Monitor(tmp_path) as monitor:
        assert status(tmp_path)['owner_alive']
        with pytest.raises(RuntimeError,match='alive'): launch(tmp_path/'source',tmp_path,resume=True)
        monitor.finish({'status':'FINISHED'})
    assert status(tmp_path)['status']=='FINISHED' and not (tmp_path/'launch.guard').exists()
