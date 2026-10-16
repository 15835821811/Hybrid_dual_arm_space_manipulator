import time
import numpy as np
import pytest
from ff_prior.storage import read_json,atomic_json,digest
from ff_prior.pipeline import acceptance
from ff_prior.guidance import public_goal_initializer
from ff_prior_loop import data,training
from ff_prior_feasible.guidance import accept_update,constrained_step,FeasibleProjector
from ff_prior_feasible.validation import validate
from ff_prior_feasible.experiment import candidate,PREVIOUS


def profile(): return acceptance(read_json('artifacts/v7_1/smoke_002/run.json')['config'])


def test_collision_improvement_cannot_destroy_goal_feasibility():
    a=np.full(12,.1); b=a.copy(); b[0]=.81
    assert not accept_update(a,np.array([-.002]),b,np.array([.001]),True)
    assert accept_update(a,np.array([-.002]),a,np.array([-.001]),True)
    assert not accept_update(a,np.array([-.001]),a,np.array([-.002]),True)


def test_constrained_collision_step_uses_redundancy_without_goal_tradeoff():
    e=np.full(12,.2); jg=np.c_[4*np.eye(12),np.zeros((12,12))]
    jd=np.zeros((2,24)); jd[:,12]=.1
    x,info=constrained_step(e,np.array([-.004,-.003]),jg,jd,True)
    assert info['linear_pose_feasible']
    assert np.max(np.linalg.norm((e+jg@x).reshape(4,3),axis=1))<=.650001
    assert np.min(np.array([-.004,-.003])+jd@x)>0
    assert np.max(abs(x))<=.150001


def test_real_train_path_adaptive_refinement_preserves_2mm_gate(physical):
    canonical,initial,_=physical
    e=next(e for e in read_json(data.OLD/'frozen_complex_tasks.json')['entries'] if e['sample_id']=='complex_train_01_static_detour')
    r=read_json('ff_prior_feasible/evidence/train_calibration_001/1_trained_diffusion_feasible_1.json')
    rr,ii,cc,_=data.runtime(e,canonical,initial); before=rr.calls['coupled']
    result=validate(r['Z'],e['task'],rr,ii,cc,profile(),time.time()+60)
    assert result['eligible'] and rr.calls['coupled']-before==3
    assert not result['validation_grids'][0]['gates']['geometry_refinement']
    assert result['validation']['geometry']['maximum_geometry_step_m']<=.002
    assert result['validation']['geometry']['minimum_clearance_m']>=0
    assert result['validation_integration_intervals']==1792


def test_adaptive_grid_does_not_accept_real_penetration(physical):
    canonical,initial,_=physical
    e=next(e for e in read_json(data.OLD/'frozen_complex_tasks.json')['entries'] if e['sample_id']=='complex_train_00_static_detour')
    rr,ii,cc,_=data.runtime(e,canonical,initial); task=e['task']; base=public_goal_initializer(task,rr,ii)
    result=validate(base,task,rr,ii,cc,profile(),time.time()+60)
    assert not result['eligible'] and not result['validation']['gates']['geometry']
    assert result['validation']['geometry']['minimum_clearance_m']<0


def test_actual_train_sampler_preserves_goal_bounds_ddim_and_budget(physical):
    canonical,initial,_=physical
    e=next(e for e in read_json(data.OLD/'frozen_complex_tasks.json')['entries'] if e['sample_id']=='complex_train_00_static_detour')
    rr,ii,cc,_=data.runtime(e,canonical,initial); task=e['task']; before=digest(task)
    bundle=training.load(PREVIOUS/'learning',read_json(PREVIOUS/'learning/training.json')['metadata']['scope'])
    result=candidate('trained_diffusion_feasible',0,task,public_goal_initializer(task,rr,ii),20261073,bundle,rr,ii,cc,profile(),time.time()+60)
    assert result['eligible'] and result['status']=='DONE'
    assert digest(task)==before and result['model_calls']['coupled']<=113 and result['integration_intervals']<=15872
    assert result['sampling']['actual_network_calls']==16
    assert max(s['decomposition_residual_after'] for s in result['sampling']['steps'])<1e-12
    assert np.all(result['Z']>=rr.lower) and np.all(result['Z']<=rr.upper)
    guide=result['sampling']['guidance'][-1]
    for update in guide['history']:
        if update['accepted'] and max(update['before_pose_norms'])<=.8:
            assert max(update['after_pose_norms'])<=.8


def test_deadline_is_not_reported_as_candidate_success(physical):
    canonical,initial,_=physical
    e=read_json(PREVIOUS/'frozen_tasks.json')['training_generation_entries'][0]
    rr,ii,cc,_=data.runtime(e,canonical,initial); task=e['task']
    result=candidate('structured_feasible',0,task,public_goal_initializer(task,rr,ii),1,None,rr,ii,cc,profile(),time.time()-1)
    assert result['status']=='ERROR' and not result['eligible'] and result['integration_intervals'] is None


def test_empty_results_never_claim_advantage(tmp_path):
    from ff_prior_feasible.reporting import report
    from ff_prior.storage import Ledger
    atomic_json(tmp_path/'frozen_tasks.json',{'evaluation_entries':[]})
    result=report(tmp_path,Ledger(tmp_path,{}),'RUNNING')
    assert not result['stability_gate_passed'] and not result['exploratory_advantage_gate_passed'] and not result['full_goal_achieved']


def test_background_duplicate_launch_refused(tmp_path):
    from ff_prior_feasible.job import Monitor,status,launch
    atomic_json(tmp_path/'run.json',{})
    with Monitor(tmp_path) as monitor:
        with pytest.raises(RuntimeError,match='alive'): launch(tmp_path/'source',tmp_path,True)
        monitor.finish({'status':'FINISHED'})
    assert status(tmp_path)['status']=='FINISHED' and not (tmp_path/'launch.guard').exists()
