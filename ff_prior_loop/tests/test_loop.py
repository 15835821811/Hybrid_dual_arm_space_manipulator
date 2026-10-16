import time
import numpy as np
import pytest
from ff_prior.storage import read_json,atomic_json,digest
from ff_prior.pipeline import acceptance
from ff_prior.contracts import assert_task_compatible
from ff_prior.spline import Reference
from ff_prior.proposal_generator import quick_gates
from ff_prior.guidance import public_goal_initializer
from ff_prior.coupled_predictor import predict_coupled
from ff_prior_complex.tasks import task_for,scene_runtime
from ff_prior.scenes import SceneSpec
from ff_prior_loop.runtime import at_origin,origin_audit
from ff_prior_loop.guidance import JointProjector
from ff_prior_loop import data
from ff_prior_loop.experiment import candidate
from ff_prior_loop.reporting import decision
from ff_prior_cost.experiment import validate,CHECKPOINT
from ff_prior_next import learning


def profile(): return acceptance(read_json('artifacts/v7_1/smoke_002/run.json')['config'])


def test_new_origin_is_complete_state_not_physical_redefinition(physical):
    canonical,old,_=physical; q0=canonical.spec.planner_zero+np.linspace(-.01,.01,17)
    rr,ii,cc=at_origin(canonical,q0)
    assert origin_audit(canonical,rr,ii)['status']=='PASS'
    assert ii.state_id!=old.state_id and ii.model_hash==old.model_hash and rr.model is canonical.model
    assert np.array_equal(canonical.spec.planner_zero,rr.spec.physical_spec.planner_zero)
    z=np.tile(q0,(19,1)); z+=np.linspace(0,.01,19)[:,None]
    ref=Reference(z,q0,15.); assert all(quick_gates(rr,ref).values())
    pred=predict_coupled(ii,ref,rr,128,time.time()+60)
    task=task_for(rr,ii,cc,profile(),pred['ee'][-1],15.)
    assert_task_compatible(task,rr,ii,cc,profile())
    with pytest.raises(ValueError): assert_task_compatible(task,canonical,old,physical[2],profile())
    with pytest.raises(ValueError): at_origin(canonical,canonical.upper+1e-10)


def test_lineage_selection_reproducible_and_disjoint():
    index=data.old_index(); a=data.select(index); b=data.select(list(reversed(index)))
    assert a==b and len(a)==16
    assert len({e['family_id'] for e in a})==16
    assert len(data.select(index,True))==4
    assert not {e['family_id'] for e in a}&{e['family_id'] for e in data.select(index,True)}
    with pytest.raises(ValueError): data.assert_disjoint([dict(family_id='x',split='TRAIN'),dict(family_id='x',split='VAL')])


def test_fixed_world_generation_really_loads_new_initial_and_same_goal(physical,tmp_path):
    canonical,initial,_=physical; index=data.old_index(); parent=next(e for e in data.select(index) if e['split']=='TRAIN')
    e,_=data.freeze_parent(parent,0,canonical,profile(),index,[],tmp_path,time.time()+60)
    assert e['status']=='FROZEN',e
    rr,ii,cc,_=data.runtime(e,canonical,initial); h=digest(e['task'])
    for k in (0,1):
        r=data.generate_variant(e,k,rr,ii,cc,profile(),tmp_path,time.time()+60)
        assert r['eligible'],r
        row=read_json(tmp_path/'records'/f'{e["sample_id"]}_v{k}.json')
        assert digest(row['task'])==h and row['validation_level']=='MODEL_VALID' and row['execution_status']=='NOT_RUN'
    with np.load(tmp_path/'tasks'/e['sample_id']/'initial_state.npz') as a: assert np.array_equal(a['values'],ii.values)


def test_real_collision_guidance_repairs_train_case_without_relaxing_gate():
    e=next(e for e in read_json(data.OLD/'frozen_complex_tasks.json')['entries'] if e['sample_id']=='complex_train_01_static_detour')
    rr,ii,cc=scene_runtime(SceneSpec.from_dict(e['scene'])); task=e['task']; h=digest(task)
    base=public_goal_initializer(task,rr,ii)
    p=JointProjector(task,rr,ii,cc,profile(),time.time()+90); z,info=p(base,15)
    result=validate(z,task,rr,ii,cc,profile(),time.time()+60)
    assert result['eligible'] and result['validation']['geometry']['minimum_clearance_m']>=0
    assert p.used<=110 and p.geometry_states>0 and digest(task)==h
    assert result['validation']['geometry']['maximum_geometry_step_m']<=.002
    assert info['collision_enabled'] and not info['private_witness_used']


@pytest.mark.parametrize('method',['structured_joint','frozen_diffusion_joint','trained_diffusion_goal_only'])
def test_numeric_sampler_shared_budget_and_complete_origin(physical,method):
    canonical,initial,_=physical; rr,ii,cc=at_origin(canonical,canonical.spec.planner_zero+np.full(17,.005))
    z=np.tile(rr.spec.planner_zero,(19,1)); pred=predict_coupled(ii,Reference(z,rr.spec.planner_zero,15.),rr,128,time.time()+60)
    task=task_for(rr,ii,cc,profile(),pred['ee'][-1],15.); base=public_goal_initializer(task,rr,ii)
    bundle=learning.load(CHECKPOINT,initial.model_hash,initial.state_id)
    r=candidate(method,0,task,base,212,bundle,rr,ii,cc,profile(),time.time()+60)
    assert r['status']=='DONE',r.get('reason')
    assert r['model_calls']['coupled']<=113 and r['repair_coupled_calls']<=110 and r['validation_coupled_calls']<=3
    assert np.isfinite(r['Z']).all() and np.all(r['Z']>=rr.lower) and np.all(r['Z']<=rr.upper)
    if method!='structured_joint':
        assert r['sampling']['actual_network_calls']==16
        assert max(s['decomposition_residual_after'] for s in r['sampling']['steps'])<1e-12


def test_empty_or_failed_evidence_never_opens_expansion_gate(tmp_path):
    atomic_json(tmp_path/'frozen_tasks.json',{'evaluation_entries':[]})
    d=decision(tmp_path)
    assert not d['passed'] and d['decision']=='BLOCKED' and not d['TEST']


def test_new_background_duplicate_refused(tmp_path):
    from ff_prior_loop.job import Monitor,status,launch
    atomic_json(tmp_path/'run.json',{})
    with Monitor(tmp_path) as monitor:
        with pytest.raises(RuntimeError,match='alive'): launch(tmp_path/'source',tmp_path,True)
        monitor.finish({'status':'FINISHED_WITH_BLOCKED_ITEMS'})
    assert status(tmp_path)['status']=='FINISHED_WITH_BLOCKED_ITEMS' and not (tmp_path/'launch.guard').exists()
