"""Full-output C3 parity and C4 scheduling/cache/slot contracts without physics."""
import contextlib
import importlib.util
import io
from pathlib import Path
from unittest.mock import patch

import mujoco
import numpy as np
import pytest

from v6_4.c4a_frozen import RELEASE
from v6_4.c4a_portfolio import RulePreservingPortfolio, optimize_strategy
from v6_4.continuous_route_optimizer import optimize
from v6_4.route_optimizer_protocol import (TaskSpec,PreferenceSpec,read,SearchSpec,initial_candidates,
    parameter_plan,active_intervals,build_reference_definition)


@pytest.fixture()
def case():
    task=TaskSpec.from_dict(read(RELEASE/'snapshot/frozen_tasks/c1_mother_00_plus/task.json'))
    prefs=[PreferenceSpec(**p) for p in read(RELEASE/'snapshot/frozen_tasks/c1_mother_00_plus/preferences.json')]
    identity=dict(source_identity_sha256='mock',config_sha256='mock',task_sha256=task.sha256(),model_contract_sha256=task.model_contract_sha256)
    proposals={i:dict(source='diffusion',preference=p,family=f,raw_z_m=parameter_plan(task,f,x).z_m.tolist())
        for i,p,f,x in [(1,'A','v1',[.001,0,-.005,0]),(3,'B','v2',[.002,0,-.017,0])]}
    return task,prefs,identity,proposals


def evaluator(plan,cid):
    z=plan.z_m
    return dict(status='PREDICTION_ADMISSIBLE',prediction_rollout_started=True,prediction_steps=13500,
        prediction_admissible=True,prediction_task_passed=True,online_guards_passed=True,
        prediction_metrics=dict(I_support=float(.1+z[1,0]),L_full=float(1+np.linalg.norm(z)),d_support=float(.025+abs(z[2,0])*.5)),
        costs=dict(prediction_physics_steps=13500))


@pytest.mark.parametrize('budget,kind',[(8,'R'),(12,'R'),(8,'D'),(8,'invalid'),(8,'duplicate')])
def test_c3_full_outputs_identical_to_frozen_producer(tmp_path,case,budget,kind):
    task,prefs,identity,proposals=case
    if kind=='invalid': proposals[1]['raw_z_m'][1][0]=.1
    if kind=='duplicate': proposals[1]['raw_z_m']=np.zeros((6,2)).tolist()
    spec=importlib.util.spec_from_file_location('v6_4._c3_optimizer_reference',RELEASE/'frozen_source/v6_4/continuous_route_optimizer.py')
    baseline=importlib.util.module_from_spec(spec);spec.loader.exec_module(baseline)
    with contextlib.redirect_stdout(io.StringIO()),patch('time.perf_counter',return_value=100.),patch.object(mujoco,'mj_step',side_effect=AssertionError('no physics')):
        for func,name in [(baseline.optimize,'old'),(optimize,'new')]:
            func(task,prefs,identity,evaluator,tmp_path/name,search_spec=SearchSpec(candidate_budget=budget),initializer=proposals if kind!='R' else None)
    old={p.name:p.read_bytes() for p in (tmp_path/'old').iterdir() if p.is_file()}
    new={p.name:p.read_bytes() for p in (tmp_path/'new').iterdir() if p.is_file()}
    assert old==new


@pytest.mark.parametrize('kind',['legal','invalid','duplicate'])
def test_p2_exact_positions_budget_shared_pool_and_cache(tmp_path,case,kind):
    task,prefs,identity,proposals=case;calls=[]
    if kind=='invalid': proposals[1]['raw_z_m'][1][0]=.1
    if kind=='duplicate':
        proposals[1]['raw_z_m']=np.zeros((6,2)).tolist()
        seed=initial_candidates(task)[3]
        proposals[3]['raw_z_m']=parameter_plan(task,'v2',seed['x_m']).z_m.tolist()
    def observed(plan,cid):
        calls.append((cid,plan.sha256()));return evaluator(plan,cid)
    with contextlib.redirect_stdout(io.StringIO()),patch.object(mujoco,'mj_step',side_effect=AssertionError('no physics')):
        result=optimize_strategy('P2',task,prefs,identity,observed,tmp_path,proposals=proposals)
    rows=read(tmp_path/'candidate_registry.json');props=read(tmp_path/'proposals.json')
    assert len(rows)==len(props)==8 and result['budget']['shared_A_B_pool']
    assert len(calls)==result['budget']['prediction_rollouts_started']
    for i,seed in enumerate(initial_candidates(task)):
        assert rows[i]['plan_sha256']==parameter_plan(task,seed['family'],seed['x_m']).sha256()
    assert [(r['source'],r['initializer_slot']) for r in rows[4:6]]==[('diffusion',1),('diffusion',3)]
    assert [r['preference_center'] for r in rows[6:]]==['A','B']
    assert [r['coordinate'] for r in rows[6:]]==[0,0]
    assert read(tmp_path/'prefix_04.json')['candidate_ids']==['C00','C01','C02','C03']
    if kind=='invalid':
        assert rows[4]['status']=='INITIALIZER_RAW_REJECTED' and rows[4]['raw_z_m']==proposals[1]['raw_z_m']
        assert not rows[4]['raw_seed_diagnostics']['resampled'] and len(calls)==7
    if kind=='duplicate':
        assert len(calls)==6 and result['budget']['cache_hits']==2
        assert rows[4]['evidence_alias_of_candidate_id']=='C00' and rows[5]['evidence_alias_of_candidate_id']=='C03'
        assert all(rows[i]['prediction_steps']==0 and not rows[i]['prediction_rollout_started'] for i in [4,5])
        assert sum(r['costs'].get('prediction_physics_steps',0) for r in rows)==6*13500


def test_schedule_frozen_copy_and_opt_in_only(case,tmp_path):
    task,prefs,identity,proposals=case
    schedule=RulePreservingPortfolio(proposals);first=schedule(task)
    proposals[1]['raw_z_m'][1][0]=999
    assert schedule(task)==first
    with pytest.raises(ValueError): optimize(task,prefs,identity,evaluator,tmp_path,seed_schedule=schedule)
    with pytest.raises(ValueError): optimize_strategy('P0',task,prefs,identity,evaluator,tmp_path,proposals=proposals)
