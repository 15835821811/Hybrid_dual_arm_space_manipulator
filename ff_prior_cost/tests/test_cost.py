import time
import numpy as np
import pytest
from ff_prior_goal.tests.test_goal import reachable_task
from ff_prior_goal.experiment import CHECKPOINT
from ff_prior.guidance import public_goal_initializer
from ff_prior.storage import digest, atomic_json
from ff_prior.contracts import frozen_task
from ff_prior_next import learning
from ff_prior_cost.projection import Projector
from ff_prior_cost.sampling import sample
from ff_prior_cost.experiment import candidate, train_targets
from ff_prior_cost.reporting import better


def test_ddim_matches_frozen_algebra_with_same_repair(physical):
    robot,initial,collision=physical; task,profile=reachable_task(physical)
    bundle=learning.load(CHECKPOINT,initial.model_hash,initial.state_id)
    base=public_goal_initializer(task,robot,initial)
    first=Projector(task,robot,initial,collision,profile,time.time()+60,(14,15))
    expected,_=learning.sample(*bundle,base,task,123,first)
    second=Projector(task,robot,initial,collision,profile,time.time()+60,(14,15))
    actual,info=sample(bundle,base,task,123,second)
    assert np.allclose(actual,expected,atol=1e-12,rtol=0)
    assert info['actual_network_calls']==16 and first.used==second.used
    assert max(s['decomposition_residual_after'] for s in info['steps'])<1e-12


@pytest.mark.parametrize('method',['structured_final','frozen_diffusion_final','frozen_diffusion_two'])
def test_real_candidates_share_query_ceiling_and_unchanged_goals(physical,method):
    robot,initial,collision=physical; task,profile=reachable_task(physical); old=digest(task)
    bundle=learning.load(CHECKPOINT,initial.model_hash,initial.state_id)
    base=public_goal_initializer(task,robot,initial)
    result=candidate(method,0,task,base,123,bundle,robot,initial,collision,profile,time.time()+60)
    assert result['model_calls']['coupled']<=113 and result['repair_coupled_calls']<=110
    assert result['validation_coupled_calls']<=3 and result['eligible'] and digest(task)==old


def test_exhausted_budget_cannot_relabel_unreachable_world_goal(physical):
    robot,initial,collision=physical; task,profile=reachable_task(physical)
    c=dict(task['condition']); c['goals']=np.array(c['goals']); c['goals'][:,:3,3]+=10.
    task=frozen_task(c,task['contract']); old=digest(task)
    p=Projector(task,robot,initial,collision,profile,time.time()+60,(14,15))
    z=np.repeat(robot.spec.planner_zero[None],19,axis=0)
    z,a=p(z,14); z,b=p(z,15)
    assert a['coupled_calls']<=74 and p.used<=110
    assert max(b['position_error_m'])>.005 and digest(task)==old
    with pytest.raises(ValueError): p(np.full((19,17),np.nan),15)


def test_training_target_selection_excludes_other_partitions():
    rows=[({'sample_id':str(i),'family_id':str(i//2),'task':{'id':i}},'TRAIN') for i in range(60)]
    rows+=[({'sample_id':'forbidden','family_id':'x','task':{}},s) for s in ('VAL','DEV','TEST')]
    a=train_targets(rows)
    assert len(a)==30 and a==train_targets(list(reversed(rows)))
    assert all(e['sample_id']!='forbidden' for e in a)


def test_cost_advantage_requires_success_and_wall_not_just_queries():
    a={'candidate_records':20,'candidate_denominator':20,'candidate_errors':0,'eligible':20,
        'execution_verified':4,'task_denominator':4,'coupled_queries':80,'candidate_wall_s':1.}
    b=dict(a,coupled_queries=100)
    assert better(a,b)
    assert not better(dict(a,eligible=19),b)
    assert not better(dict(a,execution_verified=3),b)
    assert not better(dict(a,candidate_wall_s=2.),b)
    assert not better(dict(a,candidate_errors=1),b)


def test_background_worker_duplicate_is_refused(tmp_path):
    from ff_prior_cost.job import Monitor,status,launch
    atomic_json(tmp_path/'run.json',{'fixture':True})
    with Monitor(tmp_path) as monitor:
        assert status(tmp_path)['owner_alive']
        with pytest.raises(RuntimeError,match='alive'): launch(tmp_path/'source',tmp_path,True)
        monitor.finish({'status':'FINISHED'})
    assert not (tmp_path/'launch.guard').exists()
