import inspect
import time
import numpy as np
import pytest
import mujoco
from ff_prior.storage import digest,atomic_json
from ff_prior.scenes import SceneSpec,Obstacle
from ff_prior_goal.tests.test_goal import reachable_task
from ff_prior_goal.experiment import CHECKPOINT
from ff_prior.guidance import public_goal_initializer
from ff_prior_next import learning
from ff_prior_complex.tasks import select_parents,scene_runtime,transfer_audit,task_for
from ff_prior_complex.experiment import candidate,proposal,PROTOCOL


def test_split_lineage_and_selection_are_order_independent():
    rows=[({'sample_id':f'{s}{i}','family_id':f'{s}{i//2}'},s) for s in ('TRAIN','VAL') for i in range(10)]
    rows += [({'sample_id':s,'family_id':s},s) for s in ('DEV','TEST')]
    a=select_parents(rows)
    assert a==select_parents(list(reversed(rows))) and len(a)==6
    assert len({r['family_id'] for r,s in a})==6
    assert all(s in ('TRAIN','VAL') for r,s in a)


def test_static_scene_transfer_preserves_dynamics_and_forbids_wrong_snapshot(physical):
    robot,initial,collision=physical
    rr,ii,cc=scene_runtime(SceneSpec((Obstacle('test','sphere',(5.,5.,5.),(.012,0.,0.)),),'unit'))
    audit=transfer_audit(robot,initial,rr,ii)
    assert audit['status']=='PASS' and audit['added_world_geoms']==1
    assert ii.state_id!=initial.state_id and ii.model_hash!=initial.model_hash
    with pytest.raises(ValueError,match='identity'): initial.reset(rr)
    names=[cc.describe(*p) for p in cc.pairs]
    assert any('v7_test' in (p['geom_a'],p['geom_b']) for p in names)
    rr.model.body_mass[rr.base_body]*=1.01
    with pytest.raises(ValueError,match='changed robot'): transfer_audit(robot,initial,rr,ii)


def test_world_obstacle_is_not_excluded_from_nonpenetration(physical):
    robot,initial,collision=physical
    data=initial.reset(robot)
    # Deliberately overlap a robot geom, to verify the added scene cannot bypass collision gates.
    gi=next(g for g in collision.geoms if robot.model.geom_bodyid[g]==robot.base_body)
    mesh=robot.model.geom_dataid[gi]; start=robot.model.mesh_vertadr[mesh]; count=robot.model.mesh_vertnum[mesh]
    vertices=robot.model.mesh_vert[start:start+count]
    point=data.geom_xpos[gi]+data.geom_xmat[gi].reshape(3,3)@vertices[np.argmax(vertices[:,0])]
    rr,ii,cc=scene_runtime(SceneSpec((Obstacle('hit','sphere',tuple(point),(.015,0.,0.)),),'unit_hit'))
    result=cc.scan(ii.reset(rr),0.)
    assert not result['passed'] and result['minimum_clearance_m']<0


@pytest.mark.parametrize('method',PROTOCOL['methods'])
def test_real_comparison_keeps_task_frozen_and_same_budget(physical,method):
    robot,initial,collision=physical; task,profile=reachable_task(physical)
    before=digest(task); base=public_goal_initializer(task,robot,initial)
    bundle=learning.load(CHECKPOINT,initial.model_hash,initial.state_id)
    result=candidate(method,0,task,base,123,bundle,robot,initial,collision,profile,time.time()+60)
    assert result['status']=='DONE',result.get('reason')
    assert result['shared_coupled_ceiling']==113 and result['model_calls']['coupled']<=113
    assert result['repair_coupled_calls']<=110 and result['validation_coupled_calls']<=3
    assert result['shared_wall_ceiling_s']==8. and result['wall_s']<8.
    assert digest(task)==before and not result['private_witness_fields_accessed']
    assert result['sampling']['actual_network_calls']==(16 if method=='frozen_diffusion' else 0)
    # This API has no parameter giving access to the private witness or its terminal joint state.
    assert 'witness' not in inspect.signature(candidate).parameters


def test_expired_candidate_cannot_claim_pass(physical):
    robot,initial,collision=physical; task,profile=reachable_task(physical)
    base=public_goal_initializer(task,robot,initial)
    result=candidate('structured_narrow',0,task,base,123,None,robot,initial,collision,profile,time.time()-1)
    assert result['status']=='ERROR' and not result['eligible'] and result['model_calls']['coupled']==0


def test_wider_proposals_keep_terminal_and_change_path():
    base=np.zeros((19,17)); a=proposal(base,'structured_narrow',17); b=proposal(base,'structured_wide',17)
    assert np.max(abs(b))>np.max(abs(a))*3
    assert np.allclose(b[-1],base[-1],atol=1e-15)
    assert np.array_equal(b,proposal(base,'structured_wide',17))


def test_background_duplicate_refused(tmp_path):
    from ff_prior_complex.job import Monitor,status,launch
    atomic_json(tmp_path/'run.json',{})
    with Monitor(tmp_path) as monitor:
        with pytest.raises(RuntimeError,match='alive'): launch(tmp_path/'source',tmp_path,True)
        monitor.finish({'status':'FINISHED'})
    assert status(tmp_path)['status']=='FINISHED' and not (tmp_path/'launch.guard').exists()
