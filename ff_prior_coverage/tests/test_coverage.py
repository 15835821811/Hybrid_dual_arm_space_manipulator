"""Real physics admission checks; test outputs are not experiment outcomes."""
import time
import numpy as np
import pytest
from ff_prior.storage import read_json,atomic_json,digest
from ff_prior.pipeline import acceptance
from ff_prior.scenes import SceneSpec,Obstacle
from ff_prior_complex.tasks import scene_runtime
from ff_prior_loop.runtime import at_origin
from ff_prior_modes.data import RuntimeCache
from ff_prior_pilot_train.storage import Ledger
from ff_prior_coverage import data


def profile():
    return acceptance(read_json('artifacts/v7_1/smoke_002/run.json')['config'])


@pytest.fixture(scope='session')
def contexts(physical):
    robot,initial,_=physical
    return data.contexts(robot,initial,profile(),data.old_index())


def test_new_contexts_preserve_robot_and_expand_static_scene_coverage(physical,contexts):
    robot,initial,_=physical
    assert len(contexts)==24 and len({v[2].state_id for v in contexts})==24
    prior=read_json(data.PREVIOUS/'contexts.json')
    assert not {v[2].state_id for v in contexts[:12]} & {r['initial_state_id'] for r in prior}
    old_scenes={digest(r['scene']) for r in prior}
    new_scenes={digest(v[0]['scene']) for v in contexts[12:]}
    assert len(new_scenes)==12 and not old_scenes & new_scenes
    for meta,rr,ii,_ in contexts:
        assert np.array_equal(rr.B,robot.B) and np.array_equal(rr.model.opt.gravity,robot.model.opt.gravity)
        assert not np.any(ii.reset(rr).qvel)


def test_actual_forward_then_fixed_world_target_mode(physical,contexts,tmp_path):
    robot,initial,_=physical;pr=profile()
    context=contexts[0]
    r=data.generate(0,context,tmp_path,pr,[],set(),time.time()+60)
    assert r['admitted'],r
    assert r['calls']['physics_steps']==0 and r['calls']['coupled']<=4
    atomic_json(tmp_path/'generation'/f'{r["sample_id"]}.json',r)
    row=data.check_new_record(tmp_path,tmp_path/'records'/f'{r["sample_id"]}.json')
    with np.load(tmp_path/row['arrays']) as a:z=a['Z'].copy()
    entry={'sample_id':'coverage_mode_00','slot':0,'split':'TRAIN','parent_sample_id':row['sample_id'],
        'family_id':row['family_id'],'task':row['task'],'scene':row['scene'],'context_id':row['context_id'],'Z':z}
    identity=digest(entry['task']);cache=RuntimeCache(robot,initial,pr)
    mode=data.generate_mode(entry,0,cache,tmp_path,pr,time.time()+60)
    assert mode['admitted'],mode
    atomic_json(tmp_path/'generation'/f'{mode["sample_id"]}.json',mode)
    m=data.check_new_record(tmp_path,tmp_path/'records'/f'{mode["sample_id"]}.json')
    assert digest(m['task'])==identity==digest(entry['task'])
    assert mode['minimum_reference_rms_rad']>=.005 and mode['calls']['coupled']<=113
    assert m['validation_level']=='MODEL_VALID' and m['execution_status']=='NOT_RUN'
    assert max(m['validation']['frozen_goal_position_error_m'])<=.005


def test_heldout_goal_and_real_penetration_are_rejected(physical,contexts,tmp_path):
    robot,initial,_=physical;pr=profile()
    context=contexts[0]
    first=data.generate(0,context,tmp_path/'first',pr,[],set(),time.time()+60)
    assert first['admitted']
    task=read_json(tmp_path/'first/tasks'/first['sample_id']/'task.json')
    near=data.generate(0,context,tmp_path/'near',pr,[('reserved',task['condition']['goals'])],set(),time.time()+60)
    assert not near['admitted'] and near['reason']=='HELDOUT_GOAL_NEIGHBOUR'
    assert read_json(tmp_path/'near/tasks'/first['sample_id']/'task.json')==task
    scene=SceneSpec((Obstacle('penetrating','sphere',(0.,0.,0.),(2.,0.,0.)),),'test_penetration')
    rr,_,_=scene_runtime(scene);rr,ii,cc=at_origin(rr,context[1].spec.planner_zero)
    blocked=data.generate(0,({'context_id':'blocked','scene':scene.to_dict(),'family_id':'test_penetration'},rr,ii,cc),
                          tmp_path/'blocked',pr,[],set(),time.time()+60)
    assert not blocked['admitted'] and blocked['status']=='REJECTED',blocked
    assert blocked['reason']=='INDEPENDENT_VALIDATION_REJECTED'
    assert blocked['validation_result']['validation']['geometry']['minimum_clearance_m']<0
    assert (tmp_path/'blocked/tasks'/first['sample_id']/'task.json').exists()
    assert not (tmp_path/'blocked/records'/f'{first["sample_id"]}.json').exists()


def test_uncommitted_record_is_never_admitted(physical,contexts,tmp_path):
    robot,initial,_=physical;pr=profile()
    context=contexts[0]
    r=data.generate(0,context,tmp_path,pr,[],set(),time.time()+60)
    assert r['admitted']
    atomic_json(tmp_path/'generation'/f'{r["sample_id"]}.json',r)
    with pytest.raises(ValueError,match='uncommitted'):
        data.check_new_record(tmp_path,tmp_path/'records'/f'{r["sample_id"]}.json',Ledger(tmp_path,{}))


def test_empty_output_and_live_duplicate_are_blocked(tmp_path):
    from ff_prior_coverage.experiment import report
    from ff_prior_coverage.job import Monitor,launch
    r=report(tmp_path,Ledger(tmp_path,{}),'FINISHED')
    assert not r['ready_for_training_stage'] and not r['full_goal_achieved']
    atomic_json(tmp_path/'run.json',{})
    with Monitor(tmp_path) as monitor:
        with pytest.raises(RuntimeError,match='alive'):launch(tmp_path/'source',tmp_path,True)
        monitor.finish({'status':'FINISHED'})
