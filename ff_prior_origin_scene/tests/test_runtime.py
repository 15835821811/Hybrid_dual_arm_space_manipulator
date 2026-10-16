"""Real retained failure scenes and all frozen new task states; no mock physics."""
import copy
from pathlib import Path
import numpy as np
import pytest
from ff_prior.storage import read_json,file_digest
from ff_prior.pipeline import audit,acceptance
from ff_prior_complex.tasks import scene_runtime,transfer_audit
from ff_prior_loop.runtime import at_origin
from ff_prior.scenes import SceneSpec
from ff_prior_origin_scene.runtime import at_scene_origin,TaskOriginSceneCache

ROOT=Path(__file__).resolve().parents[2]


@pytest.fixture(scope='module')
def physical():
    config=read_json(ROOT/'artifacts/v7_1/smoke_002/run.json')['config']
    _,rr,ii,_=audit(config)
    cases=read_json(ROOT/'ff_prior_pilot_tools/evidence/scene_state_diagnostic_002/diagnostic.json')['cases']
    return rr,ii,acceptance(config),cases


@pytest.mark.parametrize('slot',[0,1])
def test_retained_home_failure_passes_exact_actual_origin(physical,slot):
    rr,ii,_,cases=physical;c=cases[slot];scene=SceneSpec.from_dict(c['scene'])
    child,home,_=scene_runtime(scene)
    with pytest.raises(ValueError,match='integration_state'):
        transfer_audit(rr,ii,child,home)
    q0=c['actual_origin_audit']['planning_origin']
    robot,initial,collision,proof=at_scene_origin(rr,ii,scene,q0)
    parent,pi,_=at_origin(rr,q0)
    assert np.array_equal(pi.values,initial.values)
    assert proof['scene_transfer']['status']=='PASS' and all(proof['scene_transfer']['checks'].values())
    assert proof['complete_state_fields_preserved'] and proof['initial_geometry']['passed']
    assert proof['initial_geometry']['minimum_clearance_m']>=0
    assert robot.calls['physics_steps']==robot.calls['torque_rollout']==robot.calls['coupled']==0


@pytest.mark.parametrize('slot',[0,1])
def test_actual_home_contact_still_rejected(physical,slot):
    rr,ii,_,cases=physical;scene=SceneSpec.from_dict(cases[slot]['scene'])
    # Using home as the actual task origin must still fail. This is not a
    # removal of the warmstart check or permission to penetrate geometry.
    with pytest.raises(ValueError):at_scene_origin(rr,ii,scene,rr.spec.planner_zero)


def test_all_twelve_frozen_tasks_preserve_saved_full_state(physical):
    rr,ii,p,_=physical;cache=TaskOriginSceneCache(rr,ii,p)
    source=ROOT/'artifacts/v7_1/independent_new_validation_001'
    entries=read_json(source/'frozen_tasks.json')['entries']
    assert len(entries)==12
    for e in entries:
        assert e['status']=='FROZEN'
        path=source/'tasks'/e['sample_id']/'initial_state.npz';before=file_digest(path)
        robot,initial,_,proof=cache.get(e)
        with np.load(path) as a:assert np.array_equal(a['values'],initial.values)
        assert initial.state_id==e['task']['contract']['initial_state_id']
        assert proof['origin']['status']=='PASS' and proof['initial_geometry']['passed']
        assert cache.get(e)[0] is robot and file_digest(path)==before
    bad=copy.deepcopy(entries[0]);bad['task']['contract']['initial_state_id']='wrong'
    with pytest.raises((ValueError,AssertionError)):cache.get(bad)
