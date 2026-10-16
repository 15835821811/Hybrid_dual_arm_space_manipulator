import numpy as np
import pytest
from ff_prior.storage import read_json,file_digest
from ff_prior_origin_scene.runtime import at_scene_origin as previous
from ff_prior_scene_guard.runtime import at_scene_origin,InitialGeometryRejected


def test_all_six_retained_initial_penetrations_are_explicitly_rejected(physical):
    report=read_json('ff_prior_pilot_tools/evidence/origin_contact_diagnostic_001/diagnostic.json')
    for case in report['cases']:
        path=case['source'];before=file_digest(path);r=read_json(path)
        with pytest.raises(InitialGeometryRejected) as rejected:
            at_scene_origin(physical[0],physical[1],r['scene'],r['q17_origin'])
        assert rejected.value.geometry['minimum_clearance_m']<0
        assert rejected.value.geometry['minimum_clearance_m']==case['actual_initial_geometry']['minimum_clearance_m']
        assert file_digest(path)==before and read_json(path)['status']=='ERROR'


def test_safe_scene_keeps_exact_full_state_and_every_transfer_check(physical):
    row=read_json('artifacts/v7_1/qualified_origin_coverage_001/records/origin_train_003_a0.json')
    args=(physical[0],physical[1],row['scene'],row['condition']['initial_features'])
    old=previous(*args);new=at_scene_origin(*args)
    assert old[1].state_id==new[1].state_id==row['task']['contract']['initial_state_id']
    assert np.array_equal(old[1].values,new[1].values)
    assert old[3]['scene_transfer']==new[3]['scene_transfer']
    assert len(new[3]['scene_transfer']['checks'])==187
    assert all(new[3]['scene_transfer']['checks'].values())


def test_canonical_state_identity_is_still_required(physical):
    from ff_prior.initial_state import InitialState
    from ff_prior.scenes import SceneSpec
    wrong=InitialState(physical[1].values,'tampered_model')
    with pytest.raises(ValueError,match='identity mismatch'):
        at_scene_origin(physical[0],wrong,SceneSpec(),physical[0].spec.planner_zero)
