import numpy as np
import pytest
from ff_prior_source_alignment.acceptance import compare
from ff_prior_source_alignment.audit_local import record_problem


def case(i, p, r):
    return dict(sample_id=str(i), t=999, seed=i, position_m=[p,p], rotation_rad=[r,r],
                loss=((p/.005)**2+(r/np.deg2rad(1))**2)/2)


def test_scalar_improvement_does_not_hide_orientation_failure():
    warm=[case(0,.02,.01)]
    after=[case(0,.004,.02)]
    r=compare(warm,after,warm)
    assert r['median_paired_loss_ratio_vs_warm'] < .8
    assert r['TRAIN_endpoint_gate']=='BLOCKED'
    assert not r['gates']['rotation_median_not_increased']


def test_one_good_case_cannot_be_sacrificed_for_aggregate_loss():
    warm=[case(0,.004,.01),case(1,.04,.04),case(2,.04,.04)]
    after=[case(0,.006,.01),case(1,.004,.01),case(2,.004,.01)]
    r=compare(warm,after,warm)
    assert r['pose_pass'] > r['warm_pose_pass']
    assert r['lost_qualified']==1 and r['TRAIN_endpoint_gate']=='BLOCKED'


def test_exact_physical_boundary_and_all_conditions():
    warm=[case(0,.01,np.deg2rad(2))]
    after=[case(0,.005,np.deg2rad(1))]
    r=compare(warm,after,warm)
    assert r['pose_pass']==1 and r['TRAIN_endpoint_gate']=='PASS'
    assert r['automatic_DEV_or_TEST_launch'] is False


@pytest.mark.parametrize('mutation', ['identity','nan','missing_arm','loss','duplicate'])
def test_bad_or_unpaired_evidence_is_rejected(mutation):
    warm=[case(0,.01,.02)]
    after=[case(0,.005,.01)]
    if mutation=='identity':after[0]['seed']=42
    if mutation=='nan':after[0]['rotation_rad'][0]=float('nan')
    if mutation=='missing_arm':after[0]['position_m']=[.005]
    if mutation=='loss':after[0]['loss']=0
    if mutation=='duplicate':after=after+after
    with pytest.raises(ValueError):compare(warm,after,warm)


def test_heldout_val_is_not_required_to_be_training_eligible():
    e={'split':'VAL','validation_level':'MODEL_VALID'}
    row={'training_eligible':False,'execution_status':'NOT_RUN'}
    assert record_problem(e,row) is None
    assert record_problem(dict(e,split='TRAIN'),row)=='TRAIN not eligible'
    assert record_problem(e,dict(row,execution_status='FAIL'))=='execution failed'


def test_test_records_and_unsupported_levels_are_rejected():
    assert record_problem({'split':'TEST'}, {})=='forbidden partition'
    assert record_problem({'split':'TRAIN','validation_level':'PROPOSED'}, {})=='unsupported validation level'
