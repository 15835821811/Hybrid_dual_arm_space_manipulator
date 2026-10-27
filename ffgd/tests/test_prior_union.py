"""Synthetic IO tests only; these fixtures are not physical training evidence."""
import hashlib
import json
import numpy as np
import pytest
from ffgd.data import merge_prior


def fixture(path,offset=0,token_count=3,origin='initial_position'):
    path.mkdir()
    arrays=dict(z=np.arange(12).reshape(4,3)+offset,control=np.arange(24).reshape(4,3,2)+offset,
        initial_state=np.arange(20).reshape(4,5)+offset,tokens=np.ones((4,token_count,32))*offset,
        padding=np.zeros((4,token_count),bool),duration=np.full(4,5.),
        is_train=np.array([True,True,False,False]),group=np.array([1,1,2,2]),attempt=np.arange(4))
    np.savez_compressed(path/'prior.npz',**arrays)
    manifest=dict(source_kind='FF_MODEL_PRIOR',accepted=4,attempts=7,model_hash='fixture_model',state_mask=1,
        control_points=3,free_dim=3,geometry_policy_hash='fixture_geometry',boundary_origin=origin,
        profile={'fixture':True},data_hash=hashlib.sha256((path/'prior.npz').read_bytes()).hexdigest(),
        reason_counts={'None':4,'COLLISION':3})
    (path/'dataset_manifest.json').write_text(json.dumps(manifest))
    return arrays


def test_union_preserves_examples_splits_and_failed_denominator(tmp_path):
    a=tmp_path/'a';b=tmp_path/'b';out=tmp_path/'out';out.mkdir()
    first=fixture(a);second=fixture(b,9,5)
    result=merge_prior(out,{'source_datasets':[str(a),str(b)]})
    assert result['attempts']==14 and result['accepted']==8 and result['reason_counts']['COLLISION']==6
    assert result['new_physical_trajectories']==0
    with np.load(out/'prior.npz') as data:
        np.testing.assert_array_equal(data['control'][:4],first['control'])
        np.testing.assert_array_equal(data['initial_state'][4:],second['initial_state'])
        assert data['padding'][:4,3:].all()
        assert not (set(data['group'][data['is_train']]) & set(data['group'][~data['is_train']]))
        assert not (set(data['group'][:4]) & set(data['group'][4:]))
        np.testing.assert_array_equal(data['is_train'],np.tile(first['is_train'],2))


def test_union_rejects_incompatible_coordinate_system(tmp_path):
    a=tmp_path/'a';b=tmp_path/'b';out=tmp_path/'out';out.mkdir()
    fixture(a);fixture(b,9,origin='pivot_zero')
    with pytest.raises(ValueError,match='PRIOR_MERGE_CONTRACT_MISMATCH'):
        merge_prior(out,{'source_datasets':[str(a),str(b)]})
    assert not (out/'prior.npz').exists()


def test_union_rejects_modified_parent_and_split_leak(tmp_path):
    a=tmp_path/'a';b=tmp_path/'b';out=tmp_path/'out';out.mkdir()
    fixture(a);arrays=fixture(b,9)
    arrays['group']=np.ones(4,dtype=int)
    np.savez_compressed(b/'prior.npz',**arrays)
    with pytest.raises(ValueError,match='SOURCE_DATASET_IDENTITY_MISMATCH'):
        merge_prior(out,{'source_datasets':[str(a),str(b)]})
    p=b/'dataset_manifest.json';m=json.loads(p.read_text());m['data_hash']=hashlib.sha256((b/'prior.npz').read_bytes()).hexdigest();p.write_text(json.dumps(m))
    with pytest.raises(ValueError,match='FAMILY_SPLIT_LEAK'):
        merge_prior(out,{'source_datasets':[str(a),str(b)]})
