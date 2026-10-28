import numpy as np
import pytest
from ffgd.model_io import bind_free_coordinates
from ffgd.trajectory import Boundary,BoundaryCodec


def checkpoint(schema='duration_squared_v1'):
    return dict(dataset_manifest={'free_coordinate_schema':schema},mean=np.array([.1,.2]),scale=np.array([.3,.4]))


def test_binding_is_reversible_nonmutating_and_rejects_unit_confusion():
    raw=checkpoint();before=raw['mean'].copy()
    for duration in (.8,1.2,4.,5.):
        bound=bind_free_coordinates(raw,duration)
        np.testing.assert_array_equal(raw['mean'],before)
        np.testing.assert_allclose(bound['mean']/duration**2,raw['mean'])
        assert bind_free_coordinates(bound,duration) is bound
        with pytest.raises(ValueError,match='ALREADY_BOUND'):
            bind_free_coordinates(bound,duration+1)
    old=checkpoint('physical_position_v1');assert bind_free_coordinates(old,.8) is old
    with pytest.raises(ValueError,match='UNSUPPORTED'):
        bind_free_coordinates(checkpoint('unknown'),1.)
    for duration in (0.,-1.,np.nan,np.inf):
        with pytest.raises(ValueError,match='INVALID'):bind_free_coordinates(raw,duration)


def test_duration_squared_coordinates_preserve_full_space_and_acceleration_units():
    rng=np.random.default_rng(713);q=rng.normal(size=17);labels=rng.normal(0,.001,323)
    accelerations=[]
    for T in (.8,1.2,4.,5.):
        boundaries=[Boundary(0,k,tuple(range(17)),tuple(q) if k==0 else (0.,)*17) for k in (0,1,2)]
        boundaries += [Boundary(T,k,tuple(range(17)),(0.,)*17) for k in (1,2)]
        codec=BoundaryCodec(24,17,T,boundaries,origin='initial_position')
        physical=labels*T*T;C=codec.decode(physical)
        assert codec.free_dim==323
        np.testing.assert_allclose(codec.encode(C)/(T*T),labels,atol=1e-13,rtol=0)
        np.testing.assert_allclose(codec.A@C.ravel(),codec.b,atol=1e-10,rtol=0)
        accelerations.append(codec.evaluate(physical,np.linspace(0,T,65),2))
    for a in accelerations[1:]:np.testing.assert_allclose(a,accelerations[0],atol=1e-10,rtol=0)


def test_dataset_reencoding_keeps_physical_examples_splits_and_attempts(tmp_path):
    # Synthetic IO evidence only; no native validation is claimed by this test.
    import json,hashlib
    from ffgd.data import reencode_duration_coordinates
    source=tmp_path/'source';source.mkdir();out=tmp_path/'out';out.mkdir()
    rng=np.random.default_rng(19);controls=[];z=rng.normal(0,.001,(2,323));durations=np.array([4.,5.])
    for i,T in enumerate(durations):
        q=rng.normal(size=17)
        bc=[Boundary(0,k,tuple(range(17)),tuple(q) if k==0 else (0.,)*17) for k in (0,1,2)]
        bc += [Boundary(T,k,tuple(range(17)),(0.,)*17) for k in (1,2)]
        controls.append(BoundaryCodec(24,17,T,bc,origin='initial_position').decode(z[i]))
    arrays=dict(z=z,control=np.array(controls),duration=durations,is_train=np.array([True,False]),
                group=np.array([0,1]),attempt=np.array([2,5]),initial_state=np.zeros((2,7)))
    np.savez_compressed(source/'prior.npz',**arrays)
    parent=dict(source_kind='FF_MODEL_PRIOR',boundary_origin='initial_position',accepted=2,attempts=7,
                data_hash=hashlib.sha256((source/'prior.npz').read_bytes()).hexdigest())
    (source/'dataset_manifest.json').write_text(json.dumps(parent))
    result=reencode_duration_coordinates(out,dict(source_dataset=str(source)))
    assert result['passed'] and result['attempts']==7 and result['new_physical_trajectories']==0
    assert result['maximum_decode_error']<1e-10
    with np.load(out/'prior.npz') as saved:
        np.testing.assert_allclose(saved['z']*durations[:,None]**2,z,atol=1e-17,rtol=0)
        for key in arrays:
            if key!='z':np.testing.assert_array_equal(saved[key],arrays[key])
