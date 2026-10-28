"""Boundary algebra and data IO only; not physical planning evidence."""
import json
import hashlib
from types import SimpleNamespace
import numpy as np
import pytest
from ffgd.api import boundary_codec
from ffgd.data import reencode_free_terminal
from ffgd.trajectory import basis
from ffgd.contracts import TaskSpec,PoseInterval


def test_all_terminal_derivatives_are_free_and_initial_memory_is_fixed():
    q=np.arange(17)*.01;v=np.linspace(-.003,.003,17)
    robot=SimpleNamespace(Bplus=np.eye(17),qpos_ids=np.arange(17),dof_ids=np.arange(17),restore=lambda _:SimpleNamespace(qpos=q,qvel=v))
    reference=np.stack((q+.0001,v*.8,np.full(17,.001)))
    codec=boundary_codec(robot,None,1.2,origin='initial_position',reference_start=reference,terminal_policy='free')
    assert codec.free_dim==357
    operators=np.vstack([np.einsum('n,njf->jf',basis(24,1.2,1.2,k),codec.N.reshape(24,17,-1)) for k in (0,1,2)])
    assert np.linalg.matrix_rank(operators)==51
    z=np.random.default_rng(52).normal(0,.001,357)
    for k in (0,1,2):np.testing.assert_allclose(codec.evaluate(z,0,k),reference[k],atol=1e-10)
    assert np.linalg.norm(codec.evaluate(z,1.2,1))>.001
    assert np.linalg.norm(codec.evaluate(z,1.2,2))>.001


def test_tracking_cannot_silently_force_terminal_stop():
    interval=PoseInterval(0,1,0,[0,0,0],np.eye(3),frame='target')
    with pytest.raises(ValueError,match='FREE_TERMINAL'):TaskSpec(1,(),(interval,))
    assert TaskSpec(1,(),(interval,),terminal_policy='free').terminal_policy=='free'


def test_reencoding_keeps_every_physical_array_split_and_denominator(tmp_path):
    source=tmp_path/'source';source.mkdir();out=tmp_path/'out';out.mkdir()
    C=np.random.default_rng(2).normal(0,.01,(3,24,17));C[:,:3]=C[:,0:1];C[:,-3:]=C[:,-1:]
    arrays=dict(control=C,z=np.zeros((3,323)),duration=np.array([1.2,4,27.5]),is_train=np.array([True,False,True]),group=np.arange(3),attempt=np.array([1,4,7]),initial_state=np.zeros((3,8)))
    np.savez_compressed(source/'prior.npz',**arrays)
    manifest=dict(source_kind='FF_MODEL_PRIOR',accepted=3,attempts=8,reason_counts={'COLLISION':5,'None':3},data_hash=hashlib.sha256((source/'prior.npz').read_bytes()).hexdigest())
    (source/'dataset_manifest.json').write_text(json.dumps(manifest))
    result=reencode_free_terminal(out,dict(source_dataset=str(source)))
    assert result['free_dim']==357 and result['attempts']==8 and result['reason_counts']['COLLISION']==5
    assert result['new_physical_trajectories']==0 and result['maximum_decode_error']<1e-10
    with np.load(out/'prior.npz') as saved:
        for key in arrays:
            if key!='z':np.testing.assert_array_equal(saved[key],arrays[key])
        assert saved['z'].shape==(3,357)
