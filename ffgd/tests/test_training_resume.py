"""Two-update CPU software fixture. Not real prior or learning-benefit evidence."""
import json
import numpy as np
import pytest
import torch
import ffgd.training as training
from ffgd.robot import Robot
from ffgd.orchestration import atomic_json
from ffgd.model_io import file_hash


def test_training_rng_optimizer_resume_exact(tmp_path,monkeypatch):
    torch.set_num_threads(1);r=Robot();root=tmp_path/'fixture';root.mkdir()
    rng=np.random.default_rng(1);z=rng.normal(size=(8,23));tokens=rng.normal(size=(8,3,32)).astype('float32')
    np.savez_compressed(root/'prior.npz',z=z,tokens=tokens,padding=np.zeros((8,3),bool),is_train=np.array([1,1,1,1,0,0,0,0],bool),group=np.repeat(np.arange(4),2))
    atomic_json(root/'dataset_manifest.json',dict(accepted=8,source_kind='SOFTWARE_FIXTURE',model_hash=r.model_hash,data_hash=file_hash(root/'prior.npz')))
    config=dict(scope='SOFTWARE_TEST',seed=55,device='cpu',dataset=str(root),minimum_dataset_size=8,
                model=dict(max_free=30,condition_dim=32,hidden=16,layers=1,heads=2),batch_size=2,
                updates=2,learning_rate=2e-4,weight_decay=.01,ema_decay=.9,log_every=1,validate_every=1)
    complete=tmp_path/'complete';complete.mkdir();training.train(complete,config)
    interrupted=tmp_path/'interrupted';interrupted.mkdir();calls=0
    def interrupt():
        nonlocal calls
        calls+=1
        if calls==2:raise RuntimeError('TEST_INTERRUPTION')
    monkeypatch.setattr(training,'check_budget',interrupt)
    with pytest.raises(RuntimeError,match='TEST_INTERRUPTION'):training.train(interrupted,config)
    monkeypatch.setattr(training,'check_budget',lambda:None)
    resumed=tmp_path/'resumed';resumed.mkdir();training.train(resumed,config,str(interrupted/'interrupted.pt'))
    a=torch.load(complete/'last.pt',weights_only=False);b=torch.load(resumed/'last.pt',weights_only=False)
    assert a['update']==b['update']==2
    for key in ('model','ema'):
        for name in a[key]:torch.testing.assert_close(a[key][name],b[key][name],atol=0,rtol=0)
    np.testing.assert_array_equal(a['mean'],z[:4].mean(0))
