import os
import time
import numpy as np
import pytest
import torch
from ff_prior.storage import atomic_json,read_json
from ff_prior.contracts import frozen_task
from ff_prior.dataset import Scaler
from ff_prior.diffusion import NoiseTransformer
from ff_prior.clean_prior import TemporalPrior
from ff_prior_next.learning import OffsetScaler,sample,train,load
from ff_prior_next.job import Monitor,status,launch


def condition():
    return {'initial_features':np.zeros(17),'goals':np.tile(np.eye(4),(2,1,1)),
            'duration':10.,'pose_mask':np.ones((2,6))}


def test_residual_sampling_returns_physical_guided_bound_without_roundtrip():
    torch.set_num_threads(2)
    c=condition(); base=np.full((19,17),.123456789)
    residual=np.stack([np.linspace(0,1,19)[:,None]*np.ones((1,17))*i*.02 for i in range(1,5)])
    scaler=Scaler.fit(residual,[c]*4,['TRAIN']*4,'robot','initial')
    prior=TemporalPrior.fit(residual,np.zeros(17),list('abcd'),['TRAIN']*4,'robot','initial')
    transform=OffsetScaler(scaler,base)
    np.testing.assert_allclose(transform.inverse_z(transform.z(base+residual[0])),base+residual[0],atol=1e-16)
    model=NoiseTransformer(len(scaler.condition(c)),hidden=32,layers=1,heads=4); model.prediction_type='v'
    target=np.full((19,17),np.pi); seen=[]
    def guide(z,i):
        seen.append(i); return target.copy(),{'iteration':i}
    z,log=sample(model,scaler,prior,base,frozen_task(c,{}),9,guide)
    np.testing.assert_array_equal(z,target)
    assert seen==[14,15] and log['actual_network_calls']==16
    assert max(v['decomposition_residual_after'] for v in log['steps'])<1e-12


def test_residual_training_rejects_dev_or_test_before_writing(tmp_path):
    for split in ('DEV','TEST'):
        with pytest.raises(ValueError,match='TRAIN/VAL'):
            train(tmp_path,np.zeros((2,19,17)),np.zeros((2,19,17)),[condition()]*2,
                ['a','b'],['TRAIN',split],'robot','initial',lambda *a,**k:None,time.time()+10)
    assert not (tmp_path/'model.pt').exists()


def test_monitor_terminal_status_and_duplicate_launch_protection(tmp_path):
    atomic_json(tmp_path/'run.json',{'test_fixture':True})
    with Monitor(tmp_path) as monitor:
        monitor.update('software_test',completed_candidates=7)
        state=status(tmp_path)
        assert state['status']=='RUNNING' and state['owner_alive'] and state['pid']==os.getpid()
        with pytest.raises(RuntimeError,match='alive'): launch(tmp_path/'source',tmp_path,resume=True)
        assert not (tmp_path/'launch.guard').exists()
        monitor.finish({'status':'FINISHED'})
    assert status(tmp_path)['status']=='FINISHED'
    assert len((tmp_path/'events.jsonl').read_text().splitlines())==2


def test_status_detects_dead_process_without_deleting_lock(tmp_path):
    import subprocess,sys
    child=subprocess.Popen([sys.executable,'-c','pass'],creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
    child.wait()
    atomic_json(tmp_path/'status.json',{'status':'RUNNING','pid':child.pid,'heartbeat':time.time()})
    atomic_json(tmp_path/'run.lock',{'pid':child.pid,'created':time.time()})
    assert status(tmp_path)['status']=='INTERRUPTED_REQUIRES_EXPLICIT_RESUME'
    assert (tmp_path/'run.lock').exists()
