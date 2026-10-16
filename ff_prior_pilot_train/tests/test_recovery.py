import time
import numpy as np
import pytest
from ff_prior.storage import read_json
from ff_prior_pilot_train import storage,sensitivity,learning


def test_sharing_violation_retries_same_write_and_preserves_value(tmp_path,monkeypatch):
    original=storage.base.atomic_json;calls=[]
    def busy(path,value):
        calls.append(path)
        if len(calls)<3:raise PermissionError('injected Windows sharing failure')
        return original(path,value)
    monkeypatch.setattr(storage.base,'atomic_json',busy)
    storage.atomic_json(tmp_path/'proof.json',{'actual':17})
    assert len(calls)==3 and read_json(tmp_path/'proof.json')=={'actual':17}


def test_persistent_denial_is_bounded_and_never_passes(monkeypatch):
    attempts=[]
    def denied():
        attempts.append(1);raise PermissionError('injected persistent denial')
    monkeypatch.setattr(storage.time,'sleep',lambda _:None)
    with pytest.raises(PermissionError):storage.retry_write(denied)
    assert len(attempts)==8


def test_progress_io_failure_closes_item_and_allows_independent_work(physical,tmp_path):
    from ff_prior.pipeline import acceptance
    from ff_prior_modes.data import RuntimeCache
    index=read_json('artifacts/v7_1/qualified_expansion_002/training_index.json')['records']
    e=next(e for e in index if e['split']=='TRAIN');r=read_json(e['record'])
    with np.load(e['arrays']) as a:z=a['Z'].copy()
    # Use saved full-batch fitted codec; this is failure handling, no synthetic physics pass.
    from ff_prior_latent.learning import Codec
    codec=Codec(read_json('artifacts/v7_1/qualified_pilot_train_001/learning/codec.json'))
    arrays=(z[None],None,None,[e['sample_id']],['TRAIN'],None)
    canonical,initial,_=physical
    cache=RuntimeCache(canonical,initial,acceptance(read_json('artifacts/v7_1/smoke_002/run.json')['config']))
    ledger=storage.Ledger(tmp_path,{})
    def fail(*a,**k):raise PermissionError('injected progress write denial')
    matrices,report=sensitivity.prepare(tmp_path,arrays,[r],codec,cache,ledger,time.time()+30,fail)
    assert matrices is None and report['status']=='BLOCKED' and ledger.data['inflight'] is None
    assert ledger.data['calls']['sensitivity.coupled']==0
    ledger.begin('independent','candidate',30,1);ledger.finish({'status':'TEST_ONLY'})
    assert ledger.data['inflight'] is None


def test_numeric_ast_change_detected():
    from ff_prior_pilot_train.reuse import function_ast
    assert function_ast('def build(x):\n return x+1','build')!=function_ast('def build(x):\n return x+2','build')
