"""Real saved TRAIN/VAL and optimizer smoke; not formal 12k results."""
import copy
import os
import time
from pathlib import Path
import numpy as np
import pytest
from ff_prior.storage import read_json,atomic_json,file_digest
from ff_prior.diffusion import weight_hash
from ff_prior_latent.learning import Codec
from ff_prior_pilot_train.storage import Ledger
from ff_prior_coverage_train import learning,sensitivity


@pytest.fixture(scope='session')
def actual():
    source=learning.FROZEN.parent.parent
    index=read_json(source/'training_index.json')['records']
    rows=[read_json(e['record']) for e in index]
    with np.load(source/'loaded_training_arrays.npz') as a:z,bases=a['Z'].copy(),a['bases'].copy()
    arrays=(z,bases,[r['condition'] for r in rows],[e['sample_id'] for e in index],
            [e['split'] for e in index],[r['task']['contract']['initial_state_id'] for r in rows])
    return arrays,rows


def test_frozen_train_codec_and_exact_cache_then_real_weight_update(actual,tmp_path,monkeypatch):
    arrays,rows=actual;codec=learning.fit_codec(tmp_path/'learning',arrays)
    slots=[i for i,s in enumerate(arrays[4]) if s=='TRAIN'][:6]+[i for i,s in enumerate(arrays[4]) if s=='VAL'][:2]
    small=tuple(v[slots] if isinstance(v,np.ndarray) else [v[i] for i in slots] for v in arrays)
    sr=[rows[i] for i in slots];out=tmp_path/'run';ledger=Ledger(out,{})
    # These exact anchors were physically propagated in the completed parent.
    matrices,proof=sensitivity.prepare(out,small,sr,codec,None,ledger,time.time()+120,lambda *a,**k:None)
    assert proof['status']=='PASS' and len(matrices)==6
    assert ledger.data['calls']['cached_sensitivity.coupled']==39*proof['anchors']
    assert ledger.data['calls']['cached_sensitivity.new_coupled']==0 and 'sensitivity.coupled' not in ledger.data['calls']
    monkeypatch.setitem(learning.CONTRACT,'updates',4);monkeypatch.setitem(learning.CONTRACT,'batch',2)
    result=learning.train(out/'model',small,codec,matrices,'goal_loss',proof,lambda *a,**k:None,time.time()+120)
    parent,_=learning.frozen_learning.load(learning.FROZEN)
    assert result['status']=='PASS' and result['initial_weight_hash']==weight_hash(parent)
    assert result['trained_weight_hash']!=result['initial_weight_hash'] and result['VAL_gradient_updates']==0
    assert sum(result['train_draw_counts'].values())==8
    before={str(p):file_digest(p) for p in (out/'model').rglob('*') if p.is_file()}
    again=learning.train(out/'model',small,codec,matrices,'goal_loss',proof,lambda *a,**k:None,time.time()+120)
    assert again==result and before=={str(p):file_digest(p) for p in (out/'model').rglob('*') if p.is_file()}


def test_mismatched_basis_or_reference_cannot_reuse_cache(actual,tmp_path):
    arrays,rows=actual;codec=Codec(read_json(learning.FROZEN/'codec.json'))
    i=next(i for i,s in enumerate(arrays[4]) if s=='TRAIN');key=rows[i]['task']['task_hash']
    state=copy.deepcopy(codec.state);state['basis']=np.asarray(state['basis']);state['basis'][:,0]*=-1
    with pytest.raises(ValueError,match='identical frozen codec'):
        sensitivity.import_anchor(tmp_path,key,i,arrays,Codec(state),Ledger(tmp_path,{}))
    bad=list(arrays);bad[0]=arrays[0].copy();bad[0][i,0,0]+=1e-5
    with pytest.raises(ValueError,match='anchor/proof mismatch'):
        sensitivity.import_anchor(tmp_path,key,i,bad,codec,Ledger(tmp_path,{}))


def test_live_parent_and_empty_run_do_not_pass_gates(tmp_path):
    from ff_prior_coverage_train.experiment import require_data,report
    source=tmp_path/'source'
    atomic_json(source/'status.json',{'status':'FINISHED','pid':os.getpid()})
    atomic_json(source/'summary.json',{'ready_for_training_stage':True,'coverage_data_gate_passed':True})
    with pytest.raises(ValueError,match='completed and exited'):require_data(source)
    r=report(tmp_path/'out',Ledger(tmp_path/'out',{}),'FINISHED')
    assert not r['coverage_training_closure_passed'] and not r['actual_training_passed']


def test_missing_sensitivity_blocks_parameter_updates(tmp_path):
    r=learning.train(tmp_path,None,None,None,'goal_loss',{},lambda *a,**k:None,time.time()+10)
    assert r['status']=='BLOCKED' and not (tmp_path/'model.pt').exists()
