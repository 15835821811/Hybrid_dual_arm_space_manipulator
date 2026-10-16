import os
import time
from pathlib import Path
import numpy as np
import pytest
from ff_prior.storage import read_json,atomic_json,Ledger,file_digest
from ff_prior.pipeline import acceptance
from ff_prior_modes.data import RuntimeCache
from ff_prior_pilot_train import learning,sensitivity


def test_actual_saved_arrays_train_with_real_coupled_loss_and_completed_reuse(physical,tmp_path,monkeypatch):
    source=Path('artifacts/v7_1/qualified_expansion_002');index=read_json(source/'training_index.json')['records']
    slots=[i for i,e in enumerate(index) if e['split']=='TRAIN'][:6]+[i for i,e in enumerate(index) if e['split']=='VAL'][:2]
    selected=[index[i] for i in slots];rows=[read_json(e['record']) for e in selected]
    with np.load(source/'loaded_training_arrays.npz') as a:z,bases=a['Z'][slots],a['bases'][slots]
    arrays=(z,bases,[r['condition'] for r in rows],[e['sample_id'] for e in selected],
        [e['split'] for e in selected],[r['task']['contract']['initial_state_id'] for r in rows])
    canonical,initial,_=physical;profile=acceptance(read_json('artifacts/v7_1/smoke_002/run.json')['config'])
    codec=learning.fit_codec(tmp_path,arrays);ledger=Ledger(tmp_path,{})
    matrices,report=sensitivity.prepare(tmp_path,arrays,rows,codec,RuntimeCache(canonical,initial,profile),ledger,time.time()+120,lambda *a,**k:None)
    assert report['status']=='PASS' and len(matrices)==6 and report['anchors']<=6
    assert ledger.data['calls']['sensitivity.coupled']==39*report['anchors']
    assert report['contract']['max_anchors']==11000
    # Small real optimizer smoke is software evidence, not the 10000-row experiment.
    monkeypatch.setitem(learning.CONTRACT,'updates',4);monkeypatch.setitem(learning.CONTRACT,'batch',2)
    out=tmp_path/'model'
    r=learning.train(out,arrays,codec,matrices,'goal_loss',report,lambda *a,**k:None,time.time()+120)
    assert r['status']=='PASS' and r['updates']==4 and sum(r['train_draw_counts'].values())==8
    assert len(r['train_draw_counts'])==6 and r['VAL_gradient_updates']==0 and r['trained_weight_hash']!=r['initial_weight_hash']
    assert set(codec.state['fit_ids'])==set(arrays[3][:6])
    before={str(p):file_digest(p) for p in out.rglob('*') if p.is_file()}
    again=learning.train(out,arrays,codec,matrices,'goal_loss',report,lambda *a,**k:None,time.time()+120)
    assert again==r and before=={str(p):file_digest(p) for p in out.rglob('*') if p.is_file()}


def test_completed_marker_cannot_bypass_live_data_worker(tmp_path):
    from ff_prior_pilot_train.experiment import require_data
    atomic_json(tmp_path/'status.json',{'status':'FINISHED','pid':os.getpid()})
    atomic_json(tmp_path/'summary.json',{'ready_for_training_stage':True,'pilot_data_gate_passed':True})
    with pytest.raises(ValueError,match='completed and exited'):require_data(tmp_path)


def test_empty_output_never_passes_actual_ingestion_or_closure(tmp_path):
    from ff_prior_pilot_train.experiment import report
    result=report(tmp_path,Ledger(tmp_path,{}),'FINISHED')
    assert not result['pilot_training_closure_passed'] and not result['actual_training_passed'] and not result['full_goal_achieved']


def test_missing_sensitivities_block_neural_training(tmp_path):
    r=learning.train(tmp_path,None,None,None,'goal_loss',{},lambda *a,**k:None,time.time()+10)
    assert r['status']=='BLOCKED' and not (tmp_path/'model.pt').exists()
