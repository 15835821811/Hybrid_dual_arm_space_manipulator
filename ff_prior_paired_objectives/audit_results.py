import hashlib,zipfile,subprocess
import numpy as np
import torch
from ff_prior.storage import read_json,file_digest,digest
from ff_prior.diffusion import weight_hash
from ff_prior_origin_learn_bounded.distribution import CoverageSampler
from ff_prior_world_endpoint.audit_results import verify_gradient,verify_smoke
from ff_prior_world_endpoint.endpoint import loss_from_ee
from .protocol import ROOT,SOURCE,WARM,ARMS,PROTOCOL as P
from .identity import sources
from .objectives import predict,random_batch
from .learning import load,select

def verify_training(out,codec,data,arrays,rows):
    ledger=read_json(out/'ledger.json');mapping={sid:i for i,sid in enumerate(arrays[3])}
    seq=read_json(out/'paired_random_sequence.json');assert len(seq)==6000
    sampler=CoverageSampler(data['keys'],P['training_seed']+2)
    for step,r in enumerate(seq):
        batch=sampler.take(64);t,noise=random_batch(step,64,102)
        expected={'update':step+1,'batch':batch,'t':t.numpy(),'noise_hash':digest(noise.numpy())}
        assert r['update']==step+1 and r['batch']==batch and np.array_equal(r['t'],t.numpy()) and r['noise_hash']==expected['noise_hash']
        assert r['arms']=={a:digest(expected) for a in ARMS}
    tids=[arrays[3][i] for i in data['slots']];events=0;probe_results={}
    from ff_prior_origin_learn_bounded.learning import load as old_load
    warm=old_load(WARM/'learning/goal_loss')[0];initial=weight_hash(warm)
    for arm in ARMS:
        tr=read_json(out/'learning'/arm/'training.json');model,_=load(out/'learning'/arm)
        assert tr['status']=='PASS' and tr['updates']==6000 and tr['initial_weight_hash']==initial
        assert tr['metadata']['warm_checkpoint_sha256']==file_digest(WARM/'learning/goal_loss/model.pt')
        assert tr['metadata']['loss_weights']==ARMS[arm] and tr['metadata']['protocol']==P
        assert tr['metadata']['input_proof_sha256']==file_digest(out/'input_verification.json')
        assert tr['actual_consumed_sequence_hashes']==[r['arms'][arm] for r in seq]
        assert tr['random_sequence_sha256']==file_digest(out/'paired_random_sequence.json')
        assert all(tr[k]==v for k,v in sampler.report(tids).items())
        assert tr['train_draw_counts']==dict(zip(tids,(sampler.row_counts+sampler.origin_row_counts).tolist()))
        assert all(v>0 for v in tr['train_draw_counts'].values()) and sum(tr['train_draw_counts'].values())==384000
        expected_events=120 if ARMS[arm][1] else 0
        assert len(tr['world_records'])==tr['world_gradient_updates']==expected_events
        assert tr['VAL_gradient_updates']==0 and not tr['VAL_checkpoint_selection']
        for e in tr['world_records']:
            i=mapping[e['sample_id']];path=out/'world_gradients'/arm/f'{e["event"]:03d}.json'
            proof=verify_gradient(path,codec,arrays[1][i],data['centers'][i],rows[i])
            assert file_digest(path)==e['proof_sha256'] and proof['input_checkpoint_sha256']==file_digest(path.with_suffix('.pt'))
            saved=torch.load(path.with_suffix('.pt'),map_location='cpu',weights_only=False)
            warm.load_state_dict(saved['weights']);assert weight_hash(warm)==e['input_weight_hash']==proof['input_weight_hash']
            event=e['event'];group=data['groups'][event%16];expected_i=group[(event//16)%4]
            assert i==expected_i and proof['update']==50*(event+1) and proof['noise_seed']==P['world_noise_seed']+event
            assert proof['noise_t']==P['world_noise_times'][(event//16)%2] and arrays[4][i]=='TRAIN'
            with torch.no_grad():x=predict(warm,data,i,proof['noise_t'],proof['noise_seed']).numpy().astype(float)
            assert digest(x)==proof['x_hash']
            assert ledger['completed'][f'{arm}_gradient_{event:03d}']['status']=='PASS'
            events+=1
    assert events==240 and ledger['calls']['world_gradient.coupled']==240*212
    for name in ['warm']+list(ARMS):
        r=read_json(out/'probes'/f'{name}.json');assert len(r['cases'])==32
        assert r['weight_hash']==(initial if name=='warm' else read_json(out/'learning'/name/'training.json')['trained_weight_hash'])
        for c in r['cases']:
            i=mapping[c['sample_id']];assert arrays[4][i]=='TRAIN'
            assert np.isclose(c['loss'],loss_from_ee(c['ee'],rows[i]['task']['condition']['goals'])[0],rtol=1e-12,atol=1e-12)
        probe_results[name]=r
    assert read_json(out/'TRAIN_selection.json')==select(probe_results)
    assert not ledger.get('unresolved_call_accounting')
    return {'status':'PASS','actual_updates_each':6000,'actual_draws_each':384000,'paired_steps':6000,
        'current_prediction_checkpoints_verified':240,'full102_gradients':240,'TRAIN_probes':160,
        'additional_physics_calls':0,'selection_recomputed':True}

def verify(out,identity,codec,data,arrays,rows,training=False,smoke=False):
    ledger=read_json(out/'ledger.json');problems=[]
    assert ledger['inflight'] is None and ledger['identity']=={'identity_hash':digest(identity)}
    assert sources()==identity['source_hashes']
    assert all(file_digest(ROOT/p)==h for p,h in identity['source_files'].items())
    with zipfile.ZipFile(out/'executed_source.zip') as z:
        assert all(hashlib.sha256(z.read(p)).hexdigest()==h for p,h in sources().items())
    protected=read_json(ROOT/'ff_prior_expand/evidence/preservation_before.json')
    for group in ('old_files','user_edits'):assert all(file_digest(ROOT/p)==h for p,h in protected[group].items())
    for branch,h in protected['branches'].items():assert subprocess.check_output(['git','rev-parse',branch],cwd=ROOT,text=True).strip()==h
    assert read_json(out/'run.json')['acceptance']==read_json(SOURCE/'run.json')['acceptance']
    single=read_json(out/'single_step_summary.json');assert len(single['cases'])==8
    mapping={sid:i for i,sid in enumerate(arrays[3])}
    for c in single['cases']:
        i=mapping[c['sample_id']]
        if c['status']=='PASS':verify_gradient(out/'single_step'/f'{c["case"]:02d}_gradient.json',codec,arrays[1][i],data['centers'][i],rows[i])
        for r in c['negative_gradient_steps']+list(c['optimizer_steps'].values()):
            assert np.isclose(r['loss'],loss_from_ee(r['ee'],rows[i]['task']['condition']['goals'])[0])
    result={'single_step_cases_audited':8,'training':'NOT_RUN','smoke':'NOT_RUN'}
    if training:
        proof=read_json(out/'training_verification.json');assert proof['status']=='PASS';result['training']=proof
    if smoke:result['smoke']=verify_smoke(out,identity,ledger,problems)
    else:assert not (out/'candidates').exists() and not (out/'executions').exists()
    if ledger.get('unresolved_call_accounting'):problems.append('UNRESOLVED_ACCOUNTING')
    return dict(result,status='PASS' if not problems else 'BLOCKED',problems=problems,
        source_and_protected_files_verified=True,TEST_executed=False,additional_physics_calls=0)
