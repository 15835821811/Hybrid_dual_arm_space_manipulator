"""Bounded independent review; reads saved evidence, writes only beside this script.

No training/optimizer steps, DDIM sampling, propagation, geometry checks or execution.
Fixed recorded noise is reconstructed only to verify saved neural predictions.
"""
import sys
sys.dont_write_bytecode = True
from pathlib import Path
import json, hashlib, datetime, collections
import numpy as np
import torch
OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[3]
sys.path.insert(0, str(ROOT))
from ff_prior_latent.learning import Codec, LatentDenoiser
from ff_prior.storage import digest
from ff_prior.diffusion import weight_hash
torch.set_num_threads(2)
torch.use_deterministic_algorithms(True)
RUN = ROOT/'artifacts/v7_1/paired_objectives_smoke_001'
WARM = ROOT/'artifacts/v7_1/qualified_origin_learn_002/learning/goal_loss'
tracked = {}
checks = []

def track(path):
    path = Path(path).resolve()
    if str(path) not in tracked:
        stat=path.stat()
        h=hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda:stream.read(1024*1024),b''): h.update(block)
        tracked[str(path)]={'path':str(path),'sha256':h.hexdigest(),'bytes':stat.st_size,
          'modified_utc':datetime.datetime.fromtimestamp(stat.st_mtime,datetime.timezone.utc).isoformat()}
    return path

def read(path): return json.loads(track(path).read_text(encoding='utf-8-sig'))
def filehash(path): return tracked[str(track(path))]['sha256']
def check(label, value):
    checks.append({'check':label,'passed':bool(value)})
    if not value: raise AssertionError(label)
def stats(a):
    a=np.asarray(a,dtype=float)
    return {'n':int(a.size),'min':float(a.min()),'median':float(np.median(a)),
            'max':float(a.max()),'mean':float(a.mean())}

def endpoint(ee,goals):
    """Independent rotation-angle formula (atan2 skew norm, trace)."""
    a,b=np.asarray(ee),np.asarray(goals)
    pos=np.linalg.norm(a[...,:3,3]-b[...,:3,3],axis=-1)
    r=np.swapaxes(b[...,:3,:3],-1,-2)@a[...,:3,:3]
    skew=np.stack((r[...,2,1]-r[...,1,2],r[...,0,2]-r[...,2,0],r[...,1,0]-r[...,0,1]),axis=-1)
    rot=np.arctan2(np.linalg.norm(skew,axis=-1)/2,(np.trace(r,axis1=-2,axis2=-1)-1)/2)
    loss=np.sum((pos/.005)**2+(rot/np.deg2rad(1))**2,axis=-1)/4
    return loss,pos,rot

index=read(RUN/'training_index.json')['records']; byid={r['sample_id']:i for i,r in enumerate(index)}
targets=read(RUN/'fixed_TRAIN_targets.json')['entries']
inputproof=read(RUN/'input_verification.json')
check('input proof index equals frozen index',inputproof['records']==index)
check('frozen TRAIN/VAL counts',collections.Counter(r['split'] for r in index)=={'TRAIN':13838,'VAL':48})
seq=read(RUN/'paired_random_sequence.json')
tr={a:read(RUN/'learning'/a/'training.json') for a in ['A_v','B_local','C_world','D_both']}
probes={a:read(RUN/'probes'/f'{a}.json') for a in ['warm',*tr]}
single=read(RUN/'single_step_summary.json')
ledger=read(RUN/'ledger.json')
selection=read(RUN/'TRAIN_selection.json')
codec=Codec(read(RUN/'learning/A_v/codec.json'))
model=LatentDenoiser(codec.dim,len(codec.feature_mean))
alpha=torch.cumprod(1-torch.linspace(1e-4,.02,1000),dim=0)
rowcache={}
def row(sid):
    if sid not in rowcache:
        entry=index[byid[sid]]
        path=ROOT/entry['record']
        r=read(path);check('source record hash '+sid,filehash(path)==entry['record_sha256'])
        rowcache[sid]=r
    return rowcache[sid]

new_ids=[e['sample_id'] for e in index if e.get('producer_run_id')=='world_endpoint_smoke_001']
groups=[[sid for sid in sorted(new_ids) if row(sid)['parent_sample_id']==target['parent_sample_id']] for target in targets]
check('16 frozen TRAIN targets with four inherited variants',len(groups)==16 and all(len(g)==4 for g in groups))
for target,group in zip(targets,groups):
    for sid in group:
        check('fixed target goals preserved '+sid,row(sid)['task']['condition']['goals']==target['task']['condition']['goals'])

cached_arrays=track(ROOT/'artifacts/v7_1/world_endpoint_smoke_001/loaded_training_arrays.npz')
check('cached array hash bound by input proof',filehash(cached_arrays)==inputproof['parent_array_sha256'])
with np.load(cached_arrays) as a: zall=a['Z']; ball=a['bases']
prepared={}
def prediction(sid,t,seed):
    if sid not in prepared:
        i=byid[sid]; r=row(sid); base=ball[i]
        center=codec.center(r['condition'],base)
        clean=torch.tensor(codec.normalized(codec.encode(zall[i],base),center),dtype=torch.float32)[None]
        cond=torch.tensor(codec.condition(r['condition'],base),dtype=torch.float32)[None]
        prepared[sid]=(clean,cond)
    clean,cond=prepared[sid]
    noise=torch.randn(clean.shape,generator=torch.Generator().manual_seed(seed))
    a=alpha[t]; noisy=a.sqrt()*clean+(1-a).sqrt()*noise
    return (a.sqrt()*noisy-(1-a).sqrt()*model(noisy,torch.tensor([t]),cond))[0]

probe_rows=[]; metrics={}; losses={}; ids=None
probe_error={'loss':0.,'position':0.,'rotation':0.}
for arm,p in probes.items():
    identities=[(c['sample_id'],c['t'],c['seed']) for c in p['cases']]
    if ids is None: ids=identities
    check('paired 32 probe identities '+arm,len(identities)==32 and identities==ids)
    check('frozen probe identities '+arm,identities==[(g[0],t,20262303+j*2+k) for j,g in enumerate(groups) for k,t in enumerate([500,999])])
    mp=(WARM if arm=='warm' else RUN/'learning'/arm)/'model.pt'
    checkpoint=torch.load(track(mp),map_location='cpu',weights_only=False)
    model.load_state_dict(checkpoint['weights']); model.eval()
    check('probe model weight hash '+arm,weight_hash(model)==p['weight_hash'])
    if arm!='warm':check('final checkpoint file hash '+arm,filehash(mp)==tr[arm]['checkpoint_sha256'])
    values=[]
    for c in p['cases']:
        sid=c['sample_id']; check('TRAIN probe '+arm+'/'+sid,index[byid[sid]]['split']=='TRAIN')
        f,po,ro=endpoint(c['ee'],row(sid)['task']['condition']['goals'])
        errors=[abs(float(f)-c['loss']),float(np.max(np.abs(po-c['position_m']))),float(np.max(np.abs(ro-c['rotation_rad'])))]
        for k,e in zip(probe_error,errors):probe_error[k]=max(probe_error[k],e)
        check('pose arithmetic '+arm+'/'+sid+'/'+str(c['t']),np.allclose([f,*po,*ro],[c['loss'],*c['position_m'],*c['rotation_rad']],rtol=1e-10,atol=1e-11))
        with torch.no_grad(): xx=prediction(sid,c['t'],c['seed']).numpy().astype(float)
        check('saved neural probe x hash '+arm+'/'+sid+'/'+str(c['t']),digest(xx)==c['x_hash'])
        values.append(float(f))
        probe_rows.append({'arm':arm,'sample_id':sid,'t':c['t'],'seed':c['seed'],'loss':float(f),
          'position_m':po.tolist(),'rotation_rad':ro.tolist(),'max_position_mm':float(max(po)*1000),
          'max_rotation_deg':float(np.rad2deg(max(ro))),
          'position_pass':bool(max(po)<=.005),'rotation_pass':bool(max(ro)<=np.deg2rad(1)),
          'both_endpoint_pass':bool(max(po)<=.005 and max(ro)<=np.deg2rad(1))})
    losses[arm]=np.asarray(values)
for arm in probes:
    rows=[r for r in probe_rows if r['arm']==arm]
    m={'loss':stats(losses[arm]),'ratio_to_warm':stats(losses[arm]/losses['warm']),
       'max_position_mm':stats([r['max_position_mm'] for r in rows]),
       'max_rotation_deg':stats([r['max_rotation_deg'] for r in rows]),
       'position_pass':sum(r['position_pass'] for r in rows),'rotation_pass':sum(r['rotation_pass'] for r in rows),
       'both_endpoint_pass':sum(r['both_endpoint_pass'] for r in rows),
       'improved_cases_vs_warm':int(np.sum(losses[arm]<losses['warm'])),'by_t':{}}
    for t in [500,999]:
        mask=np.asarray([r['t']==t for r in rows]); rr=[r for r in rows if r['t']==t]
        m['by_t'][str(t)]={'ratio_to_warm':stats((losses[arm]/losses['warm'])[mask]),'loss':stats(losses[arm][mask]),
         'both_endpoint_pass':sum(r['both_endpoint_pass'] for r in rr),
         'improved_cases_vs_warm':int(np.sum((losses[arm]<losses['warm'])[mask]))}
    if arm in tr:
        m['ratio_to_B_local']=stats(losses[arm]/losses['B_local'])
        m['eligible']=arm in ['C_world','D_both'] and m['ratio_to_warm']['median']<=.8 and m['ratio_to_B_local']['median']<=1
        ss=selection['metrics'][arm]
        check('selection recomputed '+arm,np.isclose(m['ratio_to_warm']['median'],ss['median_ratio_vs_warm'],rtol=1e-10)
          and np.isclose(m['ratio_to_B_local']['median'],ss['median_ratio_vs_local_control'],rtol=1e-10)
          and m['improved_cases_vs_warm']==ss['improved_cases'] and m['eligible']==ss['eligible'])
    metrics[arm]=m

pairwise={f'{a}/{b}':{'ratio':stats(losses[a]/losses[b]),'wins':int(np.sum(losses[a]<losses[b]))}
 for a,b in [('B_local','A_v'),('D_both','C_world'),('C_world','A_v'),('D_both','B_local')]}
warm_rows=[r for r in probe_rows if r['arm']=='warm']
d_rows=[r for r in probe_rows if r['arm']=='D_both']
d_pose_tradeoff={'definition':'position/rotation comparisons use the maximum error across both arms for each probe',
 'position_improved':sum(d['max_position_mm']<w['max_position_mm'] for d,w in zip(d_rows,warm_rows)),
 'rotation_improved':sum(d['max_rotation_deg']<w['max_rotation_deg'] for d,w in zip(d_rows,warm_rows)),
 'loss_better_rotation_worse':sum(d['loss']<w['loss'] and d['max_rotation_deg']>w['max_rotation_deg'] for d,w in zip(d_rows,warm_rows)),
 'warm_pass_retained':sum(d['both_endpoint_pass'] and w['both_endpoint_pass'] for d,w in zip(d_rows,warm_rows)),
 'warm_pass_lost':sum(not d['both_endpoint_pass'] and w['both_endpoint_pass'] for d,w in zip(d_rows,warm_rows)),
 'new_passes':sum(d['both_endpoint_pass'] and not w['both_endpoint_pass'] for d,w in zip(d_rows,warm_rows))}
trainids=[r['sample_id'] for r in index if r['split']=='TRAIN']
counts=np.zeros(len(trainids),dtype=int)
check('6000 sequence rows',len(seq)==6000)
for k,s in enumerate(seq):
    expected={q:s[q] for q in ['update','batch','t','noise_hash']}
    check('shared step digest '+str(k+1),s['update']==k+1 and len(s['batch'])==64 and len(s['t'])==64
      and all(v==digest(expected) for v in s['arms'].values()) and set(s['arms'])==set(tr))
    np.add.at(counts,s['batch'],1)
random_checks=[]
for k in [0,49,1599,3199,5999]:
    generator=torch.Generator().manual_seed(20262501+k)
    t=torch.randint(0,1000,(64,),generator=generator)
    noise=torch.randn((64,102),generator=generator)
    check('fixed step random hash spot '+str(k+1),seq[k]['t']==t.tolist() and seq[k]['noise_hash']==digest(noise.numpy()))
    random_checks.append({'update':k+1,'t_and_noise_hash_match':True})
initial=probes['warm']['weight_hash']
training_check={}
progress=torch.load(track(RUN/'learning/progress.pt'),map_location='cpu',weights_only=False)
check('joint progress at 6000',progress['updates']==6000)
for arm,r in tr.items():
    check('same initial/input/codec '+arm,r['initial_weight_hash']==initial and r['metadata']['initial_weight_hash']==initial
      and r['metadata']['training_ids']==trainids and r['metadata']['input_proof_sha256']==filehash(RUN/'input_verification.json')
      and r['metadata']['codec_hash']==digest(codec.state) and r['metadata']['warm_checkpoint_sha256']==filehash(WARM/'model.pt'))
    check('draw counts and sequences '+arm,r['updates']==6000 and all(counts>0) and int(sum(counts))==384000
      and r['train_draw_counts']==dict(zip(trainids,counts.tolist()))
      and r['actual_consumed_sequence_hashes']==[s['arms'][arm] for s in seq]
      and r['random_sequence_sha256']==filehash(RUN/'paired_random_sequence.json'))
    check('no VAL updates or selection '+arm,r['VAL_gradient_updates']==0 and r['VAL_checkpoint_selection']==False)
    o=progress['optimizers'][arm]; steps={int(s['step']) for s in o['state'].values()}
    check('optimizer state step6000 '+arm,steps=={6000})
    model.load_state_dict(progress['weights'][arm]);check('progress equals final weights '+arm,weight_hash(model)==r['trained_weight_hash'])
    training_check[arm]={'updates':r['updates'],'optimizer_state_steps':sorted(steps),'optimizer_groups':o['param_groups'],
      'TRAIN_rows':len(trainids),'draws':int(sum(counts)),'min_draws':int(min(counts)),'max_draws':int(max(counts)),
      'world_events':r['world_gradient_updates'],'initial_weight_hash':r['initial_weight_hash'],
      'history_clip_over_one':sum(h['gradient_norm']>1 for h in r['history']),'history_logged_points':len(r['history'])}

def gradient_saved(path):
    p=read(path); sid=p['sample_id']; goals=row(sid)['task']['condition']['goals']
    with np.load(track(path.with_suffix('.npz'))) as a: arr={k:a[k] for k in a.files}
    check('gradient npz binding '+str(path),filehash(path.with_suffix('.npz'))==p['arrays_sha256'])
    f,_,_=endpoint(arr['perturbation_ee'],goals); g=(f[:,1]-f[:,0])/(2*p['epsilon'])
    err=float(np.max(np.abs(g-arr['gradient'])))
    check('independent full102 arithmetic '+str(path),g.shape==(102,) and np.allclose(g,arr['gradient'],rtol=1e-7,atol=2e-8))
    check('perturbation loss arithmetic '+str(path),np.allclose(f,arr['perturbation_losses'],rtol=1e-10,atol=1e-10))
    f0,_,_=endpoint(p['central']['ee'],goals)
    check('central loss arithmetic '+str(path),np.isclose(f0,p['true_loss'],rtol=1e-10))
    half=[]
    for h in p['half_checks']:
        fs=[float(endpoint(v['ee'],goals)[0]) for v in h['samples']]
        slope=(fs[1]-fs[0])/p['epsilon']; gj=g[h['dimension']]
        e=abs(slope-gj)/max(abs(slope),abs(gj),1e-6);half.append(e)
        check('half difference check '+str(path)+'/'+str(h['dimension']),e<=.02)
    fs=[float(endpoint(v['ee'],goals)[0]) for v in p['direction_samples']]
    ds=(fs[1]-fs[0])/(2*p['epsilon']);expect=float(g@np.asarray(p['direction']))
    de=abs(ds-expect)/max(abs(ds),abs(expect),1e-6)
    _,dp,dr=endpoint(p['central']['ee'],p['convergence']['fine']['ee'])
    check('direction and convergence '+str(path),de<=.05 and max(dp)<=.001 and max(dr)<=np.deg2rad(.2))
    return p,arr,{'path':str(path.relative_to(ROOT)),'max_gradient_abs_error':err,'half_relative_errors':half,
     'direction_relative_error':de,'max_convergence_position_m':float(max(dp)),
     'max_convergence_rotation_rad':float(max(dr))}

single_results=[]
model.load_state_dict(torch.load(WARM/'model.pt',map_location='cpu',weights_only=False)['weights'])
for c in single['cases']:
    path=RUN/'single_step'/f'{c["case"]:02d}_gradient.json'
    p,arr,gcheck=gradient_saved(path)
    pred=prediction(c['sample_id'],c['t'],c['seed'])
    check('single prediction hash '+str(c['case']),digest(pred.detach().numpy().astype(float))==p['x_hash'])
    vector=torch.cat([g.reshape(-1) for g in torch.autograd.grad(torch.sum(pred*torch.tensor(arr['gradient'],dtype=pred.dtype)),tuple(model.parameters()))])
    norm=float(vector.norm()); expected=c['component_gradients']['norms']['world']/.1
    check('single world parameter VJP norm '+str(c['case']),np.isclose(norm,expected,rtol=1e-6))
    goals=row(c['sample_id'])['task']['condition']['goals']
    before=float(endpoint(p['central']['ee'],goals)[0]); descent=[]; central_errors=[]; opt={}
    for d in c['negative_gradient_steps']:
        f=float(endpoint(d['ee'],goals)[0]);check('single descent stored loss',np.isclose(f,d['loss'],rtol=1e-10))
        descent.append({'step':d['step'],'loss':f,'relative_drop':(before-f)/before})
    for d in c['parameter_central_checks']:
        fs=[float(endpoint(d[k]['ee'],goals)[0]) for k in ['minus','plus']]
        slope=(fs[1]-fs[0])/(2*d['epsilon']);e=abs(slope-norm)/max(abs(slope),norm,1e-8)
        central_errors.append(e);check('single parameter central error',e<=.05)
    for a,d in c['optimizer_steps'].items():
        f=float(endpoint(d['ee'],goals)[0]);check('single optimizer stored loss',np.isclose(f,d['loss'],rtol=1e-10));opt[a]={'loss':f,'decreased':f<before}
    passed=any(d['loss']<before-max(1e-7,abs(before)*1e-7) for d in descent)
    check('single descent gate '+str(c['case']),passed)
    single_results.append({'case':c['case'],'t':c['t'],'before_loss':before,'parameter_VJP_norm':norm,
      'central_relative_errors':central_errors,'descent':descent,'optimizer':opt,'gradient_check':gcheck})

event_summaries={}; sampled=[]
for arm in ['C_world','D_both']:
    records=tr[arm]['world_records'];check('120 event records '+arm,len(records)==120)
    proofs=[]
    for e in records:
        path=RUN/'world_gradients'/arm/f'{e["event"]:03d}.json';p=read(path);proofs.append(p)
        check('world event proof binding '+arm+'/'+str(e['event']),filehash(path)==e['proof_sha256'] and p['status']=='PASS'
          and p['dimensions']==102 and p['calls']['coupled']==212 and p['update']==50*(e['event']+1)
          and p['input_weight_hash']==e['input_weight_hash'] and p['noise_t']==[500,999][(e['event']//16)%2]
          and ledger['completed'][f'{arm}_gradient_{e["event"]:03d}']['status']=='PASS')
        check('world sample schedule '+arm+'/'+str(e['event']),p['sample_id']==groups[e['event']%16][(e['event']//16)%4]
          and p['noise_seed']==20262502+e['event'] and p['task_hash']==row(p['sample_id'])['task']['task_hash']
          and p['codec_hash']==digest(codec.state) and p['base_hash']==digest(ball[byid[p['sample_id']]])
          and p['center_hash']==digest(codec.center(row(p['sample_id'])['condition'],ball[byid[p['sample_id']]])))
    event_summaries[arm]={'events':len(records),'t_counts':dict(collections.Counter(e['noise_t'] for e in records)),
      'mixed_norm':stats([e['parameter_gradient_norm'] for e in records]),
      'clip_coefficient':stats([min(1.,1./(e['parameter_gradient_norm']+1e-6)) for e in records]),
      'world_to_v_norm':stats([e['components']['norms']['world']/e['components']['norms']['v'] for e in records]),
      'weighted_world_norm':stats([e['components']['norms']['world'] for e in records]),
      'v_norm':stats([e['components']['norms']['v'] for e in records]),
      'cos_v_world':stats([e['components']['cosines']['v__world'] for e in records]),
      'cos_local_world':stats([e['components']['cosines']['local__world'] for e in records]),
      'fraction_clipped':sum(e['parameter_gradient_norm']>1 for e in records)/len(records),
      'half_relative_error_max':max(h['relative_error'] for p in proofs for h in p['half_checks']),
      'direction_relative_error_max':max(p['direction_relative_error'] for p in proofs),
      'by_t':{}}
    for t in [500,999]:
        r=[e for e in records if e['noise_t']==t]
        event_summaries[arm]['by_t'][str(t)]={'world_to_v_norm':stats([e['components']['norms']['world']/e['components']['norms']['v'] for e in r]),
          'mixed_norm':stats([e['parameter_gradient_norm'] for e in r])}
    for event in [0,31,64,119]:
        path=RUN/'world_gradients'/arm/f'{event:03d}.json';p,arr,gcheck=gradient_saved(path)
        saved=torch.load(track(path.with_suffix('.pt')),map_location='cpu',weights_only=False)
        check('sampled current checkpoint binding '+arm+'/'+str(event),filehash(path.with_suffix('.pt'))==p['input_checkpoint_sha256'])
        model.load_state_dict(saved['weights']);check('sampled current weight hash '+arm+'/'+str(event),weight_hash(model)==p['input_weight_hash'])
        pred=prediction(p['sample_id'],p['noise_t'],p['noise_seed'])
        check('sampled current prediction hash '+arm+'/'+str(event),digest(pred.detach().numpy().astype(float))==p['x_hash'] and np.array_equal(pred.detach().numpy().astype(float),arr['x']))
        gradient=torch.cat([v.reshape(-1) for v in torch.autograd.grad(.1*torch.sum(pred*torch.tensor(arr['gradient'],dtype=pred.dtype)),tuple(model.parameters()))])
        norm=float(gradient.norm());reported=records[event]['components']['norms']['world']
        check('sampled world parameter gradient norm '+arm+'/'+str(event),np.isclose(norm,reported,rtol=1e-6))
        sampled.append(dict(gcheck,arm=arm,event=event,world_parameter_norm=norm,recorded_norm=reported))
check('ledger accounting',ledger['calls']['world_gradient.coupled']==50880 and ledger['calls']['single_step.coupled']==1784
  and ledger['calls']['TRAIN_probes.coupled']==160 and ledger['inflight'] is None and not ledger.get('unresolved_call_accounting'))
check('no DEV candidates/executions',not (RUN/'candidates').exists() and not (RUN/'executions').exists())

terminal={name:read(RUN/name) for name in ['training_verification.json','independent_verification.json','status.json']}
resume=read(ROOT/'ff_prior_paired_objectives/evidence/completed_001/resume_noop.json')
check('executor resume evidence is completed no-op',resume['status']=='PASS' and resume['before']==resume['after']
 and resume['response']['resume']=='NO_WORK_ALREADY_COMPLETE' and resume['new_training_or_physics_calls']==0)
for name,h in resume['after'].items():check('resume evidence current hash '+name,filehash(RUN/name)==h)
terminal['executor_resume_noop']=resume
source_paths=['ff_prior_paired_objectives/objectives.py','ff_prior_paired_objectives/learning.py','ff_prior_paired_objectives/diagnostic.py',
 'ff_prior_paired_objectives/audit_results.py','ff_prior_paired_objectives/inputs.py','ff_prior_paired_objectives/protocol.py',
 'ff_prior_paired_objectives/experiment.py','ff_prior_paired_objectives/identity.py','ff_prior_world_endpoint/endpoint.py',
 'ff_prior_world_endpoint/protocol.py','ff_prior_origin_learn_bounded/distribution.py','ff_prior_latent/learning.py',
 'ff_prior/diffusion.py','ff_prior/coupled_predictor.py','ff_prior/storage.py',
 'ff_prior_paired_objectives/docs/EXPERIMENT_PLAN.md','DIFFUSION_GOAL_CURRENT.md']
for path in source_paths:track(ROOT/path)
run=read(RUN/'run.json')
for path in source_paths:
    if path.endswith('.py'):check('reviewed source hash bound to run '+path,filehash(ROOT/path)==run['identity']['source_hashes'][path])
paper_audit=ROOT/'paper/PAPER_CLAIM_AUDIT.json'
audit_context=read(paper_audit) if paper_audit.exists() else 'unavailable'
result={'status':'PASS','reviewer_role':'independent paper architect/reviewer','additional_physics_calls':0,
 'scope':{'endpoint_cases':160,'fixed_one_step_neural_predictions':160,'single_step_cases':8,
 'single_step_full102_arithmetic':8,'world_event_metadata':240,'world_full102_and_weight_VJP_spot_checks':8,
 'paired_sequence_internal_checks':6000,'optimizer_final_states':4,'all_source_row_audit_repeated':False,
 'fixed_random_tensor_spot_checks':5,'resume_execution_repeated_by_reviewer':False,
 'all240_current_prediction_audit_repeated':False,'training_replayed':False,'DDIM_or_execution_run':False},
 'probe_recomputation_max_errors':probe_error,'metrics':metrics,'pairwise':pairwise,'probe_rows':probe_rows,
 'D_pose_tradeoff':d_pose_tradeoff,
 'single_step':single_results,'training_checks':training_check,'world_event_summaries':event_summaries,
 'fixed_random_tensor_spot_checks':random_checks,
 'world_samples':sampled,'ledger_calls':ledger['calls'],'terminal_evidence':terminal,
 'paper_claim_audit':audit_context,'checks':checks,'checks_passed':len(checks)}
(OUT/'verification_results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
track(Path(__file__))
(OUT/'inputs.json').write_text(json.dumps({'review_time_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
 'review_scope':result['scope'],'artifacts':list(tracked.values())},ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({'status':result['status'],'checks_passed':len(checks),'metrics':metrics,'pairwise':pairwise,
 'world_event_summaries':event_summaries,'terminal_evidence':terminal},ensure_ascii=False,indent=2))
