"""Fresh reviewer: saved-array/neural-only checks; no simulator or optimizer step."""
import os
os.environ['OMP_NUM_THREADS']='2'
import sys,json,hashlib,zipfile,datetime
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from ff_prior.storage import digest
from ff_prior_control_transfer.data import load
from ff_prior_control_transfer.model import TransferDenoiser
from ff_prior_origin_learn_bounded.distribution import CoverageSampler
torch.set_num_threads(2)
OUT=ROOT/'artifacts/v7_1/rollout_world_bounded_001'
TRACE=ROOT/'paper/review-traces/experiment-result-to-claim/20261021_rollout_world_run01'
TRACE.mkdir(parents=True,exist_ok=True)
inputs={};checks={}
def track(p):
 p=Path(p);inputs[str(p.relative_to(ROOT))]={'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'mtime_utc':datetime.datetime.fromtimestamp(p.stat().st_mtime,datetime.timezone.utc).isoformat(),'bytes':p.stat().st_size};return p
def js(p):return json.loads(track(p).read_text(encoding='utf-8-sig'))
def pt(p):return torch.load(track(p),map_location='cpu',weights_only=False)
def npz(p):return dict(np.load(track(p)))
def save():
 (TRACE/'checks.json').write_text(json.dumps(checks,ensure_ascii=False,indent=2),encoding='utf-8')
 (TRACE/'inputs.json').write_text(json.dumps(inputs,ensure_ascii=False,indent=2),encoding='utf-8')
def stats(x):
 x=np.asarray(x);return dict(mean=float(x.mean()),median=float(np.median(x)),p95=float(np.percentile(x,95)),maximum=float(x.max()))
def residual(ee,goal):
 ee=np.asarray(ee);goal=np.asarray(goal)
 pos=(ee[...,:3,3]-goal[...,:3,3])/.005
 rot=np.swapaxes(goal[...,:3,:3],-1,-2)@ee[...,:3,:3]
 rv=Rotation.from_matrix(rot.reshape(-1,3,3)).as_rotvec().reshape(rot.shape[:-2]+(3,))/np.deg2rad(1)
 return np.concatenate((pos.reshape(pos.shape[:-2]+(6,)),rv.reshape(rv.shape[:-2]+(6,))),axis=-1)
def qualified(r):
 v=r['validation'];model=v['validation_level']=='MODEL_VALID'
 goal='frozen_goal_position_error_m' in v and max(v['frozen_goal_position_error_m'])<=.005 and max(v['frozen_goal_rotation_error_rad'])<=np.pi/180
 assert bool(model and goal)==r['eligible'];return bool(model and goal)
def dist(rs):
 ps=[];angs=[]
 for r in rs:
  qualified(r);v=r['validation']
  if 'frozen_goal_position_error_m' in v:ps.append(max(v['frozen_goal_position_error_m'])*1000);angs.append(max(v['frozen_goal_rotation_error_rad'])*180/np.pi)
 return dict(eligible=sum(r['eligible'] for r in rs),total=len(rs),missing=len(rs)-len(ps),position_mm=stats(ps) if ps else None,rotation_deg=stats(angs) if angs else None)
def metrics(rs):
 q=[max(max(r['validation']['frozen_goal_position_error_m'])/.005,max(r['validation']['frozen_goal_rotation_error_rad'])/(np.pi/180)) for r in rs]
 return dict(eligible=sum(qualified(r) for r in rs),Q=q,median_Q=float(np.median(q)),mean_Q=float(np.mean(q)),roughness=float(np.sqrt(np.mean([np.mean(np.diff(r['Z'],n=2,axis=0)**2) for r in rs]))))

# Replay the diffusion equations, with batch-one kernels, without calling producer rollout/loss/audit.
def neural(model,c,seeds,batch_one=True):
 def one(cc,ss):
  x=torch.cat([torch.cat([torch.randn((1,102),generator=torch.Generator().manual_seed(int(s))),torch.randn((1,221),generator=torch.Generator().manual_seed(int(s)+971003))],1) for s in ss]).double()
  times=np.linspace(999,0,16).round().astype(int)
  for k,t in enumerate(times):
   a=model.alpha[t];v,detail=model.components(x,torch.full((len(ss),),int(t)),cc)
   clean=torch.cat((a.sqrt()*x[:,:102]-(1-a).sqrt()*v.double(),detail),1)
   noise=(x-a.sqrt()*clean)/(1-a).sqrt();na=model.alpha[times[k+1]] if k<15 else x.new_tensor(1.)
   x=na.sqrt()*clean+(1-na).sqrt()*noise
  return x
 return torch.cat([one(c[i:i+1],[s]) for i,s in enumerate(seeds)]) if batch_one else one(c,seeds)
def teacher(full,target,codec):
 s=torch.tensor(codec.latent_scale,dtype=torch.float64);b=torch.tensor(codec.basis,dtype=torch.float64)
 physical=torch.einsum('tk,nkj->ntj',b,((full-target.double())*s).reshape(-1,19,17))
 return (physical.square().mean()+.25*torch.diff(physical,n=2,dim=1).square().mean())/s.square().mean()
def manual_adam(before,after,grads,model):
 group=before['optimizer']['param_groups'][0];b1,b2=group['betas'];max_w=max_m=max_v=0.
 for (name,p),idx,grad in zip(model.named_parameters(),group['params'],grads):
  old=before['optimizer']['state'].get(idx,{});new=after['optimizer']['state'][idx]
  m=old.get('exp_avg',torch.zeros_like(p)).clone();v=old.get('exp_avg_sq',torch.zeros_like(p)).clone()
  step=int(old.get('step',0))+1
  m.lerp_(grad,1-b1);v.mul_(b2).addcmul_(grad,grad,value=1-b2)
  w=before['weights'][name].clone();w.mul_(1-group['lr']*group['weight_decay'])
  denominator=v.sqrt()/(1-b2**step)**.5+group['eps'];w.addcdiv_(m,denominator,value=-group['lr']/(1-b1**step))
  max_w=max(max_w,float((w-after['weights'][name]).abs().max()));max_m=max(max_m,float((m-new['exp_avg']).abs().max()));max_v=max(max_v,float((v-new['exp_avg_sq']).abs().max()))
 assert max_w<2e-7 and max_m<2e-8 and max_v<2e-8,(max_w,max_m,max_v)
 return dict(max_weight_error=max_w,max_first_moment_error=max_m,max_second_moment_error=max_v)

ident=js(OUT/'run.json');selection=js(OUT/'frozen_tasks.json')
with zipfile.ZipFile(track(OUT/'executed_source.zip')) as z:
 for name,h in ident['identity']['source_hashes'].items():assert hashlib.sha256(z.read(name)).hexdigest()==h==hashlib.sha256(track(ROOT/name).read_bytes()).hexdigest()
checks['source']={'producer_head':ident['producer_head'],'bound_source_files':len(ident['identity']['source_hashes']),'unchanged':True}
track(ROOT/'ff_prior_rollout_world/docs/EXPERIMENT_PLAN.md')
print('Loading bound arrays; no physical simulation',flush=True)
codec,arrays,rows,data=load()
assert len(data['tr'])==14222 and len(data['va'])==48
model=TransferDenoiser(codec,warm=False)
checks['refinements']={};checks['derivatives']=[];checks['world_updates']=[]
for stage in ('TRAIN_world','candidate_world'):
 dest=OUT/stage;gate=js(dest/'gate.json');initial=pt(dest/'initial.pt');indices=initial['indices'];seeds=initial['seeds'];c=data['c'][indices];target=data['x'][indices]
 current=js(dest/'initial_records.json');first=metrics(current);original=[r['eligible'] or i==12 for i,r in enumerate(current)];rounds=[]
 for rd in sorted((dest/'rounds').iterdir()):
  before=pt(rd/'before.pt');prop=pt(rd/'proposed.pt');retained=pt(rd/'retained.pt');result=js(rd/'result.json')
  model.load_state_dict(before['weights']);model.calls=0
  with torch.no_grad():x=neural(model,c,seeds)
  assert model.calls==256 and torch.equal(x,before['x0'])
  residuals=[];jacobians=[]
  for i in range(16):
   a=npz(rd/'jacobians'/f'{i:02d}.npz');proof=js(rd/'jacobians'/f'{i:02d}.json');goal=selection['diagnostic'][i//2]['task']['condition']['goals']
   rr=residual(a['central_ee'],goal);pert=residual(a['perturbation_ee'],goal);J=(pert[:,1]-pert[:,0]).T/.02
   assert np.max(abs(rr-a['residual']))<1e-10 and np.max(abs(pert-a['perturbation_residuals']))<1e-10 and np.max(abs(J-a['J']))<1e-9
   assert np.array_equal(a['x'],x[i].numpy()) and J.shape==(12,323)
   half=[]
   for hc in proof['half_checks']:
    vals=np.array([residual(rec['ee'],goal) for rec in hc['records']]);assert np.max(abs(vals-hc['residuals']))<1e-10
    slope=(vals[1]-vals[0])/.01;col=J[:,hc['dimension']];err=np.linalg.norm(slope-col)/max(np.linalg.norm(slope),np.linalg.norm(col),1e-6);assert err<=.02;half.append(float(err))
   d=np.sin(np.arange(323)+.37);d/=np.linalg.norm(d)
   vals=np.array([residual(rec['ee'],goal) for rec in proof['direction_records']]);assert np.max(abs(vals-proof['direction_residuals']))<1e-10
   slope=(np.sum(vals[1]**2)/4-np.sum(vals[0]**2)/4)/.02;expect=float(rr@J@d/2);err=abs(slope-expect)/max(abs(slope),abs(expect),1e-6);assert err<=.05
   fine=np.array(proof['convergence']['fine']['ee']);dp=np.linalg.norm(a['central_ee'][:,:3,3]-fine[:,:3,3],axis=-1);dr=Rotation.from_matrix(np.swapaxes(fine[:,:3,:3],-1,-2)@a['central_ee'][:,:3,:3]).magnitude()
   assert max(dp)<=.001 and max(dr)<=np.deg2rad(.2);assert proof['calls']['coupled']==654
   checks['derivatives'].append(dict(stage=stage,round=rd.name,index=i,half_max=max(half),direction_error=float(err),convergence_m=float(max(dp)),convergence_rad=float(max(dr))))
   residuals.append(rr);jacobians.append(J)
  R=torch.tensor(np.array(residuals));J=torch.tensor(np.array(jacobians));x0=before['x0'];mask=torch.tensor(before['protected'])
  assert before['protected']==[a or r['eligible'] for a,r in zip(original,current)]
  for step in ('001','064'):
   u=pt(rd/'updates'/step/'before.pt');v=pt(rd/'updates'/step/'after.pt');model.load_state_dict(u['weights']);model.zero_grad();model.calls=0
   full=neural(model,c,seeds);estimated=R+torch.einsum('nij,nj->ni',J,full-x0);q=estimated.reshape(-1,4,3).square().sum(-1)
   guard=torch.relu(q[mask]-.81).square().mean() if mask.any() else q.mean()*0
   loss=q.mean()+torch.relu(q-1).square().mean()+100*guard+.01*(full-x0).square().mean()+.05*teacher(full,target,codec)
   loss.backward();norm=torch.linalg.vector_norm(torch.stack([torch.linalg.vector_norm(p.grad) for p in model.parameters()]));clip=min(1.,float(1/(norm+1e-6)))
   grads=[p.grad*clip for p in model.parameters()];error=max(float((a-b).abs().max()) for a,b in zip(grads,u['gradients']));assert error<3e-6,error
   assert model.calls==256
   checks['world_updates'].append(dict(stage=stage,round=rd.name,step=step,full16_calls=model.calls,gradient_max_error=error,**manual_adam(u,v,u['gradients'],model)))
  chosen=None
  for trial,rec in enumerate(result['trials']):
   rs=[]
   for i in range(16):
    p=rd/f'trial_{trial}'/f'{i:02d}';r=js(p/'validation.json');ctrl=npz(p/'controls.npz');assert np.array_equal(ctrl['raw'],ctrl['prepared']);rs.append(r)
   b=metrics(rs);a=metrics(current)
   expected={'finite':all(np.isfinite(b['Q'])),'old_qualified_preserved':all(not keep or r['eligible'] for keep,r in zip(before['protected'],rs)),'no_new_model_failure':all(x['validation']['validation_level']!='MODEL_VALID' or y['validation']['validation_level']=='MODEL_VALID' for x,y in zip(current,rs)),'controlled_time_detail':b['roughness']<=gate['rough_limit'],'mean_Q_improved':b['mean_Q']<=.99*a['mean_Q'],'median_Q_nonregression':b['median_Q']<=a['median_Q'],'raw_unchanged_by_prepare':True}
   assert expected==rec['checks'] and all(expected.values())==rec['accepted']
   if rec['accepted']:chosen=rec['scale'];current=rs
  for k,w in before['weights'].items():
   expected=w if chosen is None or not w.is_floating_point() else w+chosen*(prop['weights'][k]-w)
   assert torch.equal(expected,retained['weights'][k])
  rounds.append({'round':rd.name,'trials':[{'scale':t['scale'],'accepted':t['accepted'],'checks':t['checks'],'eligible':t['metrics']['eligible']} for t in result['trials']],'protected_indices':np.flatnonzero(before['protected']).tolist()})
  print(stage,rd.name,'independent derivative/VJP/Adam/line-search PASS',flush=True);save()
 final=metrics(current);assert final['eligible']==gate['after']['eligible'] and final['median_Q']==gate['after']['median_Q']
 checks['refinements'][stage]=dict(before=first,after=final,rounds=rounds,qualified_indices=[i for i,r in enumerate(current) if r['eligible']],original12_before=bool(js(dest/'initial_records.json')[12]['eligible']),original12_after=bool(current[12]['eligible']),gate=gate['status'])
 save()

# Regenerate the complete consumed batch/noise stream only; no training is performed.
ordinary=OUT/'learning/ordinary';training=js(ordinary/'training.json');state=pt(ordinary/'progress.pt');init=pt(ordinary/'initial.pt');m1init=pt(OUT/'TRAIN_world/initial.pt')
assert all(torch.equal(v,m1init['weights'][k]) for k,v in init['weights'].items())
sampler=CoverageSampler(data['keys'],20264203);rng=torch.Generator().manual_seed(20264201);counts=np.zeros(14222,dtype=np.int64);proofs=[];segments=[]
for step in range(6000):
 batch=sampler.take(64);np.add.at(counts,batch,1);t=torch.randint(0,1000,(64,),generator=rng);noise=torch.randn((64,323),generator=rng)
 proofs.append(digest({'step':step+1,'batch':batch,'t':t.numpy(),'noise':noise.numpy()}))
 if (step+1)%20==0:segments.append(dict(start=step-18,end=step+1,draws=1280,digest=digest(proofs)));proofs=[]
assert segments==training['segments']==state['segments'] and np.array_equal(counts,state['counts']) and torch.equal(rng.get_state(),state['rng'])
assert counts.sum()==384000 and counts.min()>0 and state['updates']==6000 and state['rollouts']==1200
assert {int(s['step']) for s in state['optimizer']['state'].values()}=={6000}
checks['ordinary_training']=dict(updates=6000,segments_recomputed=300,rows_seen=int((counts>0).sum()),draws=int(counts.sum()),minimum_row_draws=int(counts.min()),maximum_row_draws=int(counts.max()),full16_batches=1200,full16_paths=9600,fresh_same_initial_weights=True,VAL_updates=training['VAL_gradient_updates'],VAL_selection=training['VAL_checkpoint_selection'])
checks['ordinary_updates']=[]
for step in ('0001','0005','3000','6000'):
 d=ordinary/'updates'/step;u=pt(d/'before.pt');v=pt(d/'after.pt');g=npz(d/'gradients.npz');info=js(d/'update.json');flat=torch.tensor(g['mixed']);clip=min(1.,1/(info['clip_norm']+1e-6));grads=[];start=0
 for p in model.parameters():grads.append((flat[start:start+p.numel()].reshape_as(p)*clip).to(p.dtype));start+=p.numel()
 checks['ordinary_updates'].append(dict(step=step,**manual_adam(u,v,grads,model)))
save();print('6000 consumed updates/300 stream hashes/manual Adam checks PASS',flush=True)

checks['comparisons']={}
newval=[js(p) for p in sorted((OUT/'unguided').glob('*/*.json'))];oldval=[js(ROOT/'artifacts/v7_1/sample_alignment_001/unguided'/str(s)/'aligned'/f'{i:02d}.json') for s in range(2) for i in range(48)]
for a,b in zip(newval,oldval):assert a['seed']==b['seed'] and a['task_hash']==b['task_hash'];assert len(a['steps'])==16 and a['physics_guidance_calls']==0
checks['comparisons']['VAL']=dict(world=dist(newval),old_aligned=dist(oldval),world_raw_equals_prepared=sum(np.array_equal(a['raw_Z'],a['prepared_Z']) for a in newval),old_raw_equals_prepared=sum(np.array_equal(a.get('raw_Z',a.get('Z')),a.get('prepared_Z',a.get('Z'))) for a in oldval),both=sum(a['eligible'] and b['eligible'] for a,b in zip(newval,oldval)),new_only=sum(a['eligible'] and not b['eligible'] for a,b in zip(newval,oldval)),old_only=sum(not a['eligible'] and b['eligible'] for a,b in zip(newval,oldval)))
checks['comparisons']['by_task_seed']=[]
for scope in ('DEV','APPLICATION_DEV'):
 recs=[js(p) for p in sorted((OUT/'comparison'/scope).glob('*/*/candidates/*.json'))];exs=[js(p) for p in sorted((OUT/'comparison'/scope).glob('*/*/execution/outcome.json'))]
 checks['comparisons'][scope]={'world323':dict(**dist(recs),execution_verified=sum(r['status']=='EXECUTION_VERIFIED' for r in exs),execution_opportunities=len(exs))}
 for r in recs:assert r['model_calls']['coupled']<=113 and r['integration_intervals']<=15872 and r['wall_s']<=15 and r['actual_network_calls']==16
 for name in ('old_aligned','structured','learned_mean'):
  reuse=[e for e in selection['reuse'] if e['scope']==scope and e['method']==name];rs=[js(ROOT/p) for e in reuse for p in e['candidates']]
  checks['comparisons'][scope][name]=dist(rs)
  checks['comparisons'][scope][name]['reuse_entries']=len(reuse)
  for e in reuse:
   ps=[js(ROOT/p) for p in e['candidates']];checks['comparisons']['by_task_seed'].append(dict(scope=scope,method=name,sample_id=ps[0]['sample_id'],seed=ps[0]['seed'],**dist(ps)))
 for d in sorted((OUT/'comparison'/scope).glob('*/*')):
  rs=[js(p) for p in sorted((d/'candidates').glob('*.json'))]
  checks['comparisons']['by_task_seed'].append(dict(scope=scope,method='world323',sample_id=d.name,seed=rs[0]['seed'],**dist(rs)))
checks['budget']=dict(actual_coupled=sum(v for k,v in js(OUT/'ledger.json')['calls'].items() if k.endswith('.coupled')),reserved=js(OUT/'budget.json'),elapsed_s=js(OUT/'status.json')['elapsed_s'],bytes=sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file()))
checks['paper_claim_audit']='unavailable' if not (ROOT/'paper/PAPER_CLAIM_AUDIT.json').exists() else js(ROOT/'paper/PAPER_CLAIM_AUDIT.json')
checks['auto_audit']=js(OUT/'independent_verification.json')
save();print(json.dumps({k:v for k,v in checks.items() if k in ('refinements','ordinary_training','budget','paper_claim_audit')},ensure_ascii=False),flush=True)
