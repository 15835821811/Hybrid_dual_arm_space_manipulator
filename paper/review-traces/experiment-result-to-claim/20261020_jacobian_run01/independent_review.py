"""Read-only independent endpoint-Jacobian result review; writes only beside this script."""
import os, sys, json, hashlib, zipfile, copy, time, traceback
from pathlib import Path
import numpy as np
import torch
from scipy.spatial.transform import Rotation
ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT));os.chdir(ROOT)
DEST=Path(__file__).resolve().parent
OUT=ROOT/'artifacts/v7_1/endpoint_jacobian_bounded_001'
OLD=ROOT/'artifacts/v7_1/expand256_train_dev_001'
torch.set_num_threads(2)
def prohibited(*a,**k):raise RuntimeError("Review prohibits optimization and physical propagation")
torch.optim.AdamW.step=prohibited
import mujoco
mujoco.mj_step=prohibited;mujoco.mj_forward=prohibited
from ff_prior.storage import digest
from ff_prior.diffusion import weight_hash
from ff_prior_delivery_train.learning import load as warm_load
from ff_prior_endpoint_jacobian.data import load as load_data
from ff_prior_endpoint_jacobian.learning import base_loss
from ff_prior_paired_objectives.objectives import predict
inputs={}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def touch(p):
 p=Path(p);inputs[str(p.relative_to(ROOT))]={'sha256':sha(p),'bytes':p.stat().st_size,'mtime_ns':p.stat().st_mtime_ns}
 return p
def read(p):return json.loads(touch(p).read_text(encoding='utf-8-sig'))
def loadpt(p):return torch.load(touch(p),map_location='cpu',weights_only=False)
def close(a,b,atol=1e-10,rtol=1e-10):
 assert np.allclose(np.asarray(a),np.asarray(b),atol=atol,rtol=rtol),(np.max(np.abs(np.asarray(a)-np.asarray(b))),atol)
def treeclose(a,b):
 if isinstance(a,dict):
  assert a.keys()==b.keys()
  for k in a:treeclose(a[k],b[k])
 elif isinstance(a,(float,int)):close(a,b,atol=1e-12,rtol=1e-12)
 else:assert a==b
def poses(a,b):
 a=np.asarray(a);b=np.asarray(b)
 p=np.linalg.norm(a[...,:3,3]-b[...,:3,3],axis=-1)
 R=np.swapaxes(b[...,:3,:3],-1,-2)@a[...,:3,:3]
 r=Rotation.from_matrix(R.reshape(-1,3,3)).magnitude().reshape(p.shape)
 return p,r
def groups(a,b):
 p,r=poses(a,b);return np.concatenate(((p/.005)**2,(r/np.deg2rad(1))**2),axis=-1)
def loss(q):
 m=np.max(q,axis=-1)
 return m+np.log(np.exp(q-m[...,None]).mean(axis=-1))+.05*q.mean(axis=-1)
def vparams(m):return torch.cat([p.detach().reshape(-1) for p in m.parameters()])
def vg(loss,m):return torch.cat([g.reshape(-1) for g in torch.autograd.grad(loss,tuple(m.parameters()))]).detach()
def setv(m,v):
 with torch.no_grad():
  i=0
  for p in m.parameters():p.copy_(v[i:i+p.numel()].reshape_as(p));i+=p.numel()
def statehash(s):
 def plain(v):
  if isinstance(v,torch.Tensor):return v.numpy()
  if isinstance(v,dict):return {str(k):plain(x) for k,x in v.items()}
  if isinstance(v,(list,tuple)):return [plain(x) for x in v]
  return v
 return digest(plain(s))
def adam(m,g,state):
 group=state['param_groups'][0];assert len(state['param_groups'])==1
 assert group['lr']==.0002 and not group['amsgrad'] and not group.get('maximize',False)
 ps=list(m.parameters());gs=[];off=0
 for p in ps:gs.append(g[off:off+p.numel()].reshape_as(p));off+=p.numel()
 clip=torch.clamp(1/(torch.linalg.vector_norm(torch.stack([x.norm() for x in gs]))+1e-6),max=1)
 result=[];s=copy.deepcopy(state);b1,b2=group['betas']
 for pid,p,x in zip(group['params'],ps,gs):
  prev=state['state'].get(pid,{});step=float(prev.get('step',0))+1
  mm=prev.get('exp_avg',torch.zeros_like(p)).clone()
  vv=prev.get('exp_avg_sq',torch.zeros_like(p)).clone()
  x=x*clip;mm.lerp_(x,1-b1);vv.mul_(b2).addcmul_(x,x,value=1-b2)
  # Independent closed-form AdamW update, no optimizer invocation.
  den=(vv.sqrt()/np.sqrt(1-b2**step)).add_(group['eps'])
  w=p.detach().clone().mul_(1-group['lr']*group['weight_decay'])
  w.addcdiv_(mm,den,value=-group['lr']/(1-b1**step))
  result.append(w.reshape(-1));s['state'][pid]={'step':torch.tensor(step),'exp_avg':mm,'exp_avg_sq':vv}
 return torch.cat(result),s
def compare_state(a,b):
 assert a['param_groups']==b['param_groups'] and a['state'].keys()==b['state'].keys()
 for k in a['state']:
  for n in a['state'][k]:
   close(a['state'][k][n],b['state'][k][n],atol=1e-9,rtol=2e-6)
def admissible(before,after):
 a=np.r_[np.array(before['position_m'])/.005,np.array(before['rotation_rad'])/np.deg2rad(1)]
 b=np.r_[np.array(after['position_m'])/.005,np.array(after['rotation_rad'])/np.deg2rad(1)]
 return bool(after['domain'] and np.isfinite(b).all() and max(b)<max(a)-1e-9 and np.all(b[a<=1]<=1))
def dist(x):
 x=np.array(x);return dict(n=len(x),mean=float(x.mean()),median=float(np.median(x)),p95=float(np.percentile(x,95)),max=float(x.max()))
def metric(r,ex=False):
 v=r if ex else r['validation']
 p=max(v['final_position_error_m' if ex else 'frozen_goal_position_error_m'])*1000
 a=float(np.rad2deg(max(v['final_rotation_error_rad' if ex else 'frozen_goal_rotation_error_rad'])))
 return dict(position_mm=p,rotation_deg=a,Q=max(p/5,a))
def aggregate(rs,ex=False):
 ms=[metric(r,ex) for r in rs]
 return dict(denominator=len(rs),missing_endpoint=0,**{k:dist([m[k] for m in ms]) for k in ms[0]},
  pose_pass=sum(m['Q']<=1 for m in ms),qualified=sum(r.get('eligible',r.get('level')=='EXECUTION_VERIFIED') for r in rs))
def paired(a,b):
 assert a.keys()==b.keys()
 d=np.array([[b[k][m]-a[k][m] for m in ('position_mm','rotation_deg','Q')] for k in sorted(a)])
 return {'n':len(d),'position_better':int((d[:,0]<0).sum()),'position_worse':int((d[:,0]>0).sum()),
 'rotation_better':int((d[:,1]<0).sum()),'rotation_worse':int((d[:,1]>0).sum()),'delta':{k:dist(d[:,j]) for j,k in enumerate(('position_mm','rotation_deg','Q'))}}
result={'status':'RUNNING','prohibited_operations_guarded':True}
try:
 run=read(OUT/'run.json');ident=run['identity'];selection=ident['selection']
 assert run['producer_head'].startswith('ecaf2519') and run['TEST'] is False
 assert len(ident['source_hashes'])==164 and len(ident['binding']['files'])==1096
 with zipfile.ZipFile(touch(OUT/'executed_source.zip')) as z:
  for p,h in ident['source_hashes'].items():
   assert sha(touch(ROOT/p))==h==hashlib.sha256(z.read(p)).hexdigest()
 for p,h in ident['binding']['files'].items():assert sha(touch(ROOT/p))==h
 touch(ROOT/'ff_prior_endpoint_jacobian/docs/EXPERIMENT_PLAN.md')
 assert selection==read(ROOT/'ff_prior_endpoint_jacobian/evidence/selection.json')
 assert ident['binding']==read(ROOT/'ff_prior_endpoint_jacobian/evidence/binding.json')
 result['identity']={'sources':164,'binding_files':1096,'producer_head':run['producer_head']}
 oldsources=read(OLD/'run.json')['identity']['source_hashes']
 common=set(oldsources)&set(ident['source_hashes'])
 assert len(common)==143 and all(oldsources[k]==ident['source_hashes'][k] for k in common)
 assert read(OUT/'physical_contract.json')['acceptance']==read(OLD/'physical_contract.json')['acceptance']
 result['identity']['unchanged_shared_control_sources']=len(common)
 print('Identity hashes verified',flush=True)
 model,codec=warm_load(ROOT/'artifacts/v7_1/delivery_train_dev_001/learning/goal_loss')
 arrays,rows,G,data=load_data(codec)
 source_index=read(OLD/'training_input_audit.json')
 for e in source_index['records']:assert sha(touch(ROOT/e['record']))==e['record_sha256']
 assert len(data['tr'])==14222 and len(data['va'])==48
 di={e['sample_id'] for e in selection['diagnostic']};ev={e['sample_id'] for e in selection['events']}
 assert len(di)==8 and len(ev)==32 and not di&ev
 for e in selection['diagnostic']+selection['events']:
  i=data['ids'][e['sample_id']];assert i in data['tr'] and rows[i]['task']['task_hash']==e['task_hash']
 result['data']={'TRAIN':len(data['tr']),'VAL':len(data['va']),'record_hashes':len(source_index['records']),
 'all_data_content_and_G_hashes_verified_by_loader':True,'disjoint_world_target_pools':True}
 events=[];numerical=[];counter=[];total_vjp=0
 paths=sorted(OUT.glob('diagnostic/*/result.json'))+sorted(OUT.glob('world_events/*/result.json'))
 for path in paths:
  e=read(path);before=loadpt(path.parent/'before.pt');after=loadpt(path.parent/'after.pt')
  if path.parent.parent.name=='diagnostic':
   frozen=selection['cases'][int(path.parent.name)]
   expected_specs=[(data['ids'][selection['diagnostic'][frozen['target']]['sample_id']],frozen['t'],frozen['noise_seed'])]
   expected_controls=[(data['ids'][selection['diagnostic'][frozen['control_target']]['sample_id']],frozen['t'],frozen['noise_seed']+100000)]
  else:
   frozen=selection['world_sequence'][int(path.parent.name)//100-1]
   assert frozen['update']==int(path.parent.name)
   expected_specs=[(data['ids'][selection['events'][frozen['targets'][0]]['sample_id']],frozen['t'],frozen['noise_seeds'][0])]
   expected_controls=[(data['ids'][selection['events'][frozen['control_targets'][0]]['sample_id']],frozen['t'],frozen['noise_seeds'][1])]
  assert before['specs']==expected_specs and before['controls']==expected_controls
  model.load_state_dict(before['weights']);model.eval();original=vparams(model)
  assert weight_hash(model)==e['input_weight_hash']
  assert statehash(before['optimizer'])==e['optimizer_before_hash'] and statehash(after['optimizer'])==e['optimizer_after_hash']
  assert sha(path.parent/'after.pt')==e['after_sha256']
  components=[]
  for gp in sorted(path.parent.glob('gradient_*.json')):
   proof=read(gp);assert sha(touch(gp.with_suffix('.npz')))==proof['arrays_sha256']
   i=data['ids'][proof['sample_id']];goals=rows[i]['task']['condition']['goals']
   assert (i,proof['t'],proof['noise_seed'])==(before['specs']+before['controls'])[int(gp.stem.split('_')[-1])]
   assert proof['task_hash']==rows[i]['task']['task_hash'] and proof['input_weight_hash']==e['input_weight_hash']
   assert proof['base_hash']==digest(arrays[1][i]) and proof['codec_hash']==digest(codec.state)
   assert proof['center_hash']==digest(data['centers'][i])
   pred=predict(model,data,i,proof['t'],proof['noise_seed'])
   with np.load(gp.with_suffix('.npz')) as a:
    assert np.array_equal(a['x'],pred.detach().numpy().astype(float)) and digest(a['x'])==proof['x_hash']
    q=groups(proof['central']['ee'],goals);qp=groups(a['perturbation_ee'],goals)
    J=(qp[:,1]-qp[:,0])/.02;w=np.exp(q-q.max());w=w/w.sum()+.0125;g=J@w
    assert g.shape==(102,) and proof['dimensions']==102 and proof['epsilon']==.01
    for n,v in [('central_groups',q),('perturbation_groups',qp),('group_jacobian',J),('analytic_smooth_weights',w),('gradient',g)]:close(a[n],v)
    lp=loss(qp);close(lp,a['perturbation_losses']);close(loss(q),proof['true_loss'])
    close((lp[:,1]-lp[:,0])/.02,a['legacy_scalar_gradient'])
    half=[]
    assert [h['dimension'] for h in proof['half_checks']]==[0,51]
    for h in proof['half_checks']:
     hh=[loss(groups(s['ee'],goals)) for s in h['samples']];slope=(hh[1]-hh[0])/.01
     err=abs(slope-g[h['dimension']])/max(abs(slope),abs(g[h['dimension']]),1e-6)
     close(err,h['relative_error']);assert err<=.02 and h['passed'];half.append(float(err))
    direction=np.sin(np.arange(102)+.37);direction/=np.linalg.norm(direction);close(direction,proof['direction'])
    ds=[loss(groups(s['ee'],goals)) for s in proof['direction_samples']]
    slope=(ds[1]-ds[0])/.02;err=abs(slope-g@direction)/max(abs(slope),abs(g@direction),1e-6)
    close(err,proof['direction_relative_error']);assert err<=.05
    cp,cr=poses(proof['central']['ee'],proof['convergence']['fine']['ee'])
    assert max(cp)<=.001 and max(cr)<=np.deg2rad(.2) and proof['status']=='PASS' and proof['calls']['coupled']==212
    # Direct neural VJP from dot product rather than producer's attach_gradient helper.
    components.append(vg(torch.sum(pred*torch.tensor(g,dtype=pred.dtype)),model));total_vjp+=1
    numerical.append({'file':str(gp.relative_to(ROOT)),'half_max':max(half),'direction_error':float(err),'convergence_position_m':float(max(cp)),'convergence_rotation_rad':float(max(cr))})
  with np.load(touch(path.parent/'parameter_proposals.npz')) as a:
   world=torch.stack(components).mean(0)
   close(torch.stack(components).numpy(),a['world_component_gradients'],atol=1e-6,rtol=1e-7);close(world.numpy(),a['world_gradient'],atol=1e-6,rtol=1e-7)
   batch=before['base_batch']
   if batch is None:
    ll=[]
    for i,t,seed in before['specs']+before['controls']:
     noise=torch.randn((1,102),generator=torch.Generator().manual_seed(seed))
     ll.append(base_loss(model,data,G,[i],torch.tensor([t]),noise))
    bl=torch.stack(ll).mean()
   else:bl=base_loss(model,data,G,batch['indices'],batch['t'],batch['noise'])
   base=vg(bl,model);close(base.numpy(),a['base_gradient'],atol=1e-7,rtol=1e-7)
   close(e['world_coefficient'],.1*float(base.norm())/float(world.norm()))
   bv,bs=adam(model,base,before['optimizer']);mv,ms=adam(model,base+e['world_coefficient']*world,before['optimizer'])
   close((bv-original).numpy(),a['base_delta'],atol=6e-8,rtol=0)
   close((mv-original).numpy(),a['combined_delta'],atol=6e-8,rtol=0)
   expected=before['optimizer'] if not e['updates_retained'] else ms if e['accepted'] else bs
   compare_state(after['optimizer'],expected)
   expected_vector=original if not e['updates_retained'] else original+e['fraction']*torch.tensor(a['combined_delta']) if e['accepted'] else original+torch.tensor(a['base_delta'])
   model.load_state_dict(after['weights']);close(vparams(model),expected_vector,atol=6e-8,rtol=0);assert weight_hash(model)==e['output_weight_hash']
   model.load_state_dict(before['weights'])
   accepted=[t['fraction'] for t in e['trials'] if all(admissible(b,v) for b,v in zip(e['before'],t['after']))]
   assert bool(accepted)==e['accepted'] and (accepted[0] if accepted else None)==e['fraction']
   assert [t['fraction'] for t in e['trials']]==[1.,.5,.25][:len(e['trials'])]
   assert [t['fraction'] for t in e['base_trials']]==[t['fraction'] for t in e['trials']]
   for kind,key in [('trials','combined_delta'),('base_trials','base_delta')]:
    for t in e[kind]:
     ok=all(admissible(b,v) for b,v in zip(e['before'],t['after']))
     assert ok==t['accepted' if kind=='trials' else 'admissible']
     setv(model,original+t['fraction']*torch.tensor(a[key]))
     for spec,v in zip(before['specs']+before['controls'],t['after']):
      i,tt,seed=spec
      with torch.no_grad():pred=predict(model,data,i,tt,seed).numpy().astype(float)
      assert digest(pred)==v['x_hash']
      pp,rr=poses(v['ee'],rows[i]['task']['condition']['goals']);close(pp,v['position_m']);close(rr,v['rotation_rad'])
      close(loss(groups(v['ee'],rows[i]['task']['condition']['goals'])),v['loss'])
   n=424+len(e['before'])+sum(len(t['after']) for t in e['trials']+e['base_trials']);assert n==e['actual_coupled_calls']
   rec=dict(path=str(path.parent.relative_to(OUT)),accepted=e['accepted'],fraction=e['fraction'],calls=n,
    nonempty_before=len(before['optimizer']['state']),combined_minus_base_delta_norm=float(np.linalg.norm(a['combined_delta']-a['base_delta'])),
    base_delta_norm=float(np.linalg.norm(a['base_delta'])),combined_delta_norm=float(np.linalg.norm(a['combined_delta'])),
    selected_base_admissible=next((t['admissible'] for t in e['base_trials'] if t['fraction']==e['fraction']),None))
   if path.parent.parent.name=='world_events':
    step=int(path.parent.name);assert all(float(s['step'])==step-1 for s in before['optimizer']['state'].values())
    assert all(float(s['step'])==step for s in after['optimizer']['state'].values())
    assert e['updates_retained'] and before['optimizer']['state'];rec['update']=step
   else:assert not e['updates_retained'] and e['input_weight_hash']==e['output_weight_hash']
   events.append(rec)
  if len(events)%12==0:print('Verified events',len(events),'gradients',total_vjp,flush=True)
 assert len(paths)==84 and total_vjp==168
 m1=events[:24];m2=events[24:]
 assert sum(x['accepted'] for x in m1)==24 and sum(x['accepted'] for x in m2)==31
 result['gradient_checks']={'count':len(numerical),'max_half_error':max(x['half_max'] for x in numerical),'max_direction_error':max(x['direction_error'] for x in numerical),'neural_VJP':total_vjp}
 result['events']=events
 (DEST/'gradient_evidence.json').write_text(json.dumps(numerical,indent=2))
 print('Algebra complete; checking ordinary random sequence',flush=True)
 # Reconstruct ordinary sampling/RNG stream without training or model updates.
 train=read(OUT/'learning/training.json');oldtrain=read(OLD/'learning/goal_loss/training.json')
 assert train['segments']==oldtrain['segments'] and len(train['segments'])==300 and train['updates']==6000 and train['VAL_updates']==0
 from ff_prior_origin_learn_bounded.distribution import CoverageSampler
 sampler=CoverageSampler(data['keys'],20263003);torch.manual_seed(20263001);proofs=[];segments=[]
 for step in range(6000):
  local=sampler.take(64);t=torch.randint(0,1000,(64,));noise=torch.randn((64,102))
  proofs.append(digest({'update':step+1,'batch':local,'t':t.numpy(),'noise':noise.numpy()}))
  if (step+1)%100==0:
   batch=loadpt(OUT/'world_events'/f'{step+1:04d}'/'before.pt')['base_batch']
   assert batch['indices']==data['tr'][local].tolist();assert torch.equal(t,batch['t']) and torch.equal(noise,batch['noise'])
  if (step+1)%20==0:
   segments.append({'start_update':step-18,'end_update':step+1,'draws':1280,'batch_rng_digest':digest(proofs)});proofs=[]
 assert segments==train['segments']
 counts=dict(zip([arrays[3][i] for i in data['tr']],(sampler.row_counts+sampler.origin_row_counts).tolist()))
 assert counts==train['sampling']['train_draw_counts']==oldtrain['train_draw_counts']
 assert len(counts)==14222 and min(counts.values())>0 and sum(counts.values())==384000
 progress=loadpt(OUT/'learning/progress.pt')
 assert progress['updates']==6000 and all(float(v['step'])==6000 for v in progress['optimizer']['state'].values())
 checkpoint=loadpt(OUT/'learning/model.pt');assert sha(OUT/'learning/model.pt')==train['checkpoint_sha256']
 for k,v in checkpoint['weights'].items():assert torch.equal(v,progress['weights'][k])
 result['ordinary_training']={'segments_rebuilt':300,'updates':6000,'TRAIN_draws':sum(counts.values()),'minimum_draws_per_row':min(counts.values()),'event_batches_matched':60,'optimizer_final_step':6000}
 # Recompute distributions directly from raw candidate and execution records.
 stats=read(OUT/'paired_statistics.json');metrics={};records={};entries=read(OUT/'frozen_tasks.json')['entries']
 assert read(OLD/'frozen_tasks.json')['entries']==entries and len(entries)==12
 for label,folder,method in [('A',OLD,'expanded_diffusion'),('structured',OLD,'structured'),('mean',OLD,'learned_mean'),('B',OUT,None)]:
  cs={};es={}
  for sj,seed in enumerate([20262119,20262123,20262129]):
   for ti,entry in enumerate(entries):
    sid=entry['sample_id'];task=entry['task'];base=folder/'candidates'/str(sj)/sid
    if method:base/=method
    group=[]
    for k in range(5):
     c=read(base/f'{k:02d}.json');assert c['task_hash']==task['task_hash'] and c['seed']==seed+ti*101+k*997
     cs[f'{sj}/{sid}/{k:02d}']=c;group.append(c)
     if label=='B':
      from ff_prior.spline import Reference
      assert Reference(c['Z'],task['condition']['initial_features'],task['condition']['duration']).reference_hash==c['reference_hash']
      v=c['validation'];valid=all(v['gates'][g] for g in v['required'])
      assert c['eligible']==bool(valid and metric(c)['Q']<=1) and c['status']=='DONE'
      assert v['geometry']['maximum_geometry_step_m']<=.002 and v['geometry']['minimum_clearance_m']>=0
      assert c['actual_network_calls']==16 and c['model_calls']['coupled']<=113 and c['repair_coupled_calls']<=110 and c['validation_coupled_calls']<=3
      assert c['integration_intervals']+c['validation_integration_intervals']<=15872 and c['wall_s']<=15
      assert len(c['sampling']['steps'])==16 and max(s['decomposition_residual_after'] for s in c['sampling']['steps'])<=1e-10
      assert [g['allocation'] for g in c['sampling']['guidance']]==[27,27,27,29]
      assert all(g['coupled_calls']<=g['allocation'] and g['latent_feedback_max_rad']<=1e-12 for g in c['sampling']['guidance'])
    d=folder/'executions'/str(sj)/sid
    if method:d/=method
    rep=read(d/'report.json');sel=read(d/'selection.json');outcome=read(d/'outcome.json')
    chosen=min([c for c in group if c['eligible']],key=lambda c:(c['score'],c['candidate_id']))
    assert sel['selected_id']==chosen['candidate_id'] and sel['selection_before_execution']
    assert rep['task_hash']==task['task_hash'] and rep['metadata']['reference_hash']==chosen['reference_hash']
    assert outcome['status']==rep['level']=='EXECUTION_VERIFIED';es[f'{sj}/{sid}']=rep
    if label=='B':
     assert read(d/'frozen_task.json')==task
     assert rep['scene_model_hash']==task['contract']['robot_hash'] and rep['initial_state_id']==task['contract']['initial_state_id']
     assert all(rep['gates'].values()) and rep['replay']['passed'] and max(rep['replay']['maximum_errors'].values())<=1e-10
     assert rep['metadata']['robot_state_writes_during_step_loop']==0 and not rep['metadata']['external_wrench_applied']
     with np.load(touch(d/'native_trace.npz')) as a:
      assert a['qpos'].shape[1]==81 and a['ctrl'].shape[1]==67 and len(a['qpos'])==len(a['ctrl'])+1
      assert digest(a['ctrl'])==rep['ctrl_hash'] and len(a['ctrl'])==rep['metadata']['physics_steps']==rep['replay']['physics_steps']
      pp,rr=poses(a['ee'],task['condition']['goals']);hold=a['time']-a['time'][0]>=task['condition']['duration']-1e-9
      close(pp[-1],rep['final_position_error_m']);close(rr[-1],rep['final_rotation_error_rad'])
      close(pp[hold].max(axis=0),rep['hold_max_position_error_m']);close(rr[hold].max(axis=0),rep['hold_max_rotation_error_rad'])
      assert pp[hold].max()<=.005 and rr[hold].max()<=np.deg2rad(1)
      # Report includes subdivisions; saved clearance_m contains native states only.
      assert 0<=rep['minimum_clearance_m']<=a['clearance_m'].min()+1e-12
      assert rep['checked_states']==2*len(a['ctrl'])+1
      close(abs(a['manifold_residual']).max(),rep['manifold_max_rad'])
      assert np.isfinite(a['qpos']).all() and np.isfinite(a['raw_torque']).all()
      assert a['time'][-1]-a['time'][0]<=30+1e-9
  assert len(cs)==180 and len(es)==36
  metrics[label]={'candidate':aggregate(list(cs.values())),'execution':aggregate(list(es.values()),True)}
  target=stats['B'] if label=='B' else stats['reused'][label]
  treeclose(metrics[label]['candidate'],target['candidate']);treeclose(metrics[label]['execution'],target['execution'])
  records[label]={'candidate':{k:metric(v) for k,v in cs.items()},'execution':{k:metric(v,True) for k,v in es.items()}}
 result['DEV_distributions']=metrics
 result['DEV_pairs']={label:{kind:paired(records[label][kind],records['B'][kind]) for kind in ('candidate','execution')} for label in ('A','structured','mean')}
 result['unguided']={}
 from ff_prior_expand256_train.learning import load as aload
 from ff_prior_endpoint_jacobian.learning import load as bload
 from ff_prior.diffusion import schedule
 bundles={'A':aload(OLD/'learning/goal_loss'),'B':bload(OUT/'learning')}
 alpha=schedule().double();times=np.linspace(999,0,16).round().astype(int);raw_rebuilt=0
 for sj in range(2):
  mm={}
  for label in ('A','B'):
   rs=[read(p) for p in sorted(OUT.glob(f'unguided/{sj}/{label}/*.json'))];assert len(rs)==48
   for r in rs:
    assert len(r['steps'])==16 and r['physics_guidance_calls']==0 and r['calls']['coupled']<=3
    assert max(s['residual'] for s in r['steps'])<=1e-10 and r['sample_id'] in {arrays[3][i] for i in data['va']}
    assert np.asarray(r['raw_Z']).shape==np.asarray(r['prepared_Z']).shape
    i=data['ids'][r['sample_id']];j=list(data['va']).index(i)
    assert r['seed']==[20263111,20263113][sj]+j*997 and r['task_hash']==rows[i]['task']['task_hash']
    assert r['endpoint_scope']=='prepared un-guided full16 DDIM; raw path stored, no claimed raw endpoint'
    net,cc=bundles[label];base=arrays[1][i];condition=rows[i]['task']['condition']
    center=cc.center(condition,base);cond=torch.tensor(cc.condition(condition,base),dtype=torch.float32)[None]
    x=torch.randn((1,102),generator=torch.Generator().manual_seed(r['seed'])).double()
    for j,t in enumerate(times):
     with torch.no_grad():v=net(x.float(),torch.tensor([int(t)]),cond).double()
     clean=alpha[t].sqrt()*x-(1-alpha[t]).sqrt()*v
     eps=(x-alpha[t].sqrt()*clean)/(1-alpha[t]).sqrt()
     an=alpha[times[j+1]] if j<15 else torch.tensor(1.,dtype=torch.float64)
     x=an.sqrt()*clean+(1-an).sqrt()*eps
    raw=cc.decode(cc.coefficients(x[0].numpy(),center),base)
    close(raw,r['raw_Z'],atol=1e-12,rtol=1e-12);raw_rebuilt+=1
   summary=aggregate(rs);treeclose(summary,stats['unguided'][f'{sj}_{label}']);result['unguided'][f'{sj}_{label}']=summary
   mm[label]={r['sample_id']:metric(r) for r in rs}
  result['unguided'][f'{sj}_paired']=paired(mm['A'],mm['B'])
 result['unguided_raw_full16_rebuilt']=raw_rebuilt
 ledger=read(OUT/'ledger.json');status=read(OUT/'status.json');budget=read(OUT/'query_budget.json')
 assert ledger['inflight'] is None and not ledger.get('unresolved_call_accounting')
 assert status['status']=='FINISHED' and status['elapsed_s']<=21600 and budget['reserved_coupled_upper']<=60000
 assert ledger['attempts']=={'candidate':180,'diagnostic':24,'execution':36,'training_segment':300,'unguided':192}
 assert len(ledger['completed'])==732 and not any(x['status'] in ('ERROR','UNKNOWN','INFLIGHT') for x in ledger['completed'].values())
 result['budget']={'actual_calls':ledger['calls'],'reserved_coupled_upper':budget['reserved_coupled_upper'],'wall_s':status['elapsed_s'],'stage_wall_s':ledger['wall_s'],
 'output_bytes':sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file()),'pending_work':None}
 assert result['budget']['output_bytes']<=20*1024**3
 audit=ROOT/'paper/PAPER_CLAIM_AUDIT.json'
 result['paper_claim_audit']=read(audit) if audit.exists() else 'unavailable'
 result['status']='PASS'
except Exception:
 result['status']='FAIL';result['error']=traceback.format_exc();print(result['error'],flush=True)
finally:
 (DEST/'independent_evidence.json').write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf-8')
 (DEST/'inputs.json').write_text(json.dumps({'reviewed_artifacts':inputs,'script_sha256':sha(__file__)},indent=2,ensure_ascii=False),encoding='utf-8')
 print(json.dumps({k:v for k,v in result.items() if k in ('status','error','gradient_checks','ordinary_training','budget')},ensure_ascii=False),flush=True)
 if result['status']!='PASS':sys.exit(1)
