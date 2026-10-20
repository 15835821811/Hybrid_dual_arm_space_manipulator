"""Fresh reviewer: read-only saved poses, neural autograd and AdamW algebra. No simulator or optimizer calls."""
from pathlib import Path
import sys,json,hashlib,zipfile,datetime,collections
import numpy as np
import torch
from scipy.spatial.transform import Rotation
ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT))
from ff_prior_latent.learning import Codec,LatentDenoiser
from ff_prior.diffusion import schedule
from ff_prior.storage import digest
OUT=Path(__file__).resolve().parent
RUN=ROOT/'artifacts/v7_1/smooth_endpoint_bounded_001'
seen=set()
def read(p):
 p=Path(p);seen.add(p);return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def tensorload(p):seen.add(p);return torch.load(p,map_location='cpu',weights_only=False)
def npzload(p):seen.add(p);return np.load(p)
def groups(p):return np.r_[np.asarray(p['position_m'])/.005,np.asarray(p['rotation_rad'])/np.deg2rad(1)]
def admiss(a,b):
 x,y=groups(a),groups(b)
 return bool(b['domain'] and np.isfinite(y).all() and y.max()<x.max()-1e-9 and np.all(y[x<=1]<=1))
def pose(ee,goals):
 ee,goals=np.asarray(ee),np.asarray(goals)
 pos=np.linalg.norm(ee[...,:3,3]-goals[...,:3,3],axis=-1)
 rot=Rotation.from_matrix((np.swapaxes(goals[...,:3,:3],-1,-2)@ee[...,:3,:3]).reshape(-1,3,3)).magnitude().reshape(pos.shape)
 return pos,rot
def smooth(ee,goals):
 pos,rot=pose(ee,goals);q=np.r_[(pos/.005)**2,(rot/np.deg2rad(1))**2];m=q.max()
 return float(m+np.log(np.exp(q-m).mean())+.05*q.mean())
run=read(RUN/'run.json');ident=run['identity'];source_check={}
with zipfile.ZipFile(RUN/'executed_source.zip') as archive:
 seen.add(RUN/'executed_source.zip')
 for p,h in ident['source_hashes'].items():
  seen.add(ROOT/p);source_check[p]=(sha(ROOT/p)==h and hashlib.sha256(archive.read(p)).hexdigest()==h)
assert all(source_check.values())
CONTROL=ROOT/'artifacts/v7_1/expand256_train_dev_001'
codec=Codec(read(ROOT/'artifacts/v7_1/delivery_train_dev_001/learning/goal_loss/codec.json'))
index=read(CONTROL/'training_input_audit.json')['records'];sens=read(CONTROL/'sensitivity_summary.json')['bindings']
arr=npzload(CONTROL/'loaded_training_arrays.npz');Z=arr['Z'];bases=arr['bases']
cache={}
for path in sorted(RUN.glob('diagnostic/*/before.pt')):
 before=tensorload(path)
 for i,t,seed in before['specs']+before['controls']:
  if i in cache:continue
  entry=index[i];row=read(ROOT/entry['record']);assert sha(ROOT/entry['record'])==entry['record_sha256'] and entry['split']=='TRAIN'
  c=row['condition'];center=codec.center(c,bases[i]);x=codec.normalized(codec.encode(Z[i],bases[i]),center)
  gb=sens[row['task']['task_hash']];G=npzload(ROOT/gb['array'])['G'];assert sha(ROOT/gb['array'])==gb['array_sha256']
  cache[i]={'x':torch.tensor(x,dtype=torch.float32)[None],'c':torch.tensor(codec.condition(c,bases[i]),dtype=torch.float32)[None],'G':torch.tensor(G*codec.latent_scale[None,:],dtype=torch.float32)[None],'row':row,'id':entry['sample_id'],'center':center}
alpha=schedule();torch.set_num_threads(2)
model=LatentDenoiser(102,len(codec.feature_mean));model.eval()
def prediction(spec):
 i,t,seed=spec;d=cache[i];a=alpha[t];noise=torch.randn((1,102),generator=torch.Generator().manual_seed(seed))
 noisy=a.sqrt()*d['x']+(1-a).sqrt()*noise
 v=model(noisy,torch.tensor([t]),d['c']);pred=a.sqrt()*noisy-(1-a).sqrt()*v
 target=a.sqrt()*noise-(1-a).sqrt()*d['x']
 loss=(v-target).square().mean()+.05*torch.einsum('bij,bj->bi',d['G'],pred-d['x']).square().mean()
 return pred[0],loss
params=lambda:tuple(model.parameters())
def grad(x):return torch.cat([p.reshape(-1) for p in torch.autograd.grad(x,params())]).detach()
def vector():return torch.cat([p.detach().reshape(-1) for p in params()])
def adam_delta(g,state):
 pg=state['param_groups'][0];assert not state['state'] # M1 is a fresh optimizer in each case.
 pieces=[];offset=0;norm=torch.linalg.vector_norm(torch.stack([torch.linalg.vector_norm(g[s:s+p.numel()]) for p,s in zip(params(),np.cumsum([0]+[p.numel() for p in params()])[:-1])]))
 factor=torch.clamp(1/(norm+1e-6),max=1.)
 for p in params():
  gg=g[offset:offset+p.numel()].view_as(p)*factor;offset+=p.numel();b1,b2=pg['betas']
  m=torch.zeros_like(p).lerp_(gg,1-b1);v=torch.zeros_like(p).addcmul_(gg,gg,value=1-b2)
  w=p.detach().clone().mul_(1-pg['lr']*pg['weight_decay'])
  den=(v.sqrt()/(1-b2)**.5).add_(pg['eps']);w.addcdiv_(m,den,value=-pg['lr']/(1-b1));pieces.append((w-p.detach()).reshape(-1))
 return torch.cat(pieces).numpy()
failures=[];cases=[];comparisons=[];errors=[];passcount=0;current_count=0;max_loss_err=0.;max_adam=0.;max_autograd=0.
for path in sorted(RUN.glob('diagnostic/*/result.json')):
 case=int(path.parent.name);r=read(path);b=tensorload(path.parent/'before.pt');a=tensorload(path.parent/'after.pt');model.load_state_dict(b['weights']);v0=vector().clone()
 assert b['optimizer']['param_groups']==a['optimizer']['param_groups'] and not b['optimizer']['state'] and not a['optimizer']['state']
 assert all(torch.equal(b['weights'][k],a['weights'][k]) for k in b['weights']) and not a['retained']
 components=[];proofpass=[];specs=b['specs']+b['controls']
 for j,spec in enumerate(specs):
  p=read(path.parent/f'gradient_{j}.json');garr=npzload(path.parent/f'gradient_{j}.npz');g=garr['gradient'];losses=garr['perturbation_losses'];goals=cache[spec[0]]['row']['task']['condition']['goals'];pred,_=prediction(spec)
  assert np.array_equal(pred.detach().numpy().astype(float),garr['x']);current_count+=1
  assert digest(garr['x'])==p['x_hash'] and sha(path.parent/f'gradient_{j}.npz')==p['arrays_sha256']
  rebuilt=np.array([[smooth(ee,goals) for ee in pair] for pair in garr['perturbation_ee']]);max_loss_err=max(max_loss_err,float(np.max(abs(rebuilt-losses))))
  assert np.allclose(rebuilt,losses,atol=1e-12,rtol=1e-12)
  assert np.array_equal((losses[:,1]-losses[:,0])/(2*p['epsilon']),g) and g.shape==(102,)
  half=[]
  for h in p['half_checks']:
   hs=[smooth(s['ee'],goals) for s in h['samples']];slope=(hs[1]-hs[0])/p['epsilon'];e=abs(slope-g[h['dimension']])/max(abs(slope),abs(g[h['dimension']]),1e-6)
   assert abs(e-h['relative_error'])<1e-9;half.append(e)
  ds=[smooth(s['ee'],goals) for s in p['direction_samples']];slope=(ds[1]-ds[0])/(2*p['epsilon']);expected=float(g@np.asarray(p['direction']));err=abs(slope-expected)/max(abs(slope),abs(expected),1e-6)
  dp,dr=pose(p['central']['ee'],p['convergence']['fine']['ee']);conv=max(dp)<=.001 and max(dr)<=np.deg2rad(.2)
  ok=bool(np.isfinite(g).all() and conv and max(half)<=.02 and err<=.05 and p['calls']['coupled']==212)
  assert ok==(p['status']=='PASS');passcount+=int(ok);proofpass.append(ok)
  if not ok:failures.append({'case':case,'gradient':j,'role':p['role'],'sample_id':p['sample_id'],'t':p['t'],'direction_slope':slope,'gradient_dot_direction':expected,'absolute_direction_mismatch':abs(slope-expected),'relative_direction_error':err,'threshold':.05,'half_relative_errors':half,'convergence_position_max_m':float(max(dp)),'convergence_rotation_max_rad':float(max(dr)),'convergence_passed':bool(conv)})
  components.append(grad((pred*torch.tensor(g,dtype=pred.dtype)).sum()))
 proposal=npzload(path.parent/'parameter_proposals.npz');component=torch.stack(components);world=component.mean(0)
 losses=[prediction(s)[1] for s in specs];base=grad((losses[0]+losses[1])/2)
 max_autograd=max(max_autograd,float(np.max(abs(base.numpy()-proposal['base_gradient']))),float(np.max(abs(component.numpy()-proposal['world_component_gradients']))))
 assert np.array_equal(base.numpy(),proposal['base_gradient']) and np.array_equal(component.numpy(),proposal['world_component_gradients']) and np.array_equal(world.numpy(),proposal['world_gradient'])
 coeff=.1*float(base.norm())/float(world.norm());assert abs(coeff-r['world_coefficient'])<1e-12
 for kind,gg in [('base',base),('combined',base+coeff*world)]:
  error=float(np.max(abs(adam_delta(gg,b['optimizer'])-proposal[kind+'_delta'])));max_adam=max(max_adam,error);assert error<=6e-8
 assert [t['fraction'] for t in r['trials']]==[t['fraction'] for t in r['base_trials']]
 for combined,ordinary in zip(r['trials'],r['base_trials']):
  co=all(admiss(x,y) for x,y in zip(r['before'],combined['after']));ba=all(admiss(x,y) for x,y in zip(r['before'],ordinary['after']))
  assert co==combined['accepted'] and ba==ordinary['admissible']
  cq=[float(max(groups(q))) for q in combined['after']];bq=[float(max(groups(q))) for q in ordinary['after']]
  comparisons.append({'case':case,'fraction':combined['fraction'],'combined_admissible':co,'base_admissible':ba,'combined_Q':cq,'base_Q':bq,'selected_combined':r['accepted'] and combined['fraction']==r['fraction']})
  for kind,trial in [('combined',combined),('base',ordinary)]:
   torch.nn.utils.vector_to_parameters((v0+trial['fraction']*torch.tensor(proposal[kind+'_delta'])).clone(),params())
   for spec,rec in zip(specs,trial['after']):
    pred,_=prediction(spec);assert digest(pred.detach().numpy().astype(float))==rec['x_hash']
    assert abs(smooth(rec['ee'],cache[spec[0]]['row']['task']['condition']['goals'])-rec['loss'])<1e-9
   model.load_state_dict(b['weights'])
 calls=424+2+4*len(r['trials']);assert calls==r['actual_coupled_calls']
 if not all(proofpass):assert not r['trials'] and not r['base_trials'] and not r['accepted']
 cases.append({'case':case,'t':specs[0][1],'accepted':r['accepted'],'fraction':r['fraction'],'gradient_pass':all(proofpass),'calls':calls,'restored':True})
selected=[r for r in comparisons if r['selected_combined']];joint=collections.Counter((r['combined_admissible'],r['base_admissible']) for r in comparisons)
ledger=read(RUN/'ledger.json');summary=read(RUN/'summary.json');status=read(RUN/'status.json');query=read(RUN/'query_budget.json');reuse=read(RUN/'reuse_manifest.json');audit=read(RUN/'independent_verification.json')
assert ledger['inflight'] is None and not ledger.get('unresolved_call_accounting') and sum(c['calls'] for c in cases)==ledger['calls']['diagnostic.coupled']==query['reserved_coupled_upper']==10392
absent={p:not (RUN/p).exists() for p in ['learning','world_events','unguided','candidates','executions']};assert all(absent.values())
result={'status':'PASS','review_method':'independent raw reconstruction, no prior review, no physics/optimizer.step/training/resume/old audit','producer_head':run['producer_head'],'frozen_source_files_verified':len(source_check),'cases':cases,'accepted':sum(c['accepted'] for c in cases),'accepted_per_noise':{str(t):sum(c['accepted'] for c in cases if c['t']==t) for t in [333,667,999]},'full102_gradients':48,'gradient_pass':passcount,'failed_gradients':failures,'current_prediction_matches':current_count,'max_pose_loss_reconstruction_abs_error':max_loss_err,'max_neural_gradient_abs_error':max_autograd,'max_adamw_delta_abs_error':max_adam,'all_24_weights_and_empty_optimizers_restored':True,'matched_trials':comparisons,'matched_trial_count':len(comparisons),'matched_admissibility_counts':{str(k):v for k,v in joint.items()},'accepted_selected_scales_base_admissible':sum(r['base_admissible'] for r in selected),'selected_scales_combined_Q_better_both':sum(all(c<b for c,b in zip(r['combined_Q'],r['base_Q'])) for r in selected),'selected_scales_base_Q_better_both':sum(all(b<c for c,b in zip(r['combined_Q'],r['base_Q'])) for r in selected),'total_forward':sum(c['calls'] for c in cases),'run_elapsed_s':status['elapsed_s'],'diagnostic_ledger_wall_s':ledger['wall_s']['diagnostic'],'unresolved_calls':False,'new_stage_directories_absent':absent,'recorded_raw_audit':audit['status'],'reuse':reuse,'summary':summary,'paper_claim_audit':'unavailable','provisional_no_paper_claim_audit':True}
(OUT/'fresh_readonly_evidence.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
seen.add(ROOT/'ff_prior_smooth_endpoint/docs/EXPERIMENT_PLAN.md')
inputs={'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'inputs':[{'path':str(p),'sha256':sha(p),'mtime_ns':p.stat().st_mtime_ns,'bytes':p.stat().st_size} for p in sorted(seen)],'script_sha256':sha(__file__),'excluded':['earlier reviewer conclusions','execution of old audits','resume','physics propagation','optimizer.step','training']}
(OUT/'inputs.json').write_text(json.dumps(inputs,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in result.items() if k not in ['cases','matched_trials','summary','reuse']},ensure_ascii=False,indent=2))
