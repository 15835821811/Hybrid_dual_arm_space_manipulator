"""Read-only saved-data review. No physics calls, existing auditor, or optimizer step."""
import sys
sys.dont_write_bytecode = True
import json, hashlib, pathlib, datetime, subprocess, zipfile, collections
import numpy as np
import torch
from scipy.spatial.transform import Rotation
ROOT = pathlib.Path(__file__).resolve().parents[4]
TRACE = pathlib.Path(__file__).resolve().parent
RUN = ROOT/'artifacts/v7_1/preserve323_bounded_001'
sys.path.insert(0, str(ROOT))
torch.set_num_threads(2)
INPUTS = {}
def sha(b): return hashlib.sha256(b).hexdigest()
def register(p):
    p=pathlib.Path(p).resolve()
    if str(p) not in INPUTS:
        st=p.stat(); INPUTS[str(p)]={'sha256':sha(p.read_bytes()),'bytes':st.st_size,'mtime_utc':datetime.datetime.fromtimestamp(st.st_mtime,datetime.timezone.utc).isoformat()}
    return p
def js(p): return json.loads(register(p).read_text(encoding='utf-8-sig'))
def npz(p): return np.load(register(p))
def pt(p): return torch.load(register(p),map_location='cpu',weights_only=False)
def out(name,value):
    (TRACE/name).write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
def emit(k,v): print(k,json.dumps(v,ensure_ascii=True),flush=True)
def residual(ee,goal):
    ee=np.asarray(ee);goal=np.asarray(goal)
    pos=(ee[...,:3,3]-goal[...,:3,3])/.005
    mat=np.swapaxes(goal[...,:3,:3],-1,-2)@ee[...,:3,:3]
    rot=Rotation.from_matrix(mat.reshape(-1,3,3)).as_rotvec().reshape(mat.shape[:-2]+(3,))/np.deg2rad(1.)
    return np.concatenate((pos.reshape(ee.shape[:-3]+(6,)),rot.reshape(ee.shape[:-3]+(6,))),axis=-1)
def stat(x):
    x=np.asarray(x);return {'mean':float(x.mean()),'median':float(np.median(x)),'p95':float(np.percentile(x,95)),'worst':float(x.max())}
def maxerr(a,b):return float(np.max(np.abs(np.asarray(a)-np.asarray(b))))
def tmax(a,b):return float(torch.max(torch.abs(a-b)))

run=js(RUN/'run.json'); identity=run['identity'];selection=js(RUN/'frozen_tasks.json')
preflight=js(ROOT/'ff_prior_preserve323/evidence/preflight.json')
register(ROOT/'ff_prior_preserve323/docs/EXPERIMENT_PLAN.md')
assert selection==identity['selection']==js(ROOT/'ff_prior_preserve323/evidence/selection.json')
assert identity['binding']==js(ROOT/'ff_prior_preserve323/evidence/binding.json')
sources=identity['source_hashes']; binding=identity['binding']['files']; source_errors=[];input_errors=[]
with zipfile.ZipFile(register(RUN/'executed_source.zip')) as z:
    assert set(z.namelist())==set(sources)
    for name,expected in sources.items():
        if sha(z.read(name))!=expected or INPUTS[str(register(ROOT/name))]['sha256']!=expected:source_errors.append(name)
for name,expected in binding.items():
    if INPUTS[str(register(ROOT/name))]['sha256']!=expected:input_errors.append(name)
tree=subprocess.check_output(['git','ls-tree','-r',run['producer_head']],cwd=ROOT,text=True)
git_blobs={line.split('\t',1)[1]:line.split()[2] for line in tree.splitlines()}
commit_errors=[];commit_crlf_normalized=[]
for name in sources:
    b=(ROOT/name).read_bytes(); blob=hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest()
    if git_blobs.get(name)!=blob:
        normalized=b.replace(b'\r\n',b'\n');lf_blob=hashlib.sha1(b'blob '+str(len(normalized)).encode()+b'\0'+normalized).hexdigest()
        if git_blobs.get(name)==lf_blob:commit_crlf_normalized.append(name)
        else:commit_errors.append(name)
recorded_modified=[n for n in commit_errors if preflight['original_modified_files'].get(n)==sources[n]]
source_report={'producer_head':run['producer_head'],'source_count':len(sources),'bound_input_count':len(binding),'source_errors':source_errors,'input_errors':input_errors,'commit_content_differences':commit_errors,'preflight_recorded_original_modifications':recorded_modified,'commit_CRLF_normalized_files':len(commit_crlf_normalized),'preflight_source_count':preflight['source_files'],'preflight_bound_count':preflight['bound_files']}
assert len(sources)==preflight['source_files']==189 and len(binding)==preflight['bound_files']==19222
emit('binding',source_report)
assert not source_errors and not input_errors and set(commit_errors)==set(recorded_modified)

# Independently rebuild task exclusions and seeded strata from original array/index data.
audit=js(ROOT/'artifacts/v7_1/expand256_train_dev_001/training_input_audit.json')
records=[js(ROOT/e['record']) for e in audit['records']]
ids=[e['sample_id'] for e in audit['records']]; byid={s:i for i,s in enumerate(ids)}
arrays=npz(ROOT/'artifacts/v7_1/expand256_train_dev_001/loaded_training_arrays.npz');Z=arrays['Z'];bases=arrays['bases'];arrays.close()
training=js(RUN/'training_input.json');tr=[i for i,r in enumerate(records) if r['split']=='TRAIN'];va=[i for i,r in enumerate(records) if r['split']=='VAL']
fit_hash={r['task']['task_hash'] for r in selection['fit_pool']};check_hash={r['task']['task_hash'] for r in selection['check_pool']}
forbidden=set(selection['forbidden_TRAIN_task_hashes'])
excluded=[i for i in tr if records[i]['task']['task_hash'] in check_hash];adapt=[i for i in tr if i not in set(excluded)]
assert len(tr)==14222 and len(va)==48 and len(excluded)==34 and len(adapt)==14188
assert set(training['TRAIN_ids'])=={ids[i] for i in adapt} and set(training['excluded_check_ids'])=={ids[i] for i in excluded}
assert set(training['TRAIN_task_hashes'])=={records[i]['task']['task_hash'] for i in adapt}
assert not check_hash&fit_hash and not (fit_hash|check_hash)&forbidden
assert not (fit_hash|check_hash)&{records[i]['task']['task_hash'] for i in va}
unique={}
for i in tr:
    h=records[i]['task']['task_hash']
    if h not in forbidden and h not in unique:unique[h]=i
complex_ids=[byid[r['sample_id']] for r in selection['diagnostic'] if r['sample_id'].startswith('complex_train_')]
complex_hashes={records[i]['task']['task_hash'] for i in complex_ids}
ordinary=[i for h,i in sorted(unique.items()) if h not in complex_hashes]
ordinary.sort(key=lambda i:(float(np.sqrt(np.mean((Z[i]-bases[i])**2))),records[i]['task']['task_hash']))
bins=np.array_split(ordinary,3);rng=np.random.default_rng(20265011); rebuilt={'fit':[],'check':[]}
for name,pool,n in [('complex_existing',complex_ids,3),('motion_small',bins[0],4),('motion_medium',bins[1],4),('motion_large',bins[2],5)]:
    perm=rng.permutation(len(pool))
    for role,slots in [('fit',perm[:n]),('check',perm[n:2*n])]:
        rebuilt[role].extend((ids[int(pool[k])],name) for k in slots)
for role in ['fit','check']:
    assert rebuilt[role]==[(r['sample_id'],r['frozen_stratum']) for r in selection[role+'_pool']]
    assert selection[role+'_seed_pairs']==[[20265021+j*997+(500003 if role=='check' else 0),20265023+j*997+(500003 if role=='check' else 0)] for j in range(16)]
pool_report={'TRAIN_source':len(tr),'TRAIN_adaptation':len(adapt),'excluded_rows':len(excluded),'fit_tasks':len(fit_hash),'check_tasks':len(check_hash),'fit_check_overlap':0,'pool_VAL_DEV_overlap':0,'seeded_selection_reproduced':True,'scope':'held out from this adaptation only; old model and codec saw original TRAIN'}
emit('pools',pool_report)

groups={'fit_baseline':(RUN/'TRAIN_baseline/fit','fit'),'check_baseline':(RUN/'TRAIN_baseline/check','check'),'check_final':(RUN/'TRAIN_check_final','check')}
for k in range(3):groups['trial_'+str(k)]=(RUN/f'TRAIN_fit/rounds/00/trial_{k}','fit')
statistics={}; rows_by_group={};record_errors=[];pose_max=0.;all_raw_equal=True
for label,(directory,role) in groups.items():
    rows=js(directory/'records.json');rows_by_group[label]=rows
    q=[];positions=[];rotations=[];rough=[]; strata={}; seeds={}; reasons=collections.Counter(); levels=collections.Counter()
    for j,row in enumerate(rows):
        task=selection[role+'_pool'][j//2]['task'];case=directory/f'{j:02d}';saved=js(case/'validation.json');ident=js(case/'identity.json')
        assert ident['task_hash']==task['task_hash']==row['task_hash'] and ident['sample_id']==selection[role+'_pool'][j//2]['sample_id'] and ident['seed']==selection[role+'_seed_pairs'][j//2][j%2]
        with npz(case/'prediction.npz') as a:ee=a['ee'][-1].copy()
        res=residual(ee,task['condition']['goals']).reshape(4,3); norms=np.linalg.norm(res,axis=1);pos=norms[:2]*.005;rot=norms[2:]*np.deg2rad(1.)
        v=row['validation'];pose_max=max(pose_max,maxerr(pos,v['frozen_goal_position_error_m']),maxerr(rot,v['frozen_goal_rotation_error_rad']))
        goal=bool(np.all(norms<=1));eligible=goal and v['validation_level']=='MODEL_VALID'
        assert goal==row['frozen_goal_pass'] and eligible==row['eligible']==saved['eligible']
        for gate,val in v['gates'].items():
            if not val:reasons['gate:'+gate]+=1
        if not goal:reasons['goal']+=1
        if not eligible:reasons['ineligible']+=1
        with npz(case/'controls.npz') as a:
            equal=np.array_equal(a['raw'],a['prepared']);all_raw_equal &= equal
            assert equal==row['raw_equal_prepared'] and np.array_equal(a['prepared'],row['Z'])
            rough.append(float(np.mean(np.diff(a['raw'],n=2,axis=0)**2)))
        q.append(float(norms.max())); positions.append(float(pos.max()*1000));rotations.append(float(np.rad2deg(rot.max())));levels[v['validation_level']]+=1
        for mapping,key in [(strata,row['stratum']),(seeds,str(j%2))]:
            cell=mapping.setdefault(key,{'n':0,'eligible':0,'Q':[]});cell['n']+=1;cell['eligible']+=int(eligible);cell['Q'].append(q[-1])
    for mapping in [strata,seeds]:
        for cell in mapping.values():cell['Q']=stat(cell['Q'])
    statistics[label]={'n':len(rows),'eligible':sum(r['eligible'] for r in rows),'failures':dict(reasons),'validation_levels':dict(levels),'Q':stat(q),'position_max_arm_mm':stat(positions),'rotation_max_arm_deg':stat(rotations),'roughness_rms_rad':float(np.sqrt(np.mean(rough))),'strata':strata,'seed_index':seeds,'all_finite':bool(np.isfinite(q).all())}
    emit(label, {k:v for k,v in statistics[label].items() if k not in ['strata','seed_index']})
baseline=statistics['fit_baseline']
for k in range(3):
    s=statistics[f'trial_{k}'];s['mean_Q_improvement_fraction']=1-s['Q']['mean']/baseline['Q']['mean'];s['median_Q_improvement_fraction']=1-s['Q']['median']/baseline['Q']['median']
    s['baseline_qualified_lost']=sum(a['eligible'] and not b['eligible'] for a,b in zip(rows_by_group['fit_baseline'],rows_by_group[f'trial_{k}']))
assert pose_max<1e-10 and all_raw_equal

# Every saved +/- pose is converted back to a world residual, then J/checks rebuilt.
jac_report=[]
for j in range(32):
    path=RUN/f'TRAIN_fit/rounds/00/jacobians/{j:02d}';proof=js(path.with_suffix('.json'))
    with npz(path.with_suffix('.npz')) as a:dat={k:a[k].copy() for k in a}
    goal=selection['fit_pool'][j//2]['task']['condition']['goals'];r=residual(dat['central_ee'],goal);pr=residual(dat['perturbation_ee'],goal);J=(pr[:,1]-pr[:,0]).T/.02
    half=[];half_res_err=0.
    for h in proof['half_checks']:
        rr=[residual(q['ee'],goal) for q in h['records']];slope=(rr[1]-rr[0])/.01;col=J[:,h['dimension']]
        half.append(float(np.linalg.norm(slope-col)/max(np.linalg.norm(slope),np.linalg.norm(col),1e-6)));half_res_err=max(half_res_err,maxerr(rr,h['residuals']))
    dr=[residual(q['ee'],goal) for q in proof['direction_records']];direction=np.sin(np.arange(323)+.37);direction/=np.linalg.norm(direction)
    slope=(np.square(dr[1]).reshape(4,3).sum(1).mean()-np.square(dr[0]).reshape(4,3).sum(1).mean())/.02;expected=float(r@J@direction/2)
    direction_error=float(abs(slope-expected)/max(abs(slope),abs(expected),1e-6))
    fine=np.asarray(proof['convergence']['fine']['ee']);coarse=np.asarray(dat['central_ee']);dp=np.linalg.norm(coarse[:,:3,3]-fine[:,:3,3],axis=1)
    drot=np.linalg.norm(Rotation.from_matrix(np.swapaxes(coarse[:,:3,:3],-1,-2)@fine[:,:3,:3]).as_rotvec(),axis=1)
    item={'j':j,'J_max_error':maxerr(J,dat['J']),'perturbed_residual_max_error':maxerr(pr,dat['perturbation_residuals']),'central_residual_max_error':maxerr(r,dat['residual']),'half_residual_max_error':half_res_err,'half_relative_max':max(half),'direction_relative':direction_error,'propagation_position_m':float(dp.max()),'propagation_rotation_deg':float(np.rad2deg(drot.max())),'coupled_calls':proof['calls']['coupled']}
    assert item['J_max_error']<1e-8 and max(half)<=.02 and direction_error<=.05 and dp.max()<=.001 and drot.max()<=np.deg2rad(.2) and proof['calls']['coupled']==654
    assert proof['task_hash']==selection['fit_pool'][j//2]['task']['task_hash'];jac_report.append(item)
emit('jacobians',{'n':32,**{key:max(x[key] for x in jac_report) for key in jac_report[0] if key!='j'}})

# Neural forward only, plus autograd VJP and hand-written algebra for saved update endpoints.
from ff_prior_preserve323.model import TransferDenoiser
from ff_prior_preserve323.codec import FullCodec
from ff_prior_control_transfer.model import rollout
from ff_prior.diffusion import schedule
codec=FullCodec(js(RUN/'codec.json'));initial=pt(RUN/'TRAIN_fit/initial.pt');final=pt(RUN/'TRAIN_fit/final.pt');before=pt(RUN/'TRAIN_fit/rounds/00/before.pt');proposed=pt(RUN/'TRAIN_fit/rounds/00/proposed.pt');retained=pt(RUN/'TRAIN_fit/rounds/00/retained.pt')
assert all(torch.equal(initial['weights'][k],final['weights'][k]) and torch.equal(initial['weights'][k],retained['weights'][k]) for k in initial['weights'])
names=[k for k in initial['weights'] if k.startswith('detail.')]
weight_report={'detail_parameters':sum(initial['weights'][k].numel() for k in names),'proposed_detail_delta_l2':float(torch.sqrt(sum((proposed['weights'][k]-initial['weights'][k]).double().square().sum() for k in names))),'proposed_detail_delta_max':max(tmax(proposed['weights'][k],initial['weights'][k]) for k in names),'retained_delta_max':max(tmax(final['weights'][k],initial['weights'][k]) for k in initial['weights'])}
old=pt(ROOT/'artifacts/v7_1/sample_alignment_001/learning/aligned/model.pt');oldw=old['weights'];assert all(torch.equal(v,oldw[k.removeprefix('low.')]) for k,v in initial['weights'].items() if k.startswith('low.'))
all_fixed=[(r,s) for role in ['fit','check'] for r,seeds in zip(selection[role+'_pool'],selection[role+'_seed_pairs']) for s in seeds]
conds=torch.tensor(np.stack([codec.condition(r['condition'],bases[byid[r['sample_id']]]) for r,s in all_fixed]),dtype=torch.float32)
seeds=[s for r,s in all_fixed];ci=conds[:32];indices=np.array([byid[r['sample_id']] for r,s in all_fixed[:32]])
assert np.array_equal(indices,before['indices']) and seeds[:32]==before['seeds']
centers=np.stack([codec.center(r['condition'],bases[byid[r['sample_id']]]) for r,s in all_fixed[:32]])
targets=torch.tensor(codec.normalized(np.stack([codec.encode(Z[i],bases[i]) for i in indices]),centers),dtype=torch.float32)
model=TransferDenoiser(codec,warm=False);model.load_state_dict(initial['weights'])
def full_forward(c,ss):return torch.cat([rollout(model,c[j:j+1],[s]) for j,s in enumerate(ss)])
with torch.no_grad():
    initial_full=full_forward(conds,seeds)
    # Rebuild old 102-dimensional DDIM directly, keeping the old noise stream.
    old_full=[];a=schedule().double();times=np.linspace(999,0,16).round().astype(int)
    for j,s in enumerate(seeds):
        x=torch.randn((1,102),generator=torch.Generator().manual_seed(s)).double()
        for k,t in enumerate(times):
            pred=model.low(x.float(),torch.tensor([t]),conds[j:j+1]).double(); clean=a[t].sqrt()*x-(1-a[t]).sqrt()*pred
            noise=(x-a[t].sqrt()*clean)/(1-a[t]).sqrt();na=a[times[k+1]] if k<15 else x.new_tensor(1.)
            x=na.sqrt()*clean+(1-na).sqrt()*noise
        old_full.append(x)
    old_full=torch.cat(old_full);assert torch.equal(initial_full[:,:102],old_full)
    assert torch.equal(initial_full[:32],before['x0']) and torch.equal(initial_full[:32],initial['old_x'])
    model.load_state_dict(proposed['weights']);proposal_full=full_forward(conds,seeds)
    assert torch.equal(proposal_full[:,:102],old_full)
    model.load_state_dict(final['weights']);retained_full=full_forward(conds,seeds);assert torch.equal(retained_full,initial_full)
weight_report.update(low_function_outputs_checked=64,low_function_max_error=tmax(proposal_full[:,:102],old_full),proposal_output_detail_delta_l2=float(torch.linalg.vector_norm(proposal_full[:,102:]-initial_full[:,102:])),final_full_output_max_error=tmax(retained_full,initial_full))
R=[];JJ=[]
for j in range(32):
    with npz(RUN/f'TRAIN_fit/rounds/00/jacobians/{j:02d}.npz') as a:R.append(a['residual']);JJ.append(a['J'])
R=torch.tensor(np.stack(R),dtype=torch.float64);JJ=torch.tensor(np.stack(JJ),dtype=torch.float64);x0=before['x0'];protect=torch.tensor(before['protected']);orig=torch.tensor(initial['original_protected']);scale=torch.tensor(codec.latent_scale,dtype=torch.float64);basis=torch.tensor(codec.basis,dtype=torch.float64)
update_reports=[]
for step in ['001','064']:
    pre=pt(RUN/f'TRAIN_fit/rounds/00/updates/{step}/before.pt');post=pt(RUN/f'TRAIN_fit/rounds/00/updates/{step}/after.pt');model.load_state_dict(pre['weights'])
    full=full_forward(ci,seeds[:32]);q=(R+torch.einsum('nij,nj->ni',JJ,full-x0)).reshape(-1,4,3).square().sum(-1)
    world=q.mean()+torch.relu(q-1).square().mean()+100*torch.relu(q[protect]-.9**2).square().mean()+.01*(full-x0).square().mean()
    replay=(full[orig]-initial['old_x'][orig]).square().mean();phys=torch.einsum('tk,nkj->ntj',basis,((full-targets.double())*scale).reshape(-1,19,17))
    teacher=(phys.square().mean()+.25*torch.diff(phys,n=2,dim=1).square().mean())/scale.square().mean();loss=world+5*replay+.05*teacher
    grads=torch.autograd.grad(loss,model.trainable());norm=torch.linalg.vector_norm(torch.stack([torch.linalg.vector_norm(g) for g in grads]));clip=torch.clamp(1/(norm+1e-6),max=1.)
    graderr=max(tmax(g*clip,saved) for g,saved in zip(grads,pre['gradients']))
    group=pre['optimizer']['param_groups'][0];b1,b2=group['betas'];paramerr=momenterr=0.
    for n,pid,g in zip(names,group['params'],pre['gradients']):
        state=pre['optimizer']['state'].get(pid,{});m=state.get('exp_avg',torch.zeros_like(g)).clone();v=state.get('exp_avg_sq',torch.zeros_like(g)).clone();t=int(state.get('step',0))+1
        m.lerp_(g,1-b1);v.mul_(b2).addcmul_(g,g,value=1-b2)
        w=pre['weights'][n].clone();w.mul_(1-group['lr']*group['weight_decay']);den=v.sqrt()/(1-b2**t)**.5+group['eps'];w.addcdiv_(m,den,value=-group['lr']/(1-b1**t))
        paramerr=max(paramerr,tmax(w,post['weights'][n]));momenterr=max(momenterr,tmax(m,post['optimizer']['state'][pid]['exp_avg']),tmax(v,post['optimizer']['state'][pid]['exp_avg_sq']))
    assert graderr<1e-6 and paramerr<1e-8 and momenterr<1e-8
    assert all(torch.equal(pre['weights'][k],post['weights'][k]) for k in pre['weights'] if not k.startswith('detail.'))
    update_reports.append({'step':int(step),'loss':float(loss.detach()),'unclipped_gradient_norm':float(norm),'clipped_gradient_max_error':graderr,'manual_adamw_weight_max_error':paramerr,'manual_adamw_moment_max_error':momenterr,'frozen_low_unchanged':True})
emit('weights',weight_report);emit('saved_updates',update_reports)

ledger=js(RUN/'ledger.json');gate=js(RUN/'TRAIN_gate.json');status=js(RUN/'status.json');js(RUN/'independent_verification.json')
ledger_report={'attempts':ledger['attempts'],'coupled_calls':sum(v for k,v in ledger['calls'].items() if k.endswith('.coupled')),'physics_steps':sum(v for k,v in ledger['calls'].items() if k.endswith('.physics_steps')),'torque_rollouts':sum(v for k,v in ledger['calls'].items() if k.endswith('.torque_rollout')),'fit_J_calls':ledger['calls']['fit_full323_J.coupled'],'inflight':ledger['inflight'],'completed_units':len(ledger['completed']),'status':status['status'],'M1_status':gate['status'],'M1_failed_checks':[k for k,v in gate['fit']['checks'].items() if not v]}
assert ledger['attempts']=={'TRAIN_check_validation':64,'TRAIN_fit_validation':128,'fit_full323_J':32,'fit_only_world_proposal':1}
assert ledger_report['coupled_calls']==21312 and ledger_report['physics_steps']==ledger_report['torque_rollouts']==0
assert gate['fit']['actual_proposed_updates']==64 and len(gate['fit']['rounds'])==1 and not gate['fit']['rounds'][0]['accepted']
absent=['learning','candidate_fit','candidate_check_final','VAL','comparison','application'];ledger_report['later_stage_path_exists']={name:(RUN/name).exists() for name in absent}
auditpath=ROOT/'paper/PAPER_CLAIM_AUDIT.json';paper_audit=js(auditpath) if auditpath.exists() else 'unavailable'
result={'scope':'saved data only; no physics, no optimizer.step, no training, no existing audit invoked','source_binding':source_report,'pools':pool_report,'statistics':statistics,'saved_pose_metric_max_error':pose_max,'all_raw_equal_prepared':all_raw_equal,'jacobians':jac_report,'weights':weight_report,'saved_updates':update_reports,'ledger':ledger_report,'paper_claim_audit':paper_audit,'status':'PASS'}
out('independent_saved_check.json',result)
out('inputs.json',{'reviewed_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'input_count':len(INPUTS),'files':INPUTS})
emit('complete',{'status':'PASS','input_count':len(INPUTS),'ledger':ledger_report})
