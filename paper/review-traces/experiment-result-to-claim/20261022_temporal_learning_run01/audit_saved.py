"""Independent new-run evidence audit: saved arrays, neural forwards and URDF algebra only.
No simulator/geometry query/optimizer step/experiment/resume call. Historical runs
are used for frozen identity, labels and public initial conditions, never reexecuted.
"""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import ast, hashlib, json, math, time, zipfile, xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
import torch
from torch import nn
from scipy.interpolate import BSpline, PPoly
from scipy.spatial.transform import Rotation

ROOT=Path(__file__).resolve().parents[4]
TRACE=Path(__file__).resolve().parent
RUN=ROOT/'artifacts/v7_1/temporal_learning_001'
INPUTS={};CHECKS=[];RAW={'review_role':'fresh independent paper architect/reviewer','units':{},'training':{}}
torch.set_num_threads(1)

def path(p):
    p=Path(p);return p if p.is_absolute() else ROOT/p
def sha(p):
    p=path(p);h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    INPUTS[p.relative_to(ROOT).as_posix()]={'sha256':h.hexdigest(),'size_bytes':p.stat().st_size,'mtime_ns':p.stat().st_mtime_ns}
    return h.hexdigest()
def read(p):
    p=path(p);sha(p);return json.loads(p.read_text(encoding='utf-8-sig'))
def load(p):
    p=path(p);sha(p)
    with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}
def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),default=lambda x:x.tolist(),allow_nan=False).encode()).hexdigest()
def err(a,b):return float(np.max(abs(np.asarray(a)-np.asarray(b)),initial=0))
def close(a,b,tol=1e-10):return np.shape(a)==np.shape(b) and err(a,b)<=tol
def check(name,passed,detail=None):
    CHECKS.append({'name':name,'passed':bool(passed),'detail':detail})
    if not passed:print('FAIL',name,detail,flush=True)
def weight_hash(weights):
    h=hashlib.sha256()
    for n,p in weights.items():h.update(n.encode());h.update(p.detach().cpu().numpy().tobytes())
    return h.hexdigest()
def closure():
    todo=list((ROOT/'ff_prior_temporal_learning').glob('*.py'));seen=set()
    def add(m):
        parts=m.split('.')
        for i in range(1,len(parts)+1):
            p=ROOT.joinpath(*parts[:i])
            for f in [p.with_suffix('.py'),p/'__init__.py']:
                if f.is_file() and f not in seen:todo.append(f)
    while todo:
        p=todo.pop()
        if p in seen:continue
        seen.add(p);pkg=p.relative_to(ROOT).with_suffix('').parts[:-1]
        for node in ast.walk(ast.parse(p.read_text(encoding='utf-8-sig'))):
            if isinstance(node,ast.Import):
                for n in node.names:add(n.name)
            if isinstance(node,ast.ImportFrom):
                prefix=list(pkg[:len(pkg)-node.level+1]) if node.level else []
                m='.'.join(prefix+([node.module] if node.module else []));add(m)
                for n in node.names:
                    if n.name!='*':add(m+'.'+n.name)
    seen.update([ROOT/'ff_prior_temporal_learning/job.ps1',ROOT/'ff_prior_temporal_learning/docs/EXPERIMENT_PLAN.md'])
    return {p.relative_to(ROOT).as_posix() for p in seen}

# Reuse only reviewed algebra helper definitions, no historical result loading or main().
helper=ROOT/'paper/review-traces/experiment-result-to-claim/20261022_temporal_data_run01/audit_saved.py'
sha(helper)
tree=ast.parse(helper.read_text(encoding='utf-8-sig'))
names={'vec','quat_rotation','transforms','mesh_radii','polynomial_extrema'}
exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names],type_ignores=[]),str(helper),'exec'))
URDF=ROOT/'dual_arm_space_robot_2026/urdf/dual_arm_space_robot_2026.urdf';sha(URDF)
xml=ET.parse(URDF).getroot();JOINTS=xml.findall('joint');LINKS={l.attrib['name']:l for l in xml.findall('link')}
knots=np.r_[np.zeros(4),np.arange(.1,27-1e-9,.1),np.full(4,27.)]
ct=np.array([knots[i+1:i+4].mean() for i in range(273)])[2:]
B=np.zeros((67,17));B[:60,:10]=np.kron(np.eye(5),np.array([[0,1],[1,0],[1,0],[0,1],[0,1],[1,0],[1,0],[0,1],[0,1],[1,0],[1,0],[0,1]])/6);B[60:,10:]=np.eye(7);Bplus=np.linalg.pinv(B)
lower=np.r_[np.full(10,-1.),np.full(7,-np.pi)];upper=-lower
vmax=np.r_[np.full(10,.8),np.full(7,1.2)];amax=np.r_[np.full(10,2.5),np.full(7,4.)];torque=np.r_[np.full(60,4.),np.full(7,80.)]

def path_samples(w,t):
    tt=np.r_[0.,w['path_start_s'],w['path_start_s']+np.cumsum(w['segment_durations_s'])]
    pp=np.vstack([w['initial_position_w'],w['waypoint_points_m']])
    i=np.clip(np.searchsorted(tt,t,side='right')-1,0,len(tt)-2)
    duration=tt[i+1]-tt[i];s=np.clip((t-tt[i])/duration,0,1)
    a=s**3*(10-15*s+6*s*s);da=30*s*s*(1-s)**2/duration
    return pp[i]+a[:,None]*(pp[i+1]-pp[i]),da[:,None]*(pp[i+1]-pp[i])

def local_features(w,t,pt,rt,v,wlocal,grasp,grasp_r,pr0,rr0,approach):
    pc,vc=path_samples(w,t);rtf=rt@Rotation.from_rotvec(t[:,None]*wlocal).as_matrix()
    offset=rtf@grasp;point=pt+t[:,None]*v+offset
    point_v=v+np.cross(rtf@wlocal,offset);rr=rtf@grasp_r
    s=np.clip(t/4.5,0,1);a=s**3*(10-15*s+6*s*s);da=30*s*s*(1-s)**2/4.5
    bend=64*s**3*(1-s)**3;dbend=192*s*s*(1-s)**2*(1-2*s)/4.5
    pr=pr0+a[:,None]*(point-pr0)+bend[:,None]*approach
    vr=a[:,None]*point_v+da[:,None]*(point-pr0)+dbend[:,None]*approach
    rv=Rotation.from_matrix(rr@rr0.T).as_rotvec()
    rref=Rotation.from_rotvec(a[:,None]*rv).as_matrix()@rr0
    wr=a[:,None]*(rtf@wlocal)+da[:,None]*rv
    return np.concatenate([pc,np.broadcast_to(np.array(w['target_rotation_world']).reshape(1,9),(len(t),9)),vc,np.zeros((len(t),3)),pr,rref.reshape(-1,9),vr,wr],axis=1)

def public_from_identity(initial,ident,approach,objects):
    robot=ident['robot'];p,r=transforms(initial[1:82][None],ident)
    ordered=sorted(robot['joint_index_map'].values(),key=lambda x:x['actuator'])
    qi=[x['qpos'] for x in ordered];vi=[x['dof'] for x in ordered]
    q0=Bplus@initial[1:82][qi];v0=Bplus@initial[82:161][vi]
    pt,rt=p['target_satellite'][0],r['target_satellite'][0];pr,rr=p['end_effector_r'][0],r['end_effector_r'][0]
    grasp=np.asarray(ident.get('grasp_local',[0.,-.205,0.]));grasp_r=rt.T@rr;w=ident['waypoints'];velocity=initial[155:161]
    obs=np.zeros((5,12))
    for i,o in enumerate(objects):obs[i]=np.r_[1.,1.,o['position'],o['size'],o['quaternion_wxyz']]
    g=np.concatenate([initial,q0,v0,np.ravel(w['waypoint_points_m']),w['segment_durations_s'],np.ravel(w['target_rotation_world']),grasp,grasp_r.ravel(),approach,pr,rr.ravel(),pt,rt.ravel(),velocity,[w['path_start_s'],w['path_end_s'],27.],obs.ravel()])
    local=local_features(w,ct,pt,rt,velocity[:3],velocity[3:],grasp,grasp_r,pr,rr,approach)
    gscale=np.r_[np.ones(836+17),np.full(17,.1),np.full(21,.2),np.full(6,27.),np.ones(9),np.full(3,.2),np.ones(9),np.full(3,.1),np.full(3,2.),np.ones(9),np.full(3,2.),np.ones(9),np.full(6,.005),np.full(3,27.),np.tile(np.r_[1.,1.,np.full(3,2.),np.full(3,.1),np.ones(4)],5)]
    lscale=np.tile(np.r_[np.full(3,.2),np.ones(9),np.full(6,.05)],2)
    return {'g':g,'local':local,'q0':q0,'v0':v0,'gscale':gscale,'lscale':lscale}

def independent_sample(model,g,local,seeds,method):
    g=torch.as_tensor(g,dtype=torch.float32);local=torch.as_tensor(local,dtype=torch.float32);n=len(g)
    with torch.no_grad():
        if method=='deterministic_sequence':return model(torch.zeros((n,271,17)),torch.zeros(n),g,local).double().numpy(),[]
        x=torch.stack([torch.randn((271,17),generator=torch.Generator().manual_seed(s),dtype=torch.float64) for s in seeds])
        alphas=torch.cumprod(1-torch.linspace(1e-4,.02,1000),0).double();ts=torch.linspace(999,0,16).round().long();recs=[]
        for j,t in enumerate(ts):
            a=alphas[t];v=model(x.float(),t.repeat(n).float(),g,local).double()
            clean=a.sqrt()*x-(1-a).sqrt()*v;noise=(1-a).sqrt()*x+a.sqrt()*v
            residual=float((a.sqrt()*clean+(1-a).sqrt()*noise-x).abs().max())
            nxt=alphas[ts[j+1]] if j<15 else torch.tensor(1.,dtype=torch.float64)
            x=nxt.sqrt()*clean+(1-nxt).sqrt()*noise
            recs.append({'t':int(t),'decomposition_residual':residual})
        return x.numpy(),recs
def objective(pred,target):
    delta=pred-target;return float(np.mean(delta**2)+.1*np.mean(np.diff(delta,axis=1)**2))

def main():
    global P,SequenceNetwork
    run=read(RUN/'run.json');ident=run['identity'];P=ident['protocol'];summary=read(RUN/'summary.json');budget=read(RUN/'budget.json')
    check('producer_frozen_commit',run['producer_head']=='681965a0889fa1f4625e66d356a34b838f77c837')
    check('source_closure_exact',closure()==set(ident['source_hashes']))
    for p,h in ident['source_hashes'].items():check('source_current:'+p,sha(p)==h)
    sha(RUN/'executed_source.zip')
    with zipfile.ZipFile(RUN/'executed_source.zip') as z:
        check('source_zip_inventory_exact',set(z.namelist())==set(ident['source_hashes']) and len(z.namelist())==len(ident['source_hashes']))
        for p,h in ident['source_hashes'].items():check('source_zip:'+p,hashlib.sha256(z.read(p)).hexdigest()==h)
        src=z.read('ff_prior_temporal_learning/models.py').decode();nodes=ast.parse(src).body
        exec(compile(ast.Module(body=[n for n in nodes if isinstance(n,ast.ClassDef)],type_ignores=[]),'frozen_models_classes','exec'),globals())
    for p,h in ident['binding']['files'].items():check('input:'+p,sha(p)==h)
    check('binding_exact',read('ff_prior_temporal_learning/evidence/binding.json')==ident['binding'])
    check('selection_exact',read('ff_prior_temporal_learning/evidence/selection.json')==ident['selection'])
    software=read('ff_prior_temporal_learning/evidence/software_tests.json');check('software_test_source_exact',software['source_hashes']==ident['source_hashes'])
    RAW['identity']={'producer_head':run['producer_head'],'sources':len(ident['source_hashes']),'bound_inputs':len(ident['binding']['files'])}
    print('identity checked',flush=True)

    contract=read(RUN/'data_consumption_contract.json');rows=contract['source_rows'];qualified=read('artifacts/v7_1/temporal_data_001/qualified_view.json')
    check('qualified_rows_exact',rows==qualified['rows']==ident['selection']['data'])
    check('train_val_disjoint',len(rows)==12 and len(set(r['unit'] for r in rows))==12 and [r['split'] for r in rows]==['train']*8+['validation']*4)
    check('no_DEV_seeds',not {r['seed'] for r in rows}&{20261020,20261021})
    codec=read(RUN/'codec.json');encoded={s:load(RUN/(s+'_encoded.npz')) for s in ['train','validation']}
    profile=read('tracking_diffusion/config/local10mm_v1.json');features=[];labels=[];ids=[]
    for row in rows:
        check('row_grade:'+row['unit'],row['grade']=='EXECUTION_VERIFIED' and sha(row['reference'])==row['reference_sha256'])
        i=read(row['identity']);initial=load(row['initial'])['values'];ref=load(row['reference']);ids.append(i)
        f=public_from_identity(initial,i,np.asarray(profile['approach_offset']),i['scene']['objects']);features.append(f);labels.append(ref['coefficients'])
        check('reference_complete:'+row['unit'],ref['coefficients'].shape==(273,17) and np.array_equal(ref['knots'],knots) and close(f['v0'],np.zeros(17)))
    residual=np.array(labels)[:8,2:]-np.stack([f['q0'] for f in features[:8]])[:,None]
    g=np.stack([f['g'] for f in features]);local=np.stack([f['local'] for f in features])
    expected={'mean':residual.mean(0),'scale':np.maximum(np.sqrt(np.mean((residual-residual.mean(0))**2,axis=(0,1))),.02),
              'global_mean':g[:8].mean(0),'global_scale':np.maximum(g[:8].std(0),features[0]['gscale']),
              'local_mean':local[:8].mean(0),'local_scale':np.maximum(local[:8].std(axis=(0,1)),features[0]['lscale'])}
    for k,v in expected.items():check('TRAIN_only_codec:'+k,close(v,codec[k]),err(v,codec[k]))
    check('codec_declared_shape',codec['fit_count']==8 and codec['fit_split']=='TRAIN' and codec['free_shape']==[271,17])
    for split,start,stop in [('train',0,8),('validation',8,12)]:
        x=(np.array(labels)[start:stop,2:]-np.stack([f['q0'] for f in features[start:stop]])[:,None]-expected['mean'])/expected['scale']
        targets={'x':x,'global_condition':(g[start:stop]-expected['global_mean'])/expected['global_scale'],'local_condition':(local[start:stop]-expected['local_mean'])/expected['local_scale']}
        for k,v in targets.items():check('encoded_'+split+':'+k,close(v,encoded[split][k]),err(v,encoded[split][k]))
    RAW['data']={'train_units':contract['train'],'validation_units':contract['validation'],'condition_dim':g.shape[1],'local_shape':list(local.shape[1:]),'output_shape':[273,17],'free_outputs':4607,'TRAIN_mean_label_RMS_rad':float(np.sqrt(np.mean(expected['mean']**2))),'TRAIN_residual_scale_min_max':[float(expected['scale'].min()),float(expected['scale'].max())],'TRAIN_zero_normalized_output_objective':objective(np.zeros_like(encoded['train']['x']),encoded['train']['x'].astype(np.float32))}
    print('data and codec checked',flush=True)

    digests=[];consumption=np.zeros(8,int);rollcons=np.zeros(8,int)
    for step in range(1,3001):
        order=np.random.default_rng(20261022+step).permutation(8);rng=torch.Generator().manual_seed(20361022+step)
        t=torch.randint(0,1000,(8,),generator=rng);noise=torch.randn((8,271,17),generator=rng);h=hashlib.sha256()
        for v in [order,t.numpy(),noise.numpy()]:h.update(np.asarray(v).tobytes())
        digests.append(h.hexdigest());consumption[order]+=1
        if step%4==0:
            for j in range(2):rollcons[(2*(step//4)+j)%8]+=1
    models={};trained=read(RUN/'training.json')
    for method in P['methods']:
        directory=RUN/'learning'/method;payloads={}
        for name in ['initial','diagnostic_after','model']:
            sha(directory/(name+'.pt'));payloads[name]=torch.load(directory/(name+'.pt'),map_location='cpu',weights_only=False)
        initial=payloads['initial'];after=payloads['diagnostic_after'];final=payloads['model'];diagnostic=read(directory/'diagnostic.json');record=read(directory/'training.json')
        check(method+':optimizer_final',len(final['optimizer']['state'])>0 and set(int(x['step']) for x in final['optimizer']['state'].values())=={3000})
        check(method+':optimizer_diagnostic',len(after['optimizer']['state'])>0 and set(int(x['step']) for x in after['optimizer']['state'].values())=={32})
        for label,payload in [('final',final),('diagnostic',after)]:
            check(method+':optimizer_hyperparameters_'+label,all(pg['lr']==.0002 and pg['weight_decay']==.01 and pg['betas']==(.9,.999) for pg in payload['optimizer']['param_groups']))
            check(method+':finite_weights_optimizer_'+label,all(torch.isfinite(x).all().item() for x in payload['weights'].values()) and all(torch.isfinite(v).all().item() for s in payload['optimizer']['state'].values() for v in s.values() if torch.is_tensor(v)))
        check(method+':3000_digest_exact',final['batch_noise_digests']==digests)
        check(method+':3000_each_TRAIN',np.array_equal(final['consumed'],consumption) and record['data_consumption']==consumption.tolist())
        check(method+':60_history_chunks',len(record['history'])==60 and all(h['step']==50*(i+1) and h['batch_noise_sha256']==hashlib.sha256(''.join(digests[50*i:50*(i+1)]).encode()).hexdigest() and h['accumulated_consumption']==[50*(i+1)]*8 for i,h in enumerate(record['history'])))
        torch.manual_seed(20261022);fresh=SequenceNetwork(1014)
        check(method+':same_seed_initial',weight_hash(fresh.state_dict())==weight_hash(initial['weights']))
        check(method+':restored_formal_initial_hash',record['initial_weight_hash']==weight_hash(initial['weights']) and diagnostic['weights_restored'] and diagnostic['retained_updates']==0)
        probe={}
        for name,payload in payloads.items():
            model=SequenceNetwork(1014);model.load_state_dict(payload['weights']);model.eval()
            x,_=independent_sample(model,encoded['train']['global_condition'],encoded['train']['local_condition'],list(range(20264200,20264208)),method)
            probe[name]=objective(x,encoded['train']['x'].astype(np.float32))
            expected_loss=diagnostic['before' if name=='initial' else 'after']['loss'] if name!='model' else record['final_TRAIN_probe']['loss']
            check(method+':probe_reconstructed_'+name,abs(probe[name]-expected_loss)<1e-10 and np.isfinite(x).all(),[probe[name],expected_loss])
        check(method+':diagnostic_32_finite_decrease',len(diagnostic['history'])==32 and probe['diagnostic_after']<probe['initial'] and all(np.isfinite(h['loss']) and np.isfinite(h['gradient_norm']) for h in diagnostic['history']))
        check(method+':reload_weight_hash',weight_hash(model.state_dict())==record['weight_hash'] and record['weight_hash']!=record['initial_weight_hash'])
        check(method+':summary_training_exact',trained['methods'][method]==record and trained['diagnostics'][method]==diagnostic)
        check(method+':full_sampling_count',record['full_sampling_training_examples']==(1500 if method=='temporal_diffusion' else 24000) and record['full_sampling_training_updates']==(750 if method=='temporal_diffusion' else 3000))
        models[method]=model
        RAW['training'][method]={'optimizer_steps':3000,'temporary_optimizer_steps':32,'optimizer_parameters':len(final['optimizer']['state']),'parameters':sum(x.numel() for x in model.parameters()),'weight_hash':record['weight_hash'],'initial_weight_hash':record['initial_weight_hash'],'probe_objectives':probe,'consumption_each':final['consumed'].tolist(),'digest_count':len(final['batch_noise_digests']),'rollout_extra_consumption':rollcons.tolist() if method=='temporal_diffusion' else None,'reload_exact':True}
    print('checkpoint and neural training diagnostics checked',flush=True)

    radius=mesh_radii();results=read(RUN/'progress_results.json');check('summary_results_exact',results==summary['results']);native_total=0;replay_total=0
    devtasks={scene:read('artifacts/v7_1/dynamic_handoff_001/'+u+'/task.json') for scene,u in [('stationary','u00'),('drifting','u06')]}
    devrobots={scene:read('artifacts/v7_1/dynamic_handoff_001/'+u+'/identity.json')['robot'] for scene,u in [('stationary','u00'),('drifting','u06')]}
    for entry in results:
        unit=entry['unit'];d=RUN/unit;method=entry['method'];rpt=read(d/'result.json');check(unit+':result_exact',rpt==entry['result']);initial=load(d/'initial.npz')['values'];task=read(d/'task.json')
        if entry['split']=='VAL':
            idx=contract['validation'].index(entry['source_unit'])+8;i=ids[idx];f=features[idx]
            check(unit+':original_identity',task==i and np.array_equal(initial,load(rows[idx]['initial'])['values']))
        else:
            old=devtasks[entry['scene']];i={'robot':devrobots[entry['scene']],'waypoints':task['waypoints'],'grasp_local':old['grasp_local']}
            check(unit+':DEV_initial_task_reused',np.array_equal(initial,old['initial']) and task['waypoints']==old['waypoints'] and task['grasp']==old['grasp_local'] and task['target_velocity']==old['target_velocity'] and task['initial_id']==old['initial_id'])
            f=public_from_identity(initial,i,np.zeros(3),[])
            check(unit+':DEV_condition_algebra_consistent',close(f['g'],(load(RUN/('d000' if entry['scene']=='stationary' else 'd100')/'proposal.npz')['global_condition']*np.asarray(codec['global_scale'])+codec['global_mean'])))
        if not rpt['new_execution']:
            source=ROOT/rpt['reused_from'];reuse=read(d/'reuse.json');previous=read(source/'result.json')
            expected=dict(previous,new_execution=False,reused_from=rpt['reused_from'])
            check(unit+':deterministic_reuse_result',method=='deterministic_sequence' and rpt==expected and reuse['not_independent_replication'] and reuse['identical_condition'])
            check(unit+':reuse_native_hash',sha(source/'native_trace.npz')==reuse['identical_control_hash']==rpt['native_trace_sha256'])
            check(unit+':reuse_condition_and_initial',task==read(source/'task.json') and np.array_equal(initial,load(source/'initial.npz')['values']))
            check(unit+':no_duplicate_physics',not (d/'native_trace.npz').exists() and not (d/'replay_trace.npz').exists())
            RAW['units'][unit]={'new_execution':False,'source':rpt['reused_from'],'status':rpt['status'],'independent_replication':False};continue
        proposal=load(d/'proposal.npz');prepared=load(d/'prepared.npz');prep=read(d/'prepare.json');sampling=read(d/'sampling.json')
        eg=(f['g']-np.asarray(codec['global_mean']))/codec['global_scale'];el=(f['local']-np.asarray(codec['local_mean']))/codec['local_scale']
        check(unit+':public_condition_algebra',close(eg,proposal['global_condition']) and close(el,proposal['local_condition']) and np.array_equal(initial,proposal['initial']),[err(eg,proposal['global_condition']),err(el,proposal['local_condition'])])
        x,recs=independent_sample(models[method],proposal['global_condition'][None],proposal['local_condition'][None],[sampling['seed']],method)
        check(unit+':complete_sampler_output',np.array_equal(x[0],proposal['normalized_output']) and recs==sampling['steps'] and len(recs)==(16 if method=='temporal_diffusion' else 0))
        check(unit+':sampler_schedule_finite',np.isfinite(x).all() and all(s['decomposition_residual']<=1e-10 for s in recs) and not sampling['world_guidance_in_sampler'])
        raw=np.vstack([f['q0'],f['q0']+f['v0']*.1/3,x[0]*codec['scale']+codec['mean']+f['q0']]);check(unit+':decode_raw',close(raw,proposal['raw_coefficients']))
        c=proposal['raw_coefficients'].copy();c[:2]=f['q0'];delta=c-f['q0'];ratios=[1.]
        for j in range(17):
            if (delta[:,j]>0).any():ratios.append(np.min((upper[j]-f['q0'][j])/delta[delta[:,j]>0,j]))
            if (delta[:,j]<0).any():ratios.append(np.min((lower[j]-f['q0'][j])/delta[delta[:,j]<0,j]))
        d1=np.diff(c,axis=0)*3/(knots[4:276]-knots[1:273])[:,None]
        d2=np.diff(d1,axis=0)*2/(knots[4:275]-knots[2:273])[:,None]
        for deriv,limit in [(d1,vmax),(d2,amax)]:
            mask=abs(deriv)>0;ratios.append(np.min(np.broadcast_to(limit,deriv.shape)[mask]/abs(deriv[mask])))
        alpha=max(0,min(ratios));alpha=alpha*(1-1e-7) if alpha<1 else alpha
        expected_c=f['q0']+alpha*delta
        check(unit+':prepare_scale_exact',abs(alpha-prep['scale'])<1e-10 and close(expected_c,prepared['coefficients']) and np.array_equal(prepared['knots'],knots),[alpha,prep['scale']])
        spline=BSpline(knots,prepared['coefficients'],3,extrapolate=False);bounds=polynomial_extrema(spline)
        check(unit+':prepared_motion_gates',np.all(bounds[0,0]>=lower) and np.all(bounds[0,1]<=upper) and np.all(abs(bounds[1])<=vmax) and np.all(abs(bounds[2])<=amax) and close(spline(0),f['q0']) and close(spline(0,1),f['v0']) and all(prep['gates'].values()))
        h=load(d/'native_trace.npz');r=load(d/'replay_trace.npz');geom=load(d/'geometry.npz');pairs=read(d/'geometry_pairs.json');n=len(h['ctrl']);native_total+=n;replay_total+=len(r['ctrl'])
        check(unit+':native_hash',sha(d/'native_trace.npz')==rpt['native_trace_sha256'])
        check(unit+':full_saved_shapes',h['integration'].shape==(n+1,836) and h['qpos'].shape==(n+1,81) and h['qvel'].shape==(n+1,79) and h['ctrl'].shape==(n,67) and rpt['steps']==n)
        check(unit+':full_836_replay_exact',all(np.array_equal(h[k],r[k]) for k in ['integration','qpos','qvel','ctrl']) and rpt['replay_full_integration_error']==0)
        check(unit+':initial_836_exact',np.array_equal(initial,h['integration'][0]))
        check(unit+':time_source',close(h['time'],np.arange(n+1)*.002,1e-8) and np.array_equal(h['time'],h['integration'][:,0]))
        check(unit+':integration_layout',np.array_equal(h['qpos'],h['integration'][:,1:82]) and np.array_equal(h['qvel'],h['integration'][:,82:161]) and np.array_equal(h['ctrl'],h['integration'][1:,240:307]))
        check(unit+':no_applied_external_wrench',np.count_nonzero(h['integration'][:,307:])==0)
        ordered=sorted(i['robot']['joint_index_map'].values(),key=lambda v:v['actuator']);qi=[v['qpos'] for v in ordered];vi=[v['dof'] for v in ordered]
        lowq=h['qpos'][:,qi];lowv=h['qvel'][:,vi];q=lowq@Bplus.T;v=lowv@Bplus.T;a=np.diff(v,axis=0)/.002;la=np.diff(lowv,axis=0)/.002
        check(unit+':actual_q17',close(q,h['q17']))
        joint_ok=np.all(q>=lower) and np.all(q<=upper) and np.all(h['qref']>=lower) and np.all(h['qref']<=upper)
        for j in JOINTS:
            if j.attrib['type']=='revolute':
                limit=j.find('limit');addr=i['robot']['joint_index_map'][j.attrib['name']]['qpos'];joint_ok &= np.all(h['qpos'][:,addr]>=float(limit.attrib['lower'])) and np.all(h['qpos'][:,addr]<=float(limit.attrib['upper']))
        ratios=[float(np.max(abs(v)/vmax)),float(np.max(abs(a)/amax)),float(np.max(abs(lowv)/(abs(B)@vmax))),float(np.max(abs(la)/(abs(B)@amax))),float(np.max(abs(h['reference_qd'])/vmax)),float(np.max(abs(h['reference_qdd'])/amax))]
        mp=np.array([np.linalg.norm(h['momentum'][:,s],axis=1).max() for s in [slice(0,3),slice(3,6)]]);me=np.array([np.linalg.norm(h['exchange'][:,s],axis=1).max() for s in [slice(0,3),slice(3,6)]])
        pos,rot=transforms(h['qpos'],i);ee=np.broadcast_to(np.eye(4),(n+1,2,4,4)).copy();ee[:,0,:3,3]=pos['link_30']+rot['link_30']@np.array([.0475,0,0]);ee[:,0,:3,:3]=rot['link_30'];ee[:,1,:3,3]=pos['end_effector_r'];ee[:,1,:3,:3]=rot['end_effector_r']
        check(unit+':URDF_ee_all_saved_states',close(ee,h['ee']),err(ee,h['ee']))
        w=i['waypoints'];pc,_=path_samples(w,h['time']);s=np.clip(h['time']/4.5,0,1);blend=s**3*(10-15*s+6*s*s);bend=64*s**3*(1-s)**3
        grasp=np.asarray(i.get('grasp_local',[0.,-.205,0.]));home=ee[0,1,:3,3];home_r=ee[0,1,:3,:3];grasp_r=rot['target_satellite'][0].T@home_r
        point=pos['target_satellite']+rot['target_satellite']@grasp;approach=np.array(profile['approach_offset']) if entry['split']=='VAL' else np.zeros(3)
        desired=np.broadcast_to(np.eye(4),(n+1,2,4,4)).copy();desired[:,0,:3,3]=pc;desired[:,0,:3,:3]=w['target_rotation_world'];desired[:,1,:3,3]=home+blend[:,None]*(point-home)+bend[:,None]*approach
        rv=Rotation.from_matrix(rot['target_satellite']@grasp_r@home_r.T).as_rotvec();desired[:,1,:3,:3]=Rotation.from_rotvec(blend[:,None]*rv).as_matrix()@home_r
        check(unit+':desired_world_from_current_target',close(desired,h['desired']),err(desired,h['desired']))
        position=np.linalg.norm(h['ee'][:,:,:3,3]-h['desired'][:,:,:3,3],axis=2);rotation=Rotation.from_matrix((h['desired'][:,:,:3,:3]@h['ee'][:,:,:3,:3].transpose(0,1,3,2)).reshape(-1,3,3)).magnitude().reshape(-1,2)
        for label,value in [('position_mm',position*1000),('rotation_deg',np.rad2deg(rotation))]:
            stats={'mean':value.mean(0),'p95':np.percentile(value,95,axis=0),'maximum':value.max(0),'final':value[-1]}
            check(unit+':reported_'+label,all(close(v,rpt[label][k],1e-8) for k,v in stats.items()))
        check(unit+':geometry_inventory',len(geom['qpos'])==2*n+1 and len(pairs)==2*n+1 and len(geom['minima'])==2*n+1)
        # Scans are actual state followed by its midpoint; no new clearance query.
        check(unit+':geometry_actual_states',np.array_equal(geom['qpos'][0],h['qpos'][0]) and np.array_equal(geom['qpos'][1::2],h['qpos'][1:]))
        reduced=np.r_[geom['minima'][0],np.minimum(geom['minima'][1::2],geom['minima'][2::2])]
        check(unit+':all_recorded_clearances',np.array_equal(reduced,h['clearance']) and abs(geom['minima'].min()*1000-rpt['minimum_clearance_mm'])<1e-10)
        midpos,midrot=transforms(geom['qpos'][2::2],i);motion=0.
        for name,rad in radius.items():
            for pp,rr in [(pos[name][:-1],rot[name][:-1]),(pos[name][1:],rot[name][1:])]:
                angle=Rotation.from_matrix(midrot[name]@rr.transpose(0,2,1)).magnitude();motion=max(motion,float(np.max(np.linalg.norm(midpos[name]-pp,axis=1)+rad*angle)))
        hold=h['time']>=25.5-1e-9;complete=n==13500 and rpt['stop_reason'] is None
        gates={'complete_native_time':complete,'independent_replay':all(np.array_equal(h[k],r[k]) for k in ['qpos','qvel']),
               'whole_body_clearance':bool(np.min(geom['minima'])>=0),'geometry_step':motion<=.002+1e-12,
               'dynamic_position':bool(np.all(position<=.005)),'dynamic_rotation':bool(np.all(rotation<=np.pi/180)),
               'relative_hold':bool(complete and hold.any() and np.all(position[hold]<=.005) and np.all(rotation[hold]<=np.pi/180)),
               'joint_limits':bool(joint_ok),'derivative_limits':max(ratios)<=1,'raw_torque':bool(np.all(abs(h['raw_torque'])<=torque+1e-12)),
               'tracking':err(q,h['qref'])<=.02,'manifold':err(lowq,q@B.T)<=.02,'momentum':bool(np.all(mp<=1e-5+.002*me)),
               'finite':all(np.isfinite(x).all() for x in h.values()),'no_external_wrench':np.count_nonzero(h['integration'][:,307:])==0,'no_hidden_reset':True}
        check(unit+':all_physical_gates_recomputed',gates==rpt['gates'],{'recomputed':gates,'saved':rpt['gates']} if gates!=rpt['gates'] else None)
        check(unit+':raw_controls_exact',np.array_equal(h['raw_torque'],h['ctrl']))
        check(unit+':report_status',rpt['status']==('EXECUTION_VERIFIED' if all(gates.values()) else 'EXECUTION_FAILED'))
        feedback_stats=None
        if entry['split']=='DEV':
            fb=read(d/'feedback.json');fil=read(d/'filter.json');t=np.array([v['time'] for v in fb]);preferred=np.array([v['sequence_preferred'] for v in fb]);world=np.array([v['world_correction'] for v in fb]);filtered=np.array([v['filtered'] for v in fb]);requested=np.array([v['requested'] for v in fb])
            check(unit+':feedback_denominator',len(fb)==1350 and len(fil)==1350 and close(t,h['time'][:-1:10]))
            check(unit+':preferred_reconstruction',close(preferred,spline(t,1)+.5*(spline(t)-h['q17'][:-1:10])))
            check(unit+':world_sum',close(requested,preferred+world) and close(filtered,np.array([v['filtered'] for v in fil])))
            old=np.vstack([np.zeros((1,17)),filtered[:-1]]);qdot=(old[:,None,:]+np.arange(1,11)[None,:,None]/10*(filtered-old)[:,None,:]).reshape(-1,17)
            check(unit+':feedback_executed_commands',close(qdot,h['reference_qd']) and close(np.repeat((filtered-old)/.02,10,axis=0),h['reference_qdd']) and close(h['qref'][1:]-h['qref'][:-1],.002*qdot))
            def normstats(a):
                v=np.linalg.norm(a,axis=1);return {'mean':float(v.mean()),'p95':float(np.percentile(v,95)),'maximum':float(v.max())}
            feedback_stats={'sequence_preferred':normstats(preferred),'world_correction':normstats(world),'constraint_correction':normstats(filtered-requested),'filtered':normstats(filtered),'world_to_preferred_l2_ratio':float(np.linalg.norm(world)/np.linalg.norm(preferred))}
        else:
            times=np.arange(1,n+1)*.002
            check(unit+':VAL_unguided_spline_commands',close(h['qref'][1:],spline(times)) and close(h['reference_qd'],spline(times,1)) and close(h['reference_qdd'],spline(times,2)))
        transitions=read(d/'transitions.json')
        for tr in transitions:
            snap=load(d/f"transition_{int(tr['time']*10)}.npz");idx=tr['index'];check(unit+':transition_'+str(tr['time']),all(np.array_equal(v,h[k][idx-1:idx+2]) for k,v in snap.items()) and tr['direct_native_step'] and not tr['state_reset'])
        peak=np.unravel_index(np.argmax(abs(a)/amax),a.shape)
        acceleration_evidence={'actual_planner_peak_time_s':float(h['time'][peak[0]+1]),'actual_planner_peak_coordinate':int(peak[1]),'actual_planner_peak_rad_s2':float(a[peak]),'actual_planner_limit_rad_s2':float(amax[peak[1]]),'reference_acceleration_at_actual_peak':float(h['reference_qdd'][peak]),'actual_planner_exceedance_samples':int(np.count_nonzero(np.any(abs(a)>amax,axis=1))),'actual_native_exceedance_samples':int(np.count_nonzero(np.any(abs(la)>abs(B)@amax,axis=1))),'raw_coefficient_velocity_hull_ratio':float(np.max(abs(d1)/vmax)),'raw_coefficient_acceleration_hull_ratio':float(np.max(abs(d2)/amax)),'prepared_exact_polynomial_acceleration_ratio':float(np.max(abs(bounds[2])/amax)),'raw_second_difference_rms_rad':float(np.sqrt(np.mean(np.diff(proposal['raw_coefficients'],n=2,axis=0)**2)))}
        RAW['units'][unit]={'new_execution':True,'split':entry['split'],'method':method,'status':rpt['status'],'steps':n,'stop_reason':rpt['stop_reason'],'failed_gates':[k for k,v in gates.items() if not v],'full_state_shape':list(h['integration'].shape),'last_time':float(h['time'][-1]),'hold_states':int(hold.sum()),'prepare_scale':prep['scale'],'prepare_maximum_change_rad':prep['maximum_change_rad'],'sampler_max_abs_reconstruction_error':err(x[0],proposal['normalized_output']),'full_replay_max_error':err(h['integration'],r['integration']),'position_max_mm':(position.max(0)*1000).tolist(),'rotation_max_deg':np.rad2deg(rotation.max(0)).tolist(),'minimum_clearance_mm':float(geom['minima'].min()*1000),'derivative_ratios':ratios,'raw_torque_ratio':float(np.max(abs(h['raw_torque'])/torque)),'geometry_halfstep_upper_bound_m':motion,'world_feedback':feedback_stats,'acceleration_evidence':acceleration_evidence}
        print(unit,'steps',n,'status',rpt['status'],'prepare',prep['scale'],flush=True)
        del h,r,geom,pos,rot,midpos,midrot
    check('20_logical_16_unique',len(results)==20 and sum(r['result']['new_execution'] for r in results)==16)
    check('DEV_12_logical_8_unique',sum(r['split']=='DEV' for r in results)==12 and sum(r['split']=='DEV' and r['result']['new_execution'] for r in results)==8)
    check('8_pass_8_fail_unique',sum(r['result']['new_execution'] and r['result']['status']=='EXECUTION_VERIFIED' for r in results)==8 and sum(r['result']['new_execution'] and r['result']['status']=='EXECUTION_FAILED' for r in results)==8)
    check('counts_exact',summary['counts']=={s:{m:{'logical':sum(r['split']==s and r['method']==m for r in results),'verified':sum(r['split']==s and r['method']==m and r['result']['status']=='EXECUTION_VERIFIED' for r in results),'unique_executions':sum(r['split']==s and r['method']==m and r['result']['new_execution'] for r in results)} for m in P['methods']} for s in ['VAL','DEV']})
    check('budget_exact_native_replay',native_total==budget['native_steps_reserved'] and replay_total==budget['replay_steps_reserved'])
    bytecount=sum(p.stat().st_size for p in RUN.rglob('*') if p.is_file());status=read(RUN/'status.json')
    check('budget_limits',native_total+replay_total<=432000 and bytecount<=16*1024**3 and budget['elapsed_s']<=18000 and trained['elapsed_s']<=10800 and budget['inflight'] is None)
    check('total_P3_budget',ident['selection']['P3_previous_consumption']['elapsed_s']+status['elapsed_s']<=21600 and ident['selection']['P3_previous_consumption']['output_bytes']+bytecount<=20*1024**3)
    check('terminal_status',status['status']==summary['status']=='FINISHED_WITH_BLOCKED_ITEMS' and status['stage']=='complete' and not (RUN/'failure.json').exists())
    sha(RUN/'events.jsonl');events=[json.loads(x) for x in (RUN/'events.jsonl').read_text().splitlines()]
    check('event_denominators',len([e for e in events if e['stage']=='VAL_initialize'])==8 and len([e for e in events if e['stage']=='DEV_initialize'])==12 and len([e for e in events if e['stage']=='formal_training'])==120)
    RAW['budget']={'native_steps_actual':native_total,'replay_steps_actual':replay_total,'combined_actual':native_total+replay_total,'saved_budget':budget,'actual_output_bytes':bytecount,'monitor_wall_s':status['elapsed_s'],'training_wall_s':trained['elapsed_s'],'time_source':'native time and complete INTEGRATION column 0; wall values are distinct source timers, not realtime certification'}
    RAW['counts']=summary['counts'];audit=ROOT/'paper/PAPER_CLAIM_AUDIT.json';RAW['paper_claim_audit']=read(audit) if audit.exists() else 'unavailable'
    RAW['prohibited_actions']={'simulator_steps':0,'geometry_queries':0,'optimizer_step':0,'experiment_start':0,'resume':0,'historical_physical_reaudit':0}

if __name__=='__main__':
    start=time.time()
    try:main()
    finally:
        RAW.update(checks_total=len(CHECKS),checks_failed=sum(not c['passed'] for c in CHECKS),failed_checks=[c for c in CHECKS if not c['passed']],audit_elapsed_s=time.time()-start)
        (TRACE/'raw_audit.json').write_text(json.dumps(RAW,ensure_ascii=False,indent=2),encoding='utf-8')
        (TRACE/'inputs.json').write_text(json.dumps({'inputs':INPUTS,'generated_at_epoch':time.time()},ensure_ascii=False,indent=2),encoding='utf-8')
        (TRACE/'checks.json').write_text(json.dumps(CHECKS,ensure_ascii=False,indent=2),encoding='utf-8')
    print('CHECKS',len(CHECKS),'FAILED',sum(not c['passed'] for c in CHECKS),flush=True)
