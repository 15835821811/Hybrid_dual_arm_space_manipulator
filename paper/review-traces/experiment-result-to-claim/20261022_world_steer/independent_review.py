"""Fresh offline reviewer: reads saved world_steer_001 evidence only.

No project modules are imported; no MuJoCo, optimizer, training or geometry
calls occur. Numerical work is array algebra, SO(3), and 15 neural replays.
The formula constants below were read from the archived production sources.
"""
from pathlib import Path
from collections import Counter
from datetime import datetime, timezone
import io, json, hashlib, zipfile, gzip, sys, time
import numpy as np
from scipy.spatial.transform import Rotation

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
RUN = ROOT / 'artifacts/v7_1/world_steer_001'
inputs = {}; checks = Counter(); failures = []; maxima = {}; totals = Counter()

def plain(x):
    if isinstance(x,np.ndarray): return x.tolist()
    if isinstance(x,np.generic): return x.item()
    raise TypeError(type(x).__name__)

def digest(x):
    return hashlib.sha256(json.dumps(x,default=plain,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()

def read(p):
    p=Path(p); b=p.read_bytes()
    if p not in inputs:
        inputs[p]={'path':str(p),'sha256':hashlib.sha256(b).hexdigest(),'bytes':len(b),'mtime_utc':datetime.fromtimestamp(p.stat().st_mtime,timezone.utc).isoformat()}
    return b

def js(p): return json.loads(read(p))
def npz(p):
    with np.load(io.BytesIO(read(p)),allow_pickle=False) as z: return {k:z[k] for k in z.files}
def check(kind,ok,where=''):
    checks[kind]+=1
    if not bool(ok): failures.append({'check':kind,'where':str(where)})
def close(kind,a,b,tol=1e-10,where=''):
    a,b=np.asarray(a),np.asarray(b)
    err=float(np.max(abs(a-b),initial=0)) if a.shape==b.shape else float('inf')
    maxima[kind]=max(maxima.get(kind,0),err);check(kind,err<=tol,where)
def same(kind,a,b,where=''): check(kind,np.array_equal(a,b),where)
def pv(ee):
    ee=np.asarray(ee)
    return np.r_[((ee[:,:3,3]-GOALS[:,:3,3])/.005).ravel(),(Rotation.from_matrix(GOALS[:,:3,:3].transpose(0,2,1)@ee[:,:3,:3]).as_rotvec()/np.deg2rad(1)).ravel()]
def groups(e): return np.linalg.norm(np.asarray(e).reshape(4,3),axis=1)
def qual(ee): return float(max(groups(pv(ee))))
def step(j,e,cap):
    d=-np.linalg.solve(j.T@j+.001*np.eye(17),j.T@e)
    return d*min(1.,cap/max(float(abs(d).max()),1e-12))
def minimum_duration(delta):
    return max(.35,1.1*max(float(np.max(1.875*abs(delta)/VELOCITY)),float(np.max(np.sqrt((10/np.sqrt(3))*abs(delta)/ACCELERATION)))))
def state_parts(kind,state,qpos=None,qvel=None,t=None,where=''):
    check(kind+'_shape',state.shape[-1]==836,where)
    if qpos is not None:close(kind+'_qpos',state[...,1:82],qpos,where=where)
    if qvel is not None:close(kind+'_qvel',state[...,82:161],qvel,where=where)
    if t is not None:close(kind+'_time',state[...,0],t,where=where)

started=time.time()
run=js(RUN/'run.json'); ident=run['identity']; protocol=ident['protocol']; selection=ident['selection']
entry=selection['entries'][0]; task=entry['task']; GOALS=np.asarray(task['condition']['goals'])
physical=js(RUN/'physical_contract.json'); ledger=js(RUN/'ledger.json'); budget=js(RUN/'budget.json')
summary=js(RUN/'summary.json'); status=js(RUN/'status.json'); original_audit=js(RUN/'independent_verification.json')
origin=js(RUN/'scenes/capture_5_obstacles/initial_audit.json'); limits=js(RUN/'scenes/capture_5_obstacles/limits.json')
LOWER=np.asarray(limits['lower']); UPPER=np.asarray(limits['upper'])
VELOCITY=np.r_[np.full(10,.8),np.full(7,1.2)]; ACCELERATION=np.r_[np.full(10,2.5),np.full(7,4.)]
check('task_digest',digest({k:v for k,v in task.items() if k!='task_hash'})==task['task_hash'])
check('task_original_hash',task['task_hash']=='e66384242aaacf174ab1d7e642f0ba84b55ea580ab1b62951988409c75115496')
check('public_scope',entry['split']=='APPLICATION_DEV' and task['condition']['duration']==27 and len(entry['scene']['objects'])==5)
old_selection=js(ROOT/'ff_prior_static_fullchain_v2/evidence/selection.json')
check('exact_public_entry_copy',[e for e in old_selection['entries'] if e['sample_id']==entry['sample_id']]==[entry])
archive=zipfile.ZipFile(io.BytesIO(read(RUN/'executed_source.zip')))
check('source_archive_members',set(archive.namelist())==set(ident['source_hashes']))
for name,h in ident['source_hashes'].items():
    check('archived_source_hash',hashlib.sha256(archive.read(name)).hexdigest()==h,name)
    check('current_frozen_source_hash',hashlib.sha256(read(ROOT/name)).hexdigest()==h,name)
for name,h in ident['binding']['files'].items():check('bound_input_hash',hashlib.sha256(read(ROOT/name)).hexdigest()==h,name)
check('producer_commit',run['producer_head']=='099f2a93a33d4f79dce1406c5087e467a7cb75dd')
check('ledger_identity',ledger['identity']['identity_hash']==digest(ident))
check('raw_audit_claims',original_audit['status']=='PASS' and all(original_audit['checks'].values()))

# Recover the exact algebraic 17-to-67 map from the frozen angle convention
# through a source module containing only constants and NumPy functions.
namespace={'__name__':'review_angle_convention'}
exec(compile(archive.read('model_test/angle_convention.py'),'archived_angle_convention','exec'),namespace)
B=np.zeros((67,17));B[:60,:10]=namespace['continuum_mujoco_map_60_to_10']();B[60:,10:]=np.eye(7)
names=physical['identity']['robot']['low_level_joint_names']; index=physical['identity']['joint_index_map']
QIDS=np.array([index[n]['qpos'] for n in names]); VIDS=np.array([index[n]['dof'] for n in names])

m=npz(RUN/'m0/propagation_check.npz'); mr=js(RUN/'m0/result.json'); ROOT_STATE=m['root_state']
close('m0_whole_segment_qpos',m['joined_qpos'],m['whole_qpos'])
close('m0_whole_segment_qvel',m['joined_velocity'],m['whole_velocity'])
same('m0_join_rebuild',m['joined_qpos'],np.concatenate([m['first_qpos'],m['second_qpos'][1:]]))
same('m0_parent_qpos',m['first_qpos'][-1],m['second_qpos'][0])
close('m0_dual_world_poses',m['whole_ee'][256:],m['second_ee'])
close('m0_uniform_absolute_time',m['joined_time'],np.linspace(0,2,513),1e-12)
state_parts('m0_root',ROOT_STATE,m['first_qpos'][0],np.zeros(79),0)
state_parts('m0_first',m['first_state'],m['first_qpos'][-1],np.zeros(79),1)
state_parts('m0_second',m['second_state'],m['second_qpos'][-1],np.zeros(79),2)
base_motion=float(np.max(abs(m['first_qpos'][-1,:7]-m['first_qpos'][0,:7])))
check('m0_base_moves',base_motion>1e-8)
for label,s0,s1 in [('first',ROOT_STATE,m['first_state']),('second',m['first_state'],m['second_state'])]:
    report=mr[label]
    check('m0_saved_numerical_bounds',report['solve_residual']<1e-10 and report['momentum_residual']<1e-10 and report['quaternion_error']<1e-12,label)
    check('m0_state_hash',digest(s0)==report['parent_state_hash'],label)
    same('m0_unprescribed_state_fields',s0[161:],s1[161:],label)
check('m0_record_eight_passes',len(mr['checks'])==8 and all(mr['checks'].values()))
check('exact_initial_state_id',digest({'values':ROOT_STATE,'model_hash':task['contract']['robot_hash'],'state_mask':physical['initial_state_mask']})==task['contract']['initial_state_id'])
check('exact_initial_origin_hash',digest(ROOT_STATE)==origin['origin']['complete_state_hash'])

# Neural reconstruction is implemented independently with the bound weights;
# detail output is analytically zero because the frozen detail final layer is zero.
import torch
from torch import nn
torch.set_num_threads(2)
weights=torch.load(io.BytesIO(read(ROOT/'artifacts/v7_1/sample_alignment_001/learning/aligned/model.pt')),map_location='cpu',weights_only=False)
net=nn.Sequential(nn.Linear(199,256),nn.SiLU(),nn.Linear(256,256),nn.SiLU(),nn.Linear(256,256),nn.SiLU(),nn.Linear(256,102)).eval()
net.load_state_dict({k.removeprefix('net.'):v for k,v in weights['weights'].items()})
codec=js(ROOT/'artifacts/v7_1/control_transfer_bounded_001/codec.json')
alpha=torch.cumprod(1-torch.linspace(1e-4,.02,1000),dim=0).double()
times=np.linspace(999,0,16).round().astype(int)
freq=torch.exp(torch.arange(16)*(-np.log(10000)/15))
def neural(base,seed):
    cond=task['condition'];g=np.asarray(cond['goals'])
    features=np.r_[cond['initial_features'],g[:,:3,3].ravel(),g[:,:3,:2].transpose(0,2,1).ravel(),cond['duration'],np.asarray(cond['pose_mask']).ravel(),base[-1]-cond['initial_features']]
    c=(features-codec['feature_mean'])/codec['feature_scale']
    ct=torch.tensor(c,dtype=torch.float32)[None]
    x=torch.randn((1,102),generator=torch.Generator().manual_seed(seed)).double()
    with torch.no_grad():
        for i,t in enumerate(times):
            phase=torch.full((1,),int(t))[:,None]*freq[None]
            pred=net(torch.cat([x.float(),ct,phase.sin(),phase.cos()],1)).double()
            a=alpha[t];clean=a.sqrt()*x-(1-a).sqrt()*pred
            eps=(x-a.sqrt()*clean)/(1-a).sqrt();na=alpha[times[i+1]] if i<15 else x.new_tensor(1.)
            x=na.sqrt()*clean+(1-na).sqrt()*eps
    latent=np.r_[x[0].numpy(),np.zeros(221)]
    y=latent*np.asarray(codec['latent_scale'])+codec['latent_mean']+np.r_[1.,c]@np.asarray(codec['ridge'])
    return base+np.asarray(codec['basis'])@y.reshape(19,17)

candidate_rows=[];mode_counts=Counter();reasons=Counter();edge_reasons=Counter();grid_counts=Counter();collision_pairs=Counter();all_world_children=[]
reserved_total=1024;queries_total=3;actual_saved_intervals=1024;unit_queries={};unit_reservations={}
seed_phase=np.r_[np.linspace(0,1,20)[1:-1],1.];seed_basis=np.eye(17)[:,None,:]*(seed_phase**2*(3-2*seed_phase))[None,:,None]
for unit in selection['units']:
    ud=RUN/'units'/unit['id'];ur=js(ud/'result.json');setup=js(ud/'public_setup.json');base=np.asarray(setup['base'])
    selection_record=js(ud/'execution_selection.json');repair_selection=js(ud/'repair_selection.json')
    check('unit_candidates_five',len(ur['candidates'])==5 and ur['candidate_denominator']==5,ud)
    check('unit_execution_none',ur['execution_level']=='NOT_EXECUTED' and ur['execution_denominator']==1 and selection_record['candidate_id'] is None,ud)
    check('unit_repair_none',repair_selection['ids']==[],ud)
    check('selection_candidate_digest',selection_record['candidate_digest']==digest(ur['candidates']),ud)
    unitq=0;uniti=0
    for k,seed in enumerate(unit['proposal_seeds']):
        cd=ud/'candidates'/f'{k:02d}';td=cd/'tree';label=f'{unit["id"]}/{k:02d}'
        controls=npz(cd/'controls.npz');prov=js(cd/'provenance.json');outcome=js(cd/'outcome.json')
        same('outcome_matches_unit',outcome,ur['candidates'][k],label)
        check('sampling_seed',prov['sampling_seed']==seed and prov['network_calls']==16 and prov['slot']==k,label)
        close('network_raw_reconstruction',controls['network_raw'],neural(base,seed),2e-12,label)
        check('network_raw_hash',digest(controls['network_raw'])==prov['network_raw_hash'],label)
        direction=np.zeros(17)
        if k:
            angle=np.random.default_rng(seed-k).uniform(0,2*np.pi);frame=np.asarray(setup['frame'])
            v=np.cos(angle)*frame[0]+np.sin(angle)*frame[1];w=-np.sin(angle)*frame[0]+np.cos(angle)*frame[1]
            err=np.r_[np.zeros(6),[v,-v,w,-w][k-1]*setup['bow_amplitude_m'],np.zeros(3)];J=np.asarray(setup['jacobian'])
            direction=np.linalg.solve(J.T@J+1e-4*np.eye(17),J.T@err);direction*=min(1.,.3/max(float(abs(direction).max()),1e-12))
        bow=np.sin(np.pi*seed_phase)**2;bow[-1]=0
        proposal=controls['network_raw']+bow[:,None]*direction
        close('public_bow_reconstruction',controls['proposal'],proposal,1e-12,label)
        close('public_bow_direction',prov['bow_direction_rad'],direction,1e-12,label)
        prep=prov['preparation'];prepared=proposal if prep['scale']==1 else base+prep['scale']*(proposal-base)
        close('preparation_reconstruction',controls['prepared'],prepared,1e-12,label)
        check('proposal_hash',digest(controls['proposal'])==prov['proposal_hash'],label)
        sd=cd/'goal_seed';initial=npz(sd/'initial.npz');z=initial['Z'];e=pv(initial['ee']);seed_queries=1
        same('seed_initial_controls',z,controls['prepared'],label);close('seed_initial_residual',e,initial['goal'],where=label)
        for sp in sorted(sd.glob('iterations/*/step.npz')):
            s=npz(sp);r=js(sp.parent/'record.json');same('seed_parent_controls',s['before_Z'],z,label);close('seed_parent_residual',s['goal'],e,where=label)
            j=np.stack([(pv(p)-e)/h if h else np.zeros(12) for p,h in zip(s['perturb_ee'],s['epsilon'])],axis=1)
            close('seed_jacobian',s['jacobian'],j,1e-8,label);close('seed_step',s['step'],step(j,e,.15),1e-9,label);same('seed_basis',s['directions'],seed_basis,label)
            seed_queries+=int(np.count_nonzero(s['epsilon']));accepted=[]
            for trial in r['trials']:
                if 'ee' in trial:
                    seed_queries+=1;new=pv(trial['ee']);close('seed_trial_residual',new,trial['goal'],where=label)
                    accept=all(trial['quick'].values()) and all(a>1 or b<=1 for a,b in zip(groups(e),groups(new))) and new@new<e@e-1e-10
                    check('seed_trial_decision',accept==trial['accepted'],label)
                else:check('seed_unpropagated_trial',not all(trial['quick'].values()) and not trial['accepted'],label)
                if trial['accepted']:accepted.append(trial)
            check('seed_one_accepted',len(accepted)==int(r['accepted']),label)
            if accepted:
                z=z+accepted[0]['scale']*np.einsum('i,ijk->jk',s['step'],seed_basis);e=pv(accepted[0]['ee'])
            close('seed_update',s['after_Z'],z,1e-12,label)
            totals['seed_iterations']+=1
        sr=js(sd/'result.json');sa=npz(sd/'result.npz');close('seed_final_controls',sa['Z'],z,1e-12,label);close('seed_final_residual',sa['goal'],e,where=label)
        close('seed_final_Q',sr['Q'],max(groups(e)),where=label)
        n=npz(td/'nodes.npz');r=js(td/'result.json');attempts=js(td/'attempts.json');nn=len(n['q'])
        check('candidate_route_absent',r['solution_node'] is None and outcome['status']=='NO_COMPLETE_GLOBAL_ROUTE',label)
        check('later_stages_absent',not any((cd/x).exists() for x in ['fit.npz','fit.json','before_repair','repair','validated']) and not (ud/'execution').exists(),label)
        check('node_counts',nn==r['nodes'] and len(attempts)==r['attempts'],label)
        same('root_state',n['state'][0],ROOT_STATE,label);same('root_q',n['q'][0],task['condition']['initial_features'],label)
        state_parts('node_state',n['state'],n['qpos'],n['qvel'],n['elapsed'],label)
        close('node_map',n['qpos'][:,QIDS],n['q']@B.T,1e-10,label)
        close('node_rest_velocity',n['qvel'],np.zeros_like(n['qvel']),1e-12,label)
        same('node_other_state_fields',n['state'][:,161:],np.tile(ROOT_STATE[161:],(nn,1)),label)
        qvalues=np.array([qual(x) for x in n['ee']]);close('node_world_Q',n['Q'],qvalues,1e-10,label)
        check('best_node',r['best_node']==int(np.argmin(qvalues)) and abs(r['best_Q']-min(qvalues))<1e-10,label)
        depths=np.zeros(nn,dtype=int)
        for i in range(1,nn):
            p=int(n['parent'][i]);check('node_parent_order',0<=p<i,label);depths[i]=depths[p]+1
            close('node_time_chain',n['elapsed'][i]-n['elapsed'][p],n['duration'][i],1e-12,label)
            check('node_remaining_time',27-n['elapsed'][i]>=.35-1e-10,label)
        linear={};lin_count=0
        for lp in sorted(td.glob('steers/*/linearization.npz')):
            s=npz(lp);meta=js(lp.with_suffix('.json'));p=int(lp.parent.name);lin_count+=1
            same('linear_parent_state',s['parent_state'],n['state'][p],label);same('linear_parent_q',s['parent_q'],n['q'][p],label)
            check('linear_parent_hash',meta['parent_hash']==digest(s['parent_state']),label)
            expected_center=n['state'][p].copy();expected_center[0]+=1
            close('linear_center_state',s['center_state'],expected_center,1e-10,label)
            close('linear_center_pose',s['center_ee'],n['ee'][p],1e-10,label)
            e0=pv(s['center_ee']);close('linear_center_residual',s['goal'],e0,where=label)
            j=np.stack([(pv(ee)-e0)/h for ee,h in zip(s['perturb_ee'],s['epsilon'])],axis=1)
            close('linear_jacobian',j,s['jacobian'],1e-8,label)
            # Normal-equation solutions vary below 1e-9 rad with BLAS reduction
            # order/layout; retain the measured error and a 1e-9 rad threshold.
            close('linear_delta',step(j,e0,.12),s['delta'],1e-9,label)
            close('linear_perturb_target',s['perturb_target']-n['q'][p],np.diag(s['epsilon']),1e-12,label)
            check('linear_signed_epsilon',np.all(abs(s['epsilon'])==.0001) and np.all(s['perturb_target']>=LOWER) and np.all(s['perturb_target']<=UPPER),label)
            state_parts('linear_perturb',s['perturb_state'],qvel=np.zeros((17,79)),t=np.full(17,n['elapsed'][p]+1),where=label)
            close('linear_perturb_map',s['perturb_state'][:,1:82][:,QIDS],s['perturb_target']@B.T,1e-10,label)
            same('linear_other_state_fields',s['perturb_state'][:,161:],np.tile(n['state'][p,161:],(17,1)),label)
            linear[p]=s['delta']
        check('linearization_count',lin_count==r['world_linearizations'],label)
        edges={};edge_intervals=0;actual_intervals=0;visited_ids=[];state_to_node={digest(s):i for i,s in enumerate(n['state'])}
        for ep in sorted(td.glob('edges/*/report.json')):
            er=js(ep);a=npz(ep.parent/'trace.npz');eid=int(ep.parent.name);edges[eid]=er
            check('edge_id',er['edge_id']==eid,label);p=state_to_node.get(digest(a['parent_state']));check('edge_known_parent',p is not None,label)
            check('edge_parent_hash',digest(a['parent_state'])==er['parent_state_hash'],label)
            same('edge_start_qpos',a['qpos'][0],n['qpos'][p],label);same('edge_start_qvel',a['qvel'][0],n['qvel'][p],label)
            close('edge_time',a['time'],n['elapsed'][p]+np.arange(len(a['time']))/er['intervals']*er['duration'],1e-12,label)
            check('edge_actual_intervals',er['actual_intervals']==len(a['time'])-1<=er['intervals'],label)
            close('edge_minimum',np.min(a['minimum']),er['minimum_clearance_m'],1e-12,label)
            close('edge_geometry_max',np.max(a['geometry_step']),er['max_geometry_step_m'],1e-12,label)
            check('edge_numerical_decision',er['numerical_ok']==(er['solve_residual']<1e-10 and er['momentum_residual']<1e-10 and er['quaternion_error']<1e-12),label)
            check('edge_complete_decision',er['complete']==bool(np.min(a['minimum'])>=0 and len(a['time'])==er['intervals']+1),label)
            expected_accept=er['complete'] and er['numerical_ok'] and np.max(a['geometry_step'])<=.002
            check('edge_acceptance',er['accepted']==expected_accept and er['geometry_checked'] and er['qualification']=='TREE_SCREEN_ONLY',label)
            if er['complete']:
                state_parts('edge_child',a['child_state'],a['qpos'][-1],a['qvel'][-1],a['time'][-1],label)
                same('edge_unprescribed_fields',a['child_state'][161:],a['parent_state'][161:],label)
            else:check('edge_no_partial_child',len(a['child_state'])==0,label)
            # Algebraically verify every saved actuated qpos/qvel sample on the quintic edge.
            s=np.arange(len(a['time']))/er['intervals'];delta=np.asarray(er['target'])-n['q'][p]
            expected_q=n['q'][p]+(s**3*(10+s*(-15+6*s)))[:,None]*delta
            expected_v=(30*s**2*(1-s)**2)[:,None]*delta/er['duration']
            close('edge_quintic_qpos',a['qpos'][:,QIDS],expected_q@B.T,1e-10,label)
            close('edge_quintic_qvel',a['qvel'][:,VIDS],expected_v@B.T,1e-10,label)
            check('edge_duration_motion',er['duration']+1e-12>=minimum_duration(delta) and np.all(np.asarray(er['target'])>=LOWER) and np.all(np.asarray(er['target'])<=UPPER),label)
            edge_intervals+=er['intervals'];actual_intervals+=er['actual_intervals'];grid_counts[str(er['intervals'])]+=1
            reason='accepted' if er['accepted'] else ('penetration' if np.min(a['minimum'])<0 else ('numerical' if not er['numerical_ok'] else 'geometry_step'))
            edge_reasons[reason]+=1
            if reason=='penetration':collision_pairs[er['minimum_pair']['geom_a']+' | '+er['minimum_pair']['geom_b']]+=1
            where_nodes=np.flatnonzero(n['edge_id']==eid)
            for i in where_nodes:
                same('node_saved_child_state',n['state'][i],a['child_state'],label);same('node_saved_child_qpos',n['qpos'][i],a['qpos'][-1],label);same('node_saved_child_pose',n['ee'][i],a['ee'][-1],label)
                # Exact duplicate full states are allowed by the production tree.
                # A hash cannot identify a unique parent index in that case.
                check('node_edge_accepted',er['accepted'] and np.array_equal(n['state'][int(n['parent'][i])],a['parent_state']),label)
        check('edge_complete_index_set',set(edges)==set(range(r['propagations'])),label)
        visits=np.zeros(nn,dtype=int);world_count=0;accepted_by_mode=Counter();nonroot_expansions=0;world_reexpanded=0;accepted_world_nodes=set()
        world_parents=Counter();failed_world_parents=Counter();world_signatures={};world_repeated_exact=0
        for ai,a in enumerate(attempts):
            p=a['parent'];visits[p]+=1;mode_counts[a['sampling']]+=1
            if p:nonroot_expansions+=1
            if p in accepted_world_nodes:world_reexpanded+=1
            if a['sampling']=='WORLD_DESCENT':
                world_parents[p]+=1
                if not a['accepted']:failed_world_parents[p]+=1
                sig=digest([{key:t.get(key) for key in ['scale','target','duration','accepted','reason']} for t in a['trials']])
                if p in world_signatures:
                    check('repeated_world_identical_trials',sig==world_signatures[p],label)
                    world_repeated_exact+=int(sig==world_signatures[p])
                else:world_signatures[p]=sig
            check('attempt_id',ai==a['attempt'],label);close('attempt_before_Q',a['before_Q'],qvalues[p],where=label)
            close('attempt_parent_time',a['parent_elapsed'],n['elapsed'][p],where=label)
            check('attempt_parent_eligible',n['elapsed'][p]+1+.35<=27 and depths[p]+1<22,label)
            accepted_trials=[]
            for ti,t in enumerate(a['trials']):
                check('line_scale_sequence',t['scale']==[1,.5,.25][ti],label);ids=t['edge_ids'];visited_ids.extend(ids)
                if 'target' in t:
                    check('trial_edges_exist',all(i in edges for i in ids),label)
                    check('trial_grid_ladder',[edges[i]['intervals'] for i in ids]==[32,64,128,256,512,1024][:len(ids)],label)
                    for edge_id in ids:
                        same('trial_edge_target',t['target'],edges[edge_id]['target'],label);close('trial_edge_duration',t['duration'],edges[edge_id]['duration'],where=label)
                    if a['sampling']=='WORLD_DESCENT':
                        close('world_trial_target',t['target'],n['q'][p]+t['scale']*linear[p],1e-12,label)
                    expected_d=max(1.,minimum_duration(np.asarray(t['target'])-n['q'][p]));close('trial_duration_formula',t['duration'],expected_d,1e-12,label)
                if t['accepted']:
                    i=t['node'];accepted_trials.append(t);check('trial_child_reference',int(n['parent'][i])==p and int(n['edge_id'][i])==t['edge'],label)
                    close('trial_child_Q',t['after_Q'],qvalues[i],where=label)
                    if a['sampling']=='WORLD_DESCENT':
                        e0,e1=pv(n['ee'][p]),pv(n['ee'][i]);check('world_strict_descent',e1@e1<e0@e0-1e-10,label);world_count+=1;accepted_world_nodes.add(i);all_world_children.append((label,int(i),float(n['elapsed'][i]),float(qvalues[i])))
                else:
                    reasons[t['reason']]+=1
                    if t['reason']=='MOTION_OR_GEOMETRY_REJECTED':check('trial_geometry_reject',not ids or not edges[ids[-1]]['accepted'],label)
                    if t['reason']=='NO_WORLD_DESCENT':
                        check('trial_non_descent',bool(ids) and edges[ids[-1]]['accepted'] and pv(t['ee'])@pv(t['ee'])>=pv(n['ee'][p])@pv(n['ee'][p])-1e-10,label)
            check('attempt_decision',len(accepted_trials)==int(a['accepted']),label)
            if a['accepted']:
                accepted_by_mode[a['sampling']]+=1;close('accepted_remaining_time',a['remaining_time'],27-n['elapsed'][a['node']],where=label)
            check('no_terminal_hold',not a.get('terminal_hold_edges'),label)
        same('visits_recount',visits,n['visits'],label)
        check('attempt_edge_coverage',sorted(visited_ids)==list(range(len(edges))),label)
        check('node_attempt_coverage',sorted(a['node'] for a in attempts if a['accepted'])==list(range(1,nn)),label)
        qcount=seed_queries+lin_count*18+len(edges);icount=seed_queries*512+lin_count*18*256+edge_intervals
        unitq+=qcount;uniti+=icount;queries_total+=qcount;reserved_total+=icount;actual_saved_intervals+=lin_count*18*256+actual_intervals
        duplicate_full_states=nn-len(np.unique(n['state'],axis=0))
        repeated_failed=sum(failed_world_parents.values())-len(failed_world_parents)
        totals.update(candidates=1,nodes=nn,accepted_nodes=nn-1,attempts=len(attempts),edges=len(edges),linearizations=lin_count,linear_endpoints=lin_count*18,seed_queries=seed_queries,nonroot_expansions=nonroot_expansions,world_nodes_reexpanded_attempts=world_reexpanded,world_accepted=world_count,duplicate_full_states=duplicate_full_states,repeated_world_identical_trials=world_repeated_exact,repeated_failed_world_attempts=repeated_failed)
        row={'candidate':label,'seed':seed,'stop':r['stop'],'attempts':len(attempts),'nodes':nn,'accepted_by_mode':dict(accepted_by_mode),'best_Q':float(min(qvalues)),'best_elapsed_s':r['best_elapsed'],'root_Q':float(qvalues[0]),'best_remaining_s':27-r['best_elapsed'],'max_depth':int(max(depths)),'max_elapsed_s':float(max(n['elapsed'])),'min_node_remaining_s':float(27-max(n['elapsed'])),'world_accepted':world_count,'nonroot_expansions':nonroot_expansions,'world_reexpanded_attempts':world_reexpanded,'edges':len(edges),'linearizations':lin_count,'seed_queries':seed_queries,'reconstructed_queries':qcount,'reserved_intervals':icount,'saved_edge_actual_intervals':actual_intervals,'seed_Q':sr['Q']}
        row.update(duplicate_full_states=duplicate_full_states,repeated_world_identical_trials=world_repeated_exact,repeated_failed_world_attempts=repeated_failed)
        candidate_rows.append(row);print(json.dumps(row),flush=True)
    calls=js(ud/'calls.json');check('unit_calls_reconstruction',unitq==calls['coupled'],unit['id']);unit_queries[unit['id']]=unitq;unit_reservations[unit['id']]=uniti
    check('unit_limits',unitq<=19980 and uniti<=35000000 and calls['physics_steps']==0,unit['id'])

check('all_coupled_reconstruction',queries_total==budget['actual_coupled']==budget['reserved_coupled']==sum(v for k,v in ledger['calls'].items() if k.endswith('.coupled')))
check('all_reserved_intervals_reconstruction',reserved_total==budget['integration_intervals'])
check('zero_native_steps',budget['actual_native_and_replay_steps']==sum(v for k,v in ledger['calls'].items() if k.endswith('.physics_steps'))==0)
check('ledger_closed',ledger['inflight'] is None and not ledger.get('unresolved_call_accounting') and set(ledger['completed'])=={'m0','u00','u01','u02'})
check('hard_limits',queries_total<=60000 and reserved_total<=110000000 and status['elapsed_s']<=21540)
check('summary_denominators',summary['methods']['diffusion_geometry']=={'candidate_denominator':15,'executed':0,'execution_denominator':3,'qualified':0,'verified':0})
check('events_stdout_bytes',read(RUN/'events.jsonl')==read(RUN/'worker.stdout.log'))
check('stderr_empty',len(read(RUN/'worker.stderr.log'))==0)
audit_path=ROOT/'paper/PAPER_CLAIM_AUDIT.json'
paper_audit=js(audit_path) if audit_path.exists() else None
result={'scope':'independent saved-array and neural reconstruction; no physical/geometry/optimizer/training/start/resume calls','check_count':sum(checks.values()),'checks_by_category':dict(checks),'failures':failures,'max_abs_differences':maxima,'totals':dict(totals),'candidates':candidate_rows,'m0_base_max_delta':base_motion,'m0_eight_recorded_checks':mr['checks'],'mode_counts':dict(mode_counts),'trial_reasons':dict(reasons),'edge_reasons':dict(edge_reasons),'edge_grid_counts':dict(grid_counts),'penetration_pairs':dict(collision_pairs),'budget':{'actual_and_reserved_coupled_rebuilt':queries_total,'reserved_intervals_rebuilt':reserved_total,'saved_edge_m0_and_linearization_actual_intervals':actual_saved_intervals,'seed_intervals_reserved_not_full_traces':totals['seed_queries']*512,'unit_coupled':unit_queries,'unit_reserved_intervals':unit_reservations,'elapsed_s':status['elapsed_s'],'native_steps':0,'unused_coupled':60000-queries_total,'unused_interval_reservation':110000000-reserved_total,'unused_quota_authorizes_extension':False},'original_raw_audit':{'status':original_audit['status'],'checks':len(original_audit['checks']),'false_checks':[k for k,v in original_audit['checks'].items() if not v]},'paper_claim_audit':'unavailable' if paper_audit is None else paper_audit,'review_elapsed_s':time.time()-started,'input_count':len(inputs),'input_manifest_sha256':digest(list(inputs.values()))}
(HERE/'compact_evidence.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,default=plain),encoding='utf-8')
with gzip.open(HERE/'inputs.files.jsonl.gz','wt',encoding='utf-8') as f:
    for row in inputs.values():f.write(json.dumps(row,ensure_ascii=False)+'\n')
(HERE/'inputs.json').write_text(json.dumps({'reviewed_run':str(RUN),'producer_head':run['producer_head'],'source_files':len(ident['source_hashes']),'bound_inputs':len(ident['binding']['files']),'read_file_count':len(inputs),'all_read_inputs':'inputs.files.jsonl.gz','all_read_inputs_uncompressed_digest':result['input_manifest_sha256'],'top_level_inputs':[v for p,v in inputs.items() if p.parent==RUN],'paper_claim_audit':result['paper_claim_audit']},ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({'checks':sum(checks.values()),'failure_count':len(failures),'failure_categories':dict(Counter(x['check'] for x in failures)),'totals':dict(totals),'queries':queries_total,'reserved_intervals':reserved_total,'elapsed_s':time.time()-started}),flush=True)
