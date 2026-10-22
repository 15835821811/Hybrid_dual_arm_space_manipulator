"""Independent read-only arithmetic/identity audit of completed contact_steer_001.

No repository imports, MuJoCo imports, physics/geometry queries, optimizer calls,
replay, network evaluation, or modification of the frozen run/source are used.
SciPy Rotation is used only for algebra on saved 3x3 matrices.
"""
import collections
import datetime
import hashlib
import json
import math
from pathlib import Path
import sys
import xml.etree.ElementTree as ET
import zipfile
import numpy as np
from scipy.spatial.transform import Rotation

ROOT=Path(__file__).resolve().parents[4]
TRACE=Path(__file__).resolve().parent
OUT=ROOT/'artifacts/v7_1/contact_steer_001'
PKG=ROOT/'ff_prior_contact_steer'
checks=[]; manifest={}; stats={'units':[]}; linear_rows=[]; edge_rows=[]; solver_rows=[]
def plain(x):
    if isinstance(x,np.ndarray):return x.tolist()
    if isinstance(x,np.generic):return x.item()
    if isinstance(x,Path):return str(x)
    raise TypeError(type(x).__name__)
def dump(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,default=plain,allow_nan=False)+'\n',encoding='utf-8')
def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),default=plain,allow_nan=False).encode()).hexdigest()
def hashed(p,scope='new evidence'):
    p=p.resolve();key=str(p);s=p.stat()
    if key not in manifest:
        h=hashlib.sha256()
        with p.open('rb') as f:
            for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
        manifest[key]={'path':key,'sha256':h.hexdigest(),'bytes':s.st_size,'mtime_utc':datetime.datetime.fromtimestamp(s.st_mtime,datetime.timezone.utc).isoformat(),'scope':scope}
    return manifest[key]['sha256']
def read(p,scope='new evidence'):
    hashed(p,scope);return json.loads(p.read_text(encoding='utf-8-sig'))
def arr(p,scope='new saved arrays'):
    hashed(p,scope)
    with np.load(p,allow_pickle=False) as a:return {k:a[k] for k in a.files}
def ck(name,truth,detail=None):
    checks.append({'name':name,'passed':bool(truth),**({'detail':detail} if detail is not None else {})})
def near(a,b,atol=1e-10):return np.allclose(a,b,atol=atol,rtol=0)
def maxerr(a,b):return float(np.max(np.abs(np.asarray(a)-np.asarray(b)),initial=0))
def finite(a):return all(np.isfinite(v).all() for v in a.values())
def pose(a,g):
    return np.r_[((a[:,:3,3]-g[:,:3,3])/.005).ravel(),(Rotation.from_matrix(np.swapaxes(g[:,:3,:3],-1,-2)@a[:,:3,:3]).as_rotvec()/np.deg2rad(1)).ravel()]
def Q(a,g):return float(np.linalg.norm(pose(a,g).reshape(4,3),axis=1).max())
def exact(n,i,c,target=None,duration=None):
    h=hashlib.sha256(json.dumps(c,sort_keys=True,separators=(',',':')).encode())
    for a in (n['state'][i],n['q'][i],n['qpos'][i],n['qvel'][i],np.asarray([n['elapsed'][i]]),np.asarray([] if target is None else target),np.asarray([] if duration is None else [duration])):
        a=np.ascontiguousarray(a,dtype=np.float64);h.update(str(a.shape).encode());h.update(a.tobytes())
    return h.hexdigest()
def blend(s):return s**3*(10+s*(-15+6*s))
def vel(s):return 30*s**2*(1-s)**2
def minduration(delta):return max(P['tree_min_duration_s'],P['tree_time_margin']*max(np.max(1.875*np.abs(delta)/vlim),np.max(np.sqrt((10/np.sqrt(3))*np.abs(delta)/alim))))
def tangent(jd,dist,nom):
    if not len(dist):return np.zeros(17)
    normal=jd[int(np.argmin(dist))].copy();normal/=max(float(np.linalg.norm(normal)),1e-12)
    g=nom/max(float(np.linalg.norm(nom)),1e-12);A=np.stack([normal,g]);proj=np.eye(17)-np.linalg.pinv(A)@A
    sens=np.linalg.norm(jd,axis=0)
    if sens.max()<1e-12:return np.zeros(17)
    k=int(np.argmax(np.linalg.norm(proj,axis=0)*sens));t=proj[:,k];t/=max(float(np.max(np.abs(t))),1e-12)
    return P['contact_tangent_rad']*t

run=read(OUT/'run.json');I=run['identity'];P=I['protocol'];S=I['selection']
contract=read(OUT/'physical_contract.json');physical=contract['identity'];budget=read(OUT/'budget.json');ledger=read(OUT/'ledger.json')
status=read(OUT/'status.json');scene=read(OUT/'scenes/capture_5_obstacles/initial_audit.json');limits=read(OUT/'scenes/capture_5_obstacles/limits.json')
lo=np.asarray(limits['lower']);hi=np.asarray(limits['upper']);task=S['entries'][0]['task'];goals=np.asarray(task['condition']['goals'])
ck('producer_commit',run['producer_head']=='73be52493b09c4ff27a34340efe36bc8a653b97d')
ck('expected_sources_inputs',len(I['source_hashes'])==140 and len(I['binding']['files'])==206)
source_bad=[];archive_bad=[];input_bad=[]
with zipfile.ZipFile(OUT/'executed_source.zip') as z:
    hashed(OUT/'executed_source.zip')
    ck('archive_exact_member_set',set(z.namelist())==set(I['source_hashes']))
    for p,h in I['source_hashes'].items():
        if hashlib.sha256(z.read(p)).hexdigest()!=h:archive_bad.append(p)
        if hashed(ROOT/p,'frozen source byte identity')!=h:source_bad.append(p)
for p,h in I['binding']['files'].items():
    if hashed(ROOT/p,'bound input byte identity only; old results/reviews not interpreted')!=h:input_bad.append(p)
ck('source_archive_hashes',not archive_bad,archive_bad);ck('working_source_hashes',not source_bad,source_bad);ck('bound_input_hashes',not input_bad,input_bad)
ck('selection_identity',read(PKG/'evidence/selection.json')==S)
ck('binding_identity',read(PKG/'evidence/binding.json')==I['binding'])
ck('software_test_source_binding',read(PKG/'evidence/software_tests.json')['source_hashes']==I['source_hashes'])
ck('task_frozen_hash',digest({k:v for k,v in task.items() if k!='task_hash'})==task['task_hash'])
ck('original_task_scope',task['condition']['duration']==27 and len(S['entries'][0]['scene']['objects'])==5 and S['candidate_denominator']==S['execution_denominator']==3)
ck('paper_claim_audit_absent',not (ROOT/'paper/PAPER_CLAIM_AUDIT.json').exists())
ck('ledger_no_unknown',ledger['inflight'] is None and not ledger.get('unresolved_call_accounting'))
ck('ledger_completed_units',set(ledger['completed'])=={x['id'] for x in S['units']})
ck('ledger_identity',ledger['identity']['identity_hash']==digest(I))
ck('frozen_budget_matches',budget['limits']==P)
ck('budget_caps',budget['reserved_coupled']<=P['coupled_cap'] and budget['integration_intervals']<=P['integration_cap'] and budget['actual_native_and_replay_steps']<=P['native_steps_cap'] and status['elapsed_s']<P['wall_s'])
ck('terminal_status',status['stage']=='complete' and status['status']=='FINISHED_WITH_BLOCKED_ITEMS')
ck('native_zero',budget['actual_native_and_replay_steps']==0 and all(v==0 for k,v in ledger['calls'].items() if k.endswith(('.physics_steps','.torque_rollout'))))
ck('task_scene_identity',task['contract']['robot_hash']==scene['scene_transfer']['scene_robot_hash'] and task['contract']['initial_state_id']==scene['scene_task_origin_state_id'] and task['contract']['pair_policy_hash']==scene['initial_geometry']['pair_policy_hash'])

# Reproduce the frozen 67x17 map and motion bounds from saved/source constants,
# without calling RobotModelSpec.from_urdf (which would instantiate MuJoCo).
vlim=np.r_[np.full(10,.8),np.full(7,1.2)];alim=np.r_[np.full(10,2.5),np.full(7,4.)]
B=np.zeros((67,17));B[:60,:10]=np.kron(np.eye(5),np.array([[0,1],[1,0],[1,0],[0,1],[0,1],[1,0],[1,0],[0,1],[0,1],[1,0],[1,0],[0,1]])/6);B[60:,10:]=np.eye(7)
joint_names=physical['robot']['low_level_joint_names'];qids=np.array([physical['joint_index_map'][j]['qpos'] for j in joint_names]);vids=np.array([physical['joint_index_map'][j]['dof'] for j in joint_names]);nq=physical['nq'];nv=physical['nv']
urdf=ROOT/'dual_arm_space_robot_2026/urdf/dual_arm_space_robot_2026.urdf';hashed(urdf,'frozen physical model source; XML limits only')
limits_xml={j.attrib['name']:(j.attrib.get('type'),j.find('limit')) for j in ET.parse(urdf).getroot().findall('joint')}
lowlo=[];lowhi=[]
for j in joint_names:
    typ,lim=limits_xml[j]
    lowlo.append(float(lim.attrib['lower']) if lim is not None and typ!='continuous' else -math.inf)
    lowhi.append(float(lim.attrib['upper']) if lim is not None and typ!='continuous' else math.inf)
lowlo=np.asarray(lowlo);lowhi=np.asarray(lowhi)
all_calls=collections.Counter();global_actions=[];global_nodes=[];full_queries=0;reserved_intervals=0;observed_edge_intervals=0;new_count=0;prefix_count=0
for spec in S['units']:
    uid=spec['id'];d=OUT/'units'/uid;n=arr(d/'tree/nodes.npz');r=read(d/'tree/result.json');result=read(d/'result.json');attempts=read(d/'tree/attempts.json');calls=read(d/'calls.json');all_calls.update(calls)
    prefix=arr(ROOT/spec['prefix_file'],'bound prefix identity');count=r['historical_prefix_nodes'];N=len(n['q']);new_count+=N-count;prefix_count+=count
    old=arr(ROOT/spec['source_tree']/'nodes.npz','selected historical prefix identity only')
    ck(uid+'/prefix_selection_arrays',all(np.array_equal(prefix[k],old[k][spec['source_node_chain']]) for k in ('q','state','qpos','qvel','elapsed','duration','ee','edge_id')) and np.array_equal(prefix['source_node_id'],spec['source_node_chain']))
    ck(uid+'/prefix_parent_identity',np.array_equal(prefix['parent'],np.arange(count)-1))
    del old
    ck(uid+'/reused_prefix_exact',all(np.array_equal(n[k][:count],prefix[k]) for k in n))
    ck(uid+'/complete_state_shape',n['state'].shape==(N,836) and finite(n))
    ck(uid+'/root_full_state_hash',digest(n['state'][0])==scene['origin']['complete_state_hash'])
    ck(uid+'/root_state_id',digest({'values':n['state'][0],'model_hash':scene['scene_transfer']['scene_robot_hash'],'state_mask':contract['initial_state_mask']})==task['contract']['initial_state_id'])
    ck(uid+'/root_q',np.array_equal(n['q'][0],task['condition']['initial_features']) and n['elapsed'][0]==0 and n['parent'][0]==-1)
    ck(uid+'/state_fields',np.array_equal(n['state'][:,0],n['elapsed']) and np.array_equal(n['state'][:,1:1+nq],n['qpos']) and np.array_equal(n['state'][:,1+nq:1+nq+nv],n['qvel']))
    ck(uid+'/rest_nodes_no_clipping',np.max(np.abs(n['qvel']))==0 and near(n['qpos'][:,qids],n['q']@B.T,1e-12))
    ck(uid+'/planner_bounds',np.all(n['q']>=lo) and np.all(n['q']<=hi))
    ck(uid+'/low_level_bounds',np.all(n['qpos'][:,qids]>=lowlo) and np.all(n['qpos'][:,qids]<=lowhi))
    proof=read(d/'prefix_identity.json');ck(uid+'/prefix_proof',proof['passed'] and all(proof['checks'].values()))
    context=r['context'];ck(uid+'/context',context['source_hashes']==digest(I['source_hashes']) and context['protocol_hash']==digest(P) and context['task_hash']==task['task_hash'] and context['scene_model']==task['contract']['robot_hash'] and context['collision_policy']==task['contract']['pair_policy_hash'] and context['initial_state_id']==task['contract']['initial_state_id'] and context['edge_grids']==P['edge_grids'])
    nk=[exact(n,i,context) for i in range(N)];global_nodes.extend(nk[count:]);ck(uid+'/exact_node_unique',len(nk)==len(set(nk)))
    qualities=np.array([Q(ee,goals) for ee in n['ee']]);best=int(np.argmin(qualities));best_e=pose(n['ee'][best],goals);norms=np.linalg.norm(best_e.reshape(4,3),axis=1)
    ck(uid+'/best_Q',qualities[best]==r['best_Q'] and qualities[count-1]==spec['source_Q'])
    ck(uid+'/all_new_parent_time',all(0<=n['parent'][i]<i and abs(n['elapsed'][i]-n['elapsed'][int(n['parent'][i])]-n['duration'][i])<1e-12 and n['elapsed'][i]<27 for i in range(count,N)))
    ck(uid+'/tail_inherited',all(np.array_equal(n['state'][i,1+nq+nv:],n['state'][int(n['parent'][i]),1+nq+nv:]) for i in range(count,N)))
    ck(uid+'/no_solution_downstream',r['solution_node'] is None and not result['eligible'] and result['qualified']==0 and result['execution_level']=='NOT_EXECUTED' and all(not (d/p).exists() for p in ('fit.npz','fit.json','before_repair','repair','validated','execution')))
    exsel=read(d/'execution_selection.json');ck(uid+'/execution_selection_blocked',not exsel['selected'] and exsel['task_hash']==task['task_hash'] and exsel['original_initial_state_id']==task['contract']['initial_state_id'])
    edge_paths=sorted((d/'tree/edges').glob('*/report.json'));edges={};eres={};edge_to_node={int(n['edge_id'][i]):i for i in range(count,N)};edge_to_trial={}
    actual_keys=[];allkeys=[];expanded=[];trialreasons=collections.Counter();accepted_scales=collections.Counter();variant_counts=collections.Counter();visible=count;seen=set();accepted_from_failed_solver=0;bound_max=0.;dtmin=math.inf;velocity_ratio=0.;accel_ratio=0.
    for iteration,a in enumerate(attempts):
        parent=a['parent'];expanded.append(parent)
        possible=[i for i in range(count-1,visible) if i not in seen and n['elapsed'][i]+1.+P['tree_min_duration_s']<=27]
        expected=min(possible,key=lambda i:(np.log1p(qualities[i])+P['contact_parent_time_weight']*n['elapsed'][i]/27,i))
        ck(f'{uid}/parent_choice/{iteration}',parent==expected and a['iteration']==iteration and a['before_Q']==qualities[parent]);seen.add(parent)
        p=d/'tree/linear'/f'{iteration:03d}'/'jacobian.npz';j=arr(p);m=read(p.with_suffix('.json'));full_queries+=37;reserved_intervals+=37*256
        ck(f'{uid}/linear_shape/{iteration}',finite(j) and j['hold_states'].shape==(17,836) and j['perturb_states'].shape==(17,836) and j['perturb_qpos'].shape==(17,257,nq) and j['half_qpos'].shape==(257,nq) and m['queries']==37 and m['intervals']==256)
        ck(f'{uid}/linear_parent/{iteration}',np.array_equal(j['parent_state'],n['state'][parent]) and np.array_equal(j['parent_q'],n['q'][parent]) and m['parent_hash']==digest(n['state'][parent]) and m['geometry_policy_hash']==context['collision_policy'])
        hold=pose(j['center_ee'],goals);goal=pose(j['nominal_ee'],goals);hj=np.stack([(pose(e,goals)-hold)/h for e,h in zip(j['hold_ee'],j['hold_epsilon'])],axis=1);gj=np.stack([(pose(e,goals)-goal)/h for e,h in zip(j['perturb_ee'],j['epsilon'])],axis=1);dj=((j['perturb_distances']-j['distances'])/j['epsilon'][:,None]).T
        ck(f'{uid}/fd_exact/{iteration}',np.array_equal(hold,j['hold_e']) and np.array_equal(goal,j['goal']) and np.array_equal(hj,j['hold_j']) and np.array_equal(gj,j['jg']) and np.array_equal(dj,j['jd']))
        # Linear-system arithmetic reconstructs the deterministic nominal; this
        # is not an optimizer invocation/step and triggers no model evaluation.
        nominal=-np.linalg.solve(hj.T@hj+.001*np.eye(17),hj.T@hold);nominal*=min(1.,.12/max(float(np.max(np.abs(nominal))),1e-12));nominal=np.clip(nominal,lo-n['q'][parent],hi-n['q'][parent])
        ck(f'{uid}/nominal/{iteration}',near(nominal,j['nominal'],1e-10))
        duration=float(j['duration']);target=n['q'][parent]+j['nominal'];h=np.full(17,1e-4);h[n['q'][parent]+h>hi]*=-1;h2=np.full(17,1e-4);h2[target+h2>hi]*=-1
        ck(f'{uid}/fd_domains/{iteration}',np.array_equal(h,j['hold_epsilon']) and np.array_equal(h2,j['epsilon']) and np.all(n['q'][parent]+h>=lo) and np.all(target+h2>=lo) and duration==max(1.,minduration(j['nominal'])))
        sample=np.arange(257)/256;nomexpected=(n['q'][parent]+blend(sample)[:,None]*j['nominal'])@B.T
        ck(f'{uid}/nominal_saved_motion/{iteration}',near(j['nominal_qpos'][:,qids],nomexpected,1e-12) and near(j['nominal_qpos'][0],n['qpos'][parent],1e-12) and np.array_equal(j['nominal_qpos'][-1],j['nominal_state'][1:1+nq]))
        state_values=np.vstack([j['center_state'][None],j['hold_states'],j['nominal_state'][None],j['perturb_states']])
        expectedtime=np.r_[np.full(18,n['elapsed'][parent]+1.),np.full(18,n['elapsed'][parent]+duration)]
        ck(f'{uid}/derivative_full_state/{iteration}',np.array_equal(state_values[:,0],expectedtime) and np.max(np.abs(state_values[:,1+nq:1+nq+nv]))==0 and all(np.array_equal(s[1+nq+nv:],n['state'][parent,1+nq+nv:]) for s in state_values))
        ck(f'{uid}/hold_endpoints/{iteration}',near(j['center_state'][1:1+nq][qids],n['q'][parent]@B.T,1e-12) and near(j['hold_states'][:,1:1+nq][:,qids],(n['q'][parent]+np.diag(h))@B.T,1e-12))
        for k in range(17):
            delta=j['nominal'].copy();delta[k]+=h2[k];expect=(n['q'][parent]+blend(sample)[:,None]*delta)@B.T
            ck(f'{uid}/perturb_motion/{iteration}/{k}',near(j['perturb_qpos'][k,:,qids].T,expect,1e-12) and near(j['perturb_qpos'][k,0],n['qpos'][parent],1e-12) and np.array_equal(j['perturb_qpos'][k,-1],j['perturb_states'][k,1:1+nq]))
        active=np.asarray(m['active']);phases=list(range(8,257,8));k=int(j['half_coordinate']);expected_k=int(np.argmax(np.linalg.norm(dj,axis=0))) if len(active) else int(np.argmax(np.linalg.norm(gj,axis=0)))
        delta=j['nominal'].copy();delta[k]+=.5*h2[k];halfexpect=(n['q'][parent]+blend(sample)[:,None]*delta)@B.T
        ck(f'{uid}/half_independent_motion/{iteration}',k==expected_k and near(j['half_qpos'][:,qids],halfexpect,1e-12) and near(j['half_qpos'][0],n['qpos'][parent],1e-12))
        scans=m['scans'];ck(f'{uid}/phase_coverage/{iteration}',m['whole_edge_sample_indices']==phases and [x['index'] for x in scans]==phases and set(active[:,0])==set(phases) and len(active)==len(set(map(tuple,active))) and len(active)<=96 and m['all_near_pairs']>=len(active))
        ck(f'{uid}/phase_minima_retained/{iteration}',all(near(np.min(j['distances'][active[:,0]==s['index']]),s['minimum'],1e-12) for s in scans))
        hd=(j['half_distances']-j['distances'])/(.5*h2[k]);hg=(pose(j['half_ee'],goals)-goal)/(.5*h2[k]);de=float(np.max(np.abs(hd-dj[:,k]),initial=0));ge=float(np.max(np.abs(hg-gj[:,k]),initial=0));drel=float(np.linalg.norm(hd-dj[:,k])/max(np.linalg.norm(dj[:,k]),1e-12));grel=float(np.linalg.norm(hg-gj[:,k])/max(np.linalg.norm(gj[:,k]),1e-12))
        t=tangent(dj,j['distances'],j['nominal']);norm=dj[int(np.argmin(j['distances']))];normal_orth=abs(float(norm@t))/max(float(np.linalg.norm(norm)*np.linalg.norm(t)),1e-12);goal_orth=abs(float(j['nominal']@t))/max(float(np.linalg.norm(j['nominal'])*np.linalg.norm(t)),1e-12)
        ck(f'{uid}/tangent_orthogonal/{iteration}',normal_orth<1e-9 and goal_orth<1e-9)
        linear_rows.append({'unit':uid,'iteration':iteration,'parent':parent,'time':float(n['elapsed'][parent]),'active':len(active),'all_near_records':m['all_near_pairs'],'omitted_count':m['all_near_pairs']-len(active),'active_dist_min_m':float(j['distances'].min()),'half_coordinate':k,'distance_half_abs_error_m_per_rad':de,'distance_half_relative_l2_error':drel,'world_half_abs_error_normalized_per_rad':ge,'world_half_relative_l2_error':grel,'nominal_Q':Q(j['nominal_ee'],goals),'tangent_normal_cosine_abs':normal_orth,'tangent_goal_cosine_abs':goal_orth})
        ck(f'{uid}/solver_variant_order/{iteration}',[s['variant'] for s in a['solvers']]==[0,1,-1])
        for s in a['solvers']:
            x=np.asarray(s['delta']);sl=np.maximum(lo-n['q'][parent],-.12);su=np.minimum(hi-n['q'][parent],.12);bias=s['variant']*t;distance=(j['distances']+dj@(x-j['nominal'])-.0002)/.003;minmargin=float(np.min(distance,initial=np.inf));admissible=bool(np.isfinite(x).all() and np.all(x>=sl-1e-9) and np.all(x<=su+1e-9) and minmargin>=-1e-6 and np.max(np.abs(x))>1e-9);er=goal+gj@(x-j['nominal']);objective=.5*float(er@er)/max(1.,float(goal@goal))+.015*float(np.sum(((x-j['nominal']-bias)/.12)**2));violation=max(float(np.max(sl-x)),float(np.max(x-su)),0.);bound_max=max(bound_max,violation)
            ck(f'{uid}/solver_return/{iteration}/{s["variant"]}',np.array_equal(s['lower'],sl) and np.array_equal(s['upper'],su) and near(s['tangent_bias'],bias,1e-10) and abs(objective-s['objective'])<1e-8 and abs(minmargin-s['minimum_normalized_linear_distance'])<1e-8 and admissible==s['linear_admissible'])
            solver_rows.append({'unit':uid,'iteration':iteration,'variant':s['variant'],'success':s['success'],'linear_admissible':admissible,'iterations':s['iterations'],'bound_violation_rad':violation,'min_normalized_linear_margin':minmargin,'message':s['message']})
        for trial in a['trials']:
            trialreasons[trial['reason']]+=1;solved=next(s for s in a['solvers'] if s['variant']==trial['variant']);delta=trial['scale']*np.asarray(solved['delta']);target=n['q'][parent]+delta;dur=trial['duration'];dtmin=min(dtmin,dur);velocity_ratio=max(velocity_ratio,float(np.max(1.875*np.abs(delta)/dur/vlim)));accel_ratio=max(accel_ratio,float(np.max((10/np.sqrt(3))*np.abs(delta)/dur**2/alim)))
            ck(f'{uid}/trial_proposal/{iteration}/{trial["variant"]}/{trial["scale"]}',np.array_equal(target,trial['target']) and dur==max(1.,minduration(delta)) and solved['linear_admissible'])
            if 'exact_action_key' in trial:
                key=exact(n,parent,context,target,dur);ck(f'{uid}/action_key/{iteration}/{trial["variant"]}/{trial["scale"]}',key==trial['exact_action_key']);allkeys.append(key)
            if trial.get('edge_ids'):
                actual_keys.append(trial['exact_action_key']);global_actions.append(trial['exact_action_key'])
                for eid in trial['edge_ids']:edge_to_trial[eid]=(parent,trial)
            if trial['accepted']:
                ni=trial['node'];ck(f'{uid}/accepted_trial/{ni}',ni==visible and n['parent'][ni]==parent and near(n['q'][ni],target,0) and n['duration'][ni]==dur and trial['after_Q']==qualities[ni] and trial['after_Q']<=1.3*qualities[parent] and n['edge_id'][ni]==trial['edge']);visible+=1;accepted_scales[str(trial['scale'])]+=1;variant_counts[str(trial['variant'])]+=1;accepted_from_failed_solver+=not solved['success']
        del j
    ck(uid+'/expanded_once',len(expanded)==len(set(expanded))==r['expanded_parents'])
    ck(uid+'/action_unique',len(actual_keys)==len(set(actual_keys)) and len(set(allkeys))==r['exact_unique_action_keys'])
    ck(uid+'/node_trial_count',visible==N and len(edge_to_node)==r['new_nodes'])
    ck(uid+'/propagation_count',len(edge_paths)==r['propagations']==len(edge_to_trial))
    clear_min=math.inf;geom_max=0.;quat_max=0.;numerical_max={'solve':0.,'momentum':0.};accepted_edges=0;complete_edges=0;colliding_edges=0;actualints=0;reservedints=0;grids=collections.Counter();edge_time_errors=0.;edge_motion_error=0.;edge_velocity_error=0.
    for p in edge_paths:
        report=read(p);a=arr(p.with_name('trace.npz'));eid=report['edge_id'];parent,trial=edge_to_trial[eid];target=np.asarray(trial['target']);dur=trial['duration'];intervals=report['intervals'];L=len(a['time']);s=np.arange(L)/intervals;actualints+=L-1;reservedints+=intervals;reserved_intervals+=intervals;observed_edge_intervals+=L-1;grids[str(intervals)]+=1
        ck(f'{uid}/edge_link/{eid}',np.array_equal(a['parent_state'],n['state'][parent]) and digest(a['parent_state'])==report['parent_state_hash'] and np.array_equal(a['target'],target) and np.array_equal(report['target'],target) and report['duration']==dur)
        ck(f'{uid}/edge_finite/{eid}',finite(a) and report['actual_intervals']==L-1 and intervals in P['edge_grids'] and report['geometry_checked'])
        timeerr=maxerr(a['time'],n['elapsed'][parent]+s*dur);edge_time_errors=max(edge_time_errors,timeerr)
        expect=(n['q'][parent]+blend(s)[:,None]*(target-n['q'][parent]))@B.T;expectv=(vel(s)[:,None]*(target-n['q'][parent])/dur)@B.T
        motionerr=maxerr(a['qpos'][:,qids],expect);velocityerr=maxerr(a['qvel'][:,vids],expectv);edge_motion_error=max(edge_motion_error,motionerr);edge_velocity_error=max(edge_velocity_error,velocityerr)
        ck(f'{uid}/edge_motion/{eid}',timeerr<1e-12 and motionerr<1e-12 and velocityerr<1e-12 and near(a['qpos'][0],n['qpos'][parent],1e-12) and np.all(a['qpos'][:,qids]>=lowlo) and np.all(a['qpos'][:,qids]<=lowhi))
        complete=report['complete'];complete_edges+=complete;colliding_edges+=not complete;numerical=(report['solve_residual']<1e-10 and report['momentum_residual']<1e-10 and report['quaternion_error']<1e-12);accepted=complete and numerical and float(np.max(a['geometry_step']))<=.002
        ck(f'{uid}/edge_gate/{eid}',report['numerical_ok']==numerical and report['accepted']==accepted and report['minimum_clearance_m']==float(np.min(a['minimum'])) and report['max_geometry_step_m']==float(np.max(a['geometry_step'])) and (complete==(L==intervals+1 and np.min(a['minimum'])>=0)) and (complete or a['minimum'][-1]<0))
        ck(f'{uid}/edge_time/{eid}',report['start_time']==n['elapsed'][parent] and report['end_time']==a['time'][-1])
        quaterr=float(np.max(np.abs(np.linalg.norm(a['qpos'][:,3:7],axis=1)-1)));quat_max=max(quat_max,quaterr);numerical_max['solve']=max(numerical_max['solve'],report['solve_residual']);numerical_max['momentum']=max(numerical_max['momentum'],report['momentum_residual'])
        if complete:ck(f'{uid}/child_full_state/{eid}',a['child_state'].shape==(836,) and a['child_state'][0]==a['time'][-1] and np.array_equal(a['child_state'][1:1+nq],a['qpos'][-1]) and np.array_equal(a['child_state'][1+nq:1+nq+nv],a['qvel'][-1]) and np.array_equal(a['child_state'][1+nq+nv:],n['state'][parent,1+nq+nv:]))
        else:ck(f'{uid}/no_collision_child/{eid}',a['child_state'].size==0)
        if eid in edge_to_node:
            ni=edge_to_node[eid];clear_min=min(clear_min,float(np.min(a['minimum'])));geom_max=max(geom_max,float(np.max(a['geometry_step'])));accepted_edges+=1
            ck(f'{uid}/accepted_node_edge/{eid}',accepted and np.array_equal(a['child_state'],n['state'][ni]) and np.array_equal(a['ee'][-1],n['ee'][ni]) and np.array_equal(a['qpos'][-1],n['qpos'][ni]) and np.array_equal(a['qvel'][-1],n['qvel'][ni]))
        edge_rows.append({'unit':uid,'edge':eid,'parent':parent,'intervals':intervals,'actual_intervals':L-1,'complete':complete,'edge_accepted':accepted,'retained_node':edge_to_node.get(eid),'minimum_clearance_m':float(np.min(a['minimum'])),'max_geometry_step_m':float(np.max(a['geometry_step'])),'quaternion_error_recomputed':quaterr})
    ck(uid+'/coupled_arithmetic',calls['coupled']==37*len(attempts)+len(edge_paths))
    ck(uid+'/call_cap',calls['coupled']<=P['search_coupled_cap'] and 37*256*len(attempts)+reservedints<=P['search_integration_cap'])
    remaining=[i for i in range(count-1,N) if i not in seen and n['elapsed'][i]+1+.35<=27]
    ck(uid+'/stopping_rule',(r['stop']=='PARENT_LIMIT' and len(attempts)==50) or (r['stop']=='NO_UNEXPANDED_TIME_FEASIBLE_PARENT' and not remaining))
    # Numerical duplicates are distinct keys. Measure siblings without merging.
    siblings=[]
    for parent in set(int(x) for x in n['parent'][count:]):
        children=np.flatnonzero(n['parent']==parent);children=children[children>=count]
        for jidx,i in enumerate(children):
            for k in children[jidx+1:]:siblings.append(float(np.max(np.abs(n['q'][i]-n['q'][k]))))
    stats['units'].append({'unit':uid,'prefix_nodes':count,'prefix_end_time_s':float(n['elapsed'][count-1]),'prefix_Q':float(qualities[count-1]),'new_nodes':N-count,'expanded_parents':len(attempts),'stop':r['stop'],'remaining_unexpanded_time_feasible_parents':len(remaining),'best_node':best,'best_Q':float(qualities[best]),'best_time_s':float(n['elapsed'][best]),'best_position_errors_m':(norms[:2]*.005).tolist(),'best_rotation_errors_deg':norms[2:].tolist(),'relative_Q_improvement':float(1-qualities[best]/qualities[count-1]),'minimum_Q_all_nodes':float(qualities.min()),'maximum_new_time_s':float(n['elapsed'][count:].max()),'maximum_new_Q':float(qualities[count:].max()),'edge_propagations':len(edge_paths),'edge_complete_count':complete_edges,'edge_collision_stops':colliding_edges,'edge_retained_count':accepted_edges,'edge_grids':dict(grids),'derivative_queries':37*len(attempts),'coupled':calls['coupled'],'reserved_intervals':37*256*len(attempts)+reservedints,'edge_actual_intervals':actualints,'trial_reasons':dict(trialreasons),'unique_action_keys':len(set(allkeys)),'actual_propagated_actions':len(actual_keys),'accepted_variant_counts':dict(variant_counts),'accepted_scale_counts':dict(accepted_scales),'accepted_from_solver_success_false':accepted_from_failed_solver,'solver_bound_violation_max_rad':bound_max,'accepted_minimum_clearance_m':clear_min,'accepted_max_geometry_step_m':geom_max,'all_edge_max_quaternion_norm_error':quat_max,'reported_numerical_max':numerical_max,'edge_time_max_error':edge_time_errors,'edge_arm_position_max_error':edge_motion_error,'edge_arm_velocity_max_error':edge_velocity_error,'max_trial_velocity_ratio':velocity_ratio,'max_trial_acceleration_ratio':accel_ratio,'smallest_sibling_q_linf_rad':min(siblings) if siblings else None,'sibling_pairs_below_1e_4_rad':sum(v<1e-4 for v in siblings),'sibling_pairs':len(siblings)})

ck('global_call_counters',all(ledger['calls']['contact_aware_prefix_repair.'+k]==v for k,v in all_calls.items()))
ck('global_coupled_accounting',all_calls['coupled']==budget['reserved_coupled']==budget['actual_coupled']==full_queries+len(edge_rows))
ck('global_interval_accounting',reserved_intervals==budget['integration_intervals'])
ck('global_expected_counts',new_count==249 and len(linear_rows)==140 and len(edge_rows)==1179)
ck('global_exact_actions_unique',len(global_actions)==len(set(global_actions)))
ck('global_new_nodes_unique',len(global_nodes)==len(set(global_nodes)))
ck('output_size',sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file())<P['output_bytes_cap'])
stats.update({'production':run['producer_head'],'source_files':len(I['source_hashes']),'bound_inputs':len(I['binding']['files']),'historical_prefix_nodes':prefix_count,'new_nodes':new_count,'linearizations':len(linear_rows),'derivative_queries':full_queries,'edge_propagations':len(edge_rows),'reserved_intervals':reserved_intervals,'edge_actual_intervals':observed_edge_intervals,'total_inferred_completed_integration_intervals':full_queries*256+observed_edge_intervals,'coupled':all_calls['coupled'],'calls':dict(all_calls),'elapsed_s':status['elapsed_s'],'candidate_denominator':3,'execution_denominator':3,'qualified':0,'executed':0,'replay_verified':0,'blocked_opportunities':3,'paper_claim_audit':'unavailable','exact_action_keys':len(global_actions),'active_records':sum(x['active'] for x in linear_rows),'near_records':sum(x['all_near_records'] for x in linear_rows),'omitted_records':sum(x['omitted_count'] for x in linear_rows),'omission_linearizations':sum(x['omitted_count']>0 for x in linear_rows),'omitted_record_identities_saved':False,'half_distance_abs_max':max(x['distance_half_abs_error_m_per_rad'] for x in linear_rows),'half_distance_relative_l2_max':max(x['distance_half_relative_l2_error'] for x in linear_rows),'half_world_relative_l2_max':max(x['world_half_relative_l2_error'] for x in linear_rows),'half_distance_relative_l2_median':float(np.median([x['distance_half_relative_l2_error'] for x in linear_rows])),'solvers':len(solver_rows),'solver_success':sum(x['success'] for x in solver_rows),'solver_linear_admissible':sum(x['linear_admissible'] for x in solver_rows),'solver_failed_but_admissible':sum(not x['success'] and x['linear_admissible'] for x in solver_rows),'solver_messages':dict(collections.Counter(x['message'] for x in solver_rows))})
for p in OUT.rglob('*'):
    if p.is_file():hashed(p,'new run evidence manifest; some media bytes only')
for p in [PKG/'evidence/authorization.json',PKG/'evidence/launch.json',Path('C:/Users/admin/.codex/skills/experiment-result-to-claim/SKILL.md'),Path('C:/Users/admin/.codex/skills/shared-references/review-tracing.md'),ROOT/'.venv/Lib/site-packages/scipy/optimize/_optimize.py',ROOT/'.venv/Lib/site-packages/scipy/optimize/_slsqp_py.py']:
    hashed(p,'review source/instruction input')
output={'status':'PASS' if all(x['passed'] for x in checks) else 'FAIL','scope':'identity and arithmetic over saved evidence, source inspection; no physics, geometry query, propagation, optimization, resume, or replay','checks_total':len(checks),'checks_passed':sum(x['passed'] for x in checks),'failed_checks':[x for x in checks if not x['passed']],'stats':stats}
dump(TRACE/'checker_result.json',output);dump(TRACE/'checker_checks.json',checks);dump(TRACE/'linearization_rows.json',linear_rows);dump(TRACE/'edge_rows.json',edge_rows);dump(TRACE/'solver_rows.json',solver_rows);dump(TRACE/'inputs.json',{'review_started_utc':'2026-10-22','manifest':list(manifest.values()),'old_review_access':'hash only; no old review content interpreted','review_script':str(Path(__file__).resolve())})
print(json.dumps(output,ensure_ascii=False,indent=2,default=plain,allow_nan=False))
sys.exit(0 if output['status']=='PASS' else 1)
