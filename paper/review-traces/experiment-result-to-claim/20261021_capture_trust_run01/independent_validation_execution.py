"""Read stored grids and full native trace; no simulation or optimization."""
import os
os.environ['OPENBLAS_NUM_THREADS']='2'
os.environ['OMP_NUM_THREADS']='2'
import ast,json,hashlib,sys,xml.etree.ElementTree as ET
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from scipy.spatial.transform import Rotation
ROOT=Path(__file__).resolve().parents[4];sys.path.insert(0,str(ROOT))
from ff_prior.spline import Reference
from model_test.angle_convention import continuum_mujoco_map_60_to_10
OUT=ROOT/'artifacts/v7_1/capture_trust_001';DEST=Path(__file__).resolve().parent
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def plain(x):return x.tolist() if isinstance(x,np.ndarray) else x.item() if isinstance(x,np.generic) else str(x)
def digest(x):return hashlib.sha256(json.dumps(x,default=plain,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def pose_errors(a,g):
    a=np.array(a);g=np.array(g);p=np.linalg.norm(a[...,:3,3]-g[...,:3,3],axis=-1)
    return p,Rotation.from_matrix((np.swapaxes(g[...,:3,:3],-1,-2)@a[...,:3,:3]).reshape(-1,3,3)).magnitude().reshape(p.shape)
def close(a,b,atol=1e-12):return np.allclose(a,b,atol=atol,rtol=0)
result={'errors':[],'validations':[]};run=read(OUT/'run.json');ident=run['identity'];task=read(OUT/'frozen_task.json');c=task['condition'];physical=read(OUT/'physical_contract.json');limits=read(OUT/'physical_limits.json');rbound=np.array(limits['geom_rbound'])
def check(ok,msg):
    if not bool(ok):result['errors'].append(msg)
check(digest({k:v for k,v in task.items() if k!='task_hash'})==task['task_hash'],'task hash')
check(ident['selection']['entry']['obstacle_count']==5,'five obstacles')
check(task['contract']['acceptance']==physical['acceptance'],'acceptance identity')
check(physical['acceptance']==read(ROOT/'artifacts/v7_1/sample_alignment_001/physical_contract.json')['acceptance'],'unchanged acceptance')
check(ident['selection']['entry']['origin_proof']==read(OUT/'scene_initial_audit.json'),'initial proof frozen')
index=physical['identity']['joint_index_map'];names=sorted(index,key=lambda name:index[name]['actuator']);qids=[index[n]['qpos'] for n in names];vids=[index[n]['dof'] for n in names]
B=np.zeros((67,17));B[:60,:10]=continuum_mujoco_map_60_to_10();B[60:,10:]=np.eye(7);Bplus=np.linalg.pinv(B)
urdf=ET.parse(ROOT/'dual_arm_space_robot_2026/urdf/dual_arm_space_robot_2026.urdf').getroot();jointmap={j.attrib['name']:j for j in urdf.findall('joint')}
ranges=np.array([[float(jointmap[n].find('limit').attrib[k]) for k in ('lower','upper')] if jointmap[n].find('limit') is not None else [-np.inf,np.inf] for n in names]);velocity=np.r_[np.full(10,.8),np.full(7,1.2)];acceleration=np.r_[np.full(10,2.5),np.full(7,4.)]
def quick(ref):
    lo,hi=ref.bounds();low=ref.control@B.T;bound=np.r_[ref.evaluate(0,1),ref.evaluate(0,2),ref.evaluate(ref.duration,1),ref.evaluate(ref.duration,2)]
    return {'finite':bool(np.isfinite(ref.control).all()),'domain':bool(np.all(lo>=limits['lower']) and np.all(hi<=limits['upper'])),'native_range':bool(np.all(low>=ranges[:,0]) and np.all(low<=ranges[:,1])),'boundary':bool(abs(bound).max()<1e-10 and np.array_equal(ref.control[0],c['initial_features'])),'motion':all(np.all(np.maximum(abs(ref.bounds(order)[0]),abs(ref.bounds(order)[1]))<=limit+1e-12) for order,limit in ((1,velocity),(2,acceleration))),'time_hold':ref.duration+ref.hold<=30 and ref.hold==.5}
def extent(a):return float(np.max(np.linalg.norm(np.diff(a['geom'],axis=0),axis=-1)+np.linalg.norm(np.diff(a['geom_rotation'],axis=0).reshape(-1,len(rbound),3,3),axis=(-1,-2))*rbound))
validations={}
for p in sorted(OUT.glob('**/validation.json')):
    r=read(p);v=r['validation'];key=str(p.parent.relative_to(OUT));ref=Reference(np.array(r['Z']),c['initial_features'],c['duration'])
    check(ref.reference_hash==r['reference_hash'],key+' reference hash');check(r['task_hash']==task['task_hash'] and not r['goal_relabel'],key+' task')
    check(all(v['gates'][k]==value for k,value in quick(ref).items()),key+' quick gates');grids=[];any_penetration=False
    coarse=dict(np.load(p.with_name('grid_256.npz')))
    for i,g in enumerate(r['validation_grids']):
        n=g['intervals'];a=dict(np.load(p.with_name(f'grid_{n}.npz')));scans=read(p.with_name(f'grid_{n}_geometry_scans.json'));sa=dict(np.load(p.with_name(f'grid_{n}_geometry_scans.npz')))
        check(len(a['qpos'])==n+1,key+' grid size');check(np.array_equal(sa['qpos'],np.concatenate([a['qpos'][::2],a['qpos'][1::2]])),key+' scan state order')
        check(np.array_equal(sa['minima'],[s['minimum_clearance_m'] for s in scans]),key+' scan minima');check(all(s['threshold_m']==0 for s in scans),key+' scan threshold')
        geom=g['geometry'];half=n//2+1;mn=float(sa['minima'].min());co=float(sa['minima'][:half].min());mid=float(sa['minima'][half:].min());ex=extent(a)
        check(close([mn,co,mid,ex],[geom['minimum_clearance_m'],geom['coarse']['minimum_clearance_m'],geom['midpoints']['minimum_clearance_m'],geom['maximum_geometry_step_m']]),key+' geometry summaries')
        check(scans[int(np.argmin(sa['minima'][:half]))]['minimum_pair']==geom['coarse']['pair'],key+' coarse pair');check(scans[half+int(np.argmin(sa['minima'][half:]))]['minimum_pair']==geom['midpoints']['pair'],key+' mid pair')
        check(g['gates']['geometry']==(mn>=0),key+' geometry gate');check(g['gates']['geometry_refinement']==(abs(co-mn)<=.001 and ex<=.002),key+' refinement gate')
        pos,rot=pose_errors(a['ee'][-1],coarse['ee'][-1]);body=np.linalg.norm(a['body'][::2]-coarse['body'],axis=-1);center=np.linalg.norm(a['geom'][::2]-coarse['geom'],axis=-1);rotation=(a['geom_rotation'][::2]-coarse['geom_rotation']).reshape(len(coarse['phase']),len(rbound),3,3);aligned=float(max(body.max(),(center+np.linalg.norm(rotation,axis=(-1,-2))*rbound).max()))
        comp=v['numerical']['comparisons'][i];check(close(pos,comp['terminal_position_m']) and close(rot,comp['terminal_rotation_rad']) and close(aligned,comp['aligned_body_extent_m']),key+' convergence arrays')
        numerical=read(p.with_name(f'grid_{n}_numerical.json'));check(g['gates']['coupled_convergence']==(max(pos)<=.001 and max(rot)<=np.deg2rad(.2) and aligned<=.001 and numerical['solve_residual']<1e-10 and numerical['robot_momentum_residual']<1e-10 and numerical['quaternion_norm_error']<1e-12),key+' convergence gate')
        check(close(a['qpos'][0],coarse['qpos'][0],0),key+' initial grid state');check(close(a['qpos'][-1,qids],B@ref.evaluate(ref.duration),1e-10),key+' terminal map')
        grids.append({'intervals':n,'states':len(scans),'minimum_m':mn,'geometry_step_m':ex});any_penetration|=mn<0;coarse=a
    pred=dict(np.load(p.with_name('prediction.npz')))
    check(all(np.array_equal(pred[k],coarse[k]) for k in pred),key+' final prediction')
    pos,rot=pose_errors(pred['ee'][-1],c['goals']);goal=bool(max(pos)<=.005 and max(rot)<=np.deg2rad(1));check(close(pos,v['frozen_goal_position_error_m']) and close(rot,v['frozen_goal_rotation_error_rad']),key+' frozen goal')
    check(r['frozen_goal_pass']==goal,key+' goal gate');check(v['gates']['geometry']==(not any_penetration),key+' retained penetrations');check(r['eligible']==(goal and all(v['gates'].values())),key+' eligibility')
    validations[r['reference_hash']]=r
    result['validations'].append({'path':key,'grids':grids,'eligible':r['eligible'],'position_m':pos,'rotation_rad':rot,'failed_gates':[k for k,b in v['gates'].items() if not b]})
    print(key,'checked',flush=True)
teacher=[(s['id'],read(OUT/'teachers'/s['id']/'validation.json')) for s in ident['selection']['seeds']];selected=read(OUT/'teacher_execution_selection.json')
expected=[sid for sid,r in sorted((v for v in teacher if v[1]['eligible']),key=lambda v:(v[1]['score'],v[0]))[:2]]
check(selected['ids']==expected and selected['candidate_digest']==digest(teacher),'preselection')
exe=OUT/'executions/07_bow_j11_1';report=read(exe/'report.json');a=dict(np.load(exe/'native_trace.npz'));ref=Reference(validations[report['metadata']['reference_hash']]['Z'],c['initial_features'],c['duration']);times=a['time']-a['time'][0];dt=.002;hold=times>=c['duration']-1e-9
check(all(np.isfinite(v).all() for v in a.values()),'native finite');check(a['ctrl'].shape==(13750,67) and a['qpos'].shape==(13751,81),'native shape');check(digest(a['ctrl'])==report['ctrl_hash'],'actual ctrl hash');check(close(np.diff(times),dt) and abs(times[-1]-27.5)<1e-9,'native time')
check(read(exe/'frozen_task.json')==task,'execution frozen task');check(report['initial_state_id']==task['contract']['initial_state_id'] and report['scene_model_hash']==task['contract']['robot_hash'],'execution identity')
q=a['qpos'][:,qids]@Bplus.T;vv=a['qvel'][:,vids]@Bplus.T;aa=np.diff(vv,axis=0)/dt
check(close(q,a['q_actual_17']),'17 map');check(close(a['qpos'][:,qids]-q@B.T,a['manifold_residual']),'manifold array')
check(np.all(q>=limits['lower']) and np.all(q<=limits['upper']) and np.all(a['qpos'][:,qids]>=ranges[:,0]) and np.all(a['qpos'][:,qids]<=ranges[:,1]),'native joint bounds')
check(np.all(abs(vv)<=velocity) and np.all(abs(aa)<=acceleration),'native derivative bounds');check(np.all(abs(a['qvel'][:,vids])<=abs(B)@velocity) and np.all(abs(np.diff(a['qvel'][:,vids],axis=0)/dt)<=abs(B)@acceleration),'native low derivative')
for order,key in ((0,'reference_q'),(1,'reference_qd'),(2,'reference_qdd')):check(close(ref.evaluate(np.arange(13750)*dt,order),a[key]),'reference '+key)
check(close(float(abs(q-ref.evaluate(times)).max()),report['tracking_max_rad']),'tracking');check(close(float(abs(a['manifold_residual']).max()),report['manifold_max_rad']),'manifold max')
torque=np.r_[np.full(60,4.),np.full(7,80.)];check(np.all(abs(a['raw_torque'])<=torque+1e-12) and close(a['ctrl'],np.clip(a['raw_torque'],-torque,torque)),'raw torque actual ctrl')
pos,rot=pose_errors(a['ee'],c['goals']);check(close(pos[hold].max(axis=0),report['hold_max_position_error_m']) and close(rot[hold].max(axis=0),report['hold_max_rotation_error_rad']),'native hold');check(max(pos[hold].max(axis=0))<=.005 and max(rot[hold].max(axis=0))<=np.deg2rad(1),'native goal gate')
momentum=a['base_generalized_momentum'];peaks=[float(np.linalg.norm(momentum[:,sl],axis=1).max()) for sl in (slice(0,3),slice(3,6))];check(close(peaks,report['momentum_peak_linear_angular']),'momentum peaks');check(np.all(np.array(peaks)<=report['momentum_threshold_linear_angular']),'momentum gate')
check(close(np.array(report['momentum_exchange_scale_linear_angular'])*.002+1e-5,report['momentum_threshold_linear_angular']),'momentum threshold')
check(a['clearance_m'].shape==(13751,) and min(a['clearance_m'])>=0,'native clearance path');check(report['minimum_clearance_m']<=min(a['clearance_m']),'midpoint-inclusive clearance')
replay=report['replay'];check(replay['passed'] and replay['physics_steps']==13750 and all(v<=1e-10 for v in replay['maximum_errors'].values()) and replay['copies_actual_state_each_step'] is False,'saved replay record')
check(all(report['gates'].values()) and report['metadata']['robot_state_writes_during_step_loop']==0 and not report['metadata']['external_wrench_applied'],'execution metadata gates')
check(close(a['qpos'][0],np.load(OUT/'teachers/07_bow_j11_1/grid_256.npz')['qpos'][0],0),'native initial qpos')
check(np.all(a['qvel'][0]==0),'native initial stationary')
result['execution']={'native_steps':13750,'saved_native_states':13751,'duration_with_hold_s':float(times[-1]),'ctrl_shape':a['ctrl'].shape,'max_raw_torque_by_group':[float(abs(a['raw_torque'][:,:60]).max()),float(abs(a['raw_torque'][:,60:]).max())],'torque_limits_by_group':[4,80],'hold_position_m':pos[hold].max(axis=0),'hold_rotation_rad':rot[hold].max(axis=0),'minimum_native_saved_clearance_m':float(min(a['clearance_m'])),'minimum_native_plus_midpoint_report_m':report['minimum_clearance_m'],'report_checked_states':report['checked_states'],'replay_report':replay,'files':[p.name for p in exe.iterdir()],'replay_second_trace_saved':False,'midpoint_clearance_array_saved':False,'complete_integration_initial_vector_saved_in_this_run':False}
no_candidates=[]
for p in OUT.glob('representation/**/execution/outcome.json'):
    outcome=read(p);sel=read(p.with_name('selection.json'));check(outcome['status']=='NO_CANDIDATE' and not sel['eligible'] and not validations[sel['reference_hash']]['eligible'],'representation no candidate');check(not p.with_name('native_trace.npz').exists(),'no trace for no candidate');no_candidates.append(str(p.parent.relative_to(OUT)))
result['no_candidate_records']=no_candidates
source=read(OUT/'representation_source.json');teacher_z=np.array(read(OUT/'teachers/07_bow_j11_1/validation.json')['Z']);pure=np.array(read(OUT/'representation/pure_codec/validation.json')['Z']);prep=np.array(read(OUT/'representation/deployed_prepare/validation.json')['Z']);repair=np.array(read(OUT/'representation/codec_repair/validation.json')['Z']);basis=np.array(read(ROOT/'artifacts/v7_1/sample_alignment_001/learning/aligned/codec.json')['basis'])
check(source['teacher_id']==expected[0] and source['before_execution_rank']==0,'representation preexecution source');check(close(abs(pure-teacher_z).max(),source['projection_max_rad']) and close(abs(prep-teacher_z).max(),source['prepare_max_rad']),'representation displacements')
check(abs((np.eye(19)-basis@basis.T)@(prep-pure)).max()<1e-12 and abs((np.eye(19)-basis@basis.T)@(repair-prep)).max()<1e-12,'representation same affine 102 subspace')
result['representation']={'source':source,'pure_prepare_same_affine_subspace_residual':float(abs((np.eye(19)-basis@basis.T)@(prep-pure)).max()),'repair_same_affine_subspace_residual':float(abs((np.eye(19)-basis@basis.T)@(repair-prep)).max()),'initializer_base_saved':False}
ledger=read(OUT/'ledger.json');budget=read(OUT/'budget.json');check(ledger['inflight'] is None and not ledger.get('unresolved_call_accounting'),'unknown ledger');check(sum(v for k,v in ledger['calls'].items() if k.endswith('.coupled'))==budget['actual_coupled']<=12000,'coupled budget');check(sum(v for k,v in ledger['calls'].items() if k.endswith('.physics_steps'))==27500,'physics budget');check(budget['integration_intervals']<=24000000,'integration budget');size=sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file());check(size<=12*1024**3,'disk budget')
result['budget']={'coupled':budget['actual_coupled'],'integration_intervals':budget['integration_intervals'],'physics_steps':27500,'disk_bytes':size,'elapsed_s':read(OUT/'status.json')['elapsed_s'],'inflight':ledger['inflight'],'unknown_work':bool(ledger.get('unresolved_call_accounting')),'calls':ledger['calls']}
result['paper_claim_audit']='unavailable' if not (ROOT/'paper/PAPER_CLAIM_AUDIT.json').exists() else read(ROOT/'paper/PAPER_CLAIM_AUDIT.json')
(DEST/'independent_validation_execution.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,default=plain),encoding='utf-8')
print(json.dumps({'errors':result['errors'],'execution':result['execution'],'budget':result['budget']},default=plain))
