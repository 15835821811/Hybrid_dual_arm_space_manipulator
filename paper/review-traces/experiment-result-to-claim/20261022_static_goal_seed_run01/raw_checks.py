"""Independent saved-evidence reconstruction. No physics, optimization or resume.

Only this review directory is written. Numeric formulas are implemented here;
the proposal model is reloaded for pure CPU inference after checking all source
and input hashes. No run audit implementation is invoked.
"""
import os
os.environ['OMP_NUM_THREADS'] = '2'
os.environ['MKL_NUM_THREADS'] = '2'
os.environ['OPENBLAS_NUM_THREADS'] = '2'
import sys
sys.dont_write_bytecode = True
import json, hashlib, zipfile, subprocess, datetime, xml.etree.ElementTree as ET
from pathlib import Path
from collections import Counter
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.interpolate import BSpline

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
RUN = ROOT / 'artifacts/v7_1/static_goal_seed_001'
sys.path.insert(0, str(ROOT))
CHECKS = {}
ERRORS = []
INPUTS = {}

def sha(data): return hashlib.sha256(data).hexdigest()
def raw(path):
    path = Path(path)
    b = path.read_bytes()
    INPUTS[str(path.relative_to(ROOT))] = {'sha256': sha(b), 'bytes':len(b), 'mtime_ns':path.stat().st_mtime_ns}
    return b
def read(path): return json.loads(raw(path).decode('utf-8-sig'))
def arr(path):
    import io
    with np.load(io.BytesIO(raw(path)), allow_pickle=False) as a: return dict(a)
def plain(x):
    if isinstance(x,np.ndarray): return x.tolist()
    if isinstance(x,np.generic): return x.item()
    raise TypeError(type(x).__name__)
def digest(x): return sha(json.dumps(x,default=plain,sort_keys=True,separators=(',',':'),allow_nan=False).encode())
def put(name, ok):
    CHECKS[name]=bool(ok)
    if not ok: ERRORS.append(name)
def close(name,a,b,tol=1e-10): put(name,np.allclose(a,b,rtol=1e-10,atol=tol))
def exact(name,a,b): put(name,np.array_equal(a,b))
def pose(ee, goals):
    ee,goals=np.asarray(ee),np.asarray(goals)
    dp=(ee[:,:3,3]-goals[:,:3,3]).reshape(-1)/.005
    dr=Rotation.from_matrix(goals[:,:3,:3].transpose(0,2,1)@ee[:,:3,:3]).as_rotvec().reshape(-1)/np.deg2rad(1)
    return np.r_[dp,dr]
def norms(e): return np.linalg.norm(np.asarray(e).reshape(4,3),axis=1)
def errors(ee,goals):
    n=norms(pose(ee,goals)); return n[:2]*.005,n[2:]*np.deg2rad(1)
def difference(z,d):
    caps=[]
    for sign in (1.,-1.):
        v=d*sign;c=np.full(v.shape,np.inf)
        np.divide(UPPER-z,v,out=c,where=v>0)
        np.divide(LOWER-z,v,out=c,where=v<0)
        caps.append(c.min())
    return (1 if caps[0]>=caps[1] else -1)*min(.0001,max(caps)*.5) if max(caps)>1e-9 else 0.
def violation(d):
    v=np.minimum((np.asarray(d)-.0002)/.003,0)**2
    return float(v.mean()+.25*v.max()) if len(v) else 0.
def merit(e,d,goal=False): return violation(d)+(.05*float(e@e) if goal else 0.)
def protected(e,n): return bool(np.all((norms(e)>1)|(norms(n)<=1)))
def reference(z):
    C=np.vstack([np.tile(Q0,(3,1)),z[:18],np.tile(z[-1],(3,1))])
    curve=BSpline(KNOTS,C,3,extrapolate=False)
    return C,curve
def quick(z):
    C,b=reference(z);low=C@B.T
    boundary=np.r_[b.derivative(1)(0)/T,b.derivative(2)(0)/T**2,b.derivative(1)(1)/T,b.derivative(2)(1)/T**2]
    motion=True
    for order,lim in [(1,np.r_[np.full(10,.8),np.full(7,1.2)]),(2,np.r_[np.full(10,2.5),np.full(7,4.)])]:
        curve=b.derivative(order);a=curve.c[:len(curve.t)-curve.k-1]/T**order
        motion &= bool(np.all(np.max(abs(a),axis=0)<=lim+1e-12))
    return {'finite':bool(np.isfinite(C).all()),'domain':bool(np.all(C.min(0)>=LOWER) and np.all(C.max(0)<=UPPER)),
        'native_range':bool(np.all(low[:,LIMITED]>=RANGES[LIMITED,0]) and np.all(low[:,LIMITED]<=RANGES[LIMITED,1])),
        'boundary':bool(np.max(abs(boundary))<1e-10 and np.array_equal(C[0],Q0)),
        'motion':motion,'time_hold':bool(T+.5<=30)}

run=read(RUN/'run.json');ident=run['identity'];selection=ident['selection'];P=ident['protocol']
status=read(RUN/'status.json');failure=read(RUN/'failure.json');ledger=read(RUN/'ledger.json');budget=read(RUN/'budget.json')
put('producer_commit',run['producer_head']=='2d0a4d4be4bf328fc39518f5dc0c3f1fc0c2a9e2')
with zipfile.ZipFile(RUN/'executed_source.zip') as z:
    sources={name:z.read(name) for name in z.namelist()}
raw(RUN/'executed_source.zip')
put('source_file_count',len(sources)==len(ident['source_hashes'])==176)
for p,h in ident['source_hashes'].items():
    put('archive_hash/'+p,sha(sources[p])==h)
    put('source_current_hash/'+p,sha(raw(ROOT/p))==h)
# Compare the archived execution bytes with the claimed commit, allowing only
# Git's CRLF checkout conversion, which is separately counted below.
requests=''.join(run['producer_head']+':'+p+'\n' for p in sources).encode()
blob=subprocess.run(['git','cat-file','--batch'],cwd=ROOT,input=requests,stdout=subprocess.PIPE,check=True).stdout
at=0;git_exact=0;git_eol=0
for p,source in sources.items():
    end=blob.index(b'\n',at);header=blob[at:end].split();size=int(header[-1]);body=blob[end+1:end+1+size];at=end+2+size
    if source==body:git_exact+=1
    elif source.replace(b'\r\n',b'\n')==body.replace(b'\r\n',b'\n'):git_eol+=1
    put('git_commit_source/'+p,source.replace(b'\r\n',b'\n')==body.replace(b'\r\n',b'\n'))
put('binding_count',len(ident['binding']['files'])==86)
for p,h in ident['binding']['files'].items(): put('input_hash/'+p,sha(raw(ROOT/p))==h)
put('frozen_selection_file',read(ROOT/'ff_prior_static_goal_seed/evidence/selection.json')==selection)
put('frozen_binding_file',read(ROOT/'ff_prior_static_goal_seed/evidence/binding.json')==ident['binding'])
task=selection['entries'][0]['task'];goals=task['condition']['goals'];Q0=np.asarray(task['condition']['initial_features']);T=task['condition']['duration']
put('public_task_hash',digest({k:v for k,v in task.items() if k!='task_hash'})==task['task_hash']=='e66384242aaacf174ab1d7e642f0ba84b55ea580ab1b62951988409c75115496')
put('application_dev_5_obstacles',selection['task_split']=='APPLICATION_DEV' and len(selection['entries'][0]['scene']['objects'])==5)
put('unchanged_duration',T==27)
physical=read(RUN/'physical_contract.json');limits=read(RUN/'scenes/capture_5_obstacles/limits.json')
LOWER=np.asarray(limits['lower']);UPPER=np.asarray(limits['upper'])
urdf=ROOT/'dual_arm_space_robot_2026/urdf/dual_arm_space_robot_2026.urdf';tree=ET.fromstring(raw(urdf));compiler=tree.find('.//compiler')
meshdir=compiler.attrib.get('meshdir','') if compiler is not None else ''
meshes={(urdf.parent/meshdir/m.attrib['filename']).resolve() for m in tree.findall('.//mesh') if 'filename' in m.attrib}
put('all_referenced_meshes_bound',all(p.relative_to(ROOT).as_posix() in ident['binding']['files'] for p in meshes))
from model_test.angle_convention import continuum_mujoco_map_60_to_10
B=np.zeros((67,17));B[:60,:10]=continuum_mujoco_map_60_to_10();B[60:,10:]=np.eye(7)
joints={j.attrib['name']:j for j in tree.findall('joint')};native_names=physical['identity']['robot']['low_level_joint_names']
RANGES=np.array([[float(joints[n].find('limit').attrib[k]) for k in ('lower','upper')] if joints[n].find('limit') is not None else [-np.inf,np.inf] for n in native_names])
LIMITED=np.array([joints[n].attrib.get('type')!='continuous' and joints[n].find('limit') is not None for n in native_names])
KNOTS=np.r_[np.zeros(4),np.arange(1,21)/21,np.ones(4)]
s=np.r_[np.linspace(0,1,20)[1:-1],1.];basis=np.eye(17)[:,None,:]*(s*s*(3-2*s))[None,:,None]
put('ledger_no_unknown',ledger['inflight'] is None and not ledger.get('unresolved_call_accounting'))
put('ledger_units_done',len(ledger['completed'])==6 and all(v['status']=='DONE' for v in ledger['completed'].values()))
put('global_budget',budget['actual_coupled']==budget['reserved_coupled']==29202 and budget['integration_intervals']==18298744 and budget['actual_native_and_replay_steps']==0)
put('global_limits',budget['actual_coupled']<=P['coupled_cap'] and budget['integration_intervals']<=P['integration_cap'] and status['elapsed_s']<=P['wall_s'])
put('no_native_replay_files',not list(RUN.glob('units/*/execution/*')))
put('postprocess_failure',status['status']=='FAILED' and 'verify_seed' in failure['traceback'] and "trial_e=pose_vector(trial['ee']" in failure['traceback'] and 'list indices' in failure['reason'])
put('failed_audit_unwritten',not (RUN/'independent_verification.json').exists() and not (RUN/'summary.json').exists())
put('no_training_test_expansion',P['training_updates']==P['TRAIN_added']==0 and not P['TEST'] and not P['expand'])

# Pure model reload and inference; safety tripwires prohibit simulator calls.
import mujoco
def prohibited(*a,**k): raise RuntimeError('PHYSICS_FORBIDDEN_IN_READONLY_REVIEW')
for n in ('mj_step','mj_step1','mj_step2','mj_forward','mj_inverse','mj_integratePos','mj_kinematics','mj_geomDistance'):
    setattr(mujoco,n,prohibited)
from ff_prior_static_goal_seed.proposals import load_bundle,make
bundle=load_bundle()
candidate_rows=[];unit_rows=[];seed_iter_count=0;seed_trial_count=0;repair_iter_count=0;repair_accepted=0;max_j_error=0.;max_step_error=0.
call_sums=Counter();reconstructed_calls=0;reconstructed_intervals=0
for spec in selection['units']:
    uid=spec['id'];ud=RUN/'units'/uid;result=read(ud/'result.json');setup=read(ud/'public_setup.json');base=np.asarray(setup['base'])
    call_sums.update(read(ud/'calls.json'));unit_calls=0;unit_intervals=0
    put(uid+'/denominators',result['candidate_denominator']==5 and result['execution_denominator']==1 and len(result['candidates'])==5)
    put(uid+'/no_execution',result['execution_level']=='NOT_EXECUTED' and result['qualified']==0)
    sr=read(ud/'repair_selection.json');ex=read(ud/'execution_selection.json')
    put(uid+'/preselection',sr['before_search'] and sr['ids']==[min(sr['screen_rows'],key=lambda x:(x['rank'],x['id']))['id']])
    put(uid+'/execution_selection',ex['candidate_id'] is None and ex['before_execution'] and ex['candidate_digest']==digest(result['candidates']))
    setup_tuple=(base,np.asarray(setup['jacobian']),np.asarray(setup['frame']),setup['bow_amplitude_m'])
    for row in result['candidates']:
        cid=row['id'];key=f'{uid}/{cid:02d}';cd=ud/'candidates'/f'{cid:02d}';controls=arr(cd/'controls.npz');prov=read(cd/'provenance.json')
        raw_new,proposal_new,meta=make(spec['method'],cid,spec['proposal_seeds'][cid],task,setup_tuple,bundle)
        exact(key+'/model_reload_raw',raw_new,controls['network_raw']);exact(key+'/public_proposal',proposal_new,controls['proposal'])
        put(key+'/provenance_hash',digest(controls['network_raw'])==prov['network_raw_hash'] and digest(controls['proposal'])==prov['proposal_hash'] and digest(base)==prov['public_base_hash'])
        put(key+'/provenance_flags',prov['network_calls']==(16 if spec['method']=='diffusion_geometry' else 0) and not prov['private_solution_used'] and not prov['fallback_to_other_method'] and not prov['new_training'] and prov['sampling_seed']==spec['proposal_seeds'][cid])
        prep=prov['preparation'];zprep=controls['proposal'] if prep['scale']==1 else base+prep['scale']*(controls['proposal']-base)
        exact(key+'/prepared',zprep,controls['prepared']);put(key+'/prepare_quick',quick(zprep)==prep['gates'])
        # All earlier proposed preparation scales must fail if shrink was used.
        prior_scales=[1.,.5,.25,.125,.0625,.03125,0.]
        put(key+'/first_feasible_prepare',all(not all(quick(base+sc*(controls['proposal']-base)).values()) for sc in prior_scales[:prior_scales.index(prep['scale'])]))
        rawtrace=arr(cd/'raw/trace.npz');rawreport=read(cd/'raw/screen.json');raw_e=pose(rawtrace['ee'][-1],goals)
        close(key+'/raw_goal',raw_e,rawreport['goal']);exact(key+'/raw_Z',rawtrace['Z'],controls['network_raw'])
        unit_calls+=1;unit_intervals+=256
        a=arr(cd/'goal_seed/initial.npz');z=a['Z'];ee=a['ee'];e=pose(ee,goals);initial_q=float(max(norms(e)))
        exact(key+'/seed_start',z,zprep);close(key+'/seed_initial_e',e,a['goal']);unit_calls+=1;unit_intervals+=512
        seed_iters=sorted((cd/'goal_seed/iterations').glob('*/step.npz'))
        for stepfile in seed_iters:
            itkey=key+'/seed/'+stepfile.parent.name;v=arr(stepfile);r=read(stepfile.parent/'record.json');seed_iter_count+=1
            exact(itkey+'/before',v['before_Z'],z);exact(itkey+'/ee',v['ee'],ee);close(itkey+'/e',v['goal'],e);exact(itkey+'/directions',v['directions'],basis)
            hs=np.array([difference(z,d) for d in basis]);exact(itkey+'/bounded_epsilon',hs,v['epsilon'])
            J=np.array([(pose(p,goals)-e)/h if h else np.zeros(12) for h,p in zip(hs,v['perturb_ee'])]).T
            close(itkey+'/J',J,v['jacobian'],1e-8);max_j_error=max(max_j_error,float(abs(J-v['jacobian']).max()))
            ds=-np.linalg.solve(J.T@J+.001*np.eye(17),J.T@e);ds*=min(1.,.15/max(abs(ds).max(),1e-12))
            close(itkey+'/DLS_step',ds,v['step'],1e-9);max_step_error=max(max_step_error,float(abs(ds-v['step']).max()))
            unit_calls+=int(np.count_nonzero(hs));unit_intervals+=512*int(np.count_nonzero(hs));delta=np.einsum('i,ijk->jk',v['step'],basis)
            close(itkey+'/beforeQ',max(norms(e)),r['before_Q']);accepted=[]
            put(itkey+'/scales',[x['scale'] for x in r['trials']]==P['line_scales'][:len(r['trials'])])
            for tr in r['trials']:
                seed_trial_count+=1;trial_z=z+tr['scale']*delta;q=quick(trial_z)
                put(itkey+f'/quick/{tr["scale"]}',q==tr['quick'])
                if 'ee' in tr:
                    ne=pose(tr['ee'],goals);close(itkey+f'/trial_e/{tr["scale"]}',ne,tr['goal'])
                    expected=all(q.values()) and protected(e,ne) and float(ne@ne)<float(e@e)-1e-10
                    put(itkey+f'/accept/{tr["scale"]}',expected==tr['accepted']);unit_calls+=1;unit_intervals+=512
                    if tr['accepted']:accepted.append(tr)
                else:put(itkey+f'/quick_rejected/{tr["scale"]}',not all(q.values()) and not tr['accepted'])
            put(itkey+'/one_accept',len(accepted)==int(r['accepted']))
            if accepted:
                tr=accepted[0];z=z+tr['scale']*delta;ee=np.asarray(tr['ee']);e=pose(ee,goals)
            exact(itkey+'/after',z,v['after_Z']);close(itkey+'/afterQ',max(norms(e)),r['after_Q'])
        end=arr(cd/'goal_seed/result.npz');seed_result=read(cd/'goal_seed/result.json')
        exact(key+'/seed_final_Z',end['Z'],z);exact(key+'/seed_final_ee',end['ee'],ee);close(key+'/seed_final_e',end['goal'],e)
        put(key+'/seed_status',seed_result['status']=='SEED_ONLY_NOT_MODEL_VALID' and not seed_result['collision_checked'] and seed_result['iterations']==len(seed_iters) and not seed_result['goal_relabel'])
        st=arr(cd/'seeded/trace.npz');screen=read(cd/'seeded/screen.json');se=pose(st['ee'][-1],goals)
        exact(key+'/seed_screen_controls',st['Z'],z);close(key+'/seed_screen_e',se,screen['goal']);exact(key+'/seed_screen_minima',st['minima'],screen['minima'])
        rank=violation(st['minima'])+.05*float(se@se);close(key+'/rank',rank,row['rank']);unit_calls+=1;unit_intervals+=512
        candidate_rows.append({'unit':uid,'method':spec['method'],'unit_seed':spec['seed'],'candidate':cid,'proposal_seed':spec['proposal_seeds'][cid],
            'raw_Q':float(max(norms(raw_e))),'prepared_Q':initial_q,'seeded_Q':float(max(norms(se))),
            'raw_max_position_mm':float(max(norms(raw_e)[:2])*.005*1000),'prepared_max_position_mm':float(max(norms(a['goal'])[:2])*.005*1000),
            'seeded_max_position_mm':float(max(norms(se)[:2])*.005*1000),'seed_iterations':len(seed_iters),'seeded_clearance_mm':float(min(st['minima'])*1000),
            'rank':rank,'selected_for_full_validation':row['selected_for_repair'],'status':row['status'],'eligible':row['eligible']})
        if not row['selected_for_repair']:
            put(key+'/screen_only',not (cd/'validated').exists() and not (cd/'repair').exists() and row['status']=='GOAL_SEEDED_SCREENED')
            continue
        init=read(cd/'repair/initial.json');exact(key+'/repair_start',init['Z'],z)
        unit_calls+=len(init['grid_history']);unit_intervals+=sum(v['intervals'] for v in init['grid_history'])
        repair_files=sorted((cd/'repair/iterations').glob('*/linearization.npz'))
        for linfile in repair_files:
            lk=key+'/repair/'+linfile.parent.name;v=arr(linfile);r=read(linfile.parent/'record.json');repair_iter_count+=1
            exact(lk+'/before',v['Z'],z);ec=pose(v['current_ee'],goals);ed=pose(v['dense_ee'],goals);close(lk+'/goal',ec,v['goal'])
            J=np.array([(pose(p,goals)-ec)/h if h else np.zeros(12) for h,p in zip(v['steps'],v['perturb_ee'])]).T
            JD=np.array([(p-v['active_distances'])/h if h else np.zeros(len(v['active_distances'])) for h,p in zip(v['steps'],v['perturb_distances'])]).T
            close(lk+'/J',J,v['jg'],1e-8);close(lk+'/JD',JD,v['jd'],1e-8)
            exact(lk+'/epsilon',v['steps'],np.array([difference(z,d) for d in np.eye(323).reshape(323,19,17)]))
            unit_calls+=len(r['sparse_agreement'])+int(np.count_nonzero(v['steps']))
            unit_intervals+=sum(vv['actual_intervals'] for vv in r['sparse_agreement'])+int(np.count_nonzero(v['steps']))*(len(v['phase'])-1)
            solver=r['solver'];step=v['step'];delta=step.reshape(19,17);close(lk+'/solver_step',step,r['radius_rad']*np.asarray(solver['returned_u']))
            put(lk+'/step_domain',np.max(abs(delta))<=r['radius_rad']+1e-8 and np.all(z+delta>=LOWER-1e-8) and np.all(z+delta<=UPPER+1e-8))
            linear_norm=norms(ec+J@step);put(lk+'/linear_pose',np.min(1-linear_norm**2/np.maximum(1,norms(ec))**2)>=-1.1e-8)
            put(lk+'/zero_initial',solver['initialization']=='ZERO_FEASIBLE' and solver['trust_is_control_space'] and solver['variables']==323)
            accepted=[];with_goal=max(norms(ed))>1;pred_before=merit(ec,v['active_distances'],with_goal);actual_before=merit(ed,v['dense_active_distances'],with_goal)
            for j,tr in enumerate(r['trials'],1):
                tz=z+tr['scale']*delta
                if tr['status']=='QUICK_REJECTED':put(lk+f'/quick{j}',quick(tz)==tr['gates'] and not all(tr['gates'].values()));continue
                tv=arr(linfile.parent/f'trial_{j:02d}.npz');exact(lk+f'/trialZ{j}',tv['Z'],tz);ne=pose(tv['ee'],goals)
                physical=protected(ed,ne) and (violation(tv['minima'])<violation(v['dense_minima'])-1e-10 if max(norms(ed))<=1 else merit(ne,tv['minima'],True)<merit(ed,v['dense_minima'],True)-1e-10)
                pr=pred_before-merit(ec+J@(step*tr['scale']),v['active_distances']+JD@(step*tr['scale']),with_goal)
                ar=actual_before-merit(ne,tv['matched_distances'],with_goal);ratio=ar/pr if pr>1e-12 else None
                close(lk+f'/predicted{j}',pr,tr['predicted_reduction'],1e-8);close(lk+f'/actual{j}',ar,tr['actual_reduction'],1e-8)
                put(lk+f'/descent{j}',physical==tr['physical_descent']);put(lk+f'/accepted{j}',tr['accepted']==(physical and ratio is not None and ratio>=.1))
                close(lk+f'/groups{j}',norms(ne),tr['goal_groups']);close(lk+f'/clearance{j}',min(tv['minima']),tr['minimum_clearance_m'])
                unit_calls+=len(tr['grid_history']);unit_intervals+=sum(x['intervals'] for x in tr['grid_history'])
                if tr['accepted']:accepted.append(tv)
            put(lk+'/one_accept',len(accepted)==int(r['accepted']))
            if accepted:z=accepted[0]['Z'];repair_accepted+=1
            exact(lk+'/next',z,v['next_Z']);put(lk+'/hashes',digest(v['Z'])==r['before_Z_hash'] and digest(z)==r['after_Z_hash'])
        rr=read(cd/'repair/result.json');fp=arr(cd/'repair/final_path.npz');val=read(cd/'validated/validation.json');prediction=arr(cd/'validated/prediction.npz')
        exact(key+'/final_repair',z,rr['Z']);exact(key+'/final_path',z,fp['Z']);exact(key+'/validated_controls',z,val['Z'])
        C,_=reference(z);refhash=digest({'C':C,'knots':KNOTS,'degree':3,'T':T,'hold':.5})
        put(key+'/final_reference_hash',val['reference_hash']==rr['reference_hash']==refhash)
        pe,re=errors(prediction['ee'][-1],goals);close(key+'/validated_pos',pe,val['validation']['frozen_goal_position_error_m']);close(key+'/validated_rot',re,val['validation']['frozen_goal_rotation_error_rad'])
        put(key+'/goal_pass',bool(max(pe)<=.005 and max(re)<=np.deg2rad(1))==val['frozen_goal_pass'])
        final_gates=val['validation']['gates'];put(key+'/only_geometry_failure',[k for k,v in final_gates.items() if not v]==['geometry'])
        put(key+'/reject_correct',not val['eligible'] and val['validation']['validation_level']=='REJECTED' and not val['training_eligible'])
        grids=sorted(cd.glob('validated/grid_[0-9]*.npz'));gridnums=sorted(int(p.stem.split('_')[1]) for p in grids if 'geometry' not in p.stem)
        unit_calls+=len(gridnums);unit_intervals+=sum(gridnums)
        last_num=gridnums[-1];finalgrid=arr(cd/f'validated/grid_{last_num}.npz')
        exact(key+'/prediction_grid',prediction['qpos'],finalgrid['qpos']);exact(key+'/prediction_ee',prediction['ee'],finalgrid['ee'])
        for rowgrid in val['validation_grids']:
            n=rowgrid['intervals'];gv=arr(cd/f'validated/grid_{n}.npz');scans=arr(cd/f'validated/grid_{n}_geometry_scans.npz');records=read(cd/f'validated/grid_{n}_geometry_scans.json');geo=rowgrid['geometry']
            expected=np.concatenate([gv['qpos'][::2],gv['qpos'][1::2]])
            exact(key+f'/scan_states{n}',scans['qpos'],expected);exact(key+f'/scan_minima{n}',scans['minima'],[r['minimum_clearance_m'] for r in records])
            close(key+f'/scan_min{n}',min(scans['minima']),geo['minimum_clearance_m']);put(key+f'/scan_penetration{n}',min(scans['minima'])<0 and not rowgrid['gates']['geometry'])
            put(key+f'/geometry_refinement{n}',rowgrid['gates']['geometry_refinement']==(geo['refinement_difference_m']<=.001 and geo['maximum_geometry_step_m']<=.002))
            numeric=read(cd/f'validated/grid_{n}_numerical.json');put(key+f'/numeric{n}',numeric['solve_residual']<1e-10 and numeric['robot_momentum_residual']<1e-10 and numeric['quaternion_norm_error']<1e-12)
        for comp in val['validation']['numerical']['comparisons']:
            c=arr(cd/f'validated/grid_{comp["coarse"]}.npz');f=arr(cd/f'validated/grid_{comp["fine"]}.npz');p,r=errors(f['ee'][-1],c['ee'][-1]);
            close(key+f'/convergence_pos{comp["fine"]}',p,comp['terminal_position_m']);close(key+f'/convergence_rot{comp["fine"]}',r,comp['terminal_rotation_rad'])
            center=max(np.linalg.norm(f['body'][::2]-c['body'],axis=-1).max(),np.linalg.norm(f['geom'][::2]-c['geom'],axis=-1).max())
            put(key+f'/convergence_bounds{comp["fine"]}',center<=comp['aligned_body_extent_m']+1e-12 and comp['passed']==bool(max(p)<=.001 and max(r)<=np.deg2rad(.2) and comp['aligned_body_extent_m']<=.001))
        unit_rows.append({'unit':uid,'method':spec['method'],'seed':spec['seed'],'candidate':cid,'repair_iterations':len(repair_files),'repair_stop':rr['stop'],
            'repair_collision_violation':rr['collision_violation'],'max_position_mm':float(max(pe)*1000),'max_rotation_deg':float(np.rad2deg(max(re))),
            'clearance_mm':val['validation']['geometry']['minimum_clearance_m']*1000,'minimum_pair':val['validation']['geometry']['coarse']['pair'],
            'maximum_geometry_step_mm':val['validation']['geometry']['maximum_geometry_step_m']*1000,'validation_grids':gridnums,'eligible':val['eligible']})
    unit_actual=read(ud/'calls.json');put(uid+'/reconstructed_coupled',unit_calls==unit_actual['coupled']);put(uid+'/unit_call_cap',unit_calls<=P['unit_coupled_cap'])
    unit_rows[-1].update(coupled_calls=unit_calls,integration_intervals=unit_intervals)
    reconstructed_calls+=unit_calls;reconstructed_intervals+=unit_intervals
for key,value in call_sums.items():put('sum_unit_calls/'+key,ledger['calls']['static_goal_seed.'+key]==value)
put('reconstructed_global_coupled',reconstructed_calls==budget['actual_coupled'])
put('reconstructed_global_intervals',reconstructed_intervals==budget['integration_intervals'])
put('reload_inference_240_calls',bundle[0].calls==240)
put('30_candidates_6_full_24_screen',len(candidate_rows)==30 and sum(r['selected_for_full_validation'] for r in candidate_rows)==6)
put('seed_all_margin',all(r['seeded_Q']<=.5 for r in candidate_rows))
put('all_selected_goalpass_geometryfail',len(unit_rows)==6 and all(r['max_position_mm']<=5 and r['max_rotation_deg']<=1 and r['clearance_mm']<0 and not r['eligible'] for r in unit_rows))
totals={'seed_iterations':seed_iter_count,'seed_trials':seed_trial_count,'repair_iterations':repair_iter_count,'repair_accepted':repair_accepted,
    'reconstructed_coupled':reconstructed_calls,'reconstructed_integrations':reconstructed_intervals,'native_and_replay_steps':0,
    'reloaded_network_calls':bundle[0].calls,'maximum_seed_J_reconstruction_error':max_j_error,'maximum_seed_step_reconstruction_error':max_step_error,
    'source_files':len(sources),'git_exact_byte_matches':git_exact,'git_only_eol_matches':git_eol,'bound_inputs':len(ident['binding']['files']),'bound_unique_meshes':len(meshes),
    'run_bytes':sum(p.stat().st_size for p in RUN.rglob('*') if p.is_file()),'elapsed_s':status['elapsed_s']}
put('output_limit',totals['run_bytes']<=P['output_bytes_cap'])
method_summary={}
for method in ('diffusion_geometry','structured_geometry'):
    cs=[r for r in candidate_rows if r['method']==method];us=[r for r in unit_rows if r['method']==method]
    method_summary[method]={'proposal_n':len(cs),'fully_validated_n':len(us),'screen_only_n':len(cs)-len(us),'qualified_n':0,'execution_verified_n':0,'execution_opportunities':3,
        'raw_max_position_mm_median':float(np.median([r['raw_max_position_mm'] for r in cs])),
        'prepared_max_position_mm_median':float(np.median([r['prepared_max_position_mm'] for r in cs])),
        'seeded_max_position_mm_median':float(np.median([r['seeded_max_position_mm'] for r in cs])),
        'seeded_Q_median':float(np.median([r['seeded_Q'] for r in cs])),
        'coupled_calls':sum(r['coupled_calls'] for r in us),'final_clearance_mm_median':float(np.median([r['clearance_mm'] for r in us]))}
report={'status':'PASS' if not ERRORS else 'FAIL','check_count':len(CHECKS),'failed_checks':ERRORS,'checks':CHECKS,'totals':totals,'methods':method_summary,'candidates':candidate_rows,'units':unit_rows,
    'limitations':['No MuJoCo propagation, FK, geometry distance, optimizer, native execution, replay or resume was called.',
        'Geometry distances and geometry support-radius extent are saved simulator outputs: raw scan state ordering, minima, predicates and center bounds are checked, but geometry is not physically recomputed.',
        '16-step model inference was reloaded and exactly matched; this is evidence reconstruction, not a new experiment.',
        'The run remains FAILED; this external review does not overwrite the absent in-run audit or summary.',
        'PAPER_CLAIM_AUDIT absent means the downstream claim verdict is provisional.']}
(HERE/'raw_checks.json').write_text(json.dumps(report,default=plain,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
(HERE/'inputs.json').write_text(json.dumps({'reviewed_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'producer_commit':run['producer_head'],'input_files':INPUTS},indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
print(json.dumps({k:report[k] for k in ['status','check_count','failed_checks','totals','methods','units']},default=plain,indent=2))
