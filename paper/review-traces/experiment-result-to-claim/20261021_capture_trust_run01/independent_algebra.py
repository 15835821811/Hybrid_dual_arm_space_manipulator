"""Fresh saved-evidence inspection; no simulator imports or optimizer calls."""
import os
os.environ['OPENBLAS_NUM_THREADS']='2'
os.environ['OMP_NUM_THREADS']='2'
os.environ['MKL_NUM_THREADS']='2'
import ast,json,hashlib,zipfile
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from scipy.spatial.transform import Rotation
ROOT=Path(__file__).resolve().parents[4]
OUT=ROOT/'artifacts/v7_1/capture_trust_001'
DEST=Path(__file__).resolve().parent
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def plain(x):return x.tolist() if isinstance(x,np.ndarray) else x.item() if isinstance(x,np.generic) else str(x)
def digest(x):return hashlib.sha256(json.dumps(x,default=plain,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
run=read(OUT/'run.json');ident=run['identity'];P=ident['protocol'];task=read(OUT/'frozen_task.json')
env={'np':np,'Rotation':Rotation,'P':P}
def extract(path,names):
    tree=ast.parse((ROOT/path).read_text(encoding='utf-8-sig'))
    for n in tree.body:
        if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in names:
            exec(compile(ast.Module(body=[n],type_ignores=[]),path,'exec'),env)
extract('ff_prior_fixed/correction.py',['pose_vector'])
extract('ff_prior_feasible/guidance.py',['group_norms'])
extract('ff_prior_capture_feasibility/solver.py',['basis','admissible_difference'])
extract('ff_prior_capture_trust/solver.py',['restore_feasible','metric','accept','active_merit','trust_decision'])
extract('ff_prior/physical_audit.py',['pose_errors'])
globals().update({k:v for k,v in env.items() if callable(v)})
codec=SimpleNamespace(basis=np.array(read(ROOT/'artifacts/v7_1/sample_alignment_001/learning/aligned/codec.json')['basis']))
limits=read(OUT/'physical_limits.json');lower=np.array(limits['lower']);upper=np.array(limits['upper'])
result={'scope':'fresh read-only algebra; no new physics, no mj calls, no optimization','errors':[],'alpha_mismatches':[],'iterations':[]}
def check(ok,msg):
    if not bool(ok): result['errors'].append(msg)
def close(a,b,atol=1e-12,rtol=0):return np.allclose(a,b,atol=atol,rtol=rtol)
check(task==ident['selection']['entry']['task'],'frozen task')
with zipfile.ZipFile(OUT/'executed_source.zip') as archive:
    for p,h in ident['source_hashes'].items():
        check(hashlib.sha256(archive.read(p)).hexdigest()==h,'archive '+p)
        check(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h,'current '+p)
for p,h in dict(ident['binding']['files'],**ident['binding']['previous_source_hashes']).items():check(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h,'binding '+p)
result['source_count']=len(ident['source_hashes']);result['binding_count']=len(ident['binding']['files']);result['previous_source_count']=len(ident['binding']['previous_source_hashes'])
result['m0']={}
with np.load(OUT/'sanity_uniform.npz') as a:
    for k in ('qpos','ee','geom','geom_rotation','velocity'):
        error=float(np.max(abs(a['old_'+k]-a['new_'+k])))
        result['m0'][k]={'shape':a['old_'+k].shape,'max_error':error};check(error<1e-12,'m0 '+k)
for s in ident['selection']['seeds']:
    previous=read(ROOT/'artifacts/v7_1/capture_refine_001/teachers'/s['id']/'validation.json')
    seed=read(OUT/'teachers'/s['id']/'seed.json')
    check(np.array_equal(s['Z'],previous['Z']) and np.array_equal(s['Z'],seed['actual_start_Z']),'exact previous final '+s['id'])
previous_radius={};previous_z={};total_callbacks=0
for p in sorted(OUT.glob('**/iterations/*/record.json')):
    key=str(p.parent.relative_to(OUT));r=read(p);s=r['solver'];a=dict(np.load(p.with_name('linearization.npz')))
    z=a['Z'];directions=basis(r['direction_basis'],codec);n=len(directions);radius=r['radius_rad'];e=pose_vector(a['current_ee'],task['condition']['goals'])
    check(close(e,a['goal']),key+' goal');check(close(s['pose_limits'],np.maximum(1,group_norms(e))),key+' pose limits')
    for j,h in enumerate(a['steps']):
        check(h==admissible_difference(z,directions[j],lower,upper),key+f' difference domain {j}')
        check(close((pose_vector(a['perturb_ee'][j],task['condition']['goals'])-e)/h,a['jg'][:,j],1e-10,1e-10) if h else not np.any(a['jg'][:,j]),key+f' jg {j}')
        check(close((a['perturb_distances'][j]-a['active_distances'])/h,a['jd'][:,j],1e-10,1e-10) if h else not np.any(a['jd'][:,j]),key+f' jd {j}')
    M=directions.reshape(n,-1).T;matrix=M*radius;G=a['jg']*radius;D=a['jd']*radius/.003;b=(a['active_distances']-.0002)/.003
    lo=np.maximum((lower-z).ravel(),-radius);hi=np.minimum((upper-z).ravel(),radius);L=np.array(s['pose_limits']);C=np.diff(directions,n=2,axis=1).reshape(n,-1).T*radius;c0=np.diff(z,n=2,axis=0).ravel()
    def residual(u):
        pp=1-np.sum((e+G@u).reshape(4,3)**2,axis=1)/(L*L);change=matrix@u
        return np.array([pp.min(),min((change-lo).min(),(hi-change).min()),abs(change).max()])
    def feasible(u):return np.isfinite(u).all() and residual(u)[:2].min()>=-1e-8
    def audit_feasible(u):
        pp=1-np.sum((e+a['jg']@(radius*u)).reshape(4,3)**2,axis=1)/(L*L);change=M@(radius*u)
        return np.isfinite(u).all() and min(pp.min(),(change-lo).min(),(hi-change).min())>=-1e-8
    def rawfun(u):
        g=e+G@u;v=np.minimum(b+D@u,0.)**2
        return (float(v.mean()+.25*v.max()) if len(v) else 0)+.01*float(g@g)+.01*float(np.mean((c0+C@u)**2))+.001*float(np.mean((matrix@u)**2))
    normalizer=max(1,rawfun(np.zeros(n)));best=np.zeros(n);bestvalue=rawfun(best)/normalizer;exact_alpha_max=0.;audit_alpha_max=0.;min_restored=0.
    for j,cb in enumerate(s['callbacks']):
        u=np.array(cb['u']);restored,alpha=restore_feasible(u,feasible);ar,aa=restore_feasible(u,audit_feasible)
        exact_alpha_max=max(exact_alpha_max,abs(alpha-cb['restoration_alpha']));audit_alpha_max=max(audit_alpha_max,abs(aa-cb['restoration_alpha']))
        saved=cb['restoration_alpha']*u;res=residual(saved);min_restored=min(min_restored,float(res[:2].min()))
        check(feasible(saved),key+f' saved restored infeasible {j}')
        check(close(res,[cb['restored_residual'][k] for k in ('pose_min','control_min','maximum_control_step_rad')],1e-12),key+f' restored residual {j}')
        check(abs(alpha-cb['restoration_alpha'])<1e-12,key+f' execution alpha {j}')
        check(close(rawfun(u)/normalizer,cb['objective'],1e-11),key+f' callback objective {j}')
        rv=rawfun(restored)/normalizer;check(close(rv,cb['restored_objective'],1e-11),key+f' restored objective {j}')
        if feasible(restored) and rv<bestvalue:best=restored.copy();bestvalue=rv
        if abs(aa-cb['restoration_alpha'])>=1e-12:result['alpha_mismatches'].append({'path':key,'callback':j,'recorded_alpha':cb['restoration_alpha'],'execution_alpha':alpha,'audit_alpha':aa,'delta':aa-cb['restoration_alpha'],'saved_residual':res.tolist(),'saved_step_change':float(np.max(abs(matrix@(ar-saved))))})
    total_callbacks+=len(s['callbacks']);check(close(best,s['returned_u'],1e-13),key+' best incumbent');check(feasible(np.array(s['returned_u'])),key+' returned feasible')
    step=a['step'];delta=np.einsum('i,ijk->jk',step,directions)
    check(close(step,radius*np.array(s['returned_u']),1e-14),key+' step');check(np.max(abs(delta))<=radius+1e-8,key+' trust');check(np.all(z+delta>=lower-1e-8) and np.all(z+delta<=upper+1e-8),key+' domain')
    parent=str(p.parents[2]);check(radius==previous_radius.get(parent,P['trust_initial_rad']),key+' radius continuity');previous_radius[parent]=r['next_radius_rad']
    if parent in previous_z:check(np.array_equal(previous_z[parent],z),key+' Z continuity')
    previous_z[parent]=a['next_Z'];check(digest(z)==r['before_Z_hash'] and digest(a['next_Z'])==r['after_Z_hash'],key+' hashes')
    agreement=r['sparse_agreement'][-1];pos,rot=pose_errors(a['current_ee'],a['dense_ee'])
    check(close(pos,agreement['position_m']) and close(rot,agreement['rotation_rad']),key+' sparse pose');check(max(pos)<=.001 and max(rot)<=P['sparse_agreement_rad'],key+' sparse limit')
    diff=np.max(abs(a['active_distances']-a['dense_active_distances']),initial=0);check(close(diff,agreement['distance_difference_m']) and diff<=.001,key+' sparse distance')
    check(all(t in a['phase'] and t in a['dense_phase'] for t,_,_ in r['active_pairs']),key+' active phases')
    accepted=[];evaluated=0
    old={'goal':pose_vector(a['dense_ee'],task['condition']['goals']),'minima':a['dense_minima']};with_goal=metric(old)['Q']>1
    for j,t in enumerate(r['trials'],1):
        if t['status']!='EVALUATED':continue
        evaluated+=1;v=dict(np.load(p.parent/f'trial_{j:02d}.npz'));pairs=read(p.parent/f'trial_{j:02d}_pairs.json')
        check(len(pairs)==len(v['minima'])==len(v['qpos'])==len(v['phase']),key+' trial scan count')
        check(np.array_equal(v['Z'],z+t['scale']*delta),key+' trial Z')
        trial={'goal':pose_vector(v['ee'],task['condition']['goals']),'minima':v['minima']}
        predicted=active_merit(e,a['active_distances'],with_goal)-active_merit(e+a['jg']@(step*t['scale']),a['active_distances']+a['jd']@(step*t['scale']),with_goal)
        actual=active_merit(old['goal'],a['dense_active_distances'],with_goal)-active_merit(trial['goal'],v['matched_distances'],with_goal)
        ratio=actual/predicted if predicted>1e-12 else None
        check(close(predicted,t['predicted_reduction'],1e-10) and close(actual,t['actual_reduction'],1e-10),key+' ratio inputs')
        check((ratio is None and t['ratio'] is None) or (ratio is not None and close(ratio,t['ratio'],1e-10)),key+' ratio')
        ok=accept(old,trial) and ratio is not None and ratio>=.1
        check(bool(accept(old,trial))==t['physical_descent'] and bool(ok)==t['accepted'],key+' acceptance')
        check(t['grid_history'][-1]['maximum_geometry_step_m']<=.002,key+' trial density')
        if ok:accepted.append(t)
    check(bool(accepted)==r['accepted'],key+' accept record');last=accepted[0] if accepted else {}
    check(np.array_equal(a['next_Z'],z+last['scale']*delta if accepted else z),key+' next Z')
    check(r['next_radius_rad']==trust_decision(radius,bool(accepted),last.get('ratio'),last.get('scale'),abs(delta).max()>=.9*radius),key+' radius rule')
    result['iterations'].append({'path':key,'columns':n,'accepted':bool(accepted),'evaluated_trials':evaluated,'radius':radius,'next_radius':r['next_radius_rad'],'callbacks':len(s['callbacks']),'exact_alpha_max_error':exact_alpha_max,'audit_alpha_max_error':audit_alpha_max,'minimum_saved_restored_residual':min_restored,'returned_max_control_step':float(abs(delta).max())})
    print(key,'checked',flush=True)
result['total_callbacks']=total_callbacks
(DEST/'independent_algebra.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,default=plain),encoding='utf-8')
print(json.dumps({'errors':result['errors'],'alpha_mismatches':result['alpha_mismatches'],'iterations':len(result['iterations']),'columns':sum(r['columns'] for r in result['iterations']),'callbacks':total_callbacks},default=plain))
