"""Read saved arrays only; no project imports, MuJoCo, integration or optimization."""
from pathlib import Path
import hashlib, json, zipfile
import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent
RUN = ROOT/'artifacts/v7_1/capture_refine_001'
def read(p): return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def arr(p):
    with np.load(p, allow_pickle=False) as a: return {k:a[k] for k in a.files}
def maxerr(a,b): return float(np.max(np.abs(np.asarray(a)-np.asarray(b)),initial=0))
def same(a,b,atol=1e-10): assert maxerr(a,b)<=atol, (maxerr(a,b),atol)
def rotvec(actual,goal): return Rotation.from_matrix(np.swapaxes(goal[...,:3,:3],-1,-2)@actual[...,:3,:3]).as_rotvec()
task=read(RUN/'frozen_task.json'); goals=np.asarray(task['condition']['goals'])
def pose(ee): return np.r_[((ee[:,:3,3]-goals[:,:3,3])/.005).ravel(),(rotvec(ee,goals)/np.deg2rad(1)).ravel()]
def groups(ee): return np.linalg.norm(pose(ee).reshape(4,3),axis=1)
def penalty(d):
    v=np.minimum((d-.0002)/.003,0)**2
    return float(np.mean(v)+.25*np.max(v))
limits=read(RUN/'physical_limits.json'); lower=np.asarray(limits['lower']); upper=np.asarray(limits['upper']); rb=np.asarray(limits['geom_rbound'])
def motion(a): return float(np.max(np.linalg.norm(np.diff(a['geom'],axis=0),axis=-1)+np.linalg.norm(np.diff(a['geom_rotation'],axis=0).reshape(-1,len(rb),3,3),axis=(-1,-2))*rb))
run=read(RUN/'run.json'); ident=run['identity']; bind=read(ROOT/'ff_prior_capture_refine/evidence/binding.json')
source_bad=[p for p,h in ident['source_hashes'].items() if sha(ROOT/p)!=h]
bound_bad=[p for p,h in dict(bind['files'],**bind['previous_source_hashes']).items() if sha(ROOT/p)!=h]
with zipfile.ZipFile(RUN/'executed_source.zip') as z:
    zip_bad=[p for p,h in ident['source_hashes'].items() if hashlib.sha256(z.read(p)).hexdigest()!=h]
manifest=read(ROOT/'ff_prior_capture_refine/evidence/completed/raw_manifest.json')
raw_bad=[p for p,h in manifest['files'].items() if sha(RUN/p)!=h]
assert not source_bad+bound_bad+zip_bad+raw_bad
selection=read(ROOT/'ff_prior_capture_refine/evidence/selection.json')
old=ROOT/'artifacts/v7_1/capture_feasibility_001'
assert ident['selection']==selection and read(old/'frozen_task.json')==task==selection['entry']['task']
assert len(selection['entry']['scene']['objects'])==5
assert read(old/'run.json')['identity']['selection']['entry']==selection['entry']
assert read(ROOT/'artifacts/v7_1/sample_alignment_001/physical_contract.json')['acceptance']==read(RUN/'physical_contract.json')['acceptance']==task['contract']['acceptance']
sanity=arr(RUN/'sanity_uniform.npz'); same(sanity['old_qpos'],sanity['new_qpos'],0);same(sanity['old_ee'],sanity['new_ee'],0)
report={'scope':'saved-array algebra only; no physical recomputation','provenance':{'source_files':len(ident['source_hashes']),'bound_files':len(dict(bind['files'],**bind['previous_source_hashes'])),'raw_manifest_files':len(manifest['files']),'mismatches':source_bad+bound_bad+zip_bad+raw_bad,'producer_head':run['producer_head'],'task_hash':task['task_hash'],'five_obstacles_and_old_entry_exact':True},'M0':{'qpos_max_difference':maxerr(sanity['old_qpos'],sanity['new_qpos']),'ee_max_difference':maxerr(sanity['old_ee'],sanity['new_ee']),'other_arrays_not_saved':['geom','geom_rotation','velocity']},'iterations':[],'teachers':[]}
coupled=2; intervals=1024; accepted_total=0; fd_total=0
for seed in selection['seeds']:
    sid=seed['id']; td=RUN/'teachers'/sid
    oldv=read(old/'teachers'/sid/'validation.json'); savedseed=read(td/'seed.json');initial=read(td/'initial.json')
    same(seed['Z'],oldv['Z'],0); assert seed['source_reference_hash']==oldv['reference_hash']
    same(savedseed['actual_start_Z'],seed['Z'],0);same(initial['Z'],seed['Z'],0)
    coupled+=len(initial['grid_history']);intervals+=sum(v['intervals'] for v in initial['grid_history'])
    preceding=np.asarray(seed['Z']); previous_accepted=None
    for rp in sorted(td.glob('iterations/*/record.json')):
        r=read(rp);a=arr(rp.parent/'linearization.npz');s=r['solver'];same(a['Z'],preceding,0)
        same(a['goal'],pose(a['current_ee']))
        h=a['steps'];assert np.all(abs(h)<=.0001) and np.all(abs(h)>0);fd_total+=len(h)
        perturb=np.asarray([pose(e) for e in a['perturb_ee']]);jg=((perturb-a['goal'])/h[:,None]).T;jd=((a['perturb_distances']-a['active_distances'])/h[:,None]).T
        same(jg,a['jg'],1e-9);same(jd,a['jd'],1e-9)
        for i,x in enumerate(h):
            z=a['Z'].copy().ravel();z[i]+=x;assert np.all(z.reshape(19,17)>=lower) and np.all(z.reshape(19,17)<=upper)
        active=np.asarray(r['active_pairs']);ts=active[:,0];expected=np.unique(np.r_[np.linspace(0,1,513),ts]);same(a['phase'],expected,0)
        assert np.isin(ts,a['dense_phase']).all() and np.isin(ts,a['phase']).all()
        assert len(active)==240 and len(set(map(tuple,active)))==240
        pe=np.linalg.norm(a['current_ee'][:,:3,3]-a['dense_ee'][:,:3,3],axis=1);re=np.linalg.norm(rotvec(a['current_ee'],a['dense_ee']),axis=1)
        same(pe,r['sparse_agreement'][-1]['position_m']);same(re,r['sparse_agreement'][-1]['rotation_rad']);assert max(pe)<=.001 and max(re)<=np.deg2rad(.2)
        coupled+=len(r['sparse_agreement'])+len(h);intervals+=sum(x['actual_intervals'] for x in r['sparse_agreement'])+len(h)*(len(a['phase'])-1)
        limits_g=np.maximum(1,np.linalg.norm(a['goal'].reshape(4,3),axis=1));same(limits_g,s['pose_limits'])
        lo=np.maximum((lower-a['Z']).ravel(),-.15);hi=np.minimum((upper-a['Z']).ravel(),.15)
        def residual(u):
            du=.15*np.asarray(u);g=a['goal']+jg@du
            return {'pose_min':float(np.min(1-np.sum(g.reshape(4,3)**2,axis=1)/limits_g**2)),'control_min':float(np.min(np.r_[du-lo,hi-du])),'maximum_control_step_rad':float(np.max(abs(du)))}
        def objective(u):
            du=.15*np.asarray(u);g=a['goal']+jg@du;v=np.minimum((a['active_distances']-.0002+jd@du)/.003,0)**2
            return float(v.mean()+.25*v.max()+.01*(g@g)+.01*np.mean(np.diff(a['Z']+du.reshape(19,17),n=2,axis=0)**2)+.001*np.mean(du**2))
        norm=max(1,objective(np.zeros(323)));same(norm,s['objective_normalizer'],1e-8)
        residuals={}
        for key,u in [('initial',np.zeros(323)),('solved',s['solved_u']),('returned',s['returned_u'])]:
            rr=residual(u); residuals[key]=rr
            for k,v in rr.items():same(v,s[key+'_residual'][k],1e-8)
        same(a['step'],.15*np.asarray(s['returned_u']),0);assert min(residuals['returned']['pose_min'],residuals['returned']['control_min'])>=-1e-8
        same(objective(np.zeros(323))/norm,s['initial_objective'],1e-10);same(objective(s['returned_u'])/norm,s['returned_objective'],1e-10)
        callbacks=s['callbacks'];assert len(callbacks)==161
        for cb in callbacks:
            assert cb['feasible']==(min(cb['residual']['pose_min'],cb['residual']['control_min'])>=-1e-8)
        best=min([s['initial_objective']]+[cb['objective'] for cb in callbacks if cb['feasible']]);same(best,s['returned_objective'],1e-10)
        beforeg=groups(a['dense_ee']);beforec=penalty(a['dense_minima']);same(beforeg,r['before']['goal_groups']);same(beforec,r['before']['collision_violation']);same(min(a['dense_minima']),r['before']['minimum_clearance_m'])
        if previous_accepted is not None:
            for k,l in [('dense_qpos','qpos'),('dense_phase','phase'),('dense_minima','minima'),('dense_ee','ee')]: same(a[k],previous_accepted[l],0)
        trials=[]
        for ti,t in enumerate(r['trials'],1):
            assert t['status']=='EVALUATED'
            ta=arr(rp.parent/f'trial_{ti:02d}.npz');same(ta['Z'],a['Z']+t['scale']*a['step'].reshape(19,17),0)
            g=groups(ta['ee']);c=penalty(ta['minima']);same(g,t['goal_groups']);same(c,t['collision_violation']);same(min(ta['minima']),t['minimum_clearance_m'])
            assert len(ta['phase'])-1>=len(a['dense_phase'])-1
            ok=not any(x<=1 and y>1 for x,y in zip(beforeg,g)) and (c<beforec-1e-10 if max(beforeg)<=1 else c+.05*sum(pose(ta['ee'])**2)<beforec+.05*sum(pose(a['dense_ee'])**2)-1e-10)
            assert ok==t['accepted'];pairs=read(rp.parent/f'trial_{ti:02d}_pairs.json');assert len(pairs)==len(ta['minima'])
            trials.append({'scale':t['scale'],'accepted':bool(ok),'goal_groups':g.tolist(),'minimum_clearance_m':float(min(ta['minima'])),'penalty':c,'min_pair':pairs[int(np.argmin(ta['minima']))],'max_control_step_rad':float(np.max(abs(ta['Z']-a['Z'])))})
            coupled+=len(t['grid_history']);intervals+=sum(x['intervals'] for x in t['grid_history'])
            if ok: same(a['next_Z'],ta['Z'],0);previous_accepted=ta;accepted_total+=1
        if not r['accepted']:same(a['next_Z'],a['Z'],0)
        preceding=a['next_Z'];report['iterations'].append({'teacher':sid,'iteration':r['iteration'],'accepted':r['accepted'],'success':s['success'],'message':s['message'],'iterations':s['iterations'],'residuals':residuals,'returned_source':s['returned_source'],'callback_count':len(callbacks),'feasible_callbacks':sum(cb['feasible'] for cb in callbacks),'objective_best_verified':best,'jg_max_difference':maxerr(jg,a['jg']),'jd_max_difference':maxerr(jd,a['jd']),'sparse_intervals':len(a['phase'])-1,'dense_intervals':len(a['dense_phase'])-1,'inserted_nodes':len(a['phase'])-513,'sparse_position_error_m':pe.tolist(),'sparse_rotation_error_rad':re.tolist(),'recorded_distance_error_m':r['sparse_agreement'][-1]['distance_difference_m'],'trials':trials})
    final=arr(td/'final_path.npz');same(final['Z'],preceding,0);v=read(td/'validation.json');same(v['Z'],final['Z'],0)
    finalgroups=groups(final['ee']);same(finalgroups,read(td/'result.json')['goal_groups'])
    gridrows=[];previous=None
    for gp in sorted(td.glob('grid_*.npz'),key=lambda p:int(p.stem.split('_')[1])):
        n=int(gp.stem.split('_')[1]);ga=arr(gp);assert all(np.isfinite(x).all() for x in ga.values());assert len(ga['phase'])==n+1;same(ga['phase'],np.linspace(0,1,n+1),0)
        coupled+=1;intervals+=n;m=motion(ga);positions=np.linalg.norm(ga['ee'][-1,:,:3,3]-goals[:,:3,3],axis=1)
        row={'intervals':n,'geometry_step_m':m,'position_errors_m':positions.tolist(),'rotation_errors_rad':np.linalg.norm(rotvec(ga['ee'][-1],goals),axis=1).tolist(),'all_arrays_finite':True}
        if previous is not None:
            prev=previous;center=np.linalg.norm(ga['geom'][::2]-prev['geom'],axis=-1);rot=(ga['geom_rotation'][::2]-prev['geom_rotation']).reshape(len(prev['phase']),len(rb),3,3)
            extent=float(max(np.linalg.norm(ga['body'][::2]-prev['body'],axis=-1).max(),(center+np.linalg.norm(rot,axis=(-1,-2))*rb).max()))
            comp=next(x for x in v['validation']['numerical']['comparisons'] if x['fine']==n);same(extent,comp['aligned_body_extent_m'])
            same(np.linalg.norm(ga['ee'][-1,:,:3,3]-prev['ee'][-1,:,:3,3],axis=1),comp['terminal_position_m']);same(np.linalg.norm(rotvec(ga['ee'][-1],prev['ee'][-1]),axis=1),comp['terminal_rotation_rad'])
            vg=next(x for x in v['validation_grids'] if x['intervals']==n);same(m,vg['geometry']['maximum_geometry_step_m']);assert vg['gates']['geometry_refinement']==(m<=.002 and vg['geometry']['refinement_difference_m']<=.001)
            row.update(aligned_extent_m=extent,stored_clearance_m=vg['geometry']['minimum_clearance_m'],gates=vg['gates'])
        previous=ga;gridrows.append(row)
    for k in ['phase','qpos']:same(final[k],previous[k],0)
    same(final['ee'],previous['ee'][-1],0);same(groups(previous['ee'][-1]),finalgroups)
    same(np.linalg.norm(previous['ee'][-1,:,:3,3]-goals[:,:3,3],axis=1),v['validation']['frozen_goal_position_error_m'])
    assert not v['eligible'] and v['frozen_goal_pass'] and not v['validation']['gates']['geometry']
    report['teachers'].append({'id':sid,'seed_equal_old_final':True,'actual_clipping_change':0,'final_minimum_clearance_m':float(min(final['minima'])),'initial_minimum_clearance_m':initial['minimum_clearance_m'],'goal_groups':finalgroups.tolist(),'final_control_change_rad':maxerr(final['Z'],seed['Z']),'eligible':v['eligible'],'stop':read(td/'result.json')['stop'],'validation_grids':gridrows})
budget=read(RUN/'budget.json');assert coupled==budget['actual_coupled'] and intervals==budget['integration_intervals']
ledger=read(RUN/'ledger.json');assert ledger['inflight'] is None and all(v==0 for k,v in ledger['calls'].items() if k.endswith(('.torque_rollout','.physics_steps')))
assert read(RUN/'teacher_execution_selection.json')['ids']==[] and read(RUN/'representation_BLOCKED.json')['status']=='BLOCKED_NO_MODEL_VALID_WITNESS'
report['counts']={'finite_difference_columns':fd_total,'accepted':accepted_total,'coupled_reconstructed':coupled,'integration_intervals_reconstructed':intervals,'raw_run_bytes':sum(p.stat().st_size for p in RUN.rglob('*') if p.is_file()),'wall_s':read(RUN/'status.json')['elapsed_s'],'executions':0,'eligible':0,'teachers':2}
report['paper_claim_audit']='unavailable' if not (ROOT/'paper/PAPER_CLAIM_AUDIT.json').exists() else read(ROOT/'paper/PAPER_CLAIM_AUDIT.json')
(OUT/'independent_algebra.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({'status':'PASS','counts':report['counts'],'provenance':report['provenance'],'paper_claim_audit':report['paper_claim_audit']},ensure_ascii=False,indent=2))
