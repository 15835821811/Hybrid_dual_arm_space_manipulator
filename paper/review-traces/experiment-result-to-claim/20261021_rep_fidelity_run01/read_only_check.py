"""Independent saved-evidence algebra only. No project imports or MuJoCo calls."""
import ast, hashlib, json, zipfile
from pathlib import Path
import numpy as np
from scipy.interpolate import BSpline
from scipy.spatial.transform import Rotation, Slerp

ROOT=Path(__file__).resolve().parents[4]
OUT=Path(__file__).resolve().parent
RUN=ROOT/'artifacts/v7_1/rep_fidelity_001'
def read(p): return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def digest(x): return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False,default=lambda v:v.tolist()).encode()).hexdigest()
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
checks=[]
def check(name,ok,details=None):
    checks.append({'name':name,'passed':bool(ok),'details':details})
def close(a,b,tol=1e-12): return np.allclose(a,b,atol=tol,rtol=0)
run=read(RUN/'run.json'); ident=run['identity']; sources=ident['source_hashes']
check('source_hashes',all(sha(ROOT/p)==h for p,h in sources.items()),len(sources))
with zipfile.ZipFile(RUN/'executed_source.zip') as z:
    check('executed_zip',set(z.namelist())==set(sources) and all(hashlib.sha256(z.read(p)).hexdigest()==h for p,h in sources.items()),len(z.namelist()))
for group,mapping in ident['binding'].items():
    check('binding_'+group,all((ROOT/p).is_file() and sha(ROOT/p)==h for p,h in mapping.items()),len(mapping))
# Independently traverse static local imports, without importing their code.
pending=list((ROOT/'ff_prior_rep_fidelity').glob('*.py')); found=set()
def add(module):
    bits=module.split('.')
    for i in range(1,len(bits)+1):
        p=ROOT.joinpath(*bits[:i])
        for q in [p.with_suffix('.py'),p/'__init__.py']:
            if q.is_file() and q not in found: pending.append(q)
while pending:
    p=pending.pop()
    if p in found: continue
    found.add(p); pkg=p.relative_to(ROOT).with_suffix('').parts[:-1]
    for n in ast.walk(ast.parse(p.read_text(encoding='utf-8-sig'))):
        if isinstance(n,ast.Import):
            for v in n.names:add(v.name)
        if isinstance(n,ast.ImportFrom):
            prefix=list(pkg[:len(pkg)-n.level+1]) if n.level else []
            module='.'.join(prefix+([n.module] if n.module else [])); add(module)
            for v in n.names:
                if v.name!='*':add(module+'.'+v.name)
found.add(ROOT/'ff_prior_rep_fidelity/job.ps1')
check('static_local_import_closure',set(sources)=={p.relative_to(ROOT).as_posix() for p in found},len(found))
task=read(RUN/'frozen_task.json'); c=task['condition']; q0=np.array(c['initial_features']); goals=np.array(c['goals']); T=c['duration']
check('task_exact_old',task==read(ROOT/'artifacts/v7_1/capture_trust_001/frozen_task.json'))
limits=read(RUN/'physical_limits.json'); limits={k:np.array(v) for k,v in limits.items()}
inp=np.load(RUN/'representation_inputs.npz'); base=inp['base']; teacher=inp['teacher']; b0=inp['original_basis']
check('teacher_exact_old',np.array_equal(teacher,ident['selection']['teacher']['Z']))
check('base_shape_public_initializer',close(base[:-1],q0+(np.linspace(0,1,20)[1:-1]**2*(3-2*np.linspace(0,1,20)[1:-1]))[:,None]*(base[-1]-q0)))
check('rank6_exact_bound_codec',np.array_equal(b0,read(ROOT/'artifacts/v7_1/sample_alignment_001/learning/aligned/codec.json')['basis']))
def basis(r):
    if r==19:return np.eye(19)
    v=list(b0.T.copy())
    for candidate in [np.eye(19)[-1]]+[np.cos(np.pi*(np.arange(19)+.5)*k/19) for k in range(19)]:
        if len(v)==r:break
        x=candidate.copy()
        for _ in range(2):
            for a in v:x-=a*np.dot(a,x)
        if np.linalg.norm(x)>1e-10:v.append(x/np.linalg.norm(x))
    return np.stack(v,axis=1)
def project(z,b):return z.copy() if b.shape[1]==19 else base+b@(b.T@(z-base))
knots=np.r_[np.zeros(4),np.arange(1,21)/21,np.ones(4)]
def decode(z):return np.vstack([np.repeat(q0[None],3,axis=0),z[:18],np.repeat(z[-1:],3,axis=0)])
def curves(z):
    a=BSpline(knots,decode(z),3,extrapolate=False);return [a,a.derivative(1),a.derivative(2)]
def quick(z):
    cv=curves(z); ctl=decode(z); low=ctl@limits['B'].T; limited=limits['limited'].astype(bool); rg=limits['ranges']
    motion=all(np.all(np.abs(cv[o].c[:len(cv[o].t)-cv[o].k-1])/T**o<=limits[k]+1e-12) for o,k in [(1,'velocity'),(2,'acceleration')])
    return dict(finite=np.isfinite(ctl).all(),domain=np.all(ctl>=limits['lower']) and np.all(ctl<=limits['upper']),native_range=np.all(low[:,limited]>=rg[limited,0]) and np.all(low[:,limited]<=rg[limited,1]),boundary=np.max(np.abs(np.r_[cv[1](0)/T,cv[2](0)/T**2,cv[1](1)/T,cv[2](1)/T**2]))<1e-10 and np.array_equal(ctl[0],limits['q0']),motion=motion,time_hold=T+.5<=30)
def pose(actual,goal):
    p=np.linalg.norm(actual[...,:3,3]-goal[...,:3,3],axis=-1)
    rot=actual[...,:3,:3]@np.swapaxes(goal[...,:3,:3],-1,-2)
    r=Rotation.from_matrix(rot.reshape(-1,3,3)).magnitude().reshape(p.shape)
    return p,r
manifest=read(RUN/'candidate_manifest.json'); rows=manifest['candidates']; metrics=[]; origins={}; unique={}; new_intervals=[]
check('manifest_digest',digest(rows)==manifest['digest'])
check('12_fixed_logical',[(x['rank'],x['mode']) for x in rows]==[(r,m) for r in [6,10,14,19] for m in ['raw','legacy_prepare','preserve_prepare']])
for row in rows:
    name=row['id']; d=RUN/'candidates'/name; a=np.load(d/'representation.npz'); r=row['rank']; b=basis(r); raw=project(teacher,b)
    check(name+'_basis',close(a['basis'],b) and close(b.T@b,np.eye(r)))
    expected={'teacher':teacher,'base':base,'basis':b,'raw':raw,'input':raw}
    if row['mode']=='legacy_prepare':
        ctl=decode(raw); D=np.diff(np.eye(24),n=2,axis=0); A=D[:,3:21]; fixed=ctl.copy();fixed[3:21]=0
        sm=raw.copy();sm[:18]=np.linalg.solve(np.eye(18)+10*A.T@A,raw[:18]-10*A.T@(D@fixed))
        cl=np.clip(sm,limits['lower'],limits['upper']); pj=project(cl,b); out=pj.copy() if row['metadata']['shrink']==1 else project(base+row['metadata']['shrink']*(pj-base),b)
        expected.update(smoothed=sm,clipped=cl,projected=pj,output=out)
    else:expected['output']=raw
    check(name+'_all_intermediate_arrays',all(k in expected and close(a[k],expected[k]) for k in a.files),list(a.files))
    check(name+'_candidate_exact',np.array_equal(a['output'],row['Z']) and read(d/'candidate.json')==row)
    q=quick(a['output']);check(name+'_quick_gates',all(bool(q[k])==v for k,v in row['quick_gates'].items()))
    rh=digest({'C':decode(a['output']),'knots':knots,'degree':3,'T':T,'hold':.5});check(name+'_ref_hash',rh==row['reference_hash'])
    check(name+'_basis_membership',close(a['output'],project(a['output'],b)))
    v=read(d/'validation.json'); origin=read(d/'validation_origin.json'); origins[name]=origin; unique.setdefault(rh,name)
    if origin['kind']=='same_run_exact_duplicate':
        src=RUN/'candidates'/origin['owner'];check(name+'_dedup_exact',v==read(src/'validation.json') and np.array_equal(a['output'],read(src/'candidate.json')['Z']))
        predsrc=ROOT/origins[origin['owner']]['source'] if origins[origin['owner']]['kind']=='historical_exact_identity' else src
    elif origin['kind']=='historical_exact_identity':
        predsrc=ROOT/origin['source'];check(name+'_historical_exact',v==read(predsrc/'validation.json') and np.array_equal(v['Z'],a['output']) and v['task_hash']==task['task_hash'])
    else:predsrc=d
    pred=np.load(predsrc/'prediction.npz'); pos,rot=pose(pred['ee'][-1],goals)
    check(name+'_endpoint_recomputed',close(pos,v['validation']['frozen_goal_position_error_m']) and close(rot,v['validation']['frozen_goal_rotation_error_rad']))
    goal=bool(np.all(pos<=.005) and np.all(rot<=np.pi/180));check(name+'_eligibility',v['eligible']==(goal and all(v['validation']['gates'].values())) and v['frozen_goal_pass']==goal)
    gridfiles=sorted(predsrc.glob('grid_*.npz')); gridfiles=[p for p in gridfiles if p.stem[5:].isdigit()];gridfiles.sort(key=lambda p:int(p.stem[5:]))
    scans=[];previous=None
    for gp in gridfiles:
        n=int(gp.stem[5:]);g=np.load(gp)
        if origin['kind']=='new_model_validation':new_intervals.append(n)
        if previous is not None:
            cp,cr=pose(g['ee'][-1],previous['ee'][-1]); ext=max(np.linalg.norm(g['body'][::2]-previous['body'],axis=-1).max(),(np.linalg.norm(g['geom'][::2]-previous['geom'],axis=-1)+np.linalg.norm((g['geom_rotation'][::2]-previous['geom_rotation']).reshape(len(previous['phase']),-1,3,3),axis=(-1,-2))*limits['geom_rbound']).max())
            saved=next(x for x in v['validation']['numerical']['comparisons'] if x['fine']==n)
            check(name+f'_convergence_{n}',close(cp,saved['terminal_position_m']) and close(cr,saved['terminal_rotation_rad']) and close(ext,saved['aligned_body_extent_m']))
        previous=g
        sp=predsrc/(gp.stem+'_geometry_scans.npz')
        if sp.exists():
            s=np.load(sp);js=read(sp.with_suffix('.json'));check(name+f'_scan_states_{n}',np.array_equal(s['qpos'],np.concatenate([g['qpos'][::2],g['qpos'][1::2]])))
            check(name+f'_scan_minima_{n}',np.array_equal(s['minima'],[x['minimum_clearance_m'] for x in js]))
            scans.append(float(s['minima'].min()))
    finalscan=np.load(predsrc/(gridfiles[-1].stem+'_geometry_scans.npz'))
    minclear=float(finalscan['minima'].min());check(name+'_reported_clearance',close(minclear,v['validation']['geometry']['minimum_clearance_m']))
    maxstep=float(np.max(np.linalg.norm(np.diff(pred['geom'],axis=0),axis=-1)+np.linalg.norm(np.diff(pred['geom_rotation'],axis=0).reshape(len(pred['phase'])-1,-1,3,3),axis=(-1,-2))*limits['geom_rbound']))
    check(name+'_geometry_density',close(maxstep,v['validation']['geometry']['maximum_geometry_step_m']) and maxstep<=.002)
    metrics.append({'id':name,'position_mm':(pos*1000).tolist(),'rotation_deg':np.rad2deg(rot).tolist(),'clearance_mm':minclear*1000,'eligible':v['eligible'],'frozen_goal_pass':goal,'model_failed_gates':[k for k,x in v['validation']['gates'].items() if not x],'origin':origin['kind'],'quick_all':all(q.values()),'max_control_difference_rad':float(np.max(abs(a['output']-teacher)))})
selection=read(RUN/'execution_selection.json'); expected=[n for n in unique.values() if read(RUN/'candidates'/n/'validation.json')['eligible']]
check('execution_selection',expected==selection['ids']==['r19_raw'] and selection['candidate_digest']==manifest['digest'])
check('8_unique',len(unique)==8)
ex=RUN/'candidates/r19_raw/execution'; native=np.load(ex/'native_trace.npz'); replay=np.load(ex/'replay_trace.npz'); report=read(ex/'report.json')
replayerr={k:float(np.max(abs(native[k]-replay[k]),initial=0)) for k in replay.files}
check('replay_saved_arrays_exact',all(v==0 for v in replayerr.values()),replayerr)
check('native_sizes',native['qpos'].shape==(13751,81) and native['ctrl'].shape==(13750,67))
check('native_complete_time',close(np.diff(native['time']),.002) and abs(native['time'][-1]-native['time'][0]-27.5)<1e-9)
initial=np.load(RUN/'initial_state.npz')['values']; ij=read(RUN/'initial_state.json')
check('full_initial_identity',digest({'values':initial,'model_hash':ij['model_hash'],'state_mask':ij['state_mask']})==ij['state_id']==task['contract']['initial_state_id'] and ij['state_mask']==8191)
check('execution_initial_exact',np.array_equal(initial,np.load(ex/'initial_state.npz')['values']) and ij==read(ex/'initial_state.json'))
check('native_initial_segments',native['time'][0]==initial[0] and np.array_equal(native['qpos'][0],initial[1:82]) and np.array_equal(native['qvel'][0],initial[82:161]))
scans=np.load(ex/'geometry_scans.npz'); sj=read(ex/'geometry_scans.json'); ni=[i for i,x in enumerate(sj) if x['fraction']==0]; mi=[i for i,x in enumerate(sj) if x['fraction']==.5]
check('execution_geometry_native',len(ni)==13751 and np.array_equal(scans['qpos'][ni],native['qpos']))
check('execution_geometry_minima',np.array_equal(scans['minima'],[x['minimum_clearance_m'] for x in sj]) and close(scans['minima'].min(),report['minimum_clearance_m']) and len(mi)==13750)
mid=(native['qpos'][:-1]+native['qpos'][1:])/2
# Two free-body quaternions (robot base and passive satellite), using wxyz storage.
for sl in [slice(3,7),slice(77,81)]:
    qa=native['qpos'][:-1,sl];qb=native['qpos'][1:,sl];sg=np.sign(np.sum(qa*qb,axis=1));sg[sg==0]=1
    qm=qa+qb*sg[:,None];mid[:,sl]=qm/np.linalg.norm(qm,axis=1)[:,None]
check('execution_geometry_midpoints',close(scans['qpos'][mi],mid,2e-12),float(np.max(abs(scans['qpos'][mi]-mid))))
pos,rot=pose(native['ee'],goals);hold=native['time']-native['time'][0]>=T-1e-9
check('execution_world_goal_hold',close(pos[hold].max(axis=0),report['hold_max_position_error_m']) and close(rot[hold].max(axis=0),report['hold_max_rotation_error_rad']) and np.all(pos[hold]<=.005) and np.all(rot[hold]<=np.pi/180))
cv=curves(teacher); qt=cv[0](np.clip((native['time']-native['time'][0])/T,0,1)); tracking=float(np.max(abs(native['q_actual_17']-qt)))
check('execution_tracking',close(tracking,report['tracking_max_rad']) and tracking<=.02)
check('execution_manifold',close(np.max(abs(native['manifold_residual'])),report['manifold_max_rad']))
check('torque_no_clipping',np.array_equal(native['ctrl'],native['raw_torque']))
momentum=native['base_generalized_momentum'];peaks=[float(np.linalg.norm(momentum[:,s],axis=1).max()) for s in [slice(0,3),slice(3,6)]]
check('momentum_saved_peaks',close(peaks,report['momentum_peak_linear_angular']))
ledger=read(RUN/'ledger.json');budget=read(RUN/'budget.json')
check('model_budget_from_grids',len(new_intervals)==budget['actual_coupled']==ledger['calls']['model_validation.coupled'] and sum(new_intervals)==budget['integration_intervals'],{'count':len(new_intervals),'intervals':sum(new_intervals)})
check('execution_budget_from_traces',len(native['ctrl'])+len(replay['ctrl'])==ledger['calls']['selected_execution.physics_steps']==27500 and ledger['calls']['selected_execution.torque_rollout']==1)
check('no_unresolved_work',ledger['inflight'] is None and not ledger.get('unresolved_call_accounting') and len(ledger['completed'])==16 and not (RUN/'run.lock').exists())
size=sum(p.stat().st_size for p in RUN.rglob('*') if p.is_file()); check('bounded',size<8*1024**3 and read(RUN/'status.json')['elapsed_s']<7200 and len(new_intervals)<=300 and sum(new_intervals)<3000000,{'bytes':size,'elapsed_s':read(RUN/'status.json')['elapsed_s']})
result={'checks':checks,'failures':[x for x in checks if not x['passed']],'metrics':metrics,'replay_max_errors':replayerr,'execution_hold_position_mm':(pos[hold].max(axis=0)*1000).tolist(),'execution_hold_rotation_deg':np.rad2deg(rot[hold].max(axis=0)).tolist(),'execution_clearance_mm':float(scans['minima'].min()*1000),'initial_state_values':len(initial),'source_count':len(sources),'paper_claim_audit':'available' if (ROOT/'paper/PAPER_CLAIM_AUDIT.json').exists() else 'unavailable'}
(OUT/'independent_saved_checks.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,default=lambda x:x.item() if isinstance(x,np.generic) else x.tolist()),encoding='utf-8')
print(json.dumps({'checks':len(checks),'failures':result['failures'],'metrics':metrics},ensure_ascii=True))
