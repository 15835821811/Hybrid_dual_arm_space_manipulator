"""Independent read-only audit: saved arrays/algebra/hashes only; never imports physics.
Writes exclusively beside this script. No simulation, optimizer, or resume calls.
"""
from pathlib import Path
import collections, datetime, hashlib, json, subprocess, zipfile
import numpy as np
from scipy.spatial.transform import Rotation

TRACE=Path(__file__).resolve().parent
ROOT=TRACE.parents[3]
RUN=ROOT/'artifacts/v7_1/global_bypass_001'
inputs={}; failures=[]; counters=collections.Counter(); maxima=collections.defaultdict(float)
def sha(b):return hashlib.sha256(b).hexdigest()
def touch(p):
    p=Path(p); b=p.read_bytes(); s=p.stat()
    inputs[str(p.relative_to(ROOT))]={'sha256':sha(b),'bytes':len(b),'modified_utc':datetime.datetime.fromtimestamp(s.st_mtime,datetime.timezone.utc).isoformat()}
    return b
def js(p):return json.loads(touch(p).decode('utf-8-sig'))
def nz(p):
    import io
    with np.load(io.BytesIO(touch(p)),allow_pickle=False) as z:return {k:z[k] for k in z.files}
def digest(v):return sha(json.dumps(v,default=lambda a:a.tolist(),sort_keys=True,separators=(',',':'),allow_nan=False).encode())
def check(k,b):
    counters['checks']+=1
    if not bool(b):failures.append(k)
def err(k,a,b):
    v=float(np.max(np.abs(np.asarray(a)-np.asarray(b)),initial=0));maxima[k]=max(maxima[k],v);return v
def pose(a,g):
    a,g=np.asarray(a),np.asarray(g)
    return np.r_[((a[:,:3,3]-g[:,:3,3])/.005).ravel(),(Rotation.from_matrix(g[:,:3,:3].transpose(0,2,1)@a[:,:3,:3]).as_rotvec()/np.deg2rad(1)).ravel()]
def Q(e):return float(np.linalg.norm(np.asarray(e).reshape(4,3),axis=1).max())
def delta(J,e,cap):
    v=-np.linalg.solve(J.T@J+1e-3*np.eye(17),J.T@e);return v*min(1,cap/max(abs(v).max(),1e-12))
def write(name,obj): (TRACE/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')

run=js(RUN/'run.json'); ident=run['identity']; P=ident['protocol']; select=ident['selection']; task=select['entries'][0]['task']; goals=task['condition']['goals']
old=js(ROOT/'ff_prior_static_fullchain_v2/evidence/selection.json')
oldentry=[e for e in old['entries'] if e['sample_id']=='capture_5_obstacles'][0]
check('original_task_entry_exact',select['entries']==[oldentry])
oldunits=[u for u in old['units'] if u['task_id']=='capture_5_obstacles']
check('original_order_seeds_methods',select['units']==[dict(u,source_unit=u['id'],id=f'u{i:02d}') for i,u in enumerate(oldunits)])
check('task_hash',task['task_hash']==digest({k:v for k,v in task.items() if k!='task_hash'}))
check('selection_identity',js(ROOT/'ff_prior_global_bypass/evidence/selection.json')==select)
check('binding_identity',js(ROOT/'ff_prior_global_bypass/evidence/binding.json')==ident['binding'])
source_changes=[]; git_diffs=[]
for p,h in ident['binding']['files'].items():check('input_hash:'+p,sha(touch(ROOT/p))==h)
with zipfile.ZipFile(RUN/'executed_source.zip') as z:
    touch(RUN/'executed_source.zip');check('archive_exact_names',set(z.namelist())==set(ident['source_hashes']))
    prefix=subprocess.check_output(['git','rev-parse','--show-prefix'],cwd=ROOT,text=True).strip()
    for p,h in ident['source_hashes'].items():
        archive=z.read(p); live=touch(ROOT/p)
        check('archive_hash:'+p,sha(archive)==h);check('live_source_hash:'+p,sha(live)==h)
        if sha(live)!=h:source_changes.append(p)
        proc=subprocess.run(['git','show',run['producer_head']+':'+prefix+p],cwd=ROOT,capture_output=True)
        if proc.returncode or proc.stdout!=archive:git_diffs.append({'path':p,'git_blob_present':proc.returncode==0,'only_line_endings':proc.returncode==0 and proc.stdout.replace(b'\r\n',b'\n')==archive.replace(b'\r\n',b'\n'),'git_sha256':sha(proc.stdout) if proc.returncode==0 else None,'archive_sha256':h})

ledger=js(RUN/'ledger.json'); budget=js(RUN/'budget.json'); status=js(RUN/'status.json'); physical=js(RUN/'physical_contract.json'); initial=js(RUN/'scenes/capture_5_obstacles/initial_audit.json')
check('no_inflight',ledger['inflight'] is None and not ledger.get('unresolved_call_accounting'))
check('all_units_completed',set(ledger['completed'])=={'m0',*[u['id'] for u in select['units']]})
check('counter_identity',ledger['identity']['identity_hash']==digest(ident))
base_ids=physical['identity']['base_qpos_indices']; nq=physical['identity']['nq']; nv=physical['identity']['nv']
m=nz(RUN/'m0/propagation_check.npz'); mr=js(RUN/'m0/result.json')
m0={
'full_qpos_max_abs':err('m0_qpos',m['joined_qpos'],m['whole_qpos']),
'full_qvel_max_abs':err('m0_qvel',m['joined_velocity'],m['whole_velocity']),
'parent_qpos_max_abs':err('m0_parent_qpos',m['first_qpos'][-1],m['second_qpos'][0]),
'full_ee_second_half_max_abs':err('m0_ee',m['whole_ee'][256:],m['second_ee']),
'base_motion_max_abs':float(abs(m['first_qpos'][-1,base_ids]-m['first_qpos'][0,base_ids]).max()),
'start_time':float(m['joined_time'][0]),'join_time':float(m['joined_time'][256]),'end_time':float(m['joined_time'][-1]),
'state_length':len(m['root_state'])}
check('m0_saved_qpos_qvel',m0['full_qpos_max_abs']<1e-10 and m0['full_qvel_max_abs']<1e-10)
check('m0_times',np.array_equal(m['joined_time'],np.linspace(0,4,513)))
check('m0_base_moves',m0['base_motion_max_abs']>1e-8)
check('root_state_hash',digest(m['root_state'])==initial['origin']['complete_state_hash'])
for name,idx in [('root',0),('first',256),('second',512)]:
    s=m[name+'_state'];check('m0_state_qpos:'+name,np.array_equal(s[1:1+nq],m['joined_qpos'][idx]));check('m0_state_qvel:'+name,np.array_equal(s[1+nq:1+nq+nv],m['joined_velocity'][idx]));check('m0_state_time:'+name,s[0]==m['joined_time'][idx])
for k in ['first','second']:
    rr=mr[k];check('m0_numeric_report:'+k,rr['solve_residual']<1e-10 and rr['momentum_residual']<1e-10 and rr['quaternion_error']<1e-12)

rows=[]; goal_rows=[]; units=[]; reserved_intervals=1024; inferred_coupled=3; observed_edge_actual=0; pose_stats={}
for spec in select['units']:
    ud=RUN/'units'/spec['id']; ur=js(ud/'result.json'); calls=js(ud/'calls.json'); setup=js(ud/'public_setup.json'); us=js(ud/'execution_selection.json'); repair=js(ud/'repair_selection.json')
    check(spec['id']+'/unit_binding',ur['unit']==spec and len(ur['candidates'])==5 and ur['candidate_denominator']==5)
    check(spec['id']+'/selection',us['candidate_id'] is None and us['candidate_digest']==digest(ur['candidates']) and repair['ids']==[])
    unit_inferred=0;unit_reserved=0
    for row in ur['candidates']:
        pd=ud/'candidates'/f'{row["id"]:02d}'; tag=f'{spec["id"]}/{row["id"]}'; co=nz(pd/'controls.npz'); prov=js(pd/'provenance.json'); outcome=js(pd/'outcome.json')
        check(tag+'/outcome',outcome==row);check(tag+'/provenance',digest(co['network_raw'])==prov['network_raw_hash'] and digest(co['proposal'])==prov['proposal_hash'] and digest(setup['base'])==prov['public_base_hash'])
        check(tag+'/proposal_seed',prov['sampling_seed']==spec['proposal_seeds'][row['id']] and prov['network_calls']==(16 if spec['method']=='diffusion_geometry' else 0))
        scale=prov['preparation']['scale'];expected=co['proposal'] if scale==1 else np.array(setup['base'])+scale*(co['proposal']-setup['base'])
        check(tag+'/prepare_algebra',np.array_equal(co['prepared'],expected))
        seed=nz(pd/'goal_seed/initial.npz');check(tag+'/seed_initial',np.array_equal(seed['Z'],co['prepared']) and err('seed_pose',pose(seed['ee'],goals),seed['goal'])<1e-10)
        z=seed['Z'];ee=seed['ee'];e=seed['goal'];seed_calls=1
        for sp in sorted((pd/'goal_seed/iterations').glob('*/step.npz')):
            a=nz(sp); rec=js(sp.parent/'record.json');check(tag+'/seed_before',np.array_equal(a['before_Z'],z) and np.array_equal(a['ee'],ee))
            J=np.asarray([(pose(pe,goals)-e)/h if h else np.zeros(12) for pe,h in zip(a['perturb_ee'],a['epsilon'])]).T
            check(tag+'/seed_J',err('seed_J',J,a['jacobian'])<1e-8);check(tag+'/seed_delta',err('seed_delta',delta(J,e,.15),a['step'])<1e-9)
            seed_calls+=int(np.count_nonzero(a['epsilon']));accepted=[]
            for trial in rec['trials']:
                if 'ee' not in trial:check(tag+'/seed_quick_reject',not all(trial['quick'].values()) and not trial['accepted']);continue
                seed_calls+=1; ne=pose(trial['ee'],goals);og=np.linalg.norm(e.reshape(4,3),axis=1);ng=np.linalg.norm(ne.reshape(4,3),axis=1)
                expect=all(x>1 or y<=1 for x,y in zip(og,ng)) and ne@ne<e@e-1e-10
                check(tag+'/seed_accept',trial['accepted']==expect and err('seed_trial_pose',ne,trial['goal'])<1e-10)
                if trial['accepted']:accepted.append(trial)
            check(tag+'/seed_one_accept',len(accepted)==int(rec['accepted']))
            if accepted:
                trial=accepted[0];z=z+trial['scale']*np.einsum('i,ijk->jk',a['step'],a['directions']);ee=np.array(trial['ee']);e=pose(ee,goals)
            check(tag+'/seed_after',np.array_equal(z,a['after_Z']))
        sf=nz(pd/'goal_seed/result.npz');sr=js(pd/'goal_seed/result.json');check(tag+'/seed_final',np.array_equal(sf['Z'],z) and np.array_equal(sf['ee'],ee) and abs(sr['Q']-Q(e))<1e-10)
        td=pd/'tree'; n=nz(td/'nodes.npz'); tr=js(td/'result.json'); attempts=js(td/'attempts.json'); edges={};tree_reserved=0;tree_actual=0
        check(tag+'/tree_identity',tr['seed']==spec['proposal_seeds'][row['id']]+440001 and tr['nodes']==len(n['q']) and tr['attempts']==len(attempts))
        check(tag+'/root',np.array_equal(n['state'][0],m['root_state']) and np.array_equal(n['q'][0],task['condition']['initial_features']))
        check(tag+'/state_fields',np.array_equal(n['state'][:,1:1+nq],n['qpos']) and np.array_equal(n['state'][:,1+nq:1+nq+nv],n['qvel']) and np.array_equal(n['state'][:,0],n['elapsed']))
        check(tag+'/finite_nodes',all(np.isfinite(a).all() for a in n.values()))
        check(tag+'/rest_nodes',np.max(abs(n['qvel']))<1e-10)
        state_map={digest(s):i for i,s in enumerate(n['state'])}
        for ep in sorted((td/'edges').glob('*/report.json')):
            er=js(ep);a=nz(ep.parent/'trace.npz');eid=er['edge_id'];tree_reserved+=er['intervals'];tree_actual+=er['actual_intervals']
            check(tag+f'/edge{eid}/finite',all(np.isfinite(x).all() for x in a.values()))
            check(tag+f'/edge{eid}/array_report',len(a['time'])-1==er['actual_intervals'] and abs(float(a['minimum'].min())-er['minimum_clearance_m'])<1e-12 and abs(float(a['geometry_step'].max())-er['max_geometry_step_m'])<1e-12)
            check(tag+f'/edge{eid}/parent',digest(a['parent_state'])==er['parent_state_hash'] and er['parent_state_hash'] in state_map)
            par=state_map[er['parent_state_hash']]
            check(tag+f'/edge{eid}/parent_fields',np.array_equal(a['parent_state'],n['state'][par]) and err('edge_start_qpos',a['qpos'][0],n['qpos'][par])<1e-10 and err('edge_start_qvel',a['qvel'][0],n['qvel'][par])<1e-10 and abs(a['time'][0]-n['elapsed'][par])<1e-10)
            numeric=(er['momentum_residual']<1e-10 and er['solve_residual']<1e-10 and er['quaternion_error']<1e-12)
            expected=er['complete'] and numeric and a['geometry_step'].max()<=.002
            check(tag+f'/edge{eid}/accept',expected==er['accepted'] and numeric==er['numerical_ok'])
            if er['accepted']:check(tag+f'/edge{eid}/clearance',a['minimum'].min()>=0)
            check(tag+f'/edge{eid}/quaternion',np.max(abs(np.linalg.norm(a['qpos'][:,base_ids[3:]],axis=1)-1))<1e-12)
            counters['edges']+=1;counters['accepted_edges']+=int(er['accepted']);counters['collision_stopped_edges']+=int(not er['complete'] and a['minimum'].min()<0)
            edges[eid]=(er,a)
        check(tag+'/edge_count',len(edges)==tr['propagations'])
        for i in range(1,len(n['q'])):
            par=int(n['parent'][i]);er,a=edges[int(n['edge_id'][i])]
            check(tag+f'/node{i}',0<=par<i and er['accepted'] and np.array_equal(a['parent_state'],n['state'][par]) and np.array_equal(a['child_state'],n['state'][i]) and np.array_equal(a['qpos'][-1],n['qpos'][i]) and np.array_equal(a['qvel'][-1],n['qvel'][i]) and np.array_equal(a['ee'][-1],n['ee'][i]) and np.array_equal(a['target'],n['q'][i]) and abs(n['elapsed'][i]-n['elapsed'][par]-n['duration'][i])<1e-10 and abs(a['time'][-1]-n['elapsed'][i])<1e-10)
        check(tag+'/attempt_links',all((not a['accepted']) or (n['parent'][a['node']]==a['parent'] and n['edge_id'][a['node']]==a['edge']) for a in attempts))
        gc_iterations=0;gc_accepts=0;gc_best=[];gc_count=0; unmatched=0
        for hp in sorted((td/'goal_connections').glob('*/iterations.json')):
            hist=js(hp); par=int(hp.parent.name);q=n['q'][par].copy();gc_count+=1;lastQ=None;nodehash=digest(n['state'][par])
            for i,h in enumerate(hist):
                a=nz(hp.parent/f'iteration_{i:02d}.npz');gc_iterations+=1
                J=np.asarray([(pose(pe,goals)-a['goal'])/eps for pe,eps in zip(a['perturb_ee'],a['epsilon'])]).T
                check(tag+'/goal_J',err('goal_J',J,a['jacobian'])<1e-8);check(tag+'/goal_delta',err('goal_delta',delta(J,a['goal'],.12),a['delta'])<1e-9)
                accepted=[];curq=q.copy()
                for trial in h['trials']:
                    proposed=curq+trial['scale']*a['delta']
                    matches=[(er,ar) for er,ar in edges.values() if er['parent_state_hash']==nodehash and np.max(abs(ar['target']-proposed))<1e-12 and abs(er['duration']-(27-n['elapsed'][par]))<1e-10]
                    if trial['edge_pass']:
                        ne=pose(trial['ee'],goals);expected=bool(ne@ne<a['goal']@a['goal']);check(tag+'/goal_trial_accept',trial['accepted']==expected)
                        check(tag+'/goal_trial_edge',any(er['accepted'] and np.array_equal(ar['ee'][-1],trial['ee']) for er,ar in matches))
                        lastQ=Q(ne)
                        if trial['accepted']:accepted.append(trial);q=proposed;gc_accepts+=1
                    elif not matches:unmatched+=1
                    else:check(tag+'/goal_failed_edge',not any(er['accepted'] for er,ar in matches))
                check(tag+'/goal_accept_count',len(accepted)==int(h['accepted']))
            if lastQ is not None:gc_best.append(lastQ)
            goal_rows.append({'candidate':tag,'parent_node':par,'iterations':len(hist),'last_evaluated_Q':lastQ,'accepted_iterations':sum(h['accepted'] for h in hist)})
        check(tag+'/search_no_route',tr['solution_node'] is None and row['status']=='NO_COMPLETE_GLOBAL_ROUTE' and not row['eligible'] and not (pd/'fit.npz').exists())
        reserved=seed_calls*512+tree_reserved+gc_iterations*18*256;coupled=seed_calls+len(edges)+gc_iterations*18
        unit_inferred+=coupled;unit_reserved+=reserved;reserved_intervals+=reserved;inferred_coupled+=coupled;observed_edge_actual+=tree_actual
        vals=[Q(pose(e,goals)) for e in n['ee']]; best=int(np.argmin(vals));depth=[]
        for i in range(len(n['q'])):
            steps=0;j=i
            while n['parent'][j]>=0:steps+=1;j=n['parent'][j]
            depth.append(steps)
        rows.append({'unit':spec['id'],'method':spec['method'],'slot':row['id'],'seed':spec['proposal_seeds'][row['id']],'seed_Q':sr['Q'],'nodes':len(n['q']),'accepted_tree_nodes':len(n['q'])-1,'attempts':len(attempts),'saved_connector_propagations':len(edges),'inferred_tree_total_coupled':len(edges)+18*gc_iterations,'inferred_seed_coupled':seed_calls,'reserved_intervals':reserved,'saved_edge_actual_intervals':tree_actual,'goal_connections_with_iteration_logs':gc_count,'goal_iterations':gc_iterations,'goal_accepted_iterations':gc_accepts,'failed_goal_trials_without_edge_trace':unmatched,'minimum_tree_Q':min(vals),'minimum_tree_Q_node':best,'best_joint_distance_node_Q':vals[tr['best_node']],'maximum_depth':max(depth),'maximum_elapsed':float(n['elapsed'].max()),'stop':tr['stop'],'goal_last_evaluated_Q_min':min(gc_best) if gc_best else None})
        counters['accepted_tree_nodes']+=len(n['q'])-1;counters['goal_iterations']+=gc_iterations;counters['goal_accepted_iterations']+=gc_accepts;counters['goal_connections']+=gc_count
        print(tag,'nodes',len(n['q']),'edges',len(edges),'goal iterations',gc_iterations,flush=True)
    check(spec['id']+'/coupled_from_saved_records',calls['coupled']==unit_inferred)
    units.append({'unit':spec['id'],'method':spec['method'],'status':ur['status'],'execution_level':ur['execution_level'],'calls':calls,'saved_records_inferred_coupled':unit_inferred,'reserved_intervals':unit_reserved})

check('total_coupled',inferred_coupled==budget['actual_coupled']==budget['reserved_coupled']==sum(v for k,v in ledger['calls'].items() if k.endswith('.coupled')))
check('total_reserved_integrations',reserved_intervals==budget['integration_intervals'])
check('native_zero',budget['actual_native_and_replay_steps']==0 and sum(v for k,v in ledger['calls'].items() if k.endswith('.physics_steps'))==0 and not list(RUN.glob('units/*/execution/native_trace.npz')))
audit=ROOT/'paper/PAPER_CLAIM_AUDIT.json';paper_audit=js(audit) if audit.exists() else None
receipt=ROOT/'ff_prior_global_bypass/evidence/completed_resume.json'
if receipt.exists():js(receipt)
write('saved_evidence_check.json',{'kind':'independent_saved_array_algebra_no_physics','failures':failures,'counters':dict(counters),'maximum_errors':dict(maxima),'producer_head':run['producer_head'],'source_count':len(ident['source_hashes']),'input_count':len(ident['binding']['files']),'live_source_changes':source_changes,'archive_vs_git':git_diffs,'m0':m0,'units':units,'candidates':rows,'goal_connections':goal_rows,'accounting':{'inferred_coupled':inferred_coupled,'reserved_integrations':reserved_intervals,'saved_connector_actual_integrations':observed_edge_actual,'remaining_actual_integration_count_is_inferred_not_fully_traced':True},'paper_claim_audit':paper_audit,'limitations':['No physics re-evaluation; collision samples and numeric residual reports checked for internal consistency only.','Goal finite-difference center/perturbation full state/qpos/qvel/time traces are not saved.','Some motion-rejected goal line-search trials have no edge trace or explicit rejection gate.','Geometry step arrays cannot be independently derived from qpos without kinematic reconstruction; no such new physics was authorized.']})
write('inputs.json',{'review_role':'fresh independent paper architect/reviewer','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'artifacts':inputs})
print(json.dumps({'checks':counters['checks'],'failures':failures,'counters':dict(counters),'inferred_coupled':inferred_coupled,'reserved_integrations':reserved_intervals,'git_diffs':git_diffs},ensure_ascii=False),flush=True)
