"""Bounded audit: saved-array algebra and hashes only. Never import project/MuJoCo.

Reads this new run in full. Historical raw inputs are not opened, even for hashing.
Only the allowed static closeout/verdict and newly bound terminal are interpreted.
Writes solely beside this script. No optimizer, FK, geometry query, or native step.
"""
from pathlib import Path
import json, hashlib, zipfile, collections, datetime, xml.etree.ElementTree as ET
import numpy as np

ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent
RUN = ROOT/'artifacts/v7_1/balanced_brake_001'
checks=[]; metrics={}; arrays={}; inputs={}; cache={}
def plain(x):
    if isinstance(x,np.ndarray): return x.tolist()
    if isinstance(x,np.generic): return x.item()
    if isinstance(x,Path):return x.as_posix()
    raise TypeError(type(x).__name__)
def dump(name,x):
    (OUT/name).write_text(json.dumps(x,default=plain,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def hfile(p):
    p=Path(p).resolve()
    if p not in cache:
        h=hashlib.sha256()
        with p.open('rb') as f:
            for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
        cache[p]=h.hexdigest()
    inputs[p.relative_to(ROOT).as_posix()]={'sha256':cache[p],'bytes':p.stat().st_size,'mtime_ns':p.stat().st_mtime_ns}
    return cache[p]
def digest(x):return hashlib.sha256(json.dumps(x,default=plain,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def read(p):
    p=Path(p);hfile(p);return json.loads(p.read_text(encoding='utf-8-sig'))
def ck(name,cond,detail=None):
    x={'name':name,'pass':bool(cond)}
    if detail is not None:x['detail']=detail
    checks.append(x)
def close(name,a,b,tol=1e-12):
    a=np.asarray(a);b=np.asarray(b)
    err=float(np.max(abs(a-b))) if a.shape==b.shape and a.size else 0. if a.shape==b.shape else None
    ck(name,a.shape==b.shape and err is not None and err<=tol,{'max_abs_error':err,'tolerance':tol})
def load(p):
    p=Path(p);hfile(p)
    with np.load(p,allow_pickle=False) as z:a={k:z[k] for k in z.files}
    stats={}
    for k,v in a.items():
        good=np.isfinite(v)
        stats[k]={'shape':list(v.shape),'dtype':str(v.dtype),'elements':v.size,'nonfinite':int((~good).sum()),
                  'minimum':float(v[good].min()) if good.any() else None,'maximum':float(v[good].max()) if good.any() else None}
        # Exactly one unavailable filtered command, at the failed terminal tick.
        allowed=p.name=='body_obstacle_trace.npz' and k in ['command_before','command_after']
        ck(p.name+':finite:'+k,good.all() or (allowed and (~good).sum()==17 and good[:-1].all() and np.isnan(v[-1]).all()))
    arrays[p.relative_to(ROOT).as_posix()]=stats
    return a

r=read(RUN/'run.json');ident=r['identity'];contract=ident['contract'];protocol=ident['protocol']
result=read(RUN/'execution/result.json');summary=read(RUN/'summary.json');budget=read(RUN/'budget.json');status=read(RUN/'status.json')
for path,expected in ident['source_hashes'].items(): ck('source:'+path,hfile(ROOT/path)==expected)
# All frozen entries counted; historical raw never reopened. Verify only new scope and approved metadata.
for path,expected in ident['binding']['files'].items():
    approved=(path.startswith('ff_prior_balanced_brake/') or path.startswith('dual_arm_space_robot_2026/') or path in ['ff_prior_fullchain/evidence/balanced_restore_closeout.json','ff_prior_fullchain/evidence/balanced_tracking_closeout.json','paper/review-traces/experiment-result-to-claim/20261023_balanced_static/verdict.json'])
    if approved:ck('bound_input_checked:'+path,hfile(ROOT/path)==expected)
ck('frozen_binding_actual_count',len(ident['binding']['files'])==265)
metrics['historical_binding_policy']='265 frozen input hashes retained; historical raw not reopened; new-scope bindings, assets, and approved predecessor closeout/verdict hashes checked.'
metrics['frozen_input_hashes']=ident['binding']['files']
z=zipfile.ZipFile(RUN/'executed_source.zip');hfile(RUN/'executed_source.zip')
ck('source_archive_exact_members',set(z.namelist())==set(ident['source_hashes']) and len(z.namelist())==len(ident['source_hashes']))
for path,expected in ident['source_hashes'].items():ck('zip:'+path,hashlib.sha256(z.read(path)).hexdigest()==expected)
ck('dynamic_source_bundle_hash',digest(ident['source_hashes'])==contract['dynamic_source_bundle_hash'])
raw=dict(contract);given=raw.pop('contract_hash');ck('contract_canonical_hash',digest(raw)==given)
ck('same_bound_contract',read(ROOT/'ff_prior_balanced_brake/evidence/contract.json')==contract)
for path in ['ff_prior_balanced_brake/docs/EXPERIMENT_PLAN.md','ff_prior_balanced_restore/docs/EXPERIMENT_PLAN.md']:hfile(ROOT/path)
static=read(ROOT/'ff_prior_fullchain/evidence/balanced_restore_closeout.json')
staticverdict=read(ROOT/'paper/review-traces/experiment-result-to-claim/20261023_balanced_static/verdict.json')
terminal=load(ROOT/'ff_prior_balanced_brake/evidence/terminal.npz')
ck('static_predecessor_closeout_accepted',static['closed'] and static['model_valid']==static['executed']==static['replay_verified']==1 and staticverdict['claims']['C1']['claim_supported']=='yes')
ck('static_terminal_digest_chain',digest(terminal['integration'])==static['terminal_state_digest']==contract['source_terminal_digest'])
ck('static_review_bound_hash',hfile(ROOT/static['independent_review'])==static['checksums']['review'])
n=load(RUN/'execution/native_trace.npz');replay=load(RUN/'execution/replay_trace.npz');initial=load(RUN/'initial.npz');geom=load(RUN/'execution/geometry.npz')
body=load(RUN/'execution/body_obstacle_trace.npz');cc=load(RUN/'execution/controller_clearance.npz')
trans=load(RUN/'execution/transition_319.npz')
bm=read(RUN/'execution/body_obstacle_metadata.json');cm=read(RUN/'execution/controller_clearance_metadata.json')
cf=read(RUN/'control_filter.json');be=read(RUN/'execution/body_filter_events.json');gp=read(RUN/'execution/geometry_pairs.json')
steps=len(n['ctrl']);states=len(n['time']);ticks=len(cf);dt=.002
ck('native_dimensions',n['integration'].shape==(steps+1,836) and n['ctrl'].shape==(steps,67) and steps==13500)
for key in ['integration','qpos','qvel','ctrl']:close('complete_replay_every_element:'+key,n[key],replay[key],0)
for key in ['integration','qpos','qvel']:close('terminal_to_initial_native:'+key,terminal[key],n[key][0],0)
close('terminal_to_initial_npz',terminal['integration'],initial['integration'],0)
close('terminal_reference',terminal['reference_q'],n['qref'][0],0)
close('initial_previous_command',terminal['reference_qd'],cf[0]['previous_command'],0)
close('all_native_absolute_clock',np.diff(n['time']),np.full(steps,dt),1e-10)
close('integration_time_layout',n['integration'][:,0],n['time'],0)
close('integration_qpos_layout',n['integration'][:,1:82],n['qpos'],0)
close('integration_qvel_layout',n['integration'][:,82:161],n['qvel'],0)
close('integration_saved_ctrl_layout',n['integration'][1:,240:307],n['ctrl'],0)
ck('saved_applied_external_wrenches_all_zero',np.count_nonzero(n['integration'][:,307:])==0)
close('base_qpos',n['base'],n['qpos'][:,:7],0)
close('inherited_satellite_velocity',n['qvel'][:,73:79],np.tile(terminal['qvel'][73:79],(states,1)),0)
close('satellite_pose_constant',n['qpos'][:,74:81],np.tile(terminal['qpos'][74:81],(states,1)),0)

# Independently transcribed frozen 17 -> 67 linear map, no RobotAdapter/model load.
map12=np.array([[0,1],[1,0],[1,0],[0,1],[0,1],[1,0],[1,0],[0,1],[0,1],[1,0],[1,0],[0,1]],float)/6
B=np.zeros((67,17));B[:60,:10]=np.kron(np.eye(5),map12);B[60:,10:]=np.eye(7);Bplus=np.linalg.pinv(B)
lowq=n['qpos'][:,7:74];lowv=n['qvel'][:,6:73]
qcalc=lowq@Bplus.T;v=lowv@Bplus.T;a=np.diff(v,axis=0)/dt;la=np.diff(lowv,axis=0)/dt
close('q17_from_all_saved_qpos',qcalc,n['q17'],1e-14)
L=np.r_[np.full(10,-1.),np.full(7,-np.pi)];U=-L
vlim=np.r_[np.full(10,.8),np.full(7,1.2)];alim=np.r_[np.full(10,2.5),np.full(7,4.)];torque=np.r_[np.full(60,4.),np.full(7,80.)]
urdf=ROOT/'dual_arm_space_robot_2026/urdf/dual_arm_space_robot_2026.urdf';hfile(urdf)
jt=[x for x in ET.fromstring(urdf.read_text(encoding='utf-8')).findall('joint') if x.get('type') in ('revolute','continuous','prismatic')]
ck('urdf_scalar_joint_count',len(jt)==67)
limited=[(i,float(x.find('limit').get('lower')),float(x.find('limit').get('upper'))) for i,x in enumerate(jt) if x.get('type')!='continuous' and x.find('limit') is not None]
low_position=all(np.all(lowq[:,i]>=l) and np.all(lowq[:,i]<=u) for i,l,u in limited)
mp=np.array([np.linalg.norm(n['momentum'][:,s],axis=1).max() for s in [slice(0,3),slice(3,6)]])
me=np.array([np.linalg.norm(n['exchange'][:,s],axis=1).max() for s in [slice(0,3),slice(3,6)]])
position=np.linalg.norm(n['ee'][:,:,:3,3]-n['desired'][:,:,:3,3],axis=-1)
relative=np.einsum('naij,nakj->naik',n['desired'][:,:,:3,:3],n['ee'][:,:,:3,:3])
skew=np.stack([relative[:,:,2,1]-relative[:,:,1,2],relative[:,:,0,2]-relative[:,:,2,0],relative[:,:,1,0]-relative[:,:,0,1]],axis=-1)
rotation=np.arctan2(np.linalg.norm(skew,axis=-1)/2,(np.trace(relative,axis1=-2,axis2=-1)-1)/2)
hold=n['time']-n['time'][0]>=25.5-1e-9
gates={
 'complete_native_time':steps==13500 and result['stop_reason'] is None,
 'independent_replay':np.array_equal(n['qpos'],replay['qpos']) and np.array_equal(n['qvel'],replay['qvel']),
 'full_integration_replay':np.array_equal(n['integration'],replay['integration']),
 'actual_terminal_handoff':np.array_equal(n['integration'][0],terminal['integration']),
 'absolute_time_continuity':np.all(abs(np.diff(n['time'])-dt)<1e-10),
 'whole_body_clearance':np.min(n['clearance'])>=0,
 'dynamic_position':np.all(position<=.005),'dynamic_rotation':np.all(rotation<=np.pi/180),
 'relative_hold':steps==13500 and hold.any() and np.all(position[hold]<=.005) and np.all(rotation[hold]<=np.pi/180),
 'joint_limits':low_position and np.all(n['q17']>=L) and np.all(n['q17']<=U) and np.all(n['qref']>=L) and np.all(n['qref']<=U),
 'derivative_limits':np.all(abs(v)<=vlim) and np.all(abs(a)<=alim) and np.all(abs(lowv)<=abs(B)@vlim) and np.all(abs(la)<=abs(B)@alim) and np.all(abs(n['reference_qd'])<=vlim) and np.all(abs(n['reference_qdd'])<=alim),
 'raw_torque':np.all(abs(n['raw_torque'])<=torque+1e-12),
 'tracking':np.max(abs(n['q17']-n['qref']))<=.02,
 'manifold':np.max(abs(lowq-n['q17']@B.T))<=.02,
 'momentum':np.all(mp<=1e-5+.002*me),
 'no_external_wrench':np.count_nonzero(n['integration'][:,307:])==0,
 'finite':all(np.isfinite(x).all() for x in n.values())}
for k,value in gates.items():ck('gate_matches_record:'+k,value==result['gates'][k])
for name,data in [('position_mm',position*1000),('rotation_deg',np.rad2deg(rotation))]:
    for stat,fun in [('mean',lambda x:x.mean(0)),('p95',lambda x:np.percentile(x,95,axis=0)),('maximum',lambda x:x.max(0)),('final',lambda x:x[-1])]:
        close(name+':'+stat,fun(data),result[name][stat],1e-9)
close('momentum_peak',mp,result['momentum_peak']);close('momentum_threshold',1e-5+.002*me,result['momentum_threshold'])
close('actual_control_clipped_raw',n['ctrl'],np.clip(n['raw_torque'],-torque,torque),0)
close('reference_step_integral',np.diff(n['qref'],axis=0),dt*n['reference_qd'],1e-15)

# Full original reference positions/rotations reconstructed algebraically.
w=contract['waypoints'];points=np.asarray(w['waypoint_points_m']);dur=np.asarray(w['segment_durations_s']);bound=np.r_[0,np.cumsum(dur)]
expected=[]
for t in n['time']-n['time'][0]:
    if t<4.5:u=t/4.5;start=np.array(w['initial_position_w']);end=points[0];seg=-1
    elif t<25.5:seg=min(5,int(np.searchsorted(bound,t-4.5,side='right')-1));u=(t-4.5-bound[seg])/dur[seg];start=points[seg];end=points[seg+1]
    else:seg=6;u=1.;start=end=points[-1]
    u=np.clip(u,0,1);s=u**3*(10-15*u+6*u*u);p=start+s*(end-start)
    if seg in [0,2,4]:p=p+64*u**3*(1-u)**3*.07*np.array([-1.,0.,0.])
    expected.append(p)
close('all_continuum_desired_positions',np.asarray(expected),n['desired'][:,0,:3,3],1e-13)
close('all_continuum_desired_rotations',n['desired'][:,0,:3,:3],np.tile(np.asarray(w['target_rotation_world']),(states,1,1)),1e-13)
close('all_rigid_desired_poses',n['desired'][:,1],np.tile(np.asarray(contract['static_goals'])[1],(states,1,1)),1e-13)
arrivals=n['time'][0]+4.5+np.r_[0,np.cumsum(dur)];wp=[]
for j,t in enumerate(arrivals):
    k=int(np.argmin(abs(n['time']-t)));obs=result['original_waypoint_observations'][j]
    reached=n['time'][-1]>=t-1e-9;err=np.linalg.norm(n['ee'][k,0,:3,3]-points[j])*1000
    ck('waypoint_arrival_coverage:'+str(j+1),reached==obs['recorded_through_arrival'])
    close('waypoint_nearest_tick:'+str(j+1),[n['time'][k],t,n['time'][k]-t,err],[obs['nearest_recorded_time'],obs['scheduled_time'],obs['time_difference'],obs['position_error_mm']],1e-10)
    wp.append({'waypoint':j+1,'scheduled_time':float(t),'recorded_through_arrival':bool(reached),'arrival_error_mm':float(err) if reached else None,'nearest_tick_time':float(n['time'][k]),'terminal_distance_if_unreached_mm':None if reached else float(err)})
for key in trans:close('transition_319_all_values:'+key,trans[key],n[key][2249:2252],0)
trans2=load(RUN/'execution/transition_529.npz')
for key in trans2:close('transition_529_all_values:'+key,trans2[key],n[key][12749:12752],0)
metrics['hold_maximum_position_mm']=position[hold].max(0)*1000
metrics['hold_maximum_rotation_deg']=np.rad2deg(rotation[hold].max(0))
metrics['hold_time_bounds']=n['time'][hold][[0,-1]]

# Recorded geometry scans: initial, then native endpoint and its midpoint.
ck('geometry_full_scan_count',len(geom['minima'])==2*steps+1==len(gp))
close('geometry_initial',geom['qpos'][0],n['qpos'][0],0)
close('geometry_native_endpoints',geom['qpos'][1::2],n['qpos'][1:],0)
mid=(n['qpos'][:-1]+n['qpos'][1:])/2
for sl in [slice(3,7),slice(77,81)]:
    qa=n['qpos'][:-1,sl];qb=n['qpos'][1:,sl];qb=qb*np.where(np.sum(qa*qb,axis=1)[:,None]<0,-1,1)
    qm=qa+qb;mid[:,sl]=qm/np.linalg.norm(qm,axis=1)[:,None]
close('geometry_all_saved_midpoints',geom['qpos'][2::2],mid,1e-12)
close('native_clearance_scan_reduction',n['clearance'],np.r_[geom['minima'][0],np.minimum(geom['minima'][1::2],geom['minima'][2::2])],0)
close('minimum_clearance_report',geom['minima'].min()*1000,result['minimum_clearance_mm'])
cat={(p['geom_a_name'],p['geom_b_name']) for p in cm['pairs']}
ck('all_minimum_pair_identities_in_complete_catalog',all((p['geom_a'],p['geom_b']) in cat for p in gp))

# Every body/obstacle sample and connector, no resampling and no new distance query.
idx=body['native_index'];ck('all_control_ticks_plus_failed_terminal',np.array_equal(idx,np.arange(0,steps+1,10)) and ticks+1==len(be)==len(idx)==1351)
close('body_tick_times',body['time'],n['time'][idx],0);close('controller_tick_times',cc['time'],body['time'][:-1],0)
close('body_all_continuum_shapes',body['body_positions'][:,bm['continuum_body_ids']],n['skeleton_c'][idx],0)
rid=[bm['body_names'].index(x) for x in ['base_link_r','shoulder_link','back_arm_link','fore_arm_link','wrist_link','end_effector_r']]
close('body_all_rigid_shapes',body['body_positions'][:,rid],n['skeleton_r'][idx],0)
segments=np.diff(body['body_positions'][:,bm['continuum_body_ids']],axis=1);length=np.linalg.norm(segments,axis=-1)
ang=np.arctan2(np.linalg.norm(np.cross(segments[:,:-1],segments[:,1:]),axis=-1),np.sum(segments[:,:-1]*segments[:,1:],axis=-1));scale=(length[:,:-1]+length[:,1:])/2
curv=np.divide(ang,scale,out=np.zeros_like(ang),where=scale>1e-12)
close('all_discrete_curvature',body['continuum_curvature'],curv,1e-12)
close('all_body_connector_lengths',np.linalg.norm(body['connector'][:,:,:,0]-body['connector'][:,:,:,1],axis=-1),body['distance_m'],1e-10)
ck('all_body_nearest_geom_memberships',all(np.isin(body['nearest_geom_id'][:,:,j],link['geom_ids']).all() for j,link in enumerate(bm['links'])))
ck('body_saved_query_account',bm['query_count']==len(idx)*7*sum(len(x['geom_ids']) for x in bm['links']))
ck('body_expected_and_actual_denominators',bm['samples']==1351 and bm['expected_samples']==1351 and bm['status']=='COMPLETE')
for f,meta in bm['files'].items():ck('body_file_binding:'+f,hfile(RUN/'execution'/f)==meta['sha256'] and (RUN/'execution'/f).stat().st_size==meta['bytes'])
ck('body_source_native_binding',hfile(RUN/'execution/native_trace.npz')==bm['source_native_sha256']==result['native_trace_sha256'])

# Complete pair table dispositions and all 139886 stored Jacobian rows/events.
sb=cc['sphere_lower_bound_m'];dist=cc['distance_or_lower_bound_m'];mask=cc['exact_query_mask'];disp=cc['disposition'];off=cc['row_offsets']
ck('pair_catalog_complete_sequential_unique',len(cm['pairs'])==3406 and [x['index'] for x in cm['pairs']]==list(range(3406)) and len({(x['geom_a'],x['geom_b']) for x in cm['pairs']})==3406)
ck('all_exact_query_decisions',np.array_equal(mask,sb<=.09))
ck('all_far_pairs_preserve_lower_bounds',np.array_equal(dist[~mask],sb[~mask]))
ck('all_dispositions',np.array_equal(disp,np.where(~mask,0,np.where(dist>.08,1,2))))
ck('all_queried_distances_respect_sphere_bound',np.all(dist[mask]>=sb[mask]-1e-10))
ck('row_offsets_cover_all_rows',off[0]==0 and off[-1]==len(cc['row_pair_index']) and np.all(np.diff(off)>=0))
close('all_row_rhs_algebra',cc['lower_without_ramp_buffer'],-8*cc['margin_m']-cc['drift'],1e-12)
reason_counts=collections.Counter();selected_obstacle=collections.Counter();binding_obstacle=collections.Counter();negative_desired=0;binding_count=0;filtered_min=float('inf');selected_total=0
maxerrors=collections.defaultdict(float);body_pair_cross=0;body_compare_errors=[];branch=[]
exec((OUT/'brake_algebra.py').read_text(encoding='utf-8'))
for ti,(f,event) in enumerate(zip(cf,be)):
    ni=int(idx[ti]);reason_counts[f['reason']]+=1
    q=n['q17'][ni];ref=n['qref'][ni];prev=np.array(f['previous_command']);desired=np.array(f['desired'])
    lows=[-.7*vlim,4*(L+.04-q),prev-.6*alim*.02];highs=[.7*vlim,4*(U-.04-q),prev+.6*alim*.02]
    for k in range(1,11):
        weight=dt*k*(k+1)/20;offset=ref+dt*(k-k*(k+1)/20)*prev
        lows.append((L+.04-offset)/weight);highs.append((U-.04-offset)/weight)
    lo=np.max(lows,axis=0);hi=np.min(highs,axis=0)
    lo,hi=check_brake(ti,f,q,ref,prev,lo,hi)
    
    ck('filter/body_time_identity:'+str(ti),f['time']==event['time']==cc['time'][ti] and event['native_index']==ni and event['filter_reason']==f['reason'])
    close('all_body_before:'+str(ti),body['command_before'][ti],desired,0)
    if ti:close('previous_control_chain:'+str(ti),prev,cf[ti-1]['filtered'],0)
    sl=slice(off[ti],off[ti+1]);pis=cc['row_pair_index'][sl];jac=cc['jacobian17'][sl];rhs=cc['lower_without_ramp_buffer'][sl]+.008
    ck('no_omitted_active_pairs:'+str(ti),np.array_equal(pis,np.flatnonzero(disp[ti]==2)))
    close('selected_distances:'+str(ti),cc['margin_m'][sl],dist[ti,pis]-.001,0)
    close('all_controller_witness_lengths:'+str(ti),np.linalg.norm(cc['witnesses'][sl,:3]-cc['witnesses'][sl,3:],axis=1),dist[ti,pis],1e-10)
    rows=event['active_constraints'];selected_total+=len(rows)
    ck('all_event_row_count:'+str(ti),len(rows)==len(pis)==f['selected_constraint_count'])
    desiredslack=jac@desired-rhs
    # Independent crosscheck of obstacle-body minima against controller query table.
    if ti==0:pairlook={(x['geom_a'],x['geom_b']):x['index'] for x in cm['pairs']}
    for oi,obs in enumerate(bm['obstacles']):
        for li,link in enumerate(bm['links']):
            ids=[pairlook.get((obs['geom_id'],g)) for g in link['geom_ids']];ids=[p for p in ids if p is not None]
            if ids and body['distance_m'][ti,oi,li]<.08:
                body_compare_errors.append(float(abs(dist[ti,ids].min()-body['distance_m'][ti,oi,li])));body_pair_cross+=1
    successful=f['filtered'] is not None
    if successful:
        filtered=np.array(f['filtered']);boxed=np.clip(desired,lo,hi);slack=jac@filtered-rhs
        close('all_body_after:'+str(ti),body['command_after'][ti],filtered,0)
        close('boxed_command:'+str(ti),f['boxed'],boxed,0)
        close('filter_delta:'+str(ti),f['delta'],np.linalg.norm(filtered-desired),1e-12)
        ck('filter_box_feasible:'+str(ti),np.all(lo<=hi) and np.all(filtered>=lo-1e-12) and np.all(filtered<=hi+1e-12))
        ck('filter_geometry_feasible:'+str(ti),np.all(slack>=-1e-6))
        active=len(rhs)>0 and np.min(jac@boxed-rhs)<-1e-8;changed=np.max(abs(boxed-desired))>1e-10
        ck('projection_flags:'+str(ti),active==f['geometry_projection_required'] and changed==f['box_projection_required'])
        ck('solver_decision:'+str(ti),f['slsqp_used']==bool(active or changed) and (not f['slsqp_used'] or f['solver_success']))
        ck('filter_reason:'+str(ti),f['reason']==('GEOMETRY_AND_BOX_QP' if active else 'BOX_TASK_QP' if f['slsqp_used'] else 'DIRECT_OR_BOX_FEASIBLE'))
        if f['slsqp_used']:close('solver_candidate_saved:'+str(ti),f['solver_candidate'],filtered,0)
        ramp=prev[None,:]+np.arange(1,11)[:,None]/10*(filtered-prev)[None,:]
        close('all_native_command_ramps:'+str(ti),n['reference_qd'][ni:ni+10],ramp,1e-15)
        close('all_native_command_acceleration:'+str(ti),n['reference_qdd'][ni:ni+10],np.tile((filtered-prev)/.02,(10,1)),1e-14)
        filtered_min=min(filtered_min,float(slack.min()));negative_desired+=int(np.sum(desiredslack<0));binding_count+=int(np.sum(abs(slack)<=1e-6))
        ck('body_command_flags:'+str(ti),event['command_available'] and event['filtered_command_available'])
    else:
        ck('failed_terminal_command_unavailable',ti==ticks-1 and np.isnan(body['command_after'][ti]).all() and event['command_available'] and not event['filtered_command_available'])
        branch={'conflicting_indices':np.flatnonzero(lo>hi).tolist(),'minimum_distance_m':float(dist[ti].min()),'minimum_margin_m':float(dist[ti].min()-.001),'slsqp_used':f['slsqp_used'],'filtered_available':False}
        for j in np.flatnonzero(lo>hi):
            branch['coordinate_'+str(j)]={'q_measured':float(q[j]),'q_reference':float(ref[j]),'previous_velocity':float(prev[j]),'lower':float(lo[j]),'upper':float(hi[j]),'gap':float(lo[j]-hi[j]),'lower_component_index':int(np.argmax(np.array(lows)[:,j])),'upper_component_index':int(np.argmin(np.array(highs)[:,j])),'lower_components':np.array(lows)[:,j].tolist(),'upper_components':np.array(highs)[:,j].tolist()}
        ck('saved_stop_branch_proven',np.any(lo>hi) and dist[ti].min()>=0 and not f['slsqp_used'] and f['error']==result['stop_reason'])
    for j,row in enumerate(rows):
        pi=int(pis[j]);pair=cm['pairs'][pi]
        for k in ['geom_a','geom_b','geom_a_name','geom_b_name']:
            if row[k]!=pair[k]:maxerrors['event_identity_mismatch']+=1
        if row['pair_index']!=pi or not row['selected'] or row['reason']!='EXACT_DISTANCE_LE_ACTIVATION_80MM':maxerrors['event_identity_mismatch']+=1
        for name,x,y in [('event_jacobian',row['jacobian17'],jac[j]),('event_rhs',row['lower_rhs'],rhs[j]),('event_margin',row['margin_m'],cc['margin_m'][off[ti]+j]),('event_distance',row['distance_m'],dist[ti,pi]),('event_witness',np.r_[row['witness_a'],row['witness_b']],cc['witnesses'][off[ti]+j])]:
            maxerrors[name]=max(maxerrors[name],float(np.max(abs(np.asarray(x)-y))))
        if pair['geom_a']<7:selected_obstacle[pair['geom_a_name']]+=1
        if successful:
            maxerrors['unfiltered_slack']=max(maxerrors['unfiltered_slack'],abs(row['unfiltered_slack']-desiredslack[j]))
            maxerrors['filtered_slack']=max(maxerrors['filtered_slack'],abs(row['filtered_slack']-slack[j]))
            if row['binding_after_filter']!=(abs(slack[j])<=1e-6):maxerrors['binding_flag_mismatch']+=1
            if pair['geom_a']<7 and abs(slack[j])<=1e-6:binding_obstacle[pair['geom_a_name']]+=1
        elif any(k in row for k in ['filtered_slack','unfiltered_slack','binding_after_filter']):maxerrors['failed_row_fabricated_slack']+=1
for name,error in maxerrors.items():ck('all_rows:'+name,error<=1e-12,{'max_error_or_mismatch_count':error})
ck('all_body_near_minima_crosscheck',max(body_compare_errors,default=0)<1e-10,{'count':body_pair_cross,'max_error':max(body_compare_errors,default=0)})

ck('terminal_body_event_no_fabricated_control',be[-1]['native_index']==13500 and be[-1]['filter_reason']=='ACTUAL_TERMINAL_NO_FOLLOWING_FILTER' and not be[-1]['command_available'] and not be[-1]['filtered_command_available'] and be[-1]['active_constraints']==[] and np.isnan(body['command_before'][-1]).all() and np.isnan(body['command_after'][-1]).all())
import csv
with (OUT/'brake_effect_ticks.csv').open('w',newline='',encoding='utf-8') as f:
    writer=csv.DictWriter(f,fieldnames=list(brake_effect_rows[0]));writer.writeheader();writer.writerows(brake_effect_rows)
worker_warning=(RUN/'worker.stderr.log').read_bytes()
dump('brake_effect_summary.json',dict(scope='new run only; no causal comparison',ticks=ticks,lower_tightened_per_dimension=brake_effect_lower,upper_tightened_per_dimension=brake_effect_upper,minimum_command_bound_margin_per_dimension=brake_effect_command_margin,minimum_original_bound_margin_per_dimension=brake_effect_original_margin,minimum_future_braking_margin_per_dimension=brake_effect_future_margin,index5_first_tightened=next((row for row in brake_effect_rows if row['5_tightened_lower'] or row['5_tightened_upper']),None),all_final_commands_in_all_bounds=bool(np.min(brake_effect_command_margin)>=-2e-12 and np.min(brake_effect_original_margin)>=-2e-12),worker_stderr_bytes=len(worker_warning),slsqp_runtime_warning_occurrences=worker_warning.count(b'RuntimeWarning:'),warning_interpretation='SLSQP internal candidate clipping warning; every actual final command separately checked against all original and tightened bounds; no evidence of actual physical-state clipping.',braking_statistics=brake_stats))
metrics['braking_audit']=brake_stats
metrics['braking_audit']['future_barrier_min_slack']=float(brake_future_min)
metrics['braking_audit']['rounding_tolerance']=2e-12
# Completed resume is inspected, never invoked by this reviewer.
completed=ROOT/'ff_prior_balanced_brake/evidence/completed_001'
receipt=read(completed/'receipt.json');before=read(completed/'core_before.json');after=read(completed/'core_after.json')
current={p.relative_to(RUN).as_posix():{'bytes':p.stat().st_size,'sha256':hfile(p)} for p in RUN.rglob('*') if p.is_file()}
ck('complete_run_manifest_before_after_current',before['files']==after['files']==current)
ck('completed_resume_once_no_work',receipt['status']=='PASS' and receipt['exactly_one_invocation'] and receipt['pre_manifest_persisted_first'] and receipt['response']['resume']=='NO_WORK_ALREADY_COMPLETE' and receipt['returncode']==0)
for stream in ['stdout','stderr']:
    p=completed/(stream+'.bin');ck('resume_raw_'+stream,p.stat().st_size==receipt[stream]['bytes'] and hfile(p)==receipt[stream]['sha256'])
ck('all_budget_native_replay_accounting',budget['counts']==budget['reserved']=={'native':steps,'replay':steps} and budget['pending'] is None and budget['inflight'] is None and budget['extra_coupled']==0)
ck('stopped_before_budget',sum(budget['counts'].values())==27000 and budget['elapsed_s']<1200 and sum(x['bytes'] for x in current.values())<2*1024**3)
ck('summary_denominator_and_no_expansion',summary['execution_denominator']==1 and all(summary[k]==0 for k in ['new_neural_samples','new_coupled_calls','training_updates','comparison_runs']) and not summary['TEST'] and not summary['full_long_term_goal_complete'])
ck('summary_success_consistent',summary['status']==status['status']=='FINISHED' and summary['execution_level']==result['status']=='EXECUTION_VERIFIED' and summary['engineering_dynamic_continuation_pass'])
auditpath=ROOT/'paper/PAPER_CLAIM_AUDIT.json';paperclaim=read(auditpath) if auditpath.exists() else None

metrics.update(producer_git_head=r['producer_head'],source_files=len(ident['source_hashes']),bound_inputs=len(ident['binding']['files']),run_files=len(current),run_bytes=sum(x['bytes'] for x in current.values()),
    steps=steps,planned_steps=13500,actual_states=states,replayed_full_state_values=int(n['integration'].size),replay_max_abs_error=float(np.max(abs(n['integration']-replay['integration']))),
    absolute_start=float(n['time'][0]),absolute_end=float(n['time'][-1]),duration=float(n['time'][-1]-n['time'][0]),planned_duration=27.,unexecuted_steps=13500-steps,
    gates_reconstructed=gates,gates_not_independently_reconstructed=['geometry_step','no_hidden_reset (source/replay/clock supported; not independently proven from state values alone)'],
    position_max_mm=position.max(0)*1000,rotation_max_deg=np.rad2deg(rotation.max(0)),clearance_min_mm=geom['minima'].min()*1000,
    q_tracking_max=float(np.max(abs(n['q17']-n['qref']))),manifold_max=float(np.max(abs(lowq-n['q17']@B.T))),velocity_ratio_max=float(np.max(abs(v)/vlim)),acceleration_ratio_max=float(np.max(abs(a)/alim)),command_velocity_ratio_max=float(np.max(abs(n['reference_qd'])/vlim)),command_acceleration_ratio_max=float(np.max(abs(n['reference_qdd'])/alim)),raw_torque_ratio_max=float(np.max(abs(n['raw_torque'])/torque)),
    momentum_peak=mp,momentum_threshold=1e-5+.002*me,waypoints=wp,hold_samples=int(hold.sum()),geometry_scans=len(geom['minima']),
    filter_ticks=ticks,successful_filter_ticks=ticks,filter_reasons=dict(reason_counts),all_pair_tick_values=dist.size,disposition_counts={str(i):int(np.sum(disp==i)) for i in range(4)},selected_rows=selected_total,omitted_active_rows=0,filtered_min_slack=filtered_min,negative_desired_row_slacks=negative_desired,binding_rows=binding_count,selected_obstacle_rows=dict(selected_obstacle),binding_obstacle_rows=dict(binding_obstacle),failure_branch=branch,
    body_samples=len(idx),body_expected=1351,body_missing_planned=1351-len(idx),body_distance_values=body['distance_m'].size,body_query_count=bm['query_count'],body_minimum_mm=float(body['distance_m'].min()*1000),body_near_crosschecks=body_pair_cross,
    budgets=budget,completed_resume_receipt=receipt,paper_claim_audit=paperclaim if paperclaim else 'unavailable')
dump('numerical_audit.json',{'generated_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'checks_total':len(checks),'checks_passed':sum(c['pass'] for c in checks),'checks_failed':sum(not c['pass'] for c in checks),'failures':[c for c in checks if not c['pass']],'metrics':metrics,'array_coverage':arrays,'checks':checks,'new_physics_steps':0,'new_geometry_queries':0,'new_coupled_calls':0,'resume_invocations':0})
dump('inputs.json',{'policy':'New run interpreted in full; historical raw not opened; 265 frozen input hash values preserved; only approved closed static verdict/closeout, failed closeout and newly bound terminal interpreted.','files':inputs})
print(json.dumps({'checks':len(checks),'passed':sum(c['pass'] for c in checks),'failures':[c for c in checks if not c['pass']],'metrics':{k:metrics[k] for k in ['failure_branch','filter_reasons','selected_rows','disposition_counts','body_distance_values','position_max_mm','rotation_max_deg','clearance_min_mm','raw_torque_ratio_max']}},default=plain,indent=2))
