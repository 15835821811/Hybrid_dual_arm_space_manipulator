"""Algebra-only second stage, consumes delivery completion and model metadata.
Reads complete saved arrays without invoking any model or physics API.
"""
from pathlib import Path
import json,hashlib,datetime
import numpy as np
from scipy.interpolate import BSpline
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3];RUN=ROOT/'artifacts/v7_1/balanced_restore_001';UNIT=RUN/'units/u00'
CHECKS=[];INPUTS={};OUT={}
def plain(v):
    if isinstance(v,np.ndarray):return v.tolist()
    if isinstance(v,np.generic):return v.item()
    raise TypeError(type(v).__name__)
def fh(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
    return h.hexdigest()
def track(p):
    p=p.resolve();key=str(p.relative_to(ROOT))
    if key not in INPUTS:
        s=p.stat();INPUTS[key]=dict(path=str(p),sha256=fh(p),bytes=s.st_size,mtime_utc=datetime.datetime.fromtimestamp(s.st_mtime,datetime.timezone.utc).isoformat(),role='full_saved_array_or_completion_source_audit')
    return INPUTS[key]['sha256']
def read(p):track(p);return json.loads(p.read_text(encoding='utf-8-sig'))
def load(p):
    track(p)
    with np.load(p,allow_pickle=False) as f:return {k:f[k] for k in f.files}
def ck(name,ok,detail=None):CHECKS.append(dict(name=name,passed=bool(ok),detail=detail))
def close(name,a,b,tol=1e-12):
    a,b=np.asarray(a),np.asarray(b)
    if a.shape!=b.shape:ck(name,False,dict(shapes=[a.shape,b.shape]));return
    delta=float(np.max(abs(a-b),initial=0));ck(name,np.isfinite(delta) and delta<=tol,dict(max_abs_error=delta,atol=tol))
meta=read(ROOT/'ff_prior_balanced_delivery/model_metadata.json');report=read(UNIT/'execution/report.json');validation=read(UNIT/'validated/validation.json');selection=read(ROOT/'ff_prior_balanced_restore/evidence/selection.json')
track(ROOT/'ff_prior_fullchain/tools/export_balanced_model_metadata.py');track(ROOT/'ff_prior_fullchain/tools/complete_balanced_saved_diagnostics.py')
ck('model_metadata_scene_identity',meta['scene_model_hash']==report['scene_model_hash']);ck('metadata_all_zero_calls',all(v==0 for v in meta['calls'].values()))
B=np.asarray(meta['B']);BP=np.asarray(meta['Bplus']);qids=np.asarray(meta['qpos_ids']);vids=np.asarray(meta['dof_ids']);rbound=np.asarray(meta['geom_rbound']);jointids=np.array([meta['index_map'][name]['joint'] for name in sorted(meta['index_map'],key=lambda n:meta['index_map'][n]['actuator'])]);limited=np.asarray(meta['jnt_limited'])[jointids].astype(bool);ranges=np.asarray(meta['jnt_range'])[jointids];q0=np.asarray(selection['entries'][0]['task']['condition']['initial_features']);knots=np.r_[np.zeros(4),np.arange(1,21)/21,np.ones(4)]
close('B_left_inverse',BP@B,np.eye(17));close('index_qpos_consistent',np.asarray(meta['jnt_qposadr'])[jointids],qids,0);close('index_dof_consistent',np.asarray(meta['jnt_dofadr'])[jointids],vids,0)
def curve(z):return BSpline(knots,np.vstack([np.repeat(q0[None],3,axis=0),z[:18],np.repeat(z[-1:],3,axis=0)]),3)
def valid(z):
    c=curve(z);low=c.c@B.T
    return bool(np.all(c.c>=meta['planner_lower']) and np.all(c.c<=meta['planner_upper']) and np.all(low[:,limited]>=ranges[limited,0]-1e-9) and np.all(low[:,limited]<=ranges[limited,1]+1e-9) and all(np.all(np.abs(c.derivative(d).c[:len(c.derivative(d).t)-c.derivative(d).k-1]/27**d)<=.995*lim+1e-9) for d,lim in [(1,np.r_[np.full(10,.8),np.full(7,1.2)]),(2,np.r_[np.full(10,2.5),np.full(7,4.)])]))
# Reconstruct every saved geometry displacement/convergence bound with unchanged model radii.
previous=None;rows=[]
for grid in [256,512,1024,2048,4096]:
    a=load(UNIT/f'validated/grid_{grid}.npz')
    extent=np.linalg.norm(np.diff(a['geom'],axis=0),axis=-1)+np.linalg.norm(np.diff(a['geom_rotation'],axis=0).reshape(grid,161,3,3),axis=(-1,-2))*rbound
    step=float(extent.max());row=dict(intervals=grid,maximum_geometry_step_m=step)
    if previous is not None:
        vr=validation['validation_grids'][[512,1024,2048,4096].index(grid)];close(f'model{grid}_full_geometry_step',step,vr['geometry']['maximum_geometry_step_m'],0)
        body=np.linalg.norm(a['body'][::2]-previous['body'],axis=-1).max();geo=np.linalg.norm(a['geom'][::2]-previous['geom'],axis=-1)+np.linalg.norm((a['geom_rotation'][::2]-previous['geom_rotation']).reshape(len(previous['phase']),161,3,3),axis=(-1,-2))*rbound
        aligned=max(body,geo.max());comp=validation['validation']['numerical']['comparisons'][[512,1024,2048,4096].index(grid)];close(f'model{grid}_full_aligned_geometry',aligned,comp['aligned_body_extent_m'],0);row['aligned_geometry_m']=float(aligned)
    previous=a;rows.append(row)
OUT['model_geometry_reconstructed']=rows
# All saved perturbation and probe planner coordinates, objective and accepted trust steps.
for it in [0,1]:
    p=UNIT/f'restore/iter/{it:02d}';j=load(p/'jacobian.npz');rec=read(p/'record.json');pr=read(p/'probes.json');phase=j['active'][:,0]
    close(f'iter{it}_center_active_configuration',curve(j['Z'])(phase)@B.T,j['active_qpos'][:,qids],1e-14);close(f'iter{it}_center_terminal_configuration',curve(j['Z'])(1)@B.T,j['terminal_qpos'][qids],1e-14)
    perturb_error=0.;term_error=0.
    for k in range(323):
        z=j['Z'].copy();z.flat[k]+=j['steps'][k];c=curve(z)
        perturb_error=max(perturb_error,float(np.max(abs(c(phase)@B.T-j['perturb_active_qpos'][k][:,qids]))));term_error=max(term_error,float(np.max(abs(c(1)@B.T-j['perturb_terminal_qpos'][k,qids]))))
    ck(f'iter{it}_all323_active_configuration',perturb_error<1e-14,dict(max_abs_error=perturb_error));ck(f'iter{it}_all323_terminal_configuration',term_error<1e-14,dict(max_abs_error=term_error))
    for pi,v in enumerate(pr['observations']):
        z=j['Z'].copy()
        if v['kind']=='coordinate_half':z.flat[v['coordinate']]+=v['step']
        else:z+=np.asarray(v['direction']).reshape(19,17)*v['sign']*v['step']
        c=curve(z);close(f'iter{it}_probe{pi}_active_configuration',c(phase)@B.T,np.asarray(v['active_qpos'])[:,qids],1e-14);close(f'iter{it}_probe{pi}_terminal_configuration',c(1)@B.T,np.asarray(v['terminal_qpos'])[qids],1e-14)
    solver=rec['solver'];u=np.asarray(solver['returned_u']);radius=rec['radius_rad'];G=j['jg']*radius;D=j['jd']*radius/.003;b=(j['distances']-.0002)/.003;rr=j['goal'];normal=max(1.,.05*float(rr@rr)+np.mean(np.minimum(b,0)**2)+.25*max(np.minimum(b,0)**2));vv=np.minimum(b+D@u,0.);gg=rr+G@u
    objective=(.05*float(gg@gg)+np.mean(vv*vv)+.25*max(vv*vv)+1e-4*float(u@u)/323)/normal
    k=int(np.argmax(vv*vv));gradient=(.1*G.T@gg+2e-4*u/323+2*D.T@vv/len(vv)+.5*vv[k]*D[k])/normal
    close(f'iter{it}_solver_objective',objective,solver['returned_objective']);close(f'iter{it}_solver_gradient',gradient,solver['returned_gradient'],1e-11);ck(f'iter{it}_return_motion',valid(j['Z']+j['step'].reshape(19,17)));ck(f'iter{it}_trust_radius',np.max(abs(j['step']))<=radius+1e-12)

delivery=ROOT/'ff_prior_balanced_delivery/static_body_completion';original=UNIT/'body_diagnostics';suffix=delivery/'suffix';dm=read(delivery/'body_obstacle_metadata.json');om=read(original/'body_obstacle_metadata.json');sm=read(suffix/'body_obstacle_metadata.json');reservation=read(delivery/'reservation.json')
ck('original_partial_retained',om['status']=='PARTIAL' and om['samples']==9912 and om['failure']=="TimeoutError('BODY_DIAGNOSTIC_WALL_BUDGET')")
ck('completion_metadata_identity',dm['source_native_sha256']==sm['source_native_sha256']==om['source_native_sha256']==track(UNIT/'execution/native_trace.npz') and dm['original_diagnostic_metadata_sha256']==track(original/'body_obstacle_metadata.json'))
ck('full_completion_status',dm['status']=='COMPLETE' and dm['samples']==dm['expected_samples']==13751 and dm['failure'] is None)
for label,folder,m in [('original',original,om),('suffix',suffix,sm),('complete',delivery,dm)]:
    for name,desc in m['files'].items():ck(f'{label}_{name}_hash_size',track(folder/name)==desc['sha256'] and (folder/name).stat().st_size==desc['bytes'])
    ck(label+'_no_new_native_or_coupled',m['new_native_steps']==m['new_coupled_calls']==0)
ck('completion_topology',om['links']==sm['links']==dm['links'] and om['obstacles']==sm['obstacles']==dm['obstacles'] and om['geom_names']==sm['geom_names']==dm['geom_names'])
full=load(delivery/'body_obstacle_trace.npz');prefix=load(original/'body_obstacle_trace.npz');tail=load(suffix/'body_obstacle_trace.npz')
ck('all_completion_keys',set(full)==set(prefix)==set(tail))
for key in full:
    ck(key+'_full_prefix_exact',np.array_equal(full[key][:9912],prefix[key],equal_nan=True));ck(key+'_full_suffix_exact',np.array_equal(full[key][9912:],tail[key],equal_nan=True))
    ck(key+'_finite_or_declared_nan',np.isnan(full[key]).all() if key in ['command_before','command_after'] else np.isfinite(full[key]).all())
del prefix,tail
with np.load(UNIT/'execution/native_trace.npz') as f:n={k:f[k] for k in ['time','clearance_m','ee']}
close('full_body_time',full['time'],n['time'],0);close('full_body_native_index',full['native_index'],np.arange(13751),0)
events=read(delivery/'body_filter_events.json');op=read(original/'body_filter_events.json');sp=read(suffix/'body_filter_events.json');ck('all_event_concat',events==op+sp);ck('full_events_time_index',all(e['time']==full['time'][i] and e['native_index']==i for i,e in enumerate(events)))
ck('no_actual_online_filter_events',all(e['active_constraints']==[] and e['command_available'] is False and e['filtered_command_available'] is False for e in events))
ck('all_events_describe_static_no_filter',all('no online geometry velocity filter' in e['filter_reason'] for e in events))
dist=full['distance_m'];geom=full['nearest_geom_id'];connect=full['connector'];ck('full_body_shape',dist.shape==(13751,7,71));ck('all_body_nonpenetrating',np.all(dist>=0))
close('all_connector_lengths',np.linalg.norm(connect[...,1,:]-connect[...,0,:],axis=-1),dist,2e-12)
for li,link in enumerate(dm['links']):
    ck(f'all_nearest_geom_in_link{li}',np.isin(geom[:,:,li],link['geom_ids']).all());ck(f'link{li}_bodyid',all(meta['geom_bodyid'][g]==link['body_id'] for g in link['geom_ids']))
links=dm['links'];obs=dm['obstacles'];bodyid=dm['continuum_body_ids'];shape=full['body_positions'][:,bodyid];segments=np.diff(shape,axis=1);length=np.linalg.norm(segments,axis=-1);a,b=segments[:,:-1],segments[:,1:];angle=np.arctan2(np.linalg.norm(np.cross(a,b),axis=-1),np.sum(a*b,axis=-1));scale=.5*(length[:,:-1]+length[:,1:]);curv=np.divide(angle,scale,out=np.zeros_like(angle),where=scale>1e-12)
close('all_curvature_rederived',curv,full['continuum_curvature'],1e-14)
ck('all_body_minima_ge_whole_machine',np.all(dist.min(axis=(1,2))>=n['clearance_m']-1e-12))
# Final body origins agree with saved EE pose origins, accounting for continuum tip offset.
cb=dm['body_names'].index('link_30');rb=dm['body_names'].index('end_effector_r');tip=n['ee'][:,0,:3,3]-np.einsum('nij,j->ni',n['ee'][:,0,:3,:3],np.array([.0475,0,0]));close('all_continuum_tip_origin',full['body_positions'][:,cb],tip,1e-12);close('all_rigid_tip_origin',full['body_positions'][:,rb],n['ee'][:,1,:3,3],1e-12)
queries_per_state=len(obs)*sum(len(x['geom_ids']) for x in links);ck('all_query_count_exact',dm['query_count']==queries_per_state*13751 and sm['query_count']==queries_per_state*3839 and om['query_count']==queries_per_state*9912)
rows=[]
for oi,ob in enumerate(obs):
    ti,li=np.unravel_index(dist[:,oi,:].argmin(),dist[:,oi,:].shape)
    rows.append(dict(obstacle=ob['object_id'],minimum_m=float(dist[ti,oi,li]),native_index=int(ti),time=float(full['time'][ti]),link=links[li]['body_name'],arm=links[li]['arm_name'],nearest_geom=dm['geom_names'][int(geom[ti,oi,li])],samples_with_any_link_within_display_10mm=int((dist[:,oi,:].min(axis=1)<=.01).sum())))
OUT['body_observation']=dict(samples=13751,prefix_samples=9912,suffix_samples=3839,links=len(links),obstacles=len(obs),geometries_per_state=queries_per_state,all_distance_values=dist.size,all_connector_values=connect.size,full_exact_queries=dm['query_count'],additional_exact_queries=sm['query_count'],original_partial_status=om['status'],original_elapsed_s=om['elapsed_s'],additional_wall_s=dm['additional_wall_s'],combined_elapsed_field_inherited_not_total=dm['elapsed_s'],obstacle_minima=rows,maximum_discrete_curvature=float(curv.max()),no_online_filter=True)
close('body_global_minimum_matches_execution',dist.min(),report['minimum_clearance_m'],0)

# Unique completed resume: verify full manifest against present immutable run, raw bytes intact.
resume=ROOT/'ff_prior_balanced_restore/evidence/completed_001';before=read(resume/'core_before.json');after=read(resume/'core_after.json');receipt=read(resume/'receipt.json')
ck('resume_hash_size_manifests_identical',before['files']==after['files'] and before['bytes']==after['bytes']);actualfiles={str(p.relative_to(RUN)).replace('\\','/') for p in RUN.rglob('*') if p.is_file()};ck('resume_manifest_complete_file_set',actualfiles==set(before['files']))
for rel,info in before['files'].items():ck('resume_file_'+rel,track(RUN/rel)==info['sha256'] and (RUN/rel).stat().st_size==info['bytes'])
ck('resume_totals',len(before['files'])==receipt['core_files']==100 and sum(v['bytes'] for v in before['files'].values())==receipt['core_bytes']==570527800)
for stream in ['stdout','stderr']:ck('resume_raw_'+stream,track(resume/(stream+'.bin'))==receipt[stream]['sha256'] and (resume/(stream+'.bin')).stat().st_size==receipt[stream]['bytes'])
ck('resume_no_work_completed',receipt['status']=='PASS' and receipt['returncode']==0 and receipt['response']['resume']=='NO_WORK_ALREADY_COMPLETE' and receipt['exactly_one_invocation'] and not receipt['new_physical_work_allowed'])
OUT['resume']=dict(status=receipt['status'],files=receipt['core_files'],bytes=receipt['core_bytes'],stdout_bytes=receipt['stdout']['bytes'],stderr_bytes=receipt['stderr']['bytes'],independent_reviewer_did_not_resume=True)
OUT['checks']=CHECKS;OUT['summary']=dict(checks=len(CHECKS),passed=sum(x['passed'] for x in CHECKS),failed=[x for x in CHECKS if not x['passed']])
(HERE/'completion_numerical_audit.json').write_text(json.dumps(OUT,default=plain,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8');(HERE/'inputs_completion.json').write_text(json.dumps(dict(generated_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),files=list(INPUTS.values())),ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(OUT['summary'],default=plain,ensure_ascii=False,indent=2));print(json.dumps(OUT['body_observation'],default=plain,ensure_ascii=False,indent=2))
