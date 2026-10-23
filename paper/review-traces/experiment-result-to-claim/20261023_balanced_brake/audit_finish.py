"""Finish metadata/coordinate provenance checks; no simulation or project imports."""
from pathlib import Path
import json, hashlib, datetime, xml.etree.ElementTree as ET
import numpy as np
OUT=Path(__file__).resolve().parent;ROOT=Path(__file__).resolve().parents[4];RUN=ROOT/'artifacts/v7_1/balanced_brake_001'
audit=json.loads((OUT/'numerical_audit.json').read_text(encoding='utf-8'));manifest=json.loads((OUT/'inputs.json').read_text(encoding='utf-8'));checks=[]
def ck(name,yes,detail=None):
    x={'name':name,'pass':bool(yes)}
    if detail is not None:x['detail']=detail
    checks.append(x)
def load(p):
    p=Path(p);b=p.read_bytes();manifest['files'][p.relative_to(ROOT).as_posix()]={'sha256':hashlib.sha256(b).hexdigest(),'bytes':len(b),'mtime_ns':p.stat().st_mtime_ns};return b
def read(p):return json.loads(load(p))
bm=read(RUN/'execution/body_obstacle_metadata.json');cm=read(RUN/'execution/controller_clearance_metadata.json')
xml=ET.fromstring(load(ROOT/'dual_arm_space_robot_2026/urdf/dual_arm_space_robot_2026.urdf'))
bychild={j.find('child').get('link'):j for j in xml.findall('joint')}
qposadr=qveladr=0;derived=[]
for name in bm['body_names'][1:]:
    if name not in bychild:continue
    j=bychild[name];typ=j.get('type')
    if typ=='floating':qposadr+=7;qveladr+=6
    elif typ in ['continuous','revolute','prismatic']:
        limit=j.find('limit');derived.append({'name':j.get('name'),'body':name,'type':typ,'qpos':qposadr,'qvel':qveladr,'lower':float(limit.get('lower')) if limit is not None and 'lower' in limit.attrib else None,'upper':float(limit.get('upper')) if limit is not None and 'upper' in limit.attrib else None});qposadr+=1;qveladr+=1
ck('coordinate_map_derived_from_frozen_urdf_and_saved_body_order',[x['qpos'] for x in derived]==list(range(7,74)) and [x['qvel'] for x in derived]==list(range(6,73)) and qposadr==81 and qveladr==79)
with np.load(RUN/'execution/native_trace.npz') as n:
    ck('all_scalar_joint_limits_in_derived_order',all(x['type']=='continuous' or (np.all(n['qpos'][:,x['qpos']]>=x['lower']) and np.all(n['qpos'][:,x['qpos']]<=x['upper'])) for x in derived))
    # Orthogonality, homogeneous frames, and unit quaternion checks address all saved pose elements.
    for key in ['ee','desired']:
        R=n[key][:,:,:3,:3];err=np.max(abs(R@np.swapaxes(R,-1,-2)-np.eye(3)))
        ck('all_pose_rotation_matrices:'+key,err<1e-12 and np.all(abs(np.linalg.det(R)-1)<1e-12),{'max_orthogonality_error':float(err)})
        ck('all_pose_homogeneous_rows:'+key,np.array_equal(n[key][:,:,3,:],np.broadcast_to([0.,0.,0.,1.],n[key][:,:,3,:].shape)))
    for a,b in [(3,7),(77,81)]:ck('all_native_unit_quaternions:'+str(a),np.all(abs(np.linalg.norm(n['qpos'][:,a:b],axis=1)-1)<1e-12))
with np.load(RUN/'execution/body_obstacle_trace.npz') as b:
    bodymin=[]
    for oi,obs in enumerate(bm['obstacles']):
        ti,li=np.unravel_index(np.argmin(b['distance_m'][:,oi,:]),b['distance_m'][:,oi,:].shape)
        bodymin.append({'obstacle':obs['object_id'],'minimum_m':float(b['distance_m'][ti,oi,li]),'absolute_time':float(b['time'][ti]),'native_index':int(b['native_index'][ti]),'body':bm['links'][li]['body_name'],'geom':bm['geom_names'][b['nearest_geom_id'][ti,oi,li]],'display_band_samples_under_10mm':int(np.sum(np.min(b['distance_m'][:,oi,:],axis=1)<.01))})
with np.load(RUN/'execution/controller_clearance.npz') as c:
    capped=int(np.sum(c['exact_query_mask'] & (c['distance_or_lower_bound_m']>=.09-1e-15)))
    ck('no_active_distance_at_query_cap',not np.any((c['disposition']==2)&(c['distance_or_lower_bound_m']>=.09-1e-15)))
    counts=c['row_offsets'][1:]-c['row_offsets'][:-1]
events=[json.loads(line) for line in load(RUN/'events.jsonl').decode().splitlines()]
ck('all_event_timestamps_monotonic',all(events[i]['timestamp']>=events[i-1]['timestamp'] for i in range(1,len(events))))
ck('raw_stdout_equals_events',load(RUN/'events.jsonl')==load(RUN/'worker.stdout.log'))
ck('raw_worker_stderr_preserved_warning',len(load(RUN/'worker.stderr.log'))==320 and load(RUN/'worker.stderr.log').count(b'RuntimeWarning:')==1)
summary=read(RUN/'summary.json');vis=summary['visualization']
ck('original_media_partial_preserved',vis['status']=='PARTIAL_MEDIA' and vis['body_figures']['source_statuses'][0]['samples']==9912 and vis['body_figures']['source_statuses'][1]['samples']==1351)
ck('original_video_time_mismatch_explicit',all(x['fps']==15 and x['frames']==820 and abs(x['playback_duration_s']-820/15)<1e-12 and abs(x['recorded_duration_s']-54.5)<1e-8 for x in vis['videos']))
audit['supplement']={'coordinate_map':derived,'body_minima':bodymin,'queried_at_90mm_cap':capped,'constraint_rows_per_tick_min':int(counts.min()),'constraint_rows_per_tick_max':int(counts.max()),'production_events':len(events),'original_media_status':vis['status'],'original_video_playback_s':820/15,'original_recorded_combined_duration_s':vis['combined_actual_duration_s'],'new_physics_steps':0,'new_geometry_queries':0}

run=read(RUN/'run.json');identity=run['identity'];contract=identity['contract']
from subprocess import run as process_run
base=process_run(['git','show',run['producer_head']+':v6_lite/hierarchical_qp.py'],cwd=ROOT,capture_output=True,check=True).stdout
actual_hash=hashlib.sha256(load(ROOT/'v6_lite/hierarchical_qp.py')).hexdigest()
ck('actual_user_hierarchical_qp_archived',identity['source_hashes']['v6_lite/hierarchical_qp.py']==actual_hash)
audit['supplement']['source_identity']={'producer_head':run['producer_head'],'actual_source_count':len(identity['source_hashes']),'actual_source_bundle_hash':contract['dynamic_source_bundle_hash'],'executed_source_zip_sha256':hashlib.sha256(load(RUN/'executed_source.zip')).hexdigest(),'hierarchical_qp_actual_sha256':actual_hash,'hierarchical_qp_commit_sha256':hashlib.sha256(base).hexdigest(),'hierarchical_qp_differs_from_commit':actual_hash!=hashlib.sha256(base).hexdigest(),'not_pure_git_HEAD':True}
for filename,count in [('software_tests_source_only.json',18),('software_tests.json',26)]:
    sw=read(ROOT/'ff_prior_balanced_brake/evidence'/filename)
    ck('saved_software_result_'+filename,sw['failures']==sw['errors']==0 and ('Ran '+str(count)+' tests') in sw['output'] and all(sw[k]==0 for k in ['new_coupled_calls','new_native_steps','new_neural_samples','new_geometry_queries']))
    ck('software_source_identity_'+filename,sw['source_hashes']==identity['source_hashes'])
auth=read(ROOT/'ff_prior_balanced_brake/evidence/authorization.json');bud=read(ROOT/'ff_prior_balanced_brake/evidence/budget_derivation.json');binding=read(ROOT/'ff_prior_balanced_brake/evidence/binding.json')
ck('authorization_one_trial_only',auth['physical_trial_authorized'] and auth['candidate_denominator']==1 and not any(auth[k] for k in ['training','new_neural_sampling','advantage_comparison','TEST','expansion']))
ck('complete_frozen_binding_identity',binding==identity['binding'] and len(binding['files'])==265)
for family,caps in [('D3_after_caps','original_D3_caps'),('conservative_fullcycle_after','original_fullcycle_caps')]:
    ck('budget_below_'+caps,all(bud[family][k]<=v for k,v in bud[caps].items()))
prev=bud['previous_D3'];new=bud['new_repair'];media=bud['readonly_final_media'];cap=bud['D3_after_caps']
ck('new_D3_allocation_arithmetic',cap['bytes']==prev['core_bytes']+new['bytes']+media['bytes'] and cap['coupled']==prev['coupled']+new['coupled_limit'] and cap['native_including_replay']==prev['native_including_replay']+new['native_including_replay'] and abs(cap['wall_s']-prev['wall_s']-new['wall_s']-media['wall_s'])<1e-10)
failed=read(ROOT/'ff_prior_fullchain/evidence/balanced_tracking_closeout.json')
ck('failed_attempt_remains_closed_and_failed',failed['closed'] and failed['execution_status']=='EXECUTION_FAILED' and not failed['full_dynamic_pass'] and failed['completed_resume_calls']==1)
# Only predecessor completion metadata/receipt is read; no historical geometry raw.
static_delivery=read(ROOT/'ff_prior_balanced_delivery/readonly_completion_receipt.json')
static_meta=read(ROOT/'ff_prior_balanced_delivery/static_body_completion/body_obstacle_metadata.json')
final_delivery=read(ROOT/'ff_prior_balanced_delivery/final_brake_001/manifest.json')
audit['supplement']['inherited_static_completion']={'metadata_status':static_meta.get('status'),'metadata_samples':static_meta.get('samples'),'metadata_expected':static_meta.get('expected_samples'),'receipt':static_delivery,'independently_reaudited_historical_raw':False}
audit['supplement']['external_final_delivery']={k:final_delivery[k] for k in final_delivery if k not in ['files','media_frame_native_indices','native_indices','frame_times','body_frame_times','times','media_frame_times']}
# These two gates have no standalone native raw geometry poses or intervention ledger to fully regenerate.
result=read(RUN/'execution/result.json')
ck('geometry_step_report_threshold',result['maximum_geometry_step_m']<=.002+1e-12 and result['gates']['geometry_step'])
ck('no_hidden_reset_report_and_full_replay_source_consistent',result['gates']['no_hidden_reset'] and audit['metrics']['gates_reconstructed']['full_integration_replay'] and audit['metrics']['gates_reconstructed']['actual_terminal_handoff'] and audit['metrics']['gates_reconstructed']['absolute_time_continuity'])
audit['supplement']['independent_reconstruction_limits']=['Native geom_xpos/geom_xmat/geom radii sequence not saved: reported maximum geometry surface motion cannot be independently reconstructed without prohibited FK/geometry.','Recorded distance minima/witnesses/Jacobians/selection are exhaustively algebra-crosschecked; primitive distance/Jacobian correctness not re-derived from robot model.','Mass matrices/inertial quantities not saved: momentum and raw torque are checked from saved signals against gates, not independently recomputed from rigid-body dynamics.','no_hidden_reset is a source/replay/clock/terminal-supported provenance assertion, not a theorem inferable from state arrays alone.','SLSQP Hessian/KKT evidence not saved: feasibility and recorded decision logic checked; optimality not certified.','Historical 265-entry binding contains old raw hashes; historical raw are not reopened; predecessor qualification is inherited from closed review/metadata.']
audit['supplement']['no_new_work']={'native_steps':0,'replay_steps':0,'geometry_queries':0,'FK':0,'optimizer_calls':0,'training':0,'neural_samples':0,'resume_calls':0}
audit['checks'].extend(checks);audit['checks_total']=len(audit['checks']);audit['checks_passed']=sum(x['pass'] for x in audit['checks']);audit['checks_failed']=sum(not x['pass'] for x in audit['checks']);audit['failures']=[x for x in audit['checks'] if not x['pass']]
for name,data in [('numerical_audit.json',audit),('inputs.json',manifest)]: (OUT/name).write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
print(json.dumps({'checks':audit['checks_total'],'failed':audit['failures'],'supplement':audit['supplement']},ensure_ascii=False,indent=2))
