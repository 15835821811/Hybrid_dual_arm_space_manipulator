"""Finish metadata/coordinate provenance checks; no simulation or project imports."""
from pathlib import Path
import json, hashlib, datetime, xml.etree.ElementTree as ET
import numpy as np
OUT=Path(__file__).resolve().parent;ROOT=Path(__file__).resolve().parents[4];RUN=ROOT/'artifacts/v7_1/balanced_tracking_001'
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
ck('raw_worker_stderr_empty',load(RUN/'worker.stderr.log')==b'')
summary=read(RUN/'summary.json');vis=summary['visualization']
ck('original_media_partial_preserved',vis['status']=='PARTIAL_MEDIA')
ck('original_video_time_mismatch_explicit',all(x['fps']==15 and x['frames']==820 and abs(x['playback_duration_s']-820/15)<1e-12 and abs(x['recorded_duration_s']-47.16)<1e-8 for x in vis['videos']))
audit['supplement']={'coordinate_map':derived,'body_minima':bodymin,'queried_at_90mm_cap':capped,'constraint_rows_per_tick_min':int(counts.min()),'constraint_rows_per_tick_max':int(counts.max()),'production_events':len(events),'original_media_status':vis['status'],'original_video_playback_s':820/15,'original_recorded_combined_duration_s':vis['combined_actual_duration_s'],'new_physics_steps':0,'new_geometry_queries':0}
audit['checks'].extend(checks);audit['checks_total']=len(audit['checks']);audit['checks_passed']=sum(x['pass'] for x in audit['checks']);audit['checks_failed']=sum(not x['pass'] for x in audit['checks']);audit['failures']=[x for x in audit['checks'] if not x['pass']]
for name,data in [('numerical_audit.json',audit),('inputs.json',manifest)]: (OUT/name).write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
print(json.dumps({'checks':audit['checks_total'],'failed':audit['failures'],'supplement':audit['supplement']},ensure_ascii=False,indent=2))
