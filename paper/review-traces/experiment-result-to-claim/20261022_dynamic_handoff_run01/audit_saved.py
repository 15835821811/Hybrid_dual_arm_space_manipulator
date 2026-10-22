"""Read-only saved-data/NN/algebra audit. Writes only beside this script.

No MjModel/MjData construction, stepping, kinematics, optimizer, training or resume.
"""
import os
os.environ['OMP_NUM_THREADS']='1'
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['MKL_NUM_THREADS']='1'
import sys, json, hashlib, zipfile, datetime, pathlib, xml.etree.ElementTree as ET
import numpy as np
from scipy.spatial.transform import Rotation

OUT=pathlib.Path(__file__).resolve().parent
ROOT=OUT.parents[3]
sys.path.insert(0,str(ROOT))
RUN=ROOT/'artifacts/v7_1/dynamic_handoff_001'
from ff_prior.storage import digest
from ff_prior.robot_adapter import physical_spec
from v6_lite.irregular_waypoints import IrregularWaypointTarget

def read(p): return json.loads(pathlib.Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
    h=hashlib.sha256()
    with pathlib.Path(p).open('rb') as f:
        while b:=f.read(1024*1024): h.update(b)
    return h.hexdigest()
def save(name,value):
    (OUT/name).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def maximum(x): return float(np.max(np.abs(x)))
def stat(x): return {k:v.tolist() for k,v in {'mean':np.mean(x,axis=0),'p95':np.percentile(x,95,axis=0),'maximum':np.max(x,axis=0),'final':x[-1]}.items()}
def intervals(t,m):
    ids=np.flatnonzero(m)
    groups=np.split(ids,np.flatnonzero(np.diff(ids)>1)+1) if len(ids) else []
    return [{'first_s':float(t[g[0]]),'last_s':float(t[g[-1]]),'frames':len(g),'sample_span_s':float(t[g[-1]]-t[g[0]])} for g in groups]

manifest=read(RUN/'run.json'); ident=manifest['identity']; summary=read(RUN/'summary.json')
reviewed=set(RUN.rglob('*'))
reviewed={p for p in reviewed if p.is_file()}
source=ident['source_hashes']; binding=ident['binding']['files']
source_mismatch=[p for p,h in source.items() if sha(ROOT/p)!=h]
binding_mismatch=[p for p,h in binding.items() if sha(ROOT/p)!=h]
reviewed.update(ROOT/p for p in source);reviewed.update(ROOT/p for p in binding)
with zipfile.ZipFile(RUN/'executed_source.zip') as z:
    zip_mismatch=[p for p,h in source.items() if hashlib.sha256(z.read(p)).hexdigest()!=h]
    zip_set_equal=set(z.namelist())==set(source)
audit={'source_count':len(source),'binding_count':len(binding),'producer_head':manifest['producer_head'],
       'source_mismatches':source_mismatch,'binding_mismatches':binding_mismatch,'zip_hash_mismatches':zip_mismatch,
       'zip_exact_source_set':zip_set_equal,'scope':'saved arrays, frozen source reading, pure algebra and neural inference only',
       'additional_physics_steps':0,'additional_optimizer_calls':0,'additional_training_updates':0,'units':[]}
assert not (source_mismatch or binding_mismatch or zip_mismatch) and zip_set_equal
spec=physical_spec(); B=spec.planner_to_low_level; Bplus=spec.low_level_to_planner
urdf=ET.parse(spec.source_urdf).getroot()
jlimits={j.attrib['name']: (None if j.attrib['type'] in ('continuous','floating','fixed') else
    (float(j.find('limit').attrib['lower']),float(j.find('limit').attrib['upper']))) for j in urdf.findall('joint')}
unit_lookup={x['unit']['id']:x for x in summary['results']}
for row in ident['selection']['units']:
    uid=row['id']; d=RUN/uid; result=read(d/'result.json'); identity=read(d/'identity.json'); task=read(d/'task.json')
    initial=np.load(d/'initial.npz')['values']; old=ROOT/'artifacts/v7_1/waypoint_domain_002'/row['scene']/'task_controller/task.json'
    robot=identity['robot']; index=robot['joint_index_map']; names=robot['robot']['low_level_joint_names']
    qids=np.array([index[n]['qpos'] for n in names]); vids=np.array([index[n]['dof'] for n in names]); acts=np.array([index[n]['actuator'] for n in names])
    ur={'id':uid,'scene':row['scene'],'method':row['method'],'proposal_seed':row['proposal_seed'],
        'new_execution':result['new_execution'],'status':result['status'],
        'task_same_as_frozen_original':digest(task)==digest(read(old))==row['task_hash']==identity['task_hash'],
        'initial_matches_task':np.array_equal(initial,task['initial']),
        'initial_id_matches':digest({'values':initial,'model_hash':robot['scene_model_hash'],'state_mask':8191})==identity['initial_state_id']==task['initial_id']}
    assert ur['task_same_as_frozen_original'] and ur['initial_matches_task'] and ur['initial_id_matches']
    if not result['new_execution']:
        src=ROOT/result['reused_from']; sr=read(src/'result.json'); si=read(src/'identity.json'); reuse=read(d/'reuse.json')
        ur.update(reused_from=result['reused_from'],actual_new_steps=result['actual_new_steps'],
          reused_task_exact=digest(read(src/'task.json'))==digest(task),
          reused_initial_exact=np.array_equal(np.load(src/'initial.npz')['values'],initial),
          reused_robot_exact=digest(si['robot'])==digest(robot),
          reused_trace_hash_exact=sha(src/'native_trace.npz')==result['native_trace_sha256']==reuse['source_control_sha256'],
          reused_metrics_gates_exact=all(result[k]==v for k,v in sr.items() if k not in ('new_execution','actual_new_steps')),
          new_independent_replication=reuse['new_independent_replication'])
        audit['units'].append(ur);continue
    h={k:v for k,v in np.load(d/'native_trace.npz').items()}; rp={k:v for k,v in np.load(d/'replay_trace.npz').items()}
    geo={k:v for k,v in np.load(d/'geometry.npz').items()}; pairs=read(d/'geometry_pairs.json')
    t=h['time']; pos=np.linalg.norm(h['ee'][...,:3,3]-h['desired'][...,:3,3],axis=-1)
    rot=Rotation.from_matrix((np.swapaxes(h['desired'][...,:3,:3],-1,-2)@h['ee'][...,:3,:3]).reshape(-1,3,3)).magnitude().reshape(pos.shape)
    lowq=h['qpos'][:,qids]; lowv=h['qvel'][:,vids]; v=lowv@Bplus.T; a=np.diff(v,axis=0)/.002; la=np.diff(lowv,axis=0)/.002
    lo=np.array(robot['planner_lower']);hi=np.array(robot['planner_upper']); hold=t>=25.5-1e-9
    mp=np.array([np.max(np.linalg.norm(h['momentum'][:,s],axis=1)) for s in (slice(0,3),slice(3,6))]);me=np.array([np.max(np.linalg.norm(h['exchange'][:,s],axis=1)) for s in (slice(0,3),slice(3,6))])
    lowlimits=all(jlimits[n] is None or (np.all(lowq[:,j]>=jlimits[n][0]) and np.all(lowq[:,j]<=jlimits[n][1])) for j,n in enumerate(names))
    replay_error=max(maximum(h[k]-rp[k]) for k in ('qpos','qvel','integration','ctrl'))
    gates={'complete_native_time':len(h['ctrl'])==13500 and result['stop_reason'] is None and abs(t[-1]-27)<1e-8,
      'independent_replay':replay_error<=1e-10,'whole_body_clearance':np.min(h['clearance'])>=0,
      'geometry_step':result['maximum_geometry_step_m']<=.002+1e-12,
      'dynamic_position':np.all(pos<=.005),'dynamic_rotation':np.all(rot<=np.pi/180),
      'relative_hold':np.all(pos[hold]<=.005) and np.all(rot[hold]<=np.pi/180),
      'joint_limits':np.all(h['q17']>=lo) and np.all(h['q17']<=hi) and np.all(h['qref']>=lo) and np.all(h['qref']<=hi) and lowlimits,
      'derivative_limits':np.all(abs(v)<=spec.planner_velocity_limits) and np.all(abs(a)<=spec.planner_acceleration_limits) and np.all(abs(lowv)<=abs(B)@spec.planner_velocity_limits) and np.all(abs(la)<=abs(B)@spec.planner_acceleration_limits) and np.all(abs(h['reference_qd'])<=spec.planner_velocity_limits) and np.all(abs(h['reference_qdd'])<=spec.planner_acceleration_limits),
      'raw_torque':np.all(abs(h['raw_torque'])<=spec.torque_limits+1e-12),
      'tracking':maximum(h['q17']-h['qref'])<=.02,'manifold':maximum(lowq-h['q17']@B.T)<=.02,
      'momentum':np.all(mp<=1e-5+.002*me),'finite':all(np.isfinite(v).all() for v in h.values()),
      'no_hidden_reset':replay_error==0 and maximum(np.diff(t)-.002)<1e-10,
      'no_external_wrench':maximum(h['integration'][:,307:])==0}
    gates={k:bool(v) for k,v in gates.items()}
    w=task['waypoints']; path=IrregularWaypointTarget(np.array(w['initial_position_w']),np.array(w['waypoint_points_m']),np.array(w['target_rotation_world']),w['transition_duration_s'],w['path_duration_s'],np.array(w['segment_durations_s']),{})
    expected_c=np.array([path.sample(tt)[0] for tt in t])
    # Free target qpos is world position plus wxyz quaternion at qpos[74:81].
    rt=Rotation.from_quat(h['qpos'][:,[78,79,80,77]]).as_matrix();home=h['ee'][0,1,:3,3];rh=h['ee'][0,1,:3,:3]
    s=np.clip(t/4.5,0,1);alpha=s**3*(10-15*s+6*s*s)
    point=h['qpos'][:,74:77]+np.einsum('nij,j->ni',rt,np.array(task['grasp_local']))
    expected_r=home+alpha[:,None]*(point-home)
    rr=rt@rt[0].T@rh;rv=Rotation.from_matrix(rr@rh.T).as_rotvec();expected_rr=Rotation.from_rotvec(alpha[:,None]*rv).as_matrix()@rh
    geometry_aggregate=np.r_[geo['minima'][0],np.minimum(geo['minima'][1::2],geo['minima'][2::2])]
    midpoint=(h['qpos'][:-1]+h['qpos'][1:])/2
    for start in (3,77):
        r0=Rotation.from_quat(h['qpos'][:-1][:,[start+1,start+2,start+3,start]])
        r1=Rotation.from_quat(h['qpos'][1:][:,[start+1,start+2,start+3,start]])
        rm=(r0*Rotation.from_rotvec((r0.inv()*r1).as_rotvec()/2)).as_quat()[:,[3,0,1,2]]
        rm*=np.where(np.sum(rm*h['qpos'][:-1,start:start+4],axis=1)<0,-1,1)[:,None]
        midpoint[:,start:start+4]=rm
    filterlog=read(d/'control_filter.json'); ft=np.array([x['time'] for x in filterlog]);fd=np.array([x['desired'] for x in filterlog]);ff=np.array([x['filtered'] for x in filterlog]);delta=np.linalg.norm(ff-fd,axis=1)
    ramp=np.vstack([np.zeros((1,17)),ff[:-1]])[:,None,:]+np.arange(1,11)[None,:,None]/10*(ff-np.vstack([np.zeros((1,17)),ff[:-1]]))[:,None,:]
    filter_audit={'ticks':len(filterlog),'timestamps_match_native':bool(maximum(ft-t[:-1:10])<1e-10),
      'delta_recompute_error':maximum(delta-np.array([x['delta'] for x in filterlog])),
      'filtered_command_ramp_error':maximum(ramp.reshape(-1,17)-h['reference_qd']),
      'qref_integrated_from_ramp_error':maximum(np.diff(h['qref'],axis=0)-.002*h['reference_qd']),
      'active_intervals':intervals(ft,delta>1e-8),'maximum_delta':float(np.max(delta)),
      'max_delta_time_s':float(ft[np.argmax(delta)]),'failed_position_intervals':intervals(t,np.any(pos>.005,axis=1)),
      'peak_position_time_s':t[np.argmax(pos,axis=0)].tolist()}
    if np.any(pos>.005):
        peak=int(np.argmax(pos[:,0]));k=min(peak//10,len(ft)-1)
        filter_audit.update(peak_control_tick_s=float(ft[k]),filter_delta_at_peak=float(delta[k]),
          max_command_deviation_at_peak_rad_s=float(np.max(abs(ff[k]-fd[k]))),
          clearance_at_peak_mm=float(h['clearance'][peak]*1000),
          causal_conclusion='Temporal association only; no counterfactual filter removal or alternate controller was run.')
    transitions=[]
    for boundary in (4.5,25.5):
        k=int(round(boundary/.002));tr=np.load(d/f'transition_{int(boundary*10)}.npz')
        transitions.append({'boundary_s':boundary,'index':k,'time_error':abs(float(t[k])-boundary),
          'slice_exact':all(np.array_equal(tr[key],h[key][k-1:k+2]) for key in tr.files),
          'replay_full_integration_slice_exact':np.array_equal(rp['integration'][k-1:k+2],h['integration'][k-1:k+2]),
          'base_qpos_change_across_sides':maximum(h['base'][k+1]-h['base'][k-1]),
          'qvel_change_across_sides':maximum(h['qvel'][k+1]-h['qvel'][k-1])})
    ur.update(steps=len(h['ctrl']),raw_shapes={k:list(v.shape) for k,v in h.items()},
      trace_sha_matches=sha(d/'native_trace.npz')==result['native_trace_sha256'],
      replay_full_error=replay_error,integration_initial_exact=np.array_equal(h['integration'][0],initial),
      integration_time_qpos_qvel_ctrl_exact=bool(np.array_equal(h['integration'][:,0],t) and np.array_equal(h['integration'][:,1:82],h['qpos']) and np.array_equal(h['integration'][:,82:161],h['qvel']) and np.array_equal(h['integration'][1:,240:307],h['ctrl'])),
      max_timestep_error_s=maximum(np.diff(t)-.002),all_finite=gates['finite'],
      position_mm=stat(pos*1000),rotation_deg=stat(np.rad2deg(rot)),
      metric_max_discrepancy=max(maximum(np.array(result['position_mm'][k])-v) for k,v in stat(pos*1000).items()),
      rotation_metric_max_discrepancy=max(maximum(np.array(result['rotation_deg'][k])-v) for k,v in stat(np.rad2deg(rot)).items()),
      gates=gates,gate_mismatches=[k for k,v in gates.items() if result['gates'][k]!=v],
      failed_gates=[k for k,v in gates.items() if not v],
      raw_torque_ratio=float(np.max(abs(h['raw_torque'])/spec.torque_limits)),raw_to_applied_error=maximum(h['raw_torque']-h['ctrl'][:,acts]),
      world_desired_algebra_errors={'continuum_position_m':maximum(expected_c-h['desired'][:,0,:3,3]),
        'continuum_rotation_matrix':maximum(h['desired'][:,0,:3,:3]-np.array(w['target_rotation_world'])),
        'rigid_position_m':maximum(expected_r-h['desired'][:,1,:3,3]),'rigid_rotation_matrix':maximum(expected_rr-h['desired'][:,1,:3,:3])},
      geometry_saved={'query_states':len(geo['qpos']),'pair_records':len(pairs),'native_states_exact':bool(np.array_equal(geo['qpos'][0],h['qpos'][0]) and np.array_equal(geo['qpos'][1::2],h['qpos'][1:])),
        'midpoint_max_error':maximum(geo['qpos'][2::2]-midpoint),'aggregate_minimum_error':maximum(geometry_aggregate-h['clearance']),
        'minimum_mm':float(np.min(geo['minima'])*1000),'reported_maximum_geometry_step_m':result['maximum_geometry_step_m'],
        'scope':'All saved scan states/minima and quaternion midpoint interpolation checked; frozen scan code/pair policy inspected. No new geometry queries or continuous-time proof.'},
      filter_audit=filter_audit,transitions=transitions,
      base_translation_range_m=np.ptp(h['base'][:,:3],axis=0).tolist(),
      actual_domain_margin_rad=float(min(np.min(h['q17']-lo),np.min(hi-h['q17']))),
      tracking_max_rad=maximum(h['q17']-h['qref']),manifold_max_rad=maximum(lowq-h['q17']@B.T))
    assert not ur['gate_mismatches'] and ur['trace_sha_matches'] and ur['integration_initial_exact']
    assert all(tr['slice_exact'] and tr['replay_full_integration_slice_exact'] for tr in transitions)
    audit['units'].append(ur)
    print(uid,result['status'],ur['position_mm']['maximum'],flush=True)

# Exact saved seed/condition/model DDIM reconstruction, no initializer or physics calls.
import torch
from ff_prior_latent.learning import Codec,LatentDenoiser
from ff_prior.diffusion import schedule,weight_hash
torch.set_num_threads(1)
modeldir=ROOT/'artifacts/v7_1/endpoint_jacobian_bounded_001/learning'
codec=Codec(read(modeldir/'codec.json')); ckpt=torch.load(modeldir/'model.pt',map_location='cpu',weights_only=False)
training=read(modeldir/'training.json');reviewed.add(modeldir/'training.json')
model=LatentDenoiser(codec.dim,len(codec.feature_mean));model.load_state_dict(ckpt['weights']);model.eval()
initial_weight=weight_hash(model)
assert sha(modeldir/'model.pt')==training['checkpoint_sha256'] and initial_weight==training['trained_weight_hash'] and ckpt['metadata']==training['metadata']
nn=[]
for row in ident['selection']['units']:
    uid=row['id'];p=RUN/uid/'prior.json'
    if not p.exists():continue
    saved=read(p);base=np.array(saved['base']);condition=saved['task']['condition'];center=codec.center(condition,base)
    c=torch.tensor(codec.condition(condition,base),dtype=torch.float32)[None]
    x=torch.randn((1,codec.dim),generator=torch.Generator().manual_seed(saved['seed'])).double()
    alpha=schedule().double();times=np.linspace(999,0,16).round().astype(int);steps=[];before=model.calls
    with torch.no_grad():
        for j,t in enumerate(times):
            pred=model(x.float(),torch.tensor([int(t)]),c).double();a=alpha[t];clean=a.sqrt()*x-(1-a).sqrt()*pred
            nexta=alpha[times[j+1]] if j<15 else torch.tensor(1.,dtype=torch.float64)
            eps=(x-a.sqrt()*clean)/(1-a).sqrt();res=float((x-a.sqrt()*clean-(1-a).sqrt()*eps).abs().max())
            x=nexta.sqrt()*clean+(1-nexta).sqrt()*eps;steps.append({'step':j,'t':int(t),'residual':res})
    z=codec.decode(codec.coefficients(x[0].numpy(),center),base)
    nn.append({'unit':uid,'seed':saved['seed'],'seed_matches':saved['seed']==row['proposal_seed'],
       'calls':model.calls-before,'Z_exact':np.array_equal(z,np.array(saved['Z'])),'Z_max_error':maximum(z-np.array(saved['Z'])),
       'DDIM_steps_exact':steps==saved['steps'],'maximum_update_residual':max(s['residual'] for s in steps),
       'condition_dimension':len(codec.feature_mean),'latent_dimension':codec.dim,'decoded_shape':list(z.shape),
       'raw_Z_dynamic_validation':'NOT_PERFORMED; not MODEL_VALID'})
    print('NN',uid,nn[-1]['Z_max_error'],flush=True)
audit['neural_reconstruction']={'samples':nn,'audit_inference_calls':model.calls,'frozen_experiment_calls':summary['network_calls'],
     'weights_unchanged':initial_weight==weight_hash(model),'new_training_updates':0,'codec_rank':codec.rank,
     'condition_scope':'65 = public endpoint condition 48 + initializer terminal delta 17; no complete waypoint/target-motion/obstacle condition.'}
audit['denominators']={'unique_executions':sum(x['new_execution'] for x in audit['units']),
    'unique_passed':sum(x['new_execution'] and x['status']=='EXECUTION_VERIFIED' for x in audit['units']),
    'logical_units':len(audit['units']),'logical_passed':sum(x['status']=='EXECUTION_VERIFIED' for x in audit['units']),
    'shared_baselines':sum(not x['new_execution'] for x in audit['units']),
    'method_scene':[{ 'scene':scene,'method':method,'unique_executed':sum(x['new_execution'] for x in audit['units'] if x['scene']==scene and x['method']==method),
      'unique_passed':sum(x['new_execution'] and x['status']=='EXECUTION_VERIFIED' for x in audit['units'] if x['scene']==scene and x['method']==method),
      'logical_passed':sum(x['status']=='EXECUTION_VERIFIED' for x in audit['units'] if x['scene']==scene and x['method']==method)}
      for scene in ('stationary','drifting') for method in ('task_controller','diffusion_null_prior')]}
audit['local_engineering_gate_passed']=all(x['unique_passed']>=1 for x in audit['denominators']['method_scene'])
audit['budget']=read(RUN/'budget.json');audit['status']=read(RUN/'status.json')
audit['saved_final_bytes']=sum(p.stat().st_size for p in RUN.rglob('*') if p.is_file())
audit['paper_claim_audit']='unavailable' if not (ROOT/'paper/PAPER_CLAIM_AUDIT.json').exists() else read(ROOT/'paper/PAPER_CLAIM_AUDIT.json')
audit['qualification_limit']='Saved execution/replay audit; no new numerical simulation, geometry query, optimizer, training, TEST, resume, or historical physical reaudit.'
save('audit.json',audit)
inputs=[]
for p in sorted(reviewed):
    inputs.append({'path':str(p),'sha256':sha(p),'bytes':p.stat().st_size,'modified_utc':datetime.datetime.fromtimestamp(p.stat().st_mtime,datetime.timezone.utc).isoformat()})
save('inputs.json',{'review_scope':'dynamic_handoff_001 new raw evidence plus frozen source/input identity only; historical physical results not reaudited',
 'reviewed_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'files':inputs})
print('COMPLETE',audit['denominators'],'local gate',audit['local_engineering_gate_passed'],flush=True)
