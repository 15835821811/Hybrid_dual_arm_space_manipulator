"""Saved-state discrete momentum defect analysis. Never advances a simulation."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='2'
from pathlib import Path
import sys,time,json,hashlib
import numpy as np
import mujoco
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
PKG=ROOT/'ff_prior_momentum_diagnosis'
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def norm2(a):return np.stack([np.linalg.norm(a[...,:3],axis=-1),np.linalg.norm(a[...,3:],axis=-1)],axis=-1)
def main():
 start=time.time();auth=read(PKG/'evidence/scope.json')
 def guard():
  if time.time()-start>900:raise TimeoutError('DIAGNOSTIC_900S')
 def forbidden(*a,**k):raise RuntimeError('NO_PHYSICAL_ADVANCE_IN_DIAGNOSIS')
 for n in ('mj_step','mj_step1','mj_step2','mj_Euler','mj_implicit','mj_RungeKutta','mj_integratePos','mj_forward'):
  if hasattr(mujoco,n):setattr(mujoco,n,forbidden)
 from ff_prior.robot_adapter import RobotAdapter
 from ff_prior.scenes import SceneSpec
 contract=read(ROOT/'ff_prior_satellite_origin/evidence/contract.json')
 r=RobotAdapter(scene=SceneSpec.from_dict(contract['scene']));m=r.model;d=mujoco.MjData(m)
 ident=r.identity()
 if ident['scene_model_hash']!=contract['scene_model_hash']:raise ValueError('MODEL_IDENTITY')
 b=r.base_dofs;j=r.dof_ids;mass=np.empty((m.nv,m.nv));bias=np.zeros(m.nv);disp=np.zeros(m.nv)
 results=[];handoffs=[];prev_tail=None;calls=0
 for idx,rel in enumerate(auth['source_native_files']):
  guard();path=ROOT/rel;before=sha(path)
  with np.load(path,allow_pickle=False) as z:
   keys=['time','qpos','qvel','integration']
   a={k:z[k] for k in keys};key='momentum' if 'momentum' in z.files else 'base_generalized_momentum'
   saved=z[key];saved_exchange=z['exchange'] if 'exchange' in z.files else None
  if prev_tail is not None:
   handoffs.append({'from':auth['source_native_files'][idx-1],'to':rel,'full836_equal':bool(np.array_equal(prev_tail['state'],a['integration'][0])),'momentum_equal':bool(np.array_equal(prev_tail['momentum'],saved[0]))})
  prev_tail={'state':a['integration'][-1].copy(),'momentum':saved[-1].copy()}
  # Full INTEGRATION arrays used only for boundary identity, not a new replay.
  del a['integration']
  world=[];gen=[];exchange=[];velocity_defect=[];transport_defect=[];delta=[];q_update=[];reconstruct=[]
  last=None
  for k,(q,v) in enumerate(zip(a['qpos'],a['qvel'])):
   if k%2500==0:guard();print(json.dumps({'source':idx,'state':k,'total':len(saved)}),flush=True)
   # Read-only detached MjData, no live Environment and no integration.
   d.qpos[:]=q;d.qvel[:]=v
   mujoco.mj_kinematics(m,d);mujoco.mj_comPos(m,d);mujoco.mj_crb(m,d);mujoco.mj_comVel(m,d);mujoco.mj_rne(m,d,0,bias)
   mujoco.mj_fullM(m,mass,d.qM);calls+=1
   pb=mass[b]@v;ex=mass[np.ix_(b,j)]@v[j]
   rot=d.xmat[r.base_body].reshape(3,3).copy();pos=d.xpos[r.base_body].copy()
   cross=np.array([[0,-pos[2],pos[1]],[pos[2],0,-pos[0]],[-pos[1],pos[0],0]])
   A=np.zeros((6,6));A[:3,:3]=np.eye(3);A[3:,:3]=cross;A[3:,3:]=rot
   G=A@mass[b];pworld=G@v;cworld=A@bias[b]
   world.append(pworld);gen.append(pb);exchange.append(ex)
   reconstruct.append(float(np.max(abs(pb-saved[k]))))
   if last is not None:
    G0,v0,c0,q0,t0,p0=last;h=a['time'][k]-t0
    # Identity: dP = G0*(v1-v0) + (G1-G0)*v1.
    # The +/- h*A*c split exposes implicit velocity solve vs transport truncation.
    vel=G0@(v-v0)+h*c0;trans=(G-G0)@v-h*c0
    velocity_defect.append(vel);transport_defect.append(trans);delta.append(pworld-p0)
    mujoco.mj_differentiatePos(m,disp,h,q0,q)
    q_update.append(float(np.max(abs(disp-v))))
   last=(G.copy(),v.copy(),cworld.copy(),q.copy(),a['time'][k],pworld.copy())
  world,gen,exchange,velocity_defect,transport_defect,delta=map(np.asarray,(world,gen,exchange,velocity_defect,transport_defect,delta))
  if calls>auth['static_kinematics_mass_bias_queries_cap']:raise RuntimeError('STATIC_QUERY_CAP')
  fit=velocity_defect+transport_defect-delta
  item={'source':rel,'sha256':before,'states':len(saved),'start_s':float(a['time'][0]),'end_s':float(a['time'][-1]),'initial_generalized':saved[0].tolist(),'final_generalized':saved[-1].tolist(),'initial_norm':norm2(saved[0]).tolist(),'final_norm':norm2(saved[-1]).tolist(),'peak_norm':norm2(saved).max(0).tolist(),'world_initial':world[0].tolist(),'world_final':world[-1].tolist(),'world_change':(world[-1]-world[0]).tolist(),'world_change_norm':norm2(world[-1]-world[0]).tolist(),'exchange_peak':norm2(exchange).max(0).tolist(),'original_dynamic_style_threshold':(1e-5+.002*norm2(exchange).max(0)).tolist(),'saved_momentum_reconstruction_max_abs':max(reconstruct),'velocity_defect_max_norm':norm2(velocity_defect).max(0).tolist(),'velocity_defect_vector_sum':velocity_defect.sum(0).tolist(),'transport_defect_vector_sum':transport_defect.sum(0).tolist(),'decomposition_identity_max_abs':float(abs(fit).max()),'sum_identity_max_abs':float(abs(velocity_defect.sum(0)+transport_defect.sum(0)-(world[-1]-world[0])).max()),'configuration_update_vs_next_velocity_max_abs':max(q_update),'first_nonzero_index_above_1e_12':int(np.where(np.max(abs(saved),axis=1)>1e-12)[0][0]),'largest_linear_increment_time_s':float(a['time'][1+np.argmax(np.linalg.norm(delta[:,:3],axis=1))]),'source_unchanged':before==sha(path)}
  if saved_exchange is not None:item['saved_exchange_reconstruction_max_abs']=float(abs(exchange-saved_exchange).max())
  np.savez_compressed(PKG/'evidence'/f'discrete_defects_{idx}.npz',time=a['time'],generalized=gen,exchange=exchange,world_momentum=world,velocity_solve_defect=velocity_defect,configuration_transport_defect=transport_defect,momentum_increment=delta)
  results.append(item)
 total=sum(p.stat().st_size for p in (PKG/'evidence').glob('*') if p.is_file())
 result={'status':'READ_ONLY_DIAGNOSIS_COMPLETE_NOT_REQUALIFICATION','mujoco_version':mujoco.__version__,'integrator':int(m.opt.integrator),'timestep_s':float(m.opt.timestep),'model_hash':ident['scene_model_hash'],'base_dofs':b.tolist(),'base_damping':m.dof_damping[b].tolist(),'base_armature':m.dof_armature[b].tolist(),'robot_actuator_generalized_base_columns':'direct joint actuators on 67 non-base hinge DOFs','gravity':m.opt.gravity.tolist(),'density':float(m.opt.density),'viscosity':float(m.opt.viscosity),'ntendon':m.ntendon,'neq':m.neq,'handoffs':handoffs,'phases':results,'static_mass_bias_states':calls,'new_native_steps':0,'new_replay_steps':0,'new_coupled_calls':0,'new_geometry_queries':0,'source_arrays_only_no_new_trajectory':True,'wall_s':time.time()-start,'output_bytes':total,'limits':'Base force check is reconstructed algebra on every saved interval; no counterfactual integrator execution. Physical cause decomposition is separate from old pass/fail review.'}
 (PKG/'evidence/diagnosis.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
 print(json.dumps(result,ensure_ascii=True),flush=True)
if __name__=='__main__':main()
