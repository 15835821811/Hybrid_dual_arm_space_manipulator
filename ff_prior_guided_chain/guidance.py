"""Whole-trajectory feedback inside DDIM; no post-sampling recovery loop."""
import numpy as np
import scipy.sparse as sp
import mujoco
from scipy.spatial.transform import Rotation
from ff_prior.geometry import CollisionBackend
from ff_prior_pilot_train.storage import atomic_json,atomic_npz
from ff_prior.coupled_predictor import pose_errors
from ff_prior_horizon_restore.solver import select_active
from .protocol import P
from .representation import solve_qp


def pose_residual(actual,target):
    p=(actual[:,:,:3,3]-target[:,:,:3,3])/.005
    rot=np.swapaxes(target[:,:,:3,:3],-1,-2)@actual[:,:,:3,:3]
    w=Rotation.from_matrix(rot.reshape(-1,3,3)).as_rotvec().reshape(-1,2,3)/P['rotation_tolerance']
    return np.concatenate([p,w],axis=2).ravel()


class Guidance:
    def __init__(self,rep,predictor,targets,out,progress):
        self.rep,self.pred,self.targets,self.out,self.progress=rep,predictor,targets,out,progress
        self.r=predictor.r;self.budget=predictor.budget;self.cc=CollisionBackend(self.r)
        self.scratch=predictor.initial.reset(self.r)
        self.world_times=np.array([27.]) if rep.phase=='A' else np.unique(np.r_[0.,[rep.knots[i+1:i+4].mean() for i in range(273)],rep.duration])

    def grid(self,n,active=None):
        fixed=np.unique(np.r_[0.,self.rep.duration,self.world_times,[] if active is None else active[:,0]])
        available=n+1-len(fixed)
        if available<2:raise ValueError('SPARSE_ACTIVE_GRID_CAP')
        # Nested dyadic grids allow exact all-body coarse/fine comparisons.
        intervals=2**int(np.floor(np.log2(available)))
        return np.unique(np.r_[fixed,np.linspace(0,self.rep.duration,intervals+1)])

    def residual(self,pred,active,torque):
        idx=np.searchsorted(pred['time'],self.world_times)
        assert np.array_equal(pred['time'][idx],self.world_times)
        world=pose_residual(pred['ee'][idx],self.targets.poses(self.world_times))
        distances=[]
        last=-1
        for t,a,b in active:
            j=int(np.searchsorted(pred['time'],t));assert pred['time'][j]==t
            if j!=last:
                self.scratch.qpos[:]=pred['qpos'][j];self.r.kinematics(self.scratch);last=j
            distances.append(mujoco.mj_geomDistance(self.r.model,self.scratch,int(a),int(b),2.5,None))
        tv=[]
        for t,j,sign in torque:
            k=int(np.searchsorted(pred['time'],t));assert pred['time'][k]==t
            tv.append(sign*pred['raw_torque'][k,int(j)]/self.r.spec.torque_limits[int(j)]-1)
        return np.r_[world,-np.asarray(distances)/.003,np.asarray(tv)],len(world)

    def dense(self,x,path):
        ref=self.rep.ref(x);history=[];previous=None;agreement=False
        for n in P['dense_grids']:
            pred=self.pred.run(ref,self.grid(n))
            ext=np.linalg.norm(np.diff(pred['geom'],axis=0),axis=-1)+np.linalg.norm(np.diff(pred['geom_rotation'],axis=0).reshape(-1,self.r.model.ngeom,3,3),axis=(-1,-2))*self.r.model.geom_rbound
            extent=float(np.max(ext));diff=None
            if previous is not None:
                i=np.searchsorted(pred['time'],previous['time']);assert np.array_equal(pred['time'][i],previous['time'])
                dp,dr=pose_errors(pred['ee'][i],previous['ee'])
                gd=np.linalg.norm(pred['geom'][i]-previous['geom'],axis=-1)
                rd=np.linalg.norm((pred['geom_rotation'][i]-previous['geom_rotation']).reshape(-1,self.r.model.ngeom,3,3),axis=(-1,-2))*self.r.model.geom_rbound
                diff=dict(position_m=float(dp.max()),rotation_rad=float(dr.max()),all_geometry_extent_m=float(np.max(gd+rd)))
                agreement=diff['position_m']<=.001 and diff['rotation_rad']<=np.deg2rad(.2) and diff['all_geometry_extent_m']<=.001
            history.append(dict(intervals=len(pred['time'])-1,extent_m=extent,agreement=diff,numerical=pred['numerical']))
            if extent<=.002 and agreement:break
            previous=pred
        near=[];minimum=float('inf');minima=[]
        pairs=self.cc.pairs;rad=self.r.model.geom_rbound
        for j,t in enumerate(pred['time']):
            self.budget.check();self.scratch.qpos[:]=pred['qpos'][j];self.r.kinematics(self.scratch)
            centers=self.scratch.geom_xpos
            lower=np.linalg.norm(centers[pairs[:,0]]-centers[pairs[:,1]],axis=1)-rad[pairs[:,0]]-rad[pairs[:,1]]
            local=float(np.min(lower))
            # Every unqueried pair has an archived geometric lower-bound proof:
            # centers, radii, complete pair IDs and the fixed threshold.
            exact=np.flatnonzero(lower<=P['active_distance'])
            values=lower.copy()
            for k in exact:
                a,b=pairs[k];dist=float(mujoco.mj_geomDistance(self.r.model,self.scratch,int(a),int(b),2.5,None))
                values[k]=dist;near.append((t,a,b,dist,k))
            local=float(values.min());minima.append(local);minimum=min(minimum,local)
            if j%64==0:self.budget.ensure_output(sum(v.nbytes for v in pred.values() if isinstance(v,np.ndarray))+len(near)*40+8*1024**2)
        near=np.asarray(near).reshape(-1,5)
        normalized=near[:,:4].copy();normalized[:,0]/=self.rep.duration
        chosen,selection_reasons=select_active(normalized,P['geometry_rows'])
        active=near[chosen,:3]
        # Strongest torque rows retain native channel, time and frozen nominal sign.
        ratios=abs(pred['raw_torque'])/self.r.spec.torque_limits
        candidates=np.argwhere(ratios>=.75)
        order=sorted(candidates.tolist(),key=lambda ij:-ratios[tuple(ij)])[:P['torque_rows']]
        torque=np.asarray([(pred['time'][i],j,np.sign(pred['raw_torque'][i,j])) for i,j in order]).reshape(-1,3)
        residual,nworld=self.residual(pred,active,torque)
        checked_times=self.world_times if self.rep.phase=='A' else pred['time']
        idx=np.searchsorted(pred['time'],checked_times)
        dp,dr=pose_errors(pred['ee'][idx],self.targets.poses(checked_times))
        geometry_delta=float(abs(min(minima[::2])-min(minima)))
        metric=dict(position_max_m=float(dp.max()),rotation_max_rad=float(dr.max()),
            clearance_lower_or_exact_m=minimum,geometry_step_m=extent,geometry_refinement_difference_m=geometry_delta,converged=agreement,
            maximum_torque_ratio=float(ratios.max()),world_rows=nworld,
            merit=float(np.mean(residual[:nworld]**2)+np.sum(np.maximum(residual[nworld:],0)**2)),
            original_model_gates_pass=bool(dp.max()<=.005 and dr.max()<=P['rotation_tolerance'] and minimum>=0 and extent<=.002 and geometry_delta<=.001 and agreement and ratios.max()<=1 and self.rep.valid(x)))
        self.budget.ensure_output(sum(v.nbytes for v in pred.values() if isinstance(v,np.ndarray))+near.nbytes+8*1024**2)
        atomic_npz(path.with_suffix('.npz'),free=x,**{k:v for k,v in pred.items() if isinstance(v,np.ndarray)},
            pair_ids=pairs,geom_rbound=rad,near=near,selected=chosen,omitted=np.setdiff1d(np.arange(len(near)),chosen),
            torque_selected=torque,minimum_path=minima,residual=residual)
        atomic_json(path.with_suffix('.json'),dict(metrics=metric,grid_history=history,
            active_selection='worst geometry pair, phase-bin coverage, remaining exact distance; strongest torque ratios',selected_reasons=selection_reasons,
            omitted_pair_proof='||saved geom centers[a]-centers[b]||-saved rbound[a]-rbound[b] > active_distance',
            active_distance=P['active_distance'],all_pair_policy_hash=self.cc.policy_hash,
            continuous_time_collision_proof=False,qualification='MODEL_ONLY'))
        return dict(pred=pred,active=active,torque=torque,residual=residual,nworld=nworld,metrics=metric,path=str(path))

    def sparse(self,x,times,active,torque):
        pred=self.pred.run(self.rep.ref(x),times)
        r,n=self.residual(pred,active,torque)
        return r,pred

    def update(self,x,index):
        folder=self.out/f'guidance_{index:02d}';folder.mkdir(parents=True,exist_ok=False)
        center=self.dense(x,folder/'center');active=center['active'];torque=center['torque']
        union=np.vstack([active,torque]) if len(torque) else active
        agreement=[]
        for n in P['sparse_grids']:
            try:times=self.grid(n,union)
            except ValueError:
                agreement.append(dict(cap=n,status='FIXED_ROWS_EXCEED_GRID'));continue
            r0,p0=self.sparse(x,times,active,torque)
            error=float(np.max(abs(r0-center['residual']),initial=0))
            agreement.append(dict(intervals=len(times)-1,normalized_residual_error=error))
            if error<=.1:break
        else:raise ValueError('SPARSE_DENSE_RESIDUAL_DISAGREEMENT')
        dim=self.rep.dim;J=np.empty((len(r0),dim));raw=np.empty((dim,len(r0)));h=P['fd_h']
        self.budget.ensure_output(J.nbytes*2+16*1024**2)
        for j in range(dim):
            if j%32==0:self.progress('guidance_finite_difference',phase=self.rep.phase,step=index,column=j,columns=dim)
            xx=x.copy();xx.ravel()[j]+=h
            rv,_=self.sparse(xx,times,active,torque);raw[j]=rv;J[:,j]=(rv-r0)/h
        probes=[];probe_pass=True
        for k in range(4):
            d=np.sin(np.arange(dim)*(k+1)+.37);d/=np.linalg.norm(d)
            vals=[]
            for scale in (1.,-1.,.5,-.5):
                rv,_=self.sparse(x+(scale*h*d).reshape(x.shape),times,active,torque);vals.append(rv)
            central=(vals[0]-vals[1])/(2*h);half=(vals[2]-vals[3])/h;nominal=J@d
            tol=P['probe_relative_tolerance']*(1+abs(half))
            good=bool(np.all(abs(central-half)<=tol) and np.all(abs(nominal-half)<=tol))
            probes.append(dict(direction=d,values=vals,central=central,half=half,linear=nominal,passed=good));probe_pass &= good
        atomic_npz(folder/'jacobian.npz',free=x,times=times,active=active,torque=torque,
            residual=r0,perturbed_residual=raw,jacobian=J,steps=np.full(dim,h))
        atomic_json(folder/'directional_checks.json',dict(probes=probes,passed=probe_pass,
            tolerance_formula='.02*(1+abs(half_central_derivative))',same_fixed_row_identities=True,
            nonsmooth_boundary='fixed geometry IDs; closest-feature switches may remain nonsmooth, and unexplained disagreement stops this unit'))
        if not probe_pass:raise ValueError('GUIDANCE_DERIVATIVE_UNRESOLVED')
        # Lift least-squares world residual and one-sided physical violations.
        # Sparse diagonal Hessian avoids explicitly forming a dense J.T@J.
        nworld=center['nworld'];rows=len(r0);D=sp.csc_matrix(J);I=sp.eye(rows,format='csc')
        A1=sp.hstack([self.rep.A,sp.csc_matrix((self.rep.A.shape[0],rows))],format='csc')
        A2=sp.hstack([D,-I],format='csc')
        A3=sp.hstack([sp.eye(dim),sp.csc_matrix((dim,rows))],format='csc')
        A4=sp.hstack([sp.csc_matrix((rows-nworld,dim+nworld)),sp.eye(rows-nworld)],format='csc')
        A=sp.vstack([A1,A2,A3,A4],format='csc')
        y=self.rep.A@x.ravel()
        low=np.r_[self.rep.lo-y,-r0[:nworld],np.full(rows-nworld,-np.inf),np.full(dim,-P['trust_radius']),np.zeros(rows-nworld)]
        high=np.r_[self.rep.hi-y,-r0,np.full(dim,P['trust_radius']),np.full(rows-nworld,np.inf)]
        weights=np.r_[np.full(dim,P['guidance_regularization']),np.full(nworld,1/max(1,nworld)),np.ones(rows-nworld)]
        sol,report=solve_qp(sp.diags(weights),np.zeros(dim+rows),A,low,high,self.budget.deadline)
        step=sol[:dim].reshape(x.shape);trials=[];result=x
        for k,scale in enumerate(P['line_scales']):
            xx=x+scale*step
            if not self.rep.valid(xx):raise ValueError('GUIDANCE_MOTION_QP_INCONSISTENT')
            trial=self.dense(xx,folder/f'trial_{k}')
            accepted=trial['metrics']['merit']<center['metrics']['merit']-1e-10
            trials.append(dict(scale=scale,accepted=accepted,metrics=trial['metrics']))
            if accepted:result=xx;break
        atomic_json(folder/'result.json',dict(solver=report,trials=trials,agreement=agreement,
            accepted=bool(np.any(result!=x)),post_denoising_repair=False))
        self.budget.save()
        return result
