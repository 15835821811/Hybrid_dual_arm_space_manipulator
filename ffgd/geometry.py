"""Whole-body geometry inventory, diagnostic boxes and native mesh distances.

The box diagnostic uses enclosing boxes in native mesh frames. SAT projection gaps are
conservative lower bounds, not exact Euclidean surface distances. All eligible
pairs are rebuilt/evaluated every call; no permanently frozen near-pair subset.
"""
import hashlib
import json
import mujoco
import numpy as np
import torch
from .physics import tensor,quat_matrix


class Geometry:
    def __init__(self,robot):
        self.robot=r=robot;m=r.model
        self.ids=np.flatnonzero((m.geom_contype != 0)|(m.geom_conaffinity != 0))
        self.bodies=m.geom_bodyid[self.ids].copy()
        self.rot=quat_matrix(tensor(m.geom_quat[self.ids]))
        self.geom_pos=tensor(m.geom_pos[self.ids])
        for i,g in enumerate(self.ids):
            sf=int(m.geom_sameframe[g]);b=int(m.geom_bodyid[g])
            if sf==1:self.geom_pos[i]=0
            if sf in (1,3):self.rot[i]=torch.eye(3,dtype=torch.float64)
            if sf==2:self.geom_pos[i]=tensor(m.body_ipos[b])
            if sf in (2,4):self.rot[i]=quat_matrix(tensor(m.body_iquat[b]))
        center=[];half=[];coverage=[]
        for g in self.ids:
            typ=int(m.geom_type[g])
            if typ==int(mujoco.mjtGeom.mjGEOM_MESH):
                mid=int(m.geom_dataid[g]);start=int(m.mesh_vertadr[mid]);n=int(m.mesh_vertnum[mid])
                vertices=m.mesh_vert[start:start+n].astype(float)
                lo=vertices.min(0);hi=vertices.max(0);c=(lo+hi)/2;h=(hi-lo)/2+1e-8
                coverage.append(dict(geom=int(g),vertices=n,max_outside=float(np.max(abs(vertices-c)-h))))
            elif typ==int(mujoco.mjtGeom.mjGEOM_BOX):
                c=np.zeros(3);h=m.geom_size[g].copy()
                coverage.append(dict(geom=int(g),vertices=8,max_outside=0.))
            elif typ==int(mujoco.mjtGeom.mjGEOM_SPHERE):
                c=np.zeros(3);h=np.full(3,m.geom_size[g,0])
                coverage.append(dict(geom=int(g),analytic_shape='sphere',radius=float(m.geom_size[g,0]),max_outside=0.))
            else:raise NotImplementedError(f'geom type {typ} has no calibrated guide')
            center.append(c);half.append(h)
        self.offset=self.geom_pos+(self.rot@tensor(center)[:,:,None]).squeeze(-1)
        self.half=tensor(half);self.coverage=coverage
        self.pairs=[];self.excluded=[]
        connected={tuple(sorted((m.body(a).id,m.body(b).id))) for a,b in r.connected}
        for i,ga in enumerate(self.ids):
            ba=int(m.geom_bodyid[ga]);wa=int(m.body_weldid[ba])
            for j in range(i+1,len(self.ids)):
                gb=self.ids[j];bb=int(m.geom_bodyid[gb]);wb=int(m.body_weldid[bb])
                reason=None
                if wa==wb:reason='same_weld'
                elif wa != 0 and wb != 0 and (m.body_weldid[m.body_parentid[wa]]==wb or m.body_weldid[m.body_parentid[wb]]==wa):reason='direct_articulated_neighbor'
                elif tuple(sorted((ba,bb))) in connected:reason='named_mechanical_connection'
                if reason:self.excluded.append(dict(geoms=[int(ga),int(gb)],reason=reason))
                else:self.pairs.append((i,j))
        self.pairs=np.array(self.pairs,dtype=int)
        self.policy_hash=hashlib.sha256(json.dumps(dict(ids=self.ids.tolist(),pairs=self.pairs.tolist(),excluded=self.excluded),sort_keys=True).encode()).hexdigest()

    def boxes(self,kin):
        R=kin['rotation'][self.bodies]
        P=kin['position'][self.bodies]+(R@self.offset[:,:,None]).squeeze(-1)
        return P,R@self.rot

    def gaps(self,kin):
        P,R=self.boxes(kin);a,b=self.pairs.T
        Ra,Rb=R[a],R[b];delta=P[b]-P[a]
        # Rows are candidate world separating axes. Cross axes are normalized.
        axes_a=Ra.transpose(-1,-2);axes_b=Rb.transpose(-1,-2)
        cross=torch.linalg.cross(axes_a[:,:,None,:],axes_b[:,None,:,:]).reshape(-1,9,3)
        axes=torch.cat((axes_a,axes_b,cross),dim=1)
        norm=torch.linalg.vector_norm(axes,dim=-1,keepdim=True)
        valid=norm.squeeze(-1)>1e-10;axes=axes/norm.clamp_min(1e-10)
        distance=torch.abs((axes*delta[:,None,:]).sum(-1))
        radius_a=(torch.abs(axes@Ra)*self.half[a,None,:]).sum(-1)
        radius_b=(torch.abs(axes@Rb)*self.half[b,None,:]).sum(-1)
        gaps=distance-radius_a-radius_b
        return torch.where(valid,gaps,torch.full_like(gaps,-torch.inf)).max(dim=1).values

    def native_distances(self,qpos,distmax=10.):
        m=self.robot.model;d=mujoco.MjData(m);d.qpos[:]=qpos
        mujoco.mj_kinematics(m,d)
        values=[];witness=np.zeros(6)
        for a,b in self.pairs:
            values.append(mujoco.mj_geomDistance(m,d,int(self.ids[a]),int(self.ids[b]),distmax,witness))
        return np.asarray(values)

    def native_scan(self,data):
        """Exact minimum with conservative sphere pruning; no contact-list shortcut."""
        m=self.robot.model;mujoco.mj_kinematics(m,data)
        a,b=self.ids[self.pairs].T
        lower=np.linalg.norm(data.geom_xpos[b]-data.geom_xpos[a],axis=1)-m.geom_rbound[a]-m.geom_rbound[b]
        best=10.;pair=None;queries=0
        for index in np.argsort(lower):
            if lower[index]>best:break
            value=mujoco.mj_geomDistance(m,data,int(a[index]),int(b[index]),10.,None);queries+=1
            if value<best:best=float(value);pair=[int(a[index]),int(b[index])]
        return dict(minimum=best,pair=pair,queries=queries,capped=best>=10.)

    def catalog(self):
        return dict(policy_hash=self.policy_hash,collision_geoms=self.ids.tolist(),pairs=self.ids[self.pairs].tolist(),
                    excluded=self.excluded,coverage=self.coverage,guide='enclosing_mesh_AABB_in_geom_frame_SAT_lower_bound',
                    exact_distance=False,continuous_time_certified=False)


class NativeWitnessDistance(torch.autograd.Function):
    """Envelope-theorem VJP at native convex distance witnesses.

    Positions/rotations are connected to the whole-path graph. Native qpos is
    only the oracle input, not a claimed differentiable NumPy conversion.
    Feature switches and zero-distance witnesses require separate diagnostics.
    """
    @staticmethod
    def forward(ctx,P,R,qpos,geometry,cap):
        m=geometry.robot.model;d=mujoco.MjData(m);d.qpos[:]=qpos.detach().numpy()
        mujoco.mj_kinematics(m,d)
        pn=P.detach().numpy();rn=R.detach().numpy()
        np.testing.assert_allclose(pn,d.geom_xpos[geometry.ids],atol=1e-10,rtol=0)
        np.testing.assert_allclose(rn,d.geom_xmat[geometry.ids].reshape(-1,3,3),atol=1e-10,rtol=0)
        a,b=geometry.pairs.T
        lower=np.linalg.norm(pn[b]-pn[a],axis=1)-m.geom_rbound[geometry.ids[a]]-m.geom_rbound[geometry.ids[b]]
        distances=np.full(len(a),cap);normals=np.zeros((len(a),3));la=np.zeros_like(normals);lb=np.zeros_like(normals)
        queried=0
        for j in np.flatnonzero(lower<cap):
            witness=np.zeros(6)
            ga,gb=int(geometry.ids[a[j]]),int(geometry.ids[b[j]])
            # MuJoCo 3.3.2 nativeCCD returns witnesses in type-sorted order:
            # engine_support.c:1318 returns before the legacy unflip branch.
            # Order explicitly and restore original pair order for the VJP.
            flip=m.geom_type[ga]>m.geom_type[gb]
            distance=mujoco.mj_geomDistance(m,d,gb if flip else ga,ga if flip else gb,cap,witness)
            if flip:witness=witness.reshape(2,3)[::-1].copy().ravel()
            distances[j]=min(distance,cap);queried+=1
            if distance>=cap:continue
            if abs(distance)<1e-10:raise ValueError('DEGENERATE_ZERO_DISTANCE_WITNESS')
            n=(witness[3:]-witness[:3])/distance
            if abs(np.linalg.norm(n)-1)>1e-4:raise ValueError('INVALID_NATIVE_DISTANCE_WITNESS')
            normals[j]=n
            la[j]=rn[a[j]].T@(witness[:3]-pn[a[j]])
            lb[j]=rn[b[j]].T@(witness[3:]-pn[b[j]])
        geometry.last_query=dict(exact_queries=queried,pairs=len(a),capped=int(np.sum(distances>=cap)),cap=cap)
        ctx.save_for_backward(torch.tensor(a),torch.tensor(b),tensor(normals),tensor(la),tensor(lb))
        ctx.count=len(P)
        return tensor(distances)

    @staticmethod
    def backward(ctx,gradient):
        a,b,n,la,lb=ctx.saved_tensors
        gp=torch.zeros(ctx.count,3,dtype=gradient.dtype)
        gr=torch.zeros(ctx.count,3,3,dtype=gradient.dtype)
        weighted=n*gradient[:,None]
        gp.index_add_(0,a,-weighted);gp.index_add_(0,b,weighted)
        gr.index_add_(0,a,-weighted[:,:,None]*la[:,None,:])
        gr.index_add_(0,b,weighted[:,:,None]*lb[:,None,:])
        return gp,gr,None,None,None


def witness_gaps(geometry,kin,qpos,cap=.02):
    m=geometry.robot.model;ids=geometry.ids;b=geometry.bodies
    R=kin['rotation'][b]
    P=kin['position'][b]+(R@geometry.geom_pos[:,:,None]).squeeze(-1)
    R=R@geometry.rot
    return NativeWitnessDistance.apply(P,R,qpos,geometry,cap)
