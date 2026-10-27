"""Whole-body finite-volume boxes and independent native mesh distances.

The guide uses enclosing boxes in native mesh frames. SAT projection gaps are
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
            else:raise NotImplementedError(f'geom type {typ} has no calibrated guide')
            center.append(c);half.append(h)
        self.offset=tensor(m.geom_pos[self.ids])+(self.rot@tensor(center)[:,:,None]).squeeze(-1)
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

    def catalog(self):
        return dict(policy_hash=self.policy_hash,collision_geoms=self.ids.tolist(),pairs=self.ids[self.pairs].tolist(),
                    excluded=self.excluded,coverage=self.coverage,guide='enclosing_mesh_AABB_in_geom_frame_SAT_lower_bound',
                    exact_distance=False,continuous_time_certified=False)
