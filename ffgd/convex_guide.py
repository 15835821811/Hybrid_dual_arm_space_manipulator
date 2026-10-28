"""Differentiable convex support lower bound; native distance remains validator.

Projects complete native convex hull vertices on a finite set of moving axes.
Each projection gap is a certified separating gap for those convex bodies;
the maximum is a conservative distance proxy, not exact Euclidean distance.
Unlike native CCD witnesses this scalar is explicitly represented in the graph.
"""
import itertools
import numpy as np
import mujoco
import torch
from .physics import tensor


class ConvexSupportGuide:
    def __init__(self,geometry):
        self.geometry=geometry;m=geometry.robot.model;hulls=[];radii=[]
        for g in geometry.ids:
            radius=float(m.geom_size[g,0]) if m.geom_type[g]==mujoco.mjtGeom.mjGEOM_SPHERE else 0.
            radii.append(radius)
            if m.geom_type[g]==mujoco.mjtGeom.mjGEOM_BOX:
                vertices=np.asarray(list(itertools.product((-1.,1.),repeat=3)))*m.geom_size[g]
            elif radius:
                # Exact support +/-radius on every unit axis; no polygonal sphere.
                vertices=np.zeros((1,3))
            else:
                mid=int(m.geom_dataid[g]);start=int(m.mesh_vertadr[mid]);n=int(m.mesh_vertnum[mid]);vertices=m.mesh_vert[start:start+n].astype(float)
                # Use native convex graph vertex IDs, with no geometric simplification.
                ga=int(m.mesh_graphadr[mid])
                if ga>=0:
                    nv=int(m.mesh_graph[ga]);indices=m.mesh_graph[ga+2+nv:ga+2+2*nv]
                    vertices=vertices[indices]
                else:
                    from scipy.spatial import ConvexHull
                    vertices=vertices[ConvexHull(vertices).vertices]
            hulls.append(vertices)
        self.counts=[len(v) for v in hulls];maxn=max(self.counts)
        padded=np.empty((len(hulls),maxn,3))
        for i,v in enumerate(hulls):padded[i]=v[0];padded[i,:len(v)]=v
        self.vertices=tensor(padded)
        self.radii=tensor(radii)
        self.hulls=hulls
        self.sizes=dict(geoms=len(hulls),max_vertices=maxn,total_vertices=sum(self.counts))
        index=np.arange(32);z=1-2*(index+.5)/32;phi=index*np.pi*(3-np.sqrt(5))
        self.fixed_axes=tensor(np.c_[np.sqrt(1-z*z)*np.cos(phi),np.sqrt(1-z*z)*np.sin(phi),z])

    def gaps(self,kin,cap=.02):
        g=self.geometry;m=g.robot.model;bodies=g.bodies
        rb=kin['rotation'][bodies];R=rb@g.rot
        P=kin['position'][bodies]+(rb@g.geom_pos[:,:,None]).squeeze(-1)
        a,b=g.pairs.T;delta=P[b]-P[a]
        lower=torch.linalg.vector_norm(delta,dim=-1)-tensor(m.geom_rbound[g.ids[a]]+m.geom_rbound[g.ids[b]])
        # All eligible pairs participate in each broadphase. Pruning only where
        # the conservative lower bound certifies zero hinge at the stated cap.
        active=torch.nonzero(lower<cap).flatten()
        self.last_query=dict(eligible_pairs=len(a),support_pairs=len(active),broadphase_only_pairs=len(a)-len(active))
        result=torch.full((len(a),),cap,dtype=P.dtype)
        if not len(active):return result+P.sum()*0
        aa=torch.as_tensor(a)[active];bb=torch.as_tensor(b)[active]
        ra,rr=R[aa],R[bb];difference=P[bb]-P[aa]
        axes_a=ra.transpose(-1,-2);axes_b=rr.transpose(-1,-2)
        cross=torch.linalg.cross(axes_a[:,:,None,:],axes_b[:,None,:,:]).reshape(-1,9,3)
        axes=torch.cat((axes_a,axes_b,cross,difference[:,None,:],self.fixed_axes[None].expand(len(aa),-1,-1)),dim=1)
        norm=torch.linalg.vector_norm(axes,dim=-1,keepdim=True);valid=norm.squeeze(-1)>1e-10
        axes=axes/norm.clamp_min(1e-10)
        pa=(axes@ra)@self.vertices[aa].transpose(-1,-2)
        pb=(axes@rr)@self.vertices[bb].transpose(-1,-2)
        dp=(axes*difference[:,None,:]).sum(-1)
        radius=self.radii[aa,None]+self.radii[bb,None]
        forward=dp+pb.min(-1).values-pa.max(-1).values-radius
        backward=-dp+pa.min(-1).values-pb.max(-1).values-radius
        gaps=torch.maximum(forward,backward)
        gaps=torch.where(valid,gaps,torch.full_like(gaps,-torch.inf)).max(-1).values
        return result.index_copy(0,active,gaps)

    def numpy_gaps(self,P,R,cap=.02,features=False):
        """Independent NumPy support evaluation on native geom transforms."""
        g=self.geometry;m=g.robot.model;a,b=g.pairs.T
        lower=np.linalg.norm(P[b]-P[a],axis=1)-m.geom_rbound[g.ids[a]]-m.geom_rbound[g.ids[b]]
        result=np.full(len(a),cap)
        identities={}
        for i in np.flatnonzero(lower<cap):
            aa,bb=a[i],b[i];delta=P[bb]-P[aa];ra,rb=R[aa],R[bb]
            axes=np.concatenate((ra.T,rb.T,np.cross(ra.T[:,None,:],rb.T[None,:,:]).reshape(-1,3),delta[None],self.fixed_axes.numpy()))
            norms=np.linalg.norm(axes,axis=1);axis_ids=np.flatnonzero(norms>1e-10);axes=axes[norms>1e-10]/norms[norms>1e-10,None]
            ap=(axes@ra)@self.hulls[aa].T;bp=(axes@rb)@self.hulls[bb].T;dp=axes@delta
            radius=float(self.radii[aa]+self.radii[bb])
            forward=dp+bp.min(-1)-ap.max(-1)-radius;backward=-dp+ap.min(-1)-bp.max(-1)-radius
            fa,ba=int(np.argmax(forward)),int(np.argmax(backward));sign=1 if forward[fa]>=backward[ba] else -1;k=fa if sign==1 else ba
            result[i]=forward[k] if sign==1 else backward[k]
            if result[i]<.002:
                va=int(np.argmax(ap[k])) if sign==1 else int(np.argmin(ap[k]))
                vb=int(np.argmin(bp[k])) if sign==1 else int(np.argmax(bp[k]))
                identities[int(i)]=(int(axis_ids[k]),sign,va,vb)
        return (result,identities) if features else result
