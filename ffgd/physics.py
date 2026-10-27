"""Float64 differentiable rigid-body tree from the compiled native model.

All moving frames, COM inertias, mass blocks and their derivatives remain in
the graph. No native finite differences in the production derivatives.
"""
import numpy as np
import torch
import mujoco


def tensor(x): return torch.as_tensor(np.array(x, copy=True), dtype=torch.float64)


def skew(v):
    x,y,z = v.unbind(-1); o = x*0
    return torch.stack((o,-z,y,z,o,-x,-y,x,o), -1).reshape(*v.shape[:-1],3,3)


def quat_matrix(q):
    q = q / torch.linalg.vector_norm(q, dim=-1, keepdim=True)
    w,x,y,z = q.unbind(-1)
    return torch.stack((1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w),
                        2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w),
                        2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)), -1).reshape(*q.shape[:-1],3,3)


def rotation(axis, angle):
    K = skew(axis)
    return torch.eye(3, dtype=angle.dtype, device=angle.device) + torch.sin(angle)*K + (1-torch.cos(angle))*(K@K)


def quat_mul(a,b):
    aw,av = a[:1],a[1:]; bw,bv=b[:1],b[1:]
    return torch.cat((aw*bw-(av*bv).sum().reshape(1), aw*bv+bw*av+torch.linalg.cross(av,bv)))


def integrate_base(pose, velocity, dt):
    theta = velocity[3:]*dt
    norm = torch.linalg.vector_norm(theta)
    dq = torch.cat((torch.cos(norm/2).reshape(1), .5*torch.sinc(norm/(2*torch.pi))*theta))
    q = quat_mul(pose[3:],dq)
    return torch.cat((pose[:3]+dt*velocity[:3],q/torch.linalg.vector_norm(q)))


class DifferentiableTree:
    def __init__(self, robot):
        m = robot.model
        self.robot, self.nv, self.nb = robot, m.nv, m.nbody
        self.parent = m.body_parentid.tolist()
        self.pos = tensor(m.body_pos)
        self.rot = quat_matrix(tensor(m.body_quat))
        self.ipos = tensor(m.body_ipos)
        self.irot = quat_matrix(tensor(m.body_iquat))
        # Match compiled same-frame shortcuts exactly (native local2Global).
        self.ipos[np.asarray(m.body_sameframe)==1]=0
        self.irot[np.isin(m.body_sameframe,[1,3])]=torch.eye(3,dtype=torch.float64)
        self.inertia = tensor(m.body_inertia)
        self.mass_values = tensor(m.body_mass)
        self.armature, self.damping = tensor(m.dof_armature), tensor(m.dof_damping)
        self.qpos0 = tensor(m.qpos0)
        self.joints=[]
        for b in range(m.nbody):
            if m.body_jntnum[b] > 1: raise NotImplementedError("multi-joint body requires explicit support")
            self.joints.append(int(m.body_jntadr[b]) if m.body_jntnum[b] else -1)
        self.jtype=m.jnt_type.tolist(); self.qadr=m.jnt_qposadr.tolist(); self.vadr=m.jnt_dofadr.tolist()
        self.jpos=tensor(m.jnt_pos); self.axis=tensor(m.jnt_axis)
        if np.any(m.jnt_stiffness) or np.any(m.dof_frictionloss):
            raise NotImplementedError("spring/friction model must be added before use")
        self.eye=torch.eye(self.nv,dtype=torch.float64)
        self.B=tensor(robot.B)
        self.robot_bodies=[b for b in range(1,m.nbody) if self._descendant(b,robot.base_body)]

    def _descendant(self,b,root):
        while b and b != root: b=self.parent[b]
        return b == root

    def kinematics(self, qpos, velocity=None, acceleration=None):
        v = torch.zeros(self.nv,dtype=qpos.dtype) if velocity is None else velocity
        a = torch.zeros_like(v) if acceleration is None else acceleration
        z=torch.zeros(3,dtype=qpos.dtype); J=torch.zeros(3,self.nv,dtype=qpos.dtype)
        ps=[z]; Rs=[torch.eye(3,dtype=qpos.dtype)]; Jps=[J]; Jws=[J]
        ws=[z]; als=[z]; accs=[z]
        for b in range(1,self.nb):
            p=self.parent[b]; j=self.joints[b]
            Rp=Rs[p]; pp=ps[p]; wp=ws[p]; alp=als[p]
            R0=Rp@self.rot[b]; p0=pp+Rp@self.pos[b]
            if j >= 0 and self.jtype[j] == int(mujoco.mjtJoint.mjJNT_FREE):
                if p != 0: raise NotImplementedError("nested free joint")
                qa,va=self.qadr[j],self.vadr[j]
                pb=qpos[qa:qa+3]; Rb=quat_matrix(qpos[qa+3:qa+7])
                Jp=self.eye[va:va+3]; Jw=Rb@self.eye[va+3:va+6]
                w=Rb@v[va+3:va+6]; al=Rb@a[va+3:va+6]; ac=a[va:va+3]
            elif j >= 0 and self.jtype[j] == int(mujoco.mjtJoint.mjJNT_HINGE):
                qa,va=self.qadr[j],self.vadr[j]
                Rb=R0@rotation(self.axis[j],qpos[qa]-self.qpos0[qa])
                anchor=p0+R0@self.jpos[j]; pb=anchor-Rb@self.jpos[j]
                axis=R0@self.axis[j]; r=pb-anchor; d=anchor-pp
                Jw=Jws[p]+axis[:,None]*self.eye[va]
                Ja=Jps[p]-skew(d)@Jws[p]
                Jp=Ja-skew(r)@Jw
                w=wp+axis*v[va]
                al=alp+axis*a[va]+torch.linalg.cross(wp,axis*v[va])
                aa=accs[p]+torch.linalg.cross(alp,d)+torch.linalg.cross(wp,torch.linalg.cross(wp,d))
                ac=aa+torch.linalg.cross(al,r)+torch.linalg.cross(w,torch.linalg.cross(w,r))
            elif j == -1:
                pb,Rb=p0,R0; d=pb-pp
                Jw=Jws[p]; Jp=Jps[p]-skew(d)@Jw
                w,al=wp,alp
                ac=accs[p]+torch.linalg.cross(al,d)+torch.linalg.cross(w,torch.linalg.cross(w,d))
            else: raise NotImplementedError("unsupported joint type")
            ps.append(pb); Rs.append(Rb); Jps.append(Jp); Jws.append(Jw); ws.append(w); als.append(al); accs.append(ac)
        P,R,Jp,Jw,w,al,ac = map(torch.stack,(ps,Rs,Jps,Jws,ws,als,accs))
        r=(R@self.ipos[:,:,None]).squeeze(-1)
        com=P+r; Jc=Jp-skew(r)@Jw
        acom=ac+torch.linalg.cross(al,r)+torch.linalg.cross(w,torch.linalg.cross(w,r))
        RI=R@self.irot
        inertia=(RI*self.inertia[:,None,:])@RI.transpose(-1,-2)
        return dict(position=P, rotation=R, com=com, Jcom=Jc, Jangular=Jw, omega=w, alpha=al, acom=acom, inertia=inertia)

    def mass(self, qpos, kin=None):
        k=self.kinematics(qpos) if kin is None else kin
        Jc,Jw=k['Jcom'],k['Jangular']
        return (Jc.transpose(-1,-2)@(self.mass_values[:,None,None]*Jc)).sum(0) + (Jw.transpose(-1,-2)@k['inertia']@Jw).sum(0) + torch.diag(self.armature)

    def inverse(self,qpos,velocity,acceleration,kin=None):
        k=self.kinematics(qpos,velocity,acceleration) if kin is None else kin
        F=self.mass_values[:,None]*k['acom']
        Iw=(k['inertia']@k['omega'][:,:,None]).squeeze(-1)
        T=(k['inertia']@k['alpha'][:,:,None]).squeeze(-1)+torch.linalg.cross(k['omega'],Iw)
        return (k['Jcom'].transpose(-1,-2)@F[:,:,None]).sum(0).squeeze(-1)+(k['Jangular'].transpose(-1,-2)@T[:,:,None]).sum(0).squeeze(-1)+self.armature*acceleration+self.damping*velocity

    def momentum(self,qpos,velocity):
        k=self.kinematics(qpos,velocity); ids=self.robot_bodies
        linear=self.mass_values[ids,None]*(k['Jcom'][ids]@velocity)
        angular=(k['inertia'][ids]@k['omega'][ids,:,None]).squeeze(-1)+torch.linalg.cross(k['com'][ids],linear)
        return torch.cat((linear.sum(0),angular.sum(0)))

    def nominal(self,qpos,velocity,qdd):
        M=self.mass(qpos); h=self.inverse(qpos,velocity,torch.zeros_like(velocity))
        b,a=self.robot.base_dofs,self.robot.dof_ids
        aa=self.B@qdd
        ab=-torch.linalg.solve(M[b[:,None],b],M[b[:,None],a]@aa+h[b])
        tau=M[a[:,None],b]@ab+M[a[:,None],a]@aa+h[a]
        return tau,ab
