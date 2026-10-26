"""Original public world reference, predicted from this phase's real initial state."""
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation
from tracking_diffusion.environment import free_slices
from v6_lite.irregular_waypoints import IrregularWaypointTarget
from ff_prior_balanced_brake.reference import bypass
from ff_prior_satellite_origin.force import analytic_motion


class Targets:
    def __init__(self,robot,initial,phase,contract,static_goals):
        self.robot,self.initial,self.phase,self.contract=robot,initial,phase,contract
        self.static_goals=np.asarray(static_goals);self.d=initial.reset(robot)
        robot.forward(self.d);self.q0=robot.Bplus@self.d.qpos[robot.qpos_ids]
        self.v0=robot.Bplus@self.d.qvel[robot.dof_ids]
        self.tp,self.tv=free_slices(robot.model,robot.spec.target_free_joint_name)
        target=robot.id(mujoco.mjtObj.mjOBJ_BODY,'target_satellite')
        self.mass=float(robot.model.body_mass[target]);self.pt=self.d.xpos[target].copy()
        self.rt=self.d.xmat[target].reshape(3,3).copy();self.vt=self.d.qvel[self.tv].copy()
        if phase!='A':
            w=contract['waypoints'];self.w=w
            self.path=IrregularWaypointTarget(np.asarray(w['initial_position_w']),np.asarray(w['waypoint_points_m']),
                np.asarray(w['target_rotation_world']),w['transition_duration_s'],w['path_duration_s'],np.asarray(w['segment_durations_s']),w)
            self.bridge=Rotation.from_matrix(self.path.target_rotation_world@self.static_goals[0,:3,:3].T).as_rotvec()
            self.grasp=np.asarray(contract['grasp_local']);self.grasp_r=np.asarray(contract['grasp_rotation_local'])

    def reference(self,t):
        if self.phase=='A':
            a,b=self.static_goals;z=np.zeros(3)
            return a[:3,3],a[:3,:3],z,z,b[:3,3],b[:3,:3],z,z
        if self.phase=='B':
            pc,vc=self.path.sample(t);pc,vc=bypass(pc,vc,t,self.w)
            s=np.clip(t/4.5,0.,1.);alpha=s**3*(10-15*s+6*s*s)
            rate=30*s*s*(1-s)**2/4.5 if 0<=t<4.5 else 0.
            rc=Rotation.from_rotvec(alpha*self.bridge).as_matrix()@self.static_goals[0,:3,:3];wc=rate*self.bridge
        else:
            pc=np.asarray(self.w['waypoint_points_m'])[-1];rc=np.asarray(self.w['target_rotation_world']);vc=wc=np.zeros(3)
        rt=self.rt@Rotation.from_rotvec(self.vt[3:]*t).as_matrix()
        pt=self.pt+self.vt[:3]*t;vt=self.vt[:3].copy()
        if self.phase=='C':
            z,v=analytic_motion(t,self.mass);pt[2]+=z;vt[2]+=v
        omega=rt@self.vt[3:];pr=pt+rt@self.grasp
        return pc,rc,vc,wc,pr,rt@self.grasp_r,vt+np.cross(omega,rt@self.grasp),omega

    def poses(self,times):
        out=[]
        for t in times:
            v=self.reference(float(t));a=np.tile(np.eye(4),(2,1,1))
            a[0,:3,3]=v[0];a[0,:3,:3]=v[1];a[1,:3,3]=v[4];a[1,:3,:3]=v[5];out.append(a)
        return np.asarray(out)

    def features(self,rep):
        # Exact frozen1014 schema. Only C1..C5 enter neural obstacle slots;
        # R1/R2 remain present in every physical/geometry evaluation.
        w=self.w;q0,v0=self.q0,self.v0
        wp=np.asarray(w['waypoint_points_m']);dur=np.asarray(w['segment_durations_s'])
        if self.phase=='C':wp=np.repeat(wp[-1:],7,axis=0)
        obs=np.zeros((5,12));objects=[o for o in self.contract['scene']['objects'] if o['object_id'].startswith('C')]
        assert len(objects)==5
        for i,o in enumerate(objects):obs[i]=np.r_[1.,1.,o['position'],o['size'],o['quaternion_wxyz']]
        g=np.r_[self.initial.values,q0,v0,wp.ravel(),dur,np.asarray(w['target_rotation_world']).ravel(),
            self.grasp,self.grasp_r.ravel(),np.zeros(3),self.static_goals[1,:3,3],self.static_goals[1,:3,:3].ravel(),
            self.pt,self.rt.ravel(),self.vt,[4.5,25.5,27.] if self.phase=='B' else [0.,29.,29.],obs.ravel()]
        times=np.asarray([rep.knots[i+1:i+4].mean() for i in range(273)])[2:]
        local=np.asarray([np.concatenate([np.asarray(x).ravel() for x in self.reference(t)]) for t in times])
        assert g.shape==(1014,) and local.shape==(271,36)
        return dict(global_condition=g,local_condition=local,q0=q0,v0=v0,coefficient_times=times,
            obstacle_slot_ids=[o['object_id'] for o in objects],all_seven_physical_obstacles=True,
            target_prediction='phase_initial_full_state_plus_public_force_analytic_motion')
