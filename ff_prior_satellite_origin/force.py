"""Public force input and independent analytic design; never writes state."""
import numpy as np
from .protocol import P

def force_at(local_time):
    t=float(local_time)
    magnitude=.5*P['peak_force_N']*(1-np.cos(2*np.pi*t/P['burn_s'])) if 0<=t<P['burn_s'] else 0.
    return magnitude*np.asarray(P['force_axis_world'])

def analytic_motion(t,mass):
    t=np.asarray(t);T=P['burn_s'];w=2*np.pi/T;A=.5*P['peak_force_N']/mass
    s=np.clip(t,0,T)
    v=A*(s-np.sin(w*s)/w)
    z=A*(s*s/2+(np.cos(w*s)-1)/w**2)+A*T*np.maximum(t-T,0)
    return z,v

def satellite_state(robot,data,target_id):
    import mujoco
    vel=np.zeros(6)
    mujoco.mj_objectVelocity(robot.model,data,mujoco.mjtObj.mjOBJ_BODY,target_id,vel,0)
    mass=float(robot.model.body_mass[target_id]);R=data.ximat[target_id].reshape(3,3)
    p=data.xipos[target_id].copy();omega,v=vel[:3],vel[3:]
    I=R@np.diag(robot.model.body_inertia[target_id])@R.T
    linear=mass*v;spin=I@omega;angular=spin+np.cross(p,linear)
    return np.r_[p,data.xmat[target_id].reshape(-1),v,omega,linear,angular,
                 .5*mass*(v@v)+.5*(omega@spin)]

STATE_COLUMNS=['position_world(3)','rotation_world(9)','linear_velocity_world(3)',
               'angular_velocity_world(3)','linear_momentum_world(3)',
               'angular_momentum_world_origin(3)','kinetic_energy_J(1)']
