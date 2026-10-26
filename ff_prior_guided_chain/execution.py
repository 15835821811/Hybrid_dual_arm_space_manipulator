"""Own actual/replay objects persist; old gate formulas copied without relaxation."""
import time
import numpy as np
from ff_prior_balanced_restore.native_budget import NativeBudget
from ff_prior_dynamic_reference.execution import integration
from ff_prior_pilot_train.storage import atomic_json,atomic_npz
from .dynamic_environment import Environment as BEnvironment
from .satellite_environment import Environment as CEnvironment
from .protocol import P


class StaticBudget(NativeBudget):
    def __init__(self,path,parent):
        self.parent=parent
        super().__init__(path,min(parent.deadline,time.time()+P['native_s']),on_check=parent.check)
    def before_step(self,kind):
        super().before_step(kind);self.parent.reserve('native')
    def after_step(self,kind):
        super().after_step(kind);self.parent.complete('native')
    def save(self):
        result=super().save();self.parent.save();return result


class DynamicBudget:
    def __init__(self,parent,path,phase):
        self.parent,self.path,self.phase=parent,path,phase
        self.counts=dict(native=0,replay=0);self.attempted=dict(self.counts);self.inflight=None
        self.deadline=min(parent.deadline,time.time()+P['native_s'])
        self.physics_deadline=self.deadline
    def check(self,*args):
        self.parent.check()
        if time.time()>self.deadline:raise RuntimeError('NATIVE_TIME_RESERVE')
    def reserve(self,kind):
        self.check()
        if self.inflight is not None or self.attempted[kind]>=P['phase_intervals'][self.phase]:raise RuntimeError('NATIVE_PHASE_RESERVATION')
        if kind=='replay' and self.attempted[kind]>=self.counts['native']:raise RuntimeError('UNKNOWN_NATIVE_PREFIX')
        self.parent.reserve('native');self.attempted[kind]+=1;self.inflight=(kind,self.attempted[kind])
    def complete(self,kind):
        assert self.inflight==(kind,self.counts[kind]+1)
        self.counts[kind]+=1;self.inflight=None;self.parent.complete('native')
    def save(self):
        atomic_json(self.path/'native_budget.json',dict(completed=self.counts,attempted=self.attempted,inflight=self.inflight,
            cap_each=P['phase_intervals'][self.phase],no_unknown_resume=True));self.parent.save()


class PlannedController:
    def velocity(self,prior=None):
        q,_=self.state();local=float(self.data.time-self.origin_time)
        planned_q=self.guided_reference.evaluate(local)
        planned_v=self.guided_reference.evaluate(local,1)+.5*(planned_q-q)
        g,drift=self.generalized_map();j,offset=self.task_jacobian(g,drift)
        weights=np.repeat([1.,2.,1.4,2.4],3);jw=weights[:,None]*j
        h=jw.T@jw+1e-5*np.eye(17)
        # Track the generated whole-joint trajectory first; feedback corrects
        # task-space error and the existing full-body safety filter is final.
        feedback=np.linalg.solve(h,jw.T@(weights*(self.desired_velocity(True)-offset-j@planned_v)))
        desired=planned_v+feedback
        filtered=self.velocity_filter(desired,g,drift,h)
        self.contribution.append(dict(time=float(self.data.time),planned_q=planned_q,
            planned_velocity=planned_v,world_feedback=feedback,pre_filter=desired,post_filter=filtered))
        return filtered


class GuidedB(PlannedController,BEnvironment):pass
class GuidedC(PlannedController,CEnvironment):pass


def dynamic(robot,terminal,contract,phase,reference,parent,path,progress):
    path.mkdir(parents=True,exist_ok=False);budget=DynamicBudget(parent,path,phase)
    env=(GuidedB if phase=='B' else GuidedC)(robot,terminal,contract,budget)
    env.guided_reference=reference;env.contribution=[];reason=None
    actual_id=id(robot._guided_actual_data);replay_id=id(robot._guided_replay_data)
    atomic_npz(path/'initial.npz',integration=integration(robot,env.data))
    for tick in range(P['phase_intervals'][phase]//10):
        try:
            budget.check();env.advance_native(env.velocity())
            if tick%25==0:
                budget.save();progress('native_tracking',phase=phase,ticks=tick+1,absolute_time=float(env.data.time))
        except Exception as error:reason=repr(error);break
    if budget.inflight is not None or len(env.controls)!=len(env.integration_history)-1:
        atomic_npz(path/'partial_native_trace.npz',integration=np.asarray(env.integration_history),ctrl=np.asarray(env.controls))
        atomic_json(path/'unknown_work.json',dict(reason=reason,inflight=budget.inflight));budget.save()
        raise RuntimeError('UNKNOWN_NATIVE_STEP_NO_REPLAY')
    atomic_json(path/'network_controller_contribution.json',env.contribution)
    atomic_json(path/'control_filter.json',env.filter_history);env.save_constraint_observations(path)
    if phase=='B':from .dynamic_evaluation import evaluate
    else:from .satellite_evaluation import evaluate
    result=evaluate(env,path,reason,progress,budget);budget.save()
    if phase=='C':
        from scipy.spatial.transform import Rotation
        from .predictor import world_momentum
        base=np.asarray(env.history['base']);generalized=np.asarray(env.history['momentum'])
        robot_world=[]
        for q,g in zip(base,generalized):
            p,h=world_momentum(g,q[:3],Rotation.from_quat(np.roll(q[3:],-1)).as_matrix());robot_world.append(np.r_[p,h])
        robot_world=np.asarray(robot_world);satellite=np.asarray(env.satellite_history)[:,18:24]
        total=robot_world+satellite
        atomic_npz(path/'whole_system_momentum.npz',time=np.asarray(env.history['time']),
            robot_world=robot_world,satellite_world=satellite,total_world=total,
            applied_linear_impulse=np.asarray(env.impulse),applied_angular_impulse=np.asarray(env.angular_impulse))
        atomic_json(path/'whole_system_momentum.json',dict(role='additional physical balance diagnostic; does not replace original robot momentum gate',
            world_origin='same fixed world origin in all terms',initial_robot_residual_subtracted_from_gate=False))
    assert id(robot._guided_actual_data)==actual_id and id(robot._guided_replay_data)==replay_id
    terminal=dict(integration=integration(robot,env.data),reference_q=env.reference_q.copy(),reference_qd=env.command.copy(),
        reference_qdd=np.asarray(env.command_qdd[-1]) if env.command_qdd else terminal['reference_qdd'])
    atomic_npz(path/'terminal.npz',**terminal)
    atomic_json(path/'object_continuity.json',dict(actual_object=actual_id,replay_object=replay_id,
        actual_phase_initializations=0,replay_phase_initializations=0,
        actual_replay_terminal_equal=np.array_equal(integration(robot,robot._guided_actual_data),integration(robot,robot._guided_replay_data))))
    return result,terminal
