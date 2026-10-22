"""Exactly four frozen TRAIN sources, new representation and new physical evidence."""
import time
import subprocess
import zipfile
import numpy as np
from tracking_diffusion.environment import TrackingEnv,Task
from ff_prior.robot_adapter import RobotAdapter
from ff_prior.scenes import SceneSpec,Obstacle
from ff_prior.storage import RunLock,read_json,digest
from ff_prior_pilot_train.storage import atomic_json,atomic_npz
from .protocol import ROOT,P
from .inputs import identity
from .representation import fit
from .execution import Environment,evaluate

class Budget:
    def __init__(self,out):
        self.out=out;self.started=time.monotonic();self.native=0;self.replay=0
        self.deadline=self.started+P['wall_s'];self.inflight=None
    def check(self):
        if time.monotonic()>self.deadline:raise TimeoutError('WALL_BUDGET')
    def step(self,kind):
        self.check()
        if self.native+self.replay>=P['max_native_replay_steps']:raise TimeoutError('STEP_BUDGET')
        setattr(self,kind,getattr(self,kind)+1)
    def save(self):
        self.check()
        size=sum(p.stat().st_size for p in self.out.rglob('*') if p.is_file())
        if size>P['max_bytes']:raise TimeoutError('OUTPUT_BUDGET')
        atomic_json(self.out/'budget.json',dict(native_steps_reserved=self.native,replay_steps_reserved=self.replay,
            elapsed_s=time.monotonic()-self.started,output_bytes=size,inflight=self.inflight,limits=P))

def run(out,notify):
    budget=Budget(out);ident=identity()
    def progress(stage,**details):
        budget.save();notify(stage,**details)
    with RunLock(out):
        atomic_json(out/'run.json',dict(identity=ident,producer_head=subprocess.check_output(
            ['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),TEST=False))
        with zipfile.ZipFile(out/'executed_source.zip','w',zipfile.ZIP_DEFLATED) as z:
            for path in ident['source_hashes']:z.write(ROOT/path,path)
        results=[]
        for i,row in enumerate(ident['selection']['rows']):
            directory=out/f't{i:02}';directory.mkdir()
            budget.inflight=dict(case=i,seed=row['seed']);progress('initialize',case=i)
            source=ROOT/row['path'];task=read_json(source/'result.json')['task']
            old=TrackingEnv(Task(**task))
            scene=SceneSpec(tuple(Obstacle(o.name,'sphere',tuple(o.center),(o.radius,0.,0.))
                for o in old.obstacles),'legacy_moving_preserved')
            robot=RobotAdapter(scene=scene)
            atomic_json(directory/'limits.json',dict(lower=robot.lower,upper=robot.upper,
                velocity=robot.spec.planner_velocity_limits,acceleration=robot.spec.planner_acceleration_limits,
                native_torque=robot.spec.torque_limits,B=robot.B,Bplus=robot.Bplus))
            with np.load(source/'trace.npz',allow_pickle=False) as a:
                t,q,dq=a['time'],a['q'],a['dq']
                assert np.max(abs(np.diff(t)-.02))<1e-10 and len(t)==1351
                np.testing.assert_array_equal(old.data.qpos,a['qpos'][0])
                np.testing.assert_array_equal(old.data.qvel,a['qvel'][0])
                spline,report=fit(t,q,dq,robot.lower,robot.upper,robot.spec.planner_velocity_limits,
                    robot.spec.planner_acceleration_limits,progress,budget.deadline)
                atomic_npz(directory/'reference.npz',knots=spline.t,coefficients=spline.c,
                    source_time=t,source_q=q,source_dq=dq)
            report.update(source=row,task=task,task_hash=digest(task),new_TRAIN_entry=False)
            atomic_json(directory/'representation.json',report)
            if not all(report['gates'].values()):
                result=dict(status='REPRESENTATION_REJECTED',steps=0,gates=report['gates'])
                atomic_json(directory/'result.json',result)
            else:
                env=Environment(old,robot)
                atomic_npz(directory/'initial.npz',values=env.initial.values)
                atomic_json(directory/'identity.json',dict(robot=robot.identity(),scene=scene.to_dict(),
                    initial_state_id=env.initial.state_id,task=task,task_hash=digest(task),
                    source=row,waypoints=env.path.to_dict(),reference='C2_CONSTRAINED_BSPLINE'))
                reason=None
                try:
                    env.record()
                    for k in range(P['steps_each']):
                        budget.step('native');env.step_reference(spline,(k+1)*P['dt'])
                        if k%500==0:progress('native_torque',case=i,steps=k+1)
                except RuntimeError as error:
                    reason=str(error)
                # All actual failures keep their attempted trajectory and denominator.
                result=evaluate(env,directory,reason,progress,budget)
            results.append(result)
            budget.inflight=None;progress('case_complete',case=i,status=result['status'])
            atomic_json(out/'progress_results.json',results)
            del robot,old
        same=identity()==ident
        atomic_json(out/'source_preservation.json',dict(identity_unchanged=same))
        if not same:raise RuntimeError('FROZEN_IDENTITY_CHANGED')
        ok=all(r['status']=='EXECUTION_VERIFIED' for r in results)
        summary=dict(status='FINISHED' if ok else 'FINISHED_WITH_BLOCKED_ITEMS',results=results,
            denominator=4,executed=sum(r['steps']>0 for r in results),
            verified=sum(r['status']=='EXECUTION_VERIFIED' for r in results),
            elapsed_s=time.monotonic()-budget.started,training_updates=0,TRAIN_added=0,TEST=False,
            next_dynamic_DEV_gate=ok,independent_review='PENDING')
        atomic_json(out/'summary.json',summary);budget.save()
        (out/'REPORT.md').write_text('# P2a 动态参考运行结果\n\n'+
            f"四条固定TRAIN来源：执行{summary['executed']}/4，验证{summary['verified']}/4。独立审查与分析报告待完成。\n",
            encoding='utf-8')
        return summary
