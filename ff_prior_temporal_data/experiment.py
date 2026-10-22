"""A fixed 8 TRAIN / 4 VAL subset; no replacement, no library writes, no learning."""
import time
import shutil
import subprocess
import zipfile
import numpy as np
from scipy.interpolate import BSpline
from tracking_diffusion.environment import TrackingEnv,Task
from ff_prior.robot_adapter import RobotAdapter
from ff_prior.scenes import SceneSpec,Obstacle
from ff_prior.storage import RunLock,read_json,digest,file_digest
from ff_prior_pilot_train.storage import atomic_json,atomic_npz
from ff_prior_dynamic_margin.representation import fit
from ff_prior_dynamic_reference.execution import Environment,evaluate
from .protocol import ROOT,P
from .inputs import identity

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

def training_gate(rows):
    counts={s:sum(r['split']==s and r['status']=='EXECUTION_VERIFIED' for r in rows)
            for s in ['train','validation']}
    return dict(counts=counts,passed=counts['train']>=P['gate_train'] and counts['validation']>=P['gate_validation'])

def reuse_result(previous,current_id,initial,spline,limits):
    old_result=read_json(previous/'result.json');old_id=read_json(previous/'identity.json')
    old_source=read_json(previous.parent/'run.json')['identity']['source_hashes']
    with np.load(previous/'reference.npz',allow_pickle=False) as a:
        reference_exact=bool(np.array_equal(spline.t,a['knots']) and np.array_equal(spline.c,a['coefficients']))
    with np.load(previous/'initial.npz',allow_pickle=False) as a:
        initial_exact=bool(np.array_equal(initial,a['values']))
    checks=dict(reference_exact=reference_exact,initial_exact=initial_exact,
        identity_exact=digest(current_id)==digest(old_id),limits_exact=digest(limits)==digest(read_json(previous/'limits.json')),
        source_exact=all(file_digest(ROOT/p)==h for p,h in old_source.items()),
        old_qualified=old_result['status']=='EXECUTION_VERIFIED' and all(old_result['gates'].values()),
        native_hash_exact=file_digest(previous/'native_trace.npz')==old_result['native_trace_sha256'])
    ok=all(checks.values())
    result=dict(old_result,new_execution=False,actual_new_steps=0,reused_from=previous.relative_to(ROOT).as_posix()) if ok else dict(status='REUSE_IDENTITY_BLOCKED',steps=0,new_execution=False)
    return result,dict(checks=checks,passed=ok,new_execution=False,reused_from=previous.relative_to(ROOT).as_posix())

def run(out,notify):
    budget=Budget(out);ident=identity()
    def progress(stage,**details):budget.save();notify(stage,**details)
    with RunLock(out):
        atomic_json(out/'run.json',dict(identity=ident,producer_head=subprocess.check_output(
            ['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),TEST=False))
        with zipfile.ZipFile(out/'executed_source.zip','w',zipfile.ZIP_DEFLATED) as z:
            for path in ident['source_hashes']:z.write(ROOT/path,path)
        # Legacy inventory is NOT upgraded by inspecting or certifying another row.
        atomic_json(out/'inventory.json',dict(counts={'train':32,'validation':8},
            selected=[r['id'] for r in ident['selection']['rows']],
            unselected_count=28,unselected_current_grade='LEGACY_NOT_REQUALIFIED',
            source=ident['selection']['inventory'],new_tasks=0))
        results=[];qualified=[]
        for item in ident['selection']['rows']:
            unit=item['id'];row=item['source'];directory=out/unit;directory.mkdir()
            budget.inflight=dict(unit=unit,seed=row['seed']);progress('initialize',unit=unit)
            source=ROOT/row['path'];task=read_json(source/'result.json')['task']
            old=TrackingEnv(Task(**task))
            scene=SceneSpec(tuple(Obstacle(o.name,'sphere',tuple(o.center),(o.radius,0.,0.))
                for o in old.obstacles),'legacy_moving_preserved')
            robot=RobotAdapter(scene=scene)
            limits=dict(lower=robot.lower,upper=robot.upper,velocity=robot.spec.planner_velocity_limits,
                acceleration=robot.spec.planner_acceleration_limits,native_torque=robot.spec.torque_limits,B=robot.B,Bplus=robot.Bplus)
            atomic_json(directory/'limits.json',limits)
            with np.load(source/'trace.npz',allow_pickle=False) as a:
                t,q,dq=a['time'].copy(),a['q'].copy(),a['dq'].copy()
                assert len(t)==1351 and np.max(abs(np.diff(t)-.02))<1e-10
                np.testing.assert_array_equal(old.data.qpos,a['qpos'][0]);np.testing.assert_array_equal(old.data.qvel,a['qvel'][0])
            if item['reuse']:
                previous=ROOT/item['reuse']
                with np.load(previous/'reference.npz',allow_pickle=False) as a:
                    spline=BSpline(a['knots'].copy(),a['coefficients'].copy(),3,extrapolate=False)
                    np.testing.assert_array_equal(t,a['source_time']);np.testing.assert_array_equal(q,a['source_q']);np.testing.assert_array_equal(dq,a['source_dq'])
                shutil.copyfile(previous/'reference.npz',directory/'reference.npz')
                report=dict(read_json(previous/'representation.json'),reused_representation=True)
            else:
                spline,report=fit(t,q,dq,robot.lower,robot.upper,robot.spec.planner_velocity_limits,
                    robot.spec.planner_acceleration_limits,progress,budget.deadline)
                atomic_npz(directory/'reference.npz',knots=spline.t,coefficients=spline.c,source_time=t,source_q=q,source_dq=dq)
                report.update(source=row,task=task,task_hash=digest(task),new_TRAIN_entry=False)
            atomic_json(directory/'representation.json',report)
            env=Environment(old,robot)
            current_id=dict(robot=robot.identity(),scene=scene.to_dict(),initial_state_id=env.initial.state_id,
                task=task,task_hash=digest(task),source=row,waypoints=env.path.to_dict(),reference='C2_CONSTRAINED_BSPLINE')
            atomic_json(directory/'identity.json',current_id);atomic_npz(directory/'initial.npz',values=env.initial.values)
            if not all(report['gates'].values()):
                result=dict(status='REPRESENTATION_REJECTED',steps=0,new_execution=False,gates=report['gates'])
            elif item['reuse']:
                result,receipt=reuse_result(previous,current_id,env.initial.values,spline,limits)
                atomic_json(directory/'reuse.json',receipt)
            else:
                reason=None
                try:
                    env.record()
                    for k in range(P['steps_each']):
                        budget.step('native');env.step_reference(spline,(k+1)*P['dt'])
                        if k%500==0:progress('native_torque',unit=unit,steps=k+1)
                except RuntimeError as error:reason=str(error)
                result=evaluate(env,directory,reason,progress,budget)
                result.update(new_execution=True,actual_new_steps=result['steps'])
            result.update(unit=unit,split=item['split'],seed=row['seed'])
            atomic_json(directory/'result.json',result);results.append(result)
            if result['status']=='EXECUTION_VERIFIED':
                qualified.append(dict(unit=unit,split=item['split'],seed=row['seed'],
                    reference=(directory/'reference.npz').relative_to(ROOT).as_posix(),
                    reference_sha256=file_digest(directory/'reference.npz'),
                    identity=(directory/'identity.json').relative_to(ROOT).as_posix(),
                    initial=(directory/'initial.npz').relative_to(ROOT).as_posix(),
                    evidence=item['reuse'] or directory.relative_to(ROOT).as_posix(),grade='EXECUTION_VERIFIED'))
            budget.inflight=None;progress('case_complete',unit=unit,status=result['status'])
            atomic_json(out/'progress_results.json',results)
            del env,robot,old
        same=identity()==ident;atomic_json(out/'source_preservation.json',dict(identity_unchanged=same))
        if not same:raise RuntimeError('FROZEN_IDENTITY_CHANGED')
        gate=training_gate(results)
        atomic_json(out/'qualified_view.json',dict(rows=qualified,gate=gate,master_library_modified=False,
            validation_used_in_training=False,new_tasks=0,not_an_upgrade_of_unselected_rows=True))
        summary=dict(status='FINISHED' if gate['passed'] else 'FINISHED_WITH_BLOCKED_ITEMS',results=results,
            denominator=12,selected_train=8,selected_validation=4,inventory_count=40,unselected_unqualified=28,
            actual_new_executions=sum(r.get('new_execution',False) for r in results),
            historical_reused=sum(bool(r.get('reused_from')) for r in results),training_gate=gate,
            elapsed_s=time.monotonic()-budget.started,training_updates=0,TRAIN_added=0,TEST=False,
            independent_review='PENDING',next='P3b_freeze_one_temporal_candidate' if gate['passed'] else 'P3_learning_BLOCKED_continue_P1_independent_repair')
        atomic_json(out/'summary.json',summary);budget.save()
        (out/'REPORT.md').write_text('# 既有动态数据资格认证\n\n'+
            f"预选12条，新增执行{summary['actual_new_executions']}，历史复用{summary['historical_reused']}。当前分级{gate}；无训练、无扩库。新独立审查待完成。\n",encoding='utf-8')
        return summary
