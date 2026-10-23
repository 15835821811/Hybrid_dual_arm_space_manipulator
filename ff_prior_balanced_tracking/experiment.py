import time,subprocess,zipfile
import numpy as np
from ff_prior.storage import RunLock,read_json,digest
from ff_prior_pilot_train.storage import atomic_json,atomic_npz
from ff_prior.robot_adapter import RobotAdapter
from ff_prior.scenes import SceneSpec
from ff_prior_dynamic_reference.execution import integration
from .protocol import ROOT,PACKAGE,OUT,P
from .inputs import identity
from .environment import Environment
from .budget import Budget
from .evaluation import evaluate

def run(out=OUT,progress=lambda *a,**k:None,started=None):
    budget=Budget(out,started)
    ident=identity()
    with RunLock(out):
        tests=read_json(PACKAGE/'evidence/software_tests.json')
        assert tests['status']=='PASS' and tests['terminal_preflight'] is True
        assert tests['source_hashes']==ident['source_hashes']
        assert not (out/'run.json').exists(),'UNKNOWN_WORK_NOT_REPLAYABLE'
        budget.save()
        atomic_json(out/'run.json',dict(identity=ident,producer_head=subprocess.check_output(
            ['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),TEST=False))
        with zipfile.ZipFile(out/'executed_source.zip','w',zipfile.ZIP_DEFLATED) as archive:
            for f in ident['source_hashes']:archive.write(ROOT/f,f)
        budget.check();budget.save();c=ident['contract']
        progress('load_actual_predecessor_terminal')
        robot=RobotAdapter(scene=SceneSpec.from_dict(c['scene']))
        terminal=dict(np.load(PACKAGE/'evidence/terminal.npz'))
        env=Environment(robot,terminal,c,budget)
        equal=np.array_equal(integration(robot,env.data),terminal['integration'])
        assert equal
        atomic_json(out/'handoff.json',dict(full836_exact=equal,absolute_time=float(env.data.time),
            qpos_exact=np.array_equal(env.data.qpos,terminal['qpos']),qvel_exact=np.array_equal(env.data.qvel,terminal['qvel']),
            source_native_sha256=c['source_native_sha256'],contract_hash=c['contract_hash'],no_target_velocity_change=True))
        atomic_npz(out/'initial.npz',integration=integration(robot,env.data))
        directory=out/'execution';directory.mkdir()
        reason=None
        for tick in range(P['control_ticks']):
            try:
                budget.check('physics');budget.pending=dict(phase='native',tick=tick,reserved_steps=10);budget.save()
                velocity=env.velocity(None)
                env.advance_native(velocity)
                budget.pending=None
                if tick%25==0:
                    budget.save();progress('continuous_native_tracking',ticks=tick+1,total=P['control_ticks'],
                        absolute_time=float(env.data.time),local_time=float(env.data.time-env.origin_time))
            except Exception as error:
                reason=repr(error);break
        atomic_json(out/'control_filter.json',env.filter_history)
        env.save_constraint_observations(directory)
        # Preserve ambiguous partial evidence, but never replay unknown work.
        if budget.inflight is not None or len(env.controls)!=len(env.history['time'])-1:
            atomic_npz(directory/'partial_native_trace.npz',
                **{k:np.asarray(v) for k,v in env.history.items()},
                integration=np.asarray(env.integration_history),ctrl=np.asarray(env.controls),raw_torque=np.asarray(env.raw))
            try:
                env.body_recorder.save(directory,source_native_path=directory/'partial_native_trace.npz',
                    expected_samples=P['control_ticks']+1,failure=reason or 'UNKNOWN_NATIVE_STEP')
            except Exception as error:
                atomic_json(directory/'body_diagnostics_failure.json',dict(status='PARTIAL',reason=repr(error)))
            atomic_json(directory/'unknown_native_stop.json',dict(reason=reason,
                controls=len(env.controls),saved_states=len(env.history['time']),inflight=budget.inflight))
            budget.save()
            raise RuntimeError('UNRESOLVED_NATIVE_STEP_REPLAY_FORBIDDEN')
        # The final actual state has no following filter command. Preserve it
        # once, without advancing dynamics or changing any physical state.
        try:
            rows=env.body_recorder.rows
            if not rows or float(env.data.time)>rows[-1]['time']+1e-10:
                env.body_recorder.sample(env.data,command_before=None,command_after=None,
                    active_constraints=[],filter_reason='ACTUAL_TERMINAL_NO_FOLLOWING_FILTER',
                    native_index=len(env.controls))
        except Exception as error:
            env.body_diagnostic_failures.append(dict(time=float(env.data.time),stage='actual_terminal',
                reason=repr(error),physical_outcome_unchanged=True))
        # Only the known actual saved prefix is independently replayed once.
        budget.pending=dict(phase='replay',reserved_steps=len(env.controls));budget.save()
        result=evaluate(env,directory,reason,progress,budget)
        budget.pending=None;budget.save()
        passed=result['status']=='EXECUTION_VERIFIED'
        summary=dict(status='FINISHED' if passed else 'FINISHED_WITH_BLOCKED_ITEMS',engineering_dynamic_continuation_pass=passed,
            execution_denominator=1,execution_level=result['status'],actual_terminal_handoff=equal,
            new_native_steps=len(env.controls),new_replay_steps=budget.counts['replay'],
            new_coupled_calls=0,new_neural_samples=0,whole_arm_diagnostics='execution/body_obstacle_trace.npz',
            moving_satellite_pass=False,moving_satellite_status='NOT_RUN_INHERITED_ZERO_VELOCITY',
            training_updates=0,comparison_runs=0,TEST=False,full_long_term_goal_complete=False)
        atomic_json(out/'summary.json',summary)
        from .visuals import generate
        try:summary['visualization']=generate(out,robot,c,progress,min(budget.deadline,time.time()+P['media_reserve_s']))
        except Exception as error:summary['visualization']=dict(status='MEDIA_PARTIAL',error=repr(error),new_physics_steps=0)
        budget.check('total');budget.save()
        assert identity()==ident
        atomic_json(out/'summary.json',summary)
        (out/'REPORT.md').write_text('# 真终态接动态航点\n\n'+str(summary)+'\n\n该组合与原独立DEV区分。刚性臂保持原捕获位姿，卫星零速度继承。\n',encoding='utf-8')
        return summary
