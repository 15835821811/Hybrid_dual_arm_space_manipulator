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

def run(out=OUT,progress=lambda *a,**k:None):
    ident=identity()
    with RunLock(out):
        assert not (out/'run.json').exists(),'UNKNOWN_WORK_NOT_REPLAYABLE'
        atomic_json(out/'run.json',dict(identity=ident,producer_head=subprocess.check_output(
            ['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),TEST=False))
        with zipfile.ZipFile(out/'executed_source.zip','w',zipfile.ZIP_DEFLATED) as archive:
            for f in ident['source_hashes']:archive.write(ROOT/f,f)
        budget=Budget(out);budget.save();c=ident['contract']
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
        for tick in range(1350):
            try:
                budget.check();budget.pending=dict(phase='native',tick=tick,reserved_steps=10);budget.save()
                velocity=env.velocity(None)
                env.advance_native(velocity)
                budget.pending=None
                if tick%25==0:
                    budget.save();progress('continuous_native_tracking',ticks=tick+1,total=1350,
                        absolute_time=float(env.data.time),local_time=float(env.data.time-env.origin_time))
            except Exception as error:
                reason=repr(error);break
        # Actual saved prefix is replayed once even if feedback cannot finish.
        assert len(env.controls)==len(env.history['time'])-1,'UNRESOLVED_NATIVE_STEP'
        budget.pending=dict(phase='replay',reserved_steps=len(env.controls));budget.save()
        result=evaluate(env,directory,reason,progress,budget)
        budget.pending=None;budget.save()
        atomic_json(out/'control_filter.json',env.filter_history)
        passed=result['status']=='EXECUTION_VERIFIED'
        summary=dict(status='FINISHED',engineering_dynamic_continuation_pass=passed,
            execution_denominator=1,execution_level=result['status'],actual_terminal_handoff=equal,
            new_native_steps=len(env.controls),new_replay_steps=budget.counts['replay'],
            moving_satellite_pass=False,moving_satellite_status='NOT_RUN_INHERITED_ZERO_VELOCITY',
            training_updates=0,comparison_runs=0,TEST=False,full_long_term_goal_complete=False)
        atomic_json(out/'summary.json',summary)
        from .visuals import generate
        summary['visualization']=generate(out,robot,c,progress,min(budget.deadline,time.time()+1200))
        assert identity()==ident
        atomic_json(out/'summary.json',summary)
        (out/'REPORT.md').write_text('# 真终态接动态航点\n\n'+str(summary)+'\n\n该组合与原独立DEV区分。刚性臂保持原捕获位姿，卫星零速度继承。\n',encoding='utf-8')
        return summary
