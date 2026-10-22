import time,subprocess,zipfile
import numpy as np
import mujoco
from ff_prior.storage import RunLock,read_json,file_digest,digest
from ff_prior_pilot_train.storage import atomic_json,atomic_npz
from ff_prior.pipeline import audit,acceptance
from ff_prior_origin_learn_bounded.runtime import TaskOriginSceneCache
from ff_prior_static_fullchain_v2.proposals import load_bundle
from ff_prior_dynamic_reference.execution import integration
from ff_prior.initial_state import InitialState
from tracking_diffusion.environment import free_slices
from .protocol import ROOT,PACKAGE,OUT,BASE,P
from .inputs import identity
from .static import run_unit
from .budget import Budget
from .dynamic_budget import Budget as DynamicBudget
from .environment import Environment
from .evaluation import evaluate

def run(out=OUT,progress=lambda *a,**k:None):
    ident=identity();start=time.time();deadline=start+P['wall_s']
    with RunLock(out):
        assert not (out/'run.json').exists(),'UNKNOWN_WORK_NOT_REPLAYABLE'
        atomic_json(out/'run.json',dict(identity=ident,producer_head=subprocess.check_output(
            ['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),TEST=False))
        with zipfile.ZipFile(out/'executed_source.zip','w',zipfile.ZIP_DEFLATED) as archive:
            for f in ident['source_hashes']:archive.write(ROOT/f,f)
        entry=ident['selection']['entries'][0];unit=ident['selection']['units'][0]
        config=read_json(BASE/'run.json')['config'];physical,canonical,ci,_=audit(config)
        profile=acceptance(config)
        rr,ii,cc,proof=TaskOriginSceneCache(canonical,ci,profile,max_entries=1,max_scene_models=1).get(entry)
        atomic_json(out/'initial_scene_audit.json',proof)
        static=out/'static';static.mkdir();budget=Budget(static,deadline);budget.begin_unit();budget.save()
        atomic_json(out/'pending.json',dict(phase='static_planning_and_execution',unknown=False))
        before=rr.calls.copy()
        result=run_unit(static/'units/u00',unit,entry,rr,ii,cc,profile,load_bundle(),budget,progress)
        delta={k:rr.calls[k]-before[k] for k in before};budget.observe(delta);budget.save()
        atomic_json(static/'calls.json',delta);atomic_json(static/'summary.json',result)
        summary=dict(status='FINISHED',static=result,dynamic_status='BLOCKED_STATIC_NOT_VERIFIED',
            new_native_and_replay_steps=delta['physics_steps'],comparison_runs=0,new_training=0,
            obstacles=len(entry['scene']['objects']),TEST=False,full_longterm_complete=False)
        atomic_json(out/'summary.json',summary)
        if result['execution_level']=='EXECUTION_VERIFIED':
            progress('actual_terminal_handoff_same_nine_obstacles')
            path=static/'units/u00/execution/native_trace.npz';a=np.load(path)
            terminal={k:a[k][-1].copy() for k in ('integration','qpos','qvel','time','reference_q','reference_qd')}
            c=dict(ident['contract']);d=InitialState(terminal['integration'],ii.model_hash).reset(rr)
            target=rr.id(mujoco.mjtObj.mjOBJ_BODY,'target_satellite');rt=d.xmat[target].reshape(3,3);pt=d.xpos[target]
            goals=np.asarray(c['static_goals']);_,tv=free_slices(rr.model,rr.spec.target_free_joint_name)
            c.update(source_native_sha256=file_digest(path),grasp_local=rt.T@(goals[1,:3,3]-pt),
                grasp_rotation_local=rt.T@goals[1,:3,:3],target_velocity=d.qvel[tv].copy(),
                absolute_start_time=float(d.time),actual_derived_handoff=True)
            dyn=out/'dynamic';dyn.mkdir();atomic_json(dyn/'contract.json',c);atomic_npz(dyn/'initial.npz',**terminal)
            db=DynamicBudget(dyn);db.deadline=deadline;db.save();env=Environment(rr,terminal,c,db)
            assert np.array_equal(integration(rr,env.data),terminal['integration'])
            atomic_json(dyn/'handoff.json',dict(full836_exact=True,absolute_time=float(env.data.time),
                source_native_sha256=c['source_native_sha256'],same_scene_model=True,target_velocity_unchanged=True))
            reason=None
            for tick in range(1350):
                try:
                    db.check();db.pending=dict(phase='native',tick=tick,reserved_steps=10);db.save()
                    env.advance_native(env.velocity(None));db.pending=None
                    if tick%25==0:db.save();progress('nine_obstacle_tracking',ticks=tick+1,total=1350,absolute_time=float(env.data.time))
                except Exception as error:reason=repr(error);break
            assert len(env.controls)==len(env.history['time'])-1,'UNRESOLVED_NATIVE_STEP'
            db.pending=dict(phase='replay',reserved_steps=len(env.controls));db.save()
            directory=dyn/'execution';directory.mkdir()
            er=evaluate(env,directory,reason,progress,db);db.pending=None;db.save()
            atomic_json(dyn/'control_filter.json',env.filter_history)
            summary.update(dynamic_status=er['status'],dynamic_result=er,
                new_native_and_replay_steps=delta['physics_steps']+sum(db.counts.values()))
            atomic_json(dyn/'summary.json',summary)
            from .visuals import generate
            summary['visualization']=generate(dyn,rr,c,progress,min(deadline,time.time()+1200))
        else:
            from ff_prior_diffusion_demo.visuals import generate
            summary['visualization']=generate(static,rr,entry,progress,min(deadline,time.time()+1200))
        assert summary['new_native_and_replay_steps']<=P['native_steps_cap']
        assert identity()==ident,'FROZEN_IDENTITY_CHANGED'
        summary['engineering_chain_pass']=summary['dynamic_status']=='EXECUTION_VERIFIED'
        atomic_json(out/'pending.json',dict(phase=None,unknown=False))
        atomic_json(out/'summary.json',summary)
        (out/'REPORT.md').write_text('# 双侧障碍完整链实际结果\n\n'+str(summary)+'\n\n有限新场景，不代表网络独立避障能力或最终TEST。\n',encoding='utf-8')
        return summary
