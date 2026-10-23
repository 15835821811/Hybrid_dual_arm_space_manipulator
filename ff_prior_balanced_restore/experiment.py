import subprocess
import time
import zipfile
import numpy as np
from ff_prior.storage import read_json,digest,RunLock
from ff_prior_pilot_train.storage import atomic_json,atomic_npz,Ledger
from ff_prior.pipeline import audit,acceptance
from ff_prior.contracts import assert_frozen,assert_task_compatible
from ff_prior_loop.runtime import at_origin
from ff_prior_complex.tasks import transfer_audit
from ff_prior.spline import Reference
from ff_prior.proposal_generator import quick_gates
from .native_budget import NativeBudget
from ff_prior_static_goal_seed.validation import validate
from .physical_audit import audit_execution
from .protocol import ROOT,PACKAGE,OUT,OLD,BASE,SOURCE,P
from .inputs import identity,runtime
from .budget import Budget,SliceBudget,BudgetExhausted
from .motion import Motion
from .solver import Evaluator,optimize


def assert_resume_safe(out):
    if (out/'ledger.json').exists():raise ValueError('UNKNOWN_OR_INTERRUPTED_WORK_NOT_REPLAYABLE')


def unit(directory,spec,entry,robot,initial,collision,proof,profile,budget,progress):
    task=entry['task']
    with np.load(ROOT/spec['guess_file']) as a:z=a['Z'].copy()
    motion=Motion(robot,task['condition']['initial_features'])
    assert motion.valid(z),'BOUND_GUESS_MOTION_INVALID'
    atomic_json(directory/'guess_identity.json',dict(source=spec,
        original_numeric_initial_state=True,new_scene_qualification='UNQUALIFIED_GUESS',
        all_323_variables_free=True,old_physics_repeated=False,source_terminal_not_injected=True))
    budget.check()
    quota=SliceBudget(budget,P['restore_coupled_cap'],P['restore_interval_cap'],
        min(time.time()+P['restore_wall_s'],budget.deadline-P['execution_reserve_s']-P['diagnostics_wall_s']-P['visualization_wall_s']))
    restored=optimize(directory/'restore',z,motion,Evaluator(task,robot,initial,collision,quota),progress)
    progress('original_complete_model_certification',unit='u00')
    budget.ensure_output(512*1024**2)
    quota=SliceBudget(budget,40,200000,budget.deadline-P['execution_reserve_s']-P['diagnostics_wall_s']-P['visualization_wall_s'])
    validation=validate(directory/'validated',np.asarray(restored['Z']),task,robot,initial,collision,profile,quota)
    result=dict(status='DONE',unit=spec,candidate_denominator=1,execution_denominator=1,
        eligible=bool(validation['eligible']),qualified=int(validation['eligible']),
        execution_level='NOT_EXECUTED',reason='MODEL_REJECTED',validation_path='validated',
        reference_hash=validation['reference_hash'],restoration=restored,
        new_network_samples=0,historical_neural_proposals_reused=1,training_updates=0,TRAIN_added=0)
    atomic_json(directory/'execution_selection.json',dict(before_execution=True,selected=result['eligible'],
        rule='one frozen full-horizon candidate; no replacements',task_hash=task['task_hash'],
        original_initial_state_id=initial.state_id,model_prefix_final_state_never_injected=True))
    if result['eligible']:
        budget.check();assert budget.native_steps+27500<=P['native_steps_cap']
        budget.ensure_output(768*1024**2)
        atomic_json(directory/'native_reservation.json',dict(execution=13750,replay=13750,before_execution=True,
            initial_state_id=initial.state_id,original_full_state=True))
        progress('native67_execution_and_independent836_replay',unit='u00')
        execution=audit_execution(task,profile,robot,initial,Reference(validation['Z'],motion.q0,27.),
            directory/'execution',budget.deadline-P['diagnostics_wall_s']-P['visualization_wall_s'],
            native_budget=NativeBudget(directory/'execution',budget.deadline-P['diagnostics_wall_s']-P['visualization_wall_s'],
                steps_each=13750,total_cap=P['native_steps_cap'],on_check=budget.check))
        result.update(execution_level=execution['level'],reason='NATIVE_EXECUTION_ATTEMPTED')
        # Persist the scientific outcome before optional saved-state observations.
        atomic_json(directory/'result.json',result)
        try:
            from ff_prior_balanced_evidence.diagnostics import record_saved_trace
            budget.ensure_output(512*1024**2)
            progress('saved_native_whole_arm_diagnostics',unit='u00')
            result['body_diagnostics']=record_saved_trace(robot,entry['scene'],directory/'execution/native_trace.npz',
                directory/'body_diagnostics',control_dt=.002,
                deadline=min(budget.deadline-P['visualization_wall_s'],time.time()+P['diagnostics_wall_s']),notify=progress)
        except Exception as error:
            result['body_diagnostics']=dict(status='PARTIAL',reason=repr(error),physical_outcome_unchanged=True)
            atomic_json(directory/'body_diagnostics_failure.json',result['body_diagnostics'])
    assert_frozen(task);atomic_json(directory/'result.json',result)
    return result


def run(out=OUT,progress=lambda *a,**k:None,started=None):
    started=time.time() if started is None else started
    assert_resume_safe(out);ident=identity()
    with RunLock(out):
        assert read_json(PACKAGE/'evidence/software_tests.json')['status']=='PASS'
        assert read_json(PACKAGE/'evidence/software_tests.json')['source_hashes']==ident['source_hashes']
        atomic_json(out/'run.json',dict(identity=ident,producer_head=subprocess.check_output(
            ['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),TEST=False))
        with zipfile.ZipFile(out/'executed_source.zip','w',zipfile.ZIP_DEFLATED) as z:
            for p in ident['source_hashes']:z.write(ROOT/p,p)
        deadline=started+P['wall_s'];config=read_json(BASE/'run.json')['config']
        physical,canonical,initial,_=audit(config);old=read_json(SOURCE/'physical_contract.json')
        assert physical['acceptance']==old['acceptance'] and digest(physical['identity'])==digest(old['identity'])
        assert physical['initial_state_id']==old['initial_state_id']
        atomic_json(out/'physical_contract.json',physical);profile=acceptance(config)
        ledger=Ledger(out,{'identity_hash':digest(ident)});budget=Budget(out,deadline);budget.begin_unit()
        spec=ident['selection']['units'][0];entry=ident['selection']['entries'][0];directory=out/'units/u00'
        with np.load(PACKAGE/'evidence/initial_state.npz') as a:values=a['values'].copy()
        rr,ii,cc=runtime(entry,values)
        parent,pi,_=at_origin(canonical,entry['task']['condition']['initial_features'])
        assert np.array_equal(pi.values,values),'FULL_INITIAL836_CHANGED'
        transfer=transfer_audit(parent,pi,rr,ii)
        scan=cc.scan(ii.reset(rr),0.)
        assert scan['passed'],'NEW_SCENE_INITIAL_PENETRATION'
        proof=dict(scene_transfer=transfer,initial_geometry=scan,original_full_numeric_initial_state_exact=True,
            scene_task_origin_state_id=ii.state_id,comparison_origin='bound_original836_full_state')
        assert_task_compatible(entry['task'],rr,ii,cc,profile)
        atomic_json(out/'scenes'/spec['task_id']/'initial_audit.json',proof)
        atomic_json(out/'scenes'/spec['task_id']/'limits.json',dict(lower=rr.lower,upper=rr.upper))
        ledger.begin('u00','full27s_323_restoration',P['wall_s'],1);before=rr.calls.copy()
        try:
            result=unit(directory,spec,entry,rr,ii,cc,proof,profile,budget,progress)
        except (BudgetExhausted,RuntimeError,ValueError) as error:
            allowed=isinstance(error,BudgetExhausted) or str(error) in (
                'BUDGET_EXHAUSTED','MODEL_NEAR_RECORD_MEMORY_BUDGET','SPARSE_DENSE_NUMERICAL_DISAGREEMENT','PREFIX_REUSE_IDENTITY_MISMATCH','MODEL_NUMERICAL_GATE_FAILED')
            if not allowed:raise
            result=dict(status='BLOCKED',reason=str(error),unit=spec,candidate_denominator=1,
                execution_denominator=1,eligible=False,qualified=0,execution_level='NOT_COMPLETED')
            atomic_json(directory/'result.json',result)
        finally:
            delta={k:rr.calls[k]-before[k] for k in before};budget.observe(delta)
            atomic_json(directory/'calls.json',delta);budget.save()
        ledger.finish(result,delta)
        assert identity()==ident,'FROZEN_SOURCE_OR_INPUT_CHANGED'
        from .visuals import generate
        progress('saved_state_visualization')
        try:media=generate(out,rr,entry,progress,min(deadline,time.time()+P['visualization_wall_s']),'u00')
        except Exception as e:media=dict(status='MEDIA_PARTIAL',error=repr(e),new_native_steps=0)
        atomic_json(out/'media.json',media)
        passed=result['execution_level']=='EXECUTION_VERIFIED'
        summary=dict(status='FINISHED' if passed else 'FINISHED_WITH_BLOCKED_ITEMS',
            candidate_denominator=1,model_valid=int(result['eligible']),execution_denominator=1,
            execution_verified=int(passed),new_native_and_replay_steps=budget.native_steps,
            actual_coupled=budget.observed,reserved_model_intervals=budget.integrations,
            result=result,media=media,dynamic_status='READY_TO_FREEZE_ACTUAL_TERMINAL_CONTRACT' if passed else 'BLOCKED_NO_VERIFIED_TERMINAL',
            new_network_samples=0,training_updates=0,TEST=False,long_term_goal_complete=False)
        atomic_json(out/'summary.json',summary)
        (out/'REPORT.md').write_text('# 双侧2+5完整初态静态工程运行摘要\n\n'+
            f"完整模型资格 {int(result['eligible'])}/1，实际执行/完整重放 {int(passed)}/1。\n\n"+
            f"耦合 {budget.observed}，模型预留区间 {budget.integrations}，原生含重放 {budget.native_steps}。\n\n"+
            '仅为本次APP_DEV工程单元；网络使用准确历史提案身份，0新采样/训练。独立审查与完整中文分析尚待收尾。\n',encoding='utf-8')
        return summary
