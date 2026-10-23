import subprocess
import time
import zipfile
import numpy as np
from ff_prior.storage import read_json,digest,RunLock
from ff_prior_pilot_train.storage import atomic_json,atomic_npz,Ledger
from ff_prior.pipeline import audit,acceptance
from ff_prior.contracts import assert_frozen,assert_task_compatible
from ff_prior_origin_learn_bounded.runtime import TaskOriginSceneCache
from ff_prior.spline import Reference
from ff_prior.proposal_generator import quick_gates
from ff_prior_contact_steer.tree import Node,root_node,Piecewise
from ff_prior_static_goal_seed.validation import validate
from ff_prior_static_goal_seed.physical_audit import audit_execution
from .protocol import ROOT,PACKAGE,OUT,OLD,BASE,SOURCE,P
from .inputs import identity
from .budget import Budget,SliceBudget,BudgetExhausted
from .motion import Motion
from .solver import Evaluator,optimize


def assert_resume_safe(out):
    if (out/'ledger.json').exists():raise ValueError('UNKNOWN_OR_INTERRUPTED_WORK_NOT_REPLAYABLE')


def prefix(spec,robot,initial,collision,proof,directory):
    with np.load(ROOT/spec['prefix_file']) as a:
        nodes=[Node(**{k:a[k][i].copy() if np.ndim(a[k][i]) else a[k][i].item()
            for k in ('q','state','qpos','qvel','elapsed','parent','duration','ee','edge_id')}) for i in range(len(a['q']))]
    original=root_node(robot,initial);old=read_json(OLD/'scenes'/spec['task_id']/'initial_audit.json')
    checks=dict(full_original_state_exact=np.array_equal(original.state,nodes[0].state),
        planner_initial_exact=np.array_equal(original.q,nodes[0].q),
        scene_model_exact=proof['scene_transfer']['scene_robot_hash']==old['scene_transfer']['scene_robot_hash'],
        initial_id_exact=initial.state_id==old['scene_task_origin_state_id'],
        collision_policy_exact=collision.policy_hash==old['initial_geometry']['pair_policy_hash'],
        full_states=all(n.state.shape==(836,) for n in nodes),
        original_time=nodes[0].elapsed==0.,time_monotone=all(a.elapsed<b.elapsed for a,b in zip(nodes[:-1],nodes[1:])),
        final_time_within27=nodes[-1].elapsed<=27.,parent_chain=all(n.parent==i-1 for i,n in enumerate(nodes)))
    atomic_json(directory/'prefix_identity.json',dict(checks=checks,passed=all(checks.values()),source=spec,
        previous_qualification='TREE_SCREEN_ONLY',old_propagation_repeated=False,
        use='initial guess q/time only; every new predictor and executor starts at original full initial state'))
    if not all(checks.values()):raise ValueError('PREFIX_REUSE_IDENTITY_MISMATCH')
    return nodes


def unit(directory,spec,entry,robot,initial,collision,proof,profile,budget,progress):
    task=entry['task'];nodes=prefix(spec,robot,initial,collision,proof,directory)
    progress('motion_constrained_initial_fit',unit='u00')
    motion=Motion(robot,task['condition']['initial_features'])
    times=np.linspace(0,27.,P['fit_samples']);desired=Piecewise(nodes,27.).evaluate(times)
    z,fit=motion.fit(times,desired)
    atomic_npz(directory/'fit.npz',Z=z,times=times,desired=desired,
        fitted=Reference(z,motion.q0,27.).evaluate(times),A=motion.A,lo=motion.lo,hi=motion.hi,
        variable_lower=motion.lower,variable_upper=motion.upper)
    atomic_json(directory/'fit.json',dict(fit,quick_gates=quick_gates(robot,Reference(z,motion.q0,27.)),
        prefix_final_state_not_injected=True,all_323_variables_free=True))
    budget.check()
    quota=SliceBudget(budget,P['restore_coupled_cap'],P['restore_interval_cap'],
        min(time.time()+P['restore_wall_s'],budget.deadline-P['execution_reserve_s']-P['visualization_wall_s']))
    restored=optimize(directory/'restore',z,motion,Evaluator(task,robot,initial,collision,quota),progress)
    progress('original_complete_model_certification',unit='u00')
    budget.ensure_output(512*1024**2)
    quota=SliceBudget(budget,40,200000,budget.deadline-P['execution_reserve_s']-P['visualization_wall_s'])
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
            directory/'execution',budget.deadline-P['visualization_wall_s'])
        result.update(execution_level=execution['level'],reason='NATIVE_EXECUTION_ATTEMPTED')
    assert_frozen(task);atomic_json(directory/'result.json',result)
    return result


def run(out=OUT,progress=lambda *a,**k:None):
    assert_resume_safe(out);ident=identity()
    with RunLock(out):
        assert read_json(PACKAGE/'evidence/software_tests.json')['source_hashes']==ident['source_hashes']
        atomic_json(out/'run.json',dict(identity=ident,producer_head=subprocess.check_output(
            ['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),TEST=False))
        with zipfile.ZipFile(out/'executed_source.zip','w',zipfile.ZIP_DEFLATED) as z:
            for p in ident['source_hashes']:z.write(ROOT/p,p)
        deadline=time.time()+P['wall_s'];config=read_json(BASE/'run.json')['config']
        physical,canonical,initial,_=audit(config);old=read_json(SOURCE/'physical_contract.json')
        assert physical['acceptance']==old['acceptance'] and digest(physical['identity'])==digest(old['identity'])
        assert physical['initial_state_id']==old['initial_state_id']
        atomic_json(out/'physical_contract.json',physical);profile=acceptance(config)
        cache=TaskOriginSceneCache(canonical,initial,profile,max_entries=1,max_scene_models=1)
        ledger=Ledger(out,{'identity_hash':digest(ident)});budget=Budget(out,deadline);budget.begin_unit()
        spec=ident['selection']['units'][0];entry=ident['selection']['entries'][0];directory=out/'units/u00'
        rr,ii,cc,proof=cache.get(entry);assert_task_compatible(entry['task'],rr,ii,cc,profile)
        atomic_json(out/'scenes'/spec['task_id']/'initial_audit.json',proof)
        atomic_json(out/'scenes'/spec['task_id']/'limits.json',dict(lower=rr.lower,upper=rr.upper))
        ledger.begin('u00','full27s_323_restoration',P['wall_s'],1);before=rr.calls.copy()
        try:
            result=unit(directory,spec,entry,rr,ii,cc,proof,profile,budget,progress)
        except (BudgetExhausted,RuntimeError,ValueError) as error:
            allowed=isinstance(error,BudgetExhausted) or str(error) in (
                'BUDGET_EXHAUSTED','SPARSE_DENSE_NUMERICAL_DISAGREEMENT','PREFIX_REUSE_IDENTITY_MISMATCH','MODEL_NUMERICAL_GATE_FAILED')
            if not allowed:raise
            result=dict(status='BLOCKED',reason=str(error),unit=spec,candidate_denominator=1,
                execution_denominator=1,eligible=False,qualified=0,execution_level='NOT_COMPLETED')
            atomic_json(directory/'result.json',result)
        finally:
            delta={k:rr.calls[k]-before[k] for k in before};budget.observe(delta)
            atomic_json(directory/'calls.json',delta);budget.save()
        ledger.finish(result,delta);atomic_json(out/'runtime_cache.json',cache.stats())
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
        (out/'REPORT.md').write_text('# 整条27秒323约束恢复运行摘要\n\n'+
            f"完整模型资格 {int(result['eligible'])}/1，实际执行/完整重放 {int(passed)}/1。\n\n"+
            f"耦合 {budget.observed}，模型预留区间 {budget.integrations}，原生含重放 {budget.native_steps}。\n\n"+
            '仅为本次APP_DEV工程单元；网络使用准确历史提案身份，0新采样/训练。独立审查与完整中文分析尚待收尾。\n',encoding='utf-8')
        return summary
