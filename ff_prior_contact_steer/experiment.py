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
from ff_prior_static_fullchain_v2.solver import Evaluator,optimize
from ff_prior_static_goal_seed.validation import validate
from ff_prior_static_goal_seed.physical_audit import audit_execution
from .protocol import ROOT,PACKAGE,OUT,OLD,BASE,SOURCE,P
from .inputs import identity
from .budget import Budget,SliceBudget,BudgetExhausted
from .tree import Node,root_node,fit323
from .contact import search


def assert_resume_safe(out):
    if (out/'ledger.json').exists():
        raise ValueError('UNKNOWN_OR_INTERRUPTED_WORK_NOT_REPLAYABLE')


def prefix_nodes(spec,robot,initial,collision,proof,directory):
    """Verify exact reuse identity. No old path repropagation or recertification."""
    with np.load(ROOT/spec['prefix_file']) as a:
        nodes=[Node(**{k:a[k][i].copy() if np.ndim(a[k][i]) else a[k][i].item()
            for k in ('q','state','qpos','qvel','elapsed','parent','duration','ee','edge_id')}) for i in range(len(a['q']))]
    root=root_node(robot,initial)
    old=read_json(OLD/'scenes'/spec['task_id']/'initial_audit.json')
    checks=dict(root_state_exact=np.array_equal(root.state,nodes[0].state),
        root_q_exact=np.array_equal(root.q,nodes[0].q),
        root_pose_exact=np.array_equal(root.qpos,nodes[0].qpos),
        same_scene=proof['scene_transfer']['scene_robot_hash']==old['scene_transfer']['scene_robot_hash'],
        same_initial_state=initial.state_id==old['scene_task_origin_state_id'],
        same_policy=collision.policy_hash==old['initial_geometry']['pair_policy_hash'],
        original_zero_time=nodes[0].elapsed==0.,
        full_state_size=all(n.state.shape==(836,) for n in nodes),
        original_parent_chain=all(n.parent==i-1 for i,n in enumerate(nodes)))
    atomic_json(directory/'prefix_identity.json',dict(checks=checks,passed=all(checks.values()),
        source=spec,qualification='HISTORICAL_MODEL_SCREEN_PREFIX',old_physics_repeated=False))
    if not all(checks.values()):raise ValueError('PREFIX_REUSE_IDENTITY_MISMATCH')
    return nodes


def unit(directory,spec,entry,robot,initial,collision,proof,profile,budget,ident,progress):
    task=entry['task'];prefix=prefix_nodes(spec,robot,initial,collision,proof,directory)
    context=dict(task_hash=task['task_hash'],scene_model=proof['scene_transfer']['scene_robot_hash'],
        initial_state_id=initial.state_id,collision_policy=collision.policy_hash,
        source_hashes=digest(ident['source_hashes']),protocol_hash=digest(P),
        edge_grids=P['edge_grids'],numeric_backend='mujoco3.3.2_f64_midpoint_rest_to_rest')
    result=dict(status='DONE',unit=spec,candidate_denominator=1,execution_denominator=1,
        eligible=False,qualified=0,execution_level='NOT_EXECUTED',training_updates=0,TRAIN_added=0,
        new_network_samples=0,historical_neural_prefixes_reused=1)
    quota=SliceBudget(budget,P['search_coupled_cap'],P['search_integration_cap'],
        min(time.time()+P['search_wall_s'],budget.unit_deadline-P['repair_wall_s']-P['execution_reserve_s']))
    path,tree=search(directory/'tree',prefix,task,robot,collision,quota,context,progress)
    result['tree']=tree;result['reason']='NO_COMPLETE_ROUTE'
    validation=None;vd=None
    if path is not None:
        z,fit,times,desired,fitted=fit323(path,robot,27.)
        gates=quick_gates(robot,Reference(z,task['condition']['initial_features'],27.))
        atomic_npz(directory/'fit.npz',Z=z,times=times,desired=desired,fitted=fitted)
        atomic_json(directory/'fit.json',dict(fit,quick_gates=gates,
            historical_prefix_nodes=len(prefix),path_nodes=len(path),qualification='UNQUALIFIED_FIT'))
        result['reason']='FIT_MOTION_REJECTED'
        if all(gates.values()):
            quota=SliceBudget(budget,40,P['validation_integration_cap'],budget.unit_deadline-P['execution_reserve_s'])
            vd='before_repair';validation=validate(directory/vd,z,task,robot,initial,collision,profile,quota)
            result['reason']='FULL_ROUTE_MODEL_REJECTED'
            if not validation['eligible']:
                quota=SliceBudget(budget,P['repair_coupled_cap'],P['repair_integration_cap'],
                    min(time.time()+P['repair_wall_s'],budget.unit_deadline-P['execution_reserve_s']))
                repaired=optimize(directory/'repair',z,'full323',None,
                    Evaluator(task,robot,initial,collision,quota),progress,P['repair_iterations'])
                quota=SliceBudget(budget,40,P['validation_integration_cap'],budget.unit_deadline-P['execution_reserve_s'])
                vd='validated';validation=validate(directory/vd,np.asarray(repaired['Z']),task,robot,initial,collision,profile,quota)
            result.update(eligible=bool(validation['eligible']),qualified=int(validation['eligible']),
                validation_path=vd,reference_hash=validation['reference_hash'])
    atomic_json(directory/'execution_selection.json',dict(before_execution=True,selected=result['eligible'],
        rule='single frozen repair candidate, no replacement',task_hash=task['task_hash'],
        original_initial_state_id=initial.state_id,model_parent_state_never_injected=True))
    if result['eligible']:
        budget.check();assert budget.native_steps+27500<=P['native_steps_cap']
        progress('native67_execution_and_independent836_replay',unit=spec['id'])
        execution=audit_execution(task,profile,robot,initial,
            Reference(validation['Z'],task['condition']['initial_features'],27.),
            directory/'execution',min(budget.deadline,budget.unit_deadline))
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
        ledger=Ledger(out,{'identity_hash':digest(ident)});budget=Budget(out,deadline)
        entries={e['sample_id']:e for e in ident['selection']['entries']}
        for spec in ident['selection']['units']:
            budget.begin_unit();directory=out/'units'/spec['id'];entry=entries[spec['task_id']]
            rr,ii,cc,proof=cache.get(entry);assert_task_compatible(entry['task'],rr,ii,cc,profile)
            atomic_json(out/'scenes'/spec['task_id']/'initial_audit.json',proof)
            atomic_json(out/'scenes'/spec['task_id']/'limits.json',dict(lower=rr.lower,upper=rr.upper))
            ledger.begin(spec['id'],'contact_aware_prefix_repair',P['wall_s'],3);before=rr.calls.copy()
            try:
                result=unit(directory,spec,entry,rr,ii,cc,proof,profile,budget,ident,progress)
            except (BudgetExhausted,RuntimeError,ValueError) as error:
                allowed=isinstance(error,BudgetExhausted) or str(error) in (
                    'BUDGET_EXHAUSTED','SPARSE_DENSE_NUMERICAL_DISAGREEMENT','PREFIX_REUSE_IDENTITY_MISMATCH')
                if not allowed:raise
                result=dict(status='BLOCKED',reason=str(error),unit=spec,candidate_denominator=1,
                    execution_denominator=1,eligible=False,qualified=0,execution_level='NOT_COMPLETED')
                atomic_json(directory/'result.json',result)
            delta={k:rr.calls[k]-before[k] for k in before};budget.observe(delta)
            atomic_json(directory/'calls.json',delta);budget.save();ledger.finish(result,delta)
        atomic_json(out/'runtime_cache.json',cache.stats())
        assert identity()==ident,'FROZEN_SOURCE_OR_INPUT_CHANGED'
        from .audit_results import verify,report
        progress('saved_evidence_accounting');checked=verify(out,ident)
        atomic_json(out/'independent_verification.json',checked)
        from .visuals import generate
        media={};media_deadline=min(deadline,time.time()+P['visualization_wall_s'])
        for spec in ident['selection']['units']:
            rr,ii,cc,proof=cache.get(entries[spec['task_id']])
            try:media[spec['id']]=generate(out,rr,entries[spec['task_id']],progress,media_deadline,spec['id'])
            except Exception as e:media[spec['id']]=dict(status='MEDIA_PARTIAL',error=repr(e),new_native_steps=0)
        atomic_json(out/'media.json',media)
        return report(out,ident,checked)
