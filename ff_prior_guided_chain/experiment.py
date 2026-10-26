"""One new chain with persistent actual/replay states and explicit stage releases."""
import time,traceback,zipfile,copy
import numpy as np
from ff_prior.storage import read_json,file_digest,digest,RunLock
from ff_prior_pilot_train.storage import atomic_json,atomic_npz
from ff_prior.initial_state import InitialState
from ff_prior.contracts import AcceptanceConfig
from ff_prior_dynamic_reference.execution import integration
from .protocol import ROOT,PACKAGE,OUT,P
from .inputs import identity,runtime
from .budget import Budget,Instrumentation
from .representation import Representation
from .predictor import Predictor
from .targets import Targets
from .guidance import Guidance
from .sampler import generate
from .qualification import qualify
from .static_audit import audit_execution
from .execution import StaticBudget,dynamic


def release(phase,directory,budget,progress):
    budget.enter('review')
    report=directory/'execution'/('report.json' if phase=='A' else 'result.json')
    manifest={p.relative_to(directory).as_posix():dict(sha256=file_digest(p),bytes=p.stat().st_size)
        for p in sorted(directory.rglob('*')) if p.is_file() and p.name not in ('review_request.json','release.json')}
    request=dict(phase=phase,report_sha256=file_digest(report),core_manifest=manifest,
        required='fresh result-to-claim review, saved-result verification and full-state seam identity; no extra physics')
    atomic_json(directory/'review_request.json',request)
    progress('stage_review_required',phase=phase,request=str(directory/'review_request.json'))
    start=time.time()
    while not (directory/'release.json').exists():
        budget.check()
        if time.time()-start>5400:raise RuntimeError('STAGE_REVIEW_RESERVE_EXHAUSTED')
        time.sleep(2)
    decision=read_json(directory/'release.json')
    if decision.get('report_sha256')!=request['report_sha256'] or decision.get('passed') is not True:
        raise RuntimeError('STAGE_REVIEW_NOT_RELEASED')
    review=ROOT/decision['review_file']
    if not review.is_file() or file_digest(review)!=decision['review_sha256']:raise RuntimeError('REVIEW_IDENTITY_MISMATCH')
    for f,v in manifest.items():
        if file_digest(directory/f)!=v['sha256']:raise RuntimeError('STAGE_EVIDENCE_CHANGED')


def run(out=OUT,progress=lambda *a,**k:None,started=None):
    budget=Budget(out);ident=identity();summary=dict(status='FINISHED_WITH_BLOCKED_ITEMS',phases={},
        training=False,expansion=False,TEST=False,advantage_comparison=False,long_term_goal_complete=False)
    with RunLock(out):
        assert not (out/'run.json').exists(),'UNKNOWN_WORK'
        atomic_json(out/'run.json',dict(identity=ident,started=time.time()))
        with zipfile.ZipFile(out/'executed_source.zip','w',zipfile.ZIP_DEFLATED) as z:
            for f in ident['source_hashes']:z.write(ROOT/f,f)
        with Instrumentation(budget):
            robot,root_initial,entry=runtime();assert root_initial.model_hash==ident['actual_model_hash']
            robot._guided_actual_data=root_initial.reset(robot)
            robot._guided_replay_data=root_initial.reset(robot)
            robot._guided_actual_initializations=1
            robot._guided_replay_initializations=1
            root_ids=(id(robot._guided_actual_data),id(robot._guided_replay_data))
            assert np.array_equal(integration(robot,robot._guided_actual_data),root_initial.values)
            terminal=dict(integration=root_initial.values.copy(),reference_q=np.asarray(entry['task']['condition']['initial_features']),
                reference_qd=np.zeros(17),reference_qdd=np.zeros(17))
            atomic_npz(out/'original_initial.npz',values=root_initial.values)
            atomic_json(out/'initialization.json',dict(actual_initializations=1,replay_initializations=1,
                original_836_exact=True,actual_object=root_ids[0],replay_object=root_ids[1],integrator='RK4',dt=.002))
            for phase in ('A','B','C'):
                directory=out/phase;directory.mkdir()
                initial=InitialState(integration(robot,robot._guided_actual_data),root_initial.model_hash)
                assert np.array_equal(initial.values,terminal['integration'])
                assert np.array_equal(integration(robot,robot._guided_replay_data),initial.values)
                atomic_npz(directory/'phase_initial.npz',values=initial.values,reference_q=terminal['reference_q'],
                    reference_qd=terminal['reference_qd'],reference_qdd=terminal['reference_qdd'])
                contract=read_json(ROOT/('ff_prior_balanced_brake/evidence/contract.json' if phase!='C' else 'ff_prior_satellite_origin/evidence/contract.json'))
                contract=copy.deepcopy(contract);contract.update(scene_model_hash=initial.model_hash,absolute_start_time=float(robot._guided_actual_data.time),
                    new_chain='guided_diffusion_chain_001',new_neural_sample=True,network_role='frozen full trajectory with DDIM full-machine guidance',
                    original_source_contract_is_reference_only=True)
                if phase=='C':
                    hold=np.eye(4);hold[:3,3]=contract['waypoints']['waypoint_points_m'][-1]
                    hold[:3,:3]=contract['waypoints']['target_rotation_world'];contract['hold_continuum_pose']=hold
                atomic_json(directory/'contract.json',contract)
                try:
                    budget.enter('planner')
                    progress('new_phase_planning',phase=phase,absolute_time=float(robot._guided_actual_data.time))
                    predictor=Predictor(robot,initial,phase,budget)
                    rep=Representation(robot,phase,predictor.q0,predictor.v0,terminal['reference_qdd'])
                    targets=Targets(robot,initial,phase,contract,entry['task']['condition']['goals'])
                    guide=Guidance(rep,predictor,targets,directory/'planning',progress)
                    reference,final=generate(rep,guide,entry['task'],progress)
                    qualification=qualify(rep,guide,reference,final,entry['task'],directory/'qualification.json')
                    if qualification['level']!='MODEL_VALID':
                        summary['phases'][phase]=dict(model=qualification,execution='BLOCKED_MODEL_REJECTED');break
                    budget.enter('native')
                    if phase=='A':
                        execution=directory/'execution';execution.mkdir()
                        native=StaticBudget(execution,budget)
                        result=audit_execution(entry['task'],AcceptanceConfig(**entry['task']['contract']['acceptance']),robot,initial,reference,
                            execution,budget.deadline,native_budget=native)
                        arrays=dict(np.load(execution/'native_trace.npz'))
                        terminal=dict(integration=integration(robot,robot._guided_actual_data),reference_q=reference.evaluate(27.5),
                            reference_qd=reference.evaluate(27.5,1),reference_qdd=reference.evaluate(27.5,2))
                        passed=result['level']=='EXECUTION_VERIFIED'
                    else:
                        result,terminal=dynamic(robot,terminal,contract,phase,reference,budget,directory/'execution',progress)
                        passed=result['status']=='EXECUTION_VERIFIED'
                    assert (id(robot._guided_actual_data),id(robot._guided_replay_data))==root_ids
                    atomic_npz(directory/'terminal.npz',**terminal)
                    summary['phases'][phase]=dict(model=qualification,execution=result,passed=passed)
                    atomic_json(out/'summary.json',summary);budget.save()
                    if not passed:break
                    release(phase,directory,budget,progress)
                except Exception as error:
                    atomic_json(directory/'failure.json',dict(reason=repr(error),traceback=traceback.format_exc()))
                    summary['phases'][phase]=dict(status='FAILED',reason=repr(error),actual_time=float(robot._guided_actual_data.time));break
            if all(summary['phases'].get(p,{}).get('passed') for p in ('A','B','C')):summary['status']='FINISHED'
            for phase in ('A','B','C'):
                if phase not in summary['phases']:summary['phases'][phase]=dict(status='DEPENDENCY_BLOCKED',native=0,neural=0)
            budget.enter('media');budget.save();summary['budget']=read_json(out/'budget.json');summary['calls']=robot.calls
            from .media import deliver
            try:summary['media']=deliver(out,robot,entry,progress,budget)
            except Exception as error:summary['media']=dict(status='PARTIAL',error=repr(error))
            assert identity()==ident,'FROZEN_INPUT_OR_SOURCE_CHANGED'
            budget.enter(None);budget.save();summary['budget']=read_json(out/'budget.json')
            atomic_json(out/'summary.json',summary)
            (out/'REPORT.md').write_text('# 新冻结 Diffusion 连续链\n\n'+str(summary)+'\n\nAPP_DEV；模型、执行、重放分别裁决；独立收尾审查待完成。\n',encoding='utf-8')
    return summary
