"""Finite physical TRAIN/VAL collection with verified, registered route counts."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
import traceback
from types import SimpleNamespace

import numpy as np

from v6_4.collect_dataset import collect
from v6_4.dataset import write_teacher_manifest
from v6_4.prepare_teacher import prepare
from v6_4.report_comparison import validate_actual_trial
from v6_4.run_planning import load_task_suite, run_attempt
from v6_4.run_teacher_comparison import _code_sources, _manifest, _physical_steps


def _write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def _sources():
    result = _code_sources()
    for name in ('build_teacher_cohort.py','prepare_teacher.py','collect_dataset.py','dataset.py','report_comparison.py'):
        result['v6_4/'+name] = hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest()
    return result


def _failure(error, phase):
    return {'phase':phase,'type':type(error).__name__,'message':str(error),'traceback':traceback.format_exc()}


def check_output_paths(output, tasks):
    output = Path(output).resolve()
    # Windows relative paths remain subject to MAX_PATH. Check the longest
    # declared leaf before any expensive proposal or actual execution begins.
    longest = max(len(str(output/'closed_loop'/t.task_id/'route_002'/'pcc_monitor'/t.task_id/
        'plots/minimum_clearance_comparison.png')) for t in tasks)
    if os.name=='nt' and longest>=250:
        raise ValueError('cohort output needs a shorter absolute path; longest declared leaf is '+str(longest))
    return output, longest


def _resume_attempts(task, source, *, rounds=None, geometry_rounds=None):
    identity = json.loads((source/'identity.json').read_text(encoding='utf-8'))
    if identity.get('task_sha256')!=task.sha256() or identity.get('starts')!=3:
        raise ValueError('resumed planning source changed its task or three-slot budget')
    if ((rounds is not None and identity.get('optimization_rounds')!=rounds)
            or (geometry_rounds is not None and identity.get('geometry_rounds')!=geometry_rounds)):
        raise ValueError('resumed planning candidate optimization budget differs')
    attempts=[]
    for index in range(3):
        candidate=source/f'candidate_{index:03d}'
        saved=json.loads((candidate/'planning_attempt.json').read_text(encoding='utf-8'))
        proposal_path=candidate/'proposal.json'
        from v6_4.contracts import TrajectoryProposal
        proposal=TrajectoryProposal.from_dict(json.loads(proposal_path.read_text(encoding='utf-8'))) if proposal_path.exists() else None
        embedded=TrajectoryProposal.from_dict(saved['proposal']) if saved.get('proposal') is not None else None
        if saved.get('attempt_index')!=index or (embedded.sha256() if embedded else None)!=(proposal.sha256() if proposal else None):
            raise ValueError('resumed attempt slot differs from its original proposal file')
        if proposal is not None and (proposal.task_sha256!=task.sha256() or proposal.task_id!=task.task_id or proposal.origin!='teacher' or proposal.postprocessing):
            raise ValueError('resumed candidate does not bind the same raw Teacher task')
        attempts.append(SimpleNamespace(attempt_index=index,proposal=proposal,metadata=saved['metadata']))
    return attempts


def recover_completed_trial(task, source, output, expected_proposal):
    """Copy pinned raw inputs; add an independent evaluation and derived result.

    There is no new closed-loop experiment here. The evaluator independently
    replays the already acquired torque, using the unchanged execution contract.
    """
    from v6_4.evaluate_planning import evaluate_trial
    from v6_lite.hierarchical_qp import HierarchicalQPConfig
    source=Path(source).resolve();output=Path(output).resolve()
    result=json.loads((source/'result.json').read_text(encoding='utf-8'))
    from v6_4.contracts import TrajectoryProposal
    proposal=TrajectoryProposal.from_dict(json.loads((source/'raw_proposal.json').read_text(encoding='utf-8')))
    if (result.get('method')!='teacher' or result.get('task_sha256')!=task.sha256()
            or proposal.sha256()!=expected_proposal.sha256() or result.get('proposal_accepted') is not True
            or result.get('raw_proposal_passed') is not True or result.get('postprocessing') or result.get('fallback_used')):
        raise ValueError('complete resumed actual input does not bind the original raw selected proposal')
    if result.get('complete') is not False or result.get('task_success') is not False or not result.get('failure'):
        raise ValueError('retry source must preserve an original failed pipeline result')
    trace=source/'traces'/f'{task.task_id}.npz'
    with np.load(trace,allow_pickle=False) as data:
        if (data['torque'].shape!=(13500,67) or data['time'].shape!=(13500,)
                or not np.all(np.isfinite(data['torque'])) or not np.all(np.isfinite(data['time']))
                or not np.allclose(data['time'],.002*np.arange(1,13501),atol=1e-9,rtol=0.)
                or not np.array_equal(data['initial_qpos'],task.initial_qpos)
                or not np.array_equal(data['initial_qvel'],task.initial_qvel)):
            raise ValueError('resumed acquired force trace is not the full frozen initial/grid protocol')
    original={p.relative_to(source).as_posix():{'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size}
        for p in source.rglob('*') if p.is_file()}
    output.mkdir(parents=True,exist_ok=False)
    for name,record in original.items():
        if name.startswith('evaluation/'):
            continue
        destination=output/('prior_failed_result.json' if name=='result.json' else name)
        destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(source/name,destination)
        if hashlib.sha256(destination.read_bytes()).hexdigest()!=record['sha256']:
            raise ValueError('resumed acquired input copy differs from original raw SHA')
    identity=json.loads((output/'planning_identity.json').read_text(encoding='utf-8'))
    reference=output/'selected_reference.npz'
    evaluation=evaluate_trial(task,output/'traces'/trace.name,reference,scenario_result=None,
        qp_config=HierarchicalQPConfig(**identity['qp_config']),output_dir=output/'evaluation')
    derived=dict(result,complete=evaluation['complete'],task_success=evaluation['task_success'],
        status='RECOVERED_TASK_COMPLETED' if evaluation['task_success'] else 'RECOVERED_TASK_FAILED',
        trace_path=(output/'traces'/trace.name).as_posix(),reference_path=reference.as_posix(),evaluation=evaluation,
        prior_pipeline_failure=result.get('failure'),new_closed_loop_physics_steps=0)
    _write(output/'result.json',derived)
    _write(output/'resume_lineage.json',{'original_trial':source.as_posix(),'original_files':original,
        'original_failed_result_sha256':original['result.json']['sha256'],
        'task_sha256':task.sha256(),'proposal_sha256':proposal.sha256(),
        'new_closed_loop_physics_steps':0,'independent_saved_torque_replay_steps':13500,
        'source_failure_preserved':True,'copied_acquired_inputs_byte_exact':True})
    return derived


def register_verified_route(task, trial, output, seen_controls, seen_traces):
    """Only independently valid actual raw Teacher runs become route labels."""
    inputs = []
    verified = validate_actual_trial(task,trial,'teacher',inputs=inputs)
    if not (verified['task_success'] and verified['complete'] and verified['evidence_valid']):
        return None, {'registered_successful_teacher':False,'status':verified['result']['status'],
            'task_success':False,'actual_physics_steps':verified['actual_physics_steps']}
    controls_hash = hashlib.sha256(np.ascontiguousarray(verified['proposal'].free_controls,dtype='<f8').tobytes()).hexdigest()
    trace_hash = verified['trace_sha256']
    if controls_hash in seen_controls or trace_hash in seen_traces:
        return None, {'registered_successful_teacher':False,'status':'DUPLICATE_ROUTE_NOT_COUNTED',
            'task_success':True,'controls_sha256':controls_hash,'trace_sha256':trace_hash,
            'actual_physics_steps':verified['actual_physics_steps']}
    registration = collect([trial],output)
    manifest = json.loads(Path(registration['manifest']).read_text(encoding='utf-8'))
    if registration['successful_teacher_count'] != 1 or len(manifest['samples']) != 1:
        raise ValueError('verified Teacher trial was not registered as exactly one successful sample')
    sample = manifest['samples'][0]
    if sample['task_id'] != task.task_id or sample['split'] != task.split:
        raise ValueError('registered Teacher sample changed its immutable task/split')
    seen_controls.add(controls_hash)
    seen_traces.add(trace_hash)
    return sample, {'registered_successful_teacher':True,'status':verified['result']['status'],
        'task_success':True,'controls_sha256':controls_hash,'trace_sha256':trace_hash,
        'actual_physics_steps':verified['actual_physics_steps'],
        'registration_manifest':registration['manifest'],
        'registration_manifest_sha256':hashlib.sha256(Path(registration['manifest']).read_bytes()).hexdigest(),
        'verified_input_files':[{'path':p.resolve().as_posix(),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
            for p in sorted(set(inputs))]}


def build(tasks_path, output, *, reuse_first=True, rounds=2, geometry_rounds=8,
          use_training_priors=False, resume_from=None, resume_receipt=None):
    if (isinstance(rounds,bool) or not isinstance(rounds,int) or rounds<1
            or isinstance(geometry_rounds,bool) or not isinstance(geometry_rounds,int)
            or not 0<=geometry_rounds<=32):
        raise ValueError('finite positive anchor rounds and 0..32 geometry rounds required')
    tasks = load_task_suite(tasks_path)
    selected = [t for t in tasks if t.split in ('train','val')]
    if not selected or len({t.task_id for t in selected}) != len(selected):
        raise ValueError('nonempty unique TRAIN/VAL tasks required')
    output,longest_path=check_output_paths(output,selected)
    output.mkdir(parents=True,exist_ok=False)
    source_before = _sources()
    suite_hash = hashlib.sha256(Path(tasks_path).read_bytes()).hexdigest()
    resume_inventory=None
    if resume_from is not None:
        resume_from=Path(resume_from).resolve()
        receipt=json.loads(Path(resume_receipt).read_text(encoding='utf-8'))
        if Path(receipt['previous_run']).resolve()!=resume_from:
            raise ValueError('resume receipt belongs to another source cohort')
        resume_inventory=receipt['original_files']
        actual_inventory={p.relative_to(resume_from).as_posix():{'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size}
            for p in resume_from.rglob('*') if p.is_file()}
        if actual_inventory!=resume_inventory:
            raise ValueError('interrupted cohort input file set/raw identity changed')
        prior_plan=json.loads((resume_from/'plan.json').read_text(encoding='utf-8'))
        if (prior_plan['task_suite_sha256']!=suite_hash or prior_plan['optimizer_rounds']!=rounds
                or prior_plan['geometry_rounds']!=geometry_rounds or prior_plan['geometry_line_search_steps']!=4):
            raise ValueError('resume must preserve original task/source optimization budgets')
        permitted_changes={'v6_4/build_teacher_cohort.py'}
        if any(source_before.get(name)!=value for name,value in prior_plan['source_sha256'].items()
                if name not in permitted_changes):
            raise ValueError('resumption changed an executed planner/controller/evaluator source')
    declared, train_number = [], 0
    for task in selected:
        prior_scene = min(train_number//2,2) if task.split=='train' else 3
        if task.split=='train':
            train_number += 1
        declaration = {'task_id':task.task_id,'task_sha256':task.sha256(),'prior_scene':prior_scene,
            'planning_starts':3,'physical_attempt_cap':3,'successful_routes_target':2}
        if use_training_priors and task.split=='train':
            name = {'end_effector_detour':'ee','mid_arm_detour':'mid','multiple_routes':'multiple'}[task.family]
            trial = Path('v6_4/output/teacher_bootstrap')/f"fixed_{name}_train_{task.task_id.rsplit('_',1)[1]}_01"
            declaration.update(training_prior_trial=str(trial.resolve()),training_prior_task_sha256=task.sha256())
        declared.append(declaration)
    _write(output/'plan.json',{'schema':'v6_4_teacher_cohort_plan_v1','tasks_path':str(Path(tasks_path).resolve()),
        'task_suite_sha256':suite_hash,'source_sha256':source_before,'tasks':declared,
        'test_task_ids_excluded':[t.task_id for t in tasks if t.split=='test'],
        'test_prior_scene_reserved':4,'optimizer_rounds':rounds,'geometry_rounds':geometry_rounds,
        'geometry_line_search_steps':4,'geometry_stride':10,'use_same_task_training_nominal_priors':use_training_priors,
        'additional_sampling_after_failure':False,'all_attempts_retained':True,
        'output_absolute':output.as_posix(),'longest_declared_leaf_characters':longest_path,
        'resume_from':resume_from.as_posix() if resume_from else None,
        'resume_receipt':str(Path(resume_receipt).resolve()) if resume_receipt else None,
        'resume_receipt_sha256':hashlib.sha256(Path(resume_receipt).read_bytes()).hexdigest() if resume_receipt else None,
        'resume_source_changes_allowed':['v6_4/build_teacher_cohort.py'] if resume_from else [],
        'distinct_route_definition':'unique finite raw control arrays and unique actual trace SHA; no topology claim',
        'target_does_not_establish_data_sufficiency':True})
    samples, records, all_attempts = [], [], []
    started = time.perf_counter()
    first_prefixes = {'v6_4_end_effector_detour_train_000':'ee','v6_4_mid_arm_detour_train_000':'mid',
        'v6_4_multiple_routes_train_000':'multiple'}
    for index,(task,declaration) in enumerate(zip(selected,declared)):
        print(f'[teacher-cohort] {task.task_id}',flush=True)
        task_records, seen_controls, seen_traces = [], set(), set()
        success = 0
        if reuse_first and task.task_id in first_prefixes:
            prefix = first_prefixes[task.task_id]
            for route in (0,1):
                trial = Path(f'v6_4/output/teacher_closed_loop/{prefix}_train_000_route{route}_01')
                row = {'task_id':task.task_id,'candidate_index':route,'trial_path':str(trial.resolve()),
                    'reused_actual_trial':True,'new_physics_steps':0}
                try:
                    sample, proof = register_verified_route(task,trial,output/'registrations'/task.task_id/f'reused_{route}',seen_controls,seen_traces)
                    row.update(proof)
                    if sample is not None:
                        samples.append(sample)
                        success += 1
                except Exception as error:
                    row.update(status='REUSE_EVIDENCE_ERROR',registered_successful_teacher=False,
                        task_success=False,failure=_failure(error,'reuse_evidence'))
                task_records.append(row)
        if success < 2:
            prior = Path('v6_lite/output/runs/research_acceptance_01/simulation/traces')/f"v6_lite_scenario_{declaration['prior_scene']:02d}.npz"
            if declaration.get('training_prior_trial'):
                prior = Path(declaration['training_prior_trial'])/'traces'/f'{task.task_id}.npz'
            try:
                saved_candidates=resume_from/'proposals'/task.task_id if resume_from else None
                if saved_candidates is not None and all((saved_candidates/f'candidate_{k:03d}/planning_attempt.json').is_file() for k in range(3)):
                    attempts=_resume_attempts(task,saved_candidates,rounds=rounds,geometry_rounds=geometry_rounds)
                    shutil.copytree(saved_candidates,output/'proposals'/task.task_id)
                else:
                    attempts = prepare(task,prior,output/'proposals'/task.task_id,seed=6400+index*31,
                        starts=3,rounds=rounds,geometry_rounds=geometry_rounds)
                if len(attempts)!=3 or [a.attempt_index for a in attempts]!=list(range(3)):
                    raise ValueError('planner must preserve exactly three ordered candidate slots')
            except Exception as error:
                failure = _failure(error,'prepare')
                _write(output/'failures'/task.task_id/'prepare_failure.json',failure)
                attempts = []
                task_records.extend({'task_id':task.task_id,'candidate_index':k,'status':'PLANNING_PIPELINE_FAILURE',
                    'registered_successful_teacher':False,'task_success':False,'new_physics_steps':0,'failure':failure} for k in range(3))
            for attempt in attempts:
                previous_trial=resume_from/'closed_loop'/task.task_id/f'route_{attempt.attempt_index:03d}' if resume_from else None
                previous_trace=previous_trial/'traces'/f'{task.task_id}.npz' if previous_trial else None
                recover=False
                if previous_trace is not None and previous_trace.is_file():
                    try:
                        with np.load(previous_trace,allow_pickle=False) as old_trace:
                            recover=len(old_trace['torque'])==13500
                    except Exception as error:
                        _write(output/'failures'/task.task_id/f'resume_{attempt.attempt_index}_trace_failure.json',_failure(error,'resume_trace'))
                        # An unreadable acquired input cannot justify running
                        # the same declared proposal again under a new trial.
                        task_records.append({'task_id':task.task_id,'candidate_index':attempt.attempt_index,
                            'status':'RESUME_TRACE_EVIDENCE_ERROR','registered_successful_teacher':False,
                            'task_success':False,'new_physics_steps':0})
                        continue
                row = {'task_id':task.task_id,'candidate_index':attempt.attempt_index,
                    'registered_successful_teacher':False,'task_success':False,'reused_actual_trial':False,'new_physics_steps':0}
                if attempt.proposal is None:
                    row.update(status='PLANNING_FAILED',failure=attempt.metadata['failure'])
                elif success>=2 and not recover:
                    row['status'] = 'PRESERVED_UNSELECTED_AFTER_PREDECLARED_SUCCESS_TARGET'
                else:
                    trial = output/'closed_loop'/task.task_id/f'route_{attempt.attempt_index:03d}'
                    row['trial_path'] = str(trial.resolve())
                    row['recovered_complete_prior_trial']=recover
                    try:
                        if recover:
                            recover_completed_trial(task,previous_trial,trial,attempt.proposal)
                        else:
                            run_attempt(task,trial,method='teacher',proposal=attempt.proposal,
                                attribution={'cohort':'training_teacher','candidate_index':attempt.attempt_index,
                                    'teacher_planning_cost_s':attempt.metadata['planning_wall_time_s'],
                                    'bootstrap_identity':attempt.metadata.get('bootstrap_source',{})})
                        sample, proof = register_verified_route(task,trial,
                            output/'registrations'/task.task_id/f'new_{attempt.attempt_index}',seen_controls,seen_traces)
                        row.update(proof)
                        if sample is not None:
                            samples.append(sample)
                            success += 1
                    except Exception as error:
                        row.update(status='ACTUAL_OR_REGISTRATION_FAILURE',failure=_failure(error,'actual_or_registration'))
                        _write(output/'failures'/task.task_id/f'actual_{attempt.attempt_index}_failure.json',row['failure'])
                    try:
                        row['new_physics_steps'] = 0 if recover else _physical_steps(trial)
                    except Exception as error:
                        row.update(status='ACTUAL_TRACE_EVIDENCE_ERROR',failure=_failure(error,'actual_step_count'))
                        _write(output/'failures'/task.task_id/f'actual_{attempt.attempt_index}_trace_failure.json',row['failure'])
                task_records.append(row)
                print(json.dumps({'task_id':task.task_id,'candidate_index':attempt.attempt_index,
                    'status':row['status'],'registered_success_count':success}),flush=True)
        records.append({'task_id':task.task_id,'task_sha256':task.sha256(),'successful_routes':success,
            'registered_unique_successful_routes':success,'attempts':task_records,
            'prior_real_smoke_routes_reused':any(r.get('reused_actual_trial') and r.get('registered_successful_teacher') for r in task_records)})
        all_attempts.extend(task_records)
        _write(output/f'progress_{index:03d}.json',records)
    dataset_path = output/'dataset/manifest.json'
    write_teacher_manifest(samples,dataset_path)
    _write(output/'dataset/all_attempts.json',{'attempts':all_attempts,'failure_evidence_is_retained':True,
        'failed_attempts_are_training_labels':False})
    registered = {t.task_id:sum(s['task_id']==t.task_id for s in samples) for t in selected}
    if any(r['successful_routes']!=registered[r['task_id']] for r in records):
        raise ValueError('cohort route count differs from registered valid samples')
    source_after = _sources()
    summary = {'schema':'v6_4_teacher_cohort_result_v1','tasks':records,
        'dataset':{'attempt_count':len(all_attempts),'successful_teacher_count':len(samples),'manifest':dataset_path.resolve().as_posix()},
        'elapsed_wall_s':time.perf_counter()-started,'task_group_count':len(selected),
        'all_tasks_attempted':len(records)==len(selected),'all_success_targets_met':all(n>=2 for n in registered.values()),
        'registered_successes_by_task':registered,'new_physics_steps':sum(r.get('new_physics_steps',0) for r in all_attempts),
        'reused_successful_samples':sum(bool(r.get('reused_actual_trial') and r.get('registered_successful_teacher')) for r in all_attempts),
        'source_before':source_before,'source_after':source_after,'all_sources_unchanged':source_before==source_after,
        'task_suite_unchanged':hashlib.sha256(Path(tasks_path).read_bytes()).hexdigest()==suite_hash,
        'recovered_complete_prior_trials':sum(r.get('recovered_complete_prior_trial',False) for r in all_attempts),
        'resume_source_files_unchanged':({p.relative_to(resume_from).as_posix():{'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size}
            for p in resume_from.rglob('*') if p.is_file()}==resume_inventory) if resume_from else None,
        'data_sufficiency_or_generalization_established':False,'wall_20ms_is_acceptance_gate':False}
    _write(output/'summary.json',summary)
    _write(output/'artifact_manifest.json',_manifest(output))
    print(json.dumps(summary['dataset']),flush=True)
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tasks',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--no-reuse-first',action='store_true')
    parser.add_argument('--rounds',type=int,default=2)
    parser.add_argument('--geometry-rounds',type=int,default=8)
    parser.add_argument('--use-training-priors',action='store_true')
    parser.add_argument('--resume-from',type=Path)
    parser.add_argument('--resume-receipt',type=Path)
    args = parser.parse_args()
    build(args.tasks,args.output,reuse_first=not args.no_reuse_first,rounds=args.rounds,
        geometry_rounds=args.geometry_rounds,use_training_priors=args.use_training_priors,
        resume_from=args.resume_from,resume_receipt=args.resume_receipt)
