"""Finite task-anchored residual study; no retries, repair, or TEST selection."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback

import numpy as np

from .residual_execution import execute_residual_attempt, sha, source_guard, write
from .residual_protocol import freeze_residual_protocol, validate_frozen_protocol
from .task_anchored_reference import TaskAnchoredResidualPlan, reference_precheck
from .task_protocol import TaskSpec

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def event(name, **values):
    print(json.dumps({'event': name, **values}, ensure_ascii=False, allow_nan=False), flush=True)


def freeze_source(output, checks_path):
    """Freeze bytes, not Git's newline-normalized notion of source identity."""
    output = Path(output).resolve()
    plan = validate_frozen_protocol(output)
    checks_path = Path(checks_path).resolve()
    checks = read(checks_path)
    if checks.get('passed') is not True:
        raise ValueError('required representation/execution/data/protocol unit checks did not pass')
    bootstrap = read(output/'bootstrap_identity.json')
    inherited = bootstrap['source_sha256']
    if {p: sha(ROOT/p) for p in inherited} != inherited:
        raise ValueError('inherited B.1 source bytes changed')
    additions = [p for p in (ROOT/'v6_4').glob('*.py') if p.relative_to(ROOT).as_posix() not in inherited]
    additions += [p for p in (ROOT/'v6_4/tests').glob('test_*residual*.py')]
    additions += [ROOT/'v6_4/tests/test_task_anchored_reference.py']
    names = sorted(set(inherited) | {p.relative_to(ROOT).as_posix() for p in additions})
    protected = {str((output/p).resolve()): digest for p, digest in plan['artifact_sha256'].items()}
    protected[str((output/'plan.json').resolve())] = sha(output/'plan.json')
    protected[str((output/'bootstrap_identity.json').resolve())] = sha(output/'bootstrap_identity.json')
    protected[str((output/'frozen_execution_config.json').resolve())] = sha(output/'frozen_execution_config.json')
    protected[str(checks_path)] = sha(checks_path)
    for record in plan['historical_inputs']:
        protected[record['path']] = record['sha256']
    identity = {'schema': 'v64_b2_frozen_source_identity_v1',
        'git_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'git_status': subprocess.check_output(['git', 'status', '--short'], cwd=ROOT, text=True),
        'historical_base_git_head': bootstrap['base_git_head'],
        'source_sha256': {p: sha(ROOT/p) for p in names}, 'protected_artifacts': protected,
        'inherited_B1_source_byte_parity': True, 'unit_checks_path': str(checks_path),
        'unit_checks_sha256': sha(checks_path), 'python': sys.version, 'python_executable': sys.executable,
        'platform': platform.platform(), 'frozen_utc': datetime.now(timezone.utc).isoformat(),
        'wall_deadline_is_research_gate': False, 'deployment': 'NOT_MET',
        'supplemental_runtime_goal': 'end-to-end dispatch validity remains a separate deployment requirement; frozen execution unchanged'}
    write(output/'source_identity.json', identity)
    source_guard(output/'source_identity.json')
    event('SOURCE_FROZEN', files=len(names), identity_sha256=sha(output/'source_identity.json'))
    return identity


def missing_evaluation(output, task, reason):
    path = Path(output)/'evaluation_not_run.json'
    if not path.exists():
        write(path, {'schema': 'v64_b2_unexecuted_evaluation_v1', 'task_id': task.task_id,
            'task_sha256': task.sha256(), 'complete': False, 'evidence_valid': False,
            'task_success': False, 'full_task_success': False, 'status': 'NOT_RUN', 'reason': reason,
            'native_geometry': {'passed': False, 'status': 'NOT_RUN'},
            'independent_interval': {'passed': False, 'status': 'NOT_RUN'},
            'reference_binding': {'passed': False, 'status': 'NOT_RUN'},
            'execution_contract': {'passed': False, 'status': 'NOT_RUN'},
            'task_requirements': {'passed': False, 'status': 'NOT_RUN'}})
    return path


def reject_slot(output, task, raw, reason, identity_path, slot_id, reuse=False):
    """An invalid candidate consumes its slot but creates no physical attempt."""
    output = Path(output)
    source_guard(identity_path)
    if output.exists():
        if not reuse or not (output/'attempt_result.json').exists():
            raise FileExistsError('unfinished or retained rejected slot')
        result = read(output/'attempt_result.json')
        if (result['slot_id'] != slot_id or result['task_sha256'] != task.sha256()
                or result['source_identity_sha256'] != sha(identity_path)
                or read(output/'raw_candidate.json') != raw):
            raise ValueError('rejected slot reuse identity differs')
        return result
    output.mkdir(parents=True, exist_ok=False)
    write(output/'task.json', task.to_dict())
    write(output/'raw_candidate.json', raw)
    evaluation_path = missing_evaluation(output, task, reason)
    result = {'schema': 'v64_b2_residual_attempt_v1', 'slot_id': slot_id, 'task_id': task.task_id,
        'task_sha256': task.sha256(), 'split': task.split, 'status': 'REFERENCE_PRECHECK_REJECTED',
        'reason': reason, 'actual_steps': 0, 'actual_runner_started': False, 'entered_actual': False,
        'full_task_success': False, 'full_27s_success': False, 'fallback_used': False,
        'source_identity_sha256': sha(identity_path), 'evaluation_path': str(evaluation_path.resolve()),
        'evaluation_sha256': sha(evaluation_path), 'evaluation': read(evaluation_path), 'deployment': 'NOT_MET'}
    write(output/'attempt_result.json', result)
    return result


def run_slot(output, task_path, plan_path, slot_id, reuse):
    task = TaskSpec.from_dict(read(task_path))
    raw = read(plan_path)
    identity = output/'source_identity.json'
    attempt_dir = output/'attempts'/slot_id
    try:
        residual = TaskAnchoredResidualPlan.from_dict(raw)
    except (ValueError, TypeError, KeyError) as error:
        result = reject_slot(attempt_dir, task, raw, str(error), identity, slot_id, reuse)
    else:
        result = execute_residual_attempt(task, residual, attempt_dir,
            qp_config_path=output/'frozen_execution_config.json', identity_path=identity,
            slot_id=slot_id, reuse_completed=reuse)
    if not result.get('evaluation_path'):
        path = missing_evaluation(attempt_dir, task, result['status'])
    else:
        path = Path(result['evaluation_path'])
    return result, path


def validate_budget(output, protocol):
    allowed = {'zero_interface'}
    allowed |= {f'teacher_{i:02d}' for i in range(24)}
    allowed |= {f'TEST_{i:02d}_{m}' for i in range(4) for m in ('E0', 'E1', 'E2')}
    records = []
    for directory in sorted((output/'attempts').glob('*')):
        if directory.name not in allowed:
            raise ValueError('undeclared actual slot: '+directory.name)
        terminal = directory/'attempt_result.json'
        if terminal.exists():
            r = read(terminal)
            records.append({'slot_id': r['slot_id'], 'status': r['status'],
                'actual_runner_started': r.get('actual_runner_started', False),
                'entered_actual': r['entered_actual'], 'actual_steps': r['actual_steps'],
                'full_task_success': r['full_task_success'], 'path': str(terminal.resolve()), 'sha256': sha(terminal)})
    started = sum(r['actual_runner_started'] for r in records)
    if started > protocol['budget']['total_actual_max'] or len(records) > 37:
        raise ValueError('fixed actual budget exceeded')
    return {'schema': 'v64_b2_budget_ledger_v1', 'maximum_actual': 37,
        'actual_runner_started': started, 'nonzero_step_actual_attempts': sum(r['entered_actual'] for r in records),
        'actual_physics_steps': sum(r['actual_steps'] for r in records), 'terminal_slots': len(records),
        'records': records, 'private_preview_replay_and_queries_separate': True,
        'replacements_or_extra_actual': 0}


def freeze_teacher_manifest(output, protocol):
    rows = []
    for i, declared in enumerate(protocol['records']):
        directory = output/'attempts'/f'teacher_{i:02d}'
        attempt = directory/'attempt_result.json'
        result = read(attempt)
        evaluation = Path(result['evaluation_path']) if result.get('evaluation_path') else directory/'evaluation_not_run.json'
        e = read(evaluation)
        binding = e.get('reference_binding', {})
        rows.append({**declared, 'plan_path': str((output/declared['plan_path']).resolve()),
            'plan_sha256': sha(output/declared['plan_path']),
            'attempt_path': str(attempt.resolve()), 'attempt_sha256': sha(attempt),
            'evaluation_path': str(evaluation.resolve()), 'evaluation_sha256': sha(evaluation),
            'full_task_success': result['full_task_success'],
            'nonzero_reference': binding.get('nonzero_reference_consumed', False),
            'consumed_reference_binding_passed': binding.get('passed', False)})
    value = {'schema': 'v64_b2_teacher_actual_manifest_v1', 'records': rows,
        'task_protocol_sha256': sha(output/'plan.json'), 'source_identity_sha256': sha(output/'source_identity.json')}
    path = output/'teacher_manifest.json'
    if path.exists():
        if read(path) != value:
            raise ValueError('retained teacher evidence identity changed')
    else:
        write(path, value)
    return path


def sample_candidates(output, protocol, device):
    from .residual_dataset import load_residual_dataset, ConditionNormalizer, retrieve_train_residual
    from .residual_diffusion import load_residual_sampler
    manifest = output/'test_candidate_manifest.json'
    if manifest.exists():
        retained = read(manifest)
        checkpoint=output/'training/model/selected.pt'
        if (retained['checkpoint_sha256']!=sha(checkpoint)
                or retained['source_identity_sha256']!=sha(output/'source_identity.json')
                or retained['dataset_sha256']!=sha(output/'dataset/manifest.json')
                or retained['training_config_sha256']!=sha(output/'training/training_config.json')
                or retained['K1_slot']!=0 or retained['K4_actual']!='NOT_RUN'
                or len(retained['records'])!=4):
            raise ValueError('retained candidate production identity differs')
        for record in retained['artifacts']:
            if sha(output/record['path']) != record['sha256']:
                raise ValueError('retained candidate changed')
        for declared,learned in zip(protocol['test_records'],retained['records']):
            if learned['task_id']!=declared['task_id'] or len(learned['E2'])!=4:
                raise ValueError('retained candidate Task/order differs')
            for slot,(seed,candidate) in enumerate(zip(declared['E2_sample_seeds'],learned['E2'])):
                raw=read(output/candidate['raw_path'])
                if (candidate['slot'],candidate['seed'],raw['slot'],raw['seed'],raw['task_id'])!=(slot,seed,slot,seed,declared['task_id']):
                    raise ValueError('retained candidate slot/seed binding differs')
                if candidate['plan_path']!=f'learned_candidates/{declared["task_id"]}/E2/slot_{slot:02d}/plan.json':
                    raise ValueError('retained K1 plan path differs')
        return retained
    dataset = load_residual_dataset(output/'dataset/manifest.json')
    normalizer = ConditionNormalizer.from_dict(read(output/'training/condition_normalizer.json'))
    checkpoint = output/'training/model/selected.pt'
    sampler = load_residual_sampler(checkpoint, device=device)
    rows, artifacts = [], []
    def save(relative, value):
        write(output/relative, value)
        artifacts.append({'path': relative, 'sha256': sha(output/relative)})
    for i, declared in enumerate(protocol['test_records']):
        task = TaskSpec.from_dict(read(output/declared['task_path']))
        definition = read(output/declared['definition_path'])
        root = f'learned_candidates/{task.task_id}'
        try:
            z, retrieval = retrieve_train_residual(dataset, task, definition, normalizer)
            transferred = TaskAnchoredResidualPlan.from_definition(definition, z)
            e1 = {'plan_path': root+'/E1/plan.json', 'status': 'DECLARED', **retrieval}
            save(e1['plan_path'], transferred.to_dict())
        except (ValueError, TypeError, KeyError) as error:
            e1 = {'plan_path': root+'/E1/rejected.json', 'status': 'REJECTED', 'reason': str(error)}
            save(e1['plan_path'], e1)
        e2 = []
        for slot, seed in enumerate(declared['E2_sample_seeds']):
            started = time.perf_counter()
            noise = np.random.default_rng(seed).standard_normal(12).astype(np.float32)
            noise[~np.repeat(np.asarray(definition['interval_mask'], dtype=bool), 2)] = 0.
            relative = root+f'/E2/slot_{slot:02d}'
            save(relative+'/started.json', {'task_id':task.task_id,'slot':slot,'seed':seed,
                'initial_noise':noise.tolist(),'checkpoint_sha256':sha(checkpoint)})
            raw = {'task_id': task.task_id, 'slot': slot, 'seed': seed,
                'initial_noise': noise.tolist(), 'raw_z_m': None,
                'checkpoint_sha256': sha(checkpoint), 'inference_repair': False}
            try:
                z, metadata = sampler.sample(task, definition, noise)
                raw['sampler_metadata']=metadata
                values=np.asarray(z)
                if np.isfinite(values).all():raw['raw_z_m']=values.tolist()
                else:
                    raw['raw_z_m']={'shape':list(values.shape),'flattened_values':[float(v) if np.isfinite(v) else str(v) for v in values.reshape(-1)]}
                proposal = TaskAnchoredResidualPlan.from_definition(definition, z)
                candidate = proposal.to_dict()
                precheck = reference_precheck(task, proposal)
                legal = True
            except (ValueError, TypeError, KeyError, FloatingPointError, RuntimeError) as error:
                raw['generation_or_format_failure']={'type':type(error).__name__,'message':str(error)}
                candidate = {'schema': 'v64_b2_rejected_raw_residual_v1', 'raw': raw, 'reason': str(error)}
                precheck = {'passed': False, 'reference_task_passed': False,
                    'reference_velocity_passed': False, 'errors': [{'message': str(error)}],
                    'whole_body_geometry': 'NOT_RUN', 'joint_control_point_checks': 'N/A_NEW_REPRESENTATION'}
                legal = False
            raw['generation_wall_s']=time.perf_counter()-started
            save(relative+'/raw.json', raw)
            save(relative+'/plan.json', candidate)
            save(relative+'/reference_precheck.json', precheck)
            e2.append({'slot': slot, 'seed': seed, 'plan_path': relative+'/plan.json',
                'raw_path': relative+'/raw.json', 'precheck_path': relative+'/reference_precheck.json',
                'finite_format_amplitude_passed': legal, 'precheck': precheck})
        rows.append({'task_id': task.task_id, 'E1': e1, 'E2': e2})
    result = {'schema': 'v64_b2_fixed_TEST_candidates_v1', 'records': rows, 'artifacts': artifacts,
        'checkpoint_path': str(checkpoint.resolve()), 'checkpoint_sha256': sha(checkpoint),
        'source_identity_sha256':sha(output/'source_identity.json'),
        'dataset_sha256':sha(output/'dataset/manifest.json'),
        'training_config_sha256':sha(output/'training/training_config.json'),
        'K1_slot': 0, 'K4_actual': 'NOT_RUN', 'TEST_actual_observed_before_sampling': 0}
    write(manifest, result)
    return result


def finalize(output, protocol, reason=None):
    """Build parallel capability/delivery/learning verdicts from bound evidence."""
    ledger = validate_budget(output, protocol)
    zero = read(output/'attempts/zero_interface/attempt_result.json')
    teacher = read(output/'teacher_manifest.json')
    dataset = read(output/'dataset/manifest.json')
    training_path = output/'training/model/training_report.json'
    trained = training_path.exists()
    training_report=read(training_path) if trained else read(output/'training/training_status.json')
    candidate_path = output/'test_candidate_manifest.json'
    candidates = read(candidate_path) if candidate_path.exists() else None
    table_a = {'denominator': 16, 'status': 'EVALUATED' if candidates else 'NOT_RUN',
        'reason': reason, 'joint_codec_geometry': 'N/A/new-representation', 'whole_body_reference_geometry': 'N/A/no_q_ref'}
    if candidates:
        slots = [s for r in candidates['records'] for s in r['E2']]
        table_a.update(finite_format_amplitude_pass_count=sum(s['finite_format_amplitude_passed'] for s in slots),
            reference_task_pass_count=sum(s['precheck'].get('reference_task_passed', False) for s in slots),
            reference_velocity_pass_count=sum(s['precheck'].get('reference_velocity_passed', False) for s in slots),
            nonzero_reference_count=sum(s['precheck'].get('nonzero_reference', False) for s in slots),
            K1_accepted_task_count=sum(r['E2'][0]['precheck']['passed'] for r in candidates['records']),
            K4_reference_accepted_task_count=sum(any(s['precheck']['passed'] for s in r['E2']) for r in candidates['records']),
            reference_offset_peak_m=[s['precheck'].get('reference_offset_analytic_peak_m') for s in slots],
            duplicate_raw_z_count=16-len({json.dumps(read(output/s['raw_path'])['raw_z_m']) for s in slots}),
            candidate_records=slots)
    table_b = []
    for method in ('E0', 'E1', 'E2'):
        rows = []
        for i, declared in enumerate(protocol['test_records']):
            path = output/'attempts'/f'TEST_{i:02d}_{method}'/'attempt_result.json'
            if path.exists():
                result = read(path)
                evaluation = result.get('evaluation') or {}
                rows.append({'task_id': declared['task_id'], 'status': result['status'],
                    'full_task_success': result['full_task_success'], 'entered_actual': result['entered_actual'],
                    'actual_runner_started':result.get('actual_runner_started',False),
                    'actual_steps': result['actual_steps'], 'reference_binding': evaluation.get('reference_binding'),
                    'task_requirements': evaluation.get('task_requirements'),
                    'native_geometry': evaluation.get('native_geometry'), 'metrics': evaluation.get('metrics'),
                    'execution_failure': result.get('execution_failure'), 'path': str(path.resolve()), 'sha256': sha(path)})
            else:
                rows.append({'task_id': declared['task_id'], 'status': 'NOT_RUN',
                    'full_task_success': False, 'entered_actual': False,'actual_runner_started':False, 'actual_steps': 0, 'reason': reason})
        table_b.append({'method': method, 'denominator': 4,
            'full_task_success_count': sum(r['full_task_success'] for r in rows),
            'actual_attempt_count': sum(r['actual_runner_started'] for r in rows),
            'nonzero_step_actual_attempts':sum(r['entered_actual'] for r in rows),
            'precheck_reject_count': sum(r['status'] == 'REFERENCE_PRECHECK_REJECTED' for r in rows),
            'not_run_count': sum(r['status'] == 'NOT_RUN' for r in rows), 'records': rows})
    successes = [r for r in teacher['records'] if r['full_task_success'] and r['nonzero_reference']]
    test_nonzero = [r for m in table_b for r in m['records'] if r['full_task_success'] and (r.get('reference_binding') or {}).get('nonzero_reference_consumed')]
    representation_verified = bool(successes or test_nonzero)
    zero_binding = (zero.get('evaluation') or {}).get('reference_binding', {})
    delivery = len(teacher['records']) == 24 and ledger['terminal_slots'] >= 25 and (not trained or all(m['not_run_count'] == 0 for m in table_b))
    verdict = {'research_delivery_complete': delivery,
        'zero_residual_parity_passed': bool(zero['full_task_success'] and zero_binding.get('zero_residual_all_10_fields_exact_base_parity') is True),
        'task_anchor_representation_verified': representation_verified,
        'nonzero_residual_full_task_count': len(successes)+len(test_nonzero),
        'teacher_successful_task_count': len({r['task_id'] for r in successes}),
        'training_executed': trained, 'DATA_LIMITED': dataset['data_status'] == 'DATA_LIMITED',
        'diffusion_full_task_success_count': table_b[2]['full_task_success_count'],
        'advantage_over_retrieval': 'not_established', 'deployment': 'NOT_MET',
        'continuous_time_certified': False, 'stop_reason': reason,
        'training_and_TEST_stopped_under_predeclared_rule': not trained}
    # A pilot advantage requires successful nonzero E2 and a strict paired success
    # gain; paths from failed tasks never establish an advantage.
    if table_b[2]['full_task_success_count'] > table_b[1]['full_task_success_count'] and any(
            r['full_task_success'] and (r.get('reference_binding') or {}).get('nonzero_reference_consumed') for r in table_b[2]['records']):
        verdict['advantage_over_retrieval'] = 'established_in_pilot'
    feasibility = {declared['task_id']: ('ESTABLISHED_BY_NONLEARNING_FULL_TASK' if any(
        m['records'][i]['full_task_success'] for m in table_b[:2]) else 'FEASIBILITY_NOT_ESTABLISHED')
        for i, declared in enumerate(protocol['test_records'])}
    paired_paths=[]
    for i,declared in enumerate(protocol['test_records']):
        for method in ('E1','E2'):
            zero_row=table_b[0]['records'][i]
            method_row=table_b[1 if method=='E1' else 2]['records'][i]
            row={'task_id':declared['task_id'],'comparison':method+'_vs_E0',
                'eligible':zero_row['full_task_success'] and method_row['full_task_success']}
            if row['eligible']:
                arrays=[]
                for group in ('E0',method):
                    result=read(output/'attempts'/f'TEST_{i:02d}_{group}'/'attempt_result.json')
                    replay=Path(result['evaluation_path']).parent/'fresh_replay.npz'
                    with np.load(replay,allow_pickle=False) as state:
                        arrays.append(state['continuum_position'].copy())
                difference=np.linalg.norm(arrays[1]-arrays[0],axis=1)
                row.update(actual_continuum_path_difference_peak_m=float(difference.max()),
                    actual_continuum_path_difference_rms_m=float(np.sqrt(np.mean(difference**2))),
                    scope='paired complete independently verified 27s Tasks')
            else:row['status']='NOT_COMPARED_incomplete_or_failed_Task_or_safety'
            paired_paths.append(row)
    summary = {'schema': 'v64_b2_final_summary_v1', 'verdict': verdict, 'table_A': table_a,
        'table_B': table_b, 'TEST_feasibility': feasibility, 'dataset_counts': dataset['counts'],
        'budget': ledger, 'teacher_successes': successes,'training':training_report,
        'paired_actual_path_comparisons':paired_paths,
        'timing_scope': 'inherited instrumented dispatch timelines plus algorithm/preparation clocks; real delayed-state validity NOT_VERIFIED',
        'source_identity_sha256': sha(output/'source_identity.json'), 'plan_sha256': sha(output/'plan.json')}
    write(output/'summary.json', summary)
    write(output/'table_A.json', table_a)
    write(output/'table_B.json', table_b)
    write(output/'budget_ledger.json', ledger)
    write(output/'paired_path_comparison.json',paired_paths)
    report = ['# V6.4-B.2 task-anchored Cartesian residual pilot', '',
        'This independent supplement preserves B.1 conclusions and its complete failure evidence.', '',
        f'Research delivery: **{delivery}**. Zero interface parity: **{verdict["zero_residual_parity_passed"]}**.',
        f'Nonzero full-task results: **{verdict["nonzero_residual_full_task_count"]}**; successful teacher tasks: **{verdict["teacher_successful_task_count"]}**.',
        f'Training executed: **{trained}**; DATA_LIMITED: **{verdict["DATA_LIMITED"]}**; learning advantage: **{verdict["advantage_over_retrieval"]}**.', '',
        f'Stop reason: {reason or "completed frozen protocol"}.', '',
        '| Group | Full 27s Task /4 | Actual attempts | Precheck rejects | Not run |', '|---|---:|---:|---:|---:|']
    report += [f'| {m["method"]} | {m["full_task_success_count"]}/4 | {m["actual_attempt_count"]} | {m["precheck_reject_count"]} | {m["not_run_count"]} |' for m in table_b]
    report += ['', 'Reference anchor acceptance is a property of the analytic representation, not an improvement over historical 0/24 joint-codec Task results.', '',
        'Geometry: original native robot-target 500Hz; original whole-body 50Hz with configuration subdivisions4. No continuous-time claim. Missing geometry is NOT_RUN, never zero clearance.', '',
        f'Budget: {ledger["actual_runner_started"]}/37 runner entries; {ledger["nonzero_step_actual_attempts"]} nonzero-step actual attempts; {ledger["actual_physics_steps"]} actual physical steps. Preview, independent replay and geometry are separate costs.', '',
        'Timing: 20ms planner / 2ms physics / 27s Task remain frozen. Inherited C.1 dispatch timelines and algorithm/preparation wall clocks are retained but do not gate research success. Deployment remains **NOT_MET**; command validity while the real state continues evolving under calculation delay remains unverified.', '',
        'See summary.json, table_A.json, table_B.json, teacher_manifest.json and dataset/manifest.json for every slot, failure and independent evidence binding. No automatic fallback, coefficient repair, additional teacher or TEST retry was used.', '']
    with (output/'REPORT.md').open('x', encoding='utf8') as stream:
        stream.write('\n'.join(report))
    return summary


def run(output, plan_path, device='cpu', reuse=False, stop_after='all'):
    output, plan_path = Path(output).resolve(), Path(plan_path).resolve()
    if plan_path != output/'plan.json':
        raise ValueError('the execution output must contain this frozen plan')
    protocol = validate_frozen_protocol(output)
    source_guard(output/'source_identity.json')
    if (output/'summary.json').exists():
        raise FileExistsError('study is terminal; no extra runs permitted')
    lock = output/'orchestrator.lock'
    with lock.open('x', encoding='utf8') as stream:
        json.dump({'pid': os.getpid(), 'argv': [sys.executable, *sys.argv]}, stream)
    command = {'argv': [sys.executable, *sys.argv], 'pid': os.getpid(), 'exit_code':0,'started_utc': datetime.now(timezone.utc).isoformat(),
        'source_identity_sha256': sha(output/'source_identity.json'), 'plan_sha256': sha(plan_path)}
    started = time.perf_counter()
    try:
        zero = protocol['zero_interface']
        result, _ = run_slot(output, output/zero['task_path'], output/zero['plan_path'], 'zero_interface', reuse)
        if not result['full_task_success']:
            raise RuntimeError('zero-residual interface failed; preserve evidence and diagnose implementation before nonzero actual')
        if stop_after == 'zero':
            return
        for i, record in enumerate(protocol['records']):
            run_slot(output, output/record['task_path'], output/record['plan_path'], f'teacher_{i:02d}', reuse)
            validate_budget(output, protocol)
        teacher = freeze_teacher_manifest(output, protocol)
        from .residual_dataset import freeze_residual_dataset
        if not (output/'dataset/manifest.json').exists():
            freeze_residual_dataset(teacher, output/'tasks.json', output/'definitions.json', output/'dataset')
        if stop_after == 'teacher':
            return
        from .residual_diffusion import prepare_training, train_model
        if not (output/'training/training_config.json').exists() and not (output/'training/training_status.json').exists():
            prepare_training(output/'dataset/manifest.json', output/'training', plan=protocol['training'], device=device)
        dataset = read(output/'dataset/manifest.json')
        if dataset['training_status'] != 'READY':
            finalize(output, protocol, reason=dataset['reason'])
            return
        if not (output/'training/model/training_report.json').exists():
            train_model(output/'training')
        from .residual_diffusion import load_training_bundle
        load_training_bundle(output/'training')
        training_report=read(output/'training/model/training_report.json')
        for name,key in (('selected.pt','selected_checkpoint_sha256'),('last.pt','last_checkpoint_sha256')):
            if sha(output/'training/model'/name)!=training_report[key]:
                raise ValueError('completed training checkpoint hash differs')
        if stop_after == 'training':
            return
        candidates = sample_candidates(output, protocol, device)
        for i, (declared, learned) in enumerate(zip(protocol['test_records'], candidates['records'])):
            if learned['task_id']!=declared['task_id'] or learned['E2'][0]['slot']!=0:
                raise ValueError('fixed TEST Task/K1 candidate binding differs')
            for method, path in (('E0', declared['E0_plan_path']), ('E1', learned['E1']['plan_path']), ('E2', learned['E2'][0]['plan_path'])):
                run_slot(output, output/declared['task_path'], output/path, f'TEST_{i:02d}_{method}', reuse)
                validate_budget(output, protocol)
        finalize(output, protocol)
    except Exception as error:
        command['exit_code']=1
        command['error'] = {'type': type(error).__name__, 'message': str(error), 'traceback': traceback.format_exc()}
        raise
    finally:
        command.update(elapsed_wall_s=time.perf_counter()-started, finished_utc=datetime.now(timezone.utc).isoformat())
        index = len(list((output/'commands').glob('*.json')))
        write(output/'commands'/f'command_{index:03d}.json', command)
        lock.unlink()


def main():
    parser = argparse.ArgumentParser(__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    freeze = commands.add_parser('freeze')
    freeze.add_argument('--output', required=True)
    freeze.add_argument('--seed', type=int, default=2026100701)
    source = commands.add_parser('freeze-source')
    source.add_argument('--output', required=True)
    source.add_argument('--checks', required=True)
    execute = commands.add_parser('run')
    execute.add_argument('--plan', required=True)
    execute.add_argument('--output', required=True)
    execute.add_argument('--device', default='cpu')
    execute.add_argument('--reuse-completed', action='store_true')
    execute.add_argument('--stop-after', choices=('zero', 'teacher', 'training', 'all'), default='all')
    args = parser.parse_args()
    if args.command == 'freeze':
        freeze_residual_protocol(args.output, args.seed)
    elif args.command == 'freeze-source':
        freeze_source(args.output, args.checks)
    else:
        run(args.output, args.plan, args.device, args.reuse_completed, args.stop_after)


if __name__ == '__main__':
    main()
