"""Preserve raw DDIM20 K1/K8 and execute the shared statically chosen proposals."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time
import traceback

from v6_4.generate_proposals import generate
from v6_4.run_planning import load_task_suite
from v6_4.train_diffusion import _source_identity
from v6_4.run_teacher_comparison import (SCORE_DEFINITION, _code_sources, _manifest,
    _physical_steps, _trial_row, _write, evaluate_candidate_set)


def _executed_sources():
    sources = _code_sources()
    sources.update(_source_identity()['learning_sources_sha256'])
    for name in ('run_diffusion_comparison.py', 'generate_proposals.py'):
        sources['v6_4/'+name] = hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
    return sources


def _failure_cell(task, k, directory, error):
    failure = {'type': type(error).__name__, 'message': str(error), 'traceback': traceback.format_exc()}
    actual = directory / f'K{k}' / 'actual'
    try:
        steps = _physical_steps(actual)
    except Exception as trace_error:
        steps = None
        failure['physical_step_count_error'] = str(trace_error)
    row = dict(task_id=task.task_id, task_sha256=task.sha256(), candidate_count=k,
        selectedIndex=None, task_success=False, complete=False, proposal_accepted=False,
        raw_proposal_passed=False, status='COMPARISON_EVIDENCE_ERROR',
        new_physics_steps=steps, reused_actual_trial=False, failure=failure)
    if actual.is_dir():
        row['actual_trial_path'] = actual.as_posix()
    path = directory / f'K{k}' / 'comparison_row.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        _write(path, row)
    return row


def run_comparison(tasks_path, checkpoint, output, *, seed=64, device='cpu', split='test'):
    tasks = tuple(t for t in load_task_suite(tasks_path) if t.split == split)
    if split == 'test' and (len(tasks) != 3 or len({t.family for t in tasks}) != 3):
        raise ValueError('all three independent test families must be retained')
    if not tasks or split not in ('train', 'val', 'test'):
        raise ValueError('nonempty declared split required')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    checkpoint = Path(checkpoint).resolve()
    checkpoint_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    sources = _executed_sources()
    _write(output / 'protocol.json', {
        'schema': 'v6_4_diffusion_test_protocol_v1', 'split': split,
        'task_suite_sha256': hashlib.sha256(Path(tasks_path).read_bytes()).hexdigest(),
        'task_hashes': {t.task_id: t.sha256() for t in tasks},
        'checkpoint_path': checkpoint.as_posix(), 'checkpoint_sha256': checkpoint_hash,
        'source_sha256': sources, 'candidate_count': 8, 'seed': seed,
        'candidate_seeds': list(range(seed, seed + 8)), 'DDIM_steps': 20,
        'K1_definition': 'candidate0 only',
        'K8_definition': 'minimum common static score among eight raw-gate-passing proposals',
        'score_definition': SCORE_DEFINITION, 'repair_used': False, 'fallback_used': False,
        'selection_uses_actual_outcomes': False, 'physics_period_s': .002, 'task_period_s': .02,
        'wall_20ms_is_acceptance_gate': False})
    started = time.perf_counter()
    rows = []
    for task in tasks:
        directory = output / task.task_id
        directory.mkdir()
        proposals, generation_failure = [None] * 8, None
        try:
            proposals, _ = generate(task, checkpoint, directory / 'generation',
                                    K=8, seed=seed, device=device)
        except Exception as error:
            generation_failure = {'type': type(error).__name__, 'message': str(error),
                                  'traceback': traceback.format_exc()}
            _write(directory / 'generation_failure.json', generation_failure)
        try:
            selection = evaluate_candidate_set(task, proposals, directory / 'screening')
        except Exception as error:
            failure = {'type': type(error).__name__, 'message': str(error),
                       'traceback': traceback.format_exc()}
            _write(directory / 'comparison_failure.json', failure)
            first, eighth = [_failure_cell(task, k, directory, error) for k in (1, 8)]
        else:
            try:
                first = _trial_row(task, 1, selection['selectedIndexK1'], proposals, directory,
                                   method='diffusion')
            except Exception as error:
                first = _failure_cell(task, 1, directory, error)
            try:
                eighth = _trial_row(task, 8, selection['selectedIndex'], proposals, directory,
                    prior_trial=first if first.get('actual_trial_path') and first.get('proposal_sha256') else None,
                    method='diffusion')
            except Exception as error:
                eighth = _failure_cell(task, 8, directory, error)
        for row in (first, eighth):
            row.update(method='diffusion', generation_scope='eight independent serial seeds; K1 is prefix comparator',
                       generation_failure=generation_failure)
        rows.extend((first, eighth))
        _write(directory / 'task_result.json', {'rows': [first, eighth]})
        print(json.dumps({'task_id': task.task_id, 'K1': first['status'],
                          'K8': eighth['status']}), flush=True)
    after = _executed_sources()
    report = {'schema': 'v6_4_diffusion_test_comparison_v1', 'split': split,
        'task_count': len(tasks), 'candidate_attempt_count': len(tasks) * 8, 'rows': rows,
        'by_K': {str(k): {'task_count': len(tasks),
            'accepted_count': sum(r['proposal_accepted'] for r in rows if r['candidate_count'] == k),
            'task_success_count': sum(r['task_success'] for r in rows if r['candidate_count'] == k)} for k in (1, 8)},
        'new_physics_steps': sum(r['new_physics_steps'] for r in rows if r['new_physics_steps'] is not None),
        'physical_step_count_exact_known': all(r['new_physics_steps'] is not None for r in rows),
        'all_tasks_attempted': len(rows) == 2 * len(tasks), 'all_sources_unchanged': sources == after,
        'checkpoint_unchanged': hashlib.sha256(checkpoint.read_bytes()).hexdigest() == checkpoint_hash,
        'elapsed_wall_s': time.perf_counter() - started, 'repair_used': False, 'fallback_used': False,
        'hardware': 'NOT_ESTABLISHED', 'wall_20ms_is_acceptance_gate': False}
    _write(output / 'report.json', report)
    _write(output / 'artifact_manifest.json', _manifest(output))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tasks', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=64)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--split', choices=('train', 'val', 'test'), default='test')
    arguments = parser.parse_args()
    result = run_comparison(arguments.tasks, arguments.checkpoint, arguments.output,
        seed=arguments.seed, device=arguments.device, split=arguments.split)
    print(json.dumps({k: result[k] for k in ('by_K', 'new_physics_steps', 'all_tasks_attempted')}), flush=True)
