"""Build a versioned local delivery packet from completed M1/M2/M3 outputs."""
from collections import Counter
from pathlib import Path
import json
import math
import statistics

from ff_prior.storage import read_json, file_digest, digest
from ff_prior_pilot_train.storage import atomic_json

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'artifacts/v7_1/delivery_release_001'
M1 = ROOT / 'artifacts/v7_1/delivery_guidance_smoke_001'
M2 = ROOT / 'artifacts/v7_1/delivery_dataset_128_001'
M3 = ROOT / 'artifacts/v7_1/delivery_train_dev_001'
OLD = ROOT / 'artifacts/v7_1/qualified_origin_learn_002'


def binding(path):
    path = Path(path)
    return {'path': path.relative_to(ROOT).as_posix(), 'sha256': file_digest(path),
            'bytes': path.stat().st_size}


def stats(values):
    return {'n': len(values), 'median': statistics.median(values),
            'maximum': max(values), 'minimum': min(values)} if values else {'n': 0}


def build():
    if (OUT / 'packet.json').exists():
        old = read_json(OUT / 'packet.json')
        for b in old['inputs'] + old['outputs']:
            if file_digest(ROOT / b['path']) != b['sha256']:
                raise ValueError('Existing release binding changed; do not overwrite')
        return {'status': 'NO_WORK_ALREADY_PACKAGED', 'packet': str(OUT / 'packet.json')}
    inputs = []
    for run in (M1, M2, M3):
        status = read_json(run / 'status.json')
        if status['status'] not in ('FINISHED', 'FINISHED_WITH_BLOCKED_ITEMS'):
            raise ValueError('Package only after worker completion')
        for name in ('run.json', 'summary.json', 'independent_verification.json', 'physical_contract.json'):
            inputs.append(binding(run / name))
        if read_json(run / 'independent_verification.json')['status'] != 'PASS':
            raise ValueError('Raw audit unresolved: ' + run.name)
    m2 = read_json(M2 / 'summary.json'); m3 = read_json(M3 / 'summary.json')
    chosen = m3['selected_method']
    if chosen not in ('expanded_diffusion', 'frozen_diffusion'):
        raise ValueError('No identity-bound engineering-qualified model')
    modeldir = (M3 if chosen == 'expanded_diffusion' else OLD) / 'learning/goal_loss'
    model = {name: binding(modeldir / name) for name in ('model.pt', 'codec.json', 'training.json')}
    index = read_json(M2 / 'training_index.json')['records']
    increment = read_json(M2 / 'qualified_increment.json')['records']
    assert len(index) == 14014 and len(increment) == 128
    assert len({e['sample_id'] for e in index}) == len(index)
    assert all(e['split'] == 'TRAIN' for e in increment)
    assert all(e['execution_status'] in ('NOT_RUN', 'PASS') for e in increment)
    train_families = {e['family_id'] for e in index if e['split'] == 'TRAIN'}
    val_families = {e['family_id'] for e in index if e['split'] == 'VAL'}
    assert not train_families & val_families
    dataset = {'version': 'delivery_dataset_13966_train_48_val_v1',
        'index': binding(M2 / 'training_index.json'),
        'increment_manifest': binding(M2 / 'qualified_increment.json'),
        'counts_by_split': dict(Counter(e['split'] for e in index)),
        'counts_by_split_and_level': dict(Counter(e['split'] + ':' + e['validation_level'] for e in index)),
        'increment_counts_by_level': dict(Counter(e['validation_level'] for e in increment)),
        'TRAIN_VAL_family_overlap': [], 'TEST_read': False,
        'source_records': index,
        'scope': 'Existing and increment labels retained individually. MODEL_VALID does not certify torque execution. No blanket promotion from sampled execution.'}
    model_manifest = {'version': 'delivery_model_v1', 'selected_method': chosen,
        'files': model, 'sampler': 'late4_diffusion', 'source_commit': '09ba8bae',
        'executed_source': binding(M3 / 'executed_source.zip'),
        'candidate_budget': {'network_calls': 16, 'coupled_queries': 113, 'repair_queries': 110,
                             'integration_intervals': 15872, 'wall_s': 15},
        'comparison': binding(M3 / 'summary.json'), 'binding': binding(M3 / 'checkpoint_binding.json'),
        'training_proof': binding(M3 / 'training_verification.json'),
        'dataset': dataset['index'], 'engineering_gate': m3['engineering_gates'][chosen],
        'world_loss_scope': 'v MSE + teacher-local pose surrogate; not exact nonlinear endpoint-gradient supervision',
        'raw_proposal_success': 'NOT_REMEASURED_IN_M3', 'TEST_final_model_freeze': False,
        'advantage': 'NOT_ESTABLISHED'}
    measurements = {}; failures = []
    for method in m3['metrics']:
        candidates = [read_json(p) for p in (M3 / 'candidates').glob(f'*/*/{method}/*.json')]
        outcomes = [read_json(p) for p in (M3 / 'executions').glob(f'*/*/{method}/outcome.json')]
        assert len(candidates) == 180 and len(outcomes) == 36
        groups = {}
        for role in ('ORIGINAL_DEV', 'ALREADY_USED_DEVELOPMENT'):
            rows = [r for r in candidates if r['development_role'] == role]
            groups[role] = {'candidates': len(rows), 'eligible': sum(r['eligible'] for r in rows)}
        for seed in range(3):
            rows = [r for r in candidates if r['seed_group'] == seed]
            groups['seed_' + str(seed)] = {'candidates': len(rows), 'eligible': sum(r['eligible'] for r in rows)}
        valid = [r for r in candidates if r['eligible']]
        measurements[method] = {'counts': m3['metrics'][method], 'groups': groups,
            'eligible_position_error_max_of_arms_mm': stats([1000 * max(r['validation']['frozen_goal_position_error_m']) for r in valid]),
            'eligible_orientation_error_max_of_arms_deg': stats([180 / math.pi * max(r['validation']['frozen_goal_rotation_error_rad']) for r in valid]),
            'all_candidate_wall_s': stats([r['wall_s'] for r in candidates]),
            'qualification_conditioned_statistics': True}
        for r in candidates:
            if not r['eligible']:
                failures.append({'phase': 'M3_candidate', 'method': method, 'sample_id': r['sample_id'],
                    'seed_group': r['seed_group'], 'candidate_id': r['candidate_id'],
                    'status': r['status'], 'reason': r.get('reason'), 'validation': r.get('validation')})
        for p in (M3 / 'executions').glob(f'*/*/{method}/outcome.json'):
            r = read_json(p)
            if r['status'] != 'EXECUTION_VERIFIED':
                failures.append({'phase': 'M3_execution', 'source': binding(p), 'outcome': r})
    expansion_allowed = (m2['M3_ready'] and m3['M4_ready'] and m2['admission_rate_actual_attempts'] >= .5)
    next_batch = {'version': 'next_256_coverage_proposal_v1', 'status': 'PROTOCOL_ONLY_NOT_STARTED',
        'engineering_prerequisites_pass': bool(expansion_allowed), 'automatic_launch': False,
        'slots': 256, 'targets': 64, 'slots_per_target': 4, 'max_attempts_per_slot': 2, 'max_attempts': 512,
        'selection': 'TRAIN only; deterministic metadata coverage ranking; 32 obstacle and 32 no-obstacle targets; 64 distinct origins where available. Exclude all used DEV and VAL/TEST families. Freeze exact task hashes, seeds, origins and execution units before generation.',
        'execution_units': 32, 'obstacle_execution_units': 16,
        'unit_selection': 'first frozen-order eligible candidate per preselected target; no replacement after execution failure',
        'individual_gates': 'unchanged historical no-penetration, both world endpoints 5mm/1degree, 2mm geometry step and all existing motion/dynamics/range/replay gates',
        'admission': 'deduplicate; complete state and lineage; MODEL_VALID unless actually executed and independently replayed; failed/unknown execution excluded',
        'stop': 'at frozen attempt/time/storage limits; never silently fill partial quota or recurse to 512/1024',
        'next_continue_gate': {'minimum_qualified': 128, 'minimum_distinct_targets': 32,
                               'minimum_execution_replay': 26, 'execution_denominator': 32,
                               'minimum_qualification_per_actual_attempt': .5},
        'superiority_time_or_query_win_required': False,
        'resource_contract': {'threads': 2, 'wall_s': 14400, 'max_output_gib': 20},
        'exact_task_seed_manifest_frozen': False, 'requires_separate_run_authorization': True}
    readiness = {'status': 'NOT_READY', 'TEST_started': False,
        'engineering_model_available': True, 'raw_audits_pass': True,
        'independent_claim_review': 'PENDING',
        'untouched_TEST_family_manifest_frozen': False, 'final_TEST_model_and_sampler_frozen': False,
        'one_shot_TEST_protocol_and_budget_frozen': False,
        'reason': 'This release is development-validated. Independent held-out family manifest and one-shot TEST model/protocol are not frozen. No TEST content accessed by this packager.',
        'next': 'After independent review, separately authorize and freeze a single untouched TEST protocol/model. Do not use TEST to tune or repeat. Broad superiority is not implied by engineering pass.'}
    OUT.mkdir(parents=True, exist_ok=True)
    products = {'dataset_manifest.json': dataset, 'model_manifest.json': model_manifest,
        'comparison_metrics.json': measurements,
        'failure_ledger.json': {'current_M3_failures': failures,
            'M1_source': binding(M1 / 'failures.json') if (M1 / 'failures.json').exists() else None,
            'M2_source': binding(M2 / 'failures.json'),
            'historical_paired_objectives': {'status': 'HISTORICAL_BLOCKED_NOT_RELABELED', 'D_ratio': .914089,
                'source': binding(ROOT / 'ff_prior_paired_objectives/evidence/completed_001/summary.json')},
            'old_models_results_preserved': True},
        'next_256_protocol.json': next_batch, 'TEST_readiness.json': readiness}
    for name, data in products.items():
        atomic_json(OUT / name, data)
    for path in (M2 / 'training_index.json', M2 / 'qualified_increment.json', M2 / 'diversity.json',
                 M3 / 'training_verification.json', M3 / 'checkpoint_binding.json',
                 M3 / 'learning/goal_loss/training.json', M3 / 'failures.json'):
        if path.exists(): inputs.append(binding(path))
    inputs.extend(model.values())
    atomic_json(OUT / 'packet.json', {'version': 'delivery_release_001', 'status': 'PACKAGED_PENDING_REVIEW',
        'inputs': inputs, 'outputs': [binding(OUT / name) for name in products],
        'physics_or_training_launched': False, 'TEST_read': False})
    return {'status': 'PACKAGED_PENDING_REVIEW', 'packet': str(OUT / 'packet.json'),
            'selected_method': chosen, 'next_256_prerequisites_pass': bool(expansion_allowed)}


if __name__ == '__main__':
    print(json.dumps(build(), ensure_ascii=False, indent=2))
