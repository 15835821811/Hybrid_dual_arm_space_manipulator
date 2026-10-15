"""Frozen checkpoint, bounded goal projection, VAL + original DEV smoke."""
from pathlib import Path
import subprocess
import time
import zipfile
import numpy as np
from ff_prior.pipeline import audit, acceptance
from ff_prior.storage import read_json, atomic_json, atomic_npz, file_digest, digest, Ledger, RunLock
from ff_prior.contracts import assert_task_compatible, assert_frozen
from ff_prior.guidance import public_goal_initializer
from ff_prior.model_gates import evaluate_reference
from ff_prior.coupled_predictor import pose_errors
from ff_prior.physical_audit import audit_execution
from ff_prior.spline import Reference
from ff_prior_next import learning
from ff_prior_fixed.experiment import inputs as original_inputs, sources as previous_sources
from .projection import GoalProjector, CONTRACT

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = ROOT/'artifacts/v7_1/fixed_train_consistency_001/learning'
METHODS = ('structured_projected', 'diffusion_projected', 'diffusion_smooth_only')
PROTOCOL = {'version': 'v71_goal_projection_smoke_1', 'profile': 'smoke',
    'methods': list(METHODS), 'checkpoint': 'fixed_train_consistency_001', 'retraining': False,
    'VAL_targets': 6, 'DEV_targets': 4, 'candidates_per_task': 5, 'ddim_steps': 16,
    'projection': CONTRACT, 'projection_iterations': [14, 15],
    'max_candidates': 150, 'max_selected_executions': 30, 'wall_budget_s': 1800,
    'pilot': False, 'TEST': False, 'no_replacement': True, 'no_private_terminal': True,
    'stability': '>=95% eligible in each split, >=4/5 per task, all 10 projected selected executions verified',
    'benefit': 'not inferred from projection success; compare same repair on structured baseline',
    'parameter_selection': 'fixed before VAL/DEV run; no sweep',
    'VAL_scope': 'development check, not independent held-out test'}


def sources():
    return dict(previous_sources(), **{p.relative_to(ROOT).as_posix(): file_digest(p)
        for p in sorted((ROOT/'ff_prior_goal').glob('*.py'))})


def inputs(source):
    old, data, tasks, rows, files = original_inputs(source)
    source = Path(source).resolve()
    for name in ('model.pt', 'training.json', 'scaler.json', 'clean_prior.json'):
        p = CHECKPOINT/name
        files[str(Path('..')/'fixed_train_consistency_001/learning'/name)] = file_digest(p)
    return old, data, tasks, rows, files


def validation_targets(rows):
    pool = sorted((r for r, s in rows if s == 'VAL'),
                  key=lambda r: digest({'selection': 'goal_projection_v1', 'sample': r['sample_id']}))
    selected, families = [], set()
    for row in pool:
        if row['family_id'] in families: continue
        selected.append({'sample_id': row['sample_id'], 'family_id': row['family_id'], 'task': row['task']})
        families.add(row['family_id'])
        if len(selected) == 6: break
    if len(selected) != 6: raise ValueError('six distinct VAL families required')
    return selected


def candidate(method, index, task, base, seed, bundle, robot, initial, collision, profile, deadline):
    assert_frozen(task)
    projector = GoalProjector(task, robot, initial, collision, profile, deadline,
                              goal_enabled=method != METHODS[2])
    before = robot.calls.copy(); wall = time.perf_counter(); cpu = time.process_time()
    if method == METHODS[0]:
        rng = np.random.default_rng(seed)
        z = base+.015*np.sin(np.linspace(0, np.pi, 19))[:, None]*rng.normal(size=(1, 17))
        proposal = z.copy(); logs = []
        for i in (14, 15):
            z, info = projector(z, i); logs.append(info)
        sampling = {'actual_network_calls': 0, 'guidance': logs, 'raw_proposal_hash': digest(proposal)}
    else:
        model, scaler, prior = bundle
        z, sampling = learning.sample(model, scaler, prior, base, task, seed, projector)
    sampling['goal_projection_enabled'] = method != METHODS[2]
    sampling['curvature_regularizer_shared'] = True
    prediction, validation = evaluate_reference(robot, initial,
        Reference(z, task['condition']['initial_features'], task['condition']['duration']),
        collision, profile, deadline)
    passed, score = False, None
    if prediction is not None:
        pos, rot = pose_errors(prediction['ee'][-1], task['condition']['goals'])
        passed = bool(np.all(pos <= profile.position_error_m) and np.all(rot <= profile.rotation_error_rad))
        score = float(np.sum((pos/profile.position_error_m)**2)+np.sum((rot/profile.rotation_error_rad)**2))
        validation.update(frozen_goal_position_error_m=pos, frozen_goal_rotation_error_rad=rot)
    assert_frozen(task)
    return {'status': 'DONE', 'method': method, 'candidate_id': index, 'seed': seed,
        'task_hash': task['task_hash'], 'public_base_hash': digest(base), 'Z': z,
        'reference_hash': Reference(z, task['condition']['initial_features'], task['condition']['duration']).reference_hash,
        'sampling': sampling, 'validation': validation, 'frozen_goal_pass': passed,
        'eligible': passed and validation['validation_level'] == 'MODEL_VALID', 'score': score,
        'wall_s': time.perf_counter()-wall, 'cpu_s': time.process_time()-cpu,
        'model_calls': {k: robot.calls[k]-before[k] for k in before}, 'private_teacher_fields_accessed': False}


def run(source, out, progress, resume=False, previous_elapsed=0.):
    from .reporting import write_report
    source, out = Path(source).resolve(), Path(out).resolve()
    if source == out or source in out.parents or out in source.parents: raise ValueError('independent output required')
    old, data, tasks, rows, files = inputs(source)
    identity = {'protocol': PROTOCOL, 'source': str(source), 'source_files': files,
                'tasks_hash': digest(tasks), 'source_hashes': sources()}
    with RunLock(out):
        if (out/'run.json').exists() and (not resume or read_json(out/'run.json')['identity'] != identity):
            raise ValueError('run identity conflict')
        proof = read_json(ROOT/'ff_prior_goal/evidence/software_tests.json')
        if proof['status'] != 'PASS' or proof['source_hashes'] != sources(): raise ValueError('software proof stale')
        if proof['junit_sha256'] != file_digest(ROOT/'ff_prior_goal/evidence/tests.xml'): raise ValueError('test evidence changed')
        audit_report, robot, initial, collision = audit(old['config']); profile = acceptance(old['config'])
        if audit_report['acceptance'] != read_json(source/'baseline_audit.json')['acceptance']: raise ValueError('physical gates changed')
        groups = {'VAL': validation_targets(rows), 'DEV': tasks}
        for entries in groups.values():
            for entry in entries: assert_task_compatible(entry['task'], robot, initial, collision, profile)
        if not (out/'run.json').exists():
            atomic_json(out/'run.json', {'identity': identity, 'acceptance': audit_report['acceptance'],
                'robot_hash': initial.model_hash, 'initial_state_id': initial.state_id,
                'producer_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()})
            atomic_json(out/'frozen_tasks.json', groups); atomic_json(out/'software_tests.json', proof)
            with zipfile.ZipFile(out/'executed_source.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
                for p in identity['source_hashes']: archive.write(ROOT/p, p)
        if read_json(out/'frozen_tasks.json') != groups: raise ValueError('frozen evaluation targets changed')
        ledger = Ledger(out, identity)
        if (out/'summary.json').exists() and read_json(out/'summary.json')['status'] == 'FINISHED':
            return read_json(out/'summary.json')
        deadline = time.time()+max(0, 1800-previous_elapsed)
        bundle = learning.load(CHECKPOINT, initial.model_hash, initial.state_id)
        # Verify stored TRAIN correction labels anew; they never initialize a candidate.
        corrections = sorted((CHECKPOINT.parent/'corrections/records').glob('*.json'))
        if len(corrections) != 16: raise ValueError('expected 16 prior corrections')
        for p in corrections:
            row = read_json(p); key = 'label_'+row['sample_id']
            if key in ledger.data['completed']: continue
            ledger.begin(key, 'labels', 1800, 16); before = robot.calls.copy()
            progress('label_revalidation', sample=row['sample_id'])
            try:
                if row['split'] != 'TRAIN' or not row['training_eligible']: raise ValueError('ineligible TRAIN source')
                z = np.load(CHECKPOINT.parent/row['arrays'])['Z']; c = row['condition']
                prediction, validation = evaluate_reference(robot, initial, Reference(z, c['initial_features'], c['duration']), collision, profile, deadline)
                pos, rot = pose_errors(prediction['ee'][-1], c['goals'])
                passed = validation['validation_level'] == 'MODEL_VALID' and np.all(pos <= profile.position_error_m) and np.all(rot <= profile.rotation_error_rad)
                result = {'status': 'PASS' if passed else 'FAIL', 'sample_id': row['sample_id'],
                    'task_hash': row['task']['task_hash'], 'validation': validation, 'position_error_m': pos,
                    'rotation_error_rad': rot, 'execution_evidence': 'UNCHANGED_NOT_UPGRADED'}
            except Exception as error: result = {'status': 'ERROR', 'reason': repr(error)}
            atomic_json(out/'labels'/p.name, result)
            ledger.finish(result, {k: robot.calls[k]-before[k] for k in before})
        for split, entries in groups.items():
            for ti, entry in enumerate(entries):
                task = entry['task']; base = public_goal_initializer(task, robot, initial)
                for method in METHODS:
                    candidates = []
                    for k in range(5):
                        key = f'candidate_{split}_{ti}_{method}_{k}'
                        path = out/'candidates'/split/method/f'{ti:03d}_{k:02d}.json'
                        if key in ledger.data['completed']:
                            if path.exists(): candidates.append(read_json(path))
                            continue
                        ledger.begin(key, 'candidates', 1800, 150); before = robot.calls.copy()
                        progress('sampling', split=split, task=ti+1, method=method, candidate=k+1,
                            completed=ledger.data['attempts']['candidates']-1, total=150)
                        try:
                            seed = (20261017 if split == 'DEV' else 20261023)+ti*101+k*997
                            result = candidate(method, k, task, base, seed, bundle, robot, initial, collision, profile, deadline)
                        except Exception as error: result = {'status': 'ERROR', 'reason': repr(error), 'eligible': False,
                            'candidate_id': k, 'task_hash': task['task_hash'], 'method': method}
                        atomic_json(path, result); candidates.append(result)
                        ledger.finish({'status': result['status'], 'split': split, 'method': method, 'kind': 'candidate'},
                            {n: robot.calls[n]-before[n] for n in before})
                    key = f'execution_{split}_{ti}_{method}'
                    if key in ledger.data['completed']: continue
                    eligible = [c for c in candidates if c.get('eligible')]
                    chosen = min(eligible, key=lambda c: (c['score'], c['candidate_id'])) if eligible else None
                    directory = out/'executions'/split/method/f'{ti:03d}'
                    atomic_json(directory/'selection.json', {'task_hash': task['task_hash'],
                        'selected_id': chosen['candidate_id'] if chosen else None, 'selection_before_execution': True,
                        'rule': 'lowest normalized pose error among eligible candidates; no execution retry'})
                    ledger.begin(key, 'execution', 1800, 30); before = robot.calls.copy()
                    progress('torque_execution', split=split, task=ti+1, method=method,
                        selected=chosen['candidate_id'] if chosen else None)
                    try:
                        if chosen:
                            ref = Reference(np.asarray(chosen['Z']), task['condition']['initial_features'], task['condition']['duration'])
                            result = audit_execution(task, profile, robot, initial, ref, directory, deadline)
                            state = result['level']
                        else: state = 'NO_CANDIDATE'
                        result = {'status': state, 'split': split, 'method': method, 'kind': 'execution'}
                    except Exception as error: result = {'status': 'EXECUTION_FAILED', 'reason': repr(error),
                        'split': split, 'method': method, 'kind': 'execution'}
                    ledger.finish(result, {n: robot.calls[n]-before[n] for n in before})
                write_report(out, ledger, 'RUNNING')
        if any(file_digest(source/p) != h for p, h in files.items()): raise ValueError('input mutated')
        if sources() != identity['source_hashes']: raise ValueError('source code changed during run')
        atomic_json(out/'verification.json', {'source_inputs_unchanged': True, 'source_code_unchanged': True,
            'same_frozen_DEV': groups['DEV'] == tasks, 'goals_not_relabelled': True,
            'physical_profile_unchanged': True, 'TEST_arrays_read': False, 'complete_model_calls': robot.calls})
        incomplete = (out/'interruption.json').exists() or ledger.data.get('unresolved_call_accounting') or any(
            v['status'] in ('ERROR', 'BLOCKED', 'INTERRUPTED') for v in ledger.data['completed'].values())
        return write_report(out, ledger, 'FINISHED_WITH_BLOCKED_ITEMS' if incomplete else 'FINISHED')
