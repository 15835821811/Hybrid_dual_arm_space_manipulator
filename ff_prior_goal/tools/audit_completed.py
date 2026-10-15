"""Read-only verification of completed evidence; no model/physics evaluations."""
from pathlib import Path
import hashlib
import shutil
import subprocess
import zipfile
import numpy as np
from ff_prior.storage import read_json, atomic_json, digest, file_digest
from ff_prior.contracts import MODEL_GATES, EXECUTION_GATES, assert_frozen
from ff_prior.spline import Reference
from ff_prior.physical_audit import pose_errors
from ff_prior_goal.experiment import inputs, sources, validation_targets, METHODS, PROTOCOL


def main():
    source = Path('artifacts/v7_1/smoke_002'); out = Path('artifacts/v7_1/goal_projection_001')
    run = read_json(out/'run.json'); summary = read_json(out/'summary.json'); ledger = read_json(out/'ledger.json')
    assert summary['status'] == 'FINISHED' and run['identity']['source_hashes'] == sources()
    assert ledger['inflight'] is None and not ledger.get('unresolved_call_accounting')
    assert ledger['attempts'] == {'labels': 16, 'candidates': 150, 'execution': 30}
    assert len(ledger['completed']) == 196
    assert run['identity']['protocol'] == PROTOCOL
    assert all(file_digest(source/p) == h for p, h in run['identity']['source_files'].items())
    with zipfile.ZipFile(out/'executed_source.zip') as z:
        assert all(hashlib.sha256(z.read(p)).hexdigest() == h for p, h in sources().items())
    old, data, dev, rows, _ = inputs(source)
    groups = read_json(out/'frozen_tasks.json')
    assert groups['DEV'] == dev and groups['VAL'] == validation_targets(rows)
    assert run['acceptance'] == read_json(source/'baseline_audit.json')['acceptance']
    evidence = {}; networks = 0; real_executions = 0; physics_steps = 0
    minimum_clearance = []; pair_diversity = {}; hold_pos = []; hold_rot = []
    for split, tasks in groups.items():
        evidence[split] = {}
        for method in METHODS:
            all_rows = []; physical = []
            pair_diversity[split+'/'+method] = []
            for ti, entry in enumerate(tasks):
                task = entry['task']; assert_frozen(task); candidates = []
                for k in range(5):
                    r = read_json(out/'candidates'/split/method/f'{ti:03d}_{k:02d}.json')
                    candidates.append(r); all_rows.append(r)
                    assert r['status'] == 'DONE' and r['task_hash'] == task['task_hash'] and r['method'] == method
                    assert r['seed'] == (20261017 if split == 'DEV' else 20261023)+ti*101+k*997
                    assert not r['private_teacher_fields_accessed']
                    ref = Reference(r['Z'], task['condition']['initial_features'], task['condition']['duration'])
                    assert ref.reference_hash == r['reference_hash']
                    s = r['sampling']; networks += s['actual_network_calls']
                    assert s['actual_network_calls'] == (0 if method == METHODS[0] else 16)
                    assert [g['iteration'] for g in s['guidance']] == [14, 15]
                    for g in s['guidance']:
                        assert not g['private_teacher_fields_used'] and g['task_hash'] == task['task_hash']
                        assert g['goal_enabled'] == (method != METHODS[2])
                        assert len(g['history']) <= 4 and g['coupled_calls'] <= 153
                    if method != METHODS[0]:
                        assert s['sampling_precision'] == 'float64_state_float32_network'
                        assert s['prediction_type'] == 'v' and len(s['clean_prior_projection']) == 16
                        assert all(step['decomposition_residual_after'] < 1e-12 for step in s['steps'])
                    v = r['validation']; model_ok = all(v['gates'].get(g, False) for g in MODEL_GATES)
                    assert model_ok == (v['validation_level'] == 'MODEL_VALID')
                    goal_ok = ('frozen_goal_position_error_m' in v and
                        max(v['frozen_goal_position_error_m']) <= .005 and max(v['frozen_goal_rotation_error_rad']) <= np.deg2rad(1))
                    assert goal_ok == r['frozen_goal_pass'] and r['eligible'] == (model_ok and goal_ok)
                all_hashes = {read_json(out/'candidates'/split/m/f'{ti:03d}_{k:02d}.json')['public_base_hash']
                              for m in METHODS for k in range(5)}
                assert len(all_hashes) == 1
                eligible = [r for r in candidates if r['eligible']]
                chosen = min(eligible, key=lambda r: (r['score'], r['candidate_id'])) if eligible else None
                directory = out/'executions'/split/method/f'{ti:03d}'
                select = read_json(directory/'selection.json')
                assert select['selected_id'] == (chosen['candidate_id'] if chosen else None)
                assert select['selection_before_execution'] and select['task_hash'] == task['task_hash']
                z = [np.array(r['Z']) for r in eligible]
                pair_diversity[split+'/'+method].append([float(np.sqrt(np.mean((a-b)**2)))
                    for i, a in enumerate(z) for b in z[i+1:]])
                if chosen is None:
                    assert not (directory/'report.json').exists(); continue
                report = read_json(directory/'report.json'); physical.append(report); real_executions += 1
                assert read_json(directory/'frozen_task.json') == task
                assert report['task_hash'] == task['task_hash']
                assert report['metadata']['reference_hash'] == chosen['reference_hash']
                assert report['scene_model_hash'] == run['robot_hash'] and report['initial_state_id'] == run['initial_state_id']
                assert report['metadata']['robot_state_writes_during_step_loop'] == 0
                assert report['metadata']['external_wrench_applied'] is False
                with np.load(directory/'native_trace.npz') as a:
                    assert digest(a['ctrl']) == report['ctrl_hash']
                    assert len(a['qpos']) == len(a['ctrl'])+1
                    assert a['ctrl'].shape[1] == 67 and a['qpos'].shape[1] == 81
                    pos, rot = pose_errors(a['ee'], task['condition']['goals'])
                    hold = a['time']-a['time'][0] >= task['condition']['duration']-1e-9
                    assert np.max(abs(pos[hold].max(axis=0)-report['hold_max_position_error_m'])) < 1e-12
                    assert np.max(abs(rot[hold].max(axis=0)-report['hold_max_rotation_error_rad'])) < 1e-12
                    assert report['replay']['physics_steps'] == len(a['ctrl'])
                    physics_steps += 2*len(a['ctrl'])
                if report['level'] == 'EXECUTION_VERIFIED':
                    assert all(report['gates'][g] for g in EXECUTION_GATES) and report['replay']['passed']
                    assert max(report['replay']['maximum_errors'].values()) <= 1e-10
                minimum_clearance.append(report['minimum_clearance_m'])
                hold_pos.append(max(report['hold_max_position_error_m']))
                hold_rot.append(max(report['hold_max_rotation_error_rad']))
            actual = {'eligible': sum(r['eligible'] for r in all_rows),
                      'execution_verified': sum(r['level'] == 'EXECUTION_VERIFIED' for r in physical)}
            for k, v in actual.items(): assert summary['methods'][split][method][k] == v
            evidence[split][method] = actual
    assert networks == 1600 and read_json(out/'verification.json')['complete_model_calls']['physics_steps'] == physics_steps
    labels = [read_json(p) for p in (out/'labels').glob('*.json')]
    assert len(labels) == 16 and summary['label_revalidation_pass'] == sum(r['status'] == 'PASS' for r in labels)
    for r in labels:
        original = read_json(Path('artifacts/v7_1/fixed_train_consistency_001/corrections/records')/(r['sample_id']+'.json'))
        assert r['task_hash'] == original['task_hash'] and r['execution_evidence'] == 'UNCHANGED_NOT_UPGRADED'
    before = read_json('ff_prior_goal/evidence/preflight.json')
    assert all(file_digest(p) == h for p, h in before['user_dirty_file_sha256'].items())
    for branch in ('main', 'v6.4-a-diffusion-planning'):
        assert subprocess.check_output(['git', 'rev-parse', branch], text=True).strip() == before['branches'][branch]
    for directory, filename in [('ff_prior/evidence/smoke_002', 'evidence_manifest.json'),
            ('ff_prior/evidence/diffusion_repair', 'manifest.json'),
            ('ff_prior_next/evidence/public_seed_residual_001', 'evidence_manifest.json'),
            ('ff_prior_fixed/evidence/fixed_train_consistency_001', 'evidence_manifest.json')]:
        p = Path(directory); assert all(file_digest(p/k) == h for k, h in read_json(p/filename)['files'].items())
    report = {'status': 'PASS', 'evidence': evidence, 'network_calls': networks, 'real_torque_executions': real_executions,
        'physics_steps': physics_steps, 'pair_control_rms_rad': pair_diversity,
        'minimum_execution_clearance_m': min(minimum_clearance), 'max_hold_position_error_m': max(hold_pos),
        'max_hold_rotation_error_rad': max(hold_rot), 'user_files_and_old_evidence_preserved': True,
        'same_frozen_DEV': True, 'validation_gates_unchanged': True, 'source_checkpoint_identity_verified': True,
        'extra_model_or_physics_calls_in_audit': 0, 'pilot': 'NOT_RUN', 'TEST': 'NOT_RUN'}
    atomic_json(out/'independent_verification.json', report)
    dest = Path('ff_prior_goal/evidence/goal_projection_001'); dest.mkdir(parents=True, exist_ok=True)
    files = [p for p in out.rglob('*.json') if p.name != 'launcher.json']
    files += [out/'REPORT.md', out/'executed_source.zip', out/'events.jsonl']
    for p in files:
        target = dest/p.relative_to(out); target.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(p, target)
    atomic_json(dest/'evidence_manifest.json', {'weights_included': False, 'native_trace_included': False,
        'files': {p.relative_to(dest).as_posix(): file_digest(p) for p in sorted(dest.rglob('*'))
                  if p.is_file() and p.name != 'evidence_manifest.json'}})
    print({k: v for k, v in report.items() if k != 'pair_control_rms_rad'})


if __name__ == '__main__': main()
