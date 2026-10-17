"""Read-only posthoc statistics for the completed frozen endpoint experiment.

No physics, model inference, new candidates, checkpoint selection or training.
Writes separate evidence only, leaving all producer outputs untouched.
"""
from pathlib import Path
from collections import Counter
import json
import numpy as np
from ff_prior.storage import read_json, file_digest, digest
from ff_prior.spline import Reference
from ff_prior.initial_state import STATE

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'artifacts/v7_1/world_endpoint_smoke_001'
DEST = ROOT / 'ff_prior_world_endpoint/evidence/completed_001'


def stats(values):
    a = np.asarray(values, dtype=float)
    return dict(n=len(a), minimum=float(a.min()), median=float(np.median(a)), maximum=float(a.max()))


def main():
    s = read_json(OUT / 'summary.json')
    tr = read_json(OUT / 'learning/goal_loss/training.json')
    before = read_json(OUT / 'learning/goal_loss/world_diagnostic_before.json')['cases']
    after = read_json(OUT / 'learning/goal_loss/world_diagnostic_after.json')['cases']
    assert len(before) == len(after) == 32
    groups = {}
    for t in (500, 999, 'all'):
        pairs = [(a, b) for a, b in zip(before, after) if t == 'all' or a['t'] == t]
        assert all((a['sample_id'], a['t'], a['seed']) == (b['sample_id'], b['t'], b['seed']) for a, b in pairs)
        g = {'paired_loss_ratio': stats([b['loss'] / a['loss'] for a, b in pairs]),
             'improved': sum(b['loss'] < a['loss'] for a, b in pairs)}
        for name, pos in [('before', 0), ('after', 1)]:
            cases = [p[pos] for p in pairs]
            g[name] = {'loss': stats([c['loss'] for c in cases]),
                       'worst_arm_position_mm': stats([1000*max(c['position_m']) for c in cases]),
                       'worst_arm_rotation_deg': stats([np.rad2deg(max(c['rotation_rad'])) for c in cases]),
                       'pose_pass': sum(int(max(c['position_m']) <= .005 and max(c['rotation_rad']) <= np.deg2rad(1)) for c in cases)}
        groups[str(t)] = g
    assert groups['all']['paired_loss_ratio']['median'] == tr['world_alignment']['median_paired_loss_ratio']
    raw = {}
    for method in s['methods']:
        records = [read_json(p) for p in sorted((OUT/'proposal_diagnostics').glob(f'*/{method}/raw.json'))]
        raw[method] = {'count': len(records), 'eligible': sum(r['eligible'] for r in records),
                      'position_gate': sum(max(r['validation']['frozen_goal_position_error_m']) <= .005 for r in records),
                      'rotation_gate': sum(int(max(r['validation']['frozen_goal_rotation_error_rad']) <= np.deg2rad(1)) for r in records),
                      'worst_arm_position_mm': stats([1000*max(r['validation']['frozen_goal_position_error_m']) for r in records]),
                      'worst_arm_rotation_deg': stats([np.rad2deg(max(r['validation']['frozen_goal_rotation_error_rad'])) for r in records])}
    proofs = [read_json(p) for p in sorted((OUT/'world_gradients').glob('*.json'))]
    assert len(proofs) == 120 and all(p['status'] == 'PASS' for p in proofs)
    grads = tr['world_records']
    gradient = {'events': len(grads), 'noise_counts': dict(Counter(g['noise_t'] for g in grads)),
                'sample_counts': dict(Counter(g['sample_id'] for g in grads)),
                'task_counts': dict(Counter(p['task_hash'] for p in proofs)),
                'true_loss': stats([g['true_world_loss'] for g in grads]),
                'latent_gradient_l2': stats([g['gradient_l2'] for g in grads]),
                'combined_parameter_gradient_before_clip': stats([g['parameter_gradient_norm'] for g in grads]),
                'combined_gradient_clipped_events': sum(g['parameter_gradient_norm'] > 1 for g in grads),
                'maximum_half_relative_error': max(h['relative_error'] for p in proofs for h in p['half_checks']),
                'maximum_direction_relative_error': max(p['direction_relative_error'] for p in proofs),
                'maximum_256_512_position_difference_m': max(max(p['convergence']['position_m']) for p in proofs),
                'maximum_256_512_rotation_difference_rad': max(max(p['convergence']['rotation_rad']) for p in proofs),
                'scope': 'combined parameter norm includes both vMSE and world loss; cannot infer their separate norm or angle'}
    records = [read_json(p) for p in sorted((OUT/'records').glob('*.json'))]
    checked_arrays = []
    for row in records:
        array_path = OUT/row['arrays']
        state_path = OUT/row['complete_initial_state']
        assert file_digest(array_path) == row['arrays_sha256']
        assert file_digest(state_path) == row['initial_state_sha256']
        with np.load(array_path) as saved:
            ref = Reference(saved['Z'], row['condition']['initial_features'], row['condition']['duration'])
            expected = ref.arrays()
            assert ref.reference_hash == row['reference_hash']
            assert set(saved.files) == set(expected)
            assert all(np.array_equal(saved[k], expected[k]) for k in expected)
        with np.load(state_path) as state:
            assert digest({'values':state['values'], 'model_hash':row['task']['contract']['robot_hash'],
                           'state_mask':int(STATE)}) == row['task']['contract']['initial_state_id']
        checked_arrays.append({'sample_id':row['sample_id'], 'all_spline_arrays_identical':True,
                               'complete_initial_state_hash_verified':True})
    new_ids = tr['metadata']['new_ids']
    index = read_json(OUT/'training_index.json')['records']
    candidate_paths = sorted((OUT/'candidates').glob('*/*/*/*.json'))
    ddim_residuals = [step['decomposition_residual_after'] for p in candidate_paths for step in read_json(p).get('sampling', {}).get('steps', [])]
    result = {'posthoc_only': True, 'new_physics_calls': 0, 'new_network_calls': 0,
              'source_summary_sha256': file_digest(OUT/'summary.json'), 'elapsed_s': read_json(OUT/'status.json')['elapsed_s'],
              'training_wall_s': tr['wall_s'], 'TRAIN_noise_probes': groups, 'raw_proposals': raw,
              'gradient': gradient, 'maximum_DDIM_decomposition_residual': max(ddim_residuals),
              'all_TRAIN_draws': stats(list(tr['train_draw_counts'].values())),
              'new_TRAIN_draws': stats([tr['train_draw_counts'][sid] for sid in new_ids]),
              'row_stream_draws': stats(list(tr['uniform_row_draw_counts'].values())),
              'origin_stream_draws': stats(list(tr['origin_stream_draw_counts'].values())),
              'new_grades': dict(Counter(r['validation_level'] for r in records)),
              'all_TRAIN_grades': dict(Counter(e['validation_level'] for e in index if e['split'] == 'TRAIN')),
              'new_maximum_goal_position_m': max(max(r['validation']['frozen_goal_position_error_m']) for r in records),
              'new_maximum_goal_rotation_rad': max(max(r['validation']['frozen_goal_rotation_error_rad']) for r in records),
              'new_minimum_distinct_control_rms_rad': min(r['minimum_reference_rms_rad'] for r in records),
              'trained_weight_hash': tr['trained_weight_hash'], 'checkpoint_sha256': tr['checkpoint_sha256'],
              'complete_array_audit': {'status':'PASS', 'records':len(checked_arrays),
                                      'all_saved_spline_views_recomputed':True, 'complete_initial_states_verified':True},
              'cache': read_json(OUT/'runtime_cache_audit.json')}
    DEST.mkdir(exist_ok=True)
    (DEST/'posthoc_diagnostics.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k not in ('gradient','cache')}, ensure_ascii=False, indent=2))
    print(json.dumps({k:v for k,v in gradient.items() if k not in ('sample_counts','task_counts')}, indent=2))


if __name__ == '__main__':
    main()
