"""Read-only quantitative summary of the pinned, finite shadow experiment."""
from pathlib import Path
import hashlib
import json

ROOT = next(p for p in Path(__file__).resolve().parents if (p / '.git').exists())
SOURCE = ROOT / 'v6_lite/output/runs/research_inertia_shadow_resumed_01'
OUT = Path(__file__).resolve().parent
PIN = 'ee98249e6ba936278e005943dc942724e57fb5ef2f59d2fe226b0e101d89158e'


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def read(p):
    return json.loads(p.read_text(encoding='utf-8'))


def main():
    assert sha(SOURCE / 'manifest.json') == PIN
    manifest = read(SOURCE / 'manifest.json')
    assert all(sha(SOURCE / k) == v for k, v in manifest.items())
    r = read(SOURCE / 'report.json')
    assert r['complete'] and r['evidence_valid'] and len(r['runs']) == 15
    rows = []
    for run in r['runs']:
        k = run['fresh_kinematics']['original_phase_window_diagnostic_statistics']
        rows.append({
            'scene': run['scenario_index'], 'alpha': run['alpha'],
            'target_min_mm': 1000 * run['robot_target_500hz']['minimum_clearance_m'],
            'whole_min_mm': 1000 * run['whole_body_dense_discrete']['minimum_clearance_m'],
            'continuum_path_max_mm': 1000 * k['continuum_original_path_window_error_m']['max'],
            'rigid_final_mm': 1000 * k['rigid_final_position_error_m'],
            'rigid_last_1_5s_max_mm': 1000 * k['rigid_original_last_1_5s_position_error_m']['max'],
            'target_below_5mm': run['robot_target_500hz']['below_required_clearance_state_count'],
            'whole_below_5mm': run['whole_body_dense_discrete']['below_required_clearance_state_count'],
            'target_negative': run['robot_target_500hz']['negative_native_signed_state_count'],
            'whole_negative': run['whole_body_dense_discrete']['negative_native_signed_state_count'],
            'target_queries': run['robot_target_500hz']['query_count'],
            'whole_queries': run['whole_body_dense_discrete']['query_count'],
            'target_min_is_censored': run['robot_target_500hz']['minimum_is_censored_lower_bound'],
            'whole_min_is_censored': run['whole_body_dense_discrete']['minimum_is_censored_lower_bound'],
        })
    by_alpha = []
    for alpha in (1., .95, 1.05):
        a = [x for x in rows if x['alpha'] == alpha]
        assert len(a) == 5
        by_alpha.append({'alpha': alpha, 'scenes': 5,
                         **{k: min(x[k] for x in a) for k in ('target_min_mm', 'whole_min_mm')},
                         **{k: max(x[k] for x in a) for k in ('continuum_path_max_mm', 'rigid_final_mm', 'rigid_last_1_5s_max_mm')},
                         **{k: sum(x[k] for x in a) for k in ('target_below_5mm', 'whole_below_5mm', 'target_negative', 'whole_negative')}})
    paired = r['paired_vs_nominal']
    assert len(paired) == 10
    delta_fields = ('low_level_joint_delta_rad', 'base_translation_delta_m', 'base_orientation_delta_deg',
                    'rigid_tip_delta_m', 'continuum_tip_delta_m', 'rigid_orientation_delta_deg',
                    'continuum_orientation_delta_deg')
    extrema = {k: max(x[k]['max'] for x in paired) for k in delta_fields}
    report = {
        'schema': 'finite_inertia_shadow_quantitative_summary_v1',
        'source_manifest_sha256': PIN, 'producer_commit': '19ad85c7db81a2e254fd176077b6aae39333a439',
        'source_file_hashes_valid': True, 'run_count': 15, 'physics_steps': r['physics_steps_completed'],
        'rows': rows, 'by_alpha': by_alpha, 'max_paired_response': extrema,
        'total_target_queries': sum(x['target_queries'] for x in rows),
        'total_whole_queries': sum(x['whole_queries'] for x in rows),
        'exact_paired_target_clearance_delta_min_m': min(x['paired_exact_target_clearance_delta_m']['min'] for x in paired),
        'exact_paired_target_clearance_delta_max_m': max(x['paired_exact_target_clearance_delta_m']['max'] for x in paired),
        'paired_exact_clearance_counts': [x['paired_exact_clearance_count'] for x in paired],
        'paired_censored_clearance_excluded_counts': [x['paired_censored_clearance_count_excluded_from_exact_delta'] for x in paired],
        'scope': 'Summary of saved full observations, not an independent replay or a new acceptance decision.',
        'limitations': r['limitations'],
    }
    (OUT / 'summary.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8', newline='\n')
    own = {p.name: sha(p) for p in sorted(OUT.iterdir()) if p.is_file() and p.name != 'manifest.json'}
    (OUT / 'manifest.json').write_text(json.dumps(own, indent=2) + '\n', encoding='utf-8', newline='\n')
    print(json.dumps({k: v for k, v in report.items() if k not in ('rows', 'limitations')}, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
