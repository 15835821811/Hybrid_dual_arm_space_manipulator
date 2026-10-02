"""Independent audit of saved evidence only: no producer import or MuJoCo calls."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import sys

import numpy as np

AUDIT = Path(__file__).resolve().parent
ROOT = next(p for p in AUDIT.parents if (p / '.git').exists() and (p / 'model_test').is_dir())
TRIAL = ROOT / 'v6_lite/output/runs/research_inertia_shadow_resumed_01'
OLD = ROOT / 'v6_lite/output/runs/research_inertia_shadow_01'
BASELINE = ROOT / 'v6_lite/output/runs/research_acceptance_01'
HEAD = '19ad85c7db81a2e254fd176077b6aae39333a439'
EXPECTED = [(i, a, f'scene_{i:02d}_alpha_{a:.2f}') for a in (1., .95, 1.05) for i in range(5)]
REUSED = [x[2] for x in EXPECTED[:-1]]
CHECKS = []
RUN_SUMMARY = []
PAIR_SUMMARY = []


def read(p):
    return json.loads(p.read_text(encoding='utf-8'))


def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for part in iter(lambda: f.read(1024 * 1024), b''):
            h.update(part)
    return h.hexdigest()


def write(p, value):
    with p.open('x', encoding='utf-8', newline='\n') as f:
        f.write(json.dumps(value, ensure_ascii=True, indent=2, allow_nan=False) + '\n')


def check(name, value, detail=None):
    row = {'name': name, 'passed': bool(value)}
    if detail is not None:
        row['detail'] = detail
    CHECKS.append(row)


def inventory(directory):
    return {p.relative_to(directory).as_posix(): {'sha256': sha(p), 'bytes': p.stat().st_size}
            for p in sorted(directory.rglob('*')) if p.is_file()}


def load(p):
    with np.load(p, allow_pickle=False) as a:
        return {k: a[k].copy() for k in a.files}


def array_id(a):
    a = np.ascontiguousarray(a)
    return {'sha256_raw_c_order': hashlib.sha256(a.tobytes()).hexdigest(),
            'dtype': a.dtype.str, 'shape': list(a.shape)}


def stats(a):
    a = np.asarray(a)
    if not a.size or not np.all(np.isfinite(a)):
        raise ValueError('statistics requires nonempty finite values')
    return {'count': int(a.size), 'mean': float(a.mean()),
            'rmse': float(np.sqrt(np.mean(a * a))), 'p95': float(np.percentile(a, 95.)),
            'max': float(a.max()), 'min': float(a.min())}


def compare_stats(name, values, declared):
    computed = stats(values)
    # Same declared statistics, without rounding data or altering experiment gates.
    check(name, computed == declared, {'computed': computed, 'declared': declared})


def angle_deg(a, b):
    return np.rad2deg(np.arccos(np.clip((np.einsum('...ij,...ij->...', a, b) - 1.) / 2., -1., 1.)))


def provenance(name, p, source):
    check(name, p['capture_complete'] and p['git_commit_unchanged'] and p['source_unchanged']
          and p['before']['git_commit'] == HEAD and p['after']['git_commit'] == HEAD
          and p['before']['files'] == source['files'] and p['after']['files'] == source['files']
          and p['before']['tracked_worktree_dirty'] is False
          and p['after']['tracked_worktree_dirty'] is False
          and p['source_changes'] == {'added': [], 'removed': [], 'modified': []})


def main():
    audit_plan = read(AUDIT / 'plan.json')
    source = read(AUDIT / 'source_verification.json')
    check('auditor_script_pinned', sha(Path(__file__)) == audit_plan['auditor_script_sha256'])
    check('source_verification_pinned', sha(AUDIT / 'source_verification.json') == audit_plan['source_verification_sha256'])
    check('source_read_before_documentation_edits', source['passed'] and source['auditor_source_commit'] == HEAD
          and source['source_trial_commit'] == HEAD and not source['tracked_worktree_dirty']
          and source['source_inventory_sha256'] == source['expected_inventory_sha256'])
    plan = read(TRIAL / 'plan.json')
    report = read(TRIAL / 'report.json')
    replay = read(TRIAL / 'replay_report.json')
    manifest = read(TRIAL / 'manifest.json')
    check('final_manifest_pin', sha(TRIAL / 'manifest.json') == audit_plan['expected_final_manifest_sha256'])
    all_files = {p.relative_to(TRIAL).as_posix() for p in TRIAL.rglob('*') if p.is_file() and p.name != 'manifest.json'}
    check('final_manifest_exact_146_file_coverage', len(manifest) == 146 and set(manifest) == all_files)
    input_inventory = {}
    for filename, expected_hash in manifest.items():
        p = (TRIAL / filename).resolve()
        valid = p.is_relative_to(TRIAL) and p.is_file()
        actual = sha(p) if valid else None
        input_inventory[filename] = {'sha256': actual, 'expected_sha256': expected_hash,
                                     'bytes': p.stat().st_size if valid else None}
        check('manifest_file:' + filename, valid and actual == expected_hash)
    write(AUDIT / 'input_manifest_verification.json', input_inventory)
    old_inventory = read(TRIAL / 'interrupted_source_inventory.json')
    check('original_139_files_unchanged', len(old_inventory) == 139 and inventory(OLD) == old_inventory)
    check('original_paused_output_not_relabelled', not (OLD / 'report.json').exists() and not (OLD / 'manifest.json').exists())
    pause = read(OLD / 'goal_pause_receipt.json')
    check('original_exit1_retained', pause['observer_terminal_exit_code'] == 1
          and not pause['producer_terminal_report_exists'] and not pause['producer_final_manifest_exists']
          and not pause['full_diagnostic_acceptance_claimed'])
    check('pause_receipt_exact_copy', sha(OLD / 'goal_pause_receipt.json') == sha(TRIAL / 'original_goal_pause_receipt.json'))
    resumption = read(TRIAL / 'resumption_plan.json')
    check('resumption_14_plus_1_no_physics', resumption['inherited_complete_keys'] == REUSED
          and resumption['new_observation_keys'] == [EXPECTED[-1][2]]
          and resumption['physics_steps_replayed_by_resumption'] == 0
          and report['resumption']['complete_observations_reused'] == 14
          and report['resumption']['new_observations_computed'] == 1
          and report['resumption']['physics_steps_recomputed'] == 0
          and report['resumption']['original_interrupted_source_unchanged']
          and not report['resumption']['original_observer_terminal_report_existed'])
    check('resumption_script_pinned', sha(TRIAL / 'resume_observations.py') == resumption['resume_script_sha256'])
    replay_manifest = read(TRIAL / 'replay_manifest.json')
    required_replay = {'plan.json', 'producer.py', 'input_identities.json', 'replay_report.json'} | {
        f'replays/{key}/{name}' for _, _, key in EXPECTED
        for name in ('model_parameters.npz', 'model.json', 'trajectory.npz', 'replay.json')}
    check('replay_pin_and_exact_64_coverage', sha(TRIAL / 'replay_manifest.json') == audit_plan['expected_replay_manifest_sha256']
          == report['caller_pinned_replay_manifest_sha256'] == resumption['pinned_replay_manifest_sha256']
          and len(replay_manifest) == 64 and set(replay_manifest) == required_replay)
    for filename, expected_hash in replay_manifest.items():
        check('unchanged_replay_copy:' + filename, expected_hash == manifest[filename] == old_inventory[filename]['sha256'])
    for key in REUSED:
        for filename in ('observer_plan.json', 'robot_target_500hz.npz', 'whole_body_dense.npz', 'fresh_kinematics.npz', 'observation.json'):
            name = f'observations/{key}/{filename}'
            check('inherited_exact_copy:' + name, manifest[name] == old_inventory[name]['sha256'])
    check('no_partial_observation_reused', len(list((OLD / 'observations').glob('*/observation.json'))) == 14
          and not (OLD / 'observations' / EXPECTED[-1][2] / 'observation.json').exists())
    check('nominal_baseline_manifest_pin', sha(BASELINE / 'manifest.json') == plan['baseline_complete_manifest_sha256']
          == '379b9977a441c540a20cfc50a9e90614a1e0985fa2a5a73bbb7c9690f823eec0'
          and plan['baseline_source_commit'] == '9cd1831e7c77a340464a669eac3b5aa7cd9c3c3c')
    check('producer_frozen_source_pin', sha(TRIAL / 'producer.py') == plan['tool_sha256']
          == source['files']['v6_lite/run_research_inertia_shadow.py']['sha256_raw'])
    qualification = read(TRIAL / 'test_qualification.json')
    tests = read(ROOT / qualification['report_path'])
    check('original_220_plus_16_regression_qualification', qualification['qualified']
          and qualification['current_tests'] == 220 and qualification['historical_tests'] == 16
          and qualification['full_test_profile_passed'] and qualification['source_commit'] == HEAD
          and qualification['before_first_physics_step_of_formal_experiment']
          and sha(ROOT / qualification['report_path']) == qualification['report_sha256'] and tests['passed'])
    actual_replays = [(r['scenario_index'], r['alpha'], r['key']) for r in replay['runs']]
    actual_observations = [(r['scenario_index'], r['alpha'], r['key']) for r in report['runs']]
    check('15_exact_unique_ordered_runs', actual_replays == EXPECTED == actual_observations and len(set(EXPECTED)) == 15)
    check('full_diagnostic_terminal_status', report['complete'] and report['evidence_valid'] and report['status'] == 'DIAGNOSTIC_COMPLETE'
          and replay['complete'] and report['physics_steps_completed'] == replay['physics_steps_completed'] == 202500
          and plan['smoke_steps'] is None)
    provenance('final_source_provenance', report['source_provenance'], plan['current_source'])
    false_claims = ('feedback_recomputed', 'new_qp_solved', 'runtime_executor_used', 'runtime_certificate_reused',
                    'closed_loop_robustness_established', 'wall_deployment_certified', 'hardware_established',
                    'continuous_time_collision_certified', 'contact_response_enabled', 'shadow_interval_cbf_recomputed',
                    'shadow_calibrated_envelope_support_established', 'old_acceptance_modified')
    check('diagnostic_claim_limits', all(report['limitations'][k] is False for k in false_claims))
    baseline_manifest = read(BASELINE / 'manifest.json')
    baseline_manifest = {k.replace('\\', '/'): v for k, v in baseline_manifest.items()}
    traces = {}
    for i in range(5):
        path = BASELINE / f'simulation/traces/v6_lite_scenario_{i:02d}.npz'
        check('baseline_trace_hash:' + str(i), sha(path) == plan['inputs'][i]['trace_sha256']
              == baseline_manifest[path.relative_to(BASELINE).as_posix()])
        traces[i] = load(path)
    trajectory, kinematics, target_geometry = {}, {}, {}
    qids = base_adr = None
    for rr, row in zip(replay['runs'], report['runs']):
        i, alpha, key = row['scenario_index'], row['alpha'], row['key']
        replay_dir = TRIAL / 'replays' / key
        obs = TRIAL / 'observations' / key
        check(key + ':complete_steps', rr['status'] == 'REPLAY_COMPLETED' and rr['full_27s_complete']
              and rr['steps_completed'] == rr['planned_steps'] == 13500 and rr['error'] is None)
        if alpha == 1.:
            p = rr['nominal_parity']
            check(key + ':nominal_nine_zero_residuals', p['passed'] and p['full_27s_nominal_reproduction']
                  and p['tolerance'] == 1e-11 and len(p['maximum_absolute_residuals']) == 9
                  and all(x == 0. for x in p['maximum_absolute_residuals'].values()))
        tr = load(replay_dir / 'trajectory.npz')
        trajectory[key] = tr
        check(key + ':trajectory_raw_shapes_finite_clock', tr['qpos'].shape == (13501, 81)
              and tr['qvel'].shape == (13501, 79) and tr['time'].shape == (13501,)
              and tr['torque_input'].shape == (13500, 67)
              and all(np.all(np.isfinite(v)) for v in tr.values())
              and np.allclose(tr['time'], np.arange(13501) * .002, rtol=0., atol=1e-9))
        check(key + ':frozen_applied_input_and_initial_state', np.array_equal(tr['torque_input'], traces[i]['torque'])
              and array_id(tr['torque_input']) == plan['inputs'][i]['torque'] == rr['torque_input_identity']
              and np.array_equal(tr['qpos'][0], traces[i]['initial_qpos'])
              and np.array_equal(tr['qvel'][0], traces[i]['initial_qvel']))
        if alpha == 1.:
            check(key + ':nominal_task_qpos_exact', np.array_equal(tr['qpos'][::10], traces[i]['task_qpos']))
        if qids is None:
            model_record = read(replay_dir / 'model.json')
            with np.load(replay_dir / 'model_parameters.npz', allow_pickle=False) as a:
                scalar_jids = np.flatnonzero(np.isin(a['before_jnt_type'], [2, 3]))
                qids = a['before_jnt_qposadr'][scalar_jids]
                base_adr = int(a['before_jnt_qposadr'][model_record['selection']['root_joint_id']])
                check('independent_joint_address_order', len(qids) == 67
                      and np.array_equal(a['before_actuator_trnid'][:, 0], scalar_jids))
        observed = read(obs / 'observation.json')
        op = read(obs / 'observer_plan.json')
        check(key + ':saved_report_exact_terminal_row', observed == row and row['status'] == 'OBSERVATION_COMPLETED')
        provenance(key + ':observation_source', row['source_provenance'], plan['current_source'])
        pairs = op['pairs']
        target_pairs = [x for x in pairs if x['pair_class'].endswith('_target')]
        check(key + ':pair_plan_identity', len(pairs) == 2927 and len(target_pairs) == 75
              and len({(x['geom_a_id'], x['geom_b_id']) for x in pairs}) == 2927
              and op['input_trajectory_sha256'] == rr['trajectory_sha256'] == sha(replay_dir / 'trajectory.npz')
              and op['observer_model_id'] == plan['inputs'][i]['nominal_compiled_model_id']
              and op['pair_policy_sha256'] == row['pair_policy_sha256'] == plan['inputs'][i]['pair_policy_sha256']
              and op['source']['files'] == plan['current_source']['files'])
        summaries = {}
        for field, filename, count, query_pairs, spacing, distmax in (
            ('robot_target_500hz', 'robot_target_500hz.npz', 13501, target_pairs, 1., .25),
            ('whole_body_dense_discrete', 'whole_body_dense.npz', 5401, pairs, 2.5, 2.5)):
            g = row[field]
            a = load(obs / filename)
            d, c, ph, idx = (a[k] for k in ('minimum_m', 'minimum_censored_lower_bound', 'physics_grid_phase', 'minimum_pair_index'))
            check(key + ':' + field + ':raw_hash_complete_coverage', sha(obs / filename) == g['arrays_sha256']
                  and bool(a['complete']) and int(a['planned_state_count']) == int(a['completed_state_count']) == count
                  and int(a['completed_query_count']) == count * len(query_pairs)
                  and g['checked_state_count'] == count and g['pair_count'] == len(query_pairs)
                  and g['query_count'] == count * len(query_pairs)
                  and d.shape == c.shape == ph.shape == idx.shape == (count,)
                  and np.all(np.isfinite(d)) and np.array_equal(ph, np.arange(count) * spacing)
                  and np.all((idx >= 0) & (idx < len(query_pairs))) and c.dtype == np.dtype(bool)
                  and np.all(d[c] == distmax) and np.all(d[~c] < distmax - 1e-12)
                  and all(a[k].size == 0 for k in ('partial_state_qpos', 'partial_state_qvel', 'partial_state_pair_distance_m', 'partial_state_pair_censored_lower_bound')))
            minimum_at = int(np.argmin(d))
            check(key + ':' + field + ':minimum_state_statistics', g['minimum_clearance_m'] == float(d.min())
                  and g['minimum_is_censored_lower_bound'] == bool(c[minimum_at])
                  and g['below_required_clearance_state_count'] == int(np.sum(d < .005))
                  and g['negative_native_signed_state_count'] == int(np.sum(d < 0.))
                  and g['measured_discrete_clearance_at_least_gate'] == bool(d.min() >= .005)
                  and not g['continuous_time_certified'] and not g['nativeccd_claimed'])
            witness = g['minimum_pair']
            wi = witness['state_index']
            wp = query_pairs[int(idx[wi])]
            check(key + ':' + field + ':minimum_witness', d[wi] == d.min()
                  and witness['physics_grid_phase'] == float(ph[wi]) and witness['time_s'] == float(ph[wi] * .002)
                  and witness['native_signed_distance_or_lower_bound_m'] == float(d[wi])
                  and witness['censored_lower_bound'] == bool(c[wi])
                  and witness['geom_a'] == wp['geom_a'] and witness['geom_b'] == wp['geom_b']
                  and witness['pair_class'] == wp['pair_class'])
            below = np.flatnonzero(d < .005)
            first = g['first_below_required_clearance']
            check(key + ':' + field + ':first_below', first is None if not len(below) else
                  first['state_index'] == int(below[0]) and first['minimum_m'] == float(d[below[0]]))
            check(key + ':' + field + ':aggregate_bounds', int(c.sum()) <= g['truncated_query_count'] <= g['query_count']
                  and g['below_required_clearance_state_count'] <= g['violating_pair_sample_count'] <= g['query_count']
                  and g['negative_native_signed_state_count'] <= g['negative_native_signed_query_count'] <= g['violating_pair_sample_count']
                  and min(g['minimum_by_class_m'].values()) == g['minimum_clearance_m'])
            summaries[field] = {'states': count, 'queries': count * len(query_pairs), 'minimum_m': float(d.min()),
                                'below_5mm_states': int(np.sum(d < .005)), 'negative_states': int(np.sum(d < 0.)),
                                'censored_minimum_states': int(c.sum())}
            if field == 'robot_target_500hz':
                target_geometry[key] = a
        k = load(obs / 'fresh_kinematics.npz')
        kinematics[key] = k
        check(key + ':fresh_arrays_finite_grid', sha(obs / 'fresh_kinematics.npz') == row['fresh_kinematics']['arrays_sha256']
              and set(k) == {'time', 'rigid_tip', 'continuum_tip', 'rigid_rotation', 'continuum_rotation', 'rigid_target', 'rigid_target_rotation'}
              and all(v.shape[0] == 13501 and np.all(np.isfinite(v)) for v in k.values())
              and np.array_equal(k['time'], tr['time']))
        rerr = np.linalg.norm(k['rigid_tip'][1:] - k['rigid_target'][1:], axis=1)
        cerr = np.linalg.norm(k['continuum_tip'][1:] - traces[i]['continuum_target'], axis=1)
        tracking = row['fresh_kinematics']['tracking_statistics']
        for metric, data in [('rigid_position_error_m', rerr), ('continuum_position_error_m', cerr),
                             ('rigid_orientation_error_deg', angle_deg(k['rigid_rotation'][1:], k['rigid_target_rotation'][1:])),
                             ('continuum_orientation_error_deg', angle_deg(k['continuum_rotation'][1:], traces[i]['continuum_target_rotation']))]:
            compare_stats(key + ':' + metric, data, tracking[metric])
        path = plan['inputs'][i]['scenario']['continuum_target']
        mask = (k['time'][1:] >= path['path_start_s']) & (k['time'][1:] <= path['path_end_s'])
        steady = k['time'][1:] >= 25.5
        phases = row['fresh_kinematics']['original_phase_window_diagnostic_statistics']
        compare_stats(key + ':original_path_window', cerr[mask], phases['continuum_original_path_window_error_m'])
        compare_stats(key + ':original_steady_window', rerr[steady], phases['rigid_original_last_1_5s_position_error_m'])
        check(key + ':final_rigid_error_and_original_claim_boundary', phases['rigid_final_position_error_m'] == float(np.linalg.norm(k['rigid_tip'][-1] - k['rigid_target'][-1]))
              and not row['fresh_kinematics']['original_acceptance_relabelled'])
        summaries['phase_windows'] = {'path_declared_start_s': path['path_start_s'], 'path_declared_end_s': path['path_end_s'],
                                     'path_actual_clock_sample_count': int(mask.sum()), 'steady_declared_start_s': 25.5,
                                     'steady_actual_clock_sample_count': int(steady.sum()), 'initial_excluded_from_error_metrics': True}
        RUN_SUMMARY.append({'key': key, 'alpha': alpha, 'scenario_index': i, **summaries})
    expected_pairs = [(i, alpha) for i in range(5) for alpha in (.95, 1.05)]
    check('ten_exact_ordered_paired_runs', [(p['scenario_index'], p['alpha']) for p in report['paired_vs_nominal']] == expected_pairs)
    for pair in report['paired_vs_nominal']:
        i, alpha = pair['scenario_index'], pair['alpha']
        key = f'scene_{i:02d}_alpha_{alpha:.2f}'; nominal = f'scene_{i:02d}_alpha_1.00'
        q, nq = trajectory[key]['qpos'], trajectory[nominal]['qpos']
        k, nk = kinematics[key], kinematics[nominal]
        qquat, nquat = q[:, base_adr + 3:base_adr + 7], nq[:, base_adr + 3:base_adr + 7]
        dots = np.abs(np.sum(qquat * nquat, axis=1) / (np.linalg.norm(qquat, axis=1) * np.linalg.norm(nquat, axis=1)))
        for metric, values in [
            ('low_level_joint_delta_rad', np.abs(q[:, qids] - nq[:, qids])),
            ('base_translation_delta_m', np.linalg.norm(q[:, base_adr:base_adr + 3] - nq[:, base_adr:base_adr + 3], axis=1)),
            ('base_orientation_delta_deg', np.rad2deg(2 * np.arccos(np.clip(dots, 0., 1.)))),
            ('rigid_tip_delta_m', np.linalg.norm(k['rigid_tip'] - nk['rigid_tip'], axis=1)),
            ('continuum_tip_delta_m', np.linalg.norm(k['continuum_tip'] - nk['continuum_tip'], axis=1)),
            ('rigid_orientation_delta_deg', angle_deg(k['rigid_rotation'], nk['rigid_rotation'])),
            ('continuum_orientation_delta_deg', angle_deg(k['continuum_rotation'], nk['continuum_rotation']))]:
            compare_stats(key + ':paired:' + metric, values, pair[metric])
        a, b = target_geometry[key], target_geometry[nominal]
        exact = ~(a['minimum_censored_lower_bound'] | b['minimum_censored_lower_bound'])
        exact_count, excluded_count = int(exact.sum()), int((~exact).sum())
        check(key + ':exact_censored_partition', exact_count + excluded_count == 13501
              and pair['paired_exact_clearance_count'] == exact_count
              and pair['paired_censored_clearance_count_excluded_from_exact_delta'] == excluded_count)
        if exact_count:
            compare_stats(key + ':paired:exact_clearance_only', a['minimum_m'][exact] - b['minimum_m'][exact], pair['paired_exact_target_clearance_delta_m'])
        else:
            check(key + ':no_exact_delta_without_uncensored_pair', pair['paired_exact_target_clearance_delta_m'] is None)
        PAIR_SUMMARY.append({'scenario_index': i, 'alpha': alpha, 'exact_clearance_count': exact_count,
                             'excluded_censored_count': excluded_count, 'delta_samples_include_initial': True})
    check('total_declared_query_coverage', sum(x['robot_target_500hz']['queries'] for x in RUN_SUMMARY) == 15188625
          and sum(x['whole_body_dense_discrete']['queries'] for x in RUN_SUMMARY) == 237130905)


if __name__ == '__main__':
    error = None
    try:
        main()
    except BaseException as e:
        error = {'type': type(e).__name__, 'message': str(e)}
        check('auditor_completed_without_exception', False, error)
    failures = [x for x in CHECKS if not x['passed']]
    audit_plan = read(AUDIT / 'plan.json')
    result = {'schema': 'inertia_shadow_saved_evidence_audit_v1', 'completed_utc': datetime.now(timezone.utc).isoformat(),
              'evidence_valid': not failures and error is None, 'audit_completed': error is None,
              'source_trial_commit': HEAD, 'auditor_source_commit': audit_plan['auditor_source_commit'],
              'auditor_script_sha256': sha(Path(__file__)), 'producer_final_manifest_sha256': sha(TRIAL / 'manifest.json'),
              'check_count': len(CHECKS), 'passed_check_count': len(CHECKS) - len(failures), 'failures': failures,
              'checks': CHECKS, 'runs': RUN_SUMMARY, 'paired_runs': PAIR_SUMMARY,
              'prior_audit_tool_error': audit_plan.get('prior_audit_tool_error'),
              'limits': {'producer_imported': False, 'physics_steps_recomputed': 0, 'geometry_queries_recomputed': 0,
                         'all_pair_distance_values_independently_recomputed': False,
                         'raw_pair_values_not_saved': 'Only per-state minima and aggregate query counts are stored; full-pair totals rely on frozen loop implementation plus completion counters.',
                         'trial_pass_established': False, 'closed_loop_robustness_established': False,
                         'continuous_time_collision_certified': False, 'wall_deployment_certified': False,
                         'hardware_established': False, 'original_acceptance_relabelled': False}}
    write(AUDIT / 'report.json', result)
    write(AUDIT / 'manifest.json', {p.name: sha(p) for p in sorted(AUDIT.iterdir()) if p.is_file() and p.name != 'manifest.json'})
    print(json.dumps({'evidence_valid': result['evidence_valid'], 'check_count': len(CHECKS),
                      'passed_check_count': result['passed_check_count'], 'failures': failures}, ensure_ascii=True))
    raise SystemExit(0 if result['evidence_valid'] else 1)
