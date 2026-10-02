"""Append actual finite-study results while preserving all historical status values."""
from pathlib import Path
import hashlib
import json
import subprocess

ROOT = next(p for p in Path(__file__).resolve().parents if (p / '.git').exists())
HEAD = '19ad85c7db81a2e254fd176077b6aae39333a439'
BASE = ROOT / 'v6_lite/output/runs'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    path = ROOT / 'v6_lite/controller_status.json'
    old = json.loads(subprocess.check_output(['git', 'show', HEAD + ':v6_lite/controller_status.json'], cwd=ROOT).decode('utf-8'))
    current = read(path)
    assert len(old) == 57 and current == old, 'status must start from preserved 57-field snapshot'
    shadow = read(BASE / 'research_inertia_shadow_resumed_01/report.json')
    native = read(BASE / 'research_inertia_shadow_independent_audit_03/report.json')
    saved = read(BASE / 'research_inertia_shadow_evidence_audit_02/report.json')
    summary = read(BASE / 'research_inertia_shadow_analysis_01/summary.json')
    qualification = read(BASE / 'research_inertia_shadow_01/test_qualification.json')
    assert shadow['complete'] and shadow['evidence_valid'] and len(shadow['runs']) == 15
    assert native['complete'] and native['evidence_valid'] and native['total_checks'] == 4293
    assert saved['evidence_valid'] and saved['audit_completed'] and saved['check_count'] == 768
    assert qualification['qualified'] and qualification['current_tests'] == 220 and qualification['historical_tests'] == 16
    current['completed_research_model_sensitivity'] = {
        'status': 'DIAGNOSTIC_COMPLETE', 'complete': True, 'evidence_valid': True,
        'source_commit': HEAD,
        'kind': 'uncertified controller-free same-torque numerical open-loop shadow',
        'factor': 'robot_subtree_principal_body_inertia_scale', 'factor_values': [0.95, 1.0, 1.05],
        'robot_subtree_body_count': 73, 'scenario_count': 5, 'full_runs': 15,
        'duration_per_run_s': 27.0, 'physics_period_s': 0.002, 'physics_steps_completed': 202500,
        'source_python_files': 239, 'model_asset_files': 76,
        'report': 'v6_lite/output/runs/research_inertia_shadow_resumed_01/report.json',
        'manifest': 'v6_lite/output/runs/research_inertia_shadow_resumed_01/manifest.json',
        'manifest_sha256': sha(BASE / 'research_inertia_shadow_resumed_01/manifest.json'),
        'nominal_full_runs': 5, 'nominal_parity_fields_per_scene': 9,
        'nominal_parity_maximum_absolute_residual': 0.0,
        'original_torque_channels': 67, 'torque_input_changed': False,
        'mass_COM_armature_geometry_actuators_options_changed': False,
        'fresh_current_qpos_observer': True,
        'robot_target_states_per_run': 13501, 'robot_target_pairs': 75,
        'whole_body_discrete_configuration_states_per_run': 5401, 'whole_body_pairs': 2927,
        'robot_target_query_count': 15188625, 'whole_body_query_count': 237130905,
        'whole_body_all_pairs_at_every_2ms_state': False,
        'minimum_measured_clearance_m': 0.009032701852940889,
        'saved_below_5mm_state_count': 0, 'saved_native_negative_state_count': 0,
        'finite_sensitivity_results_by_alpha': summary['by_alpha'],
        'max_paired_response': summary['max_paired_response'],
        'paired_exact_clearance_state_count': 131555, 'paired_censored_excluded_state_count': 3455,
        'original_interrupted_directory': 'v6_lite/output/runs/research_inertia_shadow_01',
        'original_observer_terminal_exit_code': 1,
        'original_interrupted_139_files_preserved': True,
        'resumption_observer_terminal_exit_code': 0,
        'resumption_complete_observations_reused': 14, 'resumption_new_observations': 1,
        'resumption_physics_steps_recomputed': 0,
        'independent_audit': 'v6_lite/output/runs/research_inertia_shadow_independent_audit_03/report.json',
        'independent_replay_checks': 2395, 'independent_geometry_checks': 1898,
        'independent_nominal_physics_prefix_steps': 100,
        'independent_native_geometry_queries': 254045,
        'independent_full_15_run_physics_replay': False,
        'independent_full_geometry_recalculation': False,
        'saved_evidence_audit': 'v6_lite/output/runs/research_inertia_shadow_evidence_audit_02/report.json',
        'saved_evidence_checks': 768,
        'retained_failed_auditors': [
            'v6_lite/output/runs/research_inertia_shadow_independent_audit_01/replay/report.json',
            'v6_lite/output/runs/research_inertia_shadow_evidence_audit_01/report.json'],
        'regression_qualification': 'v6_lite/output/runs/research_inertia_shadow_01/test_qualification.json',
        'document': 'docs/V6_2_RESEARCH_MODEL_SENSITIVITY.md',
        'archive_verification': 'v6_lite/output/runs/research_inertia_shadow_archive_check_01/verification.json',
        'new_25_11_interval_closed_loop_acceptance': False,
        **shadow['limitations'],
    }
    current['research_model_sensitivity_protocol_record_scope'] = (
        'next_model_sensitivity_research_protocol is the preserved historical NOT_EXECUTED pre-execution declaration; '
        'actual completed finite protocol is completed_research_model_sensitivity. '
        'Earlier current_next_research_protocol fields remain historical snapshots, not new pending experiments.')
    current['research_model_sensitivity_regression'] = {
        'source_commit': HEAD, 'current_passed': 220, 'current_total': 220,
        'historical_passed': 16, 'historical_total': 16, 'source_unchanged': True,
        'before_formal_first_physics_step': True,
        'report': 'v6_lite/output/runs/research_inertia_shadow_tests_01/report.json',
    }
    current['supplemental_research_scope_completion'] = {
        'status': 'FINITE_RESEARCH_COMPLETE', 'supplements_previous_goal': True,
        'wall_compute_20ms_is_research_prerequisite': False,
        'simulated_planning_period_s': 0.02, 'simulated_physics_period_s': 0.002,
        'original_safety_QP_execution_contracts_preserved': True,
        'nominal_research_acceptance': 'PASSED',
        'conservatism_and_shape_diagnosis': 'COMPLETED',
        'twofold_velocity_research': 'ALL_FIVE_ATTEMPTED_TRIAL_FAILED_INCOMPLETE',
        'model_caused_execution_sensitivity': 'DIAGNOSTIC_COMPLETE',
        'finite_research_completion_requires_all_stress_trials_pass': False,
        'current_scope_requires_more_unbounded_experiments': False,
        'old_C11_wall_deployment': 'NOT_MET', 'administrator_branch': 'DEFERRED',
        'administrator_platform_condition': 'BLOCKED', 'hardware_safety': 'NOT_ESTABLISHED',
        'hard_real_time_guarantee': 'NOT_ESTABLISHED',
        'old_goal_completed': False,
        'further_actuator_error_delay_robust_CBF_backup_studies_established': False,
        'root_requirement_audit': 'v6_lite/output/runs/research_supplement_completion_audit_01/report.json',
        'goal_mark_complete_only_after_raw_archive_and_committed_byte_verification': True,
    }
    assert all(current[k] == v for k, v in old.items())
    path.write_text(json.dumps(current, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8', newline='\n')
    print(json.dumps({'preserved_historical_root_fields': len(old), 'appended_root_fields': len(current) - len(old)}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
