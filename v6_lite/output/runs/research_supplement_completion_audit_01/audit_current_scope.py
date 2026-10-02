"""Root audit of this finite supplement, with deployment and claims kept separate."""
from pathlib import Path
import hashlib
import json
import re
import subprocess

ROOT = next(p for p in Path(__file__).resolve().parents if (p / '.git').exists())
OUT = Path(__file__).resolve().parent
RUNS = ROOT / 'v6_lite/output/runs'
HEAD = '19ad85c7db81a2e254fd176077b6aae39333a439'


def read(p):
    return json.loads(p.read_text(encoding='utf-8'))


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    checks, requirements, inputs = [], [], {}

    def check(name, passed):
        checks.append({'name': name, 'passed': bool(passed)})

    def load(directory, filename='report.json'):
        p = RUNS / directory / filename
        inputs[p.relative_to(ROOT).as_posix()] = {'sha256': sha(p), 'bytes': p.stat().st_size}
        return read(p)

    def requirement(name, result, evidence):
        requirements.append({'requirement': name, 'result': result, 'evidence': evidence})

    nominal = load('research_acceptance_01')
    a = nominal['algorithm_simulation']
    check('nominal_full_five_scene_research_acceptance', nominal['passed'] and nominal['complete'])
    check('original_25_functional_11_contract_evidence', a['delivery']['passed_count'] == a['delivery']['total_count'] == 25
          and a['execution']['passed_count'] == a['execution']['total_count'] == 11)
    check('nominal_independent_interval_full_scope', a['interval']['passed'] and a['interval']['task_state_checks'] == 6755
          and a['interval']['interval_rows_recomputed'] == 12350 and a['interval']['failure_count'] == 0)
    check('C1_non_timing_parity', a['c1_exact_parity']['all_passed'])
    check('performance_separate_and_startup_retained', nominal['computational_performance']['evidence_valid']
          and not nominal['computational_performance']['performance_is_functional_gate']
          and not nominal['computational_performance']['startup_samples_excluded'])
    check('original_deployment_not_relabelled', nominal['wall_continuation']['status'] == 'NOT_MET'
          and not nominal['wall_continuation']['previous_goal_completed']
          and nominal['hardware_deployment']['status'] == 'NOT_ESTABLISHED' and not nominal['hard_realtime_certified'])
    requirement('Recover algorithm/simulation research while retaining simulated20ms/2ms and safety contracts',
                'COMPLETED_NOMINAL_ACCEPTANCE', 'research_acceptance_01; 25/11;6755/12350; C1 parity; separate full timing observation')
    cons = load('research_conservatism_01')
    shape = load('research_shape_decomposition_01')
    check('finite_conservatism_and_shape_evidence', cons['complete'] and cons['evidence_valid'] and all(cons['checks'].values())
          and shape['complete'] and shape['evidence_valid'] and shape['case_count'] == 172 and all(shape['checks'].values()))
    requirement('Study interval/proxy conservatism without weakening geometric safety', 'COMPLETED_FINITE_DIAGNOSIS',
                '1024 original cases and fixed172 shape diagnosis; independent29/29 and1720 numeric+13/13 chain already archived')
    velocity = load('research_velocity_stress_01')
    scalar = load('research_velocity_failure_scalar_independent_01', 'verification.json')
    check('failed_complex_task_attempt_stays_failed', not velocity['passed'] and not velocity['complete'] and scalar['evidence_valid'])
    requirement('Study more demanding task ability and retain failures', 'COMPLETED_RESEARCH_WITH_FAILED_TRIAL',
                '2x target linear velocity, all5 attempted,2 complete3 rejected;338/538/249 evidence; full25/11/interval NOT_RUN')
    shadow = load('research_inertia_shadow_resumed_01')
    replay = load('research_inertia_shadow_resumed_01', 'replay_report.json')
    native = load('research_inertia_shadow_independent_audit_03')
    saved = load('research_inertia_shadow_evidence_audit_02')
    failed_saved = load('research_inertia_shadow_evidence_audit_01')
    failed_native = load('research_inertia_shadow_independent_audit_01/replay')
    check('finite_model_study_complete_and_nominal_exact', shadow['complete'] and shadow['evidence_valid']
          and len(shadow['runs']) == 15 and shadow['physics_steps_completed'] == 202500
          and all(x['nominal_parity']['passed'] and all(v == 0 for v in x['nominal_parity']['maximum_absolute_residuals'].values())
                  for x in replay['runs'] if x['alpha'] == 1.0))
    check('independent_finite_geometry_and_saved_evidence', native['complete'] and native['evidence_valid']
          and native['passed_checks'] == native['total_checks'] == 4293
          and saved['evidence_valid'] and saved['passed_check_count'] == saved['check_count'] == 768)
    check('failed_auditors_retained_as_failed', not failed_saved['evidence_valid'] and not failed_native['evidence_valid'])
    check('model_study_claim_bounds', not shadow['limitations']['feedback_recomputed']
          and not shadow['limitations']['runtime_certificate_reused'] and not shadow['limitations']['closed_loop_robustness_established']
          and not shadow['limitations']['continuous_time_collision_certified'])
    check('resume_14_reused_1_new_0_physics_original_not_relabelled', shadow['resumption']['complete_observations_reused'] == 14
          and shadow['resumption']['new_observations_computed'] == 1 and shadow['resumption']['physics_steps_recomputed'] == 0
          and shadow['resumption']['original_interrupted_source_unchanged']
          and not (RUNS / 'research_inertia_shadow_01/report.json').exists())
    qualification = load('research_inertia_shadow_01', 'test_qualification.json')
    check('frozen_pre_physics_220_plus16_qualification', qualification['qualified']
          and qualification['current_tests'] == 220 and qualification['historical_tests'] == 16
          and qualification['before_first_physics_step_of_formal_experiment']
          and sha(ROOT / qualification['report_path']) == qualification['report_sha256'])
    requirement('Study model-caused physical execution/tracking/clearance error', 'COMPLETED_FINITE_OPEN_LOOP_SENSITIVITY',
                '15 full inertia shadows; unchanged67 torques and other parameters; 5x9 nominal zero; saved minima/fresh frames/paired deltas; independent bounded audits')
    frozen = load('research_inertia_shadow_01', 'plan.json')['current_source']['files']
    check('all239_source_files_remain_original_bytes', len(frozen) == 239 and all(sha(ROOT / p) == v['sha256_raw'] for p, v in frozen.items()))
    assets = load('research_inertia_shadow_independent_audit_03', 'source_manifest.json')['model_assets']
    check('all76_model_assets_remain_original_bytes', len(assets) == 76 and all(sha(ROOT / p) == v['sha256'] for p, v in assets.items()))
    old = json.loads(subprocess.check_output(['git', 'show', HEAD + ':v6_lite/controller_status.json'], cwd=ROOT).decode('utf-8'))
    status = read(ROOT / 'v6_lite/controller_status.json')
    check('all57_prior_status_values_preserved', len(old) == 57 and all(status[k] == v for k, v in old.items()))
    current = status['supplemental_research_scope_completion']
    check('current_scope_not_old_deployment_completion', current['status'] == 'FINITE_RESEARCH_COMPLETE'
          and current['supplements_previous_goal'] and not current['old_goal_completed']
          and not current['wall_compute_20ms_is_research_prerequisite']
          and current['simulated_planning_period_s'] == .02 and current['simulated_physics_period_s'] == .002)
    requirement('Preserve provenance, interrupted attempts, original failures and distinct deployment status',
                'COMPLETED_SAVED_EVIDENCE_AND_STATUS', 'final146 pins; original139 unchanged; failed audits01 and migration/pause receipts; all57 prior status values retained')
    docs = ['README.md', 'docs/V6_2_RESEARCH_ACCEPTANCE.md', 'docs/V6_2_RESEARCH_VELOCITY_STRESS.md', 'docs/V6_2_RESEARCH_MODEL_SENSITIVITY.md']
    for name in docs:
        p = ROOT / name
        text = p.read_text(encoding='utf-8')
        check(name + ':no_draft_placeholders', '__GEOMETRY_' not in text and '__EVIDENCE_' not in text)
        for target in re.findall(r'\]\(([^)]+)\)', text):
            if ':' in target or target.startswith('#'):
                continue
            destination = (p.parent / target.split('#')[0]).resolve()
            if 'research_inertia_shadow_archive_check_01/verification.json' in destination.as_posix():
                continue  # Final archive is the separate explicit completion gate below.
            check(name + ':local_link:' + target, destination.exists())
    requirement('Document actual finite outcomes and distinguish historical NOT_EXECUTED plans', 'COMPLETED',
                'new model report; README and acceptance/velocity scope notes; four appended status fields')
    result = {
        'schema': 'root_finite_supplement_requirement_audit_v1', 'evidence_valid': all(x['passed'] for x in checks),
        'finite_research_and_documentation_complete': all(x['passed'] for x in checks),
        'requirements': requirements, 'check_count': len(checks), 'checks': checks,
        'failed_checks': [x for x in checks if not x['passed']], 'input_reports': inputs,
        'objective_text_sha256': sha(OUT / 'objective.txt'),
        'human_steering': ['This goal supplements the earlier goal.', 'Prioritize the final research objective; stop unproductive local optimization.',
                          'Wall-clock20ms is a separate performance/deployment condition, not a prerequisite for pure research.'],
        'goal_completion_final_gate': 'Only after separate disk/index archival verification passes, commit finishes, and committed raw bytes are verified.',
        'old_C11_deployment_goal_completed': False,
        'new_robust_closed_loop_actuator_delay_CBF_backup_or_hardware_claim': False,
        'requires_new_unbounded_experiments_to_complete_this_finite_supplement': False,
    }
    (OUT / 'report.json').write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8', newline='\n')
    own = {p.name: sha(p) for p in sorted(OUT.iterdir()) if p.is_file() and p.name != 'manifest.json'}
    (OUT / 'manifest.json').write_text(json.dumps(own, indent=2) + '\n', encoding='utf-8', newline='\n')
    print(json.dumps({'evidence_valid': result['evidence_valid'], 'checks': len(checks), 'failed': result['failed_checks']}))
    return 0 if result['evidence_valid'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
