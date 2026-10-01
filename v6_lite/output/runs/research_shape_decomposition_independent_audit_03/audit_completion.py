"""Final independent evidence reconciliation after retained auditor errors.

Numerical independence is provided by audit02: independent OBB formulas,
URDF FK, PCC matrix exponentials, all 61 candidates, and 17x5 bounds for
the fixed 172 states. This script verifies its complete immutable record
and corrects only the source_changes empty-dictionary schema test. It does
not repeat numerical queries/physics or change any prior artifact.
"""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import traceback

ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent
TRIAL = ROOT/'v6_lite/output/runs/research_shape_decomposition_01'
OLD = ROOT/'v6_lite/output/runs/research_conservatism_01'
AUDITS = [ROOT/f'v6_lite/output/runs/research_shape_decomposition_independent_audit_0{i}' for i in (1, 2)]
COMMIT = '32c3b04764669402f06d4a843b4ef36c81138908'
INPUT_SHA = 'ec620f3396bc06f98597e76270501a1bd7a88a941e5740239bb10d775d3efe8c'
IDS_SHA = '5d528aa0b01942484d9f9d0375ffe4c79c1f3c6c016b8c2cc9303af15e50c243'
CONTRACT_SHA = 'd3228423e4352d29025631c6d34663d95b6b20ee2c70a7e32db58d2076701db0'
CHECKS = []


def allfinite(v):
    if isinstance(v, dict):
        return all(allfinite(x) for x in v.values())
    if isinstance(v, list):
        return all(allfinite(x) for x in v)
    return not isinstance(v, (int, float)) or isinstance(v, bool) or math.isfinite(v)


def read(p):
    v = json.loads(Path(p).read_text(encoding='utf-8'),
                   parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))
    if not allfinite(v):
        raise ValueError('nonfinite saved JSON: '+str(p))
    return v


def rows(p):
    vs = [json.loads(x, parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))
          for x in Path(p).read_text(encoding='utf-8').splitlines()]
    if not allfinite(vs):
        raise ValueError('nonfinite saved rows: '+str(p))
    return vs


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write(p, v):
    with Path(p).open('x', encoding='utf-8', newline='\n') as f:
        json.dump(v, f, ensure_ascii=True, indent=2, allow_nan=False)
        f.write('\n')


def check(name, passed, detail=None):
    CHECKS.append({'name': name, 'passed': bool(passed), 'detail': detail})


def near(a, b):
    return isinstance(a, (int, float)) and not isinstance(a, bool) and isinstance(b, (int, float)) and not isinstance(b, bool) and math.isfinite(a) and math.isfinite(b) and abs(a-b) <= 1e-12


def empty_changes(v):
    return isinstance(v, dict) and set(v) == {'added', 'removed', 'modified'} and all(isinstance(x, list) and len(x) == 0 for x in v.values())


def verify_manifest(directory, name='artifact_manifest.json'):
    m = read(directory/name)
    entries = {x['path'].replace('\\', '/'): x for x in m['artifacts']}
    files = {p.relative_to(directory).as_posix() for p in directory.rglob('*') if p.is_file() and p.name != name}
    return len(entries) == len(m['artifacts']) and set(entries) == files and all(
        sha(directory/k) == x['sha256'] and (directory/k).stat().st_size == x['bytes'] for k, x in entries.items())


def run():
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    tracked = subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no'], cwd=ROOT, text=True).strip()
    plan, produced, provenance = [read(TRIAL/k) for k in ('plan.json', 'report.json', 'source_provenance.json')]
    final_numeric_report = read(AUDITS[1]/'report.json')
    numeric_plan = read(AUDITS[1]/'audit_plan.json')
    case_checks, numeric = read(AUDITS[1]/'by_case_checks.json'), rows(AUDITS[1]/'case_numeric_checks.jsonl')
    ids = read(OLD/'selections.json')['proven_proxy_below_actual_safe_indices']
    original = rows(OLD/'baseline.jsonl')
    expected = [x['index'] for x in original if x['actual_mujoco_distance_m'] >= .005 and x['proxy_status'] == 'PROXY_CLEARANCE_BELOW_GATE']
    ids_hash = hashlib.sha256(json.dumps(ids, separators=(',', ':')).encode('ascii')).hexdigest()
    source = plan['source']
    inventory = {k: {'sha256': sha(ROOT/k), 'bytes': (ROOT/k).stat().st_size} for k in source['files']}
    references = []
    for directory in [TRIAL, OLD, *AUDITS]:
        for p in sorted(directory.iterdir()):
            if p.is_file():
                references.append({'path': p.relative_to(ROOT).as_posix(), 'sha256': sha(p), 'bytes': p.stat().st_size})
    input_path = ROOT/'v6_lite/output/v6_2_b1/formal_audit_r02/independent_heldout_inputs.npz'
    references.append({'path': input_path.relative_to(ROOT).as_posix(), 'sha256': sha(input_path), 'bytes': input_path.stat().st_size})
    write(OUT/'audit_plan.json', {'schema': 'independent_shape_final_reconciliation_plan_v1',
        'started_utc': datetime.now(timezone.utc).isoformat(), 'declared_before_final_checks': True,
        'auditor_source_commit': commit, 'source_trial_commit': COMMIT, 'auditor_script_sha256': sha(__file__),
        'indices_sha256': ids_hash, 'indices': ids, 'references': references,
        'numerical_evidence_source': 'research_shape_decomposition_independent_audit_02; 1720 independent numerical checks passed',
        'scope': 'immutable numerical record/source/manifest reconciliation only; no numerical/physics rerun'})
    check('strict_manifest_trial_old_and_both_retained_audits', all(verify_manifest(d) for d in [TRIAL, OLD, *AUDITS]))
    old_audit = read(AUDITS[0]/'report.json')
    check('retained_tool_failures_exactly_accounted', old_audit['evidence_valid'] is False
          and old_audit['checks_passed'] == 10 and old_audit['checks_count'] == 12
          and {x['name'] for x in old_audit['failed_checks']} == {'source_commit_snapshot_and_raw_hashes', 'all_172_case_numeric_checks'}
          and final_numeric_report['evidence_valid'] is False and final_numeric_report['checks_passed'] == 12
          and final_numeric_report['checks_count'] == 13
          and [x['name'] for x in final_numeric_report['failed_checks']] == ['source_commit_snapshot_and_raw_hashes'])
    check('all_source_identity_and_source_changes_schema', commit == COMMIT == produced['source_commit']
          == source['git_commit'] == provenance['before']['git_commit'] == provenance['after']['git_commit']
          == final_numeric_report['source_trial_commit'] == numeric_plan['source_trial_commit']
          == final_numeric_report['auditor_source_commit'] == numeric_plan['auditor_source_commit']
          and not tracked and source['tracked_worktree_dirty'] is False and not source['capture_errors']
          and not provenance['before']['capture_errors'] and not provenance['after']['capture_errors']
          and provenance['capture_complete'] is True and provenance['source_unchanged'] is True
          and provenance['git_commit_unchanged'] is True and empty_changes(provenance['source_changes'])
          and source['files'] == provenance['before']['files'] == provenance['after']['files']
          and all(inventory[k]['sha256'] == v['sha256_raw'] and inventory[k]['bytes'] == v['size_bytes'] for k, v in source['files'].items())
          and sha(TRIAL/'producer.py') == sha(ROOT/'v6_lite/audit_research_shape_decomposition.py'))
    check('source_schema_counterexamples', empty_changes({'added': [], 'removed': [], 'modified': []})
          and not empty_changes({'added': ['x'], 'removed': [], 'modified': []})
          and not empty_changes({'added': [], 'removed': [], 'modified': ['x']})
          and not empty_changes({}) and not empty_changes({'added': [], 'removed': [], 'modified': False})
          and not near(float('nan'), 0) and not allfinite({'nested': [float('inf')]}))
    check('fixed_original_172_indices_and_input_identity', ids == expected == plan['indices']
          == [x['index'] for x in case_checks] == [x['index'] for x in numeric]
          and len(ids) == len(set(ids)) == 172 and ids_hash == IDS_SHA == plan['indices_sha256']
          == numeric_plan['indices_sha256'] == final_numeric_report['indices_sha256']
          and sha(input_path) == INPUT_SHA == plan['frozen_input_sha256'] == numeric_plan['input_sha256']
          and verify_manifest(input_path.parent, 'audit_manifest.json'))
    contract = hashlib.sha256(json.dumps(plan['model_contract'], sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    check('fixed_contract_sampling_and_negative_interpretation', contract == CONTRACT_SHA == plan['model_contract_sha256']
          and plan['tube_radii_m'] == read(OLD/'plan.json')['tube_radii_m'] and plan['gate_m'] == .005
          and plan['PCC_samples_per_section'] == 17 and plan['capsule_axis_samples'] == 5
          and plan['original_envelope_numerical_margin_m'] == 1e-4 and plan['capsule_count'] == 61
          and plan['fallback_geom_names'] == ['collision_0003'] and plan['fallback_included_in_envelope_radius_or_distances'] is False
          and plan['online_radius_reduction_authorized'] is False and plan['compute_performance_or_wall_test'] is False
          and 'unsafe indicators, not exact penetration depths' in plan['negative_clearance_rule']
          and 'inconclusive, not proof' in plan['negative_envelope_margin_rule'])
    required_checks = {'exact_frozen_input_arrays', 'matched_actual_configuration_target_pose_and_model',
        'all_independent_urdf_fk_matches_mujoco', 'direct_all_geoms_actual_matches_original',
        'independent_same_material_witness_and_pcc_exponential', 'independent_residual_identity_and_interpretation',
        'all_30_modules_independent_piecewise_distance', 'terminal_offset_counted_once',
        'all_61_capsule_candidates_and_minimum', 'all_capsules_original_17x5_sufficient_bound'}
    check('all_1720_independent_numerical_checks_complete', all(x['passed'] is True and set(x['checks']) == required_checks
          and all(v is True for v in x['checks'].values()) and x['checks'] == n['checks'] for x, n in zip(case_checks, numeric))
          and final_numeric_report['per_case_check_count'] == final_numeric_report['per_case_checks_passed'] == 1720
          and final_numeric_report['failed_cases'] == [])
    check('full_per_case_geometry_and_sampling_coverage', all(len(x['direct_actual_all_geoms']) == 61
          and len(x['chain_modules']) == 30 and len(x['all_capsule_candidates']) == 61
          and len(x['all_capsule_envelope_samples']) == 61 and len(x['segment_envelope']) == 5
          and len({z['geom_name'] for z in x['all_capsule_candidates']}) == 61
          and [z['module_index'] for z in x['chain_modules']] == list(range(30))
          and [z['segment_id'] for z in x['segment_envelope']] == list(range(5)) for x in numeric))
    check('raw_residual_algebra_and_lipschitz_bound', all(
          near(w['raw_proxy_m'], w['pcc_sd_m']-w['radius_m'])
          and near(w['directional_distance_effect_m'], w['actual_sd_m']-w['pcc_sd_m'])
          and abs(w['directional_distance_effect_m']) <= w['position_error_norm_m']+1e-12
          and near(w['actual_minus_raw_proxy_m'], w['radius_m']+w['directional_distance_effect_m']+w['remaining_global_surface_correspondence_m'])
          and abs(w['identity_residual_m']) <= 1e-12 for w in [x['same_material_witness'] for x in numeric]))
    check('original_per_capsule_and_per_section_envelope_algebra', all(
          near(z['sampled_required_radius_m'], z['sampled_axis_to_pcc_max_m']+z['capsule_radius_m'])
          and z['source_vertex_excess_m'] <= 0 for x in numeric for z in x['all_capsule_envelope_samples'])
          and all(near(z['sufficient_radius_upper_original_protocol_m'], z['sampled_required_radius_m']+z['axis_sampling_allowance_m']
                       +z['pcc_sampling_allowance_m']+z['numerical_margin_m'])
          and near(z['sufficient_radius_margin_m'], z['declared_radius_m']-z['sufficient_radius_upper_original_protocol_m'])
          for x in numeric for z in x['segment_envelope']))
    chain_counts = dict(Counter(x['same_radius_chain_status'] for x in numeric))
    capsule_counts = dict(Counter(x['capsule_status'] for x in numeric))
    support_count = sum(x['sufficient_coverage'] is True for x in numeric)
    check('reclassified_raw_top_counts_and_positive_minima', chain_counts == {'SAFE': 25, 'BELOW': 147}
          == produced['same_radius_chain_status_counts'] == final_numeric_report['same_radius_chain_status_counts']
          and capsule_counts == {'SAFE': 171, 'BELOW': 1} == produced['capsule_gate_status_counts'] == final_numeric_report['capsule_status_counts']
          and support_count == 172 == produced['sufficient_envelope_support_at_this_state_count']
          and all(x['same_radius_chain_status'] == ('SAFE' if min(z['net_indicator_m'] for z in x['chain_modules']) >= .005 else 'BELOW')
                  and x['capsule_status'] == ('SAFE' if min(z['capsule_indicator_m'] for z in x['all_capsule_candidates']) >= .005 else 'BELOW')
                  and x['sufficient_coverage'] is all(z['sufficient_radius_margin_m'] >= 0 for z in x['segment_envelope'])
                  and all(z['centerline_distance_or_unsafe_indicator_m'] > 0 for z in x['chain_modules'])
                  and min(z['capsule_indicator_m'] for z in x['all_capsule_candidates']) > 0 for x in numeric))
    stats = final_numeric_report['statistics']
    check('saved_independent_full_statistics_match_producer', all(set(produced[k]) == set(v)
          and all(near(produced[k][kk], vv) for kk, vv in v.items()) for k, v in stats.items()))
    check('producer_and_all_references_readonly_end', all(sha(ROOT/x['path']) == x['sha256']
          and (ROOT/x['path']).stat().st_size == x['bytes'] for x in references)
          and all(sha(ROOT/k) == v['sha256'] for k, v in inventory.items())
          and subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip() == COMMIT
          and produced['safety_contract_changed'] is False and produced['closed_loop_claim'] is False
          and produced['hardware_claim'] is False and produced['wall_deployment_status'] == 'NOT_MET')
    write(OUT/'by_item_checks.json', CHECKS)
    write(OUT/'source_inventory.json', inventory)
    write(OUT/'reference_manifest.json', {'artifacts': references})
    report = {'schema': 'independent_shape_decomposition_final_audit_v1', 'complete': True,
        'evidence_valid': all(x['passed'] for x in CHECKS), 'checks_passed': sum(x['passed'] for x in CHECKS),
        'checks_count': len(CHECKS), 'source_trial_commit': COMMIT, 'auditor_source_commit': commit,
        'auditor_script_sha256': sha(__file__), 'numerical_auditor_script_sha256': sha(AUDITS[1]/'audit_shape_decomposition.py'),
        'case_count': 172, 'indices_sha256': IDS_SHA, 'numerical_checks_passed': 1720, 'numerical_checks_count': 1720,
        'numerical_evidence_dir': AUDITS[1].relative_to(ROOT).as_posix(),
        'numerical_evidence_rows_sha256': sha(AUDITS[1]/'case_numeric_checks.jsonl'),
        'same_radius_chain_status_counts': chain_counts, 'capsule_status_counts': capsule_counts,
        'sufficient_envelope_support_count': support_count, 'statistics': stats,
        'failed_checks': [x for x in CHECKS if not x['passed']],
        'retained_auditor_failures': [{'directory': AUDITS[0].relative_to(ROOT).as_posix(), 'evidence_valid': False,
            'errors': ['Terminal cube tied PCA eigenvalues: audit01 chose last eigenvector while frozen protocol chooses first maximal via argmax.',
                       'audit01/02 mistook a nonempty schema dictionary containing three empty change lists for actual source changes.']},
            {'directory': AUDITS[1].relative_to(ROOT).as_posix(), 'evidence_valid': False,
             'error': 'All 1720 numerical checks pass; source dictionary schema flag was false. Correct independent source gate passes here.'}],
        'limits': final_numeric_report['limits'], 'wall_deployment_status': 'NOT_MET', 'closed_loop_claim': False,
        'hardware_claim': False, 'this_run_repeated_numerical_or_physics_queries': False}
    write(OUT/'report.json', report)
    files = sorted(p for p in OUT.iterdir() if p.is_file() and p.name != 'artifact_manifest.json')
    write(OUT/'artifact_manifest.json', {'schema': 'independent_shape_final_audit_manifest_v1',
        'artifacts': [{'path': p.name, 'sha256': sha(p), 'bytes': p.stat().st_size} for p in files]})
    print(json.dumps({'evidence_valid': report['evidence_valid'], 'checks': f'{report["checks_passed"]}/{report["checks_count"]}',
        'numerical_checks': '1720/1720', 'output': str(OUT)}, ensure_ascii=True))
    return 0 if report['evidence_valid'] else 1


if __name__ == '__main__':
    try:
        sys.exit(run())
    except Exception:
        error = traceback.format_exc()
        write(OUT/'audit_failure.json', {'complete': False, 'evidence_valid': False, 'error': error})
        print(error, file=sys.stderr)
        sys.exit(1)
