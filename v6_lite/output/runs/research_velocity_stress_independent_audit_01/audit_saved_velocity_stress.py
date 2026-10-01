"""Independent saved-evidence audit; no controller imports or MuJoCo stepping.

Only writes new audit products beside this script. The trial, baseline and
runtime sources are read-only. Statistics are recomputed from raw samples.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
import traceback
import xml.etree.ElementTree as ET

import numpy as np

SCENE_CHECKS = set("""continuum_initial_position_contract rigid_grasp_final_error
rigid_grasp_steady_rmse continuum_irregular_waypoint_path_rmse
rigid_orientation_error_below_limit continuum_orientation_error_below_limit
whole_body_dense_discrete_clearance continuum_target_dense_discrete_clearance
obstacle_avoidance_engaged single_qp_every_task_tick all_qp_solves_succeeded
dynamic_execution_only no_degenerate_active_clearance_gradients
reaction_map_satisfies_zero_momentum_constraint physics_and_task_rate_exact""".split())
DELIVERY_CHECKS = set("""contract_version authoritative_metrics_hash five_seeded_scenarios
exact_control_rates requested_acceptance_thresholds_locked learning_free_architecture
original_irregular_waypoint_contract continuum_end_effector_offset_contract
no_learning_runtime_imports reported_summary_passed trace_hashes
continuum_initial_position_is_1930_626_0_mm metrics_recomputed_from_traces
native_torque_replay_exact one_qp_per_50hz_tick obstacle_constraints_materially_engaged
dynamic_ctrl_mj_step_only honest_collision_claim_scope moving_target_continuum_clearance
continuum_target_all_500hz_replay_states_clear rigid_grasp_point_error
continuum_irregular_waypoint_tracking_error both_arm_orientation_error
whole_body_minimum_clearance base_pose_drift_recomputed""".split())
EXECUTION_CHECKS = set("""new_contract_version five_scenarios selected_endpoint_equals_servo_command
no_uncertified_execution current_ramp_rows_feasible predicted_next_start_rows_feasible
shared_ten_step_reference_exact reference_measured_error_recorded
clip_and_saturation_diagnostics_recorded sim_and_wall_clock_separate trace_hashes_match""".split())
PERFORMANCE_CHECKS = {"task_controller_runs_within_50hz_p95", "torque_controller_runs_within_500hz_p95"}
STEPS = ["current_and_historical_tests", "complete_five_scene_simulation", "simulation_qualification",
         "independent_torque_delivery", "independent_execution_contract", "independent_interval_recompute",
         "exact_c1_nontiming_parity", "raw_timing_audit"]


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_new(path, payload):
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def near(a, b, tolerance=1e-9):
    return finite(a) and finite(b) and abs(a - b) <= tolerance


def equal_stats(actual, expected):
    if isinstance(expected, dict):
        return isinstance(actual, dict) and actual.keys() == expected.keys() and all(
            equal_stats(actual[key], value) for key, value in expected.items())
    if isinstance(expected, float):
        return near(actual, expected, 1e-10)
    return actual == expected


def stats(raw, deadline, steady=True):
    samples = np.asarray(raw, dtype=float)
    if samples.ndim != 1 or not len(samples) or not np.all(np.isfinite(samples)) or np.any(samples < 0):
        raise ValueError("invalid raw latency samples")
    longest = streak = 0
    for exceeded in samples > deadline:
        streak = streak + 1 if exceeded else 0
        longest = max(longest, streak)
    return {"count": len(samples), "p50_ms": float(np.median(samples) * 1000),
            "p95_ms": float(np.percentile(samples, 95) * 1000),
            "p99_ms": float(np.percentile(samples, 99) * 1000),
            "max_ms": float(np.max(samples) * 1000),
            "over_deadline_count": int(np.count_nonzero(samples > deadline)),
            "longest_consecutive_over_deadline": longest,
            "first_cycle_ms": float(samples[0] * 1000),
            "steady_after_first_cycle": stats(samples[1:], deadline, False) if steady and len(samples) > 1 else None,
            "passed": bool(np.percentile(samples, 95) <= deadline)}


def audit_cycle(row, index, trace, scene):
    checks = {}
    cert, guards = row["certificate"], row["servo_dispatch_checks"]
    start, end = row["state_acquisition_monotonic_ns"], row["actual_dispatch_monotonic_ns"]
    phases = row["phases"]
    source = float(trace["task_time"][index])
    raw_boundary = 0.0 if index == 0 else float(trace["time"][index * 10 - 1])
    clocks = [start, end, row["source_simulation_time_s"], row["dispatch_latency_s"], row["thread_cpu_s"]]
    clocks.extend(value for phase in phases for value in (
        phase["start_ns"], phase["end_ns"], phase["wall_s"], phase["thread_cpu_s"]))
    cert_clocks = [cert[name] for name in ("state_acquisition_time", "target_acquisition_time",
        "planned_execution_start", "planned_execution_start_simulation_s", "solve_finished_time",
        "validation_finished_time", "valid_until", "maximum_supported_state_age", "simulation_valid_until")]
    checks["finite_nonnegative_clocks"] = all(finite(value) and value >= 0 for value in clocks + cert_clocks)
    if not checks["finite_nonnegative_clocks"]:
        return {"scene": scene["scenario"]["scenario_id"], "cycle": index,
                "passed": False, "checks": checks, "errors": ["finite_nonnegative_clocks"]}
    checks["source_time_raw_trace_and_grid"] = (near(source, index * .020) and near(source, raw_boundary)
        and near(row["source_simulation_time_s"], source)
        and near(cert["planned_execution_start_simulation_s"], source))
    checks["phase_clock_chain"] = bool(phases) and phases[0]["start_ns"] == start and phases[-1]["end_ns"] == end \
        and end >= start and all(p["end_ns"] >= p["start_ns"] and
            near(p["wall_s"], (p["end_ns"] - p["start_ns"]) * 1e-9, 1e-12) for p in phases) \
        and all(a["end_ns"] == b["start_ns"] for a, b in zip(phases, phases[1:])) \
        and near(sum(p["thread_cpu_s"] for p in phases), row["thread_cpu_s"], 1e-12)
    checks["raw_dispatch_latency"] = near((end - start) * 1e-9, row["dispatch_latency_s"], 1e-12)
    phase_ends = {p["name"]: p["end_ns"] * 1e-9 for p in phases}
    checks["certificate_raw_wall_clock_binding"] = (near(cert["state_acquisition_time"], start * 1e-9)
        and cert["target_acquisition_time"] == cert["state_acquisition_time"]
        and cert["planned_execution_start"] == cert["state_acquisition_time"]
        and near(cert["solve_finished_time"], phase_ends.get("qp_assembly_and_solve"))
        and near(cert["validation_finished_time"], phase_ends.get("execution_validation"))
        and cert["state_acquisition_time"] <= cert["solve_finished_time"] <= cert["validation_finished_time"] <= end * 1e-9)
    checks["simulation_validity_20ms_unextended"] = (cert["validity_clock"] == "simulation_time"
        and cert["maximum_supported_state_age"] == .020
        and cert["valid_until"] == cert["state_acquisition_time"] + .020
        and near(cert["simulation_valid_until"], source + .020, 1e-12))
    checks["ordered_command_and_selected_payload"] = (row["command_id"] == index == cert["command_id"]
        and cert["command_sha256"] == hashlib.sha256(trace["task_selected_command"][index].tobytes()).hexdigest()
        and np.array_equal(trace["command_velocity"][index * 10:index * 10 + 10],
            np.repeat(trace["task_selected_command"][index][None, :], 10, axis=0)))
    checks["recorded_source_model_partition_ids"] = (cert["source_model_hash"] ==
        scene["execution_contract"]["source_compiled_model_sha256"]
        and all(isinstance(cert[key], str) and re.fullmatch(r"[0-9a-f]{64}", cert[key]) is not None
                for key in ("source_state_id", "source_model_hash", "source_partition_id", "command_sha256")))
    checks["all_ten_guard_records"] = len(guards) == 10 and row["accepted"] is True \
        and row["dispatch_check"] == guards[0] if guards else False
    guard_errors = []
    previous = -math.inf
    for substep, guard in enumerate(guards):
        scalar_values = [guard.get(key) for key in ("dispatch_time", "simulation_state_age_s",
            "simulation_valid_until", "state_age_s", "target_age_s")]
        if not all(finite(value) and value >= -1e-9 for value in scalar_values):
            guard_errors.append(f"nonfinite_guard_{substep}")
            continue
        observed_simulation = source + guard["simulation_state_age_s"]
        raw_time = 0.0 if index * 10 + substep == 0 else float(trace["time"][index * 10 + substep - 1])
        if not (guard["accepted"] is True and guard["reason"] is None
                and guard["policy"] == "research_simulation" and guard["validity_clock"] == "simulation_time"
                and guard["servo_substep"] == substep and substep < 10
                and near(guard["simulation_state_age_s"], .002 * substep)
                and near(observed_simulation, raw_time) and near(observed_simulation, index * .020 + substep * .002)
                and guard["simulation_valid_until"] == cert["simulation_valid_until"]
                and observed_simulation < cert["simulation_valid_until"]
                and cert["validation_finished_time"] <= guard["dispatch_time"] and previous <= guard["dispatch_time"]
                and near(guard["state_age_s"], guard["dispatch_time"] - cert["state_acquisition_time"])
                and near(guard["target_age_s"], guard["dispatch_time"] - cert["target_acquisition_time"])
                and guard["wall_clock_current"] == (guard["dispatch_time"] <= cert["valid_until"])
                and guard["wall_deployment_certified"] is False
                and guard["continuation_guaranteed_on_reject"] is False):
            guard_errors.append(f"invalid_guard_{substep}")
        if substep == 0 and not (start * 1e-9 <= guard["dispatch_time"] <= end * 1e-9):
            guard_errors.append("first_guard_outside_raw_timeline")
        previous = guard["dispatch_time"]
    checks["accepted_ordered_finite_guards_on_raw_simulation_grid"] = not guard_errors
    errors = [key for key, value in checks.items() if not value] + guard_errors
    return {"scene": scene["scenario"]["scenario_id"], "cycle": index,
            "certificate_id": cert["command_id"], "guard_count": len(guards),
            "accepted_after_wall_valid_until_count": sum(g["wall_clock_current"] is False for g in guards),
            "passed": not errors, "checks": checks, "errors": errors}



EXPECTED_COMMIT = '9c7120e8ca8711cd58dc65471e58763c965148ff'
BASELINE_MANIFEST_SHA = '379b9977a441c540a20cfc50a9e90614a1e0985fa2a5a73bbb7c9690f823eec0'
FIXTURE_SHA = 'd9b282bc73dcfcba83d3ebfe187a89b5ee9d8eecae8d0ba8204feeece933f9b2'
IDS = [f'v6_lite_scenario_{i:02d}' for i in range(5)]
VELOCITY = 'target_satellite_linear_velocity_m_s'
SKIPPED_FULL_GATES = ['applied_physical_velocity_factor', 'simulation_qualification',
    'independent_torque_delivery', 'independent_execution_contract',
    'independent_interval_recompute', 'raw_timing_audit']


def strict_load(path):
    def pairs(values):
        result = {}
        for key, value in values:
            if key in result:
                raise ValueError('duplicate JSON field: ' + key)
            result[key] = value
        return result
    return json.loads(Path(path).read_text(encoding='utf-8'), object_pairs_hook=pairs,
        parse_constant=lambda value: (_ for _ in ()).throw(ValueError('nonfinite JSON: ' + value)))


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
        allow_nan=False).encode()).hexdigest()


def jsonl(path):
    return [json.loads(line, parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
        for line in Path(path).read_text(encoding='utf-8').splitlines()]


def inventory(directory):
    return {p.relative_to(directory).as_posix(): {'sha256': sha(p), 'bytes': p.stat().st_size}
        for p in sorted(directory.rglob('*')) if p.is_file() and p.name != 'manifest.json'}


def clock_check(row):
    phases = row['phases']
    start, end = row['state_acquisition_monotonic_ns'], row['actual_dispatch_monotonic_ns']
    values = [start, end, row['source_simulation_time_s'], row['dispatch_latency_s'], row['thread_cpu_s']]
    values.extend(v for p in phases for v in (p['start_ns'], p['end_ns'], p['wall_s'], p['thread_cpu_s']))
    return (all(finite(v) and v >= 0 for v in values) and bool(phases) and end >= start
        and phases[0]['start_ns'] == start and phases[-1]['end_ns'] == end
        and all(p['end_ns'] >= p['start_ns'] and near(p['wall_s'], (p['end_ns']-p['start_ns'])*1e-9, 1e-12) for p in phases)
        and all(a['end_ns'] == b['start_ns'] for a,b in zip(phases, phases[1:]))
        and near(sum(p['thread_cpu_s'] for p in phases), row['thread_cpu_s'], 1e-12)
        and near((end-start)*1e-9, row['dispatch_latency_s'], 1e-12))


def compiled_model_hash(model, contract):
    digest = hashlib.sha256(contract.encode())
    for name in ('body_mass','body_inertia','body_pos','body_quat','body_ipos','body_iquat',
        'dof_damping','dof_armature','dof_frictionloss','jnt_stiffness','jnt_axis','jnt_pos','jnt_range',
        'geom_pos','geom_quat','geom_size','geom_type','geom_contype','geom_conaffinity','actuator_gear',
        'actuator_gainprm','actuator_biasprm','actuator_dynprm','actuator_ctrlrange'):
        digest.update(getattr(model, name).tobytes())
    digest.update(np.asarray([model.opt.timestep,int(model.opt.integrator),int(model.opt.solver),
        int(model.opt.disableflags),int(model.opt.enableflags),model.opt.iterations,model.opt.tolerance,
        model.opt.density,model.opt.viscosity]).tobytes())
    digest.update(model.opt.gravity.tobytes())
    return digest.hexdigest()


def audit(root, trial, baseline, output):
    import sys
    sys.path.insert(0, str(root))
    import mujoco
    from model_test.robot_model_spec_v5 import RobotModelSpecV5, default_robot_model_spec_v5
    from model_test.whole_body_verifier_v5 import WholeBodyCollisionVerifier, WholeBodyVerificationConfig, WorkspaceSphere
    items, scenes, cycles, rejected, source_files = [], [], [], [], {}
    def check(name, value, detail=None):
        items.append({'name': name, 'passed': bool(value), 'detail': detail})
    ap = strict_load(output/'audit_plan.json')
    check('auditor_script_matches_prepared_identity', sha(__file__) == ap['script_sha256_raw'])
    plan, report = strict_load(trial/'plan.json'), strict_load(trial/'report.json')
    suite = trial/'simulation'
    metadata = strict_load(suite/'run_metadata.json')
    cohort = strict_load(suite/'research_trial_report.json')
    definitions = strict_load(trial/'scenario_definitions.json')
    base_metrics = strict_load(baseline/'simulation/v6_lite_metrics.json')
    original = [s['scenario'] for s in base_metrics['scenarios']]
    original_by_id = {s['scenario']['scenario_id']: s for s in base_metrics['scenarios']}
    completed = {s['scenario']['scenario_id']: s for s in cohort['completed_scenes']}
    failed = {s['scenario_id']: s for s in cohort['failed_scenarios']}
    cfg, qp = plan['run_config'], plan['qp_config']
    check('audit_prepared_plan_binding', sha(trial/'plan.json') == ap['trial_plan_sha256'])
    check('full_unique_five_attempts_retained', len(completed) == len(cohort['completed_scenes'])
        and len(failed) == len(cohort['failed_scenarios']) and not (set(completed)&set(failed))
        and set(completed)|set(failed) == set(IDS) and cohort['all_predeclared_scenes_attempted'] is True)
    check('research_failure_and_incomplete_report_honest', bool(failed) and report['passed'] is False
        and report['complete'] is False and cohort['complete_five_scene_horizons'] is False)
    steps = {s['name']: s for s in report['steps']}
    check('full_cohort_gates_explicitly_not_run', all(steps[n]['status'] == 'NOT_RUN' for n in SKIPPED_FULL_GATES)
        and not (suite/'validation.json').exists() and not (suite/'execution_validation.json').exists()
        and not (trial/'interval_recompute').exists())
    check('no_old_c1_parity_required', plan['old_c1_nontiming_parity_required'] is False
        and not any('parity' in s['name'] for s in report['steps']))
    check('partial_timing_never_full_performance_acceptance', report['computational_performance']['status'] == 'INCOMPLETE_COHORT_OBSERVATION'
        and report['computational_performance']['evidence_valid'] is False
        and report['computational_performance']['performance_target_met'] is None
        and report['computational_performance']['performance_is_functional_gate'] is False)
    for directory, expected in ((trial, None), (baseline, BASELINE_MANIFEST_SHA)):
        manifest_path = directory/'manifest.json'
        manifest = {k.replace('\\','/'):v for k,v in strict_load(manifest_path).items()}
        actual = inventory(directory)
        check(directory.name+':complete_manifest_sha_and_file_set', manifest == {k:v['sha256'] for k,v in actual.items()}
            and (expected is None or sha(manifest_path) == expected))
        source_files[str(directory)] = {'manifest': {'sha256': sha(manifest_path), 'bytes': manifest_path.stat().st_size}, 'files':actual}
    commit = subprocess.check_output(['git','rev-parse','HEAD'], cwd=root, text=True).strip()
    status = subprocess.check_output(['git','status','--porcelain','--untracked-files=no'], cwd=root, text=True).strip()
    prov = report['source_provenance']
    check('frozen_source_commit_and_clean_tracked_worktree', commit == EXPECTED_COMMIT and not status
        and plan['source']['git_commit'] == metadata['source']['git_commit'] == EXPECTED_COMMIT
        and prov['before']['git_commit'] == prov['after']['git_commit'] == EXPECTED_COMMIT)
    check('frozen_source_provenance_complete_unchanged', prov['source_unchanged'] is True
        and prov['capture_complete'] is True and prov['git_commit_unchanged'] is True
        and not prov['before']['capture_errors'] and not prov['after']['capture_errors']
        and not prov['before']['tracked_worktree_dirty'] and not prov['after']['tracked_worktree_dirty']
        and prov['before']['files'] == prov['after']['files'] == plan['source']['files'])
    for name, frozen in plan['source']['files'].items():
        raw = (root/name).read_bytes()
        current = {'sha256_raw':hashlib.sha256(raw).hexdigest(), 'bytes':len(raw),
            'sha256_lf_normalized':hashlib.sha256(raw.replace(b'\r\n',b'\n')).hexdigest()}
        source_files[name] = current
        check('source:'+name, current['sha256_raw'] == frozen['sha256_raw'] and len(raw) == frozen['size_bytes']
            and current['sha256_lf_normalized'] == metadata['source']['source_normalized_lf_sha256'].get(name))
    identities = report['committed_source_identity']
    check('required_committed_source_and_fixture_before_after', report['required_committed_files_unchanged'] is True
        and identities['before']['passed'] is True and identities['after']['passed'] is True
        and identities['before']['required_committed_files'] == identities['after']['required_committed_files'])
    for name, entry in identities['before']['required_committed_files'].items():
        raw = (root/name).read_bytes()
        git_raw = subprocess.check_output(['git','show',EXPECTED_COMMIT+':'+name], cwd=root)
        check('committed:'+name, raw.replace(b'\r\n',b'\n') == git_raw.replace(b'\r\n',b'\n')
            and sha(root/name) == entry['sha256_raw'] and len(raw) == entry['bytes'])
    fixture = root/'v6_lite/test_fixtures/target_velocity_baseline.json'
    check('immutable_baseline_fixture', sha(fixture) == FIXTURE_SHA and strict_load(fixture)['scenarios'] == original)
    expected = json.loads(json.dumps(original))
    for scene in expected:
        scene[VELOCITY] = [2.*v for v in scene[VELOCITY]]
    check('exact_original_rng_cohort_and_single_factor', definitions['original'] == definitions['scale_one'] == original
        and definitions['scale_two'] == expected == metadata['scenarios']
        and [s['scenario_id'] for s in expected] == plan['scenario_order'] == IDS
        and sha(trial/'scenario_definitions.json') == plan['scenario_definitions_file_sha256']
        and all(canonical(definitions[k]) == definitions['canonical_sha256'][k] == plan['scenario_definitions_canonical_sha256'][k]
            for k in ('original','scale_one','scale_two')))
    old_config = dict(cfg); old_config.pop('target_linear_velocity_scale')
    config_hash = canonical({'run':cfg,'qp':qp})
    check('exact_original_config_contract_except_scale', old_config == base_metrics['run_config']
        and cfg['target_linear_velocity_scale'] == 2. and qp == base_metrics['qp_config']
        and cfg == metadata['run_config'] and qp == metadata['qp_config']
        and metadata['configuration_sha256'] == config_hash
        and metadata['scenario_definitions_sha256'] == canonical(expected))
    check('locked_rates_and_safety', cfg['physics_period_s'] == .002 and cfg['task_period_s'] == .020
        and cfg['duration_s'] == 27. and cfg['whole_body_minimum_clearance_m'] == .005
        and qp['clearance_rate_tolerance_m_s'] == .0001 and qp['velocity_tolerance_rad_s'] == .0001
        and qp['enable_capsule_cbf'] is True and cfg['dispatch_clock_policy'] == 'research_simulation')
    tests = strict_load(trial/'test_profiles/report.json')
    check('current_213_and_historical_16_regressions_retained', tests['profile'] == 'all' and tests['passed'] is True
        and tests['current']['tests'] == 213 and tests['current']['failures'] == tests['current']['errors'] == 0
        and tests['historical']['tests_selected'] == 16 and tests['historical']['exit_code'] == 0
        and tests['historical_goldens_modified'] is False and tests['branch_name_used_for_selection'] is False)
    for key in ('current','historical'):
        record = tests[key]; log = trial/'test_profiles'/record['log_file']
        check(key+':log_sha_and_bytes', sha(log) == record['log_sha256_raw'] and log.stat().st_size == record['log_size_bytes'])
    spec = RobotModelSpecV5.from_urdf(default_robot_model_spec_v5().source_urdf,
        continuum_kp=100.,continuum_kd=5.,continuum_torque_limit=4.,rigid_kp=200.,rigid_kd=20.,rigid_torque_limit=80.)
    contract, bundle = spec.runtime_contract_sha256(), spec.source_bundle_sha256()
    check('declared_model_identity_preserved', metadata['model_identity']['runtime_contract_sha256'] == contract
        and metadata['model_identity']['source_bundle_sha256'] == bundle)
    needed = ('initial_qpos','initial_qvel','time','torque','task_time','task_qpos','command_velocity',
        'task_selected_command','task_full_latency','torque_latency','task_solver_status','task_solver_candidate',
        'task_failure_reason','task_execution_mode','task_interval_selected_rows','task_ramp_clearance_min_slack_m_s',
        'task_ramp_velocity_min_slack_rad_s','task_lookahead_min_slack_m_s','task_solver_latency','task_shape_clearance_latency')
    for declared in expected:
        name = declared['scenario_id']; is_complete = name in completed
        fail = None if is_complete else strict_load(suite/'failures'/(name+'_execution_failure.json'))
        path = Path(completed[name]['trace']['path']) if is_complete else Path(fail['partial_trace'])
        check(name+':unique_owned_trace_path', path.resolve().is_relative_to(suite.resolve())
            and path.name == name+('.npz' if is_complete else '_partial_trace.npz'))
        with np.load(path,allow_pickle=False) as z:
            trace = {k:z[k].copy() for k in needed}
            failure_time = None if is_complete else float(z['failure_time_s'])
        old_path = Path(original_by_id[name]['trace']['path'])
        with np.load(old_path,allow_pickle=False) as old:
            old_qpos, old_qvel = old['initial_qpos'].copy(),old['initial_qvel'].copy()
        check(name+':baseline_trace_hash', sha(old_path) == original_by_id[name]['trace']['sha256'])
        obstacles = [WorkspaceSphere(name=o['name'],center=np.asarray(o['center_w']),radius=o['radius_m']) for o in declared['workspace_obstacles']]
        model = WholeBodyCollisionVerifier(spec,obstacles,WholeBodyVerificationConfig(minimum_clearance=.005,
            query_distance_max=2.5,adaptive_subdivisions=4,self_collision_ancestor_exclusion_depth=3,include_target_satellite_pairs=True)).model
        model.geom_contype[:]=0; model.geom_conaffinity[:]=0
        model_hash = compiled_model_hash(model,contract)
        jid = mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,spec.target_free_joint_name)
        check(name+':named_target_free_joint', jid >= 0 and int(model.jnt_type[jid]) == int(mujoco.mjtJoint.mjJNT_FREE))
        qstart,vstart = int(model.jnt_qposadr[jid]),int(model.jnt_dofadr[jid]); linear=slice(vstart,vstart+3); angular=slice(vstart+3,vstart+6)
        init = mujoco.MjData(model); expected_qpos = init.qpos.copy()
        for joint,value in zip(spec.low_level_joint_names,spec.encode_position(spec.planner_zero)):
            joint_id=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,joint)
            expected_qpos[int(model.jnt_qposadr[joint_id])]=value
        expected_qpos[qstart:qstart+3]+=declared['target_satellite_position_shift_m']
        other=np.ones(model.nv,dtype=bool); other[vstart:vstart+6]=False
        qpos,qvel=trace['initial_qpos'],trace['initial_qvel']
        check(name+':actual_initial_qpos_matches_original_and_model', qpos.shape == (model.nq,)
            and np.all(np.isfinite(qpos)) and np.array_equal(qpos,old_qpos) and np.array_equal(qpos,expected_qpos))
        check(name+':actual_twofold_named_target_qvel_only', qvel.shape == old_qvel.shape == (model.nv,)
            and np.all(np.isfinite(qvel)) and np.array_equal(qvel[linear],np.asarray(declared[VELOCITY]))
            and np.array_equal(qvel[linear],2.*old_qvel[linear])
            and np.array_equal(qvel[angular],np.asarray(declared['target_satellite_angular_velocity_rad_s']))
            and np.array_equal(qvel[angular],old_qvel[angular]) and np.array_equal(qvel[other],old_qvel[other]))
        n=len(trace['time']); accepted=n//10; attempted=len(trace['task_time']); timelines=jsonl(suite/'timing'/(name+'.jsonl'))
        check(name+':saved_physical_and_task_grid', n>0 and n%10==0 and trace['torque'].shape==(n,67)
            and trace['command_velocity'].shape==(n,17) and np.all(np.isfinite(trace['torque']))
            and np.max(np.abs(trace['time']-np.arange(1,n+1)*.002))<=1e-9
            and np.max(np.abs(trace['task_time']-np.arange(attempted)*.020))<=1e-9
            and attempted==accepted+(0 if is_complete else 1) and len(timelines)==attempted
            and trace['task_qpos'].shape==(accepted+1,model.nq))
        stub = completed[name] if is_complete else {'scenario':declared,'execution_contract':{'source_compiled_model_sha256':model_hash}}
        check(name+':compiled_model_hash_independently_recreated', all(r['certificate']['source_model_hash']==model_hash for r in timelines[:accepted]))
        rows=[audit_cycle(r,index,trace,stub) for index,r in enumerate(timelines[:accepted])]
        cycles.extend(rows)
        check(name+':all_saved_accepted_certificates_guards', len(rows)==accepted and all(r['passed'] for r in rows)
            and sum(r.get('guard_count',0) for r in rows)==n)
        check(name+':all_raw_phase_chains', all(clock_check(r) for r in timelines))
        record={'scenario_id':name,'status':'COMPLETED' if is_complete else 'REJECTED_PARTIAL',
            'trace_path':str(path),'trace_sha256':sha(path),'trace_bytes':path.stat().st_size,
            'physics_steps_saved':n,'successful_task_ticks':accepted,'planning_attempts_saved':attempted,
            'timeline_count':len(timelines),'guards_saved':sum(r.get('guard_count',0) for r in rows),
            'phase_counts':dict(Counter(p['name'] for r in timelines for p in r['phases'])),
            'selected_interval_row_occurrences_executed':int(np.sum(trace['task_interval_selected_rows'][:accepted])),
            'initial_qpos':qpos.tolist(),'initial_qvel':qvel.tolist(),'named_target_dof_slice':[vstart,vstart+6],
            'actual_linear_velocity':qvel[linear].tolist(),'actual_angular_velocity':qvel[angular].tolist(),
            'compiled_model_sha256':model_hash,'source_partition_id_unique_count':len({r['certificate']['source_partition_id'] for r in timelines[:accepted]}),
            'raw_timing_observation':{'algorithm_attempts':stats(trace['task_full_latency'],.020),
                'dispatch_or_rejection_attempts':stats([r['dispatch_latency_s'] for r in timelines],.020),'executed_torque':stats(trace['torque_latency'],.002)}}
        if is_complete:
            scene=completed[name]; rates=scene['metrics']['rates_and_latency']
            check(name+':complete_declared_scene_and_full_shape', scene['scenario']==declared and n==13500 and accepted==1350
                and sha(path)==scene['trace']['sha256'] and set(scene['checks'])==SCENE_CHECKS
                and scene['passed']==all(scene['checks'].values()) and rates['physics_steps']==n and rates['task_ticks']==accepted)
            measured={'algorithm':stats(trace['task_full_latency'],.020),'dispatch':stats([r['dispatch_latency_s'] for r in timelines],.020),
                'torque':stats(trace['torque_latency'],.002)}
            check(name+':reported_timing_raw_consistency', all(equal_stats(rates[k],measured[k]) for k in ('algorithm','dispatch'))
                and near(rates['torque_latency_p95_ms'],measured['torque']['p95_ms'],1e-10))
            root_timing = {r['scenario_id']:r for r in report['computational_performance']['scenes']}[name]
            check(name+':root_partial_timing_raw_consistency', root_timing['passed'] is True and not root_timing['errors']
                and root_timing['trace_sha256'] == sha(path) and root_timing['timeline_sha256'] == sha(suite/'timing'/(name+'.jsonl'))
                and all(equal_stats(root_timing[k],measured[k]) for k in measured)
                and near(root_timing['initialization_latency_s'],rates['initialization_latency_s'],1e-12))
            check(name+':logged_nominal_contract_supported', np.all(trace['task_failure_reason']=='none')
                and np.all(np.isin(trace['task_execution_mode'],['TRACK','CAUTION']))
                and np.min(trace['task_ramp_clearance_min_slack_m_s'])>=-qp['clearance_rate_tolerance_m_s']
                and np.min(trace['task_ramp_velocity_min_slack_rad_s'])>=-qp['velocity_tolerance_rad_s']
                and np.min(trace['task_lookahead_min_slack_m_s'])>=-qp['clearance_rate_tolerance_m_s'])
            record['reported_functional_passed']=scene['passed']; record['independent_25_11_interval_full_cohort']='NOT_RUN'
        else:
            snapshot=strict_load(fail['counterexample_snapshot']); last=snapshot['planning_snapshots'][-1]; end=timelines[-1]
            check(name+':failure_artifacts_bound', snapshot['scenario_id']==fail['scenario_id']==name
                and snapshot['partial_trace']['sha256']==sha(path) and snapshot['partial_trace']['physics_steps_saved']==n
                and snapshot['run_config']==cfg and snapshot['qp_config']==qp and snapshot['configuration_sha256']==config_hash
                and snapshot['model_runtime_contract_sha256']==contract and snapshot['model_source_bundle_sha256']==bundle)
            check(name+':rejected_attempt_has_no_saved_following_servo_step', fail['next_servo_step_executed'] is False
                and snapshot['next_servo_step_executed'] is False and near(fail['time_s'],failure_time)
                and near(failure_time,n*.002) and near(float(trace['time'][-1]),failure_time)
                and near(last['time_s'],failure_time) and np.array_equal(np.asarray(last['qpos']),trace['task_qpos'][-1])
                and end['command_id']==accepted and near(end['source_simulation_time_s'],failure_time)
                and end['accepted'] is False and 'servo_dispatch_checks' not in end
                and end['reason']==fail['failure_reason']==last['failure_reason']==str(trace['task_failure_reason'][-1])
                and last['selected_command'] is None and np.all(np.isnan(trace['task_selected_command'][-1]))
                and not (suite/'traces'/(name+'.npz')).exists())
            check(name+':failure_candidate_and_old_command_identity', np.array_equal(np.asarray(fail['solver_candidate']),trace['task_solver_candidate'][-1])
                and np.array_equal(np.asarray(last['solver_candidate']),trace['task_solver_candidate'][-1])
                and np.array_equal(np.asarray(last['old_command']),trace['command_velocity'][-1])
                and fail['solver_status']==last['solver_status']==str(trace['task_solver_status'][-1])
                and fail['execution_mode']==last['execution_mode']==str(trace['task_execution_mode'][-1]))
            residual_errors=[abs(np.dot(np.asarray(r['gradient_m_per_rad']),last['solver_candidate'])-r['lower_m_s']-r['candidate_residual_m_s']) for r in last['rows']]
            check(name+':frozen_candidate_constraint_residual_arithmetic', bool(residual_errors) and max(residual_errors)<=1e-10)
            rejection={'scenario_id':name,'failure_reason':fail['failure_reason'],'solver_status':fail['solver_status'],
                'failure_time_s':failure_time,'physics_steps_saved':n,'rejected_planning_attempt':accepted,
                'next_unexecuted_physics_step':n,'last_executed_physics_step':n-1,'no_saved_next_servo_step':True,
                'rejected_timeline_phase_names':[p['name'] for p in end['phases']],
                'failed_timeline_prior_certificate_id':end['certificate']['command_id'] if end.get('certificate') else None,
                'snapshot_sha256':sha(fail['counterexample_snapshot']),'execution_failure_sha256':sha(suite/'failures'/(name+'_execution_failure.json')),
                'maximum_candidate_residual_arithmetic_error':max(residual_errors)}
            rejected.append(rejection); record['rejection']=rejection
        scenes.append(record)
    summary=report['trial_summary']
    check('root_trial_summary_matches_independent_scene_counts', summary['observed_attempt_count']==len(scenes)==5
        and summary['reported_completed_scene_count']==len(completed) and summary['reported_rejected_scene_count']==len(failed)
        and summary['all_five_attempts_observed'] is True and summary['partial_is_full_acceptance'] is False
        and {r['scenario_id']:r['status'] for r in summary['scenes']}=={s['scenario_id']:('COMPLETED_PASS' if s['reported_functional_passed'] else 'COMPLETED_FUNCTIONAL_FAILURE')
            if s['status']=='COMPLETED' else 'REJECTED_PARTIAL' for s in scenes})
    check('four_scopes_do_not_expand_claims', report['algorithm_simulation']['status']=='NOT_COMPLETED'
        and report['wall_continuation']=={'status':'NOT_MET','retested':False,'previous_goal_completed':False}
        and report['hardware_deployment']['status']=='NOT_ESTABLISHED' and report['hard_realtime_certified'] is False
        and report['delay_or_model_error_robustness_established'] is False and metadata['timing_protocol']['wall_deadline_enforced'] is False)
    check('root_completed_subset_performance_observation_honest',
        report['computational_performance']['observed_completed_scene_targets_met'] == all(
            s['raw_timing_observation'][key]['passed'] for s in scenes if s['status']=='COMPLETED'
            for key in ('algorithm_attempts','dispatch_or_rejection_attempts','executed_torque')))
    check('input_artifacts_unchanged_after_reading', source_files[str(trial)]['files']==inventory(trial)
        and source_files[str(baseline)]['files']==inventory(baseline))
    write_new(output/'source_and_artifact_inventory.json',source_files)
    write_new(output/'by_item_checks.json',items)
    write_new(output/'by_scene_checks.json',scenes)
    with (output/'certificate_guard_checks.jsonl').open('x',encoding='utf-8',newline='\n') as stream:
        for row in cycles:
            stream.write(json.dumps(row,separators=(',',':'),ensure_ascii=False,allow_nan=False)+'\n')
    failures=[r for r in items if not r['passed']]
    return {'schema':'research_velocity_stress_independent_failure_audit_v1','evidence_valid':not failures,
        'research_acceptance_passed':False,'trial_rejection_independently_confirmed':not failures,
        'source_trial_commit':EXPECTED_COMMIT,'script_sha256_raw':sha(__file__),'trial_directory':str(trial),
        'counts':{'scenarios':len(scenes),'completed_scenes':len(completed),'rejected_scenes':len(failed),
            'physics_steps_saved':sum(s['physics_steps_saved'] for s in scenes),'accepted_certificates':len(cycles),
            'guards_saved':sum(r.get('guard_count',0) for r in cycles),'planning_attempts_saved':sum(s['planning_attempts_saved'] for s in scenes),
            'checks':len(items),'failed_checks':len(failures)},
        'full_cohort_delivery_25':'NOT_RUN','full_cohort_execution_11':'NOT_RUN','full_cohort_interval_recompute':'NOT_RUN',
        'performance_is_research_gate':False,'wall_continuation_status':'NOT_MET','hardware_deployment':'NOT_ESTABLISHED',
        'continuous_time_certified':False,'hard_realtime_certified':False,'unknown_model_robustness_established':False,
        'scope':'saved hashes/config/cohort/initial state/raw clocks/all recorded guards/rejection boundary and candidate arithmetic; no MuJoCo stepping or repeated physics',
        'source_state_id_limit':'64-hex certificate source_state_id presence is checked; complete integration state including controls/activation/warmstart/history is not saved per task and cannot be independently hashed from qpos-only trace',
        'partition_id_limit':'full leaf list is not saved; certificate partition hash format/consistency is checked, not independently reconstructed from selected-row counts',
        'no_step_limit':'no saved following step is verified against partial trace grids, terminal snapshots, rejected timeline and frozen raise-before-servo control flow; this is not hardware continuation or safe backup evidence',
        'failures':failures,'rejections':rejected,'scenes':scenes}


def main():
    parser=argparse.ArgumentParser(description='Independent saved velocity stress failure audit; no physics replay')
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[4])
    parser.add_argument('--trial-dir',type=Path,required=True)
    parser.add_argument('--baseline-dir',type=Path,required=True)
    args=parser.parse_args(); output=Path(__file__).resolve().parent
    if (output/'report.json').exists():
        raise FileExistsError('preserve existing audit report and failures')
    try:
        result=audit(args.root.resolve(),args.trial_dir.resolve(),args.baseline_dir.resolve(),output)
    except Exception as error:
        result={'schema':'research_velocity_stress_independent_failure_audit_v1','evidence_valid':False,
            'research_acceptance_passed':False,'script_sha256_raw':sha(__file__),
            'error':{'type':type(error).__name__,'message':str(error),'traceback':traceback.format_exc()}}
    result['completed_utc']=datetime.now(timezone.utc).isoformat()
    write_new(output/'report.json',result)
    exit_code=0 if result['evidence_valid'] else 1
    write_new(output/'terminal_execution.json',{'expected_exit_code':exit_code,'script_sha256_raw':sha(__file__),
        'command_arguments':{'root':str(args.root),'trial':str(args.trial_dir),'baseline':str(args.baseline_dir)}})
    write_new(output/'artifact_manifest.json',{p.name:{'sha256':sha(p),'bytes':p.stat().st_size}
        for p in output.iterdir() if p.is_file() and p.name!='artifact_manifest.json'})
    print(json.dumps({'evidence_valid':result['evidence_valid'],'research_acceptance_passed':False,
        'counts':result.get('counts'),'error':result.get('error'),'failures':result.get('failures'),
        'report':str(output/'report.json')},ensure_ascii=False,indent=2))
    raise SystemExit(exit_code)


if __name__=='__main__':
    main()
