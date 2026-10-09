"""Predeclared full-five-scene 2x target translation research experiment.

Only target linear velocity changes. The same 27 s horizon also doubles
accumulated target translation. This is not delay/model-error robustness or
wall deployment qualification. Existing artifacts and safety gates remain.
"""
from __future__ import annotations
import argparse
import copy
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import traceback

import numpy as np

from v6_lite.run_evidence import _write_json
from v6_lite.run_research_acceptance import (
    audit_research_timing, qualify_research_suite, sha,
)
from v6_lite.run_test_profiles import current_source_snapshot, source_provenance

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REFERENCE = ROOT / "v6_lite/output/runs/research_acceptance_01"
REFERENCE_COMMIT = "9cd1831e7c77a340464a669eac3b5aa7cd9c3c3c"
REFERENCE_MANIFEST_SHA = "379b9977a441c540a20cfc50a9e90614a1e0985fa2a5a73bbb7c9690f823eec0"
VELOCITY_KEY = "target_satellite_linear_velocity_m_s"
SCENE_IDS = [f"v6_lite_scenario_{i:02d}" for i in range(5)]
REQUIRED_COMMITTED_PATHS = (
    "v6_lite/run_research_velocity_stress.py", "v6_lite/test_research_velocity_stress.py",
    "v6_lite/run_v6_lite.py", "v6_lite/test_target_velocity_scale.py",
    "v6_lite/test_fixtures/target_velocity_baseline.json",
    "v6_lite/run_research_acceptance.py", "v6_lite/run_test_profiles.py",
    "v6_lite/validate_v6_lite.py", "v6_lite/audit_b2_online_interval_recompute.py",
    "v6_lite/hierarchical_qp.py", "v6_lite/runtime_command.py",
    "model_test/whole_body_verifier_v5.py",
)


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode("utf-8")).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"),
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))


def load_frozen_reference(reference_dir):
    """Use the original accepted cohort; no replacement reference or goldens."""
    reference_dir = Path(reference_dir).resolve()
    manifest_path = reference_dir / "manifest.json"
    if sha(manifest_path) != REFERENCE_MANIFEST_SHA:
        raise ValueError("original research reference manifest identity changed")
    manifest = read_json(manifest_path)
    normalized = {name.replace("\\", "/"): digest for name, digest in manifest.items()}
    actual = {path.relative_to(reference_dir).as_posix() for path in reference_dir.rglob("*")
              if path.is_file() and path.name != "manifest.json"}
    if len(normalized) != len(manifest) or set(normalized) != actual:
        raise ValueError("original research reference file set changed")
    for name, digest in normalized.items():
        path = (reference_dir / name).resolve()
        if not path.is_relative_to(reference_dir) or sha(path) != digest:
            raise ValueError("original research reference artifact changed: " + name)
    report = read_json(reference_dir / "report.json")
    provenance = report["source_provenance"]
    if (report.get("passed") is not True or report.get("complete") is not True
            or provenance.get("source_unchanged") is not True
            or provenance["before"]["git_commit"] != REFERENCE_COMMIT
            or provenance["after"]["git_commit"] != REFERENCE_COMMIT):
        raise ValueError("original research reference is not the accepted frozen run")
    metrics = read_json(reference_dir / "simulation/v6_lite_metrics.json")
    metadata = read_json(reference_dir / "simulation/run_metadata.json")
    scenes = [scene["scenario"] for scene in metrics["scenarios"]]
    if [scene["scenario_id"] for scene in scenes] != SCENE_IDS or scenes != metadata["scenarios"]:
        raise ValueError("original five scenario definitions disagree")
    return {"source_commit": REFERENCE_COMMIT, "manifest_sha256": sha(manifest_path),
            "metrics_sha256": sha(reference_dir / "simulation/v6_lite_metrics.json"),
            "scenario_definitions_sha256": canonical_sha(scenes), "scenarios": scenes,
            "run_config": metrics["run_config"], "qp_config": metrics["qp_config"]}


def compare_declared_scenes(original, scale_one, scale_two):
    """All fields match; only the already sampled linear velocity scales."""
    try:
        expected_two = copy.deepcopy(original)
        for scene in expected_two:
            scene[VELOCITY_KEY] = [2.0 * value for value in scene[VELOCITY_KEY]]
        checks = {
            "original_ordered_five_scenes": len(original) == len(scale_one) == len(scale_two) == 5
                and [s["scenario_id"] for s in original] == SCENE_IDS,
            "scale_one_exact_original_definitions": scale_one == original,
            "scale_two_only_linear_velocity_doubled": scale_two == expected_two,
            "finite_serializable_definitions": bool(canonical_sha(original) and canonical_sha(scale_one)
                                                   and canonical_sha(scale_two)),
        }
    except (KeyError, TypeError, ValueError):
        checks = {"valid_declared_scene_definitions": False}
    return {"passed": all(checks.values()), "checks": checks,
            "failures": [name for name, passed in checks.items() if not passed]}


def qualify_velocity_stress_suite(payload, scenes, run_config, qp_config, applied_factor):
    """Retain every research gate, then bind results to the exact new cohort."""
    try:
        base = qualify_research_suite(payload)
        checks = dict(base["checks"])
        actual_scenes = [scene["scenario"] for scene in payload.get("scenarios", [])]
        checks.update({
            "declared_twofold_velocity_factor": payload.get("run_config", {}).get("target_linear_velocity_scale") == 2.0,
            "exact_declared_run_config": payload.get("run_config") == run_config,
            "exact_declared_qp_and_capsule_cbf": payload.get("qp_config") == qp_config
                and qp_config.get("enable_capsule_cbf") is True,
            "exact_declared_ordered_scenario_definitions": actual_scenes == scenes
                and [scene["scenario_id"] for scene in actual_scenes] == SCENE_IDS
                and canonical_sha(actual_scenes) == canonical_sha(scenes),
            "declared_factor_applied_to_physical_initial_velocity": bool(applied_factor and applied_factor.get("passed") is True),
        })
    except (KeyError, TypeError, ValueError, AttributeError):
        checks = {"valid_full_stress_suite_evidence": False}
    return {"passed": all(checks.values()), "checks": checks,
            "failures": [name for name, passed in checks.items() if not passed]}


def applied_factor_audit(payload, reference_dir, declared_scenes, suite_dir, run_config):
    """Verify the actual initial state, using named free-joint model addresses."""
    from model_test.whole_body_verifier_v5 import WholeBodyCollisionVerifier, WholeBodyVerificationConfig, WorkspaceSphere
    from v6_lite.hierarchical_qp import free_joint_slices
    from v6_lite.run_v6_lite import default_v6_lite_robot_spec
    baseline = read_json(Path(reference_dir) / "simulation/v6_lite_metrics.json")
    baseline_scenes = {s["scenario"]["scenario_id"]: s for s in baseline["scenarios"]}
    records = []
    spec = default_v6_lite_robot_spec()
    for item, declared in zip(payload.get("scenarios", []), declared_scenes):
        scene_id = declared["scenario_id"]
        checks = {}
        record = {"scenario_id": scene_id, "checks": checks}
        try:
            if item["scenario"]["scenario_id"] != scene_id:
                raise ValueError("physical factor audit scene order mismatch")
            source = Path(item["trace"]["path"]).resolve()
            previous = Path(baseline_scenes[scene_id]["trace"]["path"]).resolve()
            source_sha, previous_sha = sha(source), sha(previous)
            checks["new_and_baseline_trace_identity"] = (source.is_relative_to(Path(suite_dir).resolve())
                and previous.is_relative_to(Path(reference_dir).resolve())
                and source_sha == item["trace"]["sha256"]
                and previous_sha == baseline_scenes[scene_id]["trace"]["sha256"])
            obstacles = tuple(WorkspaceSphere(name=o["name"], center=np.asarray(o["center_w"]), radius=float(o["radius_m"]))
                              for o in declared["workspace_obstacles"])
            verifier = WholeBodyCollisionVerifier(spec, obstacles, WholeBodyVerificationConfig(
                minimum_clearance=run_config["whole_body_minimum_clearance_m"], query_distance_max=2.5,
                adaptive_subdivisions=run_config["verification_subdivisions"],
                self_collision_ancestor_exclusion_depth=3, include_target_satellite_pairs=True))
            _, target = free_joint_slices(verifier.model, spec.target_free_joint_name)
            with np.load(source, allow_pickle=False) as trace, np.load(previous, allow_pickle=False) as old:
                actual, before = trace["initial_qvel"].copy(), old["initial_qvel"].copy()
                initial_qpos, old_initial_qpos = trace["initial_qpos"].copy(), old["initial_qpos"].copy()
            if (actual.shape != (verifier.model.nv,) or before.shape != actual.shape
                    or target.stop-target.start != 6 or not np.all(np.isfinite(actual)) or not np.all(np.isfinite(before))):
                raise ValueError("invalid physical initial velocity layout")
            if (initial_qpos.shape != (verifier.model.nq,) or old_initial_qpos.shape != initial_qpos.shape
                    or not np.all(np.isfinite(initial_qpos)) or not np.all(np.isfinite(old_initial_qpos))):
                raise ValueError("invalid physical initial position layout")
            checks["all_initial_position_unchanged"] = np.array_equal(initial_qpos, old_initial_qpos)
            linear = slice(target.start, target.start+3)
            angular = slice(target.start+3, target.stop)
            other = np.ones(verifier.model.nv, dtype=bool)
            other[target] = False
            checks["actual_twofold_linear_velocity"] = np.array_equal(actual[linear], np.asarray(declared[VELOCITY_KEY]))
            checks["original_angular_velocity"] = (np.array_equal(actual[angular], np.asarray(declared["target_satellite_angular_velocity_rad_s"]))
                and np.array_equal(actual[angular], before[angular]))
            checks["all_other_initial_velocity_unchanged"] = np.array_equal(actual[other], before[other])
            checks["baseline_linear_velocity_and_factor"] = (np.array_equal(before[linear], np.asarray(baseline_scenes[scene_id]["scenario"][VELOCITY_KEY]))
                and np.array_equal(actual[linear], 2.0*before[linear]))
            record.update({"source_trace_sha256": source_sha, "baseline_trace_sha256": previous_sha,
                "target_free_joint_name": spec.target_free_joint_name, "target_dof_slice": [target.start, target.stop],
                "actual_initial_qvel": actual.tolist(), "baseline_initial_qvel": before.tolist(),
                "actual_initial_qpos": initial_qpos.tolist(), "baseline_initial_qpos": old_initial_qpos.tolist(),
                "declared_linear_velocity_m_s": declared[VELOCITY_KEY],
                "declared_angular_velocity_rad_s": declared["target_satellite_angular_velocity_rad_s"]})
        except (OSError, KeyError, TypeError, ValueError) as error:
            checks["complete_physical_factor_evidence"] = False
            record["error"] = {"type": type(error).__name__, "message": str(error)}
        record["passed"] = bool(checks and all(checks.values()))
        records.append(record)
    return {"schema": "research_applied_velocity_factor_audit_v1",
            "passed": len(records) == len(payload.get("scenarios", [])) == len(declared_scenes) == 5
                and [r["scenario_id"] for r in records] == SCENE_IDS and all(r["passed"] for r in records),
            "scenes": records, "independent_physics_replay": False,
            "scope": "exact saved initial velocity factor, not independent collision or CBF certification"}


def committed_source_identity(root, snapshot):
    """Bind required new/core files to HEAD; disclose other inventory sources."""
    root = Path(root)
    try:
        tracked = set(subprocess.check_output(["git", "ls-files"], cwd=root).decode("utf-8").splitlines())
    except (OSError, subprocess.CalledProcessError) as error:
        return {"passed": False, "required_committed_files": {}, "errors": ["Git source identity unavailable: " + str(error)],
                "untracked_inventory_sources": None, "untracked_inventory_sources_imported": None,
                "all_inventory_sources_claimed_in_head": False}
    records, errors = {}, []
    for name in REQUIRED_COMMITTED_PATHS:
        try:
            blob = subprocess.check_output(["git", "show", "HEAD:" + name], cwd=root, stderr=subprocess.PIPE)
            raw = (root / name).read_bytes()
            same = raw.replace(b"\r\n", b"\n") == blob.replace(b"\r\n", b"\n")
            records[name] = {"sha256_raw": hashlib.sha256(raw).hexdigest(), "bytes": len(raw),
                "sha256_lf_normalized": hashlib.sha256(raw.replace(b"\r\n", b"\n")).hexdigest(),
                "matches_committed_head_normalized_lf": same}
            if not same:
                errors.append("required source differs from HEAD: " + name)
        except (OSError, subprocess.CalledProcessError):
            errors.append("required source missing from HEAD: " + name)
    untracked = sorted(set(snapshot["files"]) - tracked)
    imported_paths = set()
    for module in tuple(sys.modules.values()):
        file = getattr(module, "__file__", None)
        if file:
            path = Path(file).resolve()
            if path.is_relative_to(root.resolve()):
                imported_paths.add(path.relative_to(root.resolve()).as_posix())
    imported_untracked = sorted(set(untracked) & imported_paths)
    if imported_untracked:
        errors.append("untracked inventory source was imported: " + ", ".join(imported_untracked))
    return {"passed": not errors, "required_committed_files": records, "errors": errors,
            "untracked_inventory_sources": untracked, "untracked_inventory_sources_imported": imported_untracked,
            "all_inventory_sources_claimed_in_head": False}


def summarize_trials(suite_dir, payload=None):
    """Observed attempts remain observations, never partial full acceptance."""
    suite_dir = Path(suite_dir)
    reports, read_errors = {}, []
    for name in ("research_trial_report.json", "run_failure.json"):
        path = suite_dir / name
        if path.is_file():
            try:
                reports[name] = read_json(path)
            except (OSError, ValueError):
                read_errors.append(name)
    trial = reports.get("research_trial_report.json", {})
    completed = (payload or {}).get("scenarios", trial.get("completed_scenes", []))
    completed_by_id = {s.get("scenario", {}).get("scenario_id"): s for s in completed}
    rejected = {s.get("scenario_id"): s for s in trial.get("failed_scenarios", [])}
    failure = reports.get("run_failure.json", {})
    records, observed_attempts = [], set()
    all_files = [p for p in suite_dir.rglob("*") if p.is_file()] if suite_dir.exists() else []
    for scene_id in SCENE_IDS:
        artifacts = [p.relative_to(suite_dir).as_posix() for p in all_files if scene_id in p.name]
        if scene_id in completed_by_id:
            scene = completed_by_id[scene_id]
            status = "COMPLETED_PASS" if scene.get("passed") is True else "COMPLETED_FUNCTIONAL_FAILURE"
            record = {"scenario_id": scene_id, "status": status,
                      "reported_rates": scene.get("metrics", {}).get("rates_and_latency", {}),
                      "reported_checks": scene.get("checks", {}), "artifacts": artifacts}
        elif scene_id in rejected:
            record = {"scenario_id": scene_id, "status": "REJECTED_PARTIAL",
                      "error": rejected[scene_id], "artifacts": artifacts}
        elif failure.get("scenario_id") == scene_id:
            record = {"scenario_id": scene_id, "status": "ABORTED_PARTIAL",
                      "error": failure, "artifacts": artifacts}
        elif artifacts:
            record = {"scenario_id": scene_id, "status": "PARTIAL_ARTIFACTS_ONLY", "artifacts": artifacts}
        else:
            record = {"scenario_id": scene_id, "status": "NOT_OBSERVED_ATTEMPTED", "artifacts": []}
        if record["status"] != "NOT_OBSERVED_ATTEMPTED":
            observed_attempts.add(scene_id)
        records.append(record)
    return {"schema": "research_velocity_stress_trial_summary_v1", "scenes": records,
        "observed_attempt_count": len(observed_attempts), "all_five_attempts_observed": observed_attempts == set(SCENE_IDS),
        "reported_completed_scene_count": len(completed_by_id), "reported_rejected_scene_count": len(rejected),
        "reported_full_cohort_attempts": trial.get("all_predeclared_scenes_attempted") is True,
        "read_errors": read_errors, "partial_is_full_acceptance": False,
        "reported_trial_file": "research_trial_report.json" if trial else None}


def run(output_dir, reference_dir=DEFAULT_REFERENCE):
    from v6_lite.run_v6_lite import V6LiteRunConfig, build_scenarios, default_v6_lite_robot_spec, run_suite
    from v6_lite.hierarchical_qp import HierarchicalQPConfig
    output_dir, reference_dir = Path(output_dir).resolve(), Path(reference_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    frozen = current_source_snapshot(ROOT)
    config = V6LiteRunConfig(scenario_count=5, seed=20260801, duration_s=27., physics_period_s=.002,
                            task_period_s=.020, pcc_mode="bounded_interval_pcc",
                            dispatch_clock_policy="research_simulation", target_linear_velocity_scale=2.)
    qp = HierarchicalQPConfig(enable_capsule_cbf=True)
    cfg_dict, qp_dict = asdict(config), asdict(qp)
    result = {"schema": "v6_2_research_velocity_stress_result_v1", "passed": False, "complete": False,
        "complete_definition": "all five full horizons and declared validation calls completed; acceptance additionally requires every functional/evidence gate",
        "steps": [], "algorithm_simulation": {"status": "NOT_COMPLETED"},
        "computational_performance": {"status": "NOT_MEASURED", "performance_is_functional_gate": False},
        "wall_continuation": {"status": "NOT_MET", "retested": False, "previous_goal_completed": False},
        "hardware_deployment": {"status": "NOT_ESTABLISHED"}, "hard_realtime_certified": False,
        "claim_scope": "nominal seeded target translation speed and accumulated displacement stress",
        "delay_or_model_error_robustness_established": False}
    required_identity = None
    suite_dir = output_dir / "simulation"
    payload = None

    def step(name, action):
        print("[velocity-stress] " + name, flush=True)
        before = current_source_snapshot(ROOT)
        if (not source_provenance(frozen, before)["source_unchanged"] or before["tracked_worktree_dirty"]):
            raise RuntimeError("declared source changed before " + name)
        try:
            value = action()
            record = {"name": name, "status": "COMPLETED", "result": value}
        except BaseException as error:
            value = None
            record = {"name": name, "status": "ABORTED" if isinstance(error, (KeyboardInterrupt, SystemExit)) else "FAILED",
                      "error_type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()}
        after = current_source_snapshot(ROOT)
        provenance = source_provenance(frozen, after)
        record["source_unchanged"] = provenance["source_unchanged"]
        result["steps"].append(record)
        _write_json(output_dir / "report.json", result)
        if not provenance["source_unchanged"] or after["tracked_worktree_dirty"]:
            raise RuntimeError("declared source changed during " + name)
        return value

    try:
        if frozen["capture_errors"] or frozen["tracked_worktree_dirty"] or not frozen["git_commit"]:
            raise RuntimeError("freeze clean tracked source before the velocity stress experiment")
        required_identity = committed_source_identity(ROOT, frozen)
        if not required_identity["passed"]:
            raise RuntimeError("required committed source identity failed: " + "; ".join(required_identity["errors"]))
        config.validate(); qp.validate()
        reference = load_frozen_reference(reference_dir)
        spec = default_v6_lite_robot_spec()
        scale_one = [s.to_dict() for s in build_scenarios(spec, replace(config, target_linear_velocity_scale=1.))]
        scale_two = [s.to_dict() for s in build_scenarios(spec, config)]
        declaration = compare_declared_scenes(reference["scenarios"], scale_one, scale_two)
        scale_one_config = dict(cfg_dict)
        scale_one_config.pop("target_linear_velocity_scale")
        if (not declaration["passed"] or scale_one_config != reference["run_config"]
                or qp_dict != reference["qp_config"]):
            raise ValueError("baseline definitions/configuration changed beyond the declared linear velocity factor")
        scenes = {"original": reference["scenarios"], "scale_one": scale_one, "scale_two": scale_two,
                  "comparison": declaration,
                  "canonical_sha256": {"original": canonical_sha(reference["scenarios"]),
                                       "scale_one": canonical_sha(scale_one), "scale_two": canonical_sha(scale_two)}}
        _write_json(output_dir / "scenario_definitions.json", scenes)
        _write_json(output_dir / "plan.json", {"schema": "v6_2_research_velocity_stress_plan_v1",
            "declared_before_tests_and_simulation": True, "supplements_previous_c11_goal": True,
            "source": frozen, "committed_source_identity": required_identity,
            "run_config": cfg_dict, "qp_config": qp_dict, "baseline": {k: v for k, v in reference.items() if k != "scenarios"},
            "scenario_definitions_file": "scenario_definitions.json", "scenario_definitions_file_sha256": sha(output_dir / "scenario_definitions.json"),
            "scenario_definitions_canonical_sha256": scenes["canonical_sha256"], "scenario_order": SCENE_IDS,
            "only_changed_factor": {"name": "target_linear_velocity_scale", "value": 2., "default": 1.,
                                    "also_doubles_full_horizon_target_translation": True},
            "steps": ["current_and_historical_tests", "complete_five_scene_simulation", "applied_physical_velocity_factor",
                      "simulation_qualification", "independent_torque_delivery", "independent_execution_contract", "independent_interval_recompute", "raw_timing_audit"],
            "old_c1_nontiming_parity_required": False, "safety_contract_changed": False,
            "performance_targets": {"planning_p95_ms": 20, "torque_p95_ms": 2, "research_blocking": False},
            "wall_deployment_status": "NOT_MET", "hardware_deployment": "NOT_ESTABLISHED",
            "delay_or_model_error_robustness_established": False,
            "failure_policy": "exclusive output; all five declared scenes attempted by the suite; retain failures/aborts and report unobserved attempts"})
        from v6_lite.run_test_profiles import run as test_profiles
        tests = step("current_and_historical_tests", lambda: test_profiles("all", output_dir / "test_profiles"))
        if tests is not True:
            raise RuntimeError("full current/historical regression prerequisite failed")
        payload = step("complete_five_scene_simulation", lambda: run_suite(config, qp, suite_dir, continue_failed_scenarios=True))
        result["trial_summary"] = summarize_trials(suite_dir, payload)
        if payload is None:
            trial_path = suite_dir / "research_trial_report.json"
            partial = read_json(trial_path).get("completed_scenes", []) if trial_path.is_file() else []
            if partial:
                timing = step("completed_scene_partial_timing_observation", lambda: audit_research_timing({"scenarios": partial}, suite_dir))
                if timing:
                    timing["observed_completed_scene_targets_met"] = timing["performance_target_met"]
                    timing["performance_target_met"] = None
                    timing["evidence_valid"] = False
                    timing["status"] = "INCOMPLETE_COHORT_OBSERVATION"
                    result["computational_performance"] = timing
            for name in ("applied_physical_velocity_factor", "simulation_qualification", "independent_torque_delivery", "independent_execution_contract",
                         "independent_interval_recompute", "raw_timing_audit"):
                result["steps"].append({"name": name, "status": "NOT_RUN", "reason": "complete five-scene metrics unavailable; partial evidence cannot qualify"})
            raise RuntimeError("five-scene simulation failed or aborted; retained partial trials do not qualify")
        def factor_check():
            factor = applied_factor_audit(payload, reference_dir, scale_two, suite_dir, cfg_dict)
            _write_json(output_dir / "applied_factor_audit.json", factor)
            return factor
        factor = step("applied_physical_velocity_factor", factor_check)
        qualification = step("simulation_qualification", lambda: qualify_velocity_stress_suite(payload, scale_two, cfg_dict, qp_dict, factor))
        from v6_lite.validate_v6_lite import validate_delivery, validate_execution_contract
        delivery = step("independent_torque_delivery", lambda: validate_delivery(ROOT / "v6_lite", suite_dir, acceptance_profile="research_simulation"))
        execution = step("independent_execution_contract", lambda: validate_execution_contract(suite_dir))
        from v6_lite.audit_b2_online_interval_recompute import run as interval_recompute
        interval = step("independent_interval_recompute", lambda: interval_recompute(suite_dir, output_dir / "interval_recompute"))
        timing = step("raw_timing_audit", lambda: audit_research_timing(payload, suite_dir))
        result["algorithm_simulation"] = {"status": "PASSED" if all(x and x.get("passed") is True for x in (qualification, delivery, execution, interval)) else "FAILED",
            "qualification": qualification, "applied_physical_velocity_factor": factor,
            "delivery": delivery, "execution": execution, "interval": interval,
            "old_c1_nontiming_parity_required": False}
        result["computational_performance"] = timing or {"status": "EVIDENCE_INVALID", "performance_is_functional_gate": False}
        result["complete"] = bool(qualification and qualification["checks"].get("complete_fixed_rate_execution")
                                  and all(x["status"] == "COMPLETED" for x in result["steps"]))
        result["passed"] = bool(result["complete"] and result["algorithm_simulation"]["status"] == "PASSED"
                                and timing and timing.get("evidence_valid") is True)
    except BaseException as error:
        result["error"] = {"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()}
    final_source = current_source_snapshot(ROOT)
    result["source_provenance"] = source_provenance(frozen, final_source)
    final_identity = committed_source_identity(ROOT, final_source)
    result["committed_source_identity"] = {"before": required_identity, "after": final_identity}
    required_unchanged = bool(required_identity and final_identity["passed"]
                              and required_identity["required_committed_files"] == final_identity["required_committed_files"])
    result["required_committed_files_unchanged"] = required_unchanged
    result["passed"] = bool(result["passed"] and required_unchanged and result["source_provenance"]["source_unchanged"]
                            and not final_source["tracked_worktree_dirty"])
    result.setdefault("trial_summary", summarize_trials(suite_dir, payload))
    _write_json(output_dir / "report.json", result)
    _write_json(output_dir / "manifest.json", {p.relative_to(output_dir).as_posix(): sha(p)
                for p in sorted(output_dir.rglob("*")) if p.is_file() and p.name != "manifest.json"})
    print(json.dumps({"passed": result["passed"], "complete": result["complete"],
                      "report": str(output_dir / "report.json"), "wall_continuation": "NOT_MET"}), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    raise SystemExit(0 if run(args.output_dir)["passed"] else 1)


if __name__ == "__main__":
    main()
