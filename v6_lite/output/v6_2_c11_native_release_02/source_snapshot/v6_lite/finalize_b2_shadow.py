"""Generate a hash-bound B.2 stage-2 gate report from immutable audits."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def _gate_blockers(frontier: dict, refined: dict) -> dict[str, dict]:
    """Separate witnessed proxy penetration from old-velocity CBF failures."""

    if frontier["budgets"][-1] != 511 or frontier["gate_m"] != .005:
        raise ValueError("the frozen high-budget witness protocol changed")
    blockers = {}
    for mode in ("baseline", "enabled"):
        entries = [item for item in frontier["entries"] if item["mode"] == mode]
        records = [item for item in refined["records"] if item["mode"] == mode]
        witnessed_below = []
        for entry in entries:
            last = entry["budget_ladder"][-1]
            if last["point_budget"] != 511:
                raise ValueError("missing high-budget witness result")
            if last["proxy_status"] == "PROXY_CLEARANCE_BELOW_GATE":
                if not last["upper_m"] < frontier["gate_m"]:
                    raise ValueError("below-gate status lacks a point witness")
                witnessed_below.append(entry)
        start_bad = [item for item in records if item["feasibility"]["status"]
                     == "START_CLEARANCE_VIOLATION"]
        initial = [item for item in records if item["tick"] == 0]
        positive_h_bad = [item for item in start_bad
                          if item["worst_interval_start_terms"] is not None
                          and item["worst_interval_start_terms"]["h_m"] >= 0.0]
        if (len(witnessed_below) != frontier["modes"][mode]["status_by_budget"]
                ["511"].get("PROXY_CLEARANCE_BELOW_GATE", 0)
                or len(start_bad) != refined["modes"][mode]
                ["diagnostic_refined_start_violation_count"]
                or len(positive_h_bad) != refined["modes"][mode]
                ["start_violations_with_positive_interval_h"]
                or len(initial) != 5
                or len({item["scenario_id"] for item in initial}) != 5):
            raise ValueError(f"gate blocker totals disagree with source: {mode}")

        def earliest_by_scenario(items: list[dict]) -> dict[str, int]:
            first = {}
            for item in items:
                name = item["scenario_id"]
                first[name] = min(first.get(name, item["tick"]), item["tick"])
            return dict(sorted(first.items()))

        blockers[mode] = {
            "initial_proxy_safe_and_frozen_rows_feasible": sum(
                item["proxy_status"] == "PROXY_CLEARANCE_AT_LEAST_GATE"
                and item["feasibility"]["status"] == "FROZEN_ROWS_FEASIBLE"
                for item in initial),
            "sampled_proxy_below_gate_with_point_witness": len(witnessed_below),
            "first_witnessed_below_tick_by_scenario": earliest_by_scenario(
                witnessed_below),
            "sampled_old_start_clearance_violations_after_refinement": len(start_bad),
            "first_old_start_violation_tick_by_scenario": earliest_by_scenario(
                start_bad),
            "old_start_violations_with_nonnegative_interval_h": len(positive_h_bad),
        }
    return blockers


def finalize(root: Path, output_dir: Path) -> dict:
    root = Path(root)
    output_dir.mkdir(parents=True, exist_ok=False)
    sources = {
        "cold_shadow": root / "shadow_a1" / "shadow_report.json",
        "warm_shadow": root / "shadow_warm" / "shadow_report.json",
        "near_gate": root / "near_gate_audit" / "near_gate_audit.json",
        "b1_holdout_geometry": root / "heldout_geometry" / "heldout_geometry_report.json",
        "budget_frontier": root / "budget_frontier" / "budget_frontier.json",
        "refined_start": root / "refined_start" / "refined_start_report.json",
        "repartition_counterfactual": root / "repartition_counterfactual" / "repartition_counterfactual.json",
        "repartition_handoff": root / "repartition_handoff" / "repartition_handoff.json",
        "weighted_qp_probe": root / "qp_probe_early" / "weighted_qp_probe.json",
        "initial_qp_probe": root / "qp_probe_initial255" / "weighted_qp_probe.json",
        "initial_qp_probe_failure": root / "qp_probe_initial255_failures" / "attempt1.json",
        "candidate_ramp_early": root / "candidate_ramp_early" / "candidate_ramp_summary.json",
        "candidate_prediction": root / "candidate_prediction_early" / "candidate_prediction_summary.json",
        "candidate_prediction_failure": root / "candidate_prediction_early_failures" / "attempt1.json",
        "repeated_qp_probe": root / "repeated_qp_probe_early" / "repeated_probe_summary.json",
        "blas_thread_trial": root / "blas_thread_trial" / "abba_summary.json",
        "mujoco_sphere_screen": root / "mujoco_sphere_screen" / "sphere_screen_summary.json",
        "weighted_qp_sphere_trial": root / "qp_sphere_trial" / "qp_sphere_summary.json",
        "private_five_scene_rollout": root / "private_rollout_400_summary" / "private_five_scene_summary.json",
        "private_rollout_recompute": root / "private_recompute_400" / "private_recompute_summary.json",
        "private_qp_sphere_trial": root / "private_qp_sphere_trial" / "private_qp_sphere_summary.json",
        "private_qp_sphere_full": root / "private_qp_sphere_full" / "private_full_qp_sphere_summary.json",
        "servo_subspace_origin": root / "servo_subspace_origin" / "servo_subspace_origin_summary.json",
        "discrete_servo_first_tick": root / "discrete_servo_first_tick" / "discrete_servo_summary.json",
        "discrete_private_five_scene": root / "discrete_private_400_summary" / "discrete_private_five_scene_summary.json",
        "discrete_private_recompute": root / "discrete_private_recompute_400" / "private_recompute_summary.json",
        "discrete_screened_scene00": root / "discrete_screened_private_rollout_400" / "scene_00" / "private_rollout_summary.json",
        "discrete_private_paired_qp": root / "discrete_private_full_qp_sphere" / "private_full_qp_sphere_summary.json",
        "discrete_screened_parity": root / "discrete_screened_parity" / "discrete_screened_parity_summary.json",
        "full_state_envelope": root / "full_state_envelope" / "envelope_sweep_summary.json",
        "microstep_envelope": root / "microstep_envelope" / "microstep_envelope_summary.json",
        "full_torque_envelope": root / "full_torque_envelope" / "full_torque_envelope_summary.json",
        "batched_point_trial": root / "batched_query_trial" / "batched_query_summary.json",
        "prepared_cold_trial": root / "prepared_query_trial" / "prepared_query_summary.json",
        "prepared_full_trace": root / "prepared_full_trace" / "prepared_full_trace_summary.json",
        "root_rescue_frontier": root / "root_rescue_frontier" / "root_rescue_summary.json",
        "full_root255_census": root / "full_root255_census" / "full_root255_summary.json",
    }
    values = {name: json.loads(path.read_text(encoding="utf-8"))
              for name, path in sources.items()}
    cold, warm, near, heldout, frontier, refined, repartition, handoff, probe, initial_probe, initial_failure, candidate_ramp, candidate_prediction, prediction_failure, repeated_qp, blas_trial, sphere_screen, qp_sphere, private_rollout, private_recompute, private_qp_sphere, private_qp_full, servo_origin, discrete_first_tick, discrete_private, discrete_recompute, screened_scene, paired_discrete_qp, screened_parity, sweep, micro, full_torque, batched, prepared_cold, prepared_full, root_rescue, full_root = (
        values[name] for name in sources
    )
    sources.update({
        "prepared_500hz_envelope": root / "prepared_envelope_500hz" /
            "prepared_envelope_summary.json",
        "prepared_private_query": root / "prepared_private_query_2005" /
            "prepared_private_query_summary.json",
        "admm_solver_trial": root / "admm_solver_trial_2000" /
            "admm_solver_trial_summary.json",
        "optimized_private_recompute": root / "optimized_private_recompute_400" /
            "private_recompute_summary.json",
        "optimized_private_five_scene": root / "optimized_private_400_summary" /
            "optimized_private_five_scene_summary.json",
        "prepared_screened_scene00": root / "prepared_screened_private_rollout_400" /
            "scene_00" / "private_rollout_summary.json",
        "optimized_private_repeat_protocol": root / "optimized_private_repeats_3x5" /
            "repeat_protocol.json",
        "optimized_private_repeats": root / "optimized_private_repeats_3x5" /
            "repeat_summary.json",
        "qp_phase_probe": root / "qp_phase_probe_scene02_400" /
            "qp_phase_summary.json",
        "cholesky_private_recompute": root / "cholesky_private_recompute_400" /
            "private_recompute_summary.json",
        "cholesky_repeat_protocol": root / "cholesky_private_repeats_3x5" /
            "repeat_protocol.json",
        "cholesky_repeat_summary": root / "cholesky_private_repeats_3x5" /
            "repeat_summary.json",
        "cholesky_private_five_scene": root / "cholesky_private_400_summary" /
            "cholesky_private_five_scene_summary.json",
        "cholesky_full_scene03": root / "cholesky_private_full_scene03_1350" /
            "private_rollout_summary.json",
        "cholesky_full_scene03_recompute": root /
            "cholesky_private_full_scene03_recompute" /
            "private_recompute_summary.json",
        "cholesky_full_scene03_budget": root /
            "cholesky_private_full_scene03_budget" /
            "unified_budget_summary.json",
        "full_private_five_gate": root / "full_private_five_gate" /
            "private_five_gate_summary.json",
        "strict_domain_stop": root / "strict_cholesky_scene01_900" /
            "private_rollout_summary.json",
        "strict_domain_replay": root / "strict_domain_replay_scene01_900" /
            "strict_domain_replay.json",
        "domain_endpoint_private_five": root / "domain_endpoint_private_five_gate" /
            "domain_private_five_summary.json",
        "domain_endpoint_private_five_visual": root /
            "domain_endpoint_private_five_gate" / "b2-domain-private-five.png",
        "domain_endpoint_private_five_visual_meta": root /
            "domain_endpoint_private_five_gate" / "b2-domain-private-five.json",
    })
    optimized = json.loads(sources["optimized_private_five_scene"].read_text(
        encoding="utf-8"))
    optimized_dir = root / "optimized_private_400_summary"
    optimized_manifest = json.loads((optimized_dir /
        "optimized_private_five_scene_manifest.json").read_text(encoding="utf-8"))
    prepared_screened = json.loads(sources["prepared_screened_scene00"].read_text(
        encoding="utf-8"))
    prepared_screened_dir = root / "prepared_screened_private_rollout_400" / "scene_00"
    prepared_screened_manifest = json.loads((prepared_screened_dir /
        "private_rollout_manifest.json").read_text(encoding="utf-8"))
    for label, filename in {
        "summary": "private_rollout_summary.json",
        "records": "private_rollout_records.jsonl",
        "trace": "private_rollout_trace.npz",
        "document": "PRIVATE_ROLLOUT.md",
    }.items():
        if prepared_screened_manifest[f"{label}_sha256"] != _sha(
                prepared_screened_dir / filename):
            raise ValueError(f"prepared screened {label} hash changed")
    for name, digest in prepared_screened["source_sha256"].items():
        if _source_sha(Path("v6_lite") / name) != digest:
            raise ValueError(f"prepared screened source changed: {name}")
    if (optimized_manifest["summary_sha256"]
            != _sha(sources["optimized_private_five_scene"])
            or optimized_manifest["document_sha256"]
            != _sha(optimized_dir / "OPTIMIZED_PRIVATE_FIVE_SCENE.md")
            or optimized["generator_source_sha256"]
            != _source_sha(Path("v6_lite/finalize_b2_optimized_private.py"))
            or optimized["prepared_envelope_summary_sha256"]
            != _sha(sources["prepared_500hz_envelope"])
            or optimized["prepared_query_summary_sha256"]
            != _sha(sources["prepared_private_query"])
            or optimized["admm_solver_trial_summary_sha256"]
            != _sha(sources["admm_solver_trial"])
            or optimized["independent_recompute_summary_sha256"]
            != _sha(sources["optimized_private_recompute"])
            or optimized["scene_count"] != 5
            or optimized["ticks_per_scene"] != 400
            or optimized["checked_500hz_states"] != 20005
            or optimized["checked_task_states"] != 2005
            or optimized["independent_interval_rows"] != 5257
            or optimized["independent_recompute_failure_count"]
            or len(optimized["p95_over_20ms_scenes"]) != 3
            or optimized["production_online_controller_changed"]
            or optimized["stage3_admission"]
            or optimized["full_cycle_20ms_acceptance"]
            or optimized["continuous_time_certified"]):
        raise ValueError("optimized private diagnostic hashes or gate changed")
    if (prepared_screened["scenario_id"] != "v6_lite_scenario_00"
            or prepared_screened["executed_ticks"] != 400
            or prepared_screened["private_preflight_plus_qp_timing"]["p95_ms"] <= 20.0
            or prepared_screened["production_online_controller_changed"]
            or prepared_screened["stage3_admission"]):
        raise ValueError("prepared screened timing failure changed")
    repeat_dir = root / "optimized_private_repeats_3x5"
    repeat_protocol = json.loads(sources["optimized_private_repeat_protocol"].read_text(
        encoding="utf-8"))
    repeat = json.loads(sources["optimized_private_repeats"].read_text(
        encoding="utf-8"))
    repeat_manifest = json.loads((repeat_dir / "repeat_manifest.json").read_text(
        encoding="utf-8"))
    if (repeat_manifest["protocol_sha256"]
            != _sha(sources["optimized_private_repeat_protocol"])
            or repeat_manifest["summary_sha256"]
            != _sha(sources["optimized_private_repeats"])
            or repeat_manifest["document_sha256"]
            != _sha(repeat_dir / "REPEAT_TIMING.md")
            or repeat["protocol_sha256"]
            != _sha(sources["optimized_private_repeat_protocol"])
            or repeat_protocol["round_scene_order"] != [
                [f"v6_lite_scenario_{i:02d}" for i in order]
                for order in ((0, 1, 2, 3, 4), (4, 3, 2, 1, 0),
                              (0, 1, 2, 3, 4))]
            or repeat_protocol["ticks_per_scene"] != 400
            or repeat_protocol["diagnostic_budget_ms"] != 20.0
            or repeat["run_count"] != 15
            or repeat["round_count"] != 3
            or repeat["scene_count"] != 5
            or repeat["p95_over_20ms_run_count"] != 13
            or repeat["status"] != "PRIVATE_REPEAT_TIMING_GATE_NOT_MET"
            or repeat["production_online_controller_changed"]
            or repeat["stage3_admission"]
            or repeat["full_cycle_20ms_acceptance"]
            or repeat["continuous_time_certified"]):
        raise ValueError("optimized repeat timing protocol or summary changed")
    for name, digest in repeat_protocol["source_sha256"].items():
        if _source_sha(Path("v6_lite") / name) != digest:
            raise ValueError(f"optimized repeat timing source changed: {name}")
    expected_runs = [(round_number, scene_id)
                     for round_number, order in enumerate(
                         repeat_protocol["round_scene_order"], start=1)
                     for scene_id in order]
    if ([(item["round"], item["scenario_id"]) for item in repeat["runs"]]
            != expected_runs):
        raise ValueError("optimized repeat scene order changed")
    for item in repeat["runs"]:
        scene_index = int(item["scenario_id"][-2:])
        folder = repeat_dir / f"round_{item['round']:02d}" / f"scene_{scene_index:02d}"
        summary_path = folder / "private_rollout_summary.json"
        manifest_path = folder / "private_rollout_manifest.json"
        trace_path = folder / "private_rollout_trace.npz"
        records_path = folder / "private_rollout_records.jsonl"
        document_path = folder / "PRIVATE_ROLLOUT.md"
        baseline_trace = (root / "discrete_private_rollout_400" /
                          f"scene_{scene_index:02d}" / "private_rollout_trace.npz")
        if (item["summary_sha256"] != _sha(summary_path)
                or item["manifest_sha256"] != _sha(manifest_path)
                or item["trace_sha256"] != _sha(trace_path)
                or item["baseline_trace_sha256"] != _sha(baseline_trace)
                or item["trace_sha256"] != item["baseline_trace_sha256"]):
            raise ValueError(f"optimized repeat run changed: {folder}")
        run_summary = json.loads(summary_path.read_text(encoding="utf-8"))
        run_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (run_summary["private_preflight_plus_qp_timing"]
                != item["preflight_plus_qp_timing"]
                or run_manifest["summary_sha256"] != _sha(summary_path)
                or run_manifest["trace_sha256"] != _sha(trace_path)
                or run_manifest["records_sha256"] != _sha(records_path)
                or run_manifest["document_sha256"] != _sha(document_path)
                or run_summary["records_sha256"] != _sha(records_path)
                or run_summary["stage3_admission"]):
            raise ValueError(f"optimized repeat records changed: {folder}")
    if repeat["p95_over_20ms_runs"] != [
            {"round": item["round"], "scenario_id": item["scenario_id"],
             "p95_ms": item["preflight_plus_qp_timing"]["p95_ms"]}
            for item in repeat["runs"]
            if item["preflight_plus_qp_timing"]["p95_ms"] > 20.0]:
        raise ValueError("optimized repeat timing failure list changed")
    cholesky = json.loads(sources["cholesky_private_five_scene"].read_text(
        encoding="utf-8"))
    qp_phase = json.loads(sources["qp_phase_probe"].read_text(
        encoding="utf-8"))
    cholesky_dir = root / "cholesky_private_400_summary"
    cholesky_manifest = json.loads((cholesky_dir /
        "cholesky_private_five_scene_manifest.json").read_text(encoding="utf-8"))
    if (cholesky_manifest["summary_sha256"]
            != _sha(sources["cholesky_private_five_scene"])
            or cholesky_manifest["document_sha256"]
            != _sha(cholesky_dir / "CHOLESKY_PRIVATE_FIVE_SCENE.md")
            or cholesky["generator_source_sha256"]
            != _source_sha(Path("v6_lite/finalize_b2_cholesky_private.py"))
            or cholesky["phase_probe_summary_sha256"]
            != _sha(sources["qp_phase_probe"])
            or cholesky["independent_recompute_summary_sha256"]
            != _sha(sources["cholesky_private_recompute"])
            or cholesky["repeat_protocol_sha256"]
            != _sha(sources["cholesky_repeat_protocol"])
            or cholesky["repeat_summary_sha256"]
            != _sha(sources["cholesky_repeat_summary"])
            or cholesky["status"]
            != "PRIVATE_400_TICK_TIMING_DIAGNOSTIC_PASS_NOT_ONLINE"
            or cholesky["scene_count"] != 5
            or cholesky["repeat_run_count"] != 15
            or cholesky["repeat_p95_over_20ms_count"]
            or cholesky["repeat_over_20ms_cycle_count"] != 4
            or cholesky["independent_task_states"] != 2005
            or cholesky["independent_interval_rows"] != 5257
            or cholesky["independent_failure_count"]
            or cholesky["production_online_controller_changed"]
            or cholesky["stage3_admission"]
            or cholesky["full_cycle_20ms_acceptance"]
            or cholesky["continuous_time_certified"]):
        raise ValueError("cholesky private evidence or scope changed")
    if (qp_phase["scenario_id"] != "v6_lite_scenario_02"
            or qp_phase["ticks"] != 400
            or qp_phase["dense_solve_call_counts"] != [2]
            or qp_phase["production_online_controller_changed"]
            or qp_phase["stage3_admission"]):
        raise ValueError("QP phase timing diagnostic changed")
    cholesky_repeat = json.loads(sources["cholesky_repeat_summary"].read_text(
        encoding="utf-8"))
    cholesky_protocol = json.loads(sources["cholesky_repeat_protocol"].read_text(
        encoding="utf-8"))
    cholesky_repeat_dir = root / "cholesky_private_repeats_3x5"
    cholesky_repeat_manifest = json.loads((cholesky_repeat_dir /
        "repeat_manifest.json").read_text(encoding="utf-8"))
    if (cholesky_repeat_manifest["protocol_sha256"]
            != _sha(sources["cholesky_repeat_protocol"])
            or cholesky_repeat_manifest["summary_sha256"]
            != _sha(sources["cholesky_repeat_summary"])
            or cholesky_repeat_manifest["document_sha256"]
            != _sha(cholesky_repeat_dir / "REPEAT_TIMING.md")
            or cholesky_repeat["protocol_sha256"]
            != _sha(sources["cholesky_repeat_protocol"])
            or cholesky_protocol["ticks_per_scene"] != 400
            or cholesky_protocol["diagnostic_budget_ms"] != 20.0
            or cholesky_repeat["p95_over_20ms_run_count"]
            or cholesky_repeat["cholesky_unconstrained_fallback_count"]
            or cholesky_repeat["run_count"] != 15):
        raise ValueError("cholesky repeat provenance changed")
    for name, digest in cholesky_protocol["source_sha256"].items():
        if _source_sha(Path("v6_lite") / name) != digest:
            raise ValueError(f"cholesky repeat source changed: {name}")
    for scene_index, item in enumerate(cholesky["scenes"]):
        folder = root / "cholesky_private_rollout_400" / f"scene_{scene_index:02d}"
        trace = folder / "private_rollout_trace.npz"
        baseline_trace = (root / "discrete_private_rollout_400" /
                          f"scene_{scene_index:02d}" / "private_rollout_trace.npz")
        if (item["summary_sha256"] != _sha(folder / "private_rollout_summary.json")
                or item["manifest_sha256"]
                != _sha(folder / "private_rollout_manifest.json")
                or item["trace_sha256"] != _sha(trace)
                or item["baseline_trace_sha256"] != _sha(baseline_trace)
                or item["trace_sha256"] != item["baseline_trace_sha256"]
                or any(item["exact_trace_parity"].values())
                or item["preflight_plus_qp_timing"]["p95_ms"] > 20.0):
            raise ValueError(f"cholesky first-round scene changed: {scene_index}")
    if (len(cholesky_repeat["runs"]) != 15
            or any(item["preflight_plus_qp_timing"]["p95_ms"] > 20.0
                   for item in cholesky_repeat["runs"])):
        raise ValueError("cholesky repeat p95 result changed")
    for item in cholesky_repeat["runs"]:
        scene_index = int(item["scenario_id"][-2:])
        folder = (cholesky_repeat_dir / f"round_{item['round']:02d}" /
                  f"scene_{scene_index:02d}")
        trace = folder / "private_rollout_trace.npz"
        if (item["summary_sha256"] != _sha(folder / "private_rollout_summary.json")
                or item["manifest_sha256"]
                != _sha(folder / "private_rollout_manifest.json")
                or item["trace_sha256"] != _sha(trace)
                or item["trace_sha256"]
                != cholesky["scenes"][scene_index]["trace_sha256"]):
            raise ValueError(f"cholesky repeat raw run changed: {folder}")
    full_scene_dir = root / "cholesky_private_full_scene03_1350"
    full_scene = json.loads(sources["cholesky_full_scene03"].read_text(
        encoding="utf-8"))
    full_scene_manifest = json.loads((full_scene_dir /
        "private_rollout_manifest.json").read_text(encoding="utf-8"))
    for label, filename in {
        "summary": "private_rollout_summary.json",
        "records": "private_rollout_records.jsonl",
        "trace": "private_rollout_trace.npz",
        "document": "PRIVATE_ROLLOUT.md",
    }.items():
        if full_scene_manifest[f"{label}_sha256"] != _sha(full_scene_dir / filename):
            raise ValueError(f"full-scene private {label} hash changed")
    full_scene_timing = full_scene["private_preflight_plus_qp_timing"]
    if (full_scene_manifest["failure_sha256"] is not None
            or full_scene["scenario_id"] != "v6_lite_scenario_03"
            or full_scene["predeclared_horizon_ticks"] != 1350
            or full_scene["attempted_ticks"] != 1350
            or full_scene["executed_ticks"] != 1350
            or full_scene["stop_reason"] != "HORIZON_COMPLETE"
            or full_scene["trace_sha256"] != full_scene_manifest["trace_sha256"]
            or full_scene["records_sha256"] != full_scene_manifest["records_sha256"]
            or full_scene["cholesky_unconstrained_call_count"] != 1350
            or full_scene["cholesky_unconstrained_fallback_count"] != 0
            or not full_scene["strict_online_domain_all_executed_ticks"]
            or full_scene_timing["count"] != 1350
            or full_scene_timing["p95_ms"] > 20.0
            or full_scene["production_online_controller_changed"]
            or full_scene["stage3_admission"]
            or full_scene["full_cycle_20ms_acceptance"]
            or full_scene["continuous_time_certified"]):
        raise ValueError("full-scene private scope or timing changed")
    for name, digest in full_scene["source_sha256"].items():
        if _source_sha(Path("v6_lite") / name) != digest:
            raise ValueError(f"full-scene private source changed: {name}")
    full_recompute_dir = root / "cholesky_private_full_scene03_recompute"
    full_recompute = json.loads(sources["cholesky_full_scene03_recompute"].read_text(
        encoding="utf-8"))
    full_recompute_manifest = json.loads((full_recompute_dir /
        "private_recompute_manifest.json").read_text(encoding="utf-8"))
    for label, filename in {
        "summary": "private_recompute_summary.json",
        "checks": "private_recompute_checks.jsonl",
        "interval_rows": "private_recompute_interval_rows.jsonl",
        "failures": "private_recompute_failures.jsonl",
        "document": "PRIVATE_RECOMPUTE.md",
    }.items():
        if full_recompute_manifest[f"{label}_sha256"] != _sha(
                full_recompute_dir / filename):
            raise ValueError(f"full-scene recompute {label} hash changed")
    if (full_recompute["input_layout"] != "direct_single_scene"
            or full_recompute["scenes"] != ["v6_lite_scenario_03"]
            or full_recompute["task_ticks_per_scene"] != 1350
            or full_recompute["torque_steps_per_scene"] != 13500
            or full_recompute["checked_task_states"] != 1351
            or full_recompute["checked_interval_rows"] != 2371
            or full_recompute["failure_count"] != 0
            or not full_recompute["pass_recompute"]
            or not full_recompute["independent_strict_online_domain_all_executed_ticks"]
            or not full_recompute["all_500hz_states_on_declared_shape_subspace"]
            or full_recompute["torque_limit_violation_count"] != 0
            or full_recompute["inputs"][0]["summary_sha256"]
            != _sha(sources["cholesky_full_scene03"])
            or full_recompute["inputs"][0]["trace_sha256"]
            != full_scene["trace_sha256"]
            or full_recompute["inputs"][0]["records_sha256"]
            != full_scene["records_sha256"]
            or full_recompute["new_mode_online_admitted"]
            or full_recompute["full_cycle_20ms_acceptance"]
            or full_recompute["continuous_time_certified"]
            or (full_recompute_dir / "private_recompute_failures.jsonl").stat().st_size):
        raise ValueError("full-scene private recompute or scope changed")
    for name, digest in full_recompute["source_sha256"].items():
        if _source_sha(Path("v6_lite") / name) != digest:
            raise ValueError(f"full-scene recompute source changed: {name}")
    budget_dir = root / "cholesky_private_full_scene03_budget"
    budget_audit = json.loads(sources["cholesky_full_scene03_budget"].read_text(
        encoding="utf-8"))
    budget_manifest = json.loads((budget_dir /
        "unified_budget_manifest.json").read_text(encoding="utf-8"))
    for label, filename in {
        "summary": "unified_budget_summary.json",
        "rows": "unified_budget_rows.jsonl",
        "document": "UNIFIED_BUDGET.md",
    }.items():
        if budget_manifest[f"{label}_sha256"] != _sha(budget_dir / filename):
            raise ValueError(f"full-scene unified budget {label} hash changed")
    if (budget_audit["scenario_id"] != "v6_lite_scenario_03"
            or budget_audit["task_ticks"] != 1350
            or budget_audit["budget_protocol"] != "bounded_cold_shadow_default"
            or budget_audit["frozen_shadow_budget"]
            != cold["modes"]["enabled"]["budget"]
            or budget_audit["budget_overrun_tick_count"] != 0
            or budget_audit["parity_failure_count"] != 0
            or budget_audit["query_budget_exhausted_count"] != 0
            or not budget_audit["within_frozen_shadow_budget_on_saved_states"]
            or budget_audit["inputs"]["private_summary_sha256"]
            != _sha(sources["cholesky_full_scene03"])
            or budget_audit["inputs"]["private_records_sha256"]
            != full_scene["records_sha256"]
            or budget_audit["inputs"]["private_trace_sha256"]
            != full_scene["trace_sha256"]
            or budget_audit["inputs"]["independent_recompute_sha256"]
            != _sha(sources["cholesky_full_scene03_recompute"])
            or budget_audit["action_time_budget_enforced_by_private_controller"]
            or budget_audit["production_online_controller_changed"]
            or budget_audit["stage3_admission"]
            or budget_audit["full_control_time_evaluated"]
            or budget_audit["continuous_time_certified"]):
        raise ValueError("full-scene unified budget or scope changed")
    for name, digest in budget_audit["source_sha256"].items():
        if _source_sha(Path("v6_lite") / name) != digest:
            raise ValueError(f"full-scene budget source changed: {name}")
    full_private_gate_dir = root / "full_private_five_gate"
    full_private_gate = json.loads(sources["full_private_five_gate"].read_text(
        encoding="utf-8"))
    full_private_gate_manifest = json.loads((full_private_gate_dir /
        "private_five_gate_manifest.json").read_text(encoding="utf-8"))
    for label, filename in {
        "summary": "private_five_gate_summary.json",
        "failures": "private_five_gate_failures.jsonl",
        "document": "FULL_PRIVATE_FIVE_GATE.md",
    }.items():
        if full_private_gate_manifest[f"{label}_sha256"] != _sha(
                full_private_gate_dir / filename):
            raise ValueError(f"full private five {label} changed")
    if (full_private_gate["generator_source_sha256"]
            != _source_sha(Path("v6_lite/finalize_b2_full_private_five.py"))
            or full_private_gate["scene_count"] != 5
            or full_private_gate["executed_task_ticks"] != 6750
            or full_private_gate["recomputed_task_states"] != 6755
            or full_private_gate["recomputed_interval_rows"] != 12549
            or full_private_gate["strict_domain_violation_tick_count"] != 63
            or full_private_gate["track_commands_issued_outside_declared_domain"]
            != 63
            or full_private_gate["failure_records_sha256"]
            != _sha(full_private_gate_dir / "private_five_gate_failures.jsonl")
            or full_private_gate["stage2_status"] != "GATE_NOT_MET"
            or full_private_gate["stage3_admission"]
            or full_private_gate["production_online_controller_changed"]):
        raise ValueError("full private five scope or gate changed")
    if [(item["scenario_id"], item["strict_domain_violation_ticks"])
            for item in full_private_gate["scenes"]] != [
                (f"v6_lite_scenario_{index:02d}", 63 if index == 1 else 0)
                for index in range(5)]:
        raise ValueError("full private five domain counterexample changed")
    strict_stop = json.loads(sources["strict_domain_stop"].read_text(
        encoding="utf-8"))
    strict_replay = json.loads(sources["strict_domain_replay"].read_text(
        encoding="utf-8"))
    strict_dir = root / "strict_cholesky_scene01_900"
    strict_manifest = json.loads((strict_dir / "private_rollout_manifest.json")
                                 .read_text(encoding="utf-8"))
    if (strict_manifest["summary_sha256"] != _sha(sources["strict_domain_stop"])
            or strict_manifest["trace_sha256"] != _sha(
                strict_dir / "private_rollout_trace.npz")
            or strict_manifest["records_sha256"] != _sha(
                strict_dir / "private_rollout_records.jsonl")
            or strict_stop["executed_ticks"] != 887
            or strict_stop["attempted_ticks"] != 888
            or strict_stop["first_preflight_domain_rejection_tick"] != 887
            or strict_stop["stop_reason"]
            != "RAMP_MICROSTATE_OUTSIDE_DECLARED_WORK_DOMAIN"
            or not strict_stop["strict_gate_rejected_before_servo"]
            or strict_stop["failure_record"]["selected_command"] is not None
            or strict_stop["failure_record"]["next_servo_step_executed"] is not False
            or strict_replay["executed_ticks"] != 887
            or strict_replay["native_torque_steps"] != 8870
            or strict_replay["checked_native_states"] != 8871
            or strict_replay["native_replay_max_state_error"] > 1e-8
            or strict_replay["minimum_executed_work_domain_margin_rad"] < 0
            or strict_replay["first_rejected_candidate_tick"] != 887
            or strict_replay["first_predicted_outside_servo_substep"] != 4
            or not strict_replay["rejected_candidate_excluded_from_torque_trace"]
            or strict_replay["independent_interval_rows_recomputed_here"]
            or strict_replay["production_online_admitted"]):
        raise ValueError("strict domain stop or independent replay changed")
    for name, digest in strict_stop["source_sha256"].items():
        if _source_sha(Path("v6_lite") / name) != digest:
            raise ValueError(f"strict domain source changed: {name}")
    if strict_replay["source_sha256"] != _source_sha(
            Path("v6_lite/audit_b2_strict_domain_replay.py")):
        raise ValueError("strict domain replay source changed")
    for name, digest in strict_replay["inputs_sha256"].items():
        filename = {"summary": "private_rollout_summary.json",
                    "manifest": "private_rollout_manifest.json",
                    "records": "private_rollout_records.jsonl",
                    "trace": "private_rollout_trace.npz",
                    "failure": "private_rollout_failure.json"}[name]
        if _sha(strict_dir / filename) != digest:
            raise ValueError(f"strict replay input changed: {name}")
    domain_five_dir = root / "domain_endpoint_private_five_gate"
    domain_five = json.loads(sources["domain_endpoint_private_five"].read_text(
        encoding="utf-8"))
    domain_five_manifest = json.loads((domain_five_dir /
        "domain_private_five_manifest.json").read_text(encoding="utf-8"))
    if (domain_five_manifest["summary_sha256"]
            != _sha(sources["domain_endpoint_private_five"])
            or domain_five_manifest["document_sha256"]
            != _sha(domain_five_dir / "DOMAIN_PRIVATE_FIVE.md")
            or domain_five_manifest["failures_sha256"]
            != _sha(domain_five_dir / "domain_private_five_failures.jsonl")
            or domain_five["generator_source_sha256"] != _source_sha(
                Path("v6_lite/finalize_b2_domain_private_five.py"))
            or domain_five["private_diagnostic_status"]
            != "PASS_PRIVATE_FIVE_WITH_LIMITS"
            or domain_five["stage2_online_gate_status"] != "GATE_NOT_MET"
            or domain_five["scene_count"] != 5
            or domain_five["executed_task_ticks"] != 6750
            or domain_five["independent_recomputed_task_states"] != 6755
            or domain_five["independent_recomputed_interval_rows"] != 12350
            or domain_five["independent_checked_2ms_states"] != 67505
            or domain_five["failure_count"] != 0
            or domain_five["preflight_plus_qp_over_20ms_tick_count"] != 6
            or domain_five["stage3_admission"]
            or domain_five["full_cycle_20ms_acceptance"]):
        raise ValueError("new private five evidence or scope changed")
    for name, entry in domain_five_manifest["sources"].items():
        if _sha(Path(entry["path"])) != entry["sha256"]:
            raise ValueError(f"new private five source changed: {name}")
    domain_visual_meta = json.loads(sources[
        "domain_endpoint_private_five_visual_meta"].read_text(encoding="utf-8"))
    if (domain_visual_meta["image_sha256"] != _sha(sources[
            "domain_endpoint_private_five_visual"])
            or domain_visual_meta["summary_sha256"]
            != _sha(sources["domain_endpoint_private_five"])
            or domain_visual_meta["manifest_sha256"] != _sha(
                domain_five_dir / "domain_private_five_manifest.json")
            or domain_visual_meta["stage2_online_gate_status"] != "GATE_NOT_MET"):
        raise ValueError("new private five visualization changed")
    if not cold["passed_as_read_only_audit"] or not warm["passed_as_read_only_audit"]:
        raise ValueError("native replay shadow integrity failed")
    if not near["passed"] or not heldout["passed"]:
        raise ValueError("frozen geometry audit failed")
    if frontier["input_shadow"]["sha256"] != _sha(sources["warm_shadow"]):
        raise ValueError("budget frontier is not tied to the current warm shadow")
    if refined["input_frontier"]["sha256"] != _sha(sources["budget_frontier"]):
        raise ValueError("refined start is not tied to the current budget frontier")
    if repartition["input_shadow"]["sha256"] != _sha(sources["warm_shadow"]):
        raise ValueError("repartition counterfactual is not tied to the current warm shadow")
    if (repartition["point_budget"] != 64 or repartition["gate_m"] != .005
            or repartition["repartition_applied_to_execution"]):
        raise ValueError("repartition counterfactual changed the frozen comparison")
    for mode in ("baseline", "enabled"):
        counts = repartition["modes"][mode]["counts"]
        expected = warm["modes"][mode]["counts"].get("warm_all_task_unknown", 0)
        expected_ticks = warm["modes"][mode]["summaries"]["warm_all_task_query_ms"]["count"]
        if (counts.get("warm_UNKNOWN_CROSSES_GATE", 0) != expected
                or counts.get("task_ticks", 0) != expected_ticks):
            raise ValueError(f"repartition counterfactual warm count changed: {mode}")
    if handoff["input_repartition"]["sha256"] != _sha(sources["repartition_counterfactual"]):
        raise ValueError("repartition handoff is not tied to the current counterfactual")
    if (handoff["repartition_admitted_online"]
            or len(handoff["native_replay_checks"]) != 10
            or any(item["max_state_error"] > 1e-8
                   for item in handoff["native_replay_checks"])):
        raise ValueError("repartition handoff native replay integrity failed")
    for mode in ("baseline", "enabled"):
        expected_samples = warm["modes"][mode]["counts"].get("UNKNOWN_CROSSES_GATE", 0)
        if handoff["modes"][mode]["sample_count"] != expected_samples:
            raise ValueError(f"repartition handoff sampled count changed: {mode}")
    if (len(refined["native_replay_checks"]) != 10
            or any(x["max_state_error"] > 1e-8
                   for x in refined["native_replay_checks"])):
        raise ValueError("refined native torque replay integrity failed")
    if (not probe["passed_as_read_only_integrity"]
            or probe["online_control_changed"]
            or probe["new_mode_closed_loop_acceptance"]
            or probe["probe_ticks"] != [50, 100, 150]
            or len(probe["native_replay_checks"]) != 10
            or len(probe["records"]) != 30
            or probe["input_refined_start_sha256"] != _sha(sources["refined_start"])):
        raise ValueError("weighted QP probe integrity or scope failed")
    if probe["source_hash_newline_policy"] != "LF_NORMALIZED":
        raise ValueError("weighted QP probe source hash policy changed")
    for name, digest in probe["source_sha256"].items():
        if _source_sha(Path("v6_lite") / name) != digest:
            raise ValueError(f"weighted QP probe source changed: {name}")
    for item in probe["inputs"].values():
        if _sha(Path(item["metrics_path"])) != item["metrics_sha256"]:
            raise ValueError("weighted QP probe metrics changed")
        for trace in item["traces"]:
            if _sha(Path(trace["path"])) != trace["sha256"]:
                raise ValueError("weighted QP probe trace changed")
    if (not initial_probe["passed_as_read_only_integrity"]
            or initial_probe["online_control_changed"]
            or initial_probe["new_mode_closed_loop_acceptance"]
            or initial_probe["probe_ticks"] != [0]
            or initial_probe["probe_protocol"] != "initial_preflight"
            or initial_probe["point_budget"] != 255
            or initial_probe["source_hash_newline_policy"] != "LF_NORMALIZED"
            or initial_probe["input_refined_start_sha256"]
            != _sha(sources["refined_start"])
            or len(initial_probe["native_replay_checks"]) != 10
            or any(item["max_state_error"] > 1e-8
                   for item in initial_probe["native_replay_checks"])
            or len(initial_probe["records"]) != 10
            or initial_probe["source_sha256"] != probe["source_sha256"]):
        raise ValueError("initial weighted QP probe integrity or scope failed")
    for item in initial_probe["inputs"].values():
        if (_sha(Path(item["metrics_path"])) != item["metrics_sha256"]
                or len(item["traces"]) != 5):
            raise ValueError("initial weighted QP five-scenario metrics changed")
        for trace in item["traces"]:
            if _sha(Path(trace["path"])) != trace["sha256"]:
                raise ValueError("initial weighted QP trace changed")
    for mode in ("baseline", "enabled"):
        item = initial_probe["summary"][mode]
        if (item["probe_count"] != 5
                or item["query_budget_ok_count"] != 5
                or item["proxy_safe_count"] != 5
                or item["envelope_supported_count"] != 5
                or item["validated_command_count"] != 5
                or item["admission_preconditions_met_count"] != 5):
            raise ValueError("initial QP preflight did not retain five validated starts")
        rows = [row for row in initial_probe["records"] if row["mode"] == mode]
        if (len(rows) != 5 or len({row["scenario_id"] for row in rows}) != 5
                or any(row["tick"] != 0 or row["selected_interval_count"] != 0
                       or row["query_points"] != 5
                       or row["selected_command"] is None for row in rows)):
            raise ValueError("initial QP records changed")
    initial_dir = root / "qp_probe_initial255"
    initial_manifest = json.loads((initial_dir / "weighted_qp_probe_manifest.json")
                                  .read_text(encoding="utf-8"))
    if (initial_manifest["report_sha256"] != _sha(sources["initial_qp_probe"])
            or initial_manifest["document_sha256"]
            != _sha(initial_dir / "WEIGHTED_QP_PROBE.md")):
        raise ValueError("initial weighted QP output hash changed")
    if (initial_failure["status"] != "SCRIPT_ERROR_BEFORE_QP_SOLVE"
            or initial_failure["exception"] != "TypeError: 'float' object is not iterable"
            or initial_failure["failure_output_file_count"] != 0):
        raise ValueError("initial probe first-attempt failure record changed")
    if (candidate_ramp["input_probe_sha256"] != _sha(sources["weighted_qp_probe"])
            or candidate_ramp["source_hash_newline_policy"] != "LF_NORMALIZED"
            or candidate_ramp["new_interval_mode_executed"]
            or candidate_ramp["continuous_time_certified"]
            or not candidate_ramp["private_counterfactual_servo_executed"]
            or candidate_ramp["probe_ticks"] != [50, 100, 150]
            or candidate_ramp["servo_steps_per_task"] != 10
            or len(candidate_ramp["records"]) != 30):
        raise ValueError("one-ramp candidate branch provenance failed")
    for name, digest in candidate_ramp["source_sha256"].items():
        if _source_sha(Path("v6_lite") / name) != digest:
            raise ValueError(f"one-ramp candidate source changed: {name}")
    for item in candidate_ramp["inputs"].values():
        if (_sha(Path(item["metrics_path"])) != item["metrics_sha256"]
                or len(item["traces"]) != 5):
            raise ValueError("one-ramp candidate metrics changed")
        for trace in item["traces"]:
            if _sha(Path(trace["path"])) != trace["sha256"]:
                raise ValueError("one-ramp candidate trace changed")
    candidate_dir = root / "candidate_ramp_early"
    candidate_outputs = {
        "summary": "candidate_ramp_summary.json",
        "microstates": "candidate_ramp_states.jsonl",
        "failures": "candidate_ramp_failures.jsonl",
        "document": "CANDIDATE_RAMP.md",
    }
    candidate_manifest = json.loads((candidate_dir / "candidate_ramp_manifest.json")
                                    .read_text(encoding="utf-8"))
    for name, filename in candidate_outputs.items():
        if candidate_manifest[f"{name}_sha256"] != _sha(candidate_dir / filename):
            raise ValueError(f"one-ramp candidate {name} hash changed")
    if (candidate_ramp["microstate_records_sha256"]
            != _sha(candidate_dir / candidate_outputs["microstates"])
            or candidate_ramp["failure_records_sha256"]
            != _sha(candidate_dir / candidate_outputs["failures"])):
        raise ValueError("one-ramp candidate records changed")
    if (candidate_prediction["input_probe_sha256"]
            != _sha(sources["weighted_qp_probe"])
            or candidate_prediction["input_candidate_ramp_sha256"]
            != _sha(sources["candidate_ramp_early"])
            or candidate_prediction["source_hash_newline_policy"] != "LF_NORMALIZED"
            or candidate_prediction["new_interval_mode_executed"]
            or candidate_prediction["continuous_time_certified"]
            or not candidate_prediction["margin_is_empirical_not_certified_bound"]
            or candidate_prediction["probe_ticks"] != [50, 100, 150]
            or len(candidate_prediction["cases"]) != 30):
        raise ValueError("candidate fixed-partition prediction provenance failed")
    for name, digest in candidate_prediction["source_sha256"].items():
        if _source_sha(Path("v6_lite") / name) != digest:
            raise ValueError(f"candidate prediction source changed: {name}")
    for item in candidate_prediction["inputs"].values():
        if (_sha(Path(item["metrics_path"])) != item["metrics_sha256"]
                or len(item["traces"]) != 5):
            raise ValueError("candidate prediction metrics changed")
        for trace in item["traces"]:
            if _sha(Path(trace["path"])) != trace["sha256"]:
                raise ValueError("candidate prediction trace changed")
    prediction_dir = root / "candidate_prediction_early"
    prediction_outputs = {
        "summary": "candidate_prediction_summary.json",
        "rows": "candidate_prediction_rows.jsonl",
        "failures": "candidate_prediction_failures.jsonl",
        "document": "CANDIDATE_PREDICTION.md",
    }
    prediction_manifest = json.loads((prediction_dir / "candidate_prediction_manifest.json")
                                     .read_text(encoding="utf-8"))
    for name, filename in prediction_outputs.items():
        if prediction_manifest[f"{name}_sha256"] != _sha(prediction_dir / filename):
            raise ValueError(f"candidate prediction {name} hash changed")
    if (candidate_prediction["row_records_sha256"]
            != _sha(prediction_dir / prediction_outputs["rows"])
            or candidate_prediction["failure_records_sha256"]
            != _sha(prediction_dir / prediction_outputs["failures"])):
        raise ValueError("candidate prediction records changed")
    partial_dir = root / "candidate_prediction_early_failures" / "attempt1_partial"
    if (prediction_failure["status"] != "SCRIPT_ERROR_AFTER_TWO_COMPLETED_PROBES"
            or prediction_failure["full_result_generated"]
            or prediction_failure["partial_row_count"] != 5
            or prediction_failure["partial_rows_sha256"]
            != _sha(partial_dir / "candidate_prediction_rows.jsonl")
            or prediction_failure["partial_failure_rows_sha256"]
            != _sha(partial_dir / "candidate_prediction_failures.jsonl")):
        raise ValueError("candidate prediction first-attempt failure changed")
    if (repeated_qp["input_original_probe_sha256"]
            != _sha(sources["weighted_qp_probe"])
            or repeated_qp["source_hash_newline_policy"] != "LF_NORMALIZED"
            or repeated_qp["new_interval_mode_executed"]
            or repeated_qp["full_control_cycle_measured"]
            or repeated_qp["round_count"] != 5
            or repeated_qp["complete_round_count"] != 5
            or repeated_qp["probe_ticks"] != [50, 100, 150]
            or repeated_qp["point_budget"] != 63
            or repeated_qp["candidate_mismatch_count"] != 0):
        raise ValueError("repeated read-only QP timing provenance failed")
    for name, digest in repeated_qp["source_sha256"].items():
        if _source_sha(Path("v6_lite") / name) != digest:
            raise ValueError(f"repeated QP probe source changed: {name}")
    repeated_dir = root / "repeated_qp_probe_early"
    repeated_outputs = {
        "summary": "repeated_probe_summary.json",
        "records": "repeated_probe_records.jsonl",
        "document": "REPEATED_QP_PROBE.md",
    }
    repeated_manifest = json.loads((repeated_dir / "repeated_probe_manifest.json")
                                   .read_text(encoding="utf-8"))
    for name, filename in repeated_outputs.items():
        if repeated_manifest[f"{name}_sha256"] != _sha(repeated_dir / filename):
            raise ValueError(f"repeated QP probe {name} hash changed")
    if repeated_qp["records_sha256"] != _sha(repeated_dir / repeated_outputs["records"]):
        raise ValueError("repeated QP probe records changed")
    for round_index, item in enumerate(repeated_qp["rounds"], 1):
        if (item["round"] != round_index or item["status"] != "COMPLETE"
                or item["candidate_mismatch_count"] != 0
                or item["record_count"] != 30
                or item["report_sha256"] != _sha(Path(item["report_path"]))):
            raise ValueError("repeated QP probe round changed")
        round_dir = Path(item["report_path"]).parent
        if (item["document_sha256"] != _sha(round_dir / "WEIGHTED_QP_PROBE.md")
                or item["manifest_sha256"]
                != _sha(round_dir / "weighted_qp_probe_manifest.json")):
            raise ValueError("repeated QP probe round outputs changed")
    if (blas_trial["input_original_probe_sha256"]
            != _sha(sources["weighted_qp_probe"])
            or blas_trial["source_hash_newline_policy"] != "LF_NORMALIZED"
            or blas_trial["new_interval_mode_executed"]
            or blas_trial["full_control_cycle_measured"]
            or blas_trial["complete_group_count"] != 4
            or blas_trial["candidate_mismatch_count"] != 0
            or blas_trial["probe_ticks"] != [50, 100, 150]
            or blas_trial["point_budget"] != 63
            or [item["requested_openblas_threads"] for item in blas_trial["groups"]]
            != [2, 1, 1, 2]):
        raise ValueError("OpenBLAS ABBA trial provenance failed")
    for name, digest in blas_trial["source_sha256"].items():
        if _source_sha(Path("v6_lite") / name) != digest:
            raise ValueError(f"OpenBLAS ABBA trial source changed: {name}")
    blas_dir = root / "blas_thread_trial"
    blas_manifest = json.loads((blas_dir / "abba_manifest.json").read_text(encoding="utf-8"))
    for label, filename in (("summary", "abba_summary.json"),
                            ("records", "abba_records.jsonl"),
                            ("document", "BLAS_THREAD_TRIAL.md")):
        if blas_manifest[f"{label}_sha256"] != _sha(blas_dir / filename):
            raise ValueError(f"OpenBLAS ABBA trial {label} hash changed")
    if blas_trial["records_sha256"] != _sha(blas_dir / "abba_records.jsonl"):
        raise ValueError("OpenBLAS ABBA trial records changed")
    for item in blas_trial["groups"]:
        folder = blas_dir / item["name"]
        if (item["status"] != "COMPLETE" or item["record_count"] != 30
                or item["candidate_mismatch_count"] != 0
                or item["report_sha256"] != _sha(folder / "weighted_qp_probe.json")
                or item["document_sha256"] != _sha(folder / "WEIGHTED_QP_PROBE.md")
                or item["manifest_sha256"]
                != _sha(folder / "weighted_qp_probe_manifest.json")
                or item["stdout_sha256"] != _sha(blas_dir / f"{item['name']}_stdout.txt")
                or item["stderr_sha256"] != _sha(blas_dir / f"{item['name']}_stderr.txt")):
            raise ValueError("OpenBLAS ABBA trial group outputs changed")
    if (sphere_screen["record_count"] != 70
            or sphere_screen["failure_count"] != 0
            or sphere_screen["probe_ticks"]
            != [50, 100, 150, 250, 350, 500, 1000]
            or sphere_screen["query_max_m"] != .09
            or sphere_screen["activation_m"] != .08
            or sphere_screen["source_hash_newline_policy"] != "LF_NORMALIZED"
            or sphere_screen["new_interval_mode_executed"]
            or sphere_screen["online_controller_changed"]
            or sphere_screen["full_control_cycle_measured"]):
        raise ValueError("MuJoCo sphere screen protocol changed")
    for name, digest in sphere_screen["source_sha256"].items():
        if _source_sha(Path("v6_lite") / name) != digest:
            raise ValueError(f"MuJoCo sphere screen source changed: {name}")
    sphere_dir = root / "mujoco_sphere_screen"
    sphere_manifest = json.loads((sphere_dir / "sphere_screen_manifest.json")
                                 .read_text(encoding="utf-8"))
    for label, filename in (("summary", "sphere_screen_summary.json"),
                            ("records", "sphere_screen_records.jsonl"),
                            ("failures", "sphere_screen_failures.jsonl"),
                            ("document", "MUJOCO_SPHERE_SCREEN.md")):
        if sphere_manifest[f"{label}_sha256"] != _sha(sphere_dir / filename):
            raise ValueError(f"MuJoCo sphere screen {label} hash changed")
    if (sphere_screen["records_sha256"]
            != _sha(sphere_dir / "sphere_screen_records.jsonl")
            or sphere_screen["failures_sha256"]
            != _sha(sphere_dir / "sphere_screen_failures.jsonl")):
        raise ValueError("MuJoCo sphere screen raw records changed")
    for item in sphere_screen["inputs"]:
        metrics = Path("v6_lite/output/v6_2_a1") / f"{item['mode']}_root" / "output" / "v6_lite_metrics.json"
        if (_sha(metrics) != item["metrics_sha256"]
                or _sha(Path(item["trace_path"])) != item["trace_sha256"]):
            raise ValueError("MuJoCo sphere screen input hashes changed")
    for mode in ("baseline", "enabled"):
        item = sphere_screen["modes"][mode]
        if (item["state_count"] != 35 or item["pair_count_per_state"] != 2927
                or item["missed_active_count"] != 0
                or item["max_active_distance_error_m"] != 0.0
                or item["max_active_witness_error_m"] != 0.0
                or item["max_minimum_error_m"] != 0.0):
            raise ValueError("MuJoCo sphere screen did not preserve frozen rows")
    if (qp_sphere["input_original_probe_sha256"]
            != _sha(sources["weighted_qp_probe"])
            or qp_sphere["source_hash_newline_policy"] != "LF_NORMALIZED"
            or qp_sphere["online_controller_changed"]
            or qp_sphere["new_interval_mode_executed"]
            or qp_sphere["full_control_cycle_measured"]
            or qp_sphere["groups_predeclared"] != [
                {"name": "a1", "method": "reference"},
                {"name": "b1", "method": "sphere_screen"},
                {"name": "b2", "method": "sphere_screen"},
                {"name": "a2", "method": "reference"},
            ]
            or qp_sphere["probe_ticks"] != [50, 100, 150]
            or qp_sphere["point_budget"] != 63
            or qp_sphere["complete_group_count"] != 4
            or qp_sphere["candidate_mismatch_count"] != 0):
        raise ValueError("weighted QP sphere-screen provenance failed")
    for name, digest in qp_sphere["source_sha256"].items():
        if _source_sha(Path("v6_lite") / name) != digest:
            raise ValueError(f"weighted QP sphere-screen source changed: {name}")
    qp_sphere_dir = root / "qp_sphere_trial"
    qp_sphere_manifest = json.loads((qp_sphere_dir / "qp_sphere_manifest.json")
                                    .read_text(encoding="utf-8"))
    for label, filename in (("summary", "qp_sphere_summary.json"),
                            ("records", "qp_sphere_records.jsonl"),
                            ("document", "QP_SPHERE_TRIAL.md")):
        if qp_sphere_manifest[f"{label}_sha256"] != _sha(qp_sphere_dir / filename):
            raise ValueError(f"weighted QP sphere-screen {label} hash changed")
    if qp_sphere["records_sha256"] != _sha(qp_sphere_dir / "qp_sphere_records.jsonl"):
        raise ValueError("weighted QP sphere-screen records changed")
    for group in qp_sphere["groups"]:
        folder = qp_sphere_dir / group["name"]
        if (group["status"] != "COMPLETE" or group["record_count"] != 30
                or group["candidate_mismatch_count"] != 0
                or group["report_sha256"] != _sha(folder / "weighted_qp_probe.json")
                or group["document_sha256"] != _sha(folder / "WEIGHTED_QP_PROBE.md")
                or group["manifest_sha256"]
                != _sha(folder / "weighted_qp_probe_manifest.json")
                or group["stdout_sha256"]
                != _sha(qp_sphere_dir / f"{group['name']}_stdout.txt")):
            raise ValueError("weighted QP sphere-screen group outputs changed")
    for mode in ("baseline", "enabled"):
        if any(qp_sphere["modes"][mode][method]["record_count"] != 30
               for method in ("reference", "sphere_screen")):
            raise ValueError("weighted QP sphere-screen mode count changed")
    if (private_rollout["input_refined_start_sha256"]
            != _sha(sources["refined_start"])
            or private_rollout["source_hash_newline_policy"] != "LF_NORMALIZED"
            or private_rollout["predeclared_horizon_ticks_per_scene"] != 400
            or private_rollout["predeclared_scenarios"]
            != [f"v6_lite_scenario_{index:02d}" for index in range(5)]
            or private_rollout["complete_horizon_count"] != 5
            or private_rollout["old_violation_tick_reached_count"] != 5
            or private_rollout["old_violation_tick_start_satisfied_count"] != 5
            or private_rollout["strict_online_admission"]
            or private_rollout["new_mode_five_scene_acceptance"]
            or private_rollout["full_cycle_20ms_acceptance"]):
        raise ValueError("private multi-cycle scope or provenance changed")
    for name, digest in private_rollout["source_sha256"].items():
        if _source_sha(Path("v6_lite") / name) != digest:
            raise ValueError(f"private multi-cycle source changed: {name}")
    private_summary_dir = root / "private_rollout_400_summary"
    private_manifest = json.loads((private_summary_dir /
                                   "private_five_scene_manifest.json")
                                  .read_text(encoding="utf-8"))
    if (private_manifest["summary_sha256"]
            != _sha(sources["private_five_scene_rollout"])
            or private_manifest["document_sha256"]
            != _sha(private_summary_dir / "PRIVATE_FIVE_SCENE.md")):
        raise ValueError("private multi-cycle aggregate changed")
    for scene in private_rollout["scenes"]:
        folder = Path(scene["folder"])
        report_path = folder / "private_rollout_summary.json"
        item = json.loads(report_path.read_text(encoding="utf-8"))
        if (scene["summary_sha256"] != _sha(report_path)
                or scene["manifest_sha256"]
                != _sha(folder / "private_rollout_manifest.json")
                or scene["executed_ticks"] != 400
                or scene["stop_reason"] != "HORIZON_COMPLETE"
                or scene["private_start_status_at_old_violation_tick"]
                != "START_ROWS_SATISFIED"
                or item["native_replay_max_qpos_error"] > 1e-8
                or item["production_online_controller_changed"]
                or item["stage3_admission"]
                or item["strict_online_domain_all_executed_ticks"]):
            raise ValueError("private multi-cycle scene evidence changed")
        for name, digest in item["source_sha256"].items():
            if _source_sha(Path("v6_lite") / name) != digest:
                raise ValueError(f"private scene source changed: {name}")
        scene_manifest = json.loads((folder / "private_rollout_manifest.json")
                                    .read_text(encoding="utf-8"))
        for label, filename in (("summary", "private_rollout_summary.json"),
                                ("records", "private_rollout_records.jsonl"),
                                ("trace", "private_rollout_trace.npz"),
                                ("document", "PRIVATE_ROLLOUT.md")):
            if scene_manifest[f"{label}_sha256"] != _sha(folder / filename):
                raise ValueError(f"private scene {label} changed")
    if (not private_recompute["pass_recompute"]
            or private_recompute["failure_count"] != 0
            or private_recompute["checked_task_states"] != 2005
            or private_recompute["checked_interval_rows"] != 5257
            or private_recompute["maximum_native_state_error"] > 1e-8
            or private_recompute["maximum_query_error_m"] > 1e-8
            or private_recompute["maximum_start_slack_error_m_s"] > 1e-7
            or private_recompute["new_mode_online_admitted"]
            or private_recompute["full_cycle_20ms_acceptance"]
            or private_recompute["continuous_time_certified"]
            or not private_recompute["shared_model_geometry_with_private_controller"]
            or private_recompute["source_hash_newline_policy"] != "LF_NORMALIZED"):
        raise ValueError("private torque independent recomputation changed")
    for name, digest in private_recompute["source_sha256"].items():
        if _source_sha(Path("v6_lite") / name) != digest:
            raise ValueError(f"private torque recomputation source changed: {name}")
    recompute_dir = root / "private_recompute_400"
    recompute_manifest = json.loads((recompute_dir / "private_recompute_manifest.json")
                                    .read_text(encoding="utf-8"))
    for label, filename in (("summary", "private_recompute_summary.json"),
                            ("checks", "private_recompute_checks.jsonl"),
                            ("interval_rows", "private_recompute_interval_rows.jsonl"),
                            ("failures", "private_recompute_failures.jsonl"),
                            ("document", "PRIVATE_RECOMPUTE.md")):
        if recompute_manifest[f"{label}_sha256"] != _sha(recompute_dir / filename):
            raise ValueError(f"private torque recomputation {label} changed")
    for item in private_recompute["inputs"]:
        if (item["summary_sha256"] != _sha(Path(item["summary_path"]))
                or item["trace_sha256"]
                != _sha(Path(item["summary_path"]).parent /
                        "private_rollout_trace.npz")
                or item["records_sha256"]
                != _sha(Path(item["summary_path"]).parent /
                        "private_rollout_records.jsonl")):
            raise ValueError("private recomputation input changed")
    if (private_qp_sphere["schema"] != "v6_2_b2_private_qp_sphere_abba_v1"
            or private_qp_sphere["scope"]
            != "read_only_QP_on_private_torque_replayed_frozen_states"
            or private_qp_sphere["source_hash_newline_policy"] != "LF_NORMALIZED"
            or private_qp_sphere["groups_predeclared"] != [
                {"name": "a1", "method": "reference"},
                {"name": "b1", "method": "sphere_screen"},
                {"name": "b2", "method": "sphere_screen"},
                {"name": "a2", "method": "reference"},
            ]
            or private_qp_sphere["ticks_per_scene"] != [1, 50, 100, 200, 300, 399]
            or private_qp_sphere["point_budget"] != 255
            or private_qp_sphere["record_count"] != 120
            or private_qp_sphere["failure_count"] != 0
            or private_qp_sphere["online_controller_changed"]
            or private_qp_sphere["private_servo_commanded_by_trial"]
            or private_qp_sphere["strict_online_domain_accepted"]
            or private_qp_sphere["full_cycle_timing_measured"]):
        raise ValueError("private QP sphere trial protocol changed")
    for name, digest in private_qp_sphere["source_sha256"].items():
        if _source_sha(Path("v6_lite") / name) != digest:
            raise ValueError(f"private QP sphere trial source changed: {name}")
    if private_qp_sphere["input_metrics_sha256"] != _sha(
            Path("v6_lite/output/v6_2_a1/enabled_root/output/v6_lite_metrics.json")):
        raise ValueError("private QP sphere trial A.1 input changed")
    private_qp_dir = root / "private_qp_sphere_trial"
    private_qp_manifest = json.loads((private_qp_dir / "private_qp_sphere_manifest.json")
                                     .read_text(encoding="utf-8"))
    for label, filename in (("summary", "private_qp_sphere_summary.json"),
                            ("records", "private_qp_sphere_records.jsonl"),
                            ("failures", "private_qp_sphere_failures.jsonl"),
                            ("document", "PRIVATE_QP_SPHERE.md")):
        if private_qp_manifest[f"{label}_sha256"] != _sha(private_qp_dir / filename):
            raise ValueError(f"private QP sphere {label} changed")
    for label in ("records", "failures"):
        if private_qp_sphere[f"{label}_sha256"] != private_qp_manifest[f"{label}_sha256"]:
            raise ValueError(f"private QP sphere {label} summary changed")
    for index in range(5):
        scene_id = f"v6_lite_scenario_{index:02d}"
        folder = root / "private_rollout_400" / f"scene_{index:02d}"
        for label, filename in (("summary", "private_rollout_summary.json"),
                                ("records", "private_rollout_records.jsonl"),
                                ("trace", "private_rollout_trace.npz")):
            if private_qp_sphere["inputs"][scene_id][f"{label}_sha256"] != _sha(
                    folder / filename):
                raise ValueError(f"private QP sphere input changed: {scene_id} {label}")
    if (len(private_qp_sphere["groups"]) != 4
            or any(group["record_count"] != 30 or group["mismatch_count"] != 0
                   for group in private_qp_sphere["groups"])
            or any(private_qp_sphere["summary"][method]["record_count"] != 60
                   for method in ("reference", "sphere_screen"))):
        raise ValueError("private QP sphere group comparison changed")
    if (private_qp_full["schema"] != "v6_2_b2_private_full_qp_sphere_v1"
            or private_qp_full["scope"] != "read_only_all_saved_private_planning_states"
            or private_qp_full["max_ticks_per_scene"] != 400
            or private_qp_full["scene_count"] != 5
            or private_qp_full["point_budget"] != 255
            or private_qp_full["record_count"] != 4000
            or private_qp_full["failure_count"] != 0
            or private_qp_full["source_hash_newline_policy"] != "LF_NORMALIZED"
            or private_qp_full["production_online_controller_changed"]
            or private_qp_full["private_servo_commanded_by_trial"]
            or private_qp_full["full_cycle_timing_measured"]
            or private_qp_full["strict_online_domain_accepted"]):
        raise ValueError("private full-trace QP sphere protocol changed")
    for name, digest in private_qp_full["source_sha256"].items():
        if _source_sha(Path("v6_lite") / name) != digest:
            raise ValueError(f"private full-trace QP source changed: {name}")
    if private_qp_full["input_metrics_sha256"] != private_qp_sphere[
            "input_metrics_sha256"]:
        raise ValueError("private full-trace QP A.1 input changed")
    if (private_qp_full["method_order_by_scene"] != {
            f"v6_lite_scenario_{index:02d}":
            (["reference", "sphere_screen"] if index % 2 == 0
             else ["sphere_screen", "reference"])
            for index in range(5)}):
        raise ValueError("private full-trace method order changed")
    for index in range(5):
        scene_id = f"v6_lite_scenario_{index:02d}"
        if private_qp_full["input_private_hashes"][scene_id] != private_qp_sphere[
                "inputs"][scene_id]:
            raise ValueError(f"private full-trace QP input changed: {scene_id}")
    for method in ("reference", "sphere_screen"):
        item = private_qp_full["summary"][method]
        if (item["preflight_plus_qp_ms"]["count"] != 2000
                or item["saved_command_max_error"]["max"] > 1e-8):
            raise ValueError(f"private full-trace QP result changed: {method}")
    full_qp_dir = root / "private_qp_sphere_full"
    full_qp_manifest = json.loads((full_qp_dir /
                                   "private_full_qp_sphere_manifest.json")
                                  .read_text(encoding="utf-8"))
    for label, filename in (("summary", "private_full_qp_sphere_summary.json"),
                            ("records", "private_full_qp_sphere_records.jsonl"),
                            ("failures", "private_full_qp_sphere_failures.jsonl")):
        if full_qp_manifest[f"{label}_sha256"] != _sha(full_qp_dir / filename):
            raise ValueError(f"private full-trace QP {label} changed")
    for label in ("records", "failures"):
        if private_qp_full[f"{label}_sha256"] != full_qp_manifest[
                f"{label}_sha256"]:
            raise ValueError(f"private full-trace QP {label} summary changed")
    if (private_recompute["all_500hz_states_on_declared_shape_subspace"]
            or private_recompute[
                "independent_strict_online_domain_all_executed_ticks"]
            or private_recompute["torque_limit_violation_count"]
            or not discrete_recompute["pass_recompute"]
            or not discrete_recompute[
                "independent_strict_online_domain_all_executed_ticks"]
            or not discrete_recompute[
                "all_500hz_states_on_declared_shape_subspace"]
            or discrete_recompute["torque_limit_violation_count"]
            or discrete_recompute["failure_count"]
            or discrete_recompute["checked_task_states"] != 2005
            or discrete_recompute["checked_interval_rows"] != 5257
            or discrete_recompute["maximum_native_state_error"] > 1e-8
            or discrete_recompute["maximum_500hz_subspace_residual_linf_rad"]
            > 1e-10):
        raise ValueError("original/compensated strict-domain replay changed")
    if (servo_origin["record_count"] != 50
            or any(scene["first_step_outside_1e_10_rad"] != 1
                   or scene["torque_saturation_step_count"]
                   or scene["acceleration_clip_step_count"]
                   for scene in servo_origin["scenes"])
            or servo_origin["maximum_saved_torque_error_nm"] != 0.0
            or servo_origin["maximum_saved_qpos_error"] != 0.0
            or discrete_first_tick["record_count"] != 50
            or any(scene["maximum_position_residual_rad"] > 1e-10
                   or scene["torque_saturation_step_count"]
                   for scene in discrete_first_tick["scenes"])
            or discrete_first_tick["online_controller_changed"]
            or discrete_first_tick["stage3_admission"]
            or discrete_first_tick["continuous_time_certified"]
            or discrete_private["status"]
            != "PRIVATE_DIAGNOSTIC_COMPLETE_NOT_ONLINE_ADMISSION"
            or discrete_private["complete_400_tick_scene_count"] != 5
            or discrete_private["checked_500hz_states"] != 20005
            or discrete_private["maximum_500hz_subspace_residual_linf_rad"]
            > 1e-10
            or discrete_private["independent_interval_row_count"] != 5257
            or discrete_private["independent_recompute_failure_count"]
            or discrete_private["torque_limit_violation_count"]
            or discrete_private["production_online_controller_changed"]
            or discrete_private["stage3_admission"]
            or discrete_private["full_cycle_20ms_acceptance"]
            or discrete_private["continuous_time_certified"]):
        raise ValueError("compensated private diagnostic provenance failed")
    for source in (servo_origin, discrete_first_tick, discrete_recompute):
        for name, digest in source["source_sha256"].items():
            if _source_sha(Path("v6_lite") / name) != digest:
                raise ValueError(f"compensated source changed: {name}")
    for directory, manifest_name, names in (
        (root / "servo_subspace_origin", "servo_subspace_origin_manifest.json",
         {"summary": "servo_subspace_origin_summary.json",
          "records": "servo_subspace_origin_records.jsonl"}),
        (root / "discrete_servo_first_tick", "discrete_servo_manifest.json",
         {"summary": "discrete_servo_summary.json",
          "records": "discrete_servo_records.jsonl"}),
        (root / "discrete_private_recompute_400",
         "private_recompute_manifest.json",
         {"summary": "private_recompute_summary.json",
          "checks": "private_recompute_checks.jsonl",
          "interval_rows": "private_recompute_interval_rows.jsonl",
          "failures": "private_recompute_failures.jsonl",
          "document": "PRIVATE_RECOMPUTE.md"}),
        (root / "discrete_private_400_summary",
         "discrete_private_five_scene_manifest.json",
         {"summary": "discrete_private_five_scene_summary.json",
          "document": "DISCRETE_PRIVATE_FIVE_SCENE.md"}),
    ):
        manifest = json.loads((directory / manifest_name).read_text(
            encoding="utf-8"))
        for label, filename in names.items():
            if manifest[f"{label}_sha256"] != _sha(directory / filename):
                raise ValueError(f"compensated {directory.name} {label} changed")
    if (discrete_private["origin_summary_sha256"]
            != _sha(sources["servo_subspace_origin"])
            or discrete_private["first_tick_summary_sha256"]
            != _sha(sources["discrete_servo_first_tick"])
            or discrete_private["recompute_summary_sha256"]
            != _sha(sources["discrete_private_recompute"])):
        raise ValueError("compensated aggregate input hashes changed")
    for index, item in enumerate(discrete_private["scenes"]):
        folder = root / "discrete_private_rollout_400" / f"scene_{index:02d}"
        if (item["scenario_id"] != f"v6_lite_scenario_{index:02d}"
                or item["executed_ticks"] != 400
                or item["maximum_subspace_residual_linf_rad"] > 1e-10
                or item["minimum_ramp_envelope_margin_m"] <= 0.0
                or item["minimum_next_start_slack_m_s"] < 0.0
                or item["summary_sha256"]
                != _sha(folder / "private_rollout_summary.json")
                or item["records_sha256"]
                != _sha(folder / "private_rollout_records.jsonl")
                or item["trace_sha256"]
                != _sha(folder / "private_rollout_trace.npz")
                or item["manifest_sha256"]
                != _sha(folder / "private_rollout_manifest.json")):
            raise ValueError("compensated private scene changed")
    if (screened_scene["schema"]
            != "v6_2_b2_discrete_screened_private_multicycle_v1"
            or screened_scene["scenario_id"] != "v6_lite_scenario_00"
            or screened_scene["executed_ticks"] != 400
            or screened_scene["stop_reason"] != "HORIZON_COMPLETE"
            or not screened_scene["strict_online_domain_all_executed_ticks"]
            or screened_scene["native_replay_max_qpos_error"] > 1e-8
            or len(screened_scene["sphere_screen_exact_pair_calls"]) != 400
            or screened_scene["private_preflight_plus_qp_timing"]["p95_ms"]
            <= 20.0
            or screened_scene["stage3_admission"]
            or paired_discrete_qp["record_count"] != 4000
            or paired_discrete_qp["failure_count"]
            or paired_discrete_qp["private_servo_commanded_by_trial"]
            or paired_discrete_qp["full_cycle_timing_measured"]
            or paired_discrete_qp["summary"]["sphere_screen"][
                "preflight_plus_qp_ms"]["p95"] <= 20.0
            or screened_parity["status"]
            != "SCREEN_PARITY_PASS_TIMING_GATE_NOT_MET"
            or screened_parity["maximum_qpos_difference_m_or_rad"] > 1e-8
            or screened_parity["maximum_torque_difference_nm"] > 1e-8
            or screened_parity["paired_failure_count"]
            or screened_parity["stage3_admission"]
            or screened_parity["full_cycle_20ms_acceptance"]
            or screened_parity["continuous_time_certified"]):
        raise ValueError("compensated pair screen result changed")
    for source in (screened_scene, paired_discrete_qp, screened_parity):
        for name, digest in source["source_sha256"].items():
            if _source_sha(Path("v6_lite") / name) != digest:
                raise ValueError(f"compensated pair-screen source changed: {name}")
    for directory, manifest_name, names in (
        (root / "discrete_screened_private_rollout_400" / "scene_00",
         "private_rollout_manifest.json",
         {"summary": "private_rollout_summary.json",
          "records": "private_rollout_records.jsonl",
          "trace": "private_rollout_trace.npz",
          "document": "PRIVATE_ROLLOUT.md"}),
        (root / "discrete_private_full_qp_sphere",
         "private_full_qp_sphere_manifest.json",
         {"summary": "private_full_qp_sphere_summary.json",
          "records": "private_full_qp_sphere_records.jsonl",
          "failures": "private_full_qp_sphere_failures.jsonl"}),
        (root / "discrete_screened_parity",
         "discrete_screened_parity_manifest.json",
         {"summary": "discrete_screened_parity_summary.json",
          "document": "DISCRETE_SCREENED_PARITY.md"}),
    ):
        manifest = json.loads((directory / manifest_name).read_text(
            encoding="utf-8"))
        for label, filename in names.items():
            if manifest[f"{label}_sha256"] != _sha(directory / filename):
                raise ValueError(f"compensated {directory.name} {label} changed")
    if (screened_parity["reference_summary_sha256"]
            != discrete_private["scenes"][0]["summary_sha256"]
            or screened_parity["screened_summary_sha256"]
            != _sha(sources["discrete_screened_scene00"])
            or screened_parity["paired_qp_summary_sha256"]
            != _sha(sources["discrete_private_paired_qp"])):
        raise ValueError("compensated pair-screen input hashes changed")
    if (sweep["warm_shadow_sha256"] != _sha(sources["warm_shadow"])
            or sweep["source_hash_newline_policy"] != "LF_NORMALIZED"
            or sweep["online_control_changed"]
            or sweep["fallback_geom_names"] != ["collision_0003"]):
        raise ValueError("full task-state envelope sweep provenance failed")
    for name, digest in sweep["source_sha256"].items():
        if _source_sha(Path("v6_lite") / name) != digest:
            raise ValueError(f"full envelope sweep source changed: {name}")
    for mode in ("baseline", "enabled"):
        item = sweep["inputs"][mode]
        if _sha(Path(item["metrics_path"])) != item["metrics_sha256"]:
            raise ValueError("full envelope sweep metrics changed")
        if len(item["traces"]) != 5:
            raise ValueError("full envelope sweep lacks five scenarios")
        for trace in item["traces"]:
            if _sha(Path(trace["path"])) != trace["sha256"]:
                raise ValueError("full envelope sweep trace changed")
    sweep_dir = root / "full_state_envelope"
    if (_sha(sweep_dir / "envelope_sweep_states.jsonl")
            != sweep["state_records_sha256"]
            or _sha(sweep_dir / "envelope_sweep_failures.jsonl")
            != sweep["failure_records_sha256"]):
        raise ValueError("full envelope sweep records changed")
    if (micro["input_refined_start_sha256"] != _sha(sources["refined_start"])
            or micro["input_full_sweep_sha256"] != _sha(sources["full_state_envelope"])
            or micro["input_full_sweep_states_sha256"]
            != sweep["state_records_sha256"]
            or micro["source_hash_newline_policy"] != "LF_NORMALIZED"
            or micro["new_interval_mode_executed"]
            or micro["continuous_time_certified"]
            or len(micro["native_replay_checks"]) != 10
            or any(item["max_state_error"] > 1e-8
                   for item in micro["native_replay_checks"])):
        raise ValueError("microstep envelope native replay provenance failed")
    for name, digest in micro["source_sha256"].items():
        if _source_sha(Path("v6_lite") / name) != digest:
            raise ValueError(f"microstep envelope source changed: {name}")
    for item in micro["inputs"].values():
        if _sha(Path(item["metrics_path"])) != item["metrics_sha256"]:
            raise ValueError("microstep envelope metrics changed")
        if len(item["traces"]) != 5:
            raise ValueError("microstep envelope lacks five scenarios")
        for trace in item["traces"]:
            if _sha(Path(trace["path"])) != trace["sha256"]:
                raise ValueError("microstep envelope trace changed")
    micro_dir = root / "microstep_envelope"
    if (_sha(micro_dir / "microstep_envelope_states.jsonl")
            != micro["state_records_sha256"]
            or _sha(micro_dir / "microstep_envelope_failures.jsonl")
            != micro["failure_records_sha256"]):
        raise ValueError("microstep envelope records changed")
    if (full_torque["input_full_sweep_summary_sha256"]
            != _sha(sources["full_state_envelope"])
            or full_torque["input_full_sweep_states_sha256"]
            != sweep["state_records_sha256"]
            or full_torque["input_selected_summary_sha256"]
            != _sha(sources["microstep_envelope"])
            or full_torque["input_selected_states_sha256"]
            != micro["state_records_sha256"]
            or full_torque["source_hash_newline_policy"] != "LF_NORMALIZED"
            or full_torque["new_interval_mode_executed"]
            or full_torque["continuous_time_certified"]
            or full_torque["cross_checks"] != {
                "saved_task_states": 13500, "selected_microstates": 220,
            }
            or len(full_torque["scenes"]) != 10):
        raise ValueError("full old-torque envelope provenance failed")
    for name, digest in full_torque["source_sha256"].items():
        if _source_sha(Path("v6_lite") / name) != digest:
            raise ValueError(f"full old-torque source changed: {name}")
    for item in full_torque["inputs"].values():
        if (_sha(Path(item["metrics_path"])) != item["metrics_sha256"]
                or len(item["traces"]) != 5):
            raise ValueError("full old-torque metrics changed")
        for trace in item["traces"]:
            if _sha(Path(trace["path"])) != trace["sha256"]:
                raise ValueError("full old-torque trace changed")
    full_torque_dir = root / "full_torque_envelope"
    full_torque_outputs = {
        "summary": "full_torque_envelope_summary.json",
        "states": "full_torque_envelope_states.jsonl",
        "failures": "full_torque_envelope_failures.jsonl",
        "document": "FULL_TORQUE_ENVELOPE.md",
    }
    full_torque_manifest = json.loads((full_torque_dir / "full_torque_envelope_manifest.json")
                                      .read_text(encoding="utf-8"))
    for name, filename in full_torque_outputs.items():
        if full_torque_manifest[f"{name}_sha256"] != _sha(full_torque_dir / filename):
            raise ValueError(f"full old-torque {name} hash changed")
    trials = (
        (batched, root / "batched_query_trial", {
            "summary": "batched_query_summary.json",
            "document": "BATCHED_QUERY_TRIAL.md",
        }, "batched_query_manifest.json", "input_refined_start_sha256",
         sources["refined_start"]),
        (prepared_cold, root / "prepared_query_trial", {
            "summary": "prepared_query_summary.json",
            "document": "PREPARED_QUERY_TRIAL.md",
        }, "prepared_query_manifest.json", "input_warm_shadow_sha256",
         sources["warm_shadow"]),
        (prepared_full, root / "prepared_full_trace", {
            "summary": "prepared_full_trace_summary.json",
            "states": "prepared_full_trace_states.jsonl",
            "failures": "prepared_full_trace_failures.jsonl",
            "document": "PREPARED_FULL_TRACE.md",
        }, "prepared_full_trace_manifest.json", "input_warm_shadow_sha256",
         sources["warm_shadow"]),
    )
    for trial, directory, outputs, manifest_name, input_key, input_path in trials:
        if (trial[input_key] != _sha(input_path)
                or trial["source_hash_newline_policy"] != "LF_NORMALIZED"
                or trial["online_control_changed"]
                or trial["new_interval_mode_executed"]):
            raise ValueError("read-only performance trial provenance failed")
        for name, digest in trial["source_sha256"].items():
            if _source_sha(Path("v6_lite") / name) != digest:
                raise ValueError(f"performance trial source changed: {name}")
        for item in trial["inputs"].values():
            if _sha(Path(item["metrics_path"])) != item["metrics_sha256"]:
                raise ValueError("performance trial metrics changed")
            if len(item["traces"]) != 5:
                raise ValueError("performance trial lacks five scenarios")
            for trace in item["traces"]:
                if _sha(Path(trace["path"])) != trace["sha256"]:
                    raise ValueError("performance trial trace changed")
        trial_manifest = json.loads((directory / manifest_name).read_text(
            encoding="utf-8"))
        for name, filename in outputs.items():
            if trial_manifest[f"{name}_sha256"] != _sha(directory / filename):
                raise ValueError(f"performance trial {name} hash changed")
    if (len(batched["records"]) != 60
            or len(prepared_cold["records"]) != 136
            or prepared_full["state_records_sha256"] != _sha(
                root / "prepared_full_trace" / "prepared_full_trace_states.jsonl")
            or prepared_full["failure_records_sha256"] != _sha(
                root / "prepared_full_trace" / "prepared_full_trace_failures.jsonl")
            or (root / "prepared_full_trace" /
                "prepared_full_trace_failures.jsonl").stat().st_size != 0):
        raise ValueError("performance trial records or parity changed")
    for mode in ("baseline", "enabled"):
        item = prepared_full["modes"][mode]
        if (item["counts"]["task_states"] != 6750
                or item["counts"].get("parity_failures", 0)
                or item["counts"]["UNKNOWN_CROSSES_GATE"]
                != warm["modes"][mode]["counts"]["warm_all_task_unknown"]
                or item["maximum_parity_error_m"] > 1e-12):
            raise ValueError("full saved-state prepared query parity failed")
    prediction_margins = {}
    for mode in ("baseline", "enabled"):
        metrics_path = Path(probe["inputs"][mode]["metrics_path"])
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        run_config = metrics["run_config"]
        if (run_config["task_period_s"] != .02
                or run_config["physics_period_s"] != .002
                or run_config["task_latency_p95_threshold_s"] != .02
                or metrics["qp_config"]["task_period_s"] != .02):
            raise ValueError("A.1 task cadence or latency threshold changed")
        prediction_margins[mode] = metrics["qp_config"]["lookahead_model_margin_m_s"]
        if prediction_margins[mode] != .005:
            raise ValueError("original lookahead margin changed")
        summaries = warm["modes"][mode]["summaries"]
        h = summaries["next_h_prediction_abs_error_m"]
        residual = summaries["next_start_residual_abs_error_m_s"]
        if (not 0 < residual["count"] == h["count"]
                or any(not (0 <= series["p50"] <= series["p95"]
                            <= series["p99"] <= series["max"])
                       for series in (h, residual))):
            raise ValueError("fixed-partition prediction summary invalid")
    rescue_dir = root / "root_rescue_frontier"
    if (root_rescue["input_repartition_counterfactual_sha256"]
            != _sha(sources["repartition_counterfactual"])
            or root_rescue["source_hash_newline_policy"] != "LF_NORMALIZED"
            or root_rescue["online_control_changed"]
            or root_rescue["new_interval_mode_executed"]
            or root_rescue["budgets"] != [127, 255]
            or root_rescue["max_leaves"] != 256):
        raise ValueError("root rescue frontier provenance or scope failed")
    for name, digest in root_rescue["source_sha256"].items():
        if _source_sha(Path("v6_lite") / name) != digest:
            raise ValueError(f"root rescue source changed: {name}")
    for mode in ("baseline", "enabled"):
        item = root_rescue["inputs"][mode]
        if (_sha(Path(item["metrics_path"])) != item["metrics_sha256"]
                or len(item["traces"]) != 5):
            raise ValueError("root rescue five-scenario input changed")
        for trace in item["traces"]:
            if _sha(Path(trace["path"])) != trace["sha256"]:
                raise ValueError("root rescue trace changed")
        result = root_rescue["modes"][mode]
        counts = result["counts"]
        frozen = repartition["modes"][mode]["counts"]["cold_UNKNOWN_CROSSES_GATE"]
        if (counts["frozen_states"] != frozen
                or result["maximum_parity_error_m"] > 1e-12):
            raise ValueError("root rescue parity or frozen population failed")
        for budget in root_rescue["budgets"]:
            prefix = str(budget)
            classified = sum(counts.get(f"{prefix}_{status}", 0) for status in (
                "PROXY_CLEARANCE_AT_LEAST_GATE", "PROXY_CLEARANCE_BELOW_GATE",
                "UNKNOWN_CROSSES_GATE"))
            if (classified != frozen
                    or result["budgets"][prefix]["prepared_query_ms"]["count"]
                    != frozen):
                raise ValueError("root rescue status totals changed")
    rescue_outputs = {
        "summary": "root_rescue_summary.json",
        "states": "root_rescue_states.jsonl",
        "failures": "root_rescue_failures.jsonl",
        "document": "ROOT_RESCUE_FRONTIER.md",
    }
    rescue_manifest = json.loads((rescue_dir / "root_rescue_manifest.json").read_text(
        encoding="utf-8"))
    for name, filename in rescue_outputs.items():
        if rescue_manifest[f"{name}_sha256"] != _sha(rescue_dir / filename):
            raise ValueError(f"root rescue {name} hash changed")
    if (rescue_dir / rescue_outputs["failures"]).stat().st_size != 0:
        raise ValueError("root rescue parity failures present")
    full_root_dir = root / "full_root255_census"
    if (full_root["input_warm_shadow_sha256"] != _sha(sources["warm_shadow"])
            or full_root["input_root_rescue_summary_sha256"]
            != _sha(sources["root_rescue_frontier"])
            or full_root["input_root_rescue_states_sha256"]
            != _sha(rescue_dir / "root_rescue_states.jsonl")
            or full_root["source_hash_newline_policy"] != "LF_NORMALIZED"
            or full_root["point_budget"] != 255
            or full_root["max_leaves"] != 256
            or full_root["online_control_changed"]
            or full_root["new_interval_mode_executed"]):
        raise ValueError("full root-255 census provenance or scope failed")
    for name, digest in full_root["source_sha256"].items():
        if _source_sha(Path("v6_lite") / name) != digest:
            raise ValueError(f"full root-255 source changed: {name}")
    for mode in ("baseline", "enabled"):
        item = full_root["inputs"][mode]
        if (_sha(Path(item["metrics_path"])) != item["metrics_sha256"]
                or len(item["traces"]) != 5):
            raise ValueError("full root-255 five-scenario input changed")
        for trace in item["traces"]:
            if _sha(Path(trace["path"])) != trace["sha256"]:
                raise ValueError("full root-255 trace changed")
        result = full_root["modes"][mode]
        counts = result["counts"]
        classified = sum(counts.get(status, 0) for status in (
            "PROXY_CLEARANCE_AT_LEAST_GATE", "PROXY_CLEARANCE_BELOW_GATE",
            "UNKNOWN_CROSSES_GATE"))
        if (counts["task_states"] != 6750
                or classified != counts["task_states"]
                or counts["root_rescue_matches"]
                != root_rescue["modes"][mode]["counts"]["frozen_states"]
                or counts.get("parity_failures", 0)
                or result["maximum_parity_error_m"] > 1e-12):
            raise ValueError("full root-255 census parity or population failed")
    full_outputs = {
        "summary": "full_root255_summary.json",
        "states": "full_root255_states.jsonl",
        "failures": "full_root255_failures.jsonl",
        "document": "FULL_ROOT255_CENSUS.md",
    }
    full_manifest = json.loads((full_root_dir / "full_root255_manifest.json").read_text(
        encoding="utf-8"))
    for name, filename in full_outputs.items():
        if full_manifest[f"{name}_sha256"] != _sha(full_root_dir / filename):
            raise ValueError(f"full root-255 {name} hash changed")
    if (full_root_dir / full_outputs["failures"]).stat().st_size != 0:
        raise ValueError("full root-255 parity failures present")
    gate = warm["online_admission_gate"]
    blockers = _gate_blockers(frontier, refined)
    remaining = sum(refined["modes"][mode]["old_bad_remains_unexecutable"]
                    for mode in ("baseline", "enabled"))
    probe_ready = all(
        probe["summary"][mode]["probe_count"] == 15
        and probe["summary"][mode]["admission_preconditions_met_count"] == 15
        and probe["summary"][mode]["validated_command_count"] == 15
        for mode in ("baseline", "enabled")
    )
    sweep_ready = all(
        sweep["modes"][mode]["state_count"] == 6750
        and sweep["modes"][mode]["status_counts"].get(
            "COVERED_AT_THIS_STATE", 0) == 6750
        for mode in ("baseline", "enabled")
    )
    micro_ready = all(
        micro["modes"][mode]["window_count"] == 10
        and micro["modes"][mode]["microstate_count"] == 110
        and micro["modes"][mode]["covered_microstate_count"] == 110
        for mode in ("baseline", "enabled")
    )
    candidate_ramp_ready = all(
        candidate_ramp["modes"][mode]["ramp_count"] == 15
        and candidate_ramp["modes"][mode]["covered_ramp_count"] == 15
        and candidate_ramp["modes"][mode]["realized_next_start_satisfied_count"] == 15
        and candidate_ramp["modes"][mode]["next_proxy_safe_count"] == 15
        and candidate_ramp["modes"][mode]["whole_body_violation_ramp_count"] == 0
        and candidate_ramp["modes"][mode]["maximum_old_torque_error"] <= 1e-8
        and candidate_ramp["modes"][mode]["maximum_old_next_qpos_error"] <= 1e-8
        for mode in ("baseline", "enabled")
    )
    candidate_prediction_ready = all(
        candidate_prediction["modes"][mode]["case_count"] == 15
        and candidate_prediction["modes"][mode]["selected_row_count"] == 50
        and candidate_prediction["modes"][mode]["cbf_optimism_over_frozen_margin_count"] == 0
        and candidate_prediction["modes"][mode]["realized_start_violation_count"] == 0
        for mode in ("baseline", "enabled")
    )
    repeated_partial_budget_ready = all(
        repeated_qp["modes"][mode]["record_count"] == 75
        and repeated_qp["modes"][mode]["query_plus_qp_probe_ms"]["p95"] <= 20.0
        for mode in ("baseline", "enabled")
    )
    full_torque_ready = all(
        full_torque["modes"][mode]["scene_count"] == 5
        and full_torque["modes"][mode]["torque_steps"] == 67500
        and full_torque["modes"][mode]["checked_states"] == 67505
        and full_torque["modes"][mode]["covered_states"] == 67505
        and full_torque["modes"][mode]["maximum_native_replay_error"] <= 1e-8
        for mode in ("baseline", "enabled")
    )
    status = ("GATE_NOT_MET" if gate["status"] == "NOT_MET" or remaining
              or full_private_gate["strict_domain_violation_tick_count"]
              or not probe_ready or not sweep_ready or not micro_ready
              or not candidate_ramp_ready or not candidate_prediction_ready
              or not repeated_partial_budget_ready
              or not full_torque_ready
              else "NEEDS_TRUE_ONLINE_TIMING")
    a1_enabled = json.loads(Path(full_root["inputs"]["enabled"]["metrics_path"])
                            .read_text(encoding="utf-8"))
    a1_task_p95_max = max(
        scene["metrics"]["rates_and_latency"]["task_full_latency_p95_ms"]
        for scene in a1_enabled["scenarios"])
    a1_task_single_max = max(
        scene["metrics"]["rates_and_latency"]["task_full_latency_max_ms"]
        for scene in a1_enabled["scenarios"])
    query_diagnostic_ceiling_ms = cold["modes"]["enabled"]["budget"][
        "total_query_time_ms"]
    lines = [
        "# V6.2-B.2 第二阶段：影子评估与在线接入门禁", "",
        f"当前门禁：**{status}**。因此本证据不作为新区间模式的五场景闭环验收。",
        "", "## 20 ms 门槛的来源", "",
        "旧 A.1 两组已发布配置均固定 50 Hz 规划周期 (`task_period_s=0.02`)，"
        "500 Hz MuJoCo/力矩周期 (`physics_period_s=0.002`)，"
        "即每个规划命令对应共享十步斜坡。原任务控制器 `QP.solve` 的"
        " `full_latency` 从 `mj_forward` 前开始，到该次候选动作验证及"
        "求解器状态更新后结束；它覆盖任务层几何、约束装配、QP 和动作"
        "验证，不含之后十步力矩伺服、`mj_step` 或真实系统调度。"
        "任务层 p95 验收阈值设为 0.02 s，力矩计算另有 0.002 s p95 阈值；"
        "B.2 沿用 20 ms 作为周期预算，但新预检＋QP 诊断覆盖额外的区间"
        "查询和包络核对，计时范围与旧指标不同，不能直接比较速度提升。"
        "没有从 CBF 数学性质推得这个数值。只读查询或查询加 QP 探针"
        "不是完整任务层计时；即使该 p95 达标，也不构成每周期无超时的"
        "硬实时证明。",
        f"旧 A.1 enabled 五场景中最差的任务层 p95 为 {a1_task_p95_max:.3f} ms，"
        f"单次最大值为 {a1_task_single_max:.3f} ms；这也说明 p95 门槛允许个别周期超时。",
        f"当前影子配置的 `total_query_time_ms={query_diagnostic_ceiling_ms:.1f}` "
        "限制影子几何预检的诊断计时，包含区间查询、相关 Jacobian 和"
        "几何核对；它占满整个任务周期，"
        "没有给旧 MuJoCo 约束装配、QP、动作验证和调度预留时间，"
        "不能单独作为在线准入预算。",
        f"新区间私有 scene_03 的 {full_scene_timing['count']} 次预检＋QP "
        f"p95 为 {full_scene_timing['p95_ms']:.3f} ms，"
        f"最大值为 {full_scene_timing['max_ms']:.3f} ms；"
        "这个局部指标没有完成生产在线模式的完整任务层验收。",
        "", "## A.1 原生力矩重放上的只读几何", "",
        "两组旧控制 trace 均按 500 Hz 力矩原生重放；每个规划边界先调用 `mj_forward`"
        " 更新空间几何量，再与保存的 50 Hz 状态及哈希核对。"
        "每组五场景，每场景预定抽取 27 个规划状态；持久查询则在每组 "
        f"{warm['modes']['baseline']['summaries']['warm_all_task_query_ms']['count']:,} "
        "个规划 tick 上更新。", "",
        "| 模式 | 冷查询 p95 ms | 持久查询采样 p95 ms | 持久查询全部 tick 未知 | 最大叶数 | 配对容量估计 p95 ms | 真实几何误拒绝 | 子空间外状态 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for mode in ("baseline", "enabled"):
        c, w = cold["modes"][mode], warm["modes"][mode]
        lines.append(
            f"| {mode} | {c['summaries']['query_time_ms']['p95']:.3f} | "
            f"{w['summaries']['query_time_ms']['p95']:.3f} | "
            f"{w['counts'].get('warm_all_task_unknown', 0)} | "
            f"{w['summaries']['warm_all_task_partition_size']['max']:.0f} | "
            f"{w['summaries']['paired_replacement_estimate_ms']['p95']:.3f} | "
            f"{w['counts'].get('proxy_false_reject_vs_mujoco', 0)} | "
            f"{w['counts'].get('off_shape_subspace', 0)} |"
        )
    lines += [
        "", "持久区间未合并时可增长到预算边界，许多贴近 5 mm 门槛的状态保持未知。"
        "排除区间在所测下一 tick 进入激活距离的计数为 0，但此有限重放结果不是一般动态保证。",
        "配对容量估计使用旧全链耗时减旧形状查询耗时再加影子查询耗时；"
        "它不是新模式实测，也不能单独证明 20 ms 全链 p95。", "",
        "区间导数对每个状态的基座和目标刚体原点各调用一次 MuJoCo Jacobian，"
        "再将刚体速度精确平移至各中点及目标 witness；"
        "下面的计数包含形状 Jacobian 与这两次 MuJoCo 调用。", "",
        "| 模式 | 抽样 Jacobian 次数 p95 | 统一预算超限 | 持久热查询 p95 ms |",
        "| --- | ---: | ---: | ---: |",
    ]
    for mode in ("baseline", "enabled"):
        item = warm["modes"][mode]
        lines.append(
            f"| {mode} | {item['summaries']['jacobian_evaluations']['p95']:.0f} | "
            f"{item['counts'].get('unified_budget_exceeded', 0)} | "
            f"{item['summaries']['query_time_ms']['p95']:.3f} |"
        )
    lines += [
        "", "调用次数下降不等于墙钟时延同比例下降；这里仍无新区间模式的实测全链时延。", "",
        "## 固定区间的下一周期预测误差", "",
        "在旧 A.1 力矩原生重放中，用当前冻结区间和共享十步斜坡预测下一规划边界的"
        "区间安全函数及起点残差，再于下一边界重算同一固定区间。"
        "下表只统计具备配对下一状态和所需导数的行。", "",
        "| 模式 | 配对区间行 | 安全函数绝对误差 p95 / 最大 (µm) | 起点残差绝对误差 p95 / 最大 (mm/s) | 原前瞻裕度 (mm/s) |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for mode in ("baseline", "enabled"):
        summaries = warm["modes"][mode]["summaries"]
        h = summaries["next_h_prediction_abs_error_m"]
        residual = summaries["next_start_residual_abs_error_m_s"]
        lines.append(
            f"| {mode} | {residual['count']} | "
            f"{1e6 * h['p95']:.3f} / {1e6 * h['max']:.3f} | "
            f"{1e3 * residual['p95']:.3f} / {1e3 * residual['max']:.3f} | "
            f"{1e3 * prediction_margins[mode]:.3f} |"
        )
    lines += [
        "", "所测两组最大绝对残差均低于原 5 mm/s 前瞻裕度，但这只是旧轨迹、"
        "抽样状态和固定区间上的经验观察。绝对误差不保留偏差方向，"
        "本审计也没有分离区间切换与新命令执行误差；不能把该裕度称为严格误差上界，"
        "更不能用它放行当前起点违反。", "",
        "持久未知状态另以保存的 A.1 规划位置做同状态对照：保持 64 点预算不变，"
        "只从五段根区间重新查询，未重放力矩，也未将新分区用于动作。", "",
        "| 模式 | 持久未知 | 从根区间后确定安全 | 确定低于门槛 | 仍未知 | 额外冷查询 p95 ms |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for mode in ("baseline", "enabled"):
        item = repartition["modes"][mode]
        c = item["counts"]
        lines.append(
            f"| {mode} | {c.get('warm_UNKNOWN_CROSSES_GATE', 0)} | "
            f"{c.get('cold_PROXY_CLEARANCE_AT_LEAST_GATE', 0)} | "
            f"{c.get('cold_PROXY_CLEARANCE_BELOW_GATE', 0)} | "
            f"{c.get('cold_UNKNOWN_CROSSES_GATE', 0)} | "
            f"{item['counterfactual_cold_query_time_ms'].get('p95', float('nan')):.3f} |"
        )
    lines += [
        "", "重分区能解释部分持久未知，但会改变区间安全函数集合。"
        "这项额外耗时不能与热查询并列当作新模式全链计时；"
        "分区切换后的斜坡起点、必要约束和预算仍须单独验证。", "",
        "对持久未知的抽样 tick，又以 500 Hz 旧力矩原生重放，从根分区重建区间行，"
        "加入独立 MuJoCo／胶囊行，按原速度盒与十步斜坡做冻结 LP。", "",
        "| 模式 | 抽样未知 | 根分区确定安全 | 安全且旧起点可执行 | 安全但旧起点不可执行 | Jacobian 超预算 | 下一 tick 漏选激活 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for mode in ("baseline", "enabled"):
        item = handoff["modes"][mode]
        lines.append(
            f"| {mode} | {item['sample_count']} | "
            f"{item['cold_status'].get('PROXY_CLEARANCE_AT_LEAST_GATE', 0)} | "
            f"{item['safe_repartition_executable_candidate_count']} | "
            f"{item['safe_repartition_not_executable_count']} | "
            f"{item['jacobian_budget_exceeded']} | "
            f"{item['excluded_reached_activation_next_tick']} |"
        )
    lines += [
        "", "根分区代理安全只完成几何判定；旧斜坡起点仍可能不合格，"
        "且此计算时间不含独立原约束装配、LP 或新控制器全链。"
        "这个有限旧轨迹审计不能批准在线重分区或证明新控制轨迹可行。", "",
        "区间激活筛选从现有速度、加速度和关节限位得到候选盒，"
        "并将十步斜坡起点速度纳入逐轴绝对上界；"
        "空盒退回全局速度上界并单独使门禁失败。"
        "以下计数仅比较两种冻结模型筛选，均未修改安全距离或真实执行动作。", "",
        "| 查询 | 模式 | 比全局速度筛选少选区间 | 多选区间 | 空候选盒 | 排除区间下一 tick 激活 |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for query_name, report in (("cold", cold), ("persistent", warm)):
        for mode in ("baseline", "enabled"):
            counts = report["modes"][mode]["counts"]
            lines.append(
                f"| {query_name} | {mode} | "
                f"{counts.get('ramp_box_newly_excluded', 0)} | "
                f"{counts.get('ramp_box_newly_selected', 0)} | "
                f"{counts.get('empty_candidate_velocity_box', 0)} | "
                f"{counts.get('excluded_reached_activation_next_tick', 0)} |"
            )
    lines += [
        "", "速度盒只约束声明的斜坡起点和候选终点；反作用映射、目标漂移"
        "和几何灵敏度在筛选中仍冻结。零漏选是有限重放观察，不是跨周期证明。", "",
        "## 同一冻结行的动作可行性", "",
        "在抽样 A.1 状态上，独立重算 MuJoCo 和实际链胶囊行后加入必要的区间行；"
        "只读 LP 使用原 17 维速度盒、十步斜坡起点、原容差和冻结前瞻约束。"
        "LP 可行仅表示存在终点，不表示历史斜坡起点或任何实际命令通过。", "",
        "| 模式 | 冷查询起点违反 | 持久查询起点违反 | 持久查询无可行终点 | 持久查询历史动作被拒 |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for mode in ("baseline", "enabled"):
        c = cold["modes"][mode]["counts"]
        w = warm["modes"][mode]["counts"]
        lines.append(
            f"| {mode} | {c.get('frozen_START_CLEARANCE_VIOLATION', 0) + c.get('frozen_START_VELOCITY_VIOLATION', 0)} | "
            f"{w.get('frozen_START_CLEARANCE_VIOLATION', 0) + w.get('frozen_START_VELOCITY_VIOLATION', 0)} | "
            f"{w.get('frozen_NO_FEASIBLE_ENDPOINT', 0)} | "
            f"{w.get('historical_endpoint_rejected_by_new_rows', 0)} |"
        )
    failures = [
        record for report in (cold, warm) for mode in ("baseline", "enabled")
        for record in report["modes"][mode]["frozen_feasibility_records"]
        if record["feasibility"]["status"].startswith("START_")
    ]
    if failures and all(
        record["feasibility"].get("worst_start_source", "").startswith("pcc_interval:")
        for record in failures
    ):
        lines.append(
            "四组所列起点反例的最差行均来自 PCC 区间；该代理残差不能解释为实际链碰撞。"
        )
    lines += [
        "", "逐状态失败、区间 ID、起点残差和 LP 结果保留在两份影子 JSON 中。"
        "这些是旧轨迹的反例，不可引用为新控制器已经失败或已完成闭环。", "",
        "## 预算阶梯与细分后起点", "",
        "固定预算阶梯 31/63/127/255/511 从完整根分区重新查询保存状态。"
        "随后在原生力矩重放中，以 255 点诊断预算重建新区间行与斜坡起点。"
        "高预算只用于辨别原因，不是在线接入配置。", "",
        "| 模式 | 511 点确定代理低于 5 mm | 511 点仍未知 | 旧起点违例经细分消除 | 细分后旧违例仍不可执行 | 细分计算超过 20 ms |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for mode in ("baseline", "enabled"):
        front = frontier["modes"][mode]
        ref = refined["modes"][mode]
        lines.append(
            f"| {mode} | {front['status_by_budget']['511'].get('PROXY_CLEARANCE_BELOW_GATE', 0)} | "
            f"{front['status_by_budget']['511'].get('UNKNOWN_CROSSES_GATE', 0)} | "
            f"{ref['old_bad_cured_by_diagnostic_refinement']} | "
            f"{ref['old_bad_remains_unexecutable']} | "
            f"{ref['interval_20ms_exceeded']} |"
        )
    lines += [
        "", "代理低于门槛不是实际链碰撞；细分后区间函数与梯度必须重新计算，"
        "且分区切换还可能产生新的起点违例。逐状态预算、失败原因及运行耗时均保留在"
        "预算阶梯和细分起点 JSON 中。", "",
        "最差 PCC 起点行进一步拆为静态区间 h、形变速度、基座反作用、目标漂移"
        "和屏障项，各项和须重组为保存的起点残差：", "",
        "| 模式 | 细分后起点违反且最差区间 h >= 0 | 细分后起点违反且最差区间 h < 0 |",
        "| --- | ---: | ---: |",
    ]
    for mode in ("baseline", "enabled"):
        ref = refined["modes"][mode]
        lines.append(
            f"| {mode} | {ref['start_violations_with_positive_interval_h']} | "
            f"{ref['start_violations_with_negative_interval_h']} |"
        )
    enabled_bad_terms = [
        item["worst_interval_start_terms"] for item in refined["records"]
        if item["mode"] == "enabled"
        and item["feasibility"]["status"] == "START_CLEARANCE_VIOLATION"
        and item["worst_interval_start_terms"] is not None
    ]
    if enabled_bad_terms:
        def mean_rate_mm_s(name: str) -> float:
            return (1000.0 * sum(item[name] for item in enabled_bad_terms)
                    / len(enabled_bad_terms))

        lines.append(
            f"启用组这 {len(enabled_bad_terms)} 条最差行的平均形变速率 "
            f"{mean_rate_mm_s('shape_rate_m_s'):.2f} mm/s、基座反作用 "
            f"{mean_rate_mm_s('base_reaction_rate_m_s'):.2f} mm/s、目标漂移 "
            f"{mean_rate_mm_s('target_drift_m_s'):.2f} mm/s、屏障项 "
            f"{mean_rate_mm_s('barrier_rate_m_s'):.2f} mm/s。"
        )
    lines += [
        "", "启用组静态下界仍达到 5 mm 门槛的起点也可能因旧动作的相对逼近速率"
        "违反新区间 CBF；进一步提高静态查询预算本身不能使这些旧斜坡起点合格。"
        "这不证明新区间模式一定无法找到不同轨迹，也不批准在当前门禁下接入。", "",
        "## 门禁障碍的首次出现", "",
        "511 点诊断中的代理低于门槛状态都有 PCC 曲线中点 witness：该点的"
        " PCC 净空上界小于 5 mm。继续细分不能把同一旧状态变成代理安全；"
        "这仍不等于实际离散链碰撞。下表的 tick 是 50 Hz 规划 tick，"
        "来自每 50 tick 抽样，"
        "只是所测首次出现位置。", "",
        "| 模式 | 五场景初始可行 | 代理低于门槛的 witness 状态 | 细分后旧起点违反 | 其中最差行 h >= 0 | 各场景首次 witness tick | 各场景首次起点违反 tick |",
        "| --- | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for mode in ("baseline", "enabled"):
        item = blockers[mode]

        def ticks(name: str) -> str:
            values = item[name]
            return ", ".join(f"{scenario[-2:]}:{tick}"
                             for scenario, tick in values.items()) or "无"

        lines.append(
            f"| {mode} | "
            f"{item['initial_proxy_safe_and_frozen_rows_feasible']}/5 | "
            f"{item['sampled_proxy_below_gate_with_point_witness']} | "
            f"{item['sampled_old_start_clearance_violations_after_refinement']} | "
            f"{item['old_start_violations_with_nonnegative_interval_h']} | "
            f"{ticks('first_witnessed_below_tick_by_scenario')} | "
            f"{ticks('first_old_start_violation_tick_by_scenario')} |"
        )
    lines += [
        "", "开启组的静态 witness 未发现低于门槛，但旧起点仍可因相对逼近过快"
        "而失效。要验证不同的控制轨迹，必须先解决在线接入门禁；"
        "旧 trace 的继续重放不能代替新区间闭环。", "",
        "## 原加权 QP 的早期只读探针", "",
        "在旧力矩 trace 的每场景 tick 50/100/150，用 63 点根分区构造区间行，"
        "与原 MuJoCo 和胶囊行放进同一个 17 维加权 QP。逐行结果与独立重算对照；"
        "候选仅做原动作验证，不发送给执行器。每个探针清空对偶热启动，"
        "因此下面的耗时既不是连续新模式全链，也不能代替其 20 ms 验收。", "",
        "| 模式 | 冻结状态 | 查询预算合格 | 静态代理安全 | 包络证据支持 | QP 验证命令 | 接入前提合格 | 查询加 QP 探针 p95 ms |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for mode in ("baseline", "enabled"):
        item = probe["summary"][mode]
        lines.append(
            f"| {mode} | {item['probe_count']} | {item['query_budget_ok_count']} | "
            f"{item['proxy_safe_count']} | {item['envelope_supported_count']} | "
            f"{item['validated_command_count']} | "
            f"{item['admission_preconditions_met_count']} | "
            f"{item['query_plus_qp_probe_ms']['p95']:.3f} |"
        )
    lines += [
        "", "早期探针证明原求解及动作验证接口能够处理这些冻结新区间行；"
        "但现有严格形状子空间判据未支持真实链包络，不能把 QP 命令验证"
        "改写成新区间安全接入。"
        f"子空间阈值 {probe['shape_subspace_membership_tolerance_rad']:.1e} rad；"
        f"两组残差 p95 分别为 "
        f"{probe['summary']['baseline']['subspace_residual_linf_rad']['p95']:.2e}、"
        f"{probe['summary']['enabled']['subspace_residual_linf_rad']['p95']:.2e} rad。"
        "实际与投影 URDF 离散骨架的解析位移上界 p95 分别为 "
        f"{probe['summary']['baseline']['discrete_backbone_residual_upper_m']['p95']:.2e}、"
        f"{probe['summary']['enabled']['discrete_backbone_residual_upper_m']['p95']:.2e} m；"
        "该只读诊断没有建立 PCC 到投影离散链的包络，也不覆盖物理碰撞几何。"
        "另以实际 MuJoCo 状态逐个检查胶囊是否包含于现有 PCC 管："
        f"baseline {probe['summary']['baseline']['state_local_capsule_covered_count']}/15、"
        f"enabled {probe['summary']['enabled']['state_local_capsule_covered_count']}/15，"
        "最小半径余量分别为 "
        f"{probe['summary']['baseline']['state_local_capsule_envelope_min_margin_m']['min']:.2e}、"
        f"{probe['summary']['enabled']['state_local_capsule_envelope_min_margin_m']['min']:.2e} m。"
        "该证据只覆盖冻结时刻的实际几何；未证明命令执行期间的包络保持。"
        "检查耗时单列，未计入上表 QP 探针：两组 p95 分别为 "
        f"{probe['summary']['baseline']['state_local_capsule_envelope_check_ms']['p95']:.3f}、"
        f"{probe['summary']['enabled']['state_local_capsule_envelope_check_ms']['p95']:.3f} ms。"
        "所测耗时尾部和后续旧轨迹反例也仍存在。", "",
        "## 早期只读 QP 的五轮重复计时", "",
        "在相同两组五场景 tick 50/100/150 冻结状态预先固定五轮，每轮重新原生重放"
        "旧力矩状态、建立模型并清空 QP 对偶热启动。150 条候选命令及区间选择"
        "与首次探针逐条一致；逐轮原始记录和哈希保留。", "",
        "| 模式 | 记录 | 查询加 QP p95 / p99 / 最大 ms | 超 20 ms | 再加当前状态包络 p95 / 最大 ms |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for mode in ("baseline", "enabled"):
        item = repeated_qp["modes"][mode]
        partial = item["query_plus_qp_probe_ms"]
        envelope = item["probe_plus_envelope_ms"]
        lines.append(
            f"| {mode} | {item['record_count']} | "
            f"{partial['p95']:.3f} / {partial['p99']:.3f} / {partial['max']:.3f} | "
            f"{partial['over_20ms_count']} | "
            f"{envelope['p95']:.3f} / {envelope['max']:.3f} |"
        )
    lines += [
        "", "两组的查询加 QP p95 均已超过既有 20 ms 周期。该探针只测部分链路；"
        "没有计入全部状态采样、监控及 500 Hz 力矩计算和执行，"
        "包络列也只是同冻结状态计时相加。它不能替代新区间模式真实全链计时，"
        "超 20 ms 的只读探针记录也不能称为在线漏期。", "",
        "## 线性代数线程数的 ABBA 只读对照", "",
        "在相同 30 个冻结状态上，按请求 OpenBLAS 线程数 2→1→1→2 "
        "分别启动独立进程；所有候选与首次探针核对，保留每组原始报告。", "",
        "| 模式 | 请求线程数 | 记录 | 查询加 QP p95 / p99 / 最大 ms | 超 20 ms | QP 求解 p95 ms |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for mode in ("baseline", "enabled"):
        for threads in (2, 1):
            item = blas_trial["modes"][mode][str(threads)]
            partial = item["query_plus_qp_probe_ms"]
            lines.append(
                f"| {mode} | {threads} | {item['record_count']} | "
                f"{partial['p95']:.3f} / {partial['p99']:.3f} / "
                f"{partial['max']:.3f} | {partial['over_20ms_count']} | "
                f"{item['qp_solver_ms']['p95']:.3f} |"
            )
    lines += [
        "", "只改变子进程的 `OPENBLAS_NUM_THREADS` 请求值；未固定操作系统调度，"
        "也未核实每个库的实际线程数。这是本机性能敏感性诊断，"
        "不能与旧影子的容量估计拼接成全链 20 ms 验收。"
        "同一次 ABBA 对照中 1 与 2 的 p95 差异很小，且两者均未复现前一次"
        "五轮重复探针的耗时尾部；因此不能把跨次差异归因于线程数，"
        "也不能丢弃先前超 20 ms 的原始记录。"
        "新区间接入门禁仍需独立满足几何、动作与真实闭环要求。", "",
        "## 原 MuJoCo 碰撞对的只读包围球筛选", "",
        "在旧力矩原生重放的两组五场景各七个规划状态，以原 0.09 m 查询上限"
        "和 0.08 m 激活距离对 2,927 个碰撞对作全对→筛选→筛选→全对配对测量。"
        "只有包围球下界严格超过查询范围再加 1 μm 时才试验跳过精确查询；"
        "若筛选集合没有近距见证，则回退全对。", "",
        "| 模式 | 状态 | 每状态精确调用 p95：全对→筛选 | 对查询 p95 ms：全对→筛选 | 漏失激活行 | 最大见证点/最小值差异 m |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for mode in ("baseline", "enabled"):
        item = sphere_screen["modes"][mode]
        lines.append(
            f"| {mode} | {item['state_count']} | "
            f"{item['pair_count_per_state']}→{item['exact_call_count']['p95']:.1f} | "
            f"{item['reference_pair_time_ms']['p95']:.3f}→"
            f"{item['screened_pair_time_ms']['p95']:.3f} | "
            f"{item['missed_active_count']} | "
            f"{item['max_active_witness_error_m']:.1e} / "
            f"{item['max_minimum_error_m']:.1e} |"
        )
    lines += [
        "", "所测 70 个状态的激活距离、见证点、全局及目标最小值逐项一致，"
        "原生力矩重放状态误差为零。MuJoCo 的远距离返回值有时为极大哨兵值，"
        "因此不能声称被跳过的每一对距离数值都等于查询上限。"
        "此筛选尚未接入 QP；上表只计碰撞对距离查询，不包含 Jacobian、"
        "区间行、QP、状态监控或力矩伺服，也不是 20 ms 全链证明。", "",
        "## 原加权 QP 的包围球筛选只读交叉试验", "",
        "在两组五场景 tick 50/100/150 的旧力矩重放状态中，按全对→筛选→"
        "筛选→全对顺序重新求解同一个 17 维加权 QP。每组重建模型、清空"
        "对偶热启动，并将候选及独立重算约束行与冻结基准逐状态核对。", "",
        "| 模式 | 方法 | 记录 | 查询加 QP p95 / 最大 ms | 超 20 ms | 精确距离调用 p95 |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for mode in ("baseline", "enabled"):
        for method in ("reference", "sphere_screen"):
            item = qp_sphere["modes"][mode][method]
            timing = item["query_plus_qp_probe_ms"]
            lines.append(
                f"| {mode} | {method} | {item['record_count']} | "
                f"{timing['p95']:.3f} / {timing['max']:.3f} | "
                f"{timing['over_20ms_count']} | "
                f"{item['exact_call_count']['p95']:.1f} |"
            )
    lines += [
        "", f"四组完整，候选不一致 {qp_sphere['candidate_mismatch_count']}。"
        "筛选降低原 MuJoCo 碰撞对精确查询数，但两组筛选后的部分链路 p95 "
        "仍超过原 20 ms 周期。该试验未发出候选、未执行新区间闭环，"
        "没有计入完整监控和力矩伺服；不能将局部加速算作全链接入通过。", "",
        "## 五场景初始状态的加权 QP 只读预检", "",
        "另在两组五场景的 tick 0 从根区间以最多 255 点查询，并将所需区间行"
        "送入原单个 17 维加权 QP；候选只经原动作验证，未驱动力矩伺服。"
        "本次 10 个初始状态都只用五个根中点完成静态判定，冻结可达筛选"
        "不要求任何 PCC 导数行，仍保留原 MuJoCo 和胶囊行。"
        "首次试运行因只读 QP 包装器在空区间行集合上调用 `min` 而在求解前报错；"
        "失败快照已保存，修复后重新生成下表。", "",
        "| 模式 | 初始状态 | 代理安全 | 包络证据支持 | 验证出命令 | 接入前提合格 | 查询加 QP 探针 p95 / 最大 ms |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for mode in ("baseline", "enabled"):
        item = initial_probe["summary"][mode]
        lines.append(
            f"| {mode} | {item['probe_count']} | {item['proxy_safe_count']} | "
            f"{item['envelope_supported_count']} | "
            f"{item['validated_command_count']} | "
            f"{item['admission_preconditions_met_count']} | "
            f"{item['query_plus_qp_probe_ms']['p95']:.3f} / "
            f"{item['query_plus_qp_probe_ms']['max']:.3f} |"
        )
    lines += [
        "", "初始预检说明原求解器能够在当前冻结起点生成并验证候选；"
        "空区间行并不意味着取消完整覆盖或未来的区间约束。"
        "这些计时不含状态包络检查和连续执行；enabled 的所测尾部还超过"
        "20 ms，故不能当作全链性能通过。后续状态的子空间残差、"
        "旧速度起点违例和预算未知仍阻止在线门禁。", "",
        "## 早期候选命令的单周期私有力矩分支", "",
        "对上面的 tick 50/100/150 只读 QP 候选，在旧 A.1 冻结状态另开"
        "私有 MuJoCo 分支。先以旧命令逐步复现保存的十步力矩和下一规划 qpos；"
        "再用同一参考斜坡及 67 路力矩伺服执行候选，并在实际到达的下一起点"
        "重新查询 255 点根区间、独立装配原 MuJoCo/胶囊及新区间行。"
        "这检验冻结预测之后的实际一步结果，不是新模式在线闭环。", "",
        "| 模式 | 单周期分支 | 11 状态均包含 | 实现下一起点满足 | 代理安全 | 全身几何违例窗口 | 最小包络余量 mm |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for mode in ("baseline", "enabled"):
        item = candidate_ramp["modes"][mode]
        lines.append(
            f"| {mode} | {item['ramp_count']} | {item['covered_ramp_count']} | "
            f"{item['realized_next_start_satisfied_count']} | "
            f"{item['next_proxy_safe_count']} | "
            f"{item['whole_body_violation_ramp_count']} | "
            f"{1000*item['minimum_envelope_margin_m']:.3f} |"
        )
    lines += [
        "", "候选和旧命令的差异在这 30 个分支中确实非零；旧命令分支"
        "的力矩及下一 qpos 最大误差均为零。"
        "全身几何采用原验证器有限插值检查，仍不是连续时间认证。"
        "这只是从历史状态出发的单周期反事实；不能覆盖新区间闭环累计误差、"
        "停止策略、完整五场景或 20 ms 全链时延。", "",
        "## 从初态出发的五场景私有多周期分支", "",
        "从发布 A.1 的 enabled 五个初态启动私有 MuJoCo 模型，每场景预声明"
        "400 个规划周期；之后只使用新区间 QP 验证出的命令，经原十步斜坡"
        "和 67 路力矩伺服推进。旧 trace 只作初始条件和逐时刻差异对照。"
        "这不是生产控制器在线接入。", "",
        "| 场景 | 执行周期 | 旧起点首次违例抽样 tick | 私有对应起点 | "
        "最小下一起点松弛 mm/s | 最小 500 Hz 包络余量 mm | 预检＋QP p95 ms | 超 20 ms |",
        "| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: |",
    ]
    for item in private_rollout["scenes"]:
        timing = item["private_preflight_plus_qp_timing"]
        lines.append(
            f"| {item['scenario_id'][-2:]} | {item['executed_ticks']} | "
            f"{item['old_first_sampled_start_violation_tick']} | "
            f"{item['private_start_status_at_old_violation_tick']} | "
            f"{1000*item['minimum_next_start_slack_m_s']:.3f} | "
            f"{1000*item['minimum_ramp_envelope_margin_m']:.3f} | "
            f"{timing['p95_ms']:.3f} | {timing['over_20ms_count']} |"
        )
    lines += [
        "", f"私有分支完整 {private_rollout['complete_horizon_count']}/5；"
        f"旧轨迹首次速度违例抽样时刻的私有路径起点满足 "
        f"{private_rollout['old_violation_tick_start_satisfied_count']}/5。"
        "这说明所测旧起点违例并非同一场景所有轨迹必然失败。"
        "五组从第 1 周期起均超出原严格形状子空间判据，且预检＋QP p95 "
        "都超过 20 ms；即使所测 500 Hz 胶囊在 PCC 管内，"
        "也不能把这些私有动作称为获准在线执行。", "",
        "## 私有轨迹冻结状态的原 QP 包围球筛选", "",
        "从上述五条私有力矩轨迹各取 tick 1/50/100/200/300/399，"
        "按全对→筛选→筛选→全对重建同一 17 维加权 QP。"
        "每个冻结状态清空对偶热启动；对比候选、执行选择、动作状态和全部约束行。"
        "本试验只读状态，没有将候选发往力矩伺服。", "",
        "| 方法 | 记录 | 预检＋QP p95 / p99 / 最大 ms | 超 20 ms | "
        "原 MuJoCo 对处理 p95 ms | 精确对查询 p95 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for method in ("reference", "sphere_screen"):
        item = private_qp_sphere["summary"][method]
        timing = item["preflight_plus_qp_ms"]
        lines.append(
            f"| {method} | {item['record_count']} | "
            f"{timing['p95']:.3f} / {timing['p99']:.3f} / "
            f"{timing['max']:.3f} | {timing['over_20ms_count']} | "
            f"{item['mujoco_pair_block_ms']['p95']:.3f} | "
            f"{item['exact_pair_calls']['p95']:.1f} |"
        )
    lines += [
        "", f"120 条只读记录的候选和约束行不一致 "
        f"{private_qp_sphere['failure_count']}。"
        "此抽样计时从 `mj_forward` 之后开始；筛选降低了所测 MuJoCo 精确查询数，"
        "但预检＋QP 的 p99 和最大值"
        "仍超过 20 ms；它也未计独立重算、十步力矩和在线调度。"
        "该局部计时不能代替完整周期验收，严格子空间门禁仍不满足。", "",
        "## 五条私有轨迹全部规划状态的 QP 筛选复算", "",
        "将上述抽样扩展到每条私有轨迹的全部 400 个已保存规划状态。"
        "参考与筛选方法各保持自己的连续对偶热启动；方法先后按场景交替，"
        "候选与真实已执行私有命令逐状态比较。计时从 `mj_forward` 之前开始，"
        "覆盖该次正运动学、当前包络、根区间查询及 QP；"
        "不含独立约束重算、力矩伺服或在线调度。这仍是冻结状态的只读复算，"
        "没有重新执行筛选候选，也没有测完整控制周期。", "",
        "| 方法 | 冻结状态 | 预检＋QP p95 / p99 / 最大 ms | 超 20 ms | "
        "原 MuJoCo 对处理 p95 ms | 精确对查询 p95 | 对已执行命令最大差 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for method in ("reference", "sphere_screen"):
        item = private_qp_full["summary"][method]
        timing = item["preflight_plus_qp_ms"]
        lines.append(
            f"| {method} | {timing['count']} | "
            f"{timing['p95']:.3f} / {timing['p99']:.3f} / "
            f"{timing['max']:.3f} | {timing['over_20ms_count']} | "
            f"{item['mujoco_pair_block_ms']['p95']:.3f} | "
            f"{item['exact_pair_calls']['p95']:.1f} | "
            f"{item['saved_command_max_error']['max']:.2e} |"
        )
    lines += [
        "", f"全量 {private_qp_full['record_count']} 条复算记录，"
        f"逐状态候选、约束行哈希、动作状态和保存命令不一致 "
        f"{private_qp_full['failure_count']}。"
        "筛选组的局部 p95 低于 20 ms，但 p99、最大值和超周期计数表明尾部仍越过"
        "规划周期；缺少完整周期计时，新执行路径的严格子空间条件也未成立。", "",
        "## 私有力矩轨迹的单独重放与区间重算", "",
        "另启程序从保存的私有初态和 67 路力矩逐步重放，"
        "不调用 QP 求解，也不复用控制分支的区间批次；每个规划边界"
        "重查根区间，重算固定区间的 17 维梯度、目标漂移、原 MuJoCo／"
        "胶囊约束和下一起点松弛。", "",
        f"共检查 {private_recompute['checked_task_states']} 个规划状态、"
        f"{private_recompute['checked_interval_rows']} 条区间行，"
        f"不一致 {private_recompute['failure_count']}；最大状态误差 "
        f"{private_recompute['maximum_native_state_error']:.2e}，"
        f"最大代理查询差 {private_recompute['maximum_query_error_m']:.2e} m，"
        f"最大起点松弛差 "
        f"{private_recompute['maximum_start_slack_error_m_s']:.2e} m/s。"
        "重算仍共享声明的 MuJoCo/PCC 几何模型，不是独立物理测量。"
        "原未补偿伺服在首个 2 ms 状态已偏离严格子空间；"
        "全链时延、未测场景及连续时间证据缺口仍在。", "",
        "## 隐式积分阻尼补偿的私有力矩分支", "",
        "对原伺服前十个 500 Hz 步的逐步重放表明：五场景均在第一步离开"
        "原 1e-10 rad 子空间判据，且没有加速度裁剪或力矩饱和。"
        "原模型使用 `mjINT_IMPLICITFAST`，其速度更新包含有效惯量 "
        "`M + dt·D`；当前模型所测关节阻尼为 0.05。"
        "只在私有分支的 67 路伺服逆动力学中加入这一阻尼项，"
        "保持原机器人映射、力矩上限、十步斜坡及 MuJoCo 积分器。"
        "这不是对原 1e-10 判据的放宽。", "",
        "| 场景 | 原伺服前十步最大残差 rad | 补偿后前十步最大残差 rad | "
        "补偿最大力矩变化 Nm | 触限步 |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for original, modified in zip(servo_origin["scenes"],
                                  discrete_first_tick["scenes"]):
        lines.append(
            f"| {original['scene_id'][-2:]} | "
            f"{original['maximum_first_tick_position_residual_rad']:.2e} | "
            f"{modified['maximum_position_residual_rad']:.2e} | "
            f"{modified['maximum_torque_change_nm']:.6f} | "
            f"{modified['torque_saturation_step_count']} |"
        )
    lines += [
        "", "上表原列是前十步的最大残差，五场景首个伺服步均已超判据；"
        "补偿列仅是同初态反事实，不能用它声称已完成新闭环。"
        "MuJoCo 的有效惯量和积分方法说明见 "
        "[官方文档](https://mujoco.readthedocs.io/en/latest/computation/)。", "",
        "再从发布 A.1 的五个初态启动补偿后的私有区间 QP，"
        "每一规划周期用新状态重新查询、装配、验证并执行命令。"
        "所测新轨迹不是旧 trace 的后续引用。", "",
        "| 场景 | 执行周期 | 最大规划边界子空间残差 rad | "
        "最小 500 Hz 包络余量 mm | 最小下一起点松弛 mm/s | "
        "预检＋QP p95 / 最大 ms | 超 20 ms |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for item in discrete_private["scenes"]:
        timing = item["preflight_plus_qp_timing"]
        lines.append(
            f"| {item['scenario_id'][-2:]} | {item['executed_ticks']} | "
            f"{item['maximum_subspace_residual_linf_rad']:.2e} | "
            f"{1000*item['minimum_ramp_envelope_margin_m']:.3f} | "
            f"{1000*item['minimum_next_start_slack_m_s']:.3f} | "
            f"{timing['p95_ms']:.3f} / {timing['max_ms']:.3f} | "
            f"{timing['over_20ms_count']} |"
        )
    lines += [
        "", f"独立原生力矩重放检查了 "
        f"{discrete_private['checked_500hz_states']} 个 500 Hz 状态，"
        f"最大严格子空间残差 "
        f"{discrete_private['maximum_500hz_subspace_residual_linf_rad']:.2e} rad；"
        f"重算 {discrete_private['independent_interval_row_count']} 条区间行，"
        f"不一致 {discrete_private['independent_recompute_failure_count']}，"
        f"原力矩上限违例 {discrete_private['torque_limit_violation_count']}。"
        "这仍共享 MuJoCo/PCC 模型，没有独立物理测量或连续时间认证。"
        "该私有控制链的预检＋QP p95 全部超过 20 ms，"
        "且未完成完整周期和生产在线故障注入；第三阶段门禁继续关闭。", "",
        "## 批量查询与等价 ADMM 的私有五场景结果", "",
        "在补偿伺服、原安全行及 17 维加权 QP 不变的条件下，"
        "私有路径试验批量点查询、保守球界筛选及 ADMM 乘积复用。"
        "与上述五场景补偿轨迹逐项比较，67 路力矩、位置与任务速度差均为零；"
        f"独立重放 {optimized['checked_task_states']} 个任务状态、"
        f"{optimized['independent_interval_rows']} 条区间行，不一致 "
        f"{optimized['independent_recompute_failure_count']}。", "",
        "| 场景 | 私有预检＋QP p95 ms | p99 ms | 最大 ms | 超 20 ms 周期 |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for item in optimized["scenes"]:
        timing = item["preflight_plus_qp_timing"]
        lines.append(
            f"| {item['scenario_id'][-2:]} | {timing['p95_ms']:.3f} | "
            f"{timing['p99_ms']:.3f} | {timing['max_ms']:.3f} | "
            f"{timing['over_20ms_count']} |"
        )
    lines += [
        "", f"p95 超过 20 ms：{', '.join(optimized['p95_over_20ms_scenes'])}。"
        "这是单轮 Windows 私有闭环计时，不能把不同运行之间的全部时延变化"
        "归因于优化。此前仅批量点和球界筛选的场景 00 独立运行 p95 为 "
        f"{prepared_screened['private_preflight_plus_qp_timing']['p95_ms']:.3f} ms，"
        "也保留为超预算实测。仪器化全 QP 试验每输入多次求解，其总时延无效；"
        "独立 ABBA 的求解器计时与五场景轨迹见各自哈希绑定报告。"
        "尚未完成正式全链计时或在线接入，门禁继续关闭。", "",
        "### 预声明三轮私有计时", "",
        "固定三轮各五场景、每场景 400 周期，第二轮逆序。"
        f"15 个场景轮次中 {repeat['p95_over_20ms_run_count']} 个 p95 "
        "超 20 ms；全部力矩轨迹哈希与独立重放的基线完全相同。"
        "这是私有预检＋QP 范围，不是正式任务全链。", "",
        "| 轮次 | 场景 | p95 ms | p99 ms | 最大 ms | 超 20 ms 周期 | 最长连续超时 |",
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for item in repeat["runs"]:
        timing = item["preflight_plus_qp_timing"]
        lines.append(
            f"| {item['round']} | {item['scenario_id'][-2:]} | "
            f"{timing['p95_ms']:.3f} | {timing['p99_ms']:.3f} | "
            f"{timing['max_ms']:.3f} | {timing['over_20ms_count']} | "
            f"{timing['longest_over_20ms_run']} |"
        )
    lines += [
        "", "重复结果包含通过和失败轮次；同一轨迹的 Windows 计时波动"
        "不能作为控制器等价性或硬实时保证。现有三轮诊断明确未达到"
        "预声明的全轮次 p95 门槛。", "",
        "## 17 维正定诊断解的私有 Cholesky 复用", "",
        "单次场景 02 阶段计时显示，原 `np.linalg.solve` 的 17×17 "
        f"调用 p95 为 {qp_phase['phase_statistics']['dense_solve_ms']['p95_ms']:.3f} ms；"
        "原 6×6 反作用求解、碰撞行与 ADMM 分别计时。"
        "私有分支仅将正定加权 Hessian 的无约束诊断解换为 Cholesky，"
        "不改变候选优化问题或安全约束。", "",
        "| 场景 | 首轮预检＋QP p95 ms | p99 ms | 最大 ms | 超 20 ms 周期 |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for item in cholesky["scenes"]:
        timing = item["preflight_plus_qp_timing"]
        lines.append(
            f"| {item['scenario_id'][-2:]} | {timing['p95_ms']:.3f} | "
            f"{timing['p99_ms']:.3f} | {timing['max_ms']:.3f} | "
            f"{timing['over_20ms_count']} |"
        )
    lines += [
        "", f"预声明三轮共 {cholesky['repeat_run_count']} 个场景轮次，"
        f"p95 超 20 ms 为 {cholesky['repeat_p95_over_20ms_count']}；"
        f"总超周期 {cholesky['repeat_over_20ms_cycle_count']}，"
        f"最大单周期 {cholesky['repeat_maximum_preflight_plus_qp_ms']:.3f} ms。"
        f"独立重放 {cholesky['independent_task_states']} 个任务状态、"
        f"{cholesky['independent_interval_rows']} 条区间行，"
        f"不一致 {cholesky['independent_failure_count']}；"
        "五场景轨迹哈希与原补偿基线完全相同。", "",
        "因此 400 周期私有预检＋QP 重复时延诊断通过，但不是完整任务周期"
        "或硬实时证明。旧影子门禁、生产在线完整五场景闭环、原 26/11 项新 trace "
        "与故障注入仍未全部验收；阶段二总门禁保持关闭。", "",
        "### 单场景完整 27 s 私有诊断", "",
        "从已发布启用组场景 03 的初态开始，将同一私有 Cholesky 诊断"
        "延长至预声明的 1,350 个规划周期；旧轨迹此后只作为差异对照。"
        "另启进程从保存的 67 路力矩原生重放并重算固定区间行。", "",
        "| 指标 | 结果 |", "| --- | ---: |",
        f"| 执行 / 预声明周期 | {full_scene['executed_ticks']} / "
        f"{full_scene['predeclared_horizon_ticks']} |",
        f"| 预检＋QP p95 / p99 / 最大 ms | {full_scene_timing['p95_ms']:.3f} / "
        f"{full_scene_timing['p99_ms']:.3f} / {full_scene_timing['max_ms']:.3f} |",
        f"| 超 20 ms 周期 / 最长连续次数 | "
        f"{full_scene_timing['over_20ms_count']} / "
        f"{full_scene_timing['longest_over_20ms_run']} |",
        f"| 独立重放规划状态 / 区间行 / 不一致 | "
        f"{full_recompute['checked_task_states']} / "
        f"{full_recompute['checked_interval_rows']} / "
        f"{full_recompute['failure_count']} |",
        f"| 最小下一起点松弛 mm/s | "
        f"{1000 * full_scene['minimum_realized_next_start_slack_m_s']:.3f} |",
        f"| 最小原生 500 Hz 包络余量 mm | "
        f"{1000 * full_scene['minimum_ramp_envelope_margin_m']:.3f} |",
        f"| 17 维诊断解回退 / 力矩上限违例 | "
        f"{full_scene['cholesky_unconstrained_fallback_count']} / "
        f"{full_recompute['torque_limit_violation_count']} |",
        "", "这次单场景长时轨迹满足所测代理、起点、离散包络和严格"
        "形状子空间条件；独立重放最大状态误差为 "
        f"{full_recompute['maximum_native_state_error']:.1e}。"
        "计时仅含私有预检＋QP，不含独立重算、十步力矩伺服、"
        "状态监控和实时调度。正式生产切换、"
        "原 26/11 项新 trace 验收及故障注入仍未完成；"
        "阶段二接入门禁保持关闭。", "",
        "同一条私有轨迹再按原冷影子预算独立复算 1,350 个保存规划状态。"
        "每状态将分支查询点、全部区间中点重算、形状与 MuJoCo Jacobian、"
        "局部细化及查询装配耗时纳入一张账；查询选择与保存记录逐项一致。"
        f"点总量最大 {budget_audit['count_statistics']['total_point_evaluation_count']['max']:.0f} "
        f"（冻结预算 {cold['modes']['enabled']['budget']['total_point_evaluations']}），"
        f"Jacobian 最大 {budget_audit['count_statistics']['jacobian_evaluation_count']['max']:.0f} "
        f"（预算 {cold['modes']['enabled']['budget']['jacobian_evaluations']}），"
        f"查询与区间装配最大 {budget_audit['count_statistics']['query_and_interval_assembly_ms']['max']:.3f} ms；"
        f"超限周期 {budget_audit['budget_overrun_tick_count']}，"
        f"对照不一致 {budget_audit['parity_failure_count']}。"
        "局部细化在该路径没有启用，因此实际计数为 0。"
        "这属于保存状态的事后预算核对；私有执行器尚未把总预算"
        "作为动作放行条件，也未测完整控制周期。", "",
        "### 五场景完整 27 s 私有诊断与工作域门禁", "",
        "其余四个场景也以相同私有路径独立执行到 1,350 周期；"
        "每条新轨迹分别做原生力矩重放、区间行重算与统一预算审计。"
        "以下计时仅是预检＋QP，未完成生产在线任务层验收。", "",
        "| 场景 | 周期 | 独立区间行 | 预算超限 | 域外仍 TRACK | 预检＋QP p95 / p99 / 最大 ms | 超 20 ms 周期 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for item in full_private_gate["scenes"]:
        timing = item["preflight_plus_qp_timing"]
        lines.append(
            f"| {item['scenario_id'][-2:]} | {item['executed_ticks']} | "
            f"{item['recomputed_interval_rows']} | "
            f"{item['budget_overrun_ticks']} | "
            f"{item['strict_domain_violation_ticks']} | "
            f"{timing['p95_ms']:.3f} / {timing['p99_ms']:.3f} / "
            f"{timing['max_ms']:.3f} | {timing['over_20ms_count']} |")
    lines += [
        "", f"五场景共 {full_private_gate['executed_task_ticks']} 个私有执行周期、"
        f"{full_private_gate['recomputed_task_states']} 个独立重放任务状态、"
        f"{full_private_gate['recomputed_interval_rows']} 条独立区间行，"
        "重算不一致及冻结影子预算超限均为 0。",
        "scene_01 在 tick 888–950 连续 63 个周期越出 B.1 声明的工作域，"
        "私有诊断仍发出 TRACK 命令。原生重放一致性和当前状态代理安全"
        "均不能替代声明域内的包络证据；这些周期在单独失败记录中逐项保存。"
        "因此五场景完整私有试验暴露了明确的在线准入阻断条件，"
        "阶段二总门禁仍为 GATE_NOT_MET。", "",
        "### 声明工作域越界前的严格私有拒绝", "",
        "在场景 01 的独立私有复跑中，原候选在 tick 887 的第 4 个 "
        "2 ms 力矩微步将越出声明工作域；门禁在提交该周期的任何力矩前"
        "将动作标为 UNCERTIFIED，记录失败原因并拒绝候选。"
        f"保存轨迹只有 {strict_stop['executed_ticks']} 个已执行任务周期、"
        f"{strict_replay['native_torque_steps']} 个力矩步。单独进程原生重放"
        f" {strict_replay['checked_native_states']} 个状态，最大误差 "
        f"{strict_replay['native_replay_max_state_error']:.3e}；执行状态距工作域"
        f"边界最小 {strict_replay['minimum_executed_work_domain_margin_rad']:.6g} rad。"
        "已执行力矩和 qpos 与先前完整诊断前缀逐值一致。"
        "这证明了私有试验中拒绝发生在下一伺服步前；该轨迹提前停止，"
        "仍不能替代五场景完整新模式验收。十步预测本身尚未计入"
        "预检＋QP 时延，也没有真实在线截止期限证据。", "",
        "### 同一 QP 的工作域端点约束：五场景新私有闭环", "",
        "旧私有轨迹的 scene_01 越界反例促使在同一个 17 维 QP 中增加"
        "冻结十步斜坡的形状工作域端点行；原速度盒、碰撞行和安全裕度"
        "未放宽。执行前仍预测 11 个 2 ms MuJoCo 状态，任何域外状态"
        "都拒绝提交整段候选。新变体完整运行五场景各 1,350 周期。", "",
        "| 场景 | 独立区间行 | 最小工作域余量 rad | 整机最小净空 m | 路径 RMSE m | 预检＋QP p95 / 最大 ms | 超 20 ms |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for item in domain_five["scenes"]:
        timing = item["preflight_plus_qp_timing"]
        lines.append(
            f"| {item['scenario_id'][-2:]} | {item['recomputed_interval_rows']} | "
            f"{item['minimum_work_domain_margin_rad']:.6g} | "
            f"{item['whole_body_minimum_clearance_m']:.6g} | "
            f"{item['continuum_path_rmse_m']:.6g} | "
            f"{timing['p95_ms']:.3f} / {timing['max_ms']:.3f} | "
            f"{timing['over_20ms_count']} |")
    lines += [
        "", f"五场景独立原生重放 {domain_five['independent_recomputed_task_states']} "
        f"个任务状态、重算 {domain_five['independent_recomputed_interval_rows']} "
        f"条区间行，并扫描 {domain_five['independent_checked_2ms_states']} "
        "个实际微状态；原任务跟踪与整机净空阈值均通过。"
        "冻结预算在保存状态上未超限。"
        f"预检＋QP 共 {domain_five['preflight_plus_qp_over_20ms_tick_count']} "
        "个单周期超过 20 ms，且未计十步预测预检或完整在线调度。"
        "这是新私有分支的模型内闭环证据；生产在线开关、原 26/11 项"
        "正式新模式验收和全链时限仍未完成，阶段门禁继续关闭。", "",
        "[五场景工作域、路径误差和时延图]"
        "(../domain_endpoint_private_five_gate/b2-domain-private-five.png)", "",
        "## 补偿轨迹上的球界筛选时延核对", "",
        "在上述补偿私有轨迹的全部五场景 2,000 个保存规划状态上，"
        "原精确碰撞对与保守球界筛选各重新求解一次 17 维 QP，"
        "共 4,000 次；约束行、候选与已验证命令的配对失败数为 "
        f"{paired_discrete_qp['failure_count']}。"
        "另以筛选 QP 独立运行场景 00 的 400 周期私有闭环，"
        "与未筛选补偿分支的状态最大差 "
        f"{screened_parity['maximum_qpos_difference_m_or_rad']:.2e}、"
        "力矩最大差 "
        f"{screened_parity['maximum_torque_difference_nm']:.2e} Nm。", "",
        "| 测量 | 原对查询 | 球界筛选 |",
        "| --- | ---: | ---: |",
        f"| 五场景冻结状态预检＋QP p95 ms | "
        f"{paired_discrete_qp['summary']['reference']['preflight_plus_qp_ms']['p95']:.3f} | "
        f"{paired_discrete_qp['summary']['sphere_screen']['preflight_plus_qp_ms']['p95']:.3f} |",
        f"| 精确碰撞对查询 p95 次／状态 | "
        f"{paired_discrete_qp['summary']['reference']['exact_pair_calls']['p95']:.0f} | "
        f"{paired_discrete_qp['summary']['sphere_screen']['exact_pair_calls']['p95']:.0f} |",
        f"| 场景 00 筛选私有闭环预检＋QP p95 ms | — | "
        f"{screened_scene['private_preflight_plus_qp_timing']['p95_ms']:.3f} |",
        "", "球界筛选减少了原 MuJoCo 对的精确查询数，"
        "但成对只读 p95 和场景 00 私有闭环 p95 均超过原 20 ms 周期。"
        "后者仍未测完整调度和伺服时延，不能作为第三阶段在线接入依据。", "",
        "## 候选分支的同区间前瞻误差", "",
        "对同一 30 个私有候选分支，重建原只读 QP 的区间划分与筛选 ID，"
        "在执行后一规划起点冻结相同 ID 重算 h、17 维广义梯度及目标漂移。"
        "因此下表的误差来自同一材料区间函数的状态变化。首次运行因下一"
        "qpos 数组切片少一项而报错；部分行和错误快照已单独保留，修复后"
        "在新目录完整重跑。", "",
        "| 模式 | 单周期分支 | 固定区间行 | h 最大预测高估 mm/s | CBF 起点松弛最大预测高估 mm/s | 超 5 mm/s 经验裕度 | 实际起点违例 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for mode in ("baseline", "enabled"):
        item = candidate_prediction["modes"][mode]
        lines.append(
            f"| {mode} | {item['case_count']} | {item['selected_row_count']} | "
            f"{1000*item['h_optimism_rate_m_s']['max']:.3f} | "
            f"{1000*item['cbf_start_slack_optimism_m_s']['max']:.4f} | "
            f"{item['cbf_optimism_over_frozen_margin_count']} | "
            f"{item['realized_start_violation_count']} |"
        )
    lines += [
        "", "h 残差除以原 20 ms 周期转成速度单位；CBF 松弛残差还包含"
        "广义梯度与目标漂移变化。这里 5 mm/s 仍是原经验配置，不是"
        "通用或严格误差上界；30 个单周期样本不能推出连续闭环可靠性。", "",
        "## 全部保存规划状态的实际胶囊包络", "",
        "对旧 A.1 两组五场景全部保存的规划 qpos 直接执行 MuJoCo 正运动学，"
        "逐状态检查实际胶囊是否包含于原 PCC 管；所用 trace 与上述原生重放审计逐场景哈希一致。"
        "这不是第二次力矩重放，也不是新区间模式动作。", "",
        "| 模式 | 保存状态 | 当前状态包含 | 未包含 | 最小余量 mm | 包络检查 p95 ms |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for mode in ("baseline", "enabled"):
        item = sweep["modes"][mode]
        lines.append(
            f"| {mode} | {item['state_count']} | "
            f"{item['status_counts'].get('COVERED_AT_THIS_STATE', 0)} | "
            f"{item['status_counts'].get('NOT_COVERED_AT_THIS_STATE', 0)} | "
            f"{1000 * item['minimum_margin_m']['min']:.3f} | "
            f"{item['check_ms_excluding_mj_forward']['p95']:.3f} |"
        )
    lines += [
        "", "全量余量均为正，但这只证明保存时刻的模型几何包含。"
        "检查耗时不含 MuJoCo 正运动学、区间查询和 QP；无法单独证明 20 ms 全链。"
        "本节全量扫描不包括斜坡内部；下节另检查预定的 20 个斜坡窗口。"
        "仍无跨规划周期的包络保持证明。", "",
        "## 原生力矩重放的十步斜坡中间状态", "",
        "每场景预定 tick 50 和首次抽样旧速度 CBF 起点违例 tick 两个窗口，"
        "原生重放旧 trace 的 500 Hz 力矩，对十步斜坡两端及九个内部状态"
        "分别检查实际胶囊包络。规划边界与保存状态逐步一致；未执行新区间 QP 命令。", "",
        "| 模式 | 窗口 | 500 Hz 状态 | 当前状态包含 | 最小余量 mm | 内部余量低于两端的窗口 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for mode in ("baseline", "enabled"):
        item = micro["modes"][mode]
        lines.append(
            f"| {mode} | {item['window_count']} | {item['microstate_count']} | "
            f"{item['covered_microstate_count']} | "
            f"{1000 * item['minimum_margin_m']['min']:.3f} | "
            f"{item['interior_below_both_endpoints_count']} |"
        )
    lines += [
        "", "所选 20 个窗口不能代表全部 13,500 个斜坡；500 Hz 离散观察"
        "不证明两次观测之间、连续时间或新区间闭环安全。", "",
        "## 旧 A.1 全轨迹每个物理步的实际胶囊包络", "",
        "对原两组五场景的全部力矩按 MuJoCo 原生重放，在每个 500 Hz 物理步"
        "起点和最终端点检查实际胶囊与原 PCC 管。全部 13,500 个保存规划状态"
        "和先前 220 个选定斜坡状态逐状态交叉核对；逐步记录和失败快照保存在"
        "独立的不可覆盖审计目录。", "",
        "| 模式 | 力矩步 | 500 Hz 检查状态 | 当前状态包含 | 最小余量 mm | 最大重放误差 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for mode in ("baseline", "enabled"):
        item = full_torque["modes"][mode]
        lines.append(
            f"| {mode} | {item['torque_steps']} | {item['checked_states']} | "
            f"{item['covered_states']} | {1000*item['minimum_margin_m']:.3f} | "
            f"{item['maximum_native_replay_error']:.2e} |"
        )
    lines += [
        "", "这是历史旧控制轨迹的离散时刻几何证据，仍不能推断相邻 2 ms"
        "状态之间的连续时间包含，也不能替代新区间模式的五场景闭环和独立重放。"
        "包络检查耗时不含 MuJoCo 正运动学、区间查询、Jacobian、QP 和执行，"
        "不能作为 20 ms 全链验收。", "",
        "## 只读查询计算复用试验", "",
        "首次批量点模型试验在 30 个冻结状态上没有获得稳定提速，"
        "因此未用于控制或正式影子结果。随后将同一状态的五段完整变换仅计算一次，"
        "并保持参考查询的分区细分、区间下界及终止规则。"
        "冷查询试验覆盖 58 个已保存的持久热查询抽样未知状态和 10 个早期状态；"
        "以下全量对照使用旧 A.1 的 13,500 个保存规划 qpos。", "",
        "| 模式 | 全量状态 | 分区/判定不一致 | 最大距离偏差 m | 参考查询 p95 ms | 前缀复用 p95 ms | 单状态配对加速比 p50 | 持久未知 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for mode in ("baseline", "enabled"):
        item = prepared_full["modes"][mode]
        lines.append(
            f"| {mode} | {item['counts']['task_states']} | "
            f"{item['counts'].get('parity_failures', 0)} | "
            f"{item['maximum_parity_error_m']:.2e} | "
            f"{item['reference_query_ms']['p95']:.3f} | "
            f"{item['prepared_query_ms']['p95']:.3f} | "
            f"{item['per_state_speedup']['p50']:.3f} | "
            f"{item['counts']['UNKNOWN_CROSSES_GATE']} |"
        )
    lines += [
        "", "全量试验以参考结果推进持久分区；每状态另用试验实现检查同一输入。"
        "查询时间只在本次配对运行内比较，不能与先前报告的 p95 直接拼接。"
        "该改进降低了计算时间，没有消除未知或旧起点违例；"
        "也未计入 MuJoCo 正运动学、广义 Jacobian、原约束、QP 和力矩执行。"
        "没有新区间模式全链 20 ms 验收或连续时间认证。", "",
        "## 根区间未知状态的预算敏感性", "",
        "针对 64 点根区间重查询后仍未知的 998 个冻结 A.1 状态，"
        "分别以 127 和 255 点预算从五段根区间独立重查。"
        "下表仅统计这些冻结状态；每个查询均与参考实现逐状态核对分区和判定。", "",
        "| 模式 | 冻结未知 | 点预算 | 确定安全 | 确定低于门槛 | 仍未知 | 前缀复用查询 p95 ms |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for mode in ("baseline", "enabled"):
        item = root_rescue["modes"][mode]
        counts = item["counts"]
        for budget in root_rescue["budgets"]:
            prefix = str(budget)
            lines.append(
                f"| {mode} | {counts['frozen_states']} | {budget} | "
                f"{counts.get(prefix + '_PROXY_CLEARANCE_AT_LEAST_GATE', 0)} | "
                f"{counts.get(prefix + '_PROXY_CLEARANCE_BELOW_GATE', 0)} | "
                f"{counts.get(prefix + '_UNKNOWN_CROSSES_GATE', 0)} | "
                f"{item['budgets'][prefix]['prepared_query_ms']['p95']:.3f} |"
            )
    lines += [
        "", "255 点预算使 enabled 组这 874 个冻结未知状态全部判为安全，"
        "baseline 组仍有 17 个未知。该结果仅说明预算敏感性；"
        "不覆盖其他已确定低于门槛的状态、旧速度起点违例或全任务动态演化。"
        "查询耗时不含 MuJoCo 正运动学、广义 Jacobian、原约束、QP 与力矩伺服；"
        "较大预算尚未进入在线控制，不能由此宣称 20 ms 全链通过。", "",
        "## 全部保存状态的 255 点根区间查询", "",
        "为避免只看先前未知状态带来的选择偏差，对两组五场景全部 13,500 个"
        "保存规划状态分别从五段根区间重新查询；每状态参考实现与前缀复用实现"
        "逐项比较，先前 998 个未知状态还与独立预算阶梯交叉核对。", "",
        "| 模式 | 状态 | 代理安全 | 代理低于 5 mm | 未知 | 预算耗尽 | 分区/判定不一致 | 参考查询 p95 ms | 前缀复用 p95 ms |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for mode in ("baseline", "enabled"):
        item = full_root["modes"][mode]
        counts = item["counts"]
        lines.append(
            f"| {mode} | {counts['task_states']} | "
            f"{counts.get('PROXY_CLEARANCE_AT_LEAST_GATE', 0)} | "
            f"{counts.get('PROXY_CLEARANCE_BELOW_GATE', 0)} | "
            f"{counts.get('UNKNOWN_CROSSES_GATE', 0)} | "
            f"{counts['budget_exhausted']} | "
            f"{counts.get('parity_failures', 0)} | "
            f"{item['reference_query_ms']['p95']:.3f} | "
            f"{item['prepared_query_ms']['p95']:.3f} |"
        )
    lines += [
        "", "enabled 旧轨迹的保存状态在此高预算根区间诊断中全部确定代理安全；"
        "baseline 则有大量确定低于门槛的旧状态，继续增加查询预算无法使同一"
        "状态的真实代理净空变大。代理分类不等于 MuJoCo 实际碰撞。"
        "这些是冷查询计时，不含 MuJoCo 正运动学、广义 Jacobian、原约束、"
        "QP 和力矩执行；也没有解决旧速度起点违例或证明新模式全链 20 ms。", "",
        "## 冻结留出与真实几何", "",
        f"旧 B.1 独立留出 {heldout['counts']['checked_count']} 例完成新补的 MuJoCo 离散链对照："
        f"代理假安全 {heldout['counts']['empirical_proxy_false_safe']}、"
        f"代理误拒绝 {heldout['counts']['empirical_proxy_false_reject']}。",
        f"新冻结的 5 mm 附近 {near['counts']['checked_count']} 例分布于门槛两侧："
        f"代理确定低于门槛 {near['counts'].get('PROXY_CLEARANCE_BELOW_GATE', 0)}、"
        f"确定达到门槛 {near['counts'].get('PROXY_CLEARANCE_AT_LEAST_GATE', 0)}、"
        f"未知 {near['counts'].get('UNKNOWN_CROSSES_GATE', 0)}；"
        f"独立 MuJoCo 对照中的代理假安全 {near['counts']['empirical_proxy_false_safe']}。",
        "样本由 B.1 区间查询定位到代理门槛附近，冻结后才计算真实几何；"
        "密采样或 B.1 区间中点均不被当作连续真值。", "",
        "## 证据边界与下一动作", "",
        "`interval_well_formed`、完整覆盖、精确模型下 K=1 假设、双精度认证状态、"
        "实际链子空间残差与有限样本包络证据仍分开报告。"
        "当前在线接入门禁未通过，应继续处理持久分区未知、预算与保守性；"
        "不接入新区间 QP，也不引用旧 trace 冒充新控制运行。", "",
    ]
    doc = output_dir / "STAGE2_EVIDENCE.md"
    with doc.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write("\n".join(lines))
    manifest = {
        "schema": "v6_2_b2_stage2_hash_manifest_v1",
        "status": status,
        "gate_blockers": blockers,
        "sources": {name: {"path": path.as_posix(), "sha256": _sha(path),
                           "bytes": path.stat().st_size}
                    for name, path in sources.items()},
        "generated_document": {"path": doc.as_posix(),
                               "sha256": _sha(doc), "bytes": doc.stat().st_size},
    }
    with (output_dir / "stage2_manifest.json").open(
        "x", encoding="utf-8", newline="\n"
    ) as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path,
                        default=Path("v6_lite/output/v6_2_b2"))
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = finalize(args.root, args.output_dir)
    print(json.dumps({"status": result["status"],
                      "document": result["generated_document"]}, indent=2))


if __name__ == "__main__":
    main()
