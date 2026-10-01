"""Arithmetic analysis of existing, hash-bound research traces; no physics run.

Writes only beside this script, exclusively. --verify-only recomputes and
checks the saved summary and artifact hashes without writing any files.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np

OUTPUT = Path(__file__).resolve().parent
ROOT = OUTPUT.parents[3]
REFERENCE = ROOT / "v6_lite/output/runs/research_acceptance_01"
REFERENCE_MANIFEST_SHA = "379b9977a441c540a20cfc50a9e90614a1e0985fa2a5a73bbb7c9690f823eec0"
REFERENCE_COMMIT = "9cd1831e7c77a340464a669eac3b5aa7cd9c3c3c"
BINDING_TOLERANCE_M_S = 2e-5
SOURCE_PATHS = (
    "v6_lite/hierarchical_qp.py", "v6_lite/execution_ramp.py",
    "v6_lite/b2_interval_online.py", "v6_lite/b2_interval_online_optimized.py",
    "v6_lite/pcc_monitor.py", "v6_lite/run_v6_lite.py",
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_exclusive(name, value):
    with (OUTPUT / name).open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def analyze():
    manifest_path = REFERENCE / "manifest.json"
    assert sha(manifest_path) == REFERENCE_MANIFEST_SHA
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    references = []
    for rel, digest in manifest.items():
        source = (REFERENCE / rel.replace("\\", "/")).resolve()
        assert source.is_relative_to(REFERENCE.resolve())
        assert sha(source) == digest, str(source)
        references.append({"path": str(source), "bytes": source.stat().st_size,
                           "sha256": digest})
    assert len(references) == 46
    metadata = json.loads((REFERENCE / "simulation/run_metadata.json").read_text(encoding="utf-8"))
    assert metadata["source"]["git_commit"] == REFERENCE_COMMIT
    source_rows = []
    source_text = {}
    for rel in SOURCE_PATHS:
        result = subprocess.run(
            ["git", "show", f"{REFERENCE_COMMIT}:{rel}"], cwd=ROOT,
            check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        normalized = result.stdout.replace(b"\r\n", b"\n")
        digest = hashlib.sha256(normalized).hexdigest()
        assert digest == metadata["source"]["source_normalized_lf_sha256"][rel]
        source_text[rel] = normalized.decode("utf-8")
        source_rows.append({
            "repository_path": str(ROOT / rel),
            "git_source": f"{REFERENCE_COMMIT}:{rel}",
            "git_blob_sha256": hashlib.sha256(result.stdout).hexdigest(),
            "normalized_lf_sha256": digest,
            "matches_reference_metadata": True,
        })
    qp_source = source_text["v6_lite/hierarchical_qp.py"]
    assert 'startswith("pcc:")' in qp_source
    assert "slacks <= 2e-5" in qp_source
    assert "cfg.lookahead_model_margin_m_s" in qp_source
    metrics = json.loads((REFERENCE / "simulation/v6_lite_metrics.json").read_text(encoding="utf-8"))
    config = metrics["qp_config"]
    assert metrics["run_config"]["dispatch_clock_policy"] == "research_simulation"
    assert config["enable_pcc_cbf"] is False
    gain = float(config["pcc_clearance_barrier_gain"])
    dt = float(config["task_period_s"])
    gate = float(config["pcc_clearance_safe_m"])
    lookahead_margin = float(config["lookahead_model_margin_m_s"])
    rate_tolerance = float(config["clearance_rate_tolerance_m_s"])
    assert dt == .020 and gate == .005 and rate_tolerance == 1e-4
    # Exact mean of the ten existing ramp reference velocities, alpha=k/10.
    weights = np.asarray([(1. - float(step) / 10., float(step) / 10.)
                          for step in range(1, 11)])
    old_weight = float(np.mean(weights[:, 0]))
    new_weight = float(np.mean(weights[:, 1]))
    scenes = {}
    scene_arrays = {}
    for result in metrics["scenarios"]:
        sid = result["scenario"]["scenario_id"]
        trace_path = REFERENCE / "simulation/traces" / f"{sid}.npz"
        monitor_path = REFERENCE / "simulation/pcc_monitor" / sid / "clearance_compare.json"
        monitor = json.loads(monitor_path.read_text(encoding="utf-8"))
        keys = ("task_time", "task_interval_proxy_lower_m", "task_interval_selected_rows",
                "task_selected_command", "task_solver_candidate", "task_pcc_clearance")
        with np.load(trace_path, allow_pickle=False) as trace:
            arrays = {key: trace[key].copy() for key in keys}
        scene_arrays[sid] = arrays
        times = arrays["task_time"]
        assert len(times) == 1350
        assert np.allclose([row["time"] for row in monitor["samples"]], times,
                           rtol=0, atol=1e-12)
        closest = int(np.argmin(arrays["task_interval_proxy_lower_m"]))
        scenes[sid] = {
            "trace_path": str(trace_path), "monitor_path": str(monitor_path),
            "planning_periods": len(times),
            "ticks_with_selected_interval_rows": int(np.count_nonzero(arrays["task_interval_selected_rows"])),
            "selected_interval_row_occurrences": int(np.sum(arrays["task_interval_selected_rows"])),
            "minimum_interval_lower_m": float(arrays["task_interval_proxy_lower_m"][closest]),
            "minimum_interval_h_m": float(arrays["task_interval_proxy_lower_m"][closest] - gate),
            "closest_tick_zero_based": closest, "closest_time_s": float(times[closest]),
            "legacy_pcc_active_row_occurrences": int(monitor["pcc_constraint_active_count"]),
            "legacy_pcc_binding_row_occurrences": int(monitor["pcc_binding_constraint_count"]),
            "disabled_pcc_nonfinite_placeholder_count": int(np.count_nonzero(~np.isfinite(arrays["task_pcc_clearance"]))),
            "target_linear_velocity_m_s": result["scenario"]["target_satellite_linear_velocity_m_s"],
            "target_angular_velocity_rad_s": result["scenario"]["target_satellite_angular_velocity_rad_s"],
        }
    interval_path = REFERENCE / "interval_recompute/online_recompute_interval_rows.jsonl"
    records = [json.loads(line) for line in interval_path.read_text(encoding="utf-8").splitlines()]
    rows = []
    counts_by_tick = defaultdict(Counter)
    terminal_counts = Counter()
    for line_number, row in enumerate(records, 1):
        sid, tick = row["scenario_id"], row["tick"]
        arrays = scene_arrays[sid]
        if tick >= len(arrays["task_time"]):
            terminal_counts[sid] += 1
            continue
        counts_by_tick[sid][tick] += 1
        assert row["derivative_status"] == "SUPPORTED"
        assert abs(row["distance_lower_bound_m"] - gate - row["h_m"]) <= 1e-12
        gradient = np.asarray(row["gradient_17d"], dtype=float)
        assert gradient.shape == (17,) and np.all(np.isfinite(gradient))
        endpoint = arrays["task_selected_command"][tick]
        candidate = arrays["task_solver_candidate"][tick]
        previous = np.zeros(17) if tick == 0 else arrays["task_selected_command"][tick - 1]
        lower = -gain * row["h_m"] - row["target_drift_m_s"]
        endpoint_slack = float(gradient @ endpoint - lower)
        candidate_slack = float(gradient @ candidate - lower)
        lookahead_lower = lower - gain * dt * (
            old_weight * (gradient @ previous) + row["target_drift_m_s"]) + lookahead_margin
        lookahead_slack = float((1. + gain * dt * new_weight) * (gradient @ endpoint) - lookahead_lower)
        rows.append({
            "scenario_id": sid, "tick_zero_based": tick,
            "time_s": float(arrays["task_time"][tick]),
            "interval_id": row["interval_id"], "h_m": row["h_m"],
            "lower_m": row["distance_lower_bound_m"],
            "endpoint_slack_m_s": endpoint_slack,
            "candidate_slack_m_s": candidate_slack,
            "lookahead_slack_m_s": lookahead_slack,
            "source_path": str(interval_path), "source_jsonl_line_one_based": line_number,
        })
    for sid, scene in scenes.items():
        selected = scene_arrays[sid]["task_interval_selected_rows"]
        assert all(counts_by_tick[sid][tick] == int(selected[tick]) for tick in range(len(selected)))
        local = [row for row in rows if row["scenario_id"] == sid]
        instantaneous = [row for row in local if row["endpoint_slack_m_s"] <= BINDING_TOLERANCE_M_S]
        lookahead = [row for row in local if row["lookahead_slack_m_s"] <= BINDING_TOLERANCE_M_S]
        scene.update({
            "independent_terminal_interval_rows": terminal_counts[sid],
            "instantaneous_binding_row_occurrences": len(instantaneous),
            "instantaneous_binding_unique_ticks": len({row["tick_zero_based"] for row in instantaneous}),
            "lookahead_binding_row_occurrences": len(lookahead),
            "lookahead_binding_unique_ticks": len({row["tick_zero_based"] for row in lookahead}),
            "minimum_interval_endpoint_slack_m_s": min(row["endpoint_slack_m_s"] for row in local),
            "minimum_interval_lookahead_slack_m_s": min(row["lookahead_slack_m_s"] for row in local),
        })
    summary = {
        "schema": "saved_research_interval_activity_baseline_v1",
        "evidence_scope": "Arithmetic analysis of existing accepted traces and independently recomputed interval rows; no new geometry query, physics or timing experiment",
        "reference_run": str(REFERENCE), "reference_commit": REFERENCE_COMMIT,
        "reference_manifest_path": str(manifest_path),
        "reference_manifest_sha256": REFERENCE_MANIFEST_SHA, "verified_reference_file_count": 46,
        "binding_rule": "slack <= 2e-5 m/s, matching original QP diagnostic; includes tolerated negative slacks and does not establish causal/dual activity",
        "binding_threshold_m_s": BINDING_TOLERANCE_M_S,
        "original_clearance_rate_tolerance_m_s": rate_tolerance,
        "pcc_barrier_gain_s_inv": gain, "task_period_s": dt,
        "ramp_mean_weights": {"old": old_weight, "new": new_weight},
        "lookahead_model_margin_m_s": lookahead_margin,
        "planning_period_count": sum(scene["planning_periods"] for scene in scenes.values()),
        "ticks_with_selected_interval_rows": sum(scene["ticks_with_selected_interval_rows"] for scene in scenes.values()),
        "selected_interval_row_occurrences": len(rows),
        "instantaneous_binding_row_occurrences": sum(scene["instantaneous_binding_row_occurrences"] for scene in scenes.values()),
        "lookahead_binding_row_occurrences": sum(scene["lookahead_binding_row_occurrences"] for scene in scenes.values()),
        "lookahead_binding_unique_ticks": sum(scene["lookahead_binding_unique_ticks"] for scene in scenes.values()),
        "closest_interval_boundary": min(rows, key=lambda row: row["h_m"]),
        "minimum_endpoint_slack_row": min(rows, key=lambda row: row["endpoint_slack_m_s"]),
        "minimum_lookahead_slack_row": min(rows, key=lambda row: row["lookahead_slack_m_s"]),
        "lookahead_rows_below_original_contract_tolerance": sum(row["lookahead_slack_m_s"] < -rate_tolerance for row in rows),
        "scenarios": scenes,
        "limitations": [
            "pcc_monitor legacy active/binding masks match pcc: sources, excluding pcc_interval: sources. Disabled legacy zero counts and nonfinite distance placeholders do not measure interval CBF activity.",
            "The optimized screened MuJoCo continuum-target field can return query-max 0.09 m as a supported lower bound. It is not an exact measured distance; this analysis does not use it for geometry attribution.",
            "Selected interval rows and near-binding slacks do not independently measure avoidance intervention or establish which constraint caused a trajectory change.",
            "No new closed-loop, robustness, wall deployment, hard realtime or hardware claim is added.",
        ],
    }
    assert summary["planning_period_count"] == 6750
    assert summary["ticks_with_selected_interval_rows"] == 4906
    assert summary["selected_interval_row_occurrences"] == 12350
    assert summary["instantaneous_binding_row_occurrences"] == 0
    assert summary["lookahead_binding_row_occurrences"] == 840
    assert summary["lookahead_binding_unique_ticks"] == 840
    assert summary["lookahead_rows_below_original_contract_tolerance"] == 0
    assert sha(manifest_path) == REFERENCE_MANIFEST_SHA
    assert all(sha(item["path"]) == item["sha256"] for item in references)
    sources = {"schema": "saved_research_interval_activity_source_manifest_v1",
               "analysis_path": str(Path(__file__).resolve()), "analysis_sha256": sha(__file__),
               "reference_manifest_sha256": REFERENCE_MANIFEST_SHA,
               "reference_artifacts": references, "frozen_baseline_code_sources": source_rows}
    return summary, sources


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    summary, sources = analyze()
    if args.verify_only:
        assert json.loads((OUTPUT / "summary.json").read_text(encoding="utf-8")) == summary
        assert json.loads((OUTPUT / "source_manifest.json").read_text(encoding="utf-8")) == sources
        manifest = json.loads((OUTPUT / "artifact_manifest.json").read_text(encoding="utf-8"))
        assert all(sha(OUTPUT / row["path"]) == row["sha256"] for row in manifest["artifacts"])
    else:
        write_exclusive("source_manifest.json", sources)
        write_exclusive("summary.json", summary)
        write_exclusive("artifact_manifest.json", {
            "schema": "saved_research_interval_activity_artifact_manifest_v1",
            "artifacts": [{"path": path.name, "bytes": path.stat().st_size, "sha256": sha(path)}
                          for path in sorted(OUTPUT.iterdir()) if path.is_file()],
        })
    print(json.dumps({"verified": True, "verify_only": args.verify_only,
                      "analysis_sha256": sources["analysis_sha256"],
                      "planning_periods": summary["planning_period_count"],
                      "selected_interval_rows": summary["selected_interval_row_occurrences"],
                      "ticks_with_selected_rows": summary["ticks_with_selected_interval_rows"],
                      "instantaneous_binding_rows": summary["instantaneous_binding_row_occurrences"],
                      "lookahead_binding_rows": summary["lookahead_binding_row_occurrences"],
                      "closest_interval_boundary": summary["closest_interval_boundary"],
                      "minimum_lookahead_slack_row": summary["minimum_lookahead_slack_row"]},
                     indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
