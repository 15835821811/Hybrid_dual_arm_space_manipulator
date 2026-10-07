"""Compare all 61 capsule results at every saved 2 ms private state."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np

from model_test.whole_body_verifier_v5 import (
    WholeBodyCollisionVerifier, WholeBodyVerificationConfig,
)
from v6_lite.continuum_shape_model import transform_from_free_qpos
from v6_lite.pcc_fixed_arc_state_envelope import (
    FixedArcStateLocalPCCEnvelopeAudit,
)
from v6_lite.pcc_interval_cbf import FixedIntervalCBFEvaluator
from v6_lite.pcc_vectorized_state_envelope import (
    VectorizedStateLocalPCCEnvelopeAudit,
)
from v6_lite.recompute_execution_constraints import _obstacles
from v6_lite.run_v6_lite import V6LiteRunConfig, default_v6_lite_robot_spec


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def run(output_dir: Path, trace_dir: Path, a1_root: Path,
        scenario_id: str) -> dict:
    output_dir.mkdir(parents=True, exist_ok=False)
    metrics_path = a1_root / "enabled_root/output/v6_lite_metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    saved = next(item for item in metrics["scenarios"]
                 if item["scenario"]["scenario_id"] == scenario_id)
    run_cfg = V6LiteRunConfig(**metrics["run_config"])
    robot = default_v6_lite_robot_spec()
    verifier = WholeBodyCollisionVerifier(
        robot, _obstacles(saved["scenario"]), WholeBodyVerificationConfig(
            minimum_clearance=run_cfg.whole_body_minimum_clearance_m,
            query_distance_max=2.5,
            adaptive_subdivisions=run_cfg.verification_subdivisions,
            self_collision_ancestor_exclusion_depth=3,
            include_target_satellite_pairs=True,
        ),
    )
    model = verifier.model
    evaluator = FixedIntervalCBFEvaluator(robot, model)
    original = VectorizedStateLocalPCCEnvelopeAudit(
        model, evaluator.shape_spec)
    fixed = FixedArcStateLocalPCCEnvelopeAudit(model, evaluator.shape_spec)
    data = mujoco.MjData(model)
    trace_path = trace_dir / "private_rollout_trace.npz"
    summary_path = trace_dir / "private_rollout_summary.json"
    trace_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if trace_summary["trace_sha256"] != _sha(trace_path):
        raise ValueError("private trace hash changed")
    with np.load(trace_path, allow_pickle=False) as trace:
        qpos_states = trace["qpos_states"].copy()
    expected_states = 10 * trace_summary["executed_ticks"] + 1
    if len(qpos_states) != expected_states:
        raise ValueError("private 2 ms state count changed")
    rows_path = output_dir / "fixed_arc_all_state_rows.jsonl"
    failures_path = output_dir / "fixed_arc_all_state_failures.jsonl"
    mismatches = 0
    first_mismatch = None
    capsule_count = len(original.envelopes.capsules)
    with rows_path.open("x", encoding="utf-8", newline="\n") as rows_out, \
            failures_path.open("x", encoding="utf-8", newline="\n") as fail_out:
        for index, qpos in enumerate(qpos_states):
            data.qpos[:] = qpos
            mujoco.mj_forward(model, data)
            projection = evaluator.shape_spec.project_actual_configuration(
                data.qpos[evaluator.qpos_ids[:60]])
            base = transform_from_free_qpos(
                data.qpos[evaluator.base_qpos_slice])
            a = original.evaluate(data, projection.planner_configuration, base)
            b = fixed.evaluate(data, projection.planner_configuration, base)
            exact = a == b
            row = {"state_index": index, "time_s": index * 0.002,
                   "capsule_count": capsule_count,
                   "status": b.status, "minimum_margin_m": b.min_margin_m,
                   "complete_capsule_result_exact": exact}
            rows_out.write(json.dumps(row, separators=(",", ":"),
                                      allow_nan=False) + "\n")
            if not exact:
                mismatches += 1
                if first_mismatch is None:
                    first_mismatch = index
                fail_out.write(json.dumps(row, separators=(",", ":")) + "\n")
    report = {
        "schema": "v6_2_b2_fixed_arc_envelope_all_states_parity_v1",
        "scenario_id": scenario_id,
        "checked_2ms_states": len(qpos_states),
        "capsules_per_state": capsule_count,
        "complete_capsule_result_mismatches": mismatches,
        "first_mismatch_state": first_mismatch,
        "all_results_exact": mismatches == 0,
        "production_online_controller_changed": False,
        "continuous_time_certified": False,
        "source_sha256": {name: _source_sha(Path("v6_lite") / name)
                          for name in (
                              "audit_b2_fixed_arc_envelope_all_states.py",
                              "pcc_fixed_arc_positions.py",
                              "pcc_fixed_arc_state_envelope.py",
                              "pcc_vectorized_state_envelope.py")},
        "inputs_sha256": {
            "metrics": _sha(metrics_path),
            "private_summary": _sha(summary_path),
            "private_trace": _sha(trace_path),
        },
        "rows_sha256": _sha(rows_path),
        "failures_sha256": _sha(failures_path),
    }
    (output_dir / "fixed_arc_all_state_summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8", newline="\n")
    (output_dir / "FIXED_ARC_ALL_STATES.md").write_text(
        "# B.2 固定弧长实际状态全胶囊逐值核对\n\n"
        f"场景 {scenario_id} 的 {len(qpos_states)} 个保存的 2 ms 状态，"
        f"每状态 {capsule_count} 个实际链胶囊；"
        f"完整结果不一致数 {mismatches}。\n\n"
        "本检查只比较同一 MuJoCo 模型上的两种计算路径。"
        "它不是连续时间或真实机器人保证。\n",
        encoding="utf-8", newline="\n")
    if mismatches:
        raise ValueError(f"fixed-arc result changed at {first_mismatch}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--trace-dir", type=Path, required=True)
    parser.add_argument("--a1-root", type=Path,
                        default=Path("v6_lite/output/v6_2_a1"))
    parser.add_argument("--scenario-id", default="v6_lite_scenario_01")
    args = parser.parse_args()
    print(json.dumps(run(args.output_dir, args.trace_dir, args.a1_root,
                         args.scenario_id), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
