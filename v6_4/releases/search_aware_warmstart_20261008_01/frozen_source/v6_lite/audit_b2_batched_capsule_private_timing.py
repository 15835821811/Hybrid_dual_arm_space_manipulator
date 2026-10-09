"""Trial batched capsule screening inside the unchanged private B.2 rollout."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from v6_lite import hierarchical_qp as qp_module
from v6_lite import recompute_execution_constraints as replay_module
from v6_lite.audit_b2_compact_envelope_timing import run as compact_run
from v6_lite.b2_batched_capsule_bounds import BatchedCapsuleMidpointBounds


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def run(output_dir: Path, reference_dir: Path, screened_dir: Path,
        fixed_dir: Path, baseline_dir: Path, a1_root: Path,
        scenario_id: str, ticks: int) -> dict:
    original_qp = qp_module.minimum_capsule_clearance
    original_replay = replay_module.minimum_capsule_clearance
    screens = {}
    counters = {"batched_calls": 0, "gradient_fallback_calls": 0}

    def batched_minimum(model, data, envelopes, box, *, spec=None,
                        compute_planner_gradient=True):
        if compute_planner_gradient:
            counters["gradient_fallback_calls"] += 1
            return original_qp(model, data, envelopes, box, spec=spec,
                               compute_planner_gradient=True)
        key = id(envelopes)
        cached = screens.get(key)
        if cached is None or cached[0] is not envelopes:
            cached = (envelopes, BatchedCapsuleMidpointBounds(envelopes))
            screens[key] = cached
        counters["batched_calls"] += 1
        return cached[1].minimum(model, data, box)

    try:
        qp_module.minimum_capsule_clearance = batched_minimum
        replay_module.minimum_capsule_clearance = batched_minimum
        trial = compact_run(output_dir, reference_dir, screened_dir,
                            fixed_dir, baseline_dir, a1_root,
                            scenario_id, ticks)
    finally:
        qp_module.minimum_capsule_clearance = original_qp
        replay_module.minimum_capsule_clearance = original_replay
    if (counters["batched_calls"] < 2 * ticks
            or not trial["all_non_timing_records_exact"]):
        raise ValueError("batched capsule trial did not cover QP and next start")
    old = json.loads((reference_dir / "compact_envelope_timing_summary.json")
                     .read_text(encoding="utf-8"))
    ignored_timings = {"qp_full_ms", "preflight_plus_qp_ms",
                       "independent_row_recompute_ms"}
    new_records = [json.loads(line) for line in (
        output_dir / "private_rollout_records.jsonl").read_text(
            encoding="utf-8").splitlines()]
    reference_records = [json.loads(line) for line in (
        reference_dir / "private_rollout_records.jsonl").read_text(
            encoding="utf-8").splitlines()]
    if len(new_records) != ticks or len(reference_records) != ticks:
        raise ValueError("batched capsule trial lacks full tick coverage")
    mismatch = next((tick for tick, (row, old_row) in enumerate(zip(
        new_records, reference_records)) if {
            key: value for key, value in row.items()
            if key not in ignored_timings} != {
            key: value for key, value in old_row.items()
            if key not in ignored_timings}), None)
    if mismatch is not None:
        raise ValueError(f"batched capsule changed tick {mismatch}")
    report = {
        "schema": "v6_2_b2_batched_capsule_private_timing_v1",
        "scenario_id": scenario_id,
        "task_ticks": ticks,
        "all_non_timing_records_exact": True,
        "same_native_67_channel_torque_trace": True,
        "batched_calls": counters["batched_calls"],
        "gradient_fallback_calls": counters["gradient_fallback_calls"],
        "cached_envelope_sets": len(screens),
        "reference_necessary_component_sum": old["trial_necessary_component_sum"],
        "trial_necessary_component_sum": trial["trial_necessary_component_sum"],
        "reference_ten_step_branch": old["trial_ten_step_branch"],
        "trial_ten_step_branch": trial["trial_ten_step_branch"],
        "full_online_cycle_evaluated": False,
        "online_20ms_admitted": False,
        "production_online_controller_changed": False,
        "source_sha256": {name: _source_sha(Path("v6_lite") / name)
                          for name in (
                              "audit_b2_batched_capsule_private_timing.py",
                              "audit_b2_compact_envelope_timing.py",
                              "b2_batched_capsule_bounds.py",
                              "shape_clearance.py")},
        "inputs_sha256": {
            "reference_records": _sha(
                reference_dir / "private_rollout_records.jsonl"),
            "reference_trace": _sha(
                reference_dir / "private_rollout_trace.npz"),
            "trial_records": _sha(output_dir / "private_rollout_records.jsonl"),
            "trial_trace": _sha(output_dir / "private_rollout_trace.npz"),
        },
    }
    (output_dir / "batched_capsule_private_timing_summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2,
                   allow_nan=False) + "\n", encoding="utf-8", newline="\n")
    lines = ["# B.2 私有全周期批量胶囊筛选试验", "",
             "61 条原胶囊、原精确线段—OBB 距离、原阈值和全部 500 Hz 包络状态仍在。"
             "仅批量计算胶囊中点的筛选下界，1 pm 向下保守修正只影响候选排序。"
             "1350 周期的非计时记录和 67 路力矩 trace 与参考逐值相同。", "",
             "| 指标 | 参考 p95 ms | 试验 p95 ms | 试验 p99 ms | 最大 ms |",
             "| --- | ---: | ---: | ---: | ---: |"]
    for label, old_key, new_key in (
            ("十步候选分支", "reference_ten_step_branch", "trial_ten_step_branch"),
            ("必要计算和", "reference_necessary_component_sum",
             "trial_necessary_component_sum")):
        old_item, new_item = report[old_key], report[new_key]
        lines.append(f"| {label} | {old_item['p95_ms']:.3f} | "
                     f"{new_item['p95_ms']:.3f} | {new_item['p99_ms']:.3f} | "
                     f"{new_item['max_ms']:.3f} |")
    lines += ["", "这仍是私有必要计算的计时，不是完整生产在线周期。"
              "阶段二门槛未达到，生产控制器不变。", ""]
    (output_dir / "BATCHED_CAPSULE_PRIVATE_TIMING.md").write_text(
        "\n".join(lines), encoding="utf-8", newline="\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--screened-dir", type=Path, required=True)
    parser.add_argument("--fixed-dir", type=Path, required=True)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--a1-root", type=Path,
                        default=Path("v6_lite/output/v6_2_a1"))
    parser.add_argument("--scenario-id", default="v6_lite_scenario_01")
    parser.add_argument("--ticks", type=int, default=1350)
    args = parser.parse_args()
    report = run(args.output_dir, args.reference_dir, args.screened_dir,
                 args.fixed_dir, args.baseline_dir, args.a1_root,
                 args.scenario_id, args.ticks)
    print(json.dumps({key: report[key] for key in (
        "all_non_timing_records_exact", "batched_calls",
        "reference_necessary_component_sum", "trial_necessary_component_sum")},
        indent=2))


if __name__ == "__main__":
    main()
