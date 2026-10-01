"""Predeclared three-round, five-scene private preflight-plus-QP timing."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from v6_lite.audit_b2_optimized_private_rollout import run as run_scene


ORDERS = ((0, 1, 2, 3, 4), (4, 3, 2, 1, 0), (0, 1, 2, 3, 4))
TICKS = 400


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def _write(path: Path, value: dict) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def run(output_dir: Path, a1_root: Path, baseline_root: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=False)
    protocol = {
        "schema": "v6_2_b2_optimized_private_repeat_protocol_v1",
        "round_scene_order": [[f"v6_lite_scenario_{i:02d}" for i in order]
                              for order in ORDERS],
        "ticks_per_scene": TICKS,
        "timing_scope": "private_preflight_plus_qp_only",
        "metric": "per_scene_per_round_numpy_p95_ms",
        "diagnostic_budget_ms": 20.0,
        "gate_rule": "all_15_per_scene_round_p95_values_at_most_20ms",
        "also_report": ["p99_ms", "max_ms", "over_20ms_count",
                        "longest_over_20ms_run"],
        "failed_runs_retained": True,
        "production_online_controller_changed": False,
        "full_cycle_20ms_acceptance": False,
        "continuous_time_certified": False,
        "source_sha256": {
            name: _source_sha(Path("v6_lite") / name)
            for name in ("audit_b2_optimized_repeat_timing.py",
                         "audit_b2_optimized_private_rollout.py",
                         "audit_b2_prepared_screened_private_rollout.py",
                         "b2_optimized_admm.py")
        },
    }
    protocol_path = output_dir / "repeat_protocol.json"
    _write(protocol_path, protocol)
    runs = []
    for round_index, order in enumerate(ORDERS, start=1):
        for scene_index in order:
            scene_id = f"v6_lite_scenario_{scene_index:02d}"
            folder = output_dir / f"round_{round_index:02d}" / f"scene_{scene_index:02d}"
            report = run_scene(folder, a1_root, scene_id, TICKS)
            baseline_trace = (baseline_root / f"scene_{scene_index:02d}" /
                              "private_rollout_trace.npz")
            if (report["executed_ticks"] != TICKS
                    or report["stop_reason"] != "HORIZON_COMPLETE"
                    or not report["strict_online_domain_all_executed_ticks"]
                    or report["trace_sha256"] != _sha(baseline_trace)
                    or report["production_online_controller_changed"]
                    or report["stage3_admission"]):
                raise ValueError(f"private repeat integrity failed: {round_index} {scene_id}")
            summary_path = folder / "private_rollout_summary.json"
            manifest_path = folder / "private_rollout_manifest.json"
            runs.append({
                "round": round_index,
                "scenario_id": scene_id,
                "summary_sha256": _sha(summary_path),
                "manifest_sha256": _sha(manifest_path),
                "trace_sha256": report["trace_sha256"],
                "baseline_trace_sha256": _sha(baseline_trace),
                "preflight_plus_qp_timing": report["private_preflight_plus_qp_timing"],
                "minimum_ramp_envelope_margin_m": report["minimum_ramp_envelope_margin_m"],
                "minimum_realized_next_start_slack_m_s":
                    report["minimum_realized_next_start_slack_m_s"],
            })
            timing = report["private_preflight_plus_qp_timing"]
            print(f"[b2-repeat] round={round_index} {scene_id} "
                  f"p95={timing['p95_ms']:.3f}ms over20={timing['over_20ms_count']}",
                  flush=True)
    failed = [{"round": item["round"], "scenario_id": item["scenario_id"],
               "p95_ms": item["preflight_plus_qp_timing"]["p95_ms"]}
              for item in runs if item["preflight_plus_qp_timing"]["p95_ms"] > 20.0]
    report = {
        "schema": "v6_2_b2_optimized_private_repeat_summary_v1",
        "status": ("PRIVATE_REPEAT_TIMING_GATE_NOT_MET" if failed else
                   "PRIVATE_REPEAT_TIMING_DIAGNOSTIC_PASS_NOT_ONLINE"),
        "protocol_sha256": _sha(protocol_path),
        "run_count": len(runs),
        "round_count": len(ORDERS),
        "scene_count": 5,
        "ticks_per_scene": TICKS,
        "timing_scope": "private_preflight_plus_qp_only",
        "p95_over_20ms_run_count": len(failed),
        "p95_over_20ms_runs": failed,
        "runs": runs,
        "production_online_controller_changed": False,
        "stage3_admission": False,
        "full_cycle_20ms_acceptance": False,
        "continuous_time_certified": False,
    }
    summary_path = output_dir / "repeat_summary.json"
    _write(summary_path, report)
    lines = ["# B.2 私有预检＋QP 三轮重复计时", "",
             "预声明三轮，各轮五场景、每场景 400 个规划周期。"
             "第二轮逆序，其他轮正序；全部完成后才计算总结果。"
             "每轮保留完整私有力矩轨迹、诊断、p95、p99、最大值、"
             "超 20 ms 周期和最长连续超时。", "",
             "| 轮次 | 场景 | p95 ms | p99 ms | 最大 ms | 超 20 ms | 最长连续超时 |",
             "| ---: | --- | ---: | ---: | ---: | ---: | ---: |"]
    for item in runs:
        timing = item["preflight_plus_qp_timing"]
        lines.append(f"| {item['round']} | {item['scenario_id'][-2:]} | "
                     f"{timing['p95_ms']:.3f} | {timing['p99_ms']:.3f} | "
                     f"{timing['max_ms']:.3f} | {timing['over_20ms_count']} | "
                     f"{timing['longest_over_20ms_run']} |")
    lines += ["", f"p95 超预算的场景轮次 {len(failed)} / {len(runs)}；"
              f"状态和力矩轨迹与已独立重放的基线 trace 哈希逐项相同。",
              "", "计时仅覆盖私有路径的预检与 QP，Windows 上的统计测试不是"
              "每周期硬实时证明；未完成正式全链五场景或生产在线切换。", ""]
    document_path = output_dir / "REPEAT_TIMING.md"
    document_path.write_text("\n".join(lines), encoding="utf-8", newline="\n")
    _write(output_dir / "repeat_manifest.json", {
        "schema": "v6_2_b2_optimized_private_repeat_manifest_v1",
        "protocol_sha256": _sha(protocol_path),
        "summary_sha256": _sha(summary_path),
        "document_sha256": _sha(document_path),
    })
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--a1-root", type=Path,
                        default=Path("v6_lite/output/v6_2_a1"))
    parser.add_argument("--baseline-root", type=Path,
                        default=Path("v6_lite/output/v6_2_b2/discrete_private_rollout_400"))
    args = parser.parse_args()
    result = run(args.output_dir, args.a1_root, args.baseline_root)
    print(json.dumps({key: result[key] for key in
                      ("status", "run_count", "p95_over_20ms_run_count")},
                     indent=2))


if __name__ == "__main__":
    main()
