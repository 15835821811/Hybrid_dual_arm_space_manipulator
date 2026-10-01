"""Offline decision-budget frontier on the frozen A.1 shadow sample plan.

The five predeclared budgets are diagnostic only. The largest one is not an
online budget, and neither a proxy-below verdict nor an unknown verdict is a
claim about collision of the actual MuJoCo chain.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import mujoco
import numpy as np

from model_test.whole_body_verifier_v5 import (
    WholeBodyCollisionVerifier, WholeBodyVerificationConfig,
)
from v6_lite.continuum_shape_model import transform_from_free_qpos
from v6_lite.pcc_interval_cbf import FixedIntervalCBFEvaluator, IntervalPartition
from v6_lite.pcc_persistent_interval_query import PersistentIntervalDecisionQuery
from v6_lite.recompute_execution_constraints import _obstacles
from v6_lite.run_v6_lite import default_v6_lite_robot_spec
from v6_lite.shape_clearance import target_box_from_mujoco


BUDGETS = (31, 63, 127, 255, 511)
GATE_M = 0.005
DEFAULT_SHADOW = Path("v6_lite/output/v6_2_b2/shadow_warm/shadow_report.json")
DEFAULT_A1_ROOT = Path("v6_lite/output/v6_2_a1")


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write(path: Path, value: dict) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def _summary(values: list[float]) -> dict:
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": len(values),
        "p50": float(np.quantile(array, .5)),
        "p95": float(np.quantile(array, .95)),
        "p99": float(np.quantile(array, .99)),
        "max": float(np.max(array)),
    } if values else {"count": 0}


def _scenario_rows(mode: str, metrics: dict, warm: dict,
                   records_by_scenario: dict[str, list[dict]]) -> tuple[list[dict], list[dict]]:
    robot = default_v6_lite_robot_spec()
    entries = []
    sources = []
    for scenario_result in metrics["scenarios"]:
        scenario = scenario_result["scenario"]
        scenario_id = scenario["scenario_id"]
        records = records_by_scenario.pop(scenario_id, [])
        if len(records) != warm["sample_count"] // len(metrics["scenarios"]):
            raise ValueError(f"unexpected frozen sample count: {mode}/{scenario_id}")
        verifier = WholeBodyCollisionVerifier(
            robot, _obstacles(scenario),
            WholeBodyVerificationConfig(
                minimum_clearance=metrics["run_config"]["whole_body_minimum_clearance_m"],
                query_distance_max=2.5,
                adaptive_subdivisions=metrics["run_config"]["verification_subdivisions"],
                self_collision_ancestor_exclusion_depth=3,
                include_target_satellite_pairs=True,
            ),
        )
        model = verifier.model
        model.geom_contype[:] = 0
        model.geom_conaffinity[:] = 0
        data = mujoco.MjData(model)
        evaluator = FixedIntervalCBFEvaluator(robot, model)
        query = PersistentIntervalDecisionQuery(evaluator.shape_model)
        trace_path = Path(scenario_result["trace"]["path"])
        trace_sha = _sha(trace_path)
        warm_scenario = next(x for x in warm["scenarios"]
                             if x["scenario_id"] == scenario_id)
        if (trace_sha != scenario_result["trace"]["sha256"]
                or trace_sha != warm_scenario["trace_sha256"]):
            raise ValueError(f"A.1 trace hash mismatch: {mode}/{scenario_id}")
        sources.append({"path": trace_path.as_posix(), "sha256": trace_sha})
        with np.load(trace_path, allow_pickle=False) as trace:
            task_qpos = trace["task_qpos"].copy()
        for record in records:
            tick = int(record["tick"])
            if tick < 0 or tick >= len(task_qpos):
                raise ValueError(f"invalid task tick {mode}/{scenario_id}/{tick}")
            data.qpos[:] = task_qpos[tick]
            mujoco.mj_forward(model, data)
            low_level = data.qpos[evaluator.qpos_ids[:60]]
            projection = evaluator.shape_spec.project_actual_configuration(low_level)
            if (abs(projection.residual_linf_rad
                    - record["subspace_residual_linf_rad"]) > 1e-12):
                raise ValueError(f"projection changed: {mode}/{scenario_id}/{tick}")
            base = transform_from_free_qpos(data.qpos[evaluator.base_qpos_slice])
            box = target_box_from_mujoco(model, data, evaluator.target_geom_id)
            ladder = []
            for budget in BUDGETS:
                result = query.evaluate(
                    projection.planner_configuration, base, box,
                    IntervalPartition.uniform(), gate_m=GATE_M,
                    max_point_evaluations=budget, max_leaves=512,
                )
                if not (result.bounds_valid
                        and result.partition.coverage(
                            evaluator.shape_spec.segment_lengths_m
                        ).coverage_complete):
                    raise ValueError(f"invalid interval cover: {mode}/{scenario_id}/{tick}/{budget}")
                ladder.append({
                    "point_budget": budget,
                    "point_evaluations": result.point_evaluation_count,
                    "leaf_count": result.interval_count,
                    "lower_m": result.distance_lower_bound_m,
                    "upper_m": result.distance_upper_bound_m,
                    "proxy_status": result.proxy_clearance_status,
                    "budget_exhausted": result.budget_exhausted,
                    "elapsed_ms": result.elapsed_ms,
                })
            decisive = {item["proxy_status"] for item in ladder}
            if {"PROXY_CLEARANCE_BELOW_GATE", "PROXY_CLEARANCE_AT_LEAST_GATE"} <= decisive:
                raise ValueError(f"contradictory valid bounds: {mode}/{scenario_id}/{tick}")
            entry = {
                "mode": mode,
                "scenario_id": scenario_id,
                "tick": tick,
                "input_qpos_sha256": hashlib.sha256(task_qpos[tick].tobytes()).hexdigest(),
                "subspace_residual_linf_rad": projection.residual_linf_rad,
                "warm_proxy_status": record["proxy_status"],
                "warm_query_budget_acceptable": record["query_budget_acceptable"],
                "warm_start_status": record["feasibility"]["status"],
                "warm_worst_start_source": record["feasibility"].get("worst_start_source"),
                "budget_ladder": ladder,
            }
            entries.append(entry)
        print(f"[b2-budget] {mode} {scenario_id}: {len(records)} states", flush=True)
    if records_by_scenario:
        raise ValueError(f"unknown scenario IDs in frozen shadow: {sorted(records_by_scenario)}")
    return entries, sources


def _aggregate(entries: list[dict]) -> dict:
    result = {}
    for mode in ("baseline", "enabled"):
        own = [x for x in entries if x["mode"] == mode]
        start_bad = [x for x in own if x["warm_start_status"].startswith("START_")]
        statuses = {str(budget): dict(Counter(
            x["budget_ladder"][index]["proxy_status"] for x in own
        )) for index, budget in enumerate(BUDGETS)}
        high = Counter(x["budget_ladder"][-1]["proxy_status"] for x in start_bad)
        timings = {str(budget): _summary([
            x["budget_ladder"][index]["elapsed_ms"] for x in own
        ]) for index, budget in enumerate(BUDGETS)}
        result[mode] = {
            "sample_count": len(own),
            "warm_start_violation_count": len(start_bad),
            "status_by_budget": statuses,
            "warm_start_violations_by_511_status": dict(high),
            "offline_cold_query_time_ms_by_budget": timings,
        }
    return result


def run(output_dir: Path, *, shadow_path: Path = DEFAULT_SHADOW,
        a1_root: Path = DEFAULT_A1_ROOT) -> dict:
    output_dir.mkdir(parents=True, exist_ok=False)
    try:
        shadow = json.loads(shadow_path.read_text(encoding="utf-8"))
        if not shadow["passed_as_read_only_audit"] or shadow["query_mode"] != "persistent_warm":
            raise ValueError("the frozen shadow sample plan is not valid")
        entries = []
        trace_sources = []
        metrics_sources = []
        for mode in ("baseline", "enabled"):
            metrics_path = a1_root / f"{mode}_root" / "output" / "v6_lite_metrics.json"
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            if metrics["qp_config"]["pcc_clearance_safe_m"] != GATE_M:
                raise ValueError(f"A.1 safety gate changed: {mode}")
            warm = shadow["modes"][mode]
            if _sha(metrics_path) != warm["metrics_sha256"]:
                raise ValueError(f"A.1 metrics hash mismatch: {mode}")
            metrics_sources.append({"path": metrics_path.as_posix(),
                                    "sha256": _sha(metrics_path)})
            grouped = defaultdict(list)
            for record in warm["frozen_feasibility_records"]:
                grouped[record["scenario_id"]].append(record)
            part, sources = _scenario_rows(mode, metrics, warm, grouped)
            entries.extend(part)
            trace_sources.extend(sources)
        counts = _aggregate(entries)
        report = {
            "schema": "v6_2_b2_frozen_budget_frontier_v1",
            "budgets": list(BUDGETS),
            "gate_m": GATE_M,
            "sample_plan": "all 135 frozen shadow samples per A.1 mode; five scenes each",
            "scope": "offline cold decision query on saved task_qpos; not native replay or online timing",
            "floating_point_certification": "NOT_FORMALLY_CERTIFIED",
            "source_sha256": {
                "v6_lite/audit_b2_budget_frontier.py": _sha(Path(__file__)),
                "v6_lite/pcc_persistent_interval_query.py": _sha(Path(__file__).with_name("pcc_persistent_interval_query.py")),
                "v6_lite/continuum_shape_model.py": _sha(Path(__file__).with_name("continuum_shape_model.py")),
                "v6_lite/pcc_interval_cbf.py": _sha(Path(__file__).with_name("pcc_interval_cbf.py")),
            },
            "input_shadow": {"path": shadow_path.as_posix(), "sha256": _sha(shadow_path)},
            "input_metrics": metrics_sources,
            "input_traces": trace_sources,
            "modes": counts,
            "entries": entries,
        }
        report_path = output_dir / "budget_frontier.json"
        _write(report_path, report)
        lines = [
            "# V6.2-B.2 冻结闭环状态的离线查询预算阶梯", "",
            "输入是已哈希核对的 A.1 trace 中与正式影子审计相同的 270 个规划状态。"
            "本程序只读取保存的 `task_qpos`；原生力矩重放一致性由影子审计另行检查。", "",
            "| 模式 | 预算 31 未知 | 预算 63 未知 | 预算 127 未知 | 预算 255 未知 | 预算 511 未知 | 起点违例在 511 时确定低于 5 mm | 起点违例在 511 时确定达到 5 mm |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        for mode in ("baseline", "enabled"):
            item = counts[mode]
            unknown = [item["status_by_budget"][str(b)].get("UNKNOWN_CROSSES_GATE", 0)
                       for b in BUDGETS]
            high = item["warm_start_violations_by_511_status"]
            lines.append(
                f"| {mode} | {' | '.join(map(str, unknown))} | "
                f"{high.get('PROXY_CLEARANCE_BELOW_GATE', 0)} | "
                f"{high.get('PROXY_CLEARANCE_AT_LEAST_GATE', 0)} |"
            )
        lines += [
            "", "各预算均从完整五段根区间重新开始，没有缓存、放宽半径或改变 5 mm 门槛。"
            "阶梯仅诊断不确定性的来源；预算 511 的耗时和分类不得用作在线准入。"
            "代理低于门槛不等于实际链碰撞。所有逐状态上下界、分区大小与耗时见 JSON。", "",
        ]
        doc_path = output_dir / "BUDGET_FRONTIER.md"
        with doc_path.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write("\n".join(lines))
        manifest = {"schema": "v6_2_b2_budget_frontier_manifest_v1",
                    "artifacts": [{"path": p.name, "sha256": _sha(p),
                                   "bytes": p.stat().st_size}
                                  for p in (report_path, doc_path)]}
        _write(output_dir / "budget_frontier_manifest.json", manifest)
        return report
    except Exception as error:
        _write(output_dir / "FAILURE.json", {
            "schema": "v6_2_b2_budget_frontier_failure_v1",
            "error_type": type(error).__name__,
            "error": str(error),
            "input_shadow": shadow_path.as_posix(),
        })
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--shadow", type=Path, default=DEFAULT_SHADOW)
    parser.add_argument("--a1-root", type=Path, default=DEFAULT_A1_ROOT)
    args = parser.parse_args()
    report = run(args.output_dir, shadow_path=args.shadow, a1_root=args.a1_root)
    print(json.dumps({mode: report["modes"][mode] for mode in ("baseline", "enabled")},
                     indent=2))


if __name__ == "__main__":
    main()
