"""ABBA read-only weighted-QP trial with optional MuJoCo sphere screening.

The original 17-D QP, interval rows, actual-chain capsule rows and action
validator remain unchanged. A derived probe class only reduces exact distance
calls for conservatively screened original MuJoCo pairs. All candidates and
assembled rows are checked against the independently recomputed reference.
No command is executed and no online controller path selects this class.
"""

from __future__ import annotations

import argparse
from contextlib import redirect_stdout
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import time
import traceback

import numpy as np

import v6_lite.audit_b2_weighted_qp_probe as probe_module
from v6_lite.audit_b2_mujoco_sphere_screen import SPHERE_SCREEN_PAD_M


GROUPS = (("a1", "reference"), ("b1", "sphere_screen"),
          ("b2", "sphere_screen"), ("a2", "reference"))
TICKS = (50, 100, 150)
POINT_BUDGET = 63
PERIOD_MS = 20.0
_TELEMETRY: list[dict] = []


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def _stats(values: list[float]) -> dict:
    data = np.asarray(values, dtype=np.float64)
    if not len(data):
        raise ValueError("empty timing group")
    return {"count": len(data), "min": float(np.min(data)),
            "p50": float(np.percentile(data, 50)),
            "p95": float(np.percentile(data, 95)),
            "p99": float(np.percentile(data, 99)),
            "max": float(np.max(data)),
            "over_20ms_count": int(np.count_nonzero(data > PERIOD_MS))}


class _SphereScreenIntervalQP(probe_module._ReadOnlyIntervalQP):
    """Read-only probe subclass; no production QP uses this route."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._sphere_pair_a = np.asarray(
            [pair.geom_a for pair in self.collision_pairs], dtype=np.int32)
        self._sphere_pair_b = np.asarray(
            [pair.geom_b for pair in self.collision_pairs], dtype=np.int32)

    def _mujoco_clearance_constraint_set(self, data, generalized_map):
        model = self.model
        a, b = self._sphere_pair_a, self._sphere_pair_b
        radius_a = np.asarray(model.geom_rbound[a], dtype=np.float64)
        radius_b = np.asarray(model.geom_rbound[b], dtype=np.float64)
        center_a = np.asarray(data.geom_xpos[a], dtype=np.float64)
        center_b = np.asarray(data.geom_xpos[b], dtype=np.float64)
        finite = (np.isfinite(radius_a) & np.isfinite(radius_b)
                  & (radius_a > 0) & (radius_b > 0)
                  & np.all(np.isfinite(center_a), axis=1)
                  & np.all(np.isfinite(center_b), axis=1))
        lower = np.linalg.norm(center_a - center_b, axis=1) - radius_a - radius_b
        far = finite & (lower > self.config.clearance_query_max_m
                        + SPHERE_SCREEN_PAD_M)
        original_pairs = self.collision_pairs
        retained = tuple(pair for index, pair in enumerate(original_pairs)
                         if not far[index])
        self.collision_pairs = retained
        try:
            block = super()._mujoco_clearance_constraint_set(data, generalized_map)
        finally:
            self.collision_pairs = original_pairs
        fallback = (not retained
                    or block.minimum_clearance_m >= self.config.clearance_query_max_m)
        if fallback:
            # Exact global diagnostic minimum is not known from the retained
            # pairs if none is strictly inside the distance query radius.
            block = super()._mujoco_clearance_constraint_set(data, generalized_map)
        elif not any(pair.pair_class == "continuum_target" for pair in retained):
            # The reference caps each target pair to query_max_m before
            # selecting its diagnostic minimum. With all target pairs far,
            # that minimum is query_max_m and has no valid gradient.
            block = replace(
                block,
                mujoco_continuum_target_distance_m=self.config.clearance_query_max_m,
                mujoco_continuum_target_gradient=np.zeros(17),
                mujoco_continuum_target_gradient_valid=False,
            )
        _TELEMETRY.append({
            "pair_count": len(original_pairs),
            "sphere_far_count": int(np.count_nonzero(far)),
            "exact_call_count": len(retained) + (len(original_pairs) if fallback else 0),
            "fallback_to_all_pairs": bool(fallback),
        })
        return block


def _candidate_parity(current: dict, original: dict) -> tuple[int, float]:
    reference = {(row["mode"], row["scenario_id"], row["tick"]): row
                 for row in original["records"]}
    mismatches = 0
    largest_candidate_error = 0.0
    if len(reference) != len(current["records"]):
        raise ValueError("read-only probe record count changed")
    for row in current["records"]:
        key = row["mode"], row["scenario_id"], row["tick"]
        if key not in reference:
            raise ValueError(f"unexpected frozen state {key}")
        expected = reference[key]
        selected = row["selected_command"]
        wanted = expected["selected_command"]
        selected_match = (selected is None and wanted is None) or (
            selected is not None and wanted is not None
            and np.allclose(selected, wanted, rtol=0.0, atol=1e-8))
        candidate_error = float(np.max(np.abs(np.asarray(row["solver_candidate"])
                                               - np.asarray(expected["solver_candidate"]))))
        largest_candidate_error = max(largest_candidate_error, candidate_error)
        stable = all(row[field] == expected[field] for field in (
            "query_status", "query_points", "selected_interval_count",
            "proxy_safe", "envelope_supported", "solver_status",
            "solver_iterations", "action_mode", "failure_reason",
            "candidate_valid", "ramp_valid",
        ))
        row_consistent = all(row["row_parity"][field] <= 1e-7
                             for field in ("matrix_max_abs_error",
                                           "lower_max_abs_error",
                                           "drift_max_abs_error",
                                           "gain_max_abs_error"))
        mismatches += not (stable and selected_match and candidate_error <= 1e-8
                           and row_consistent)
    return mismatches, largest_candidate_error


def run(output_dir: Path, a1_root: Path, original_probe_path: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=False)
    original = json.loads(original_probe_path.read_text(encoding="utf-8"))
    if (not original["passed_as_read_only_integrity"]
            or original["probe_ticks"] != list(TICKS)
            or original["point_budget"] != POINT_BUDGET
            or len(original["records"]) != 30):
        raise ValueError("original frozen QP probe protocol changed")
    summary = {
        "schema": "v6_2_b2_weighted_qp_sphere_screen_abba_v1",
        "scope": "read_only_qp_on_30_native_torque_replayed_old_A1_states_per_group",
        "groups_predeclared": [{"name": name, "method": method}
                               for name, method in GROUPS],
        "probe_ticks": list(TICKS), "point_budget": POINT_BUDGET,
        "period_reference_ms": PERIOD_MS,
        "sphere_screen_pad_m": SPHERE_SCREEN_PAD_M,
        "input_original_probe_sha256": _sha(original_probe_path),
        "source_hash_newline_policy": "LF_NORMALIZED",
        "source_sha256": {name: _source_sha(Path("v6_lite") / name) for name in (
            "audit_b2_weighted_qp_sphere_trial.py",
            "audit_b2_weighted_qp_probe.py",
            "audit_b2_mujoco_sphere_screen.py",
            "hierarchical_qp.py", "safety_contract.py",
        )},
        "online_controller_changed": False,
        "new_interval_mode_executed": False,
        "full_control_cycle_measured": False,
        "groups": [], "modes": {},
    }
    records = []
    original_class = probe_module._ReadOnlyIntervalQP
    for name, method in GROUPS:
        folder = output_dir / name
        stdout_path = output_dir / f"{name}_stdout.txt"
        group = {"name": name, "method": method, "status": "ERROR",
                 "record_count": 0}
        _TELEMETRY.clear()
        probe_module._ReadOnlyIntervalQP = (
            original_class if method == "reference" else _SphereScreenIntervalQP)
        started = time.perf_counter()
        try:
            with stdout_path.open("x", encoding="utf-8", newline="\n") as stream:
                with redirect_stdout(stream):
                    current = probe_module.run(folder, a1_root,
                                               probe_ticks=TICKS,
                                               point_budget=POINT_BUDGET)
            group["stdout_sha256"] = _sha(stdout_path)
            if (not current["passed_as_read_only_integrity"]
                    or current["inputs"] != original["inputs"]):
                raise ValueError("read-only QP replay or independent rows changed")
            mismatches, largest_error = _candidate_parity(current, original)
            if method == "sphere_screen" and len(_TELEMETRY) != 30:
                raise ValueError("sphere-screen telemetry does not cover all probes")
            for index, row in enumerate(current["records"]):
                telemetry = (_TELEMETRY[index] if method == "sphere_screen"
                             else {"pair_count": 2927, "sphere_far_count": 0,
                                   "exact_call_count": 2927,
                                   "fallback_to_all_pairs": False})
                records.append({
                    "group": name, "method": method,
                    "mode": row["mode"], "scenario_id": row["scenario_id"],
                    "tick": row["tick"],
                    "query_plus_qp_probe_ms": row["query_plus_qp_probe_ms"],
                    "qp_full_ms": row["qp_full_ms"],
                    "qp_solver_ms": row["qp_solver_ms"],
                    **telemetry,
                })
            group.update({
                "status": "COMPLETE", "record_count": 30,
                "candidate_mismatch_count": mismatches,
                "maximum_candidate_error": largest_error,
                "report_sha256": _sha(folder / "weighted_qp_probe.json"),
                "document_sha256": _sha(folder / "WEIGHTED_QP_PROBE.md"),
                "manifest_sha256": _sha(folder / "weighted_qp_probe_manifest.json"),
            })
        except Exception as exc:
            group["exception"] = f"{type(exc).__name__}: {exc}"
            group["traceback"] = traceback.format_exc()
            if stdout_path.exists():
                group["stdout_sha256"] = _sha(stdout_path)
        finally:
            probe_module._ReadOnlyIntervalQP = original_class
            group["wall_seconds"] = time.perf_counter() - started
            summary["groups"].append(group)
        print(f"[b2-qp-sphere] {name} {method}: {group['status']}, "
              f"candidates={group['record_count']}", flush=True)
    for mode in ("baseline", "enabled"):
        summary["modes"][mode] = {}
        for method in ("reference", "sphere_screen"):
            own = [item for item in records if item["mode"] == mode
                   and item["method"] == method]
            if not own:
                continue
            summary["modes"][mode][method] = {
                "record_count": len(own),
                "query_plus_qp_probe_ms": _stats([
                    item["query_plus_qp_probe_ms"] for item in own]),
                "qp_full_ms": _stats([item["qp_full_ms"] for item in own]),
                "qp_solver_ms": _stats([item["qp_solver_ms"] for item in own]),
                "exact_call_count": _stats([item["exact_call_count"] for item in own]),
                "fallback_count": sum(item["fallback_to_all_pairs"] for item in own),
            }
    summary["complete_group_count"] = sum(
        item["status"] == "COMPLETE" for item in summary["groups"])
    summary["candidate_mismatch_count"] = sum(
        item.get("candidate_mismatch_count", 0) for item in summary["groups"])
    records_path = output_dir / "qp_sphere_records.jsonl"
    with records_path.open("x", encoding="utf-8", newline="\n") as stream:
        for item in records:
            stream.write(json.dumps(item, ensure_ascii=False,
                                    separators=(",", ":"), allow_nan=False) + "\n")
    summary["records_sha256"] = _sha(records_path)
    summary_path = output_dir / "qp_sphere_summary.json"
    with summary_path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(summary, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    lines = [
        "# B.2 原加权 QP 的 MuJoCo 包围球筛选只读交叉试验", "",
        "固定全对→筛选→筛选→全对四组，每组在旧 A.1 原生力矩重放的"
        "两组五场景 tick 50/100/150 上重新构建相同单个 17 维加权 QP；"
        "每个冻结状态都清空对偶热启动。筛选只作用于原 MuJoCo 距离查询，"
        "PCC 区间行、实际链胶囊行和原动作验证仍由同一实现处理。", "",
        "| 模式 | 方法 | 记录 | 查询加 QP p95 / p99 / 最大 ms | 超 20 ms | MuJoCo 精确调用 p95 |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for mode in ("baseline", "enabled"):
        for method in ("reference", "sphere_screen"):
            item = summary["modes"][mode].get(method)
            if not item:
                continue
            timing = item["query_plus_qp_probe_ms"]
            lines.append(
                f"| {mode} | {method} | {item['record_count']} | "
                f"{timing['p95']:.3f} / {timing['p99']:.3f} / "
                f"{timing['max']:.3f} | {timing['over_20ms_count']} | "
                f"{item['exact_call_count']['p95']:.1f} |"
            )
    lines += [
        "", f"完整组 {summary['complete_group_count']}/4，"
        f"候选不一致 {summary['candidate_mismatch_count']}。"
        "每组完整探针、逐状态计时与哈希保留，失败不覆盖。"
        "逐状态还用独立的全对 MuJoCo 行重算检查 QP 约束；"
        "筛选结果若缺少近距见证，会回退完整距离查询。",
        "这是只读部分链路计时。没有发出候选命令、运行新区间闭环，"
        "也没有计入全部状态监控和 67 路力矩伺服，不能作为全链 20 ms "
        "或在线安全验收。", "",
    ]
    document_path = output_dir / "QP_SPHERE_TRIAL.md"
    document_path.write_text("\n".join(lines), encoding="utf-8", newline="\n")
    manifest = {"schema": "v6_2_b2_qp_sphere_trial_manifest_v1",
                "summary_sha256": _sha(summary_path),
                "records_sha256": _sha(records_path),
                "document_sha256": _sha(document_path)}
    with (output_dir / "qp_sphere_manifest.json").open(
        "x", encoding="utf-8", newline="\n"
    ) as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--a1-root", type=Path,
                        default=Path("v6_lite/output/v6_2_a1"))
    parser.add_argument("--original-probe", type=Path, default=Path(
        "v6_lite/output/v6_2_b2/qp_probe_early/weighted_qp_probe.json"))
    args = parser.parse_args()
    result = run(args.output_dir, args.a1_root, args.original_probe)
    print(json.dumps({"complete_group_count": result["complete_group_count"],
                      "candidate_mismatch_count": result["candidate_mismatch_count"],
                      "modes": result["modes"]}, indent=2))
    if result["complete_group_count"] != 4 or result["candidate_mismatch_count"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
