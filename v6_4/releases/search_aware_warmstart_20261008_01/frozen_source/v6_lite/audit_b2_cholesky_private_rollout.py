"""Private prepared/compensated interval QP with SPD diagnostic solve."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from v6_lite import audit_b2_discrete_screened_private_rollout as screened_module
from v6_lite.audit_b2_prepared_screened_private_rollout import run as prepared_run
from v6_lite.b2_cholesky_unconstrained import (
    CHOLESKY_TELEMETRY, CholeskyUnconstrainedScreenQP,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def run(output_dir: Path, a1_root: Path, scenario_id: str,
        max_ticks: int) -> dict:
    previous_qp = screened_module._ScreenQP
    CHOLESKY_TELEMETRY.clear()
    try:
        screened_module._ScreenQP = CholeskyUnconstrainedScreenQP
        report = prepared_run(output_dir, a1_root, scenario_id, max_ticks)
    finally:
        screened_module._ScreenQP = previous_qp
    if (len(CHOLESKY_TELEMETRY) != report["attempted_ticks"]
            or any(item["call_count"] != 1 for item in CHOLESKY_TELEMETRY)):
        raise ValueError("private SPD solve did not cover every QP call")
    report["schema"] = "v6_2_b2_cholesky_private_multicycle_v1"
    report["scope"] = (
        "private_interval_qp_with_prepared_points_pair_screen_compensated_servo_"
        "admm_product_reuse_and_spd_unconstrained_diagnostic_solve"
    )
    report["admm_variant"] = "same_17d_weighted_qp_reuse_residual_products"
    report["unconstrained_diagnostic_solve_variant"] = (
        "cholesky_same_17d_spd_weighted_hessian"
    )
    report["cholesky_unconstrained_call_count"] = sum(
        item["call_count"] for item in CHOLESKY_TELEMETRY)
    report["cholesky_unconstrained_fallback_count"] = sum(
        item["fallback_count"] for item in CHOLESKY_TELEMETRY)
    report["maximum_unconstrained_linear_residual"] = max(
        item["maximum_linear_residual"] for item in CHOLESKY_TELEMETRY)
    report["source_sha256"].update({
        name: _source_sha(Path("v6_lite") / name)
        for name in ("audit_b2_cholesky_private_rollout.py",
                     "b2_cholesky_unconstrained.py",
                     "b2_optimized_admm.py")
    })
    summary_path = output_dir / "private_rollout_summary.json"
    with summary_path.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    doc_path = output_dir / "PRIVATE_ROLLOUT.md"
    document = doc_path.read_text(encoding="utf-8")
    document = document.replace(
        "# B.2 批量点查询、球界筛选与阻尼补偿的私有区间 QP 诊断",
        "# B.2 SPD 无约束诊断解与等价 ADMM 的私有区间 QP 诊断",
    )
    document += (
        "\n仅将 17 维正定加权 Hessian 的无约束诊断解改用 Cholesky；"
        "它不参与 ADMM 初值、约束或实际命令选择。"
        "原 6 维反作用求解、所有安全裕度与执行合同不变。"
        "该实现仅用于私有计时和等价性核对，不是生产在线切换。\n"
    )
    doc_path.write_text(document, encoding="utf-8", newline="\n")
    manifest_path = output_dir / "private_rollout_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["summary_sha256"] = _sha(summary_path)
    manifest["document_sha256"] = _sha(doc_path)
    with manifest_path.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--a1-root", type=Path,
                        default=Path("v6_lite/output/v6_2_a1"))
    parser.add_argument("--scenario-id", default="v6_lite_scenario_02")
    parser.add_argument("--max-ticks", type=int, default=400)
    args = parser.parse_args()
    report = run(args.output_dir, args.a1_root, args.scenario_id,
                 args.max_ticks)
    print(json.dumps({key: report[key] for key in (
        "scenario_id", "executed_ticks", "stop_reason",
        "strict_online_domain_all_executed_ticks",
        "private_preflight_plus_qp_timing",
        "cholesky_unconstrained_call_count",
        "cholesky_unconstrained_fallback_count",
        "maximum_unconstrained_linear_residual",
    )}, indent=2))


if __name__ == "__main__":
    main()
