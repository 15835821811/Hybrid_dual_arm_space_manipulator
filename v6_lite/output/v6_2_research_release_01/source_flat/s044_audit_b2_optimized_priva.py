"""Private prepared/screened/compensated loop with product-reuse ADMM."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from v6_lite import audit_b2_discrete_screened_private_rollout as screened_module
from v6_lite.audit_b2_prepared_screened_private_rollout import run as prepared_run
from v6_lite.b2_optimized_admm import OptimizedScreenQP


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def run(output_dir: Path, a1_root: Path, scenario_id: str,
        max_ticks: int) -> dict:
    previous_qp = screened_module._ScreenQP
    try:
        screened_module._ScreenQP = OptimizedScreenQP
        report = prepared_run(output_dir, a1_root, scenario_id, max_ticks)
    finally:
        screened_module._ScreenQP = previous_qp
    report["schema"] = "v6_2_b2_optimized_private_multicycle_v1"
    report["scope"] = (
        "private_interval_qp_with_prepared_points_pair_screen_compensated_servo_and_product_reuse_admm"
    )
    report["admm_variant"] = "same_17d_weighted_qp_reuse_residual_products"
    report["source_sha256"].update({
        name: _source_sha(Path("v6_lite") / name)
        for name in ("audit_b2_optimized_private_rollout.py",
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
        "# B.2 批量点、球界筛选、阻尼补偿与等价 ADMM 的私有区间 QP 诊断",
    )
    document += (
        "\n该单一 17 维加权 QP 的 ADMM 循环仅复用每轮已计算的 Hessian/"
        "转置约束矩阵乘积；约束、阈值、热启动及 rho 更新规则不变。"
        "私有诊断，未接入生产在线控制。\n"
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
    parser.add_argument("--scenario-id", default="v6_lite_scenario_00")
    parser.add_argument("--max-ticks", type=int, default=400)
    args = parser.parse_args()
    report = run(args.output_dir, args.a1_root, args.scenario_id,
                 args.max_ticks)
    print(json.dumps({
        "scenario_id": report["scenario_id"],
        "executed_ticks": report["executed_ticks"],
        "stop_reason": report["stop_reason"],
        "strict_online_domain_all_executed_ticks": report[
            "strict_online_domain_all_executed_ticks"],
        "private_preflight_plus_qp_timing": report[
            "private_preflight_plus_qp_timing"],
    }, indent=2))


if __name__ == "__main__":
    main()
