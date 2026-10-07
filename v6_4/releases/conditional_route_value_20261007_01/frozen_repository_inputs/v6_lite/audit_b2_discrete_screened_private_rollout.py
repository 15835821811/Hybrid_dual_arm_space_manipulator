"""Private compensated servo loop with the previously parity-checked pair screen.

This diagnostic changes only the private QP's exact MuJoCo pair query path.
The screen uses a conservative bounding-sphere lower bound and falls back to
the original exact pairs when required. Independent original-row reconstruction
inside the private loop checks each selected command before the servo advances.
No production online controller is changed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from v6_lite import audit_b2_private_rollout as private_loop
from v6_lite.audit_b2_discrete_private_rollout import run as compensated_run
from v6_lite.audit_b2_private_qp_sphere import _ScreenQP, _TELEMETRY


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def run(output_dir: Path, a1_root: Path, scenario_id: str,
        max_ticks: int) -> dict:
    previous_qp = private_loop._ReadOnlyIntervalQP
    _TELEMETRY.clear()
    try:
        private_loop._ReadOnlyIntervalQP = _ScreenQP
        report = compensated_run(output_dir, a1_root, scenario_id, max_ticks)
    finally:
        private_loop._ReadOnlyIntervalQP = previous_qp
    if len(_TELEMETRY) != report["attempted_ticks"]:
        raise ValueError("sphere-screen telemetry does not cover attempted QP ticks")
    report["schema"] = "v6_2_b2_discrete_screened_private_multicycle_v1"
    report["scope"] = (
        "private_interval_qp_closed_loop_with_compensated_servo_and_sphere_screen"
    )
    report["mujoco_pair_screen_variant"] = (
        "conservative_bounding_sphere_lower_bound_original_exact_fallback"
    )
    report["sphere_screen_exact_pair_calls"] = [
        item["exact_call_count"] for item in _TELEMETRY
    ]
    report["sphere_screen_fallback_ticks"] = sum(
        item["fallback_to_all_pairs"] for item in _TELEMETRY
    )
    report["source_sha256"].update({
        name: _source_sha(Path("v6_lite") / name)
        for name in ("audit_b2_discrete_screened_private_rollout.py",
                     "audit_b2_private_qp_sphere.py",
                     "audit_b2_weighted_qp_sphere_trial.py",
                     "audit_b2_mujoco_sphere_screen.py")
    })
    summary_path = output_dir / "private_rollout_summary.json"
    with summary_path.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    doc_path = output_dir / "PRIVATE_ROLLOUT.md"
    document = doc_path.read_text(encoding="utf-8")
    document = document.replace(
        "# B.2 隐式积分阻尼补偿的私有多周期区间 QP 力矩诊断",
        "# B.2 球界筛选与阻尼补偿的私有多周期区间 QP 力矩诊断",
    )
    document += (
        "\n原 MuJoCo 碰撞对仅在保守球界严格超过原查询半径时跳过精确查询；"
        "每周期仍独立重构原碰撞约束行并核对。"
        "这是私有闭环时延诊断，不是生产在线接入。\n"
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
        "sphere_screen_fallback_ticks": report[
            "sphere_screen_fallback_ticks"],
    }, indent=2))


if __name__ == "__main__":
    main()
