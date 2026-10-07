"""Private compensated loop with pair screen and prepared PCC point queries.

The private loop retains the original weighted QP, interval rows, collision
constraints, servo limits and MuJoCo model. Two geometry calculations use
same-state shared section prefixes, and the original-row reconstruction still
checks each private command. The production online controller is untouched.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from v6_lite import audit_b2_private_rollout as private_loop
from v6_lite.audit_b2_discrete_screened_private_rollout import run as screened_run
from v6_lite.pcc_batched_distance_query import BatchedDistanceDecisionQuery
from v6_lite.pcc_prepared_state_envelope import PreparedStateLocalPCCEnvelopeAudit


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def run(output_dir: Path, a1_root: Path, scenario_id: str,
        max_ticks: int) -> dict:
    previous_query = private_loop.PersistentIntervalDecisionQuery
    previous_envelope = private_loop.StateLocalPCCEnvelopeAudit
    try:
        private_loop.PersistentIntervalDecisionQuery = BatchedDistanceDecisionQuery
        private_loop.StateLocalPCCEnvelopeAudit = (
            PreparedStateLocalPCCEnvelopeAudit)
        report = screened_run(output_dir, a1_root, scenario_id, max_ticks)
    finally:
        private_loop.PersistentIntervalDecisionQuery = previous_query
        private_loop.StateLocalPCCEnvelopeAudit = previous_envelope
    report["schema"] = "v6_2_b2_prepared_screened_private_multicycle_v1"
    report["scope"] = (
        "private_interval_qp_closed_loop_with_compensated_servo_sphere_screen_and_prepared_points"
    )
    report["pcc_query_variant"] = "batched_signed_distance_shared_section_prefixes"
    report["pcc_envelope_variant"] = "batched_points_original_capsule_inequality"
    report["source_sha256"].update({
        name: _source_sha(Path("v6_lite") / name)
        for name in ("audit_b2_prepared_screened_private_rollout.py",
                     "pcc_batched_distance_query.py",
                     "pcc_prepared_state_envelope.py",
                     "pcc_batched_point_model.py")
    })
    summary_path = output_dir / "private_rollout_summary.json"
    with summary_path.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    document_path = output_dir / "PRIVATE_ROLLOUT.md"
    document = document_path.read_text(encoding="utf-8")
    document = document.replace(
        "# B.2 球界筛选与阻尼补偿的私有多周期区间 QP 力矩诊断",
        "# B.2 批量点查询、球界筛选与阻尼补偿的私有区间 QP 诊断",
    )
    document += (
        "\n同状态 PCC 段前缀用于代理距离查询和实际链胶囊包络；"
        "包络半径、采样允差、数值余量和安全门槛均保留原值。"
        "阶段二私有试验，不是生产在线接入。\n"
    )
    document_path.write_text(document, encoding="utf-8", newline="\n")
    manifest_path = output_dir / "private_rollout_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["summary_sha256"] = _sha(summary_path)
    manifest["document_sha256"] = _sha(document_path)
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
