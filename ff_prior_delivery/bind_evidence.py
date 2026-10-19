"""One-shot binding of existing compact evidence, not a new physical audit."""
import hashlib
import json
from datetime import datetime
from pathlib import Path
from .gates import engineering_gate

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "ff_prior_delivery"
PATHS = [
    "ff_prior_origin_learn_bounded/evidence/completed_002/summary.json",
    "ff_prior_origin_learn_bounded/evidence/completed_002/independent_verification.json",
    "ff_prior_origin_learn_bounded/evidence/completed_002/checkpoint_binding.json",
    "ff_prior_source_alignment/evidence/local_alignment_002.json",
    "ff_prior_paired_objectives/evidence/completed_001/summary.json",
]


def main():
    target = PACKAGE / "evidence_reuse.json"
    if target.exists():
        raise RuntimeError("Evidence snapshot already exists; preserve it")
    raw = [json.loads((ROOT / p).read_text(encoding="utf-8-sig")) for p in PATHS]
    summary, audit, binding, inventory, paired = raw
    assert audit["status"] == "PASS" and audit["problems"] == []
    assert summary["independent_audit_passed"] is True
    assert inventory["status"] == "PASS" and inventory["problems"] == []
    assert paired["TRAIN_selection"]["status"] == "BLOCKED"
    method = summary["methods"]["expanded_diffusion"]
    gate = engineering_gate(method, 180, 36, True, summary["errors"])
    assert gate["status"] == "PASS"
    result = {
        "created_at": datetime.now().astimezone().isoformat(),
        "scope": "reuse existing compact evidence; no new audit/physics/training/DEV",
        "historical_mechanism": {
            "run": "qualified_origin_learn_002", "status": "PASS_REUSED",
            "checkpoint_sha256": binding["new"], "TRAIN_rows": audit["TRAIN_rows"],
            "eligible": method["eligible"], "candidates": method["completed"],
            "execution_verified": method["execution_verified"],
            "execution_units": method["selected_outcomes"],
            "historical_raw_audit": audit["status"], "new_engineering_policy_assessment": gate,
            "benefit": "NOT_ESTABLISHED",
        },
        "data_inventory": {"TRAIN": inventory["data"]["TRAIN"]["rows"],
                           "VAL": inventory["data"]["VAL"]["rows"],
                           "validation_levels": inventory["validation_levels"]},
        "old_world_supervision_gate": paired["TRAIN_selection"]["status"],
        "sources": [{"path": p, "sha256": hashlib.sha256((ROOT / p).read_bytes()).hexdigest()}
                    for p in PATHS],
    }
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    policy_path = PACKAGE / "policy.json"
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    policy["baseline_checkpoint_sha256"] = binding["new"]
    policy_path.write_text(json.dumps(policy, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["historical_mechanism"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
