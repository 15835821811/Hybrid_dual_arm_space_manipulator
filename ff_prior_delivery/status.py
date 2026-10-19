"""Read-only status entrypoint. It does not launch, audit, or resume experiments."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def status():
    policy = read(ROOT / "ff_prior_delivery/policy.json")
    evidence = read(ROOT / "ff_prior_delivery/evidence_reuse.json")
    changed = []
    for source in evidence["sources"]:
        path = ROOT / source["path"]
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != source["sha256"]:
            changed.append(source["path"])
    stages = []
    for item in policy["stages"]:
        entry = dict(item)
        runtime_path = ROOT / "artifacts/v7_1" / item["run"] / "status.json"
        if item["id"] != "M0" and runtime_path.is_file():
            runtime = read(runtime_path)
            entry["runtime"] = {k: runtime[k] for k in
                                ("status", "stage", "pid", "heartbeat", "report") if k in runtime}
            entry["status"] = "SEE_RUNTIME_NOT_AN_ACCEPTANCE_VERDICT"
        stages.append(entry)
    release_dir = ROOT / 'ff_prior_delivery_release/evidence'
    current_data = read(release_dir / 'dataset_manifest.json') if (release_dir / 'dataset_manifest.json').is_file() else None
    return {
        "goal_version": policy["version"],
        "source_evidence": "UNCHANGED" if not changed else "CHANGED_REVIEW_REQUIRED",
        "changed_sources": changed,
        "historical_mechanism": evidence["historical_mechanism"] if not changed else "UNCONFIRMED",
        "historical_data_inventory": evidence["data_inventory"],
        "data_inventory": {
            "counts_by_split": current_data['counts_by_split'],
            "counts_by_split_and_level": current_data['counts_by_split_and_level'],
            "index": current_data['index'],
        } if current_data else evidence["data_inventory"],
        "delivery_completion": read(release_dir / 'completion.json') if (release_dir / 'completion.json').is_file() else "PENDING",
        "old_world_supervision": "HISTORICAL_BLOCKED_NOT_RELABELED",
        "new_physical_experiments_in_this_goal_revision": sum(
            (ROOT / 'artifacts/v7_1' / item['run'] / 'run.json').exists()
            for item in policy['stages'] if item['id'] != 'M0'),
        "stages": stages,
        "forbidden_runs": policy["forbidden_runs"],
        "app_goal": policy["app_goal"],
        "query_does_not_launch_work": True,
    }


if __name__ == "__main__":
    print(json.dumps(status(), ensure_ascii=False, indent=2))
