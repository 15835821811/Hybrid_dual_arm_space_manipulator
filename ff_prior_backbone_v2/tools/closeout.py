"""Read saved evidence and verify completed resume once; no experiment work."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import hashlib
from collections import Counter
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from ff_prior.storage import read_json
from ff_prior.recovery import process_is_alive
from ff_prior_endpoint_jacobian.statistics import aggregate, paired, pose
from ff_prior_backbone_v2.protocol import CONTROL, METHODS

OUT = ROOT / "artifacts/v7_1/backbone_pair_smoke_002"
EVIDENCE = ROOT / "ff_prior_backbone_v2/evidence/completed"


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write(name, value):
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    (EVIDENCE / name).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def inventory(root):
    return {p.relative_to(root).as_posix(): sha(p) for p in sorted(root.rglob("*")) if p.is_file()}


def completed_resume_once():
    path = EVIDENCE / "resume_verification.json"
    if path.exists():
        return read_json(path)
    marker = EVIDENCE / "resume_started.json"
    if marker.exists():
        raise RuntimeError("Prior resume check uncertain; inspect evidence, do not repeat.")
    state = read_json(OUT / "status.json")
    assert state["status"] == "FINISHED" and not process_is_alive(state["pid"])
    before = inventory(OUT)
    write("core_manifest_before_resume.json", before)
    write("resume_started.json", {"time": time.time(), "pid": state["pid"], "owner_alive": False})
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    result = subprocess.run([sys.executable, "-m", "ff_prior_backbone_v2.job", "resume"],
                            cwd=ROOT, env=env, text=True, encoding="utf-8", capture_output=True)
    write("resume_command.json", {"returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr})
    assert result.returncode == 0
    answer = json.loads(result.stdout)
    after = inventory(OUT)
    write("core_manifest_after_resume.json", after)
    assert answer["resume"] == "NO_WORK_ALREADY_COMPLETE" and before == after
    proof = {"status": "PASS", "command_return": answer, "file_count": len(before),
             "full_file_list_and_sha256_unchanged": True, "actual_resume_calls": 1}
    write("resume_verification.json", proof)
    return proof


def collect():
    frozen = read_json(OUT / "frozen_tasks.json")["entries"]
    stats = read_json(OUT / "paired_statistics.json")
    evidence = {"unguided_all": {}, "unguided_failures": {}, "methods": stats["methods"],
                "by_seed": {}, "by_task": {}, "by_development_group": {}, "training": {},
                "paired": stats["paired"], "source": "independently aggregated saved rows; no new physics"}
    for method in METHODS:
        all_rows = [read_json(OUT / "unguided" / str(sj) / method / f"{j:02d}.json")
                    for sj in range(2) for j in range(48)]
        evidence["unguided_all"][method] = aggregate(all_rows)
        for sj in range(2):
            assert aggregate(all_rows[sj * 48:(sj + 1) * 48]) == stats["unguided"][f"{sj}_{method}"]
        evidence["unguided_failures"][method] = {
            "frozen_goal_pass": sum(r.get("frozen_goal_pass", False) for r in all_rows),
            "position_pass": sum(pose(r)["position_mm"] <= 5 for r in all_rows),
            "rotation_pass": sum(pose(r)["rotation_deg"] <= 1 for r in all_rows),
            "qualification": sum(r["eligible"] for r in all_rows),
            "nonfinite_prepared_trajectories": sum(not np.isfinite(r["prepared_Z"]).all() for r in all_rows),
            "physics_guidance_calls": sum(r["physics_guidance_calls"] for r in all_rows),
            "model_validation_levels": dict(Counter(r["validation"]["validation_level"] for r in all_rows)),
            "non_goal_gate_failures": dict(Counter(k for r in all_rows
                                                  for k, passed in r["validation"]["gates"].items() if not passed))}
        t = read_json(OUT / "learning" / method / "training.json")
        evidence["training"][method] = {k: t[k] for k in (
            "updates", "actual_TRAIN_draws", "TRAIN_rows_seen", "initial_weight_hash",
            "trained_weight_hash", "checkpoint_sha256", "VAL_gradient_updates",
            "VAL_checkpoint_selection", "world_events", "wall_s")}
        evidence["training"][method]["draw_count_min"] = min(t["train_draw_counts"].values())
        evidence["training"][method]["draw_count_max"] = max(t["train_draw_counts"].values())
        evidence["training"][method]["segment_count"] = len(t["segments"])
        evidence["training"][method]["last_logged_batch"] = t["history"][-1]
    for method in (*METHODS, "structured", "learned_mean"):
        root = OUT if method in METHODS else CONTROL
        cs, es = [], []
        for sj in range(3):
            sc, se = [], []
            for ti, entry in enumerate(frozen):
                sid = entry["sample_id"]
                c = [read_json(root / "candidates" / str(sj) / sid / method / f"{k:02d}.json") for k in range(5)]
                d = root / "executions" / str(sj) / sid / method
                ex = read_json(d / "report.json")
                outcome = read_json(d / "outcome.json")
                assert outcome["status"] == ex["level"] == "EXECUTION_VERIFIED"
                cs.extend(c)
                es.append(ex)
                sc.extend(c)
                se.append(ex)
                evidence["by_task"][f"{sj}/{sid}/{method}"] = {
                    "candidate": aggregate(c), "execution": aggregate([ex], True)}
            evidence["by_seed"][f"{sj}/{method}"] = {"candidate": aggregate(sc), "execution": aggregate(se, True)}
        assert aggregate(cs) == stats["methods"][method]["candidate"]
        assert aggregate(es, True) == stats["methods"][method]["execution"]
        for label, indices in (("original_DEV4", range(4)), ("used_development8", range(4, 12))):
            subset_c = [r for sj in range(3) for ti in indices for r in cs[(sj * 12 + ti) * 5:(sj * 12 + ti + 1) * 5]]
            subset_e = [es[sj * 12 + ti] for sj in range(3) for ti in indices]
            evidence["by_development_group"][f"{label}/{method}"] = {
                "candidate": aggregate(subset_c), "execution": aggregate(subset_e, True)}
    ledger = read_json(OUT / "ledger.json")
    assert ledger["inflight"] is None and not ledger.get("unresolved_call_accounting")
    evidence["ledger"] = {k: v for k, v in ledger.items() if k != "completed"}
    evidence["ledger"]["completed_count"] = len(ledger["completed"])
    evidence["elapsed_s"] = read_json(OUT / "status.json")["elapsed_s"]
    evidence["actual_coupled"] = sum(v for k, v in ledger["calls"].items() if k.endswith(".coupled"))
    evidence["query_budget"] = read_json(OUT / "query_budget.json")
    evidence["result_bytes"] = sum(p.stat().st_size for p in OUT.rglob("*") if p.is_file())
    evidence["raw_audit"] = read_json(OUT / "independent_verification.json")
    pre = read_json(ROOT / "ff_prior_backbone_v2/evidence/preflight.json")
    assert all(sha(ROOT / n) == h for n, h in pre["dirty_file_hashes"].items())
    failed = read_json(ROOT / "ff_prior_backbone_v2/evidence/failed_launch_001.json")
    assert inventory(Path(failed["failed_run"])) == failed["files"]
    evidence["preservation"] = {"four_user_modifications_hash_equal": True,
                              "failed_001_full_inventory_equal": True,
                              "failed_001_file_count": len(failed["files"]),
                              "current_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                              "note": "unrelated 755b1eaf scene-evidence commit preserved; current experiment source identity remains 4ccba221"}
    write("evidence_summary.json", evidence)
    return evidence


if __name__ == "__main__":
    if sys.argv[1:] == ["resume"]:
        print(json.dumps(completed_resume_once(), ensure_ascii=False))
    elif sys.argv[1:] == ["collect"]:
        result = collect()
        print(json.dumps({"unguided_all": result["unguided_all"],
                          "unguided_failures": result["unguided_failures"],
                          "actual_coupled": result["actual_coupled"],
                          "result_bytes": result["result_bytes"]}, ensure_ascii=False, indent=2))
    else:
        raise SystemExit("Use collect or resume")
