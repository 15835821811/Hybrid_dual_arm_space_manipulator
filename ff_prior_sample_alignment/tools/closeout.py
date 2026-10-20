"""Terminal-run collection and exactly-once completed-resume verification."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from collections import Counter

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from ff_prior.storage import read_json
from ff_prior.recovery import process_is_alive
from ff_prior_endpoint_jacobian.statistics import aggregate, pose, paired
from ff_prior_sample_alignment.protocol import DEFAULT_OUT as OUT, CONTROL, METHODS

EV = ROOT / "ff_prior_sample_alignment/evidence/completed"


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1048576), b""):
            h.update(block)
    return h.hexdigest()


def write(name, value):
    EV.mkdir(parents=True, exist_ok=True)
    (EV / name).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def inventory(root):
    return {p.relative_to(root).as_posix(): sha(p) for p in sorted(root.rglob("*")) if p.is_file()}


def require_terminal():
    state = read_json(OUT / "status.json")
    if state["status"] == "RUNNING" or process_is_alive(state["pid"]):
        raise RuntimeError("Running worker: collection deferred")
    return state


def collect():
    state = require_terminal()
    pre = read_json(ROOT / "ff_prior_sample_alignment/evidence/preflight.json")
    assert all(sha(ROOT / n) == h for n, h in pre["dirty_file_hashes"].items())
    result = {"status": state, "user_file_hashes_unchanged": True,
              "head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
              "source": "saved original result files only; no simulation or optimizer update"}
    if (OUT / "failure.json").exists():
        result["failure"] = read_json(OUT / "failure.json")
    for name in ("numerical_sanity", "summary", "independent_verification", "query_budget",
                 "paired_statistics", "application_statistics"):
        if (OUT / (name + ".json")).exists():
            result[name] = read_json(OUT / (name + ".json"))
    result["training"] = {}
    for kind in METHODS:
        path = OUT / "learning" / kind / "training.json"
        if path.exists():
            r = read_json(path)
            result["training"][kind] = {k: r[k] for k in (
                "updates", "TRAIN_rows_seen", "actual_TRAIN_draws", "initial_weight_hash",
                "trained_weight_hash", "checkpoint_sha256", "VAL_gradient_updates",
                "VAL_checkpoint_selection", "full16_teacher_batches", "wall_s")}
            result["training"][kind].update(world_events=len(r["world_events"]),
                                            rows_with_zero_draws=sum(n == 0 for n in r["train_draw_counts"].values()),
                                            minimum_row_draws=min(r["train_draw_counts"].values()),
                                            maximum_row_draws=max(r["train_draw_counts"].values()))
    if (OUT / "ledger.json").exists():
        ledger = read_json(OUT / "ledger.json")
        result["ledger"] = {k: v for k, v in ledger.items() if k != "completed"}
        result["ledger"]["completed_count"] = len(ledger["completed"])
        result["actual_coupled"] = sum(v for k, v in ledger["calls"].items() if k.endswith(".coupled"))
    result["result_bytes"] = sum(p.stat().st_size for p in OUT.rglob("*") if p.is_file())
    if not (OUT / "paired_statistics.json").exists():
        write("evidence_summary.json", result)
        return result
    result["unguided"] = {}
    result["DEV_by_seed"] = {}
    result["DEV_by_task"] = {}
    result["DEV_groups"] = {}
    result["DEV_failures"] = {}
    entries = read_json(OUT / "frozen_tasks.json")["entries"]
    for kind in METHODS:
        rs = [read_json(OUT / "unguided" / str(sj) / kind / f"{j:02d}.json") for sj in range(2) for j in range(48)]
        result["unguided"][kind] = {
            "all": aggregate(rs),
            "by_seed": {str(sj): aggregate(rs[sj*48:(sj+1)*48]) for sj in range(2)},
            "model_levels": dict(Counter(r["validation"]["validation_level"] for r in rs)),
            "failed_gates": dict(Counter(k for r in rs for k, v in r["validation"]["gates"].items() if not v)),
            "position_pass": sum(pose(r) is not None and pose(r)["position_mm"] <= 5 for r in rs),
            "rotation_pass": sum(pose(r) is not None and pose(r)["rotation_deg"] <= 1 for r in rs),
            "physics_guidance_calls": sum(r["physics_guidance_calls"] for r in rs),
        }
    for kind in (*METHODS, "structured", "learned_mean"):
        root = OUT if kind in METHODS else CONTROL
        all_c, all_e, outcomes = [], [], []
        for sj in range(3):
            cs, es = [], []
            for ti, e in enumerate(entries):
                sid = e["sample_id"]
                c = [read_json(root / "candidates" / str(sj) / sid / kind / f"{k:02d}.json") for k in range(5)]
                d = root / "executions" / str(sj) / sid / kind
                ex = read_json(d / "report.json") if (d / "report.json").exists() else {}
                outcome = read_json(d / "outcome.json")
                result["DEV_by_task"][f"{sj}/{sid}/{kind}"] = {
                    "candidate": aggregate(c), "execution": aggregate([ex], True), "outcome": outcome["status"]}
                cs += c
                es.append(ex)
                outcomes.append(outcome)
            all_c += cs
            all_e += es
            result["DEV_by_seed"][f"{sj}/{kind}"] = {"candidate": aggregate(cs), "execution": aggregate(es, True)}
        for label, indices in (("original_DEV4", range(4)), ("used_development8", range(4, 12))):
            c = [r for sj in range(3) for ti in indices for r in all_c[(sj*12+ti)*5:(sj*12+ti+1)*5]]
            e = [all_e[sj*12+ti] for sj in range(3) for ti in indices]
            result["DEV_groups"][f"{label}/{kind}"] = {"candidate": aggregate(c), "execution": aggregate(e, True)}
        result["DEV_failures"][kind] = {
            "candidate_status": dict(Counter(r.get("status", "MISSING") for r in all_c)),
            "failed_gates": dict(Counter(k for r in all_c for k, v in r.get("validation", {}).get("gates", {}).items() if not v)),
            "execution_status": dict(Counter(o["status"] for o in outcomes)),
        }
    result["application_failures"] = {}
    for e in read_json(OUT / "frozen_tasks.json")["application"]["entries"]:
        for kind in ("plain", "aligned", "structured", "learned_mean", "aligned_unguided"):
            ps = sorted((OUT / "application" / e["sample_id"] / kind / "candidates").glob("*.json"))
            rs = [read_json(p) for p in ps]
            result["application_failures"][f"{e['sample_id']}/{kind}"] = {
                "expected_denominator": 5, "found": len(rs),
                "status": dict(Counter(r.get("status", "MISSING") for r in rs)),
                "reasons": dict(Counter(r.get("reason", "") for r in rs if r.get("reason"))),
                "failed_gates": dict(Counter(k for r in rs for k, v in r.get("validation", {}).get("gates", {}).items() if not v)),
                "endpoint": aggregate(rs + [{}]*(5-len(rs))),
            }
    write("evidence_summary.json", result)
    return result


def resume_once():
    if (EV / "resume_verification.json").exists():
        return read_json(EV / "resume_verification.json")
    if (EV / "resume_started.json").exists():
        raise RuntimeError("Prior check uncertain; do not repeat resume")
    state = require_terminal()
    assert state["status"] in ("FINISHED", "FINISHED_WITH_BLOCKED_ITEMS")
    before = inventory(OUT)
    write("core_manifest_before_resume.json", before)
    write("resume_started.json", {"time": time.time(), "pid": state["pid"], "owner_alive": False})
    result = subprocess.run([sys.executable, "-m", "ff_prior_sample_alignment.job", "resume"], cwd=ROOT,
                            env=dict(os.environ, PYTHONIOENCODING="utf-8"),
                            text=True, encoding="utf-8", capture_output=True)
    write("resume_command.json", {"returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr})
    assert result.returncode == 0
    answer = json.loads(result.stdout)
    after = inventory(OUT)
    write("core_manifest_after_resume.json", after)
    assert answer["resume"] == "NO_WORK_ALREADY_COMPLETE" and before == after
    value = {"status": "PASS", "file_count": len(before), "actual_resume_calls": 1,
             "complete_file_list_and_hashes_unchanged": True, "answer": answer}
    write("resume_verification.json", value)
    return value


if __name__ == "__main__":
    if sys.argv[1:] == ["collect"]:
        result = collect()
        print(json.dumps({"status": result["status"]["status"], "result_bytes": result["result_bytes"],
                          "training": result["training"]}, ensure_ascii=False))
    elif sys.argv[1:] == ["resume"]:
        print(json.dumps(resume_once(), ensure_ascii=False))
    else:
        raise SystemExit("Use collect or resume")
