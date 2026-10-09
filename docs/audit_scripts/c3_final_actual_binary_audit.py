"""Post-run byte seals and saved cold-state audit; no experiment imports or execution.

Run only after the formal pipeline exits. This supplements the metadata audit:
it reads full actual seals and existing NPZ arrays, but never simulates, samples,
trains, or regenerates evidence. Output must be outside the frozen run.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

POLICY = "fresh runner/model/MjData/controller/reference provider/integrator/QP and ADMM cache; no previous candidate or stream state"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def digest(value, ascii=True):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    ensure_ascii=ascii, allow_nan=False).encode()).hexdigest()


class Audit:
    def __init__(self):
        self.checks = 0
        self.files = {}
        self.slots = []

    def check(self, condition, label):
        self.checks += 1
        if not condition:
            raise ValueError(label)

    def sha(self, path):
        path = Path(path).resolve()
        before = path.stat()
        previous = self.files.get(str(path))
        if previous:
            self.check((before.st_size, before.st_mtime_ns) == (previous["bytes"], previous["mtime_ns"]),
                       "input changed during audit: " + str(path))
            return previous["sha256"]
        h = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
                h.update(block)
        after = path.stat()
        self.check((before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns),
                   "input changed while hashing: " + str(path))
        value = h.hexdigest()
        self.files[str(path)] = {"bytes": after.st_size, "mtime_ns": after.st_mtime_ns, "sha256": value}
        return value

    def seal(self, directory):
        directory = directory.resolve()
        self.sha(directory / "manifest.json")
        entries = read(directory / "manifest.json")
        self.check(bool(entries), "empty seal: " + str(directory))
        for relative, expected in entries.items():
            path = (directory / relative).resolve()
            self.check(path.is_relative_to(directory), "seal path escapes directory")
            self.check(self.sha(path) == expected, "seal hash mismatch: " + str(path))
        return len(entries)


def run_audit(audit, repo, run):
    for stage in ("test-search", "execute-test", "validate", "report"):
        path = run / "command_logs" / ("phase_" + stage + ".json")
        receipt = read(path)
        audit.sha(path)
        audit.check(receipt["exit_code"] == 0 and receipt["stage"] == stage,
                    "formal phase incomplete: " + stage)
    identity_path = run / "source_identity.json"
    identity = read(identity_path)
    identity_sha = audit.sha(identity_path)
    sources = identity["source_sha256"]
    protected = identity["protected_artifacts"]
    audit.check(len(sources) == 430 and len(protected) == 37, "frozen identity inventory differs")
    for relative, expected in sources.items():
        path = (repo / relative).resolve()
        audit.check(path.is_relative_to(repo), "source path escapes repository")
        audit.check(audit.sha(path) == expected, "producer source differs: " + relative)
    for name, expected in protected.items():
        audit.check(audit.sha(name) == expected, "protected artifact differs: " + name)
    qp = read(run / "frozen_execution_config.json")
    config = read(run / "frozen_run_config.json")
    config_sha = digest({"run": config, "qp": qp}, ascii=False)
    all_slots = []
    for phase, root, expected_count in (("VAL", run / "closed_loop_val/actual", 20),
                                         ("TEST", run / "frozen_test/actual", 40)):
        paths = sorted(root.glob("*/*/slot.json"))
        audit.check(len(paths) == expected_count, phase + " logical slot count differs")
        all_slots.extend((phase, path) for path in paths)
    normalized_sources = {}
    for phase, path in all_slots:
        slot = read(path)
        count = audit.seal(path.parent)
        record = {"phase": phase, "task_id": slot["task_id"], "method": slot["method"],
                  "actual_steps": slot["actual_steps"], "slot_sha256": audit.sha(path),
                  "sealed_files": count, "unique_run": slot.get("unique_run", False),
                  "alias_of_slot": slot.get("alias_of_slot"), "cold_state_check": "NOT_APPLICABLE"}
        audit.slots.append(record)
        if slot.get("alias_of_slot"):
            original_path = Path(slot["alias_of_slot"]).resolve()
            audit.check(original_path.is_relative_to(run), "alias source outside formal run")
            audit.check(audit.sha(original_path) == slot["alias_of_slot_sha256"], "alias source hash differs")
            original = read(original_path)
            audit.check(not original.get("alias_of_slot") and all(slot[k] == original[k]
                        for k in ("task_sha256", "plan_sha256", "alias_identity")), "strict alias binding differs")
            record["cold_state_check"] = "STRICT_ALIAS_OF_AUDITED_UNIQUE"
            continue
        attempt_path = path.parent / "attempt/attempt_result.json"
        if not attempt_path.is_file():
            audit.check(slot["actual_steps"] == 0 and not slot.get("entered_actual"), "missing positive-step attempt")
            continue
        attempt = read(attempt_path)
        audit.check(attempt["source_sha256_before"] == sources == attempt["source_sha256_after"],
                    "attempt before/after source identity differs")
        audit.check(attempt["source_identity_sha256"] == identity_sha and attempt["sources_unchanged"],
                    "attempt source binding differs")
        audit.check(attempt["qp_config_sha256"] == audit.sha(run / "frozen_execution_config.json"),
                    "attempt QP configuration differs")
        task = read(run / "frozen_tasks" / slot["task_id"] / "task.json")
        history = {"policy": POLICY, **{k: task[k] for k in
                   ("initial_qpos", "initial_qvel", "initial_planner_q", "initial_planner_dq")}}
        audit.check(digest(history) == slot["actual_binding"]["initial_history_sha256"], "cold-history digest differs")
        metadata_path = path.parent / "attempt/actual/run_metadata.json"
        if metadata_path.is_file():
            metadata = read(metadata_path)
            audit.check(metadata["source"]["git_commit"] == identity["algorithm_producer_commit"]
                        and metadata["source"]["tracked_worktree_dirty"] is False, "runtime producer differs")
            audit.check(metadata["run_config"] == config and metadata["qp_config"] == qp
                        and metadata["configuration_sha256"] == config_sha, "effective runtime configuration differs")
            runtime = metadata["runtime_identity"]
            audit.check(runtime["model_runtime_contract_sha256"] == task["model_contract_sha256"], "model contract differs")
            audit.check(runtime["controller_version"] == "v6_2_research_simulation_bounded_interval_pcc"
                        and runtime["servo_law_version"] == "b2_implicitfast_compensated_torque_v1"
                        and runtime["pcc_mode"] == "bounded_interval_pcc"
                        and runtime["execution_mode"] == "research_simulation"
                        and runtime["integrator_name"] == "implicitfast"
                        and runtime["wall_deployment_certified"] is False, "runtime control identity differs")
            for name, expected in metadata["source"]["source_normalized_lf_sha256"].items():
                if name not in normalized_sources:
                    source = (repo / name).resolve()
                    audit.check(source.is_relative_to(repo), "normalized source outside repository")
                    audit.sha(source)
                    normalized_sources[name] = hashlib.sha256(source.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
                audit.check(normalized_sources[name] == expected, "runtime normalized source differs: " + name)
        trace_name = attempt.get("trace_path")
        if not trace_name:
            audit.check(slot["actual_steps"] == 0, "positive-step slot missing trace")
            continue
        trace_path = Path(trace_name).resolve()
        audit.check(trace_path.is_relative_to(path.parent), "actual trace escapes original attempt")
        audit.check(audit.sha(trace_path) == attempt["trace_sha256"], "trace binding differs")
        with np.load(trace_path, allow_pickle=False) as trace:
            audit.check(np.array_equal(trace["initial_qpos"], task["initial_qpos"])
                        and np.array_equal(trace["initial_qvel"], task["initial_qvel"]), "saved actual initial state differs")
            audit.check(len(trace["torque"]) == slot["actual_steps"], "actual trace step count differs")
            record["cold_state_check"] = "BIT_EXACT_TASK_QPOS_QVEL"
            record["initial_qpos_dimensions"] = list(trace["initial_qpos"].shape)
            record["initial_qvel_dimensions"] = list(trace["initial_qvel"].shape)
        evaluation = path.parent / "attempt/actual/evaluation"
        if (evaluation / "manifest.json").is_file():
            audit.seal(evaluation)
    return {"source_files": len(sources), "protected_artifacts": len(protected),
            "runtime_normalized_source_files": len(normalized_sources), "logical_slots": len(all_slots),
            "unique_saved_initial_states_checked": sum(s["cold_state_check"] == "BIT_EXACT_TASK_QPOS_QVEL" for s in audit.slots)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    repo, run, output = (p.resolve() for p in (args.repo, args.run, args.output))
    if output.is_relative_to(run) or output.exists():
        raise ValueError("audit output must be a fresh path outside the formal run")
    started = datetime.now(timezone.utc).isoformat()
    clock = time.perf_counter()
    audit = Audit()
    result = {"schema": "c3_actual_binary_and_cold_state_audit_v1", "started_utc": started,
              "argv": [sys.executable, *sys.argv], "script_sha256": audit.sha(__file__),
              "repo": str(repo), "run": str(run), "status": "FAIL", "physics_steps": 0,
              "model_samples": 0, "training_updates": 0,
              "scope": "Current producer/protected bytes, all VAL/TEST actual seals, saved initial qpos/qvel and original runtime identities; no new acceptance or direct controller-history telemetry."}
    try:
        result["counts"] = run_audit(audit, repo, run)
        result["status"] = "PASS"
    except Exception as error:
        result["error"] = {"type": type(error).__name__, "message": str(error)}
    result.update(checks_total=audit.checks, files=audit.files, slots=audit.slots,
                  ended_utc=datetime.now(timezone.utc).isoformat(), elapsed_wall_s=time.perf_counter() - clock)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps({k: result[k] for k in ("status", "checks_total", "elapsed_wall_s")} | {"output": str(output), "error": result.get("error")}))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
