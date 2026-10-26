"""Register actual successful teacher attempts without losing failure records."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from v6_4.contracts import TrajectoryProposal
from v6_4.dataset import register_teacher_sample, write_teacher_manifest
from v6_4.task_protocol import TaskSpec


def collect(trial_dirs, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    samples, attempts = [], []
    for index, trial in enumerate(trial_dirs):
        trial = Path(trial)
        result_path = trial / "result.json"
        result = json.loads(result_path.read_text(encoding="utf-8"))
        record = {"trial_path": trial.resolve().as_posix(), "status": result["status"],
                  "task_id": result["task_id"], "result_sha256": hashlib.sha256(result_path.read_bytes()).hexdigest(),
                  "registered_successful_teacher": False}
        if result["method"] == "teacher" and result.get("task_success") is True:
            task = TaskSpec.from_dict(json.loads((trial / "task.json").read_text(encoding="utf-8")))
            proposal = TrajectoryProposal.from_dict(json.loads((trial / "raw_proposal.json").read_text(encoding="utf-8")))
            sample = register_teacher_sample(
                sample_id=f"{task.task_id}__{trial.name}", task=task,
                controls_free=proposal.free_controls, success_report=trial / "evaluation" / "report.json",
                trace_path=Path(result["trace_path"]), output_dir=output_dir / "samples" / f"sample_{index:04d}",
                source_files=[trial / "task.json", trial / "raw_proposal.json",
                              trial / "planning_identity.json", trial / "proposal_gate.json"])
            samples.append(sample)
            record["registered_successful_teacher"] = True
            record["sample_id"] = sample["sample_id"]
        attempts.append(record)
    write_teacher_manifest(samples, output_dir / "manifest.json")
    (output_dir / "all_attempts.json").write_text(json.dumps({
        "attempts": attempts, "failure_evidence_is_retained": True,
        "failed_attempts_are_training_labels": False}, indent=2)+"\n", encoding="utf-8")
    return {"attempt_count": len(attempts), "successful_teacher_count": len(samples),
            "manifest": (output_dir / "manifest.json").resolve().as_posix()}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--trials", type=Path, nargs="+", required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    print(json.dumps(collect(a.trials, a.output)), flush=True)
