"""Freeze validation-selected deployment, then run every registered new test."""
import os
for _k in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ[_k] = "1"
import hashlib
from pathlib import Path
import subprocess
import sys
import time
from .round2_data import BASE, read, freeze
from .round2_train_all import process_alive
from .round2_runs import run
from .pipeline import write_json


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    pid = int((BASE/"validate.pid").read_text().strip())
    while True:
        path = BASE/"validation_stage.json"
        if path.exists() and read(path).get("status") == "complete":
            break
        if not process_alive(pid):
            raise RuntimeError("Validation stopped before completion; inspect logs")
        time.sleep(5)
    selected = read(path)["selected"]
    manifest = freeze(BASE)
    checkpoint = Path(selected["checkpoint"])
    options = dict(selected["options"], diagnostic=False, duration=None,
                   isolated_timing=True)
    deployment = {"checkpoint": str(checkpoint), **options,
        "checkpoint_sha256": sha(checkpoint),
        "dataset_sha256": sha(BASE/"data_500/dataset.npz"),
        "manifest_sha256": sha(BASE/"manifest.json"),
        "selected_by": "validation only; primary training seed 42",
        "test_used_at_freeze": False, "sampling_seeds": manifest["sampling_seeds"]}
    frozen = BASE/"test_config_frozen.json"
    if frozen.exists() and read(frozen) != deployment:
        raise RuntimeError("Final test configuration changed after freeze")
    write_json(frozen, deployment)
    write_json(BASE/"deployment.json", deployment)
    write_json(BASE/"final_stage.json", {"status": "new_model_test", "test_runs": 100})
    # No other GPU stage follows validation. One worker establishes latency;
    # all seeds/tasks stay in the denominator, including runtime failures.
    run(BASE, "test_final", "diffusion", checkpoint, None, "test",
        manifest["sampling_seeds"], 1, options)
    write_json(BASE/"final_stage.json", {"status": "teacher_test", "test_runs": 20})
    run(BASE, "test_teacher", "teacher", None, None, "test", [], 4,
        dict(options, isolated_timing=False))
    write_json(BASE/"final_stage.json", {"status": "old_model_test", "test_runs": 100})
    old_options = dict(options, steps=60, candidates=5, guidance=True,
        projection=True, scoring="legacy", continuity=0., isolated_timing=False)
    run(BASE, "test_old_model", "diffusion",
        Path("tracking_output/paper100_v1/train_a01/model.pt"), None, "test",
        manifest["sampling_seeds"], 4, old_options)
    subprocess.run([sys.executable, "-m", "tracking_diffusion.round2_report"], check=True)
    write_json(BASE/"final_stage.json", {"status": "complete", "test_runs": 220,
        "note": "Execution completion is separate from measured performance"})


if __name__ == "__main__":
    main()
