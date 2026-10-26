"""Execute pre-registered tasks with independent task and model sampling seeds."""
import os
for _name in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ[_name] = "1"

import argparse
import hashlib
import json
from pathlib import Path

from .environment import Task
from .pipeline import run_episode, write_json
from .task_domain import CONFIG, profile_hash


def snapshot(output, checkpoint=None):
    sources = sorted(Path(__file__).parent.glob("*.py")) + [CONFIG]
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    if checkpoint:
        hashes["checkpoint"] = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    version_file = output/"code_version.json"
    if version_file.exists():
        if json.loads(version_file.read_text(encoding="utf-8")) != hashes:
            raise ValueError("Code/checkpoint changed; use a new evaluation directory")
    else:
        write_json(version_file, hashes)
        folder = output/"source_snapshot"
        folder.mkdir(exist_ok=True)
        for p in sources:
            (folder/p.name).write_bytes(p.read_bytes())
    return hashes


def evaluate(manifest_path, output, names, mode, checkpoint=None, device="cpu",
             sampling_seeds=(73, 137, 211), steps=32, candidates=4, stop_on_failure=False,
             require_test=False, feedback=1., paper=False, guidance=True):
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["profile_sha256"] != profile_hash():
        raise ValueError("Manifest profile mismatch")
    entries = {e["id"]: e for e in manifest["tasks"]}
    if not names:
        names = list(entries)
    if any(name not in entries for name in names) or len(set(names)) != len(names):
        raise ValueError("Unknown or duplicate task IDs")
    versions = snapshot(output, checkpoint)
    protocol = {"manifest": str(manifest_path.resolve()), "episodes": names, "mode": mode,
                "sampling_seeds": list(sampling_seeds) if mode == "diffusion" else [None],
                "steps": steps, "candidates": candidates, "require_test": require_test,
                "feedback": feedback, "profile_sha256": profile_hash(), "paper":paper, "guidance":guidance}
    if (output/"protocol.json").exists():
        if json.loads((output/"protocol.json").read_text(encoding="utf-8")) != protocol:
            raise ValueError("Evaluation protocol changed")
    else:
        write_json(output/"protocol.json", protocol)
    results = []
    for name in names:
        entry = entries[name]
        if require_test and entry.get("split") != "test":
            raise ValueError("Final test evaluation requires a pre-registered test task")
        for sampling_seed in protocol["sampling_seeds"]:
            folder = output/name/(f"sample_{sampling_seed}" if sampling_seed is not None else mode)
            if (folder/"result.json").exists():
                result = json.loads((folder/"result.json").read_text(encoding="utf-8"))
            else:
                policy = None
                if mode == "diffusion":
                    if paper:
                        from .paper_model import PaperPolicy
                        policy = PaperPolicy(checkpoint, device, sampling_seed, candidates, steps, guidance)
                    else:
                        from .model import DiffusionPolicy
                        policy = DiffusionPolicy(checkpoint, device, sampling_seed, candidates, steps)
                    if require_test and entry["task"]["seed"] in policy.train_seeds + policy.val_seeds:
                        raise ValueError("Test task leaked into training/validation")
                result, _, _ = run_episode(Task(**entry["task"]), folder, policy, mode,
                                            feedback, progress=True)
            passed = result["meets_stage_tracking"]
            results.append({"id": name, "sampling_seed": sampling_seed, "passed": passed,
                            "result_path": str(folder/"result.json"), "result": result})
            write_json(output/"summary.json", {"attempted": len(results),
                "passed": sum(r["passed"] for r in results), "episodes": results,
                "profile_sha256": profile_hash(), "source_hashes": versions})
            if stop_on_failure and not passed:
                raise RuntimeError(f"Closed-loop gate failed: {name}, sampling seed {sampling_seed}")
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("tracking_output/local10mm_v1/pilot_manifest.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes", nargs="+")
    parser.add_argument("--mode", choices=("teacher", "feedback-only", "diffusion"), default="diffusion")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--sampling-seeds", nargs="+", type=int, default=[73, 137, 211])
    parser.add_argument("--sampling-steps", type=int, default=32)
    parser.add_argument("--candidates", type=int, default=4)
    parser.add_argument("--feedback", type=float, default=1.)
    parser.add_argument("--stop-on-failure", action="store_true")
    parser.add_argument("--require-test", action="store_true")
    parser.add_argument("--paper", action="store_true")
    parser.add_argument("--no-guidance", action="store_true")
    args = parser.parse_args()
    if args.mode == "diffusion" and args.checkpoint is None:
        parser.error("diffusion requires --checkpoint")
    evaluate(args.manifest, args.output, args.episodes, args.mode, args.checkpoint, args.device,
             args.sampling_seeds, args.sampling_steps, args.candidates, args.stop_on_failure,
             args.require_test, args.feedback, args.paper, not args.no_guidance)


if __name__ == "__main__":
    main()
