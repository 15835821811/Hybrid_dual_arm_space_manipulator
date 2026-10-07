"""Generate and preserve raw DDIM proposals before any gate or repair."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np

from v6_4.contracts import TrajectoryProposal
from v6_4.run_planning import load_task_suite


def generate(task, checkpoint, output, *, K=1, seed=64, device="auto"):
    import torch
    from v6_4.train_diffusion import load_sampler
    if K not in (1, 8):
        raise ValueError("predeclared candidate budgets are K=1 or K=8")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    started = time.perf_counter()
    sampler = load_sampler(Path(checkpoint), device=device)
    initialization_s = time.perf_counter() - started
    proposals, records = [], []
    started = time.perf_counter()
    for index in range(K):
        values, metadata = sampler.sample(task, K=1, seed=seed+index, ddim_steps=20)
        metadata.update(candidate_index=index, candidate_budget=K,
                        budget_sampling_strategy="independent seeds, serial DDIM20; candidate0 shared with K1")
        proposal = TrajectoryProposal.from_controls(task, values[0], origin="diffusion",
                                                    seed=seed+index, metadata=metadata)
        candidate = output / f"candidate_{index:03d}"
        candidate.mkdir()
        (candidate / "proposal.json").write_text(json.dumps(proposal.to_dict(), indent=2,
            allow_nan=False)+"\n", encoding="utf-8")
        np.save(candidate / "controls_free.npy", proposal.free_controls, allow_pickle=False)
        proposals.append(proposal)
        records.append({"index": index, "proposal_sha256": proposal.sha256(),
                        "seed": seed+index, "sampling_elapsed_s": metadata["sampling_elapsed_s"]})
    identity = {"task_id": task.task_id, "task_sha256": task.sha256(), "K": K,
                "checkpoint": str(Path(checkpoint).resolve()), "seed": seed,
                "initialization_s": initialization_s, "total_generation_wall_s": time.perf_counter()-started,
                "candidate0_shared_with_K1": True, "postprocessing": [], "simulation_selection": False,
                "candidates": records}
    (output / "generation.json").write_text(json.dumps(identity, indent=2)+"\n", encoding="utf-8")
    return proposals, identity


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tasks", type=Path, required=True)
    p.add_argument("--task-id", required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--K", type=int, choices=(1, 8), default=1)
    p.add_argument("--seed", type=int, default=64)
    p.add_argument("--device", default="auto")
    a = p.parse_args()
    task = next(t for t in load_task_suite(a.tasks) if t.task_id == a.task_id)
    _, identity = generate(task, a.checkpoint, a.output, K=a.K, seed=a.seed, device=a.device)
    print(json.dumps(identity), flush=True)
