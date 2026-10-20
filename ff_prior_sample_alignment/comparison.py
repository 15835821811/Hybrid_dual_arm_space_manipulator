from ff_prior.storage import read_json, digest
from ff_prior_pilot_train.storage import atomic_json
from ff_prior.guidance import public_goal_initializer
from ff_prior_delivery_guidance.sampling import candidate, LatentProjector
from ff_prior_endpoint_jacobian.comparison import unguided_sample
from ff_prior_feasible.validation import validate
from ff_prior_generalization.experiment import execute
from .protocol import P, METHODS


def unguided(out, bundles, arrays, rows, data, cache, profile, ledger, budget, progress):
    for sj, seed in enumerate(P["val_seeds"]):
        for j, i in enumerate(data["va"]):
            row, base = rows[i], arrays[1][i]
            task = row["task"]
            rr, ii, cc, _ = cache.get(row)
            for kind in METHODS:
                key = f"unguided_{sj}_{j}_{kind}"
                if key in ledger.data["completed"]:
                    continue
                ledger.begin(key, "unguided", P["wall_budget_s"], 192)
                budget.reserve(3)
                progress("M2_unguided", model=kind, seed=sj+1, target=j+1, total=48)
                before = rr.calls.copy()
                raw, steps = unguided_sample(bundles[kind], base, task, seed+j*997)
                projector = LatentProjector(task, rr, ii, cc, profile, budget.deadline,
                                           codec=bundles[kind][1], base=base)
                z, prep = projector.prepare(raw)
                result = validate(z, task, rr, ii, cc, profile, budget.deadline)
                result.update(method=kind, sample_id=arrays[3][i], seed=seed+j*997,
                              seed_group=sj, steps=steps, raw_Z=raw, prepared_Z=z,
                              preparation=prep, physics_guidance_calls=0, task_hash=task["task_hash"],
                              endpoint_scope="prepared unguided full16 DDIM; raw also saved",
                              calls={k: rr.calls[k]-before[k] for k in before})
                atomic_json(out / "unguided" / str(sj) / kind / f"{j:02d}.json", result)
                ledger.finish({"status": "DONE", "eligible": result["eligible"]}, result["calls"])


def dev(out, bundles, entries, cache, profile, ledger, budget, progress):
    for sj, seed in enumerate(P["sampling_seeds"]):
        for ti, e in enumerate(entries):
            task, sid = e["task"], e["sample_id"]
            rr, ii, cc, _ = cache.get(e)
            base = public_goal_initializer(task, rr, ii)
            for kind in METHODS:
                records = []
                for k in range(5):
                    path = out / "candidates" / str(sj) / sid / kind / f"{k:02d}.json"
                    key = f"candidate_{sj}_{sid}_{kind}_{k}"
                    if key not in ledger.data["completed"]:
                        ledger.begin(key, "candidate", P["wall_budget_s"], 360)
                        budget.reserve(113)
                        progress("M3_DEV_candidates", model=kind, seed=sj+1, task=ti+1, candidate=k+1)
                        r = candidate("late4_diffusion", k, task, base, seed+ti*101+k*997,
                                      bundles[kind], rr, ii, cc, profile, budget.deadline)
                        r.update(method=kind, sampler="late4_diffusion", sample_id=sid,
                                 seed_group=sj, split=e["split"])
                        atomic_json(path, r)
                        ledger.finish({"status": r["status"], "eligible": r["eligible"]}, r["model_calls"])
                    records.append(read_json(path))
                chosen = min((r for r in records if r["eligible"]),
                             key=lambda r: (r["score"], r["candidate_id"]), default=None)
                key = f"execution_{sj}_{sid}_{kind}"
                if key in ledger.data["completed"]:
                    continue
                ledger.begin(key, "execution", P["wall_budget_s"], 72)
                budget.check()
                directory = out / "executions" / str(sj) / sid / kind
                atomic_json(directory / "selection.json", {
                    "selected_id": chosen["candidate_id"] if chosen else None,
                    "selection_before_execution": True, "task_hash": task["task_hash"],
                    "candidate_pool_digest": digest(records)})
                before = rr.calls.copy()
                progress("M3_DEV_native_replay", model=kind, seed=sj+1, task=ti+1)
                if chosen:
                    result = execute(directory, task, chosen["Z"], rr, ii, cc, profile, budget.deadline)
                else:
                    result = {"status": "NO_CANDIDATE"}
                    atomic_json(directory / "outcome.json", result)
                ledger.finish(result, {k: rr.calls[k]-before[k] for k in before})
