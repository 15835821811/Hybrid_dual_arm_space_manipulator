import time
import numpy as np
import torch
from ff_prior.storage import read_json, digest, file_digest
from ff_prior.diffusion import weight_hash
from ff_prior_latent.learning import Codec, LatentDenoiser
from ff_prior_expand256_train.learning import load as warm_load
from ff_prior_origin_learn_bounded.distribution import CoverageSampler
from ff_prior_backbone_v2.learning import save_torch, loss
from ff_prior_paired_objectives.objectives import vector_gradient
from ff_prior_endpoint_jacobian import endpoint
from ff_prior_pilot_train.storage import atomic_json, atomic_npz
from .objectives import rollout, prepared_output, teacher_loss, combine, assign
from .protocol import P, CONTROL


def world_gradient(model, dest, i, seed, codec, arrays, rows, data, cache, profile, budget):
    dest.mkdir(parents=True, exist_ok=True)
    clean = rollout(model, data["c"][i:i+1], [seed])[0]
    rr, ii, cc, _ = cache.get(rows[i])
    prepared, prep = prepared_output(clean, codec, data["centers"][i], arrays[1][i],
                                     rows[i]["task"], rr, ii, cc, profile, budget.deadline)
    budget.reserve(212)
    value, g, proof, raw = endpoint.gradient(prepared.detach().numpy(), codec, data["centers"][i],
                                            arrays[1][i], rows[i]["task"], rr, ii, budget.deadline)
    atomic_npz(dest / "gradient.npz", **raw)
    proof.update(sample_id=arrays[3][i], index=int(i), seed=int(seed),
                 full_sampling_steps=16, preparation=prep, raw_clean=clean.detach().numpy(),
                 input_weight_hash=weight_hash(model), gradient_file_sha256=file_digest(dest / "gradient.npz"))
    atomic_json(dest / "gradient.json", proof)
    if proof["status"] != "PASS":
        raise ValueError("FULL_ROLLOUT_WORLD_NUMERICAL_CHECK_FAILED; no silent fallback")
    grad = vector_gradient(endpoint.attach_gradient(prepared, value, g), model).detach()
    return grad, {"loss": value, "sample_id": arrays[3][i],
                  "calls": proof["calls"], "gradient_status": proof["status"]}


def sanity(out, selection, codec, arrays, rows, data, cache, profile, ledger, budget, progress):
    model, _ = warm_load(CONTROL / "learning/goal_loss")
    original = weight_hash(model)
    proofs = []
    from ff_prior_endpoint_jacobian.comparison import unguided_sample
    for j, e in enumerate(selection["diagnostic"]):
        key = f"diagnostic_{j}"
        if key in ledger.data["completed"]:
            proofs.append(read_json(out / "diagnostic" / str(j) / "gradient.json"))
            continue
        ledger.begin(key, "TRAIN_gradient_sanity", P["wall_budget_s"], 2)
        progress("M0_full16_TRAIN_gradient", case=j+1, total=2)
        i, seed = data["ids"][e["sample_id"]], e["seed"]
        dest = out / "diagnostic" / str(j)
        raw, _ = unguided_sample((model, codec), arrays[1][i], rows[i]["task"], seed)
        x = rollout(model, data["c"][i:i+1], [seed])[0]
        rebuilt = codec.decode(codec.coefficients(x.detach().numpy(), data["centers"][i]), arrays[1][i])
        assert np.array_equal(raw, rebuilt)
        g, proof = world_gradient(model, dest, i, seed, codec, arrays, rows, data, cache, profile, budget)
        assert torch.isfinite(g).all() and weight_hash(model) == original
        ledger.finish({"status": "PASS", "updates_retained": 0}, proof["calls"])
        proofs.append(read_json(dest / "gradient.json"))
    result = {"status": "PASS", "full102_gradients": len(proofs), "updates_retained": 0,
              "full16_sampler_matches_deployment": True, "weight_hash_unchanged": weight_hash(model) == original}
    atomic_json(out / "numerical_sanity.json", result)
    return result


def load(directory):
    report = read_json(directory / "training.json")
    saved = torch.load(directory / "model.pt", map_location="cpu", weights_only=False)
    assert file_digest(directory / "model.pt") == report["checkpoint_sha256"]
    assert saved["metadata"] == report["metadata"]
    codec = Codec(read_json(directory / "codec.json"))
    model = LatentDenoiser(102, 65)
    model.load_state_dict(saved["weights"])
    assert weight_hash(model) == report["trained_weight_hash"]
    assert digest(codec.state) == report["metadata"]["codec_hash"]
    model.eval()
    return model, codec


def train(out, kind, selection, codec, arrays, rows, G, data, cache, profile, ledger, budget, progress):
    directory = out / "learning" / kind
    directory.mkdir(parents=True, exist_ok=True)
    if (directory / "training.json").exists():
        load(directory)
        return read_json(directory / "training.json")
    model, _ = warm_load(CONTROL / "learning/goal_loss")
    original = weight_hash(model)
    meta = {"kind": kind, "protocol": P, "codec_hash": digest(codec.state),
            "warm_checkpoint_sha256": file_digest(CONTROL / "learning/goal_loss/model.pt")}
    optimizer = torch.optim.AdamW(model.parameters(), lr=P["lr"], weight_decay=P["weight_decay"])
    sampler = CoverageSampler(data["keys"], P["training_seed"] + 2)
    rng = torch.Generator().manual_seed(P["training_seed"])
    counts = np.zeros(len(data["tr"]), dtype=np.int64)
    events, segments, history, start, previous_wall = [], [], [], 0, 0.
    checkpoint = directory / "progress.pt"
    if checkpoint.exists():
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        assert read_json(directory / "journal.json")["status"] == "COMMITTED" and saved["metadata"] == meta
        model.load_state_dict(saved["weights"])
        optimizer.load_state_dict(saved["optimizer"])
        sampler = CoverageSampler(data["keys"], P["training_seed"] + 2, saved["sampler"])
        rng.set_state(saved["rng"])
        counts, events, segments, history, start, previous_wall = (
            saved[k] for k in ("counts", "events", "segments", "history", "updates", "wall_s"))
    else:
        if (directory / "journal.json").exists():
            raise ValueError("unknown training segment")
        save_torch(directory / "initial.pt", {"weights": model.state_dict(), "metadata": meta})
        atomic_json(directory / "codec.json", codec.state)
    started = time.perf_counter()
    expected = read_json(CONTROL / "learning/goal_loss/training.json")["segments"]
    for begin in range(start, P["updates"], 20):
        key = f"train_{kind}_{begin//20}"
        ledger.begin(key, "training_segment", P["wall_budget_s"], 120)
        atomic_json(directory / "journal.json", {"status": "PENDING", "start_update": begin+1})
        proofs = []
        calls = {}
        for step in range(begin, begin+20):
            budget.check()
            batch = sampler.take(64)
            indices = data["tr"][batch]
            np.add.at(counts, batch, 1)
            t = torch.randint(0, 1000, (64,), generator=rng)
            noise = torch.randn((64, 102), generator=rng)
            value, _, _ = loss(model, data, G, indices, t, noise)
            base = vector_gradient(value, model).detach()
            teacher = world = None
            extras = {}
            full_indices = indices[:P["rollout_batch"]]
            full_seeds = [P["rollout_seed"] + step*101 + k for k in range(P["rollout_batch"])]
            event = kind == "aligned" and (step+1) % P["world_every"] == 0
            save = step+1 in P["audit_updates"] or event
            dest = directory / "updates" / f"{step+1:04d}"
            if save:
                save_torch(dest / "before.pt", {"weights": model.state_dict(), "optimizer": optimizer.state_dict(),
                           "indices": indices, "t": t, "noise": noise, "full_indices": full_indices,
                           "full_seeds": full_seeds})
            if kind == "aligned":
                tv, _ = teacher_loss(model, data, G, full_indices, full_seeds)
                teacher = vector_gradient(tv, model).detach()
                extras["full16_teacher_loss"] = float(tv.detach())
            if event:
                e = selection["world_events"][(step+1)//20-1]
                progress("M1_full16_world_event", update=step+1, total=P["updates"])
                world, record = world_gradient(model, dest, data["ids"][e["sample_id"]], e["seed"],
                                               codec, arrays, rows, data, cache, profile, budget)
                for k, v in record["calls"].items():
                    calls[k] = calls.get(k, 0) + v
                events.append(dict(record, update=step+1))
                extras["world"] = record
            mixed, scales = combine(base, teacher, world)
            optimizer.zero_grad()
            assign(model, mixed)
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            if not torch.isfinite(norm):
                raise ValueError("nonfinite gradient")
            optimizer.step()
            if save:
                save_torch(dest / "after.pt", {"weights": model.state_dict(), "optimizer": optimizer.state_dict()})
                atomic_npz(dest / "gradients.npz", base=base.numpy(), teacher=teacher.numpy() if teacher is not None else np.zeros(0),
                           world=world.numpy() if world is not None else np.zeros(0), mixed=mixed.numpy())
                atomic_json(dest / "update.json", {"scales": scales, "base_norm": float(base.norm()),
                            "clip_norm": float(norm), "extras": extras, "optimizer_steps": step+1})
            proofs.append(digest({"update": step+1, "batch": batch, "t": t.numpy(), "noise": noise.numpy()}))
            if (step+1) % 100 == 0:
                history.append(dict(update=step+1, base_loss=float(value.detach()), **extras))
                progress("M1_training", method=kind, updates=step+1, total=P["updates"])
        segment = {"start_update": begin+1, "end_update": begin+20, "draws": 1280, "batch_rng_digest": digest(proofs)}
        assert segment == expected[begin//20]
        segments.append(segment)
        save_torch(checkpoint, {"metadata": meta, "weights": model.state_dict(), "optimizer": optimizer.state_dict(),
                   "sampler": sampler.state(), "rng": rng.get_state(), "counts": counts, "events": events,
                   "segments": segments, "history": history, "updates": begin+20,
                   "wall_s": previous_wall+time.perf_counter()-started})
        atomic_json(directory / "journal.json", {"status": "COMMITTED", "updates": begin+20})
        ledger.finish({"status": "PASS"}, calls)
    assert np.all(counts > 0) and weight_hash(model) != original
    assert len(events) == (60 if kind == "aligned" else 0)
    save_torch(directory / "model.pt", {"metadata": meta, "weights": model.state_dict()})
    result = {"status": "PASS", "kind": kind, "metadata": meta, "updates": P["updates"],
              "TRAIN_rows_seen": int(sum(counts > 0)), "actual_TRAIN_draws": int(counts.sum()),
              "train_draw_counts": dict(zip([arrays[3][i] for i in data["tr"]], counts.tolist())),
              "segments": segments, "history": history, "world_events": events,
              "full16_teacher_batches": P["updates"] if kind == "aligned" else 0,
              "initial_weight_hash": original, "trained_weight_hash": weight_hash(model),
              "checkpoint_sha256": file_digest(directory / "model.pt"),
              "VAL_gradient_updates": 0, "VAL_checkpoint_selection": False,
              "wall_s": previous_wall+time.perf_counter()-started}
    atomic_json(directory / "training.json", result)
    load(directory)
    return result
