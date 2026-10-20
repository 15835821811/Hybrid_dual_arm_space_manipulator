import time
import numpy as np
import torch
from ff_prior.storage import read_json, digest, file_digest
from ff_prior.diffusion import weight_hash
from ff_prior_latent.learning import Codec
from ff_prior_origin_learn_bounded.learning import save_torch as atomic_save_torch
from ff_prior_origin_learn_bounded.distribution import CoverageSampler
from ff_prior_paired_objectives.objectives import terms
from ff_prior_pilot_train.storage import atomic_json
from .models import make, parameter_counts
from .protocol import P, CONTROL


def save_torch(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    return atomic_save_torch(path, payload)


def loss(model, data, G, indices, t, noise):
    slots = [data["lookup"][int(i)] for i in indices]
    v, local, _ = terms(model, data["x"][indices], data["c"][indices],
                         t, noise, data["alpha"], G[slots])
    return v + .05 * local, v, local


def sanity(out, data, G):
    records = []
    idx = data["tr"][:16]
    for kind in ("mlp", "transformer"):
        model = make(kind, P["diagnostic_seed"])
        before = weight_hash(model)
        opt = torch.optim.AdamW(model.parameters(), lr=P["lr"])
        rng = torch.Generator().manual_seed(P["diagnostic_seed"])
        for j in range(2):
            t = torch.randint(0, 1000, (16,), generator=rng)
            noise = torch.randn((16, 102), generator=rng)
            value, _, _ = loss(model, data, G, idx, t, noise)
            opt.zero_grad()
            value.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            if not torch.isfinite(value) or not torch.isfinite(norm):
                raise ValueError("nonfinite TRAIN numerical sanity")
            opt.step()
        clone = make(kind)
        clone.load_state_dict(model.state_dict())
        with torch.no_grad():
            a = model(data["x"][idx], t, data["c"][idx])
            b = clone(data["x"][idx], t, data["c"][idx])
        if not torch.equal(a, b) or weight_hash(model) == before:
            raise ValueError("TRAIN reload/update sanity failed")
        records.append({"kind": kind, "steps": 2, "finite": True, "reload_exact": True,
                        "last_loss": float(value.detach()), "weights_retained": False})
    counts = parameter_counts()
    if max(counts.values()) / min(counts.values()) > P["parameter_ratio_max"]:
        raise ValueError("parameter mismatch")
    result = {"status": "PASS", "records": records, "parameter_counts": counts,
              "training_indices": idx.tolist(), "physical_pass_claim": False}
    atomic_json(out / "numerical_sanity.json", result)
    return result


def load(directory):
    report = read_json(directory / "training.json")
    saved = torch.load(directory / "model.pt", map_location="cpu", weights_only=False)
    if file_digest(directory / "model.pt") != report["checkpoint_sha256"] or saved["metadata"] != report["metadata"]:
        raise ValueError("model identity changed")
    model = make(report["kind"])
    model.load_state_dict(saved["weights"])
    codec = Codec(read_json(directory / "codec.json"))
    if digest(codec.state) != report["metadata"]["codec_hash"] or weight_hash(model) != report["trained_weight_hash"]:
        raise ValueError("reload differs")
    model.eval()
    return model, codec


def train(out, kind, codec, arrays, G, data, deadline, progress):
    directory = out / "learning" / kind
    directory.mkdir(parents=True, exist_ok=True)
    meta = {"kind": kind, "protocol": P, "codec_hash": digest(codec.state),
            "data_hash": read_json(CONTROL / "learning/goal_loss/training.json")["metadata"]["data_hash"],
            "parameter_count": parameter_counts()[kind], "neural_warm_start": False}
    if (directory / "training.json").exists():
        report = read_json(directory / "training.json")
        if report["metadata"] != meta:
            raise ValueError("training identity changed")
        load(directory)
        return report
    model = make(kind)
    initial_hash = weight_hash(model)
    opt = torch.optim.AdamW(model.parameters(), lr=P["lr"], weight_decay=P["weight_decay"])
    sampler = CoverageSampler(data["keys"], P["training_seed"] + 2)
    rng = torch.Generator().manual_seed(P["training_seed"])
    counts = np.zeros(len(data["tr"]), dtype=np.int64)
    segments, history, start, previous_wall = [], [], 0, 0.
    jp = directory / "journal.json"
    checkpoint = directory / "progress.pt"
    if checkpoint.exists():
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        journal = read_json(jp)
        if saved["metadata"] != meta or journal["status"] != "COMMITTED" or journal["updates"] != saved["updates"]:
            raise ValueError("unknown training work must not be replayed")
        model.load_state_dict(saved["weights"])
        opt.load_state_dict(saved["optimizer"])
        sampler = CoverageSampler(data["keys"], P["training_seed"] + 2, saved["sampler"])
        rng.set_state(saved["rng"])
        counts, segments, history = saved["counts"], saved["segments"], saved["history"]
        start, previous_wall = saved["updates"], saved["wall_s"]
    elif jp.exists():
        raise ValueError("journal without checkpoint: unknown work")
    else:
        atomic_json(directory / "codec.json", codec.state)
        save_torch(directory / "initial.pt", {"weights": model.state_dict(), "metadata": meta})
    started = time.perf_counter()
    # Expected random sequence is independent of either architecture's RNG usage.
    expected = read_json(CONTROL / "learning/goal_loss/training.json")["segments"]
    for begin in range(start, P["updates"], 20):
        atomic_json(jp, {"status": "PENDING", "start_update": begin + 1, "end_update": begin + 20})
        proofs = []
        for step in range(begin, begin + 20):
            if time.time() >= deadline:
                raise RuntimeError("TRAIN_TIME_LIMIT")
            batch = sampler.take(64)
            indices = data["tr"][batch]
            np.add.at(counts, batch, 1)
            t = torch.randint(0, 1000, (64,), generator=rng)
            noise = torch.randn((64, 102), generator=rng)
            proof_dir = directory / "updates" / f"{step+1:04d}"
            if step + 1 in P["audit_updates"]:
                save_torch(proof_dir / "before.pt", {"weights": model.state_dict(), "optimizer": opt.state_dict(),
                           "indices": indices, "t": t, "noise": noise})
            model.train()
            value, vmse, local = loss(model, data, G, indices, t, noise)
            if not torch.isfinite(value):
                raise ValueError("nonfinite training loss")
            opt.zero_grad()
            value.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            if not torch.isfinite(norm):
                raise ValueError("nonfinite training gradient")
            opt.step()
            if step + 1 in P["audit_updates"]:
                save_torch(proof_dir / "after.pt", {"weights": model.state_dict(), "optimizer": opt.state_dict()})
            proofs.append(digest({"update": step+1, "batch": batch, "t": t.numpy(), "noise": noise.numpy()}))
            if (step + 1) % 100 == 0:
                history.append({"update": step+1, "loss": float(value.detach()),
                                "v_mse": float(vmse.detach()), "local_G": float(local.detach())})
                progress("M1_training", model=kind, updates=step+1, total=6000)
        segment = {"start_update": begin+1, "end_update": begin+20,
                   "draws": 1280, "batch_rng_digest": digest(proofs)}
        if segment != expected[begin//20]:
            raise ValueError("frozen training randomness mismatch")
        segments.append(segment)
        save_torch(checkpoint, {"metadata": meta, "weights": model.state_dict(),
                   "optimizer": opt.state_dict(), "sampler": sampler.state(), "rng": rng.get_state(),
                   "updates": begin+20, "counts": counts, "segments": segments, "history": history,
                   "wall_s": previous_wall+time.perf_counter()-started})
        atomic_json(jp, {"status": "COMMITTED", "updates": begin+20})
    if not np.all(counts > 0) or weight_hash(model) == initial_hash:
        raise ValueError("training consumption/update gate failed")
    save_torch(directory / "model.pt", {"metadata": meta, "weights": model.state_dict()})
    report = {"status": "PASS", "kind": kind, "metadata": meta, "updates": 6000,
              "actual_TRAIN_draws": int(counts.sum()), "TRAIN_rows_seen": int(np.sum(counts > 0)),
              "train_draw_counts": dict(zip([arrays[3][i] for i in data["tr"]], counts.tolist())),
              "segments": segments, "history": history, "initial_weight_hash": initial_hash,
              "trained_weight_hash": weight_hash(model), "checkpoint_sha256": file_digest(directory / "model.pt"),
              "VAL_gradient_updates": 0, "VAL_checkpoint_selection": False,
              "world_events": 0, "wall_s": previous_wall+time.perf_counter()-started}
    atomic_json(directory / "training.json", report)
    load(directory)
    return report
