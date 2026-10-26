"""Small conditional diffusion; the complete normalization lives in the checkpoint."""
from __future__ import annotations

import copy
import json
import math
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn


class AdaBlock(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.norm = nn.LayerNorm(width, elementwise_affine=False)
        self.mod = nn.Linear(width, 2*width)
        self.net = nn.Sequential(nn.Linear(width, 2*width), nn.SiLU(), nn.Linear(2*width, width))

    def forward(self, x, cond):
        scale, shift = self.mod(cond).chunk(2, dim=-1)
        return x + self.net(self.norm(x)*(1+scale) + shift) / math.sqrt(3)


class Denoiser(nn.Module):
    def __init__(self, condition_dim=135, width=192, output_dim=34):
        super().__init__()
        self.input = nn.Linear(output_dim, width)
        self.condition = nn.Sequential(nn.Linear(condition_dim, width), nn.SiLU(), nn.Linear(width, width))
        self.time = nn.Sequential(nn.Linear(64, width), nn.SiLU(), nn.Linear(width, width))
        self.blocks = nn.ModuleList([AdaBlock(width) for _ in range(3)])
        self.output = nn.Linear(width, output_dim)
        self.register_buffer("frequency", torch.exp(torch.linspace(0, -math.log(10000), 32)))

    def forward(self, x, t, condition):
        phase = t[:, None]*1000*self.frequency[None]
        c = self.condition(condition) + self.time(torch.cat([phase.sin(), phase.cos()], -1))
        h = self.input(x)
        for block in self.blocks:
            h = block(h, c)
        return self.output(h)


def schedule(t):
    # Continuous cosine v-prediction avoids dividing by near-zero alpha.
    return torch.cos(t*math.pi/2)[:, None], torch.sin(t*math.pi/2)[:, None]


@torch.no_grad()
def sample(model, condition, generator, steps=32, on_step=None):
    x = torch.randn((len(condition), model.output.out_features), device=condition.device, generator=generator)
    times = torch.linspace(1., 0., steps+1, device=condition.device)
    if on_step is not None:
        on_step(0, steps, x.detach().cpu().numpy())
    for k in range(steps):
        t = times[k].expand(len(x))
        a, s = schedule(t)
        v = model(x, t, condition)
        clean = a*x-s*v
        eps = s*x+a*v
        an, sn = schedule(times[k+1].expand(len(x)))
        x = an*clean + sn*eps
        if on_step is not None:
            on_step(k+1, steps, x.detach().cpu().numpy())
    return x


def train(dataset_path, output_path, steps=4000, seed=42, device="cpu"):
    if steps < 1:
        raise ValueError("steps must be positive")
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    np.random.seed(seed)
    with np.load(dataset_path, allow_pickle=False) as data:
        train_mask, val_mask = data["split"] == 0, data["split"] == 1
        if not train_mask.any() or not val_mask.any():
            raise ValueError("Need accepted training and validation episodes")
        ctrain = data["condition"][train_mask]
        cv = data["condition"][val_mask]
        xtrain = data["free"][train_mask].reshape(len(ctrain), -1)
        xv = data["free"][val_mask].reshape(len(cv), -1)
        metadata = json.loads(data["metadata"].item()) if "metadata" in data else {}
        train_seeds = np.unique(data["seed"][train_mask]).tolist()
        val_seeds = np.unique(data["seed"][val_mask]).tolist()
    # Statistics come from TRAIN only. Floors prevent nearly constant inputs
    # from amplifying numerical noise into a spurious out-of-domain condition.
    cm, cs = ctrain.mean(0), np.maximum(ctrain.std(0), .01)
    xm, xs = xtrain.mean(0), np.maximum(xtrain.std(0), .02)
    c = torch.as_tensor((ctrain-cm)/cs, device=device)
    x = torch.as_tensor((xtrain-xm)/xs, device=device)
    cval = torch.as_tensor((cv-cm)/cs, device=device)
    xval = torch.as_tensor((xv-xm)/xs, device=device)
    model = Denoiser(ctrain.shape[1], output_dim=xtrain.shape[1]).to(device)
    ema = copy.deepcopy(model).eval()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-5)
    generator = torch.Generator(device=device).manual_seed(seed+10)
    fixed_generator = torch.Generator(device=device).manual_seed(seed+20)
    val_ids = torch.linspace(0, len(xval)-1, min(256, len(xval)), device=device).long()
    vn = torch.randn(xval[val_ids].shape, device=device, generator=fixed_generator)
    vt = torch.rand((len(val_ids),), device=device, generator=fixed_generator)
    history, best, best_state = [], float("inf"), None
    started = time.perf_counter()
    for step in range(1, steps+1):
        ids = torch.randint(len(x), (min(128, len(x)),), generator=generator, device=device)
        clean, cond = x[ids], c[ids]
        t = torch.rand((len(ids),), device=device, generator=generator)
        noise = torch.randn(clean.shape, device=device, generator=generator)
        a, s = schedule(t)
        noisy, target = a*clean+s*noise, a*noise-s*clean
        prediction = model(noisy, t, cond)
        loss = (prediction-target).square().mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()
        for group in optimizer.param_groups:
            group["lr"] = 3e-4*(.1+.9*.5*(1+math.cos(math.pi*step/steps)))
        with torch.no_grad():
            decay = min(.995, (step+1)/(step+10))
            for pe, p in zip(ema.parameters(), model.parameters()):
                pe.lerp_(p, 1-decay)
        if step == 1 or step % 200 == 0 or step == steps:
            with torch.no_grad():
                a, s = schedule(vt)
                vl = (ema(a*xval[val_ids]+s*vn, vt, cval[val_ids]) -
                      (a*vn-s*xval[val_ids])).square().mean().item()
            item = {"step": step, "train_v_mse": loss.item(), "validation_v_mse": vl,
                    "elapsed_s": time.perf_counter()-started}
            history.append(item)
            print(item, flush=True)
            if vl < best:
                best, best_state = vl, copy.deepcopy(ema.state_dict())
    ema.load_state_dict(best_state)
    # Full denoising diagnostics are mandatory; denoising loss is not task success.
    with torch.no_grad():
        diagnostic_generator = torch.Generator(device=device).manual_seed(seed+30)
        generated = sample(ema, cval[val_ids], diagnostic_generator)
        rmse = torch.sqrt(((generated-xval[val_ids])*torch.tensor(xs, device=device)).square().mean()).item()
    checkpoint = {"version": 2 if metadata else 1, "metadata": metadata,
                  "output_dim": xtrain.shape[1], "state_dict": best_state, "condition_dim": c.shape[1],
                  "condition_mean": cm, "condition_scale": cs, "output_mean": xm,
                  "output_scale": xs, "train_seeds": train_seeds, "validation_seeds": val_seeds,
                  "training_seed": seed, "training_steps": steps, "history": history,
                  "full_sampling_validation_control_rmse_rad_per_s": rmse,
                  "dataset_path": str(Path(dataset_path).resolve())}
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, output_path)
    return {k: checkpoint[k] for k in ("training_steps", "train_seeds", "validation_seeds",
                                      "full_sampling_validation_control_rmse_rad_per_s", "history")}


class DiffusionPolicy:
    def __init__(self, path, device="cpu", seed=73, candidates=4, steps=32):
        torch.set_num_threads(1)
        # Local checkpoint created by this package, never load an untrusted pickle.
        ckpt = torch.load(path, map_location=device, weights_only=False)
        if ckpt["version"] not in (1,2):
            raise ValueError("Unsupported checkpoint version")
        self.version = ckpt["version"]
        self.output_dim = ckpt.get("output_dim",34)
        self.metadata = dict(ckpt.get("metadata",{}),sampling_seed=seed,sampling_steps=steps,candidates=candidates)
        self.task_domain = self.metadata.get("task_domain","legacy-wide-v1")
        if self.version == 2:
            from .spline import SplineChunk
            self.chunk_type = SplineChunk
            self.horizon = self.metadata["horizon"]
        self.model = Denoiser(ckpt["condition_dim"],output_dim=self.output_dim).to(device).eval()
        self.model.load_state_dict(ckpt["state_dict"])
        self.device, self.count, self.steps = device, candidates, steps
        self.generator = torch.Generator(device=device).manual_seed(seed)
        self.cm, self.cs = ckpt["condition_mean"], ckpt["condition_scale"]
        self.xm, self.xs = ckpt["output_mean"], ckpt["output_scale"]
        self.train_seeds, self.val_seeds = ckpt["train_seeds"], ckpt["validation_seeds"]

    def predict(self, condition, on_step=None, horizon=None, time=None):
        if self.version == 2:
            condition = np.r_[condition,self.horizon if horizon is None else horizon].astype(np.float32)
        c = torch.tensor(np.repeat(((condition-self.cm)/self.cs)[None], self.count, axis=0),
                         device=self.device)
        def callback(step, total, value):
            on_step(step, total, (value*self.xs+self.xm).reshape(-1, self.output_dim//17, 17))
        x = sample(self.model, c, self.generator, self.steps,
                   callback if on_step is not None else None).cpu().numpy()
        return (x*self.xs+self.xm).reshape(-1, self.output_dim//17, 17)
