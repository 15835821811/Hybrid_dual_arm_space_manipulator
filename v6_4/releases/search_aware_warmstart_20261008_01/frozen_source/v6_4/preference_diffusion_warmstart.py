"""C.2 masked conditional DDPM, fixed-budget training and immutable inference.

This module performs no physics. It reuses the inherited cosine100/internal-v/
DDIM20 implementation, and never repairs a generated meter coefficient.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from .dataset import object_sha, sha256
from .residual_dataset import _read, _write, ResidualNormalizer
from .residual_diffusion import (ResidualDDPM, ResidualDiffusionConfig, _configure,
                                 _runtime, _state_sha)
from .diffusion_model import (noise_schedule_identity, objective_identity,
                             parameterization_identity)
from .route_optimizer_protocol import active_intervals, parameter_plan, VERSIONS
from .task_anchored_reference import (build_reference_definition,
    TaskAnchoredResidualPlan, reference_precheck)

TRAINING_DEFAULTS = {
    "seed": 64221, "draw_seed": 64222, "validation_seed": 64223,
    "test_noise_seed": 64224, "optimizer_updates": 4000, "batch_size": 32,
    "validate_every": 250, "curve_every": 50,
    "optimizer": {"name": "AdamW", "lr": 1e-4, "weight_decay": .01,
                  "gradient_norm_clip": 1.},
    "validation_conditions": [["A", "v1"], ["B", "v2"]],
    "validation_noise_per_condition": 2,
    "validation_sample_units_limit": 128,
    "extra_ddim_smoke_sample_units_limit": 8,
}
CHECKPOINT_SCHEMA = "v6_4_preference_diffusion_checkpoint_v1"
SELECTION = ("maximum raw-amplitude-and-analytic-reference-legal sample count; "
             "then minimum task-balanced nearest matching-label decoded position RMS "
             "at 50Hz on frozen W_support; then fixed task-balanced VAL v-MSE; "
             "then earlier update; missing labels are not passes")
TEST_NOISE_DERIVATION = ("SeedSequence([64224, *little-endian uint32 words of "
                        "SHA256(task.sha256 ASCII)]); numpy default_rng; "
                        "two successive standard_normal(12).astype(float32), A-v1 then B-v2")


def _load_dataset(path):
    from .preference_teacher_dataset import load_preference_dataset
    return load_preference_dataset(path)


def search_mask(definition):
    result = np.zeros(6, dtype=bool)
    result[active_intervals(definition)] = True
    return result


def condition_key(preference, family, mask):
    if preference not in ("A", "B") or family not in VERSIONS:
        raise ValueError("unknown preference/family")
    m = np.asarray(mask)
    if m.shape != (6,) or m.dtype != np.bool_:
        raise ValueError("six boolean search intervals required")
    return f"{preference}/{family}/" + "".join("1" if x else "0" for x in m)


def supported_conditions(ds):
    result = {}
    for i in ds.indices("train"):
        s = ds.samples[int(i)]
        key = condition_key(s["preference"], s["reference_family"], ds.search_masks[i])
        row = result.setdefault(key, {"preference": s["preference"],
            "reference_family": s["reference_family"],
            "search_mask": ds.search_masks[i].tolist(), "sample_ids": [], "mother_ids": []})
        row["sample_ids"].append(s["sample_id"])
        row["mother_ids"].append(s["mother_id"])
    for row in result.values():
        row["sample_ids"] = sorted(set(row["sample_ids"]))
        row["mother_ids"] = sorted(set(row["mother_ids"]))
    return dict(sorted(result.items()))


def balanced_index(ds):
    """mother -> Task -> (preference, family) -> unique near-optimal references."""
    result = {}
    for i in ds.indices("train"):
        s = ds.samples[int(i)]
        bucket = (s["preference"], s["reference_family"])
        result.setdefault(s["mother_id"], {}).setdefault(s["task_id"], {}).setdefault(bucket, []).append(int(i))
    return {m: {t: {b: sorted(set(rows), key=lambda i: ds.samples[i]["sample_id"])
                    for b, rows in sorted(buckets.items())}
                for t, buckets in sorted(tasks.items())}
            for m, tasks in sorted(result.items())}


def draw_training_indices(index, generator, size=32):
    if not index:
        raise ValueError("no supervised TRAIN references")
    def choose(values):
        values = list(values)
        return values[int(torch.randint(len(values), (1,), generator=generator))]
    rows = []
    for _ in range(size):
        tasks = index[choose(index)]
        buckets = tasks[choose(tasks)]
        rows.append(choose(buckets[choose(buckets)]))
    return np.asarray(rows, dtype=np.int64)


def _val_task_ids(ds):
    if hasattr(ds, "task_splits"):
        return sorted(t for t, split in ds.task_splits.items() if split == "val")
    return sorted({s["task_id"] for s in ds.samples if s["split"] == "val"})


def _matching_indices(ds, task_id, preference, family, split="val"):
    return [int(i) for i in ds.indices(split) if ds.samples[int(i)]["task_id"] == task_id
            and ds.samples[int(i)]["preference"] == preference
            and ds.samples[int(i)]["reference_family"] == family]


def make_validation_draws(ds):
    """Exactly eight declared units for the two frozen VAL Tasks, never TEST."""
    tids = _val_task_ids(ds)
    if len(tids) != 2:
        raise ValueError("C.2 fixed validation requires exactly two frozen VAL Tasks")
    supported = supported_conditions(ds)
    gen = torch.Generator(device="cpu").manual_seed(64223)
    units = []
    for tid in tids:
        for pref, family in TRAINING_DEFAULTS["validation_conditions"]:
            definition = ds.definitions[(tid, family)]
            mask = search_mask(definition)
            refs = _matching_indices(ds, tid, pref, family)
            key = condition_key(pref, family, mask)
            for draw in range(2):
                noise = torch.randn(12, generator=gen).numpy()
                noise = np.where(np.repeat(mask, 2), noise, 0.)
                units.append({"unit_id": f"{tid}/{pref}/{family}/{draw}", "task_id": tid,
                    "preference": pref, "reference_family": family, "draw": draw,
                    "search_mask": mask.tolist(), "reference_indices": refs,
                    "noise": noise.tolist(), "noise_sha256": hashlib.sha256(noise.tobytes()).hexdigest(),
                    "status": ("READY" if refs and key in supported and mask.any() else
                               "MISSING_VAL_LABEL" if not refs else "UNSUPPORTED_TRAINING_CONDITION")})
    # Fixed loss draws are loss forwards, not DDIM sample units. Give each
    # near-optimal VAL reference one fixed t/noise draw, then balance by Task.
    refs = ds.indices("val")
    ts = torch.randint(100, (len(refs),), generator=gen).numpy()
    eps = torch.randn((len(refs), 12), generator=gen).numpy()
    eps = np.where(np.repeat(ds.search_masks[refs], 2, axis=1), eps, 0.)
    return {"schema": "v6_4_preference_fixed_val_draws_v1", "seed": 64223,
            "units": units, "reference_indices": refs.tolist(),
            "timesteps": ts.tolist(), "epsilon": eps.tolist(), "test_read": False}


def _source_identity():
    root = Path(__file__).resolve().parents[1]
    names = ("preference_diffusion_warmstart.py", "preference_teacher_dataset.py",
        "residual_diffusion.py", "residual_dataset.py", "diffusion_model.py",
        "task_anchored_reference.py", "route_optimizer_protocol.py")
    return {str(root / "v6_4" / n): sha256(root / "v6_4" / n) for n in names}


def prepare_training(dataset_manifest, output, *, plan=None, device="cpu"):
    """Seal data/scalers/schema/draws; never train or simulate."""
    output = Path(output)
    if (output / "training_config.json").exists() or (output / "training_status.json").exists():
        raise FileExistsError("C.2 training already prepared")
    ds = _load_dataset(dataset_manifest)
    if not len(ds.indices("train")) or not len(ds.indices("val")):
        result = {"status": "TRAINING_NOT_RUN", "reason": "NO_VALID_TRAIN_OR_VAL_SUPERVISION",
                  "training_executed": False, "checkpoint_created": False, "physics_steps": 0,
                  "counts": ds.manifest.get("counts", {}), "optimizer_updates_total": 0,
                  "selected_checkpoint_update": None, "supported_preference_family_conditions": []}
        _write(output / "training_status.json", result)
        _write(output / "training_summary.json", result)
        return result
    supplied = _read(plan) if plan is not None else {}
    supplied = supplied.get("training", supplied)
    for key, value in TRAINING_DEFAULTS.items():
        if key in supplied and supplied[key] != value:
            raise ValueError(f"frozen C.2 training field differs: {key}")
    cn, zn = ds.condition_scaler, ds.residual_scaler
    _write(output / "condition_normalizer.json", cn.to_dict())
    _write(output / "residual_normalizer.json", zn.to_dict())
    names = cn.to_dict()["feature_names_with_units"]
    schema = {"schema": "v6_4_preference_model_condition_schema_v1",
              "condition_dim": len(names), "feature_names_with_units": names,
              "literal": cn.to_dict()["literal"], "latent_dim": 12,
              "noise_and_loss_mask": "C.1 search intervals only; at most four dimensions",
              "legal_reference_mask_separately_encoded": True}
    _write(output / "condition_schema.json", schema)
    draws = make_validation_draws(ds)
    _write(output / "validation_draws.json", draws)
    _configure()
    config = ResidualDiffusionConfig(len(names))
    c = {"schema": "v6_4_preference_training_protocol_v1", **copy.deepcopy(TRAINING_DEFAULTS),
         "model": config.to_dict(), "device": str(device), "runtime": _runtime(device),
         "dataset_manifest_path": str(Path(dataset_manifest).resolve()),
         "dataset_manifest_sha256": sha256(dataset_manifest), "source_files": _source_identity(),
         "normalizer_sha256": {n: sha256(output / n) for n in
             ("condition_normalizer.json", "residual_normalizer.json", "condition_schema.json")},
         "validation_draws_sha256": sha256(output / "validation_draws.json"),
         "supported_conditions": supported_conditions(ds), "selection": SELECTION,
         "sampling_distribution": "uniform mother, uniform Task, uniform labeled preference/family bucket, uniform near-optimal reference",
         "noise": noise_schedule_identity(config), "objective": objective_identity(config),
         "parameterization": parameterization_identity(config), "fresh_initialization": True,
         "old_weights_loaded": False, "physics_steps": 0,
         "data_status": ds.manifest.get("data_status", "DATA_LIMITED"),
         "postprocessing": "inverse TRAIN residual scaler only; no clipping/projection/resampling/fallback"}
    _write(output / "training_config.json", c)
    return {"status": "READY", "checkpoint_created": False, "config": c}


def load_training_bundle(output):
    output = Path(output)
    c = _read(output / "training_config.json")
    if sha256(c["dataset_manifest_path"]) != c["dataset_manifest_sha256"]:
        raise ValueError("dataset manifest changed after preparation")
    for path, expected in c["source_files"].items():
        if sha256(path) != expected:
            raise ValueError(f"training source changed: {path}")
    for name, expected in c["normalizer_sha256"].items():
        if sha256(output / name) != expected:
            raise ValueError(f"scaler/schema changed: {name}")
    if sha256(output / "validation_draws.json") != c["validation_draws_sha256"]:
        raise ValueError("fixed VAL draws changed")
    ds = _load_dataset(c["dataset_manifest_path"])
    if object_sha(ds.condition_scaler.to_dict()) != object_sha(_read(output / "condition_normalizer.json")):
        raise ValueError("dataset condition scaler changed")
    if object_sha(ds.residual_scaler.to_dict()) != object_sha(_read(output / "residual_normalizer.json")):
        raise ValueError("dataset residual scaler changed")
    return {"config": c, "dataset": ds, "validation_draws": _read(output / "validation_draws.json")}


def make_tensors(ds, device="cpu"):
    clean = np.stack([ds.residual_scaler.normalize(z, m).reshape(12)
                      for z, m in zip(ds.z_m, ds.search_masks)])
    conditions = np.stack([ds.condition_scaler.transform(ds.tasks[s["task_id"]],
        ds.definitions[(s["task_id"], s["reference_family"])], s["preference"],
        s["reference_family"], ds.search_masks[i]) for i, s in enumerate(ds.samples)])
    return (torch.as_tensor(clean, dtype=torch.float32, device=device),
            torch.as_tensor(conditions, dtype=torch.float32, device=device),
            torch.as_tensor(ds.search_masks, dtype=torch.bool, device=device))


def support_sample_times(definition):
    """Frozen C.1 support windows; spacing at most 20 ms, endpoints included."""
    grids = [np.linspace(*definition["intervals_s"][i],
        int(np.ceil(np.diff(definition["intervals_s"][i])[0] * 50)) + 1)
        for i in active_intervals(definition)]
    if not grids:
        raise ValueError("NOT_APPLICABLE: no C.1 W_support")
    return np.unique(np.concatenate(grids))


def inverse_raw(normalizer, latent, mask):
    """Decode as-is, including invalid values, so rejection retains the raw data.

    Valid outputs equal ResidualNormalizer.inverse exactly. Invalid inactive
    coordinates and nonfinite values must survive to the seed qualifier.
    """
    x = np.asarray(latent, dtype=float).reshape(6, 2)
    m = np.asarray(mask, dtype=bool)
    raw = x * normalizer.std_m
    raw[m] += normalizer.mean_m
    return raw


def raw_reference_assessment(task, family, raw_z_m, mask):
    raw = np.asarray(raw_z_m, dtype=float)
    m = np.asarray(mask, dtype=bool)
    expected = search_mask(build_reference_definition(task, version=VERSIONS[family]))
    if m.shape != (6,) or not m.any() or not np.array_equal(m, expected):
        return {"legal": False, "reason": "NOT_C1_SEARCH_MASK"}, None
    if raw.shape != (6, 2) or not np.isfinite(raw).all():
        return {"legal": False, "reason": "NONFINITE_OR_WRONG_RAW_SHAPE"}, None
    if np.any(raw[~m] != 0):
        return {"legal": False, "reason": "INACTIVE_SEARCH_DIMENSION_NONZERO"}, None
    if np.any(np.linalg.norm(raw, axis=1) > .020):
        return {"legal": False, "reason": "RAW_AMPLITUDE_EXCEEDS_20MM"}, None
    try:
        plan = parameter_plan(task, family, raw[m].reshape(-1))
        check = reference_precheck(task, plan)
    except (ValueError, TypeError, KeyError) as exc:
        return {"legal": False, "reason": "ANALYTIC_REFERENCE_REJECTED", "error": str(exc)}, None
    return {"legal": bool(check["passed"]), "reason": check["status"],
            "plan_sha256": plan.sha256(), "checks": check["checks"]}, plan


def selection_key(validation, update):
    distance = validation["decoded_position_rms_m"]
    return (-validation["legal_count"], float("inf") if distance is None else distance,
            validation["fixed_v_mse"], update)


@torch.no_grad()
def validate_checkpoint(model, ds, tensors, draws, device="cpu"):
    """At most eight DDIM units; decoded references provide representation supervision."""
    clean, conditions, masks = tensors
    refs = np.asarray(draws["reference_indices"], dtype=np.int64)
    idx = torch.as_tensor(refs, device=device)
    losses = model.loss(clean[idx], torch.as_tensor(draws["timesteps"], dtype=torch.long, device=device),
        torch.as_tensor(draws["epsilon"], dtype=torch.float32, device=device),
        conditions[idx], masks[idx], reduction="none").cpu().numpy()
    by_task_loss = {tid: float(np.mean([losses[j] for j, i in enumerate(refs)
                                      if ds.samples[int(i)]["task_id"] == tid]))
                    for tid in sorted({ds.samples[int(i)]["task_id"] for i in refs})}
    units, distances, legal, generated = [], {}, 0, 0
    for unit in draws["units"]:
        row = {k: v for k, v in unit.items() if k != "noise"}
        if unit["status"] != "READY":
            row.update({"legal": False, "decoded_position_rms_m": None, "ddim_generated": False})
            units.append(row)
            continue
        tid, family, pref = unit["task_id"], unit["reference_family"], unit["preference"]
        task, definition = ds.tasks[tid], ds.definitions[(tid, family)]
        mask = np.asarray(unit["search_mask"], dtype=bool)
        condition = torch.as_tensor(ds.condition_scaler.transform(task, definition, pref, family, mask)[None], device=device)
        latent = model.sample_ddim(condition, torch.as_tensor(mask[None], device=device),
                                  initial_noise=np.asarray(unit["noise"], dtype=np.float32)[None])
        generated += 1
        raw = inverse_raw(ds.residual_scaler, latent.cpu().numpy(), mask)
        assessment, plan = raw_reference_assessment(task, family, raw, mask)
        distance = None
        if assessment["legal"]:
            legal += 1
            times = support_sample_times(definition)
            position = plan.offset_kinematics(times)[0]
            values = []
            for i in unit["reference_indices"]:
                target = TaskAnchoredResidualPlan.from_definition(definition, ds.z_m[i])
                difference = position - target.offset_kinematics(times)[0]
                values.append(float(np.sqrt(np.mean(np.sum(difference ** 2, axis=1)))))
            distance = min(values)
            distances.setdefault(tid, []).append(distance)
        from .route_initializers import json_raw
        row.update({**assessment, "raw_z_m": json_raw(raw), "decoded_position_rms_m": distance,
                    "ddim_generated": True})
        units.append(row)
    task_distance = {tid: float(np.mean(values)) for tid, values in sorted(distances.items())}
    return {"legal_count": legal, "declared_sample_units": len(draws["units"]),
        "ddim_sample_units": generated, "decoded_position_rms_m":
            float(np.mean(list(task_distance.values()))) if task_distance else None,
        "decoded_position_rms_by_task_m": task_distance,
        "fixed_v_mse": float(np.mean(list(by_task_loss.values()))),
        "fixed_v_mse_by_task": by_task_loss, "units": units,
        "distance_scope": "50Hz frozen W_support decoded position supervision, no actual quality",
        "physics_steps": 0, "test_read": False}


def preflight(model, ds, tensors):
    """Finite fixed-batch check on a disposable clone; zero DDIM/physics."""
    clean, conditions, masks = tensors
    refs = ds.indices("train")[:min(8, len(ds.indices("train")))]
    idx = torch.as_tensor(refs, device=clean.device)
    for i in refs:
        z = ds.residual_scaler.inverse(ds.residual_scaler.normalize(ds.z_m[i], ds.search_masks[i]), ds.search_masks[i])
        np.testing.assert_allclose(z, ds.z_m[i], atol=1e-15, rtol=1e-12)
    probe = copy.deepcopy(model)
    generator = torch.Generator(device="cpu").manual_seed(64222)
    ts = torch.randint(100, (len(refs),), generator=generator).to(clean.device)
    eps = torch.randn((len(refs), 12), generator=generator).to(clean.device)
    opt = torch.optim.AdamW(probe.parameters(), lr=1e-4, weight_decay=.01)
    initial = float(probe.loss(clean[idx], ts, eps, conditions[idx], masks[idx]).detach())
    for _ in range(8):
        opt.zero_grad(set_to_none=True)
        loss = probe.loss(clean[idx], ts, eps, conditions[idx], masks[idx])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(probe.parameters(), 1., error_if_nonfinite=True)
        opt.step()
    final = float(probe.loss(clean[idx], ts, eps, conditions[idx], masks[idx]).detach())
    if not np.isfinite(final) or final >= initial:
        raise FloatingPointError("fixed small-batch diagnostic did not reduce v-MSE")
    return {"passed": True, "disposable_clone_diagnostic_updates": 8,
            "official_training_updates": 0, "initial_fixed_loss": initial,
            "final_fixed_loss": final, "ddim_sample_units": 0, "physics_steps": 0}


def train_model(output):
    """One fresh 4000-update run, 16 VAL checkpoints; no implicit restart."""
    output = Path(output)
    if (output / "training_status.json").exists():
        status = _read(output / "training_status.json")
        if status["status"] == "TRAINING_NOT_RUN":
            return status
        if status["status"] != "COMPLETED":
            raise ValueError("unknown training completion state")
    if (output / "model" / "training_report.json").exists():
        load_training_bundle(output)
        report = _read(output / "model" / "training_report.json")
        for name in ("selected", "last"):
            if sha256(output / "model" / (name + ".pt")) != report[name + "_checkpoint_sha256"]:
                raise ValueError("completed checkpoint changed; cannot reuse training")
        if report["optimizer_updates_total"] != 4000 or report["validation_checkpoint_count"] != 16:
            raise ValueError("incomplete formal training cannot be reused")
        return report
    if (output / "model").exists():
        raise FileExistsError("training already started; cannot implicitly repeat consumed updates")
    bundle = load_training_bundle(output)
    c, ds = bundle["config"], bundle["dataset"]
    _configure()
    device = torch.device(c["device"])
    if _runtime(device) != c["runtime"]:
        raise ValueError("runtime differs from frozen training protocol")
    torch.manual_seed(c["seed"])
    if device.type == "cuda":
        torch.cuda.manual_seed_all(c["seed"])
    model = ResidualDDPM(ResidualDiffusionConfig.from_dict(c["model"])).to(device)
    initial_sha = _state_sha(model)
    tensors = make_tensors(ds, device)
    check = preflight(model, ds, tensors)
    if _state_sha(model) != initial_sha:
        raise ValueError("preflight modified formal initial weights")
    _write(output / "preflight.json", check)
    model_dir = output / "model"
    model_dir.mkdir()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=.01)
    generator = torch.Generator(device="cpu").manual_seed(c["draw_seed"])
    index = balanced_index(ds)
    clean, conditions, masks = tensors
    exposures = {s["sample_id"]: 0 for s in ds.samples if s["split"] == "train"}
    draw_hash = hashlib.sha256()
    config_sha = sha256(output / "training_config.json")
    start = time.perf_counter()
    selected, selected_val, best_key, units = None, None, None, 0
    history = []
    def log(row):
        with (model_dir / "curves.jsonl").open("a", encoding="utf8") as handle:
            handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
    def checkpoint(update, validation):
        return {"schema": CHECKPOINT_SCHEMA, "model_config": c["model"],
            "state_dict": model.state_dict(), "state_sha256": _state_sha(model),
            "initial_state_sha256": initial_sha, "training_config": c,
            "training_config_sha256": config_sha, "update": update,
            "selected_checkpoint_update": update, "optimizer_updates_experienced": update,
            "validation": validation, "condition_normalizer": ds.condition_scaler.to_dict(),
            "residual_normalizer": ds.residual_scaler.to_dict(),
            "condition_schema": _read(output / "condition_schema.json"),
            "supported_conditions": c["supported_conditions"],
            "training_draws_sha256": draw_hash.hexdigest(), "sample_exposures": dict(exposures),
            "fresh_initialization": True, "old_weights_loaded": False, "physics_steps": 0}
    model.train()
    for update in range(1, 4001):
        refs = draw_training_indices(index, generator, 32)
        ts = torch.randint(100, (32,), generator=generator)
        eps = torch.randn((32, 12), generator=generator)
        eps *= torch.as_tensor(np.repeat(ds.search_masks[refs], 2, axis=1))
        for values in (refs, ts.numpy(), eps.numpy()):
            draw_hash.update(values.tobytes())
        for i in refs:
            exposures[ds.samples[int(i)]["sample_id"]] += 1
        idx = torch.as_tensor(refs, device=device)
        opt.zero_grad(set_to_none=True)
        loss = model.loss(clean[idx], ts.to(device), eps.to(device), conditions[idx], masks[idx])
        if not torch.isfinite(loss):
            raise FloatingPointError("nonfinite C.2 training loss")
        loss.backward()
        grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        opt.step()
        log({"kind": "train", "update": update, "v_mse": float(loss.detach()),
             "gradient_norm_before_clip": float(grad), "elapsed_s": time.perf_counter() - start})
        if update % 250 == 0:
            load_training_bundle(output)
            if sha256(output / "training_config.json") != config_sha:
                raise ValueError("training config changed live")
            model.eval()
            validation = validate_checkpoint(model, ds, tensors, bundle["validation_draws"], device)
            units += validation["ddim_sample_units"]
            if units > 128:
                raise ValueError("formal VAL DDIM sample budget exceeded")
            key = selection_key(validation, update)
            improved = best_key is None or key < best_key
            snapshot = checkpoint(update, validation)
            torch.save(snapshot, model_dir / f"checkpoint_{update:04d}.pt")
            if improved:
                torch.save(snapshot, model_dir / "selected.pt")
                best_key, selected, selected_val = key, update, validation
            row = {"kind": "validation", "update": update, **validation,
                   "selected": improved, "elapsed_s": time.perf_counter() - start}
            log(row)
            history.append(row)
            model.train()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    final = checkpoint(4000, validation)
    final["optimizer_state_dict"] = opt.state_dict()
    torch.save(final, model_dir / "last.pt")
    report = {"status": "COMPLETED", "training_executed": True,
        "optimizer_updates_total": 4000, "optimizer_updates": 4000,
        "selected_checkpoint_update": selected, "selected_update": selected,
        "selected_checkpoint_updates_experienced": selected, "batch_size": 32,
        "sample_exposures_total": sum(exposures.values()), "sample_exposures": exposures,
        "validation_checkpoint_count": len(history), "validation_ddim_sample_units": units,
        "extra_ddim_smoke_sample_units": 0, "selected_validation": selected_val,
        "selection": SELECTION, "elapsed_s": time.perf_counter() - start,
        "initial_state_sha256": initial_sha, "final_state_sha256": _state_sha(model),
        "training_draws_sha256": draw_hash.hexdigest(), "test_used_for_selection": False,
        "selected_checkpoint_sha256": sha256(model_dir / "selected.pt"),
        "last_checkpoint_sha256": sha256(model_dir / "last.pt"),
        "training_config_sha256": config_sha, "supported_conditions": c["supported_conditions"],
        "data_status": c["data_status"], "fresh_initialization": True,
        "old_weights_loaded": False, "runtime": c["runtime"], "physics_steps": 0}
    _write(model_dir / "training_report.json", report)
    _write(output / "training_summary.json", {**report,
        "supported_preference_family_conditions": report["supported_conditions"]})
    _write(output / "training_status.json", {"status": "COMPLETED",
        "training_executed": True, "optimizer_updates_total": 4000,
        "selected_checkpoint_update": selected,
        "training_report_sha256": sha256(model_dir / "training_report.json")})
    return report


def freeze_model(model_output, *, retrieval_identity, freeze_path=None, output=None):
    """Seal actual selected weights, scalers, schema, seeds and retrieval before TEST."""
    model_output = Path(model_output)
    if freeze_path is not None and output is not None:
        raise ValueError("provide only one freeze destination")
    target = Path(freeze_path or output) if freeze_path or output else model_output.parent / "model_freeze.json"
    report = _read(model_output / "model" / "training_report.json")
    bundle = load_training_bundle(model_output)
    if report["status"] != "COMPLETED" or report["optimizer_updates_total"] != 4000:
        raise ValueError("only completed fixed-budget training can be frozen")
    if sha256(model_output / "model" / "selected.pt") != report["selected_checkpoint_sha256"]:
        raise ValueError("selected checkpoint changed since the completed training report")
    paths = [model_output / "model" / "selected.pt", model_output / "model" / "training_report.json",
        model_output / "training_config.json", model_output / "condition_normalizer.json",
        model_output / "residual_normalizer.json", model_output / "condition_schema.json", model_output / "validation_draws.json"]
    frozen = {"schema": "v6_4_preference_model_freeze_v1",
        "artifacts": {str(p.resolve()): sha256(p) for p in paths},
        "selected_checkpoint": str(paths[0].resolve()),
        "optimizer_updates_total": 4000, "selected_checkpoint_update": report["selected_checkpoint_update"],
        "seeds": {k: bundle["config"][k] for k in ("seed", "draw_seed", "validation_seed", "test_noise_seed")},
        "test_noise_derivation": TEST_NOISE_DERIVATION,
        "supported_conditions": report["supported_conditions"],
        "retrieval_identity": retrieval_identity, "retrieval_identity_sha256": object_sha(retrieval_identity),
        "test_results_used": False, "scalers_train_only": True}
    frozen["freeze_sha256"] = object_sha(frozen)
    if target.exists():
        if _read(target) != frozen:
            raise ValueError("model/retrieval freeze already differs")
    else:
        _write(target, frozen)
    return frozen


class PreferenceSampler:
    def __init__(self, checkpoint, device="cpu", *, freeze_path=None):
        from .preference_teacher_dataset import ConditionNormalizer
        start = time.perf_counter()
        _configure()
        self.checkpoint_path = Path(checkpoint)
        self.checkpoint_sha256 = sha256(checkpoint)
        c = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if c.get("schema") != CHECKPOINT_SCHEMA:
            raise ValueError("not a real C.2 preference checkpoint")
        for path, expected in c["training_config"]["source_files"].items():
            if sha256(path) != expected:
                raise ValueError(f"model production source changed: {path}")
        if freeze_path is not None:
            frozen = _read(freeze_path)
            for path, expected in frozen["artifacts"].items():
                if sha256(path) != expected:
                    raise ValueError("frozen model/scaler/schema artifact changed")
            if frozen["selected_checkpoint"] != str(self.checkpoint_path.resolve()):
                raise ValueError("checkpoint is not the pre-TEST selected artifact")
        self.device = torch.device(device)
        self.model = ResidualDDPM(ResidualDiffusionConfig.from_dict(c["model_config"])).to(self.device)
        self.model.load_state_dict(c["state_dict"], strict=True)
        self.model.eval()
        if _state_sha(self.model) != c["state_sha256"]:
            raise ValueError("checkpoint tensors do not match identity")
        self.checkpoint = c
        self.condition_normalizer = ConditionNormalizer.from_dict(c["condition_normalizer"])
        self.residual_normalizer = ResidualNormalizer.from_dict(c["residual_normalizer"])
        self.identity = {"checkpoint_sha256": self.checkpoint_sha256,
            "training_config_sha256": c["training_config_sha256"],
            "condition_scaler_sha256": object_sha(c["condition_normalizer"]),
            "residual_scaler_sha256": object_sha(c["residual_normalizer"]),
            "condition_schema_sha256": object_sha(c["condition_schema"]),
            "selected_checkpoint_update": c["update"],
            "supported_conditions": c["supported_conditions"]}
        self.model_load_wall_s = time.perf_counter() - start
        self.sample_units = 0

    @classmethod
    def from_frozen(cls, model_output, device="cpu"):
        model_output = Path(model_output)
        return cls(model_output / "model" / "selected.pt", device,
                   freeze_path=model_output.parent / "model_freeze.json")

    def initializer_proposals(self, task, noise_seed=64224):
        if noise_seed != 64224:
            raise ValueError("frozen TEST noise seed is 64224")
        words = np.frombuffer(hashlib.sha256(task.sha256().encode("ascii")).digest(), dtype="<u4")
        rng = np.random.default_rng(np.random.SeedSequence([noise_seed, *map(int, words)]))
        proposals = {}
        for slot, pref, family in ((1, "A", "v1"), (3, "B", "v2")):
            noise = rng.standard_normal(12).astype(np.float32)
            raw, metadata = self.sample(task, pref, family, noise)
            proposals[slot] = {**metadata, "raw_z_m": None if raw is None else raw.reshape(12).tolist(),
                "noise_seed": noise_seed, "noise_id": f"{task.sha256()}/{pref}/{family}/64224",
                "noise_derivation": TEST_NOISE_DERIVATION}
        return proposals

    @torch.no_grad()
    def sample(self, task, preference, family, initial_noise_np):
        definition = build_reference_definition(task, version=VERSIONS[family])
        mask = search_mask(definition)
        key = condition_key(preference, family, mask)
        metadata = {"source": "diffusion", "preference": preference, "family": family,
                    **self.identity, "physics_steps": 0, "raw_postprocessing": "inverse scaler only"}
        if key not in self.checkpoint["supported_conditions"]:
            return None, {**metadata, "initializer_rejection": "UNSUPPORTED_TRAINING_CONDITION",
                          "ddim_sample_units": 0}
        noise = np.asarray(initial_noise_np, dtype=np.float32)
        if noise.shape not in ((12,), (6, 2)) or not np.isfinite(noise).all():
            raise ValueError("finite fixed 12-D initial noise required")
        start = time.perf_counter()
        condition = self.condition_normalizer.transform(task, definition, preference, family, mask)
        condition = torch.as_tensor(condition[None], device=self.device)
        prepared = time.perf_counter() - start
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        start = time.perf_counter()
        latent = self.model.sample_ddim(condition, torch.as_tensor(mask[None], device=self.device),
                                        initial_noise=noise.reshape(1, 12))
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        elapsed = time.perf_counter() - start
        raw = inverse_raw(self.residual_normalizer, latent.cpu().numpy(), mask)
        self.sample_units += 1
        return raw, {**metadata, "condition_prepare_wall_s": prepared, "inference_wall_s": elapsed,
            "ddim_sample_units": 1, "noise_sha256": hashlib.sha256(noise.tobytes()).hexdigest(),
            "masked_initial_noise_sha256": hashlib.sha256(np.where(np.repeat(mask, 2), noise.reshape(12), 0.).tobytes()).hexdigest()}


load_preference_sampler = PreferenceSampler


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "train", "freeze"))
    parser.add_argument("--output", required=True)
    parser.add_argument("--dataset")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--retrieval-identity")
    parser.add_argument("--freeze-path")
    args = parser.parse_args()
    if args.action == "prepare":
        if not args.dataset:
            parser.error("--dataset required")
        result = prepare_training(args.dataset, args.output, device=args.device)
    elif args.action == "train":
        result = train_model(args.output)
    else:
        if not args.retrieval_identity:
            parser.error("--retrieval-identity required")
        result = freeze_model(args.output, retrieval_identity=_read(args.retrieval_identity), freeze_path=args.freeze_path)
    print(json.dumps(result, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
