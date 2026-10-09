"""C.3 paired fresh D/S training and immutable high-level seed inference.

The diffusion implementation is the unmodified C.2 ResidualDDPM. Regression
uses the same task condition, TRAIN scalers, twelve coordinates and declared
search mask. This module never invokes physics or repairs a proposal. Training
does not generate DDIM samples or select weights; only updates 250 and 4000
are exposed for the separate closed-loop VAL protocol.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn

from .dataset import object_sha, sha256
from .diffusion_model import noise_schedule_identity, objective_identity, parameterization_identity
from .preference_diffusion_warmstart import condition_key, inverse_raw, search_mask, supported_conditions
from .preference_teacher_dataset import ConditionNormalizer
from .residual_dataset import ResidualNormalizer, _read, _write
from .residual_diffusion import ResidualDDPM, ResidualDiffusionConfig, _configure, _mask, _runtime, _state_sha
from .route_initializers import OVERRIDE_CONDITIONS, json_raw
from .route_optimizer_protocol import VERSIONS
from .task_anchored_reference import build_reference_definition


CHECKPOINT_SCHEMA = "v64_c3_paired_warmstart_checkpoint_v1"
CHECKPOINT_UPDATES = (250, 4000)
SOURCE_TYPES = ("route_quality_example", "initializer_effect_example")
TRAINING_DEFAULTS = {
    "optimizer_updates": 4000, "batch_size": 32, "checkpoint_updates": [250, 4000],
    "initialization_seeds": {"D": 64321, "S": 64331}, "draw_seed": 64322,
    "diffusion_forward_noise_seed": 64323, "test_noise_seed": 64324, "val_noise_seed": 64325,
    "curve_every": 50,
    "optimizer": {"name": "AdamW", "lr": 1e-4, "weight_decay": .01, "gradient_norm_clip": 1.},
}
NOISE_DERIVATION = ("SeedSequence([noise_seed, *little-endian uint32 words of "
    "SHA256(task.sha256 ASCII)]); numpy default_rng; two successive "
    "standard_normal(12).astype(float32), A-v1 then B-v2")


class SimpleWarmstartRegression(nn.Module):
    """Two 128-unit SiLU layers; no noisy latent or diffusion timestep input."""
    def __init__(self, condition_dim):
        super().__init__()
        if not isinstance(condition_dim, int) or condition_dim < 1:
            raise ValueError("positive condition dimension required")
        self.condition_dim = condition_dim
        self.network = nn.Sequential(nn.Linear(condition_dim + 12, 128), nn.SiLU(),
            nn.Linear(128, 128), nn.SiLU(), nn.Linear(128, 12))

    def forward(self, condition, mask):
        if condition.ndim != 2 or condition.shape[1] != self.condition_dim or not torch.isfinite(condition).all():
            raise ValueError("finite declared condition [B,C] required")
        m = _mask(mask, len(condition), condition.device)
        # This is the declared generative space, not post-hoc raw repair.
        return self.network(torch.cat([condition, m.to(condition.dtype)], -1)) * m

    def loss(self, clean, condition, mask, *, reduction="mean"):
        m = _mask(mask, len(condition), condition.device)
        if clean.shape != (len(condition), 12) or not torch.isfinite(clean).all() or torch.any(clean[~m] != 0):
            raise ValueError("finite legal-space normalized labels required")
        losses = ((self(condition, m) - clean).square() * m).sum(-1) / m.sum(-1)
        if reduction not in ("mean", "none"):
            raise ValueError("unsupported reduction")
        return losses.mean() if reduction == "mean" else losses


def _load_dataset(path):
    from .search_effect_teacher import load_search_aware_dataset
    return load_search_aware_dataset(path)


def _source_identity():
    root = Path(__file__).resolve().parents[1]
    names = ("simple_warmstart_regression.py", "search_effect_teacher.py",
        "preference_diffusion_warmstart.py", "preference_teacher_dataset.py", "residual_diffusion.py",
        "residual_dataset.py", "diffusion_model.py", "dataset.py", "route_initializers.py",
        "route_optimizer_protocol.py", "task_anchored_reference.py", "task_protocol.py")
    return {str(root / "v6_4" / n): sha256(root / "v6_4" / n) for n in names}


def _training_runtime(device):
    return {**_runtime(device), "torch_num_threads": torch.get_num_threads(),
            "torch_num_interop_threads": torch.get_num_interop_threads()}


def source_balanced_index(ds):
    """mother -> Task -> preference/family -> available provenance -> unique z.

    A deduplicated label with both provenances remains one parameter, and can
    participate in either explicitly sampled source subpool.
    """
    result = {}
    for i in ds.indices("train"):
        i = int(i)
        s = ds.samples[i]
        if ds.task_splits.get(s["task_id"]) != "train":
            raise ValueError("label differs from external TRAIN split")
        sources = sorted(set(s.get("source_types", [])))
        if not sources or any(x not in SOURCE_TYPES for x in sources):
            raise ValueError("C.3 labels require explicit route/effect provenance")
        bucket = result.setdefault(s["mother_id"], {}).setdefault(s["task_id"], {}).setdefault(
            (s["preference"], s["reference_family"]), {})
        for source in sources:
            bucket.setdefault(source, []).append(i)
    for tasks in result.values():
        for buckets in tasks.values():
            for sources in buckets.values():
                for source, indices in sources.items():
                    unique = {}
                    for i in indices:
                        z_key = object_sha(np.asarray(ds.z_m[i], dtype=float).tolist())
                        if z_key in unique:
                            raise ValueError("duplicate parameter in unified TRAIN bucket")
                        unique[z_key] = i
                    sources[source] = sorted(indices, key=lambda i: ds.samples[i]["sample_id"])
    return result


def draw_source_balanced_indices(index, generator, size=32):
    if not index:
        raise ValueError("no legal TRAIN supervision")
    def choose(values):
        values = sorted(values) if isinstance(values, dict) else list(values)
        return values[int(torch.randint(len(values), (1,), generator=generator))]
    rows = []
    for _ in range(size):
        tasks = index[choose(index)]
        buckets = tasks[choose(tasks)]
        sources = buckets[choose(buckets)]
        rows.append(choose(sources[choose(sources)]))
    return np.asarray(rows, dtype=np.int64)


def make_train_tensors(ds, device="cpu"):
    """Do not read any VAL/TEST task or label while constructing training data."""
    rows = [int(i) for i in ds.indices("train")]
    dim = len(ds.condition_scaler.names)
    clean = np.zeros((len(ds.samples), 12), dtype=np.float32)
    conditions = np.zeros((len(ds.samples), dim), dtype=np.float32)
    masks = np.zeros((len(ds.samples), 6), dtype=bool)
    for i in rows:
        s = ds.samples[i]
        task = ds.tasks[s["task_id"]]
        definition = ds.definitions[(s["task_id"], s["reference_family"])]
        mask = np.asarray(ds.search_masks[i], dtype=bool)
        expected = search_mask(definition)
        if mask.shape != (6,) or not np.array_equal(mask, expected) or not 1 <= int(mask.sum()) <= 2:
            raise ValueError("TRAIN label is outside the at-most-four-dimensional C.1 search mask")
        z = np.asarray(ds.z_m[i], dtype=float)
        if z.shape != (6, 2) or not np.isfinite(z).all() or np.any(z[~mask] != 0) or np.any(np.linalg.norm(z, axis=1) > .020):
            raise ValueError("illegal raw TRAIN label; no training repair")
        if s.get("task_sha256", task.sha256()) != task.sha256():
            raise ValueError("TRAIN Task SHA mismatch")
        clean[i] = ds.residual_scaler.normalize(z, mask).reshape(12)
        conditions[i] = ds.condition_scaler.transform(task, definition, s["preference"], s["reference_family"], mask)
        masks[i] = mask
    if not np.isfinite(clean).all() or not np.isfinite(conditions).all():
        raise ValueError("nonfinite normalized TRAIN data")
    return tuple(torch.as_tensor(a, device=device) for a in (clean, conditions, masks))


def prepare_training_pair(dataset_manifest, output, *, device="cpu", plan=None):
    """Seal paired exposures and shared schema; zero optimizations/DDIM/physics."""
    output = Path(output)
    dataset_manifest = Path(dataset_manifest)
    if dataset_manifest.is_dir():
        dataset_manifest = dataset_manifest / "manifest.json"
    if (output / "training_config.json").exists() or (output / "training_status.json").exists():
        raise FileExistsError("C.3 training already prepared")
    supplied = _read(plan) if plan is not None else {}
    supplied = supplied.get("training", supplied)
    for key, expected in TRAINING_DEFAULTS.items():
        if key in supplied and supplied[key] != expected:
            raise ValueError("frozen C.3 training field differs: " + key)
    ds = _load_dataset(dataset_manifest)
    if not len(ds.indices("train")):
        status = {"status": "TRAINING_NOT_RUN", "reason": "NO_LEGAL_TRAIN_SUPERVISION",
            "training_executed_D": False, "training_executed_S": False,
            "checkpoint_created": False, "physics_steps": 0, "ddim_sample_units": 0}
        _write(output / "training_status.json", status)
        return status
    if not ds.manifest.get("scalers_train_only") or ds.condition_scaler is None or ds.residual_scaler is None:
        raise ValueError("both models require the same sealed TRAIN-only scalers")
    train_shas = {ds.tasks[ds.samples[int(i)]["task_id"]].sha256() for i in ds.indices("train")}
    if set(ds.condition_scaler.fit_task_sha256) != train_shas:
        raise ValueError("condition scaler TRAIN Task identities differ")
    make_train_tensors(ds)
    index = source_balanced_index(ds)
    generator = torch.Generator(device="cpu").manual_seed(TRAINING_DEFAULTS["draw_seed"])
    refs = np.stack([draw_source_balanced_indices(index, generator, 32) for _ in range(4000)])
    output.mkdir(parents=True, exist_ok=True)
    np.savez(output / "paired_reference_draws.npz", reference_indices=refs)
    draws_sha = hashlib.sha256(refs.astype("<i8", copy=False).tobytes()).hexdigest()
    for name, value in (("condition_normalizer.json", ds.condition_scaler.to_dict()),
                        ("residual_normalizer.json", ds.residual_scaler.to_dict())):
        _write(output / name, value)
    normalizer = ds.condition_scaler.to_dict()
    schema = {"schema": "v64_c3_shared_warmstart_condition_schema_v1",
        "condition_dim": len(normalizer["feature_names_with_units"]),
        "feature_names_with_units": normalizer["feature_names_with_units"], "literal": normalizer["literal"],
        "latent_dim": 12, "noise_and_loss_mask": "C.1 search intervals only; at most four dimensions",
        "legal_reference_mask_separately_encoded": True, "ids_or_outcomes_encoded": False}
    _write(output / "condition_schema.json", schema)
    _configure()
    d_config = ResidualDiffusionConfig(schema["condition_dim"])
    config = {"schema": "v64_c3_paired_training_protocol_v1", **copy.deepcopy(TRAINING_DEFAULTS),
        "models": {"D": d_config.to_dict(), "S": {"condition_dim": schema["condition_dim"],
            "latent_dim": 12, "hidden_dim": 128, "hidden_layers": 2, "activation": "SiLU",
            "input": "same normalized task condition plus same declared 12D dimension mask", "objective": "masked_normalized_z_mse"}},
        "device": str(device), "runtime": _training_runtime(device),
        "dataset_manifest_path": str(Path(dataset_manifest).resolve()), "dataset_manifest_sha256": sha256(dataset_manifest),
        "source_files": _source_identity(), "normalizer_sha256": {name: sha256(output / name) for name in
            ("condition_normalizer.json", "residual_normalizer.json", "condition_schema.json")},
        "paired_reference_draws_file_sha256": sha256(output / "paired_reference_draws.npz"),
        "paired_reference_indices_sha256": draws_sha, "sample_ids_in_dataset_order": [s["sample_id"] for s in ds.samples],
        "supported_conditions": supported_conditions(ds), "scalers_train_only": True,
        "sampling_distribution": "uniform mother -> Task -> preference/family -> available source (1:1 if both) -> unique parameter",
        "diffusion_numeric_identity": {"noise": noise_schedule_identity(d_config), "objective": objective_identity(d_config),
            "parameterization": parameterization_identity(d_config), "implementation": "unmodified C.2 ResidualDDPM"},
        "selection": "none during training; separate closed-loop VAL of updates250/4000 only",
        "fresh_initialization": True, "old_weights_loaded": False, "data_status": ds.manifest.get("data_status", "DATA_LIMITED"),
        "postprocessing": "inverse TRAIN scaler only; no clamp/project/resample/fallback",
        "physics_steps": 0, "ddim_sample_units": 0, "sample_exposures_per_model": 128000}
    _write(output / "training_config.json", config)
    return {"status": "READY", "checkpoint_created": False, "config": config,
            "training_manifest": str((output / "training_config.json").resolve())}


def load_training_pair(output):
    output = Path(output)
    c = _read(output / "training_config.json")
    if c["schema"] != "v64_c3_paired_training_protocol_v1":
        raise ValueError("not a C.3 training bundle")
    for key, value in TRAINING_DEFAULTS.items():
        if c.get(key) != value:
            raise ValueError("fixed C.3 training protocol changed: " + key)
    if sha256(c["dataset_manifest_path"]) != c["dataset_manifest_sha256"]:
        raise ValueError("dataset manifest changed")
    for path, expected in c["source_files"].items():
        if sha256(path) != expected:
            raise ValueError("training source changed: " + path)
    for name, expected in c["normalizer_sha256"].items():
        if sha256(output / name) != expected:
            raise ValueError("shared TRAIN scaler/schema changed")
    if sha256(output / "paired_reference_draws.npz") != c["paired_reference_draws_file_sha256"]:
        raise ValueError("paired reference exposure file changed")
    with np.load(output / "paired_reference_draws.npz", allow_pickle=False) as archive:
        refs = np.asarray(archive["reference_indices"], dtype=np.int64)
    if refs.shape != (4000, 32) or hashlib.sha256(refs.astype("<i8", copy=False).tobytes()).hexdigest() != c["paired_reference_indices_sha256"]:
        raise ValueError("paired exposure shape/identity differs")
    ds = _load_dataset(c["dataset_manifest_path"])
    if [s["sample_id"] for s in ds.samples] != c["sample_ids_in_dataset_order"]:
        raise ValueError("paired dataset sample order changed")
    train_indices = set(map(int, ds.indices("train")))
    if not set(refs.reshape(-1).tolist()).issubset(train_indices):
        raise ValueError("non-TRAIN exposure in paired draws")
    for name, value in (("condition_normalizer.json", ds.condition_scaler.to_dict()),
                        ("residual_normalizer.json", ds.residual_scaler.to_dict())):
        if object_sha(_read(output / name)) != object_sha(value):
            raise ValueError("dataset scaler changed")
    return {"config": c, "dataset": ds, "reference_indices": refs}


def _verify_completed_model(output, name, config_sha):
    root = Path(output) / name
    report = _read(root / "training_report.json")
    if report["status"] != "COMPLETED" or report["optimizer_updates_total"] != 4000 or report["training_config_sha256"] != config_sha:
        raise ValueError("completed training identity differs")
    for update in CHECKPOINT_UPDATES:
        if sha256(root / f"checkpoint_{update:04d}.pt") != report["checkpoint_sha256"][str(update)]:
            raise ValueError("completed checkpoint changed")
    return report


def _train_one(output, name, bundle):
    output = Path(output)
    root = output / name
    c, ds, refs = bundle["config"], bundle["dataset"], bundle["reference_indices"]
    config_sha = sha256(output / "training_config.json")
    if (root / "training_report.json").exists():
        return _verify_completed_model(output, name, config_sha)
    if root.exists():
        raise FileExistsError("training already started; consumed updates cannot be implicitly repeated: " + name)
    _configure()
    device = torch.device(c["device"])
    if _training_runtime(device) != c["runtime"]:
        raise ValueError("runtime differs from prepared C.3 protocol")
    root.mkdir()
    seed = c["initialization_seeds"][name]
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    model = (ResidualDDPM(ResidualDiffusionConfig.from_dict(c["models"]["D"])) if name == "D" else
             SimpleWarmstartRegression(c["models"]["S"]["condition_dim"])).to(device)
    initial_sha = _state_sha(model)
    parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=.01)
    clean, conditions, masks = make_train_tensors(ds, device)
    noise_generator = torch.Generator(device="cpu").manual_seed(c["diffusion_forward_noise_seed"])
    exposures = {ds.samples[int(i)]["sample_id"]: 0 for i in ds.indices("train")}
    reference_hash, loss_draw_hash = hashlib.sha256(), hashlib.sha256()
    checkpoints = {}
    start = time.perf_counter()
    status = {"status": "RUNNING", "model": name, "optimizer_updates_started": 0,
        "optimizer_updates_completed": 0, "training_config_sha256": config_sha,
        "training_executed": False, "ddim_sample_units": 0, "physics_steps": 0}
    _write(root / "training_status.json", status)
    model.train()
    for update in range(1, 4001):
        row = refs[update - 1]
        status["optimizer_updates_started"] = update
        _write(root / "training_status.json", status)
        idx = torch.as_tensor(row, device=device)
        reference_hash.update(row.astype("<i8", copy=False).tobytes())
        for i in row:
            exposures[ds.samples[int(i)]["sample_id"]] += 1
        opt.zero_grad(set_to_none=True)
        if name == "D":
            timesteps = torch.randint(100, (32,), generator=noise_generator)
            epsilon = torch.randn((32, 12), generator=noise_generator)
            epsilon *= torch.as_tensor(np.repeat(ds.search_masks[row], 2, axis=1))
            loss_draw_hash.update(timesteps.numpy().astype("<i8", copy=False).tobytes())
            loss_draw_hash.update(epsilon.numpy().astype("<f4", copy=False).tobytes())
            loss = model.loss(clean[idx], timesteps.to(device), epsilon.to(device), conditions[idx], masks[idx])
        else:
            loss = model.loss(clean[idx], conditions[idx], masks[idx])
        if not torch.isfinite(loss):
            raise FloatingPointError("nonfinite C.3 training loss: " + name)
        loss.backward()
        gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        opt.step()
        status.update(optimizer_updates_completed=update, training_executed=True)
        _write(root / "training_status.json", status)
        if update % c["curve_every"] == 0:
            with (root / "curves.jsonl").open("a", encoding="utf8") as handle:
                handle.write(json.dumps({"update": update, "loss": float(loss.detach()),
                    "gradient_norm_before_clip": float(gradient), "elapsed_s": time.perf_counter() - start},
                    sort_keys=True, allow_nan=False) + "\n")
        if update in CHECKPOINT_UPDATES:
            if sha256(output / "training_config.json") != config_sha:
                raise ValueError("training config changed live")
            load_training_pair(output)
            snapshot = {"schema": CHECKPOINT_SCHEMA, "model_name": name, "model_config": c["models"][name],
                "update": update, "optimizer_updates_experienced": update, "state_dict": model.state_dict(),
                "state_sha256": _state_sha(model), "initial_state_sha256": initial_sha,
                "training_config": c, "training_config_sha256": config_sha,
                "condition_normalizer": ds.condition_scaler.to_dict(), "residual_normalizer": ds.residual_scaler.to_dict(),
                "condition_schema": _read(output / "condition_schema.json"), "supported_conditions": c["supported_conditions"],
                "paired_reference_indices_prefix_sha256": reference_hash.hexdigest(), "sample_exposures": dict(exposures),
                "fresh_initialization": True, "old_weights_loaded": False, "effective_parameter_count": parameters,
                "ddim_sample_units": 0, "physics_steps": 0, "checkpoint_selected": False}
            path = root / f"checkpoint_{update:04d}.pt"
            torch.save(snapshot, path)
            checkpoints[str(update)] = sha256(path)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    if reference_hash.hexdigest() != c["paired_reference_indices_sha256"] or sum(exposures.values()) != 128000:
        raise ValueError("paired exposure budget differs")
    report = {"status": "COMPLETED", "model": name, "training_executed": True,
        "optimizer_updates_total": 4000, "batch_size": 32, "sample_exposures_total": sum(exposures.values()),
        "sample_exposures": exposures, "paired_reference_indices_sha256": reference_hash.hexdigest(),
        "diffusion_forward_loss_draws_sha256": loss_draw_hash.hexdigest() if name == "D" else None,
        "initialization_seed": seed, "fresh_initialization": True, "old_weights_loaded": False,
        "initial_state_sha256": initial_sha, "final_state_sha256": _state_sha(model), "effective_parameter_count": parameters,
        "checkpoint_updates": list(CHECKPOINT_UPDATES), "checkpoint_sha256": checkpoints,
        "selected_checkpoint_update": None, "selection": c["selection"], "elapsed_s": time.perf_counter() - start,
        "final_training_loss": float(loss.detach()), "training_config_sha256": config_sha,
        "runtime": c["runtime"], "data_status": c["data_status"], "ddim_sample_units": 0, "physics_steps": 0,
        "test_used_for_training_or_selection": False, "val_used_for_training_or_selection": False}
    _write(root / "training_report.json", report)
    _write(root / "training_status.json", {**status, "status": "COMPLETED",
        "training_report_sha256": sha256(root / "training_report.json")})
    return report


def train_model_pair(output):
    """Two separate fixed 4000-update runs replay the identical reference indices."""
    output = Path(output)
    if (output / "training_status.json").exists():
        status = _read(output / "training_status.json")
        if status.get("status") == "TRAINING_NOT_RUN":
            return status
    bundle = load_training_pair(output)
    reports = {name: _train_one(output, name, bundle) for name in ("D", "S")}
    if reports["D"]["sample_exposures"] != reports["S"]["sample_exposures"] or reports["D"]["paired_reference_indices_sha256"] != reports["S"]["paired_reference_indices_sha256"]:
        raise ValueError("D/S training exposures differ")
    result = {"status": "COMPLETED", "training_executed_D": True, "training_executed_S": True,
        "reports": reports, "paired_exposures_verified": True,
        "selected_checkpoint_D": None, "selected_checkpoint_S": None, "ddim_sample_units": 0, "physics_steps": 0}
    _write(output / "training_summary.json", result)
    _write(output / "training_status.json", {"status": "COMPLETED", "training_executed_D": True,
        "training_executed_S": True, "optimizer_updates_per_model": 4000,
        "training_summary_sha256": sha256(output / "training_summary.json")})
    return result


def frozen_noise(task, noise_seed):
    """One fixed A/v1 and one fixed B/v2 noise, independent of checkpoint."""
    if noise_seed not in (TRAINING_DEFAULTS["val_noise_seed"], TRAINING_DEFAULTS["test_noise_seed"]):
        raise ValueError("only predeclared C.3 VAL/TEST noise seeds are allowed")
    words = np.frombuffer(hashlib.sha256(task.sha256().encode("ascii")).digest(), dtype="<u4")
    rng = np.random.default_rng(np.random.SeedSequence([noise_seed, *map(int, words)]))
    return {slot: rng.standard_normal(12).astype(np.float32) for slot in OVERRIDE_CONDITIONS}


class SearchAwareSampler:
    """Load a fixed C.3 checkpoint; generate as-is once per requested condition."""
    def __init__(self, checkpoint, device="cpu", *, freeze_path=None):
        start = time.perf_counter()
        _configure()
        self.checkpoint_path = Path(checkpoint).resolve()
        self.checkpoint_sha256 = sha256(self.checkpoint_path)
        c = torch.load(self.checkpoint_path, map_location="cpu", weights_only=False)
        if c.get("schema") != CHECKPOINT_SCHEMA or c.get("model_name") not in ("D", "S") or c.get("update") not in CHECKPOINT_UPDATES:
            raise ValueError("not a predeclared C.3 D/S checkpoint")
        for path, expected in c["training_config"]["source_files"].items():
            if sha256(path) != expected:
                raise ValueError("model production source changed: " + path)
        if freeze_path is not None:
            frozen = _read(freeze_path)
            for path, expected in frozen["artifacts"].items():
                if sha256(path) != expected:
                    raise ValueError("frozen model/scaler artifact changed")
            selected = frozen.get("selected_checkpoints", {}).get(c["model_name"], frozen.get("selected_checkpoint"))
            if selected != str(self.checkpoint_path):
                raise ValueError("checkpoint differs from pre-TEST selected artifact")
        self.device, self.model_name = torch.device(device), c["model_name"]
        self.model = (ResidualDDPM(ResidualDiffusionConfig.from_dict(c["model_config"])) if self.model_name == "D" else
            SimpleWarmstartRegression(c["model_config"]["condition_dim"])).to(self.device)
        self.model.load_state_dict(c["state_dict"], strict=True)
        self.model.eval()
        if _state_sha(self.model) != c["state_sha256"]:
            raise ValueError("checkpoint tensors differ from retained identity")
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.checkpoint = c
        self.condition_normalizer = ConditionNormalizer.from_dict(c["condition_normalizer"])
        self.residual_normalizer = ResidualNormalizer.from_dict(c["residual_normalizer"])
        self.identity = {"checkpoint_sha256": self.checkpoint_sha256, "checkpoint_update": c["update"],
            "model": self.model_name, "training_config_sha256": c["training_config_sha256"],
            "condition_scaler_sha256": object_sha(c["condition_normalizer"]),
            "residual_scaler_sha256": object_sha(c["residual_normalizer"]),
            "condition_schema_sha256": object_sha(c["condition_schema"]),
            "effective_parameter_count": c["effective_parameter_count"]}
        self.model_load_wall_s = time.perf_counter() - start
        self.sample_units = 0

    @torch.no_grad()
    def sample(self, task, preference, family, initial_noise_np=None):
        definition = build_reference_definition(task, version=VERSIONS[family])
        mask = search_mask(definition)
        key = condition_key(preference, family, mask)
        metadata = {"source": "diffusion" if self.model_name == "D" else "regression",
            "preference": preference, "family": family, **self.identity, "physics_steps": 0,
            "raw_postprocessing": "inverse TRAIN scaler only", "raw_repaired": False, "resampled": False}
        if key not in self.checkpoint["supported_conditions"]:
            return None, {**metadata, "initializer_rejection": "UNSUPPORTED_TRAINING_CONDITION",
                "ddim_sample_units": 0, "regression_forward_units": 0}
        noise = None
        if self.model_name == "D":
            noise = np.asarray(initial_noise_np, dtype=np.float32)
            if noise.shape not in ((12,), (6, 2)) or not np.isfinite(noise).all():
                raise ValueError("finite fixed 12-D initial noise required")
        elif initial_noise_np is not None:
            raise ValueError("deterministic regression has no diffusion-noise input")
        start = time.perf_counter()
        encoded = self.condition_normalizer.transform(task, definition, preference, family, mask)
        condition = torch.as_tensor(encoded[None], device=self.device)
        mask_tensor = torch.as_tensor(mask[None], device=self.device)
        prepared = time.perf_counter() - start
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        start = time.perf_counter()
        latent = (self.model.sample_ddim(condition, mask_tensor, initial_noise=noise.reshape(1, 12)) if self.model_name == "D" else
                  self.model(condition, mask_tensor))
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        elapsed = time.perf_counter() - start
        raw = inverse_raw(self.residual_normalizer, latent.cpu().numpy(), mask)
        self.sample_units += 1
        extra = {"noise_sha256": hashlib.sha256(noise.tobytes()).hexdigest(),
            "masked_initial_noise_sha256": hashlib.sha256(np.where(np.repeat(mask, 2), noise.reshape(12), 0.).tobytes()).hexdigest()} if noise is not None else {}
        return raw, {**metadata, **extra, "condition_prepare_wall_s": prepared, "inference_wall_s": elapsed,
            "ddim_sample_units": int(self.model_name == "D"), "regression_forward_units": int(self.model_name == "S")}

    def initializer_proposals(self, task, noise_seed=64324):
        noises = frozen_noise(task, noise_seed) if self.model_name == "D" else {}
        result = {}
        for slot, (preference, family) in OVERRIDE_CONDITIONS.items():
            raw, metadata = self.sample(task, preference, family, noises.get(slot))
            result[slot] = {**metadata, "raw_z_m": None if raw is None else json_raw(raw.reshape(12)),
                "noise_seed": noise_seed if self.model_name == "D" else None,
                "noise_id": f"{task.sha256()}/{preference}/{family}/{noise_seed}" if self.model_name == "D" else None,
                "noise_derivation": NOISE_DERIVATION if self.model_name == "D" else None}
        return result


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "train"))
    parser.add_argument("--output", required=True)
    parser.add_argument("--dataset")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.action == "prepare":
        if not args.dataset:
            parser.error("--dataset is required")
        result = prepare_training_pair(args.dataset, args.output, device=args.device)
    else:
        result = train_model_pair(args.output)
    print(json.dumps(result, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
