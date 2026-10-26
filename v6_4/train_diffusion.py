"""Train the first conditional DDPM and sample raw 30x17 spline proposals.

Only complete, independently successful nominal teachers enter optimization.
Wall time is reported, not an acceptance deadline.  Sampling performs no clipping,
trajectory repair, simulation selection, feedback, or force/torque generation.
"""
from __future__ import annotations

import argparse
import copy
import json
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

from .dataset import (CONDITION_SCHEMA, REPRESENTATION, TrainingNormalizer,
                      canonical, encode_condition, load_teacher_dataset,
                      object_sha, sha256, task_sha)
from .diffusion_model import (ConditionalDDPM, DiffusionConfig, noise_schedule_identity,
                              parameterization_identity, objective_identity)

CHECKPOINT_SCHEMA = "v6_4_conditional_ddpm_checkpoint_v6"
EXPLICIT_OBJECTIVE_CHECKPOINT_SCHEMA = "v6_4_conditional_ddpm_checkpoint_v5"
PREVIOUS_CHECKPOINT_SCHEMA = "v6_4_conditional_ddpm_checkpoint_v3"
WEIGHT_SELECTION_CHECKPOINT_SCHEMA = "v6_4_conditional_ddpm_checkpoint_v4"
LEGACY_CHECKPOINT_SCHEMAS = ("v6_4_conditional_ddpm_checkpoint_v1", "v6_4_conditional_ddpm_checkpoint_v2")


def _write_json(path: Path, value: dict):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def _device(value: str) -> torch.device:
    if value == "auto":
        value = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(value)
    if device.type not in ("cuda", "cpu") or (device.type == "cuda" and not torch.cuda.is_available()):
        raise ValueError("learning device must be available CPU or CUDA")
    return device


def _source_identity() -> dict:
    repo = Path(__file__).resolve().parents[1]
    identities = {}
    for name in ("dataset.py", "diffusion_model.py", "train_diffusion.py", "trajectory_codec.py", "contracts.py", "task_protocol.py"):
        path = Path(__file__).resolve().parent / name
        identities[str(path.relative_to(repo))] = sha256(path)
    dependency = repo / "requirements-learning.txt"
    if dependency.is_file():
        identities[str(dependency.relative_to(repo))] = sha256(dependency)
    def git(args):
        result = subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True,
                                encoding="utf-8", errors="replace", check=True)
        return result.stdout.strip()
    return {"git_head": git(["rev-parse", "HEAD"]),
            "tracked_dirty": bool(git(["status", "--porcelain", "--untracked-files=no"])),
            "learning_sources_sha256": identities,
            "scope": "executed learning modules and isolated learning requirement; other controller source is not revalidated"}


def _source_unchanged(before: dict, after: dict) -> bool:
    return before["git_head"] == after["git_head"] and before["learning_sources_sha256"] == after["learning_sources_sha256"]


def _input_identity(paths: list[Path]) -> dict:
    return {str(p): {"sha256": sha256(p), "bytes": p.stat().st_size} for p in paths}


def _objective_fields(config):
    identity = objective_identity(config)
    return {"objective": config.objective, "objective_identity": identity,
            "objective_sha256": identity["objective_sha256"]}


def train_from_manifest(manifest_path: Path, output_dir: Path, *, epochs: int = 100,
                        batch_size: int = 8, seed: int = 64, device: str = "auto",
                        learning_rate: float = 1e-4, cpu_threads: int = 4) -> dict:
    """Run real epsilon-MSE optimization; freeze weights, split and provenance.

    Small data is allowed for an explicitly documented overfit chain smoke.  No
    threshold on training loss is interpreted as independent planning success.
    Validation is diagnostic only and never updates weights or normalization.
    """
    integers = (epochs, batch_size, seed, cpu_threads)
    if any(not isinstance(x, int) or isinstance(x, bool) for x in integers) or min(epochs, batch_size, cpu_threads) <= 0 or seed < 0:
        raise ValueError("positive epochs/batch/threads and nonnegative integer seed required")
    if not np.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("learning rate must be positive and finite")
    source_before = _source_identity()
    dataset = load_teacher_dataset(Path(manifest_path))
    normalizer = TrainingNormalizer.fit(dataset)
    train_indices, val_indices = dataset.indices("train"), dataset.indices("val")
    selected_device = _device(device)
    output_dir = Path(output_dir).resolve()
    if any(path.is_relative_to(output_dir) for path in dataset.source_files) or Path(__file__).resolve().is_relative_to(output_dir):
        raise ValueError("training output must not contain or overwrite source inputs")
    output_dir.mkdir(parents=True, exist_ok=False)
    inputs_before = _input_identity(dataset.source_files)
    config = DiffusionConfig(condition_dim=dataset.conditions.shape[1])
    schedule = noise_schedule_identity(config)
    parameterization = parameterization_identity(config)
    weight_selection = {"schema": "v6_4_predeclared_weight_selection_v1", "weights": "last_epoch_raw",
                        "last_epoch": epochs, "ema_used": False, "selection_predeclared": True}
    run_config = {"schema": "v6_4_ddpm_training_plan_v5", "model": config.to_dict(), "noise_schedule": schedule,
                  **_objective_fields(config),
                  "weight_selection": weight_selection,
                  "parameterization_identity": parameterization,
                  "manifest_path": str(dataset.manifest_path), "manifest_sha256": dataset.manifest_sha256,
                  "epochs": epochs, "batch_size": batch_size, "learning_rate": learning_rate,
                  "seed": seed, "device": str(selected_device), "cpu_threads": cpu_threads,
                  "optimizer": "AdamW", "weight_decay": 0., "gradient_norm_clip": 1.,
                  "tf32": False, "training_noise_seed": seed + 1, "order_seed": seed + 2,
                  "validation_noise_seed": seed + 3,
                  "model_contract_sha256": dataset.samples[0]["task"]["model_contract_sha256"],
                  "condition_schema": CONDITION_SCHEMA, "representation": REPRESENTATION,
                  "teacher_counts": {k: len(dataset.indices(k)) for k in ("train", "val", "test")},
                  "source_before": source_before, "input_files_before": inputs_before,
                  "wall_deadline_required": False, "test_used_for_optimization": False,
                  "normalizer_train_only": True, "candidate_repair_or_projection": False,
                  "scope": "real successful-teacher training; loss is not a held-out closed-loop result"}
    _write_json(output_dir / "config.json", run_config)
    _write_json(output_dir / "split.json", dataset.split)
    _write_json(output_dir / "normalizer.json", normalizer.to_dict())
    versions = {"python": sys.version, "numpy": np.__version__, "torch": str(torch.__version__),
                "cuda_runtime": torch.version.cuda, "cuda_available": torch.cuda.is_available(),
                "device": str(selected_device),
                "gpu": torch.cuda.get_device_name(selected_device) if selected_device.type == "cuda" else None}
    _write_json(output_dir / "environment.json", versions)
    torch.set_num_threads(cpu_threads)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # TF32 is explicitly disabled so saved settings are reproducible across runs.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = ConditionalDDPM(config).to(selected_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.)
    controls = torch.from_numpy(normalizer.normalize_controls(dataset.controls[train_indices])).to(selected_device)
    conditions = torch.from_numpy(normalizer.normalize_conditions(dataset.conditions[train_indices])).to(selected_device)
    val_controls = torch.from_numpy(normalizer.normalize_controls(dataset.controls[val_indices])).to(selected_device)
    val_conditions = torch.from_numpy(normalizer.normalize_conditions(dataset.conditions[val_indices])).to(selected_device)
    noise_generator = torch.Generator(device=selected_device).manual_seed(seed + 1)
    order_generator = torch.Generator(device="cpu").manual_seed(seed + 2)
    start, step, rows = time.perf_counter(), 0, []
    try:
        with (output_dir / "training_log.jsonl").open("x", encoding="utf-8") as stream:
            for epoch in range(epochs):
                model.train()
                permutation = torch.randperm(len(train_indices), generator=order_generator)
                loss_sum = 0.
                for first in range(0, len(permutation), batch_size):
                    indices = permutation[first:first + batch_size].to(selected_device)
                    optimizer.zero_grad(set_to_none=True)
                    loss = model.epsilon_loss(controls[indices], conditions[indices], generator=noise_generator)
                    if not torch.isfinite(loss):
                        raise ValueError("nonfinite training loss")
                    loss.backward()
                    gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                    if not torch.isfinite(gradient_norm):
                        raise ValueError("nonfinite training gradient")
                    optimizer.step()
                    step += 1
                    loss_sum += float(loss.detach()) * len(indices)
                validation_loss = None
                if len(val_indices):
                    model.eval()
                    # Recreate a fixed validation noise stream each epoch; held-out
                    # data neither fits the normalizer nor selects a checkpoint.
                    validation_generator = torch.Generator(device=selected_device).manual_seed(seed + 3)
                    total = 0.
                    with torch.no_grad():
                        for first in range(0, len(val_indices), batch_size):
                            value = model.epsilon_loss(val_controls[first:first + batch_size],
                                val_conditions[first:first + batch_size], generator=validation_generator)
                            if not torch.isfinite(value):
                                raise ValueError("nonfinite validation loss")
                            total += float(value) * len(val_controls[first:first + batch_size])
                    validation_loss = total / len(val_indices)
                row = {"epoch": epoch + 1, "optimizer_steps": step,
                       "train_epsilon_mse": loss_sum / len(train_indices),
                       "val_epsilon_mse": validation_loss, "elapsed_s": time.perf_counter() - start}
                rows.append(row)
                stream.write(json.dumps(row, allow_nan=False) + "\n")
                stream.flush()
        if selected_device.type == "cuda":
            torch.cuda.synchronize(selected_device)
        source_after, inputs_after = _source_identity(), _input_identity(dataset.source_files)
        source_stable = _source_unchanged(source_before, source_after)
        inputs_stable = inputs_before == inputs_after
        if not source_stable or not inputs_stable:
            raise ValueError("learning source/HEAD or frozen teacher input changed during training")
        if model.schedule_identity() != schedule:
            raise ValueError("effective diffusion schedule changed during training")
        checkpoint = {"schema": CHECKPOINT_SCHEMA, "model_config": config.to_dict(),
                      **_objective_fields(config),
                      "weight_selection": weight_selection, "weight_selection_sha256": object_sha(weight_selection),
                      "noise_schedule": model.schedule_identity(),
                      "parameterization_identity": model.parameterization_identity(),
                      "parameterization_sha256": parameterization["parameterization_sha256"],
                      "schedule_sha256": schedule["schedule_sha256"], "terminal_alpha_bar": schedule["terminal_alpha_bar"],
                      "model_state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                      "normalizer": normalizer.to_dict(), "split": dataset.split,
                      "manifest_sha256": dataset.manifest_sha256, "seed": seed,
                      "optimizer_steps": step, "source_identity": source_before,
                      "normalizer_sha256": sha256(output_dir / "normalizer.json"),
                      "split_sha256": sha256(output_dir / "split.json"),
                      "config_sha256": sha256(output_dir / "config.json"),
                      "model_contract_sha256": run_config["model_contract_sha256"],
                      "representation": REPRESENTATION, "condition_schema": CONDITION_SCHEMA}
        torch.save(checkpoint, output_dir / "checkpoint.pt")
        report = {"schema": "v6_4_ddpm_training_report_v5", "complete": True, "evidence_valid": True,
                  **_objective_fields(config),
                  "weight_selection": weight_selection, "weight_selection_sha256": object_sha(weight_selection),
                  "noise_schedule": model.schedule_identity(),
                  "parameterization_identity": model.parameterization_identity(),
                  "parameterization_sha256": parameterization["parameterization_sha256"],
                  "schedule_sha256": schedule["schedule_sha256"], "terminal_alpha_bar": schedule["terminal_alpha_bar"],
                  "optimizer_steps": step, "epochs": epochs, "train_samples": len(train_indices),
                  "val_samples": len(val_indices), "test_samples_used_for_optimization": 0,
                  "first_train_epsilon_mse": rows[0]["train_epsilon_mse"],
                  "last_train_epsilon_mse": rows[-1]["train_epsilon_mse"],
                  "last_val_epsilon_mse": rows[-1]["val_epsilon_mse"],
                  "elapsed_s": time.perf_counter() - start, "source_before": source_before,
                  "source_after": source_after, "source_unchanged": source_stable,
                  "input_files_after": inputs_after, "inputs_unchanged": inputs_stable,
                  "checkpoint_sha256": sha256(output_dir / "checkpoint.pt"),
                  "training_log_sha256": sha256(output_dir / "training_log.jsonl"),
                  "normalizer_sha256": checkpoint["normalizer_sha256"],
                  "split_sha256": checkpoint["split_sha256"], "config_sha256": checkpoint["config_sha256"],
                  "scope": "trained model only; independent raw-proposal and physical completion must be evaluated separately"}
        _write_json(output_dir / "training_report.json", report)
        artifacts = sorted(p for p in output_dir.iterdir() if p.is_file())
        _write_json(output_dir / "artifact_manifest.json", {"schema": "v6_4_training_artifacts_v1",
            "files": [{"path": p.name, "sha256": sha256(p), "bytes": p.stat().st_size} for p in artifacts]})
        return report
    except BaseException as exc:
        failure = {"complete": False, "evidence_valid": False, "optimizer_steps": step,
                   "error": f"{type(exc).__name__}: {exc}", "elapsed_s": time.perf_counter() - start,
                   "partial_log_retained": (output_dir / "training_log.jsonl").exists(),
                   "scope": "failed training; no qualified checkpoint"}
        _write_json(output_dir / "training_failure.json", failure)
        raise


class DiffusionSampler:
    def __init__(self, checkpoint_path: Path, device: str = "auto"):
        self.checkpoint_path = Path(checkpoint_path).resolve()
        self.checkpoint_sha256 = sha256(self.checkpoint_path)
        self.device = _device(device)
        checkpoint = torch.load(self.checkpoint_path, map_location="cpu", weights_only=True)
        if checkpoint.get("schema") not in (CHECKPOINT_SCHEMA, EXPLICIT_OBJECTIVE_CHECKPOINT_SCHEMA, WEIGHT_SELECTION_CHECKPOINT_SCHEMA, PREVIOUS_CHECKPOINT_SCHEMA, *LEGACY_CHECKPOINT_SCHEMAS) or checkpoint.get("representation") != REPRESENTATION or checkpoint.get("condition_schema") != CONDITION_SCHEMA:
            raise ValueError("not a qualified V6.4-A diffusion checkpoint")
        legacy = checkpoint["schema"] in LEGACY_CHECKPOINT_SCHEMAS
        old_linear = checkpoint["schema"] == LEGACY_CHECKPOINT_SCHEMAS[0]
        self.config = DiffusionConfig.from_dict(checkpoint["model_config"], allow_legacy_missing_schedule=old_linear,
                                                allow_legacy_missing_parameterization=legacy,
                                                allow_legacy_missing_objective=checkpoint["schema"] not in (CHECKPOINT_SCHEMA, EXPLICIT_OBJECTIVE_CHECKPOINT_SCHEMA))
        if checkpoint["schema"] not in (CHECKPOINT_SCHEMA, EXPLICIT_OBJECTIVE_CHECKPOINT_SCHEMA) and self.config.objective != "epsilon_mse":
            raise ValueError("historical checkpoint must not reinterpret epsilon training as v_mse")
        if checkpoint["schema"] != CHECKPOINT_SCHEMA and self.config.parameterization == "clean_x0":
            raise ValueError("historical checkpoint must not reinterpret weights as clean_x0")
        if legacy and self.config.parameterization != "direct_epsilon":
            raise ValueError("legacy direct-epsilon checkpoint must not reinterpret weights as residual")
        self.normalizer = TrainingNormalizer.from_dict(checkpoint["normalizer"])
        if self.config.condition_dim != len(self.normalizer.condition_mean):
            raise ValueError("checkpoint condition/normalizer dimensions differ")
        if set(checkpoint["split"]["sample_ids"]["train"]) != set(self.normalizer.training_sample_ids):
            raise ValueError("checkpoint normalization does not bind exactly its train split")
        for filename, field in (("normalizer.json", "normalizer_sha256"), ("split.json", "split_sha256"), ("config.json", "config_sha256")):
            path = self.checkpoint_path.parent / filename
            if not path.is_file() or sha256(path) != checkpoint[field]:
                raise ValueError("checkpoint training companion identity changed: " + filename)
        if json.loads((self.checkpoint_path.parent / "normalizer.json").read_text(encoding="utf-8")) != checkpoint["normalizer"] or json.loads((self.checkpoint_path.parent / "split.json").read_text(encoding="utf-8")) != checkpoint["split"]:
            raise ValueError("checkpoint embedded normalization/split differs from recorded artifacts")
        if checkpoint["schema"] in (CHECKPOINT_SCHEMA, EXPLICIT_OBJECTIVE_CHECKPOINT_SCHEMA, WEIGHT_SELECTION_CHECKPOINT_SCHEMA):
            self.weight_selection = checkpoint.get("weight_selection")
            if (not isinstance(self.weight_selection, dict)
                    or self.weight_selection.get("weights") not in ("last_epoch_raw", "last_step_ema")
                    or self.weight_selection.get("selection_predeclared") is not True
                    or checkpoint.get("weight_selection_sha256") != object_sha(self.weight_selection)
                    or json.loads((self.checkpoint_path.parent / "config.json").read_text(encoding="utf-8")).get("weight_selection") != self.weight_selection):
                raise ValueError("checkpoint predeclared weight selection is inconsistent")
        else:
            self.weight_selection = {"weights": "historical_last_epoch_raw", "legacy_metadata": True}
        self.objective = objective_identity(self.config)
        self.model = ConditionalDDPM(self.config).to(self.device)
        expected_betas, expected_alphas = self.model.betas.clone(), self.model.alpha_bars.clone()
        self.model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        if not torch.equal(self.model.betas, expected_betas) or not torch.equal(self.model.alpha_bars, expected_alphas):
            raise ValueError("checkpoint schedule buffers differ from its explicit model configuration")
        self.schedule = self.model.schedule_identity()
        if not old_linear and checkpoint.get("noise_schedule") != self.schedule:
            raise ValueError("checkpoint effective schedule identity is inconsistent")
        if not old_linear and (checkpoint.get("schedule_sha256") != self.schedule["schedule_sha256"]
                           or checkpoint.get("terminal_alpha_bar") != self.schedule["terminal_alpha_bar"]):
            raise ValueError("checkpoint terminal/schedule summary differs from its buffers")
        self.legacy_missing_schedule = legacy and "noise_schedule" not in checkpoint["model_config"]
        self.parameterization = self.model.parameterization_identity()
        if not legacy and (checkpoint.get("parameterization_identity") != self.parameterization
                or checkpoint.get("parameterization_sha256") != self.parameterization["parameterization_sha256"]):
            raise ValueError("checkpoint epsilon parameterization identity is inconsistent")
        if checkpoint["schema"] in (CHECKPOINT_SCHEMA, EXPLICIT_OBJECTIVE_CHECKPOINT_SCHEMA):
            plan = json.loads((self.checkpoint_path.parent / "config.json").read_text(encoding="utf-8"))
            expected = _objective_fields(self.config)
            if (any(checkpoint.get(k) != v or plan.get(k) != v for k, v in expected.items())
                    or plan.get("model") != checkpoint["model_config"]):
                raise ValueError("checkpoint explicit training objective identity is inconsistent")
        self.legacy_missing_parameterization = legacy and "parameterization" not in checkpoint["model_config"]
        self.legacy_missing_objective = "objective" not in checkpoint["model_config"]
        if not all(torch.all(torch.isfinite(value)) for value in self.model.state_dict().values()):
            raise ValueError("checkpoint has nonfinite weights")
        self.model.eval()
        self.source_identity = checkpoint["source_identity"]
        self.manifest_sha256 = checkpoint["manifest_sha256"]
        self.model_contract_sha256 = checkpoint["model_contract_sha256"]

    def sample(self, task, *, K: int = 1, seed: int = 64, ddim_steps: int = 20):
        if not isinstance(K, int) or isinstance(K, bool) or K <= 0 or not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
            raise ValueError("positive integer K and nonnegative seed required")
        condition, names = encode_condition(task, return_feature_names=True)
        task_value = task.to_dict() if hasattr(task, "to_dict") else task
        if task_value["model_contract_sha256"] != self.model_contract_sha256:
            raise ValueError("proposal task changes the trained nominal model contract")
        if names != self.normalizer.feature_names:
            raise ValueError("inference condition schema differs from frozen training")
        normalized = self.normalizer.normalize_conditions(condition)
        condition_tensor = torch.from_numpy(np.repeat(normalized[None], K, axis=0)).to(self.device)
        generator = torch.Generator(device=self.device).manual_seed(seed)
        source_before = _source_identity()
        if source_before["learning_sources_sha256"] != self.source_identity["learning_sources_sha256"]:
            raise ValueError("sampling learning modules differ from checkpoint training source")
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        start = time.perf_counter()
        raw = self.model.sample_ddim(condition_tensor, steps=ddim_steps, generator=generator).cpu().numpy()
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        elapsed = time.perf_counter() - start
        free = self.normalizer.denormalize_controls(raw)
        source_after = _source_identity()
        if not _source_unchanged(source_before, source_after) or sha256(self.checkpoint_path) != self.checkpoint_sha256:
            raise ValueError("sampling source/HEAD or checkpoint changed during generation")
        metadata = {"schema": "v6_4_raw_diffusion_proposals_v1", "origin": "diffusion",
                    "checkpoint_sha256": self.checkpoint_sha256, "training_manifest_sha256": self.manifest_sha256,
                    "training_source_identity": self.source_identity, "task_sha256": task_sha(task),
                    "sampling_source_before": source_before, "sampling_source_after": source_after,
                    "sampling_source_unchanged": True, "model_contract_sha256": self.model_contract_sha256,
                    "seed": seed, "K": K, "ddpm_training_steps": self.config.diffusion_steps,
                    "ddim_inference_steps": ddim_steps, "device": str(self.device), "sampling_elapsed_s": elapsed,
                    "representation": REPRESENTATION, "postprocessing": [],
                    "noise_schedule": self.schedule, "legacy_config_missing_schedule": self.legacy_missing_schedule,
                    "parameterization_identity": self.parameterization,
                    "weight_selection": self.weight_selection,
                    **_objective_fields(self.config),
                    "legacy_config_missing_objective": self.legacy_missing_objective,
                    "parameterization_sha256": self.parameterization["parameterization_sha256"],
                    "legacy_config_missing_parameterization": self.legacy_missing_parameterization,
                    "clipping_projection_or_trajectory_optimization": False,
                    "simulation_candidate_selection": False, "force_or_torque_output": False,
                    "scope": "raw free spline controls; safety/task gates and closed-loop execution remain required"}
        return free, metadata


@torch.no_grad()
def _update_ema(ema_model, model, decay):
    """Fixed-decay parameter EMA; schedule buffers are never averaged."""
    left, right = dict(ema_model.named_parameters()), dict(model.named_parameters())
    if left.keys() != right.keys():
        raise ValueError("EMA parameter identities differ")
    for name in left:
        left[name].mul_(decay).add_(right[name].detach(), alpha=1 - decay)


def _verified_parent(checkpoint_path: Path, dataset, device):
    """Permit optimizer changes only; freeze the parent's model/data meaning."""
    checkpoint_path = checkpoint_path.resolve()
    parent = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if parent.get("schema") not in (PREVIOUS_CHECKPOINT_SCHEMA, WEIGHT_SELECTION_CHECKPOINT_SCHEMA, EXPLICIT_OBJECTIVE_CHECKPOINT_SCHEMA, CHECKPOINT_SCHEMA):
        raise ValueError("pure epsilon continuation requires an explicit residual parent checkpoint")
    sampler = DiffusionSampler(checkpoint_path, device=str(device))
    if sampler.config.parameterization != "epsilon_residual" or sampler.config.noise_schedule != "cosine" or sampler.config.objective != "epsilon_mse":
        raise ValueError("parent must be cosine100 epsilon_residual; do not reinterpret weights")
    report_path = checkpoint_path.parent / "training_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if (not report.get("complete") or not report.get("evidence_valid")
            or not report.get("source_unchanged") or not report.get("inputs_unchanged")
            or report.get("checkpoint_sha256") != sampler.checkpoint_sha256
            or report.get("source_before") != parent["source_identity"]):
        raise ValueError("parent training evidence is not complete and identity bound")
    if (parent["manifest_sha256"] != dataset.manifest_sha256 or parent["split"] != dataset.split
            or tuple(sorted(sampler.normalizer.training_sample_ids)) != tuple(sorted(s["sample_id"] for s in dataset.samples))):
        raise ValueError("continuation must use exactly the parent's successful TRAIN dataset")
    if any(s["split"] != "train" for s in dataset.samples):
        raise ValueError("optimization sanity must not read VAL/TEST teacher datasets")
    config = json.loads((checkpoint_path.parent / "config.json").read_text(encoding="utf-8"))
    if config["input_files_before"] != _input_identity(dataset.source_files):
        raise ValueError("parent frozen teacher source identities changed")
    archive = checkpoint_path.parent / "source_snapshot"
    receipt_path = archive / "source_archive_receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt["source_identity"] != parent["source_identity"] or receipt["checkpoint_sha256"] != sampler.checkpoint_sha256:
        raise ValueError("parent source archive receipt does not bind its checkpoint")
    current = _source_identity()["learning_sources_sha256"]
    for relative, expected in parent["source_identity"]["learning_sources_sha256"].items():
        path = archive / relative
        if not path.is_file() or sha256(path) != expected:
            raise ValueError("parent learning source archive changed: " + relative)
        # Only this optimizer/loader file may change. Dataset, model math,
        # normalizer, representation and immutable task semantics stay exact.
        if Path(relative).name != "train_diffusion.py" and current.get(relative) != expected:
            raise ValueError("parent model/data source semantics changed: " + relative)
    identity = {"checkpoint_path": str(checkpoint_path), "checkpoint_sha256": sampler.checkpoint_sha256,
        "training_report_sha256": sha256(report_path), "source_identity": parent["source_identity"],
        "source_archive_receipt_sha256": sha256(receipt_path), "optimizer_steps": parent["optimizer_steps"],
        "initialization_only_not_sampling_old_weights_under_new_source": True,
        "model_data_sources_exact": True, "optimizer_state_reset": True}
    return sampler, identity


def optimize_epsilon_from_manifest(manifest_path: Path, parent_checkpoint: Path, output_dir: Path, *,
        optimizer_steps: int = 5000, noise_repeats: int = 32, seed: int = 67,
        device: str = "cuda", learning_rate: float = 3e-5, ema_decay: float = .999,
        cpu_threads: int = 4) -> dict:
    """One finite TRAIN-only pure-epsilon optimization sanity from old weights.

    Every successful label receives noise_repeats independent noise/time draws
    at EVERY optimizer step. This is not an epoch batch larger than the dataset.
    Canonical checkpoint weights are the predeclared final fixed-decay EMA.
    """
    if any(not isinstance(x, int) or isinstance(x, bool) for x in (optimizer_steps, noise_repeats, seed, cpu_threads)) or min(optimizer_steps, noise_repeats, cpu_threads) <= 0 or seed < 0:
        raise ValueError("positive integer steps/noise repeats/threads and nonnegative seed required")
    if not np.isfinite(learning_rate) or learning_rate <= 0 or not np.isfinite(ema_decay) or not 0 < ema_decay < 1:
        raise ValueError("invalid learning rate or EMA decay")
    source_before = _source_identity()
    dataset = load_teacher_dataset(Path(manifest_path))
    selected_device = _device(device)
    parent_sampler, parent_identity = _verified_parent(Path(parent_checkpoint), dataset, selected_device)
    normalizer, model = parent_sampler.normalizer, parent_sampler.model
    config = model.config
    source_files = sorted(set(dataset.source_files) | {p.resolve() for p in Path(parent_checkpoint).resolve().parent.rglob("*") if p.is_file()})
    inputs_before = _input_identity(source_files)
    output_dir = Path(output_dir).resolve()
    if any(p.is_relative_to(output_dir) for p in source_files) or Path(__file__).resolve().is_relative_to(output_dir):
        raise ValueError("optimization output must not contain parent or frozen inputs")
    output_dir.mkdir(parents=True, exist_ok=False)
    count = len(dataset.samples)
    effective_batch = count * noise_repeats
    selection = {"schema": "v6_4_predeclared_weight_selection_v1", "weights": "last_step_ema",
        "last_optimizer_step": optimizer_steps, "ema_used": True, "ema_decay": ema_decay,
        "ema_initialization": "verified_parent_weights", "ema_warmup": False, "ema_bias_correction": False,
        "selection_predeclared": True, "test_or_validation_checkpoint_selection": False}
    schedule, parameterization = model.schedule_identity(), model.parameterization_identity()
    plan = {"schema": "v6_4_pure_epsilon_optimization_plan_v1", "model": config.to_dict(),
        **_objective_fields(config),
        "noise_schedule": schedule, "parameterization_identity": parameterization, "weight_selection": selection,
        "parent": parent_identity, "manifest_path": str(dataset.manifest_path), "manifest_sha256": dataset.manifest_sha256,
        "optimizer_steps": optimizer_steps, "epochs": None, "distinct_successful_train_samples": count,
        "noise_repeats_per_label_per_step": noise_repeats, "effective_noise_batch": effective_batch,
        "independent_noise_draws_total": optimizer_steps * effective_batch, "optimizer": "fresh AdamW",
        "learning_rate": learning_rate, "weight_decay": 0., "gradient_norm_clip": 1., "seed": seed,
        "training_noise_seed": seed + 1, "device": str(selected_device), "cpu_threads": cpu_threads,
        "tf32": False, "loss": "basic_unweighted_true_forward_epsilon_MSE", "timestep_distribution": "independent_uniform_0_to99",
        "teacher_counts": {"train": count, "val": 0, "test": 0}, "normalizer_refitted": False,
        "model_contract_sha256": parent_sampler.model_contract_sha256, "condition_schema": CONDITION_SCHEMA,
        "representation": REPRESENTATION, "source_before": source_before, "input_files_before": inputs_before,
        "wall_deadline_required": False, "test_used_for_optimization": False, "candidate_repair_or_projection": False,
        "scope": "one finite pure-epsilon optimization sanity; TRAIN fit does not establish VAL or physical success"}
    _write_json(output_dir / "config.json", plan)
    _write_json(output_dir / "split.json", dataset.split)
    _write_json(output_dir / "normalizer.json", normalizer.to_dict())
    _write_json(output_dir / "environment.json", {"python": sys.version, "numpy": np.__version__, "torch": str(torch.__version__),
        "cuda_runtime": torch.version.cuda, "device": str(selected_device),
        "gpu": torch.cuda.get_device_name(selected_device) if selected_device.type == "cuda" else None})
    torch.set_num_threads(cpu_threads)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    controls = torch.from_numpy(normalizer.normalize_controls(dataset.controls)).to(selected_device).repeat_interleave(noise_repeats, dim=0)
    conditions = torch.from_numpy(normalizer.normalize_conditions(dataset.conditions)).to(selected_device).repeat_interleave(noise_repeats, dim=0)
    ema_model = copy.deepcopy(model).eval().requires_grad_(False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.)
    generator = torch.Generator(device=selected_device).manual_seed(seed + 1)
    step, start, rows = 0, time.perf_counter(), []
    try:
        with (output_dir / "training_log.jsonl").open("x", encoding="utf-8") as stream:
            for step in range(1, optimizer_steps + 1):
                model.train(); optimizer.zero_grad(set_to_none=True)
                # Exactly effective_batch independent epsilons AND timesteps,
                # not count labels despite a nominal batch size of 192.
                loss = model.epsilon_loss(controls, conditions, generator=generator)
                if not torch.isfinite(loss): raise ValueError("nonfinite optimization loss")
                loss.backward()
                gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                if not torch.isfinite(gradient): raise ValueError("nonfinite optimization gradient")
                optimizer.step(); _update_ema(ema_model, model, ema_decay)
                row = {"optimizer_step": step, "distinct_train_labels": count, "independent_noise_examples": effective_batch,
                    "train_epsilon_mse": float(loss.detach()), "ema_updates": step, "elapsed_s": time.perf_counter() - start}
                rows.append(row); stream.write(json.dumps(row, allow_nan=False) + "\n"); stream.flush()
        if selected_device.type == "cuda": torch.cuda.synchronize(selected_device)
        source_after, inputs_after = _source_identity(), _input_identity(source_files)
        if not _source_unchanged(source_before, source_after) or inputs_before != inputs_after:
            raise ValueError("optimization source/HEAD or parent/frozen teacher input changed")
        if ema_model.schedule_identity() != schedule or ema_model.parameterization_identity() != parameterization:
            raise ValueError("EMA model interpretation changed")
        cpu_state = lambda m: {k: v.detach().cpu() for k, v in m.state_dict().items()}
        torch.save({"schema": "v6_4_terminal_optimizer_weights_v1", "model_config": config.to_dict(),
            "model_state_dict": cpu_state(model), "optimizer_steps": step,
            "canonical_sampling_checkpoint": False, "parent_checkpoint_sha256": parent_identity["checkpoint_sha256"]},
            output_dir / "last_optimizer_weights.pt")
        torch.save(optimizer.state_dict(), output_dir / "optimizer_state.pt")
        checkpoint = {"schema": CHECKPOINT_SCHEMA, "model_config": config.to_dict(), "model_state_dict": cpu_state(ema_model),
            **_objective_fields(config),
            "weight_selection": selection, "weight_selection_sha256": object_sha(selection),
            "noise_schedule": schedule, "parameterization_identity": parameterization,
            "parameterization_sha256": parameterization["parameterization_sha256"], "schedule_sha256": schedule["schedule_sha256"],
            "terminal_alpha_bar": schedule["terminal_alpha_bar"], "normalizer": normalizer.to_dict(), "split": dataset.split,
            "manifest_sha256": dataset.manifest_sha256, "seed": seed, "optimizer_steps": step, "source_identity": source_before,
            "normalizer_sha256": sha256(output_dir / "normalizer.json"), "split_sha256": sha256(output_dir / "split.json"),
            "config_sha256": sha256(output_dir / "config.json"), "model_contract_sha256": parent_sampler.model_contract_sha256,
            "representation": REPRESENTATION, "condition_schema": CONDITION_SCHEMA, "parent": parent_identity}
        torch.save(checkpoint, output_dir / "checkpoint.pt")
        report = {"schema": "v6_4_pure_epsilon_optimization_report_v1", "complete": True, "evidence_valid": True,
            **_objective_fields(config),
            "weight_selection": selection, "weight_selection_sha256": object_sha(selection), "parent": parent_identity,
            "optimizer_steps": step, "epochs": None, "train_samples": count, "val_samples": 0,
            "test_samples_used_for_optimization": 0, "noise_repeats_per_label_per_step": noise_repeats,
            "effective_noise_batch": effective_batch, "independent_noise_draws_total": step * effective_batch,
            "loss": plan["loss"], "first_train_epsilon_mse": rows[0]["train_epsilon_mse"],
            "last_train_epsilon_mse": rows[-1]["train_epsilon_mse"], "last_val_epsilon_mse": None,
            "ema_parent_coefficient_at_end": ema_decay ** step, "ema_updates": step,
            "elapsed_s": time.perf_counter() - start, "source_before": source_before, "source_after": source_after,
            "source_unchanged": True, "inputs_unchanged": True, "input_files_after": inputs_after,
            "checkpoint_sha256": sha256(output_dir / "checkpoint.pt"), "training_log_sha256": sha256(output_dir / "training_log.jsonl"),
            "last_optimizer_weights_sha256": sha256(output_dir / "last_optimizer_weights.pt"),
            "optimizer_state_sha256": sha256(output_dir / "optimizer_state.pt"),
            "normalizer_sha256": checkpoint["normalizer_sha256"], "split_sha256": checkpoint["split_sha256"],
            "config_sha256": checkpoint["config_sha256"], "noise_schedule": schedule,
            "parameterization_identity": parameterization, "parameterization_sha256": parameterization["parameterization_sha256"],
            "schedule_sha256": schedule["schedule_sha256"], "terminal_alpha_bar": schedule["terminal_alpha_bar"],
            "scope": "last EMA of fixed-budget TRAIN-only optimization; loss/fit alone is not raw or closed-loop success"}
        _write_json(output_dir / "training_report.json", report)
        _write_json(output_dir / "artifact_manifest.json", {"files": [{"path": p.name, "sha256": sha256(p), "bytes": p.stat().st_size}
            for p in sorted(output_dir.iterdir()) if p.is_file()]})
        return report
    except BaseException as exc:
        _write_json(output_dir / "training_failure.json", {"complete": False, "evidence_valid": False, "optimizer_step": step,
            "error": f"{type(exc).__name__}: {exc}", "partial_log_retained": (output_dir / "training_log.jsonl").exists(),
            "scope": "failed finite optimization; no qualified EMA checkpoint"})
        raise


def train_v_from_manifest(manifest_path: Path, output_dir: Path, *, optimizer_steps: int = 5000,
        noise_repeats: int = 32, seed: int = 68, device: str = "cuda", learning_rate: float = 1e-4,
        ema_decay: float = .999, cpu_threads: int = 4, protected_inputs: tuple[Path, ...] = (),
        parameterization: str = "epsilon_residual") -> dict:
    """One from-scratch TRAIN-only v-MSE version, never reinterpreting EPS weights.

    Each label has independent noise/time repeats every step. The sole canonical
    checkpoint is the final EMA with predeclared decay min(cap,(1+s)/(10+s)).
    This changes the training objective; it is not basic epsilon-MSE training.
    """
    if any(not isinstance(x, int) or isinstance(x, bool) for x in (optimizer_steps, noise_repeats, seed, cpu_threads)) or min(optimizer_steps, noise_repeats, cpu_threads) <= 0 or seed < 0:
        raise ValueError("positive integer steps/noise repeats/threads and nonnegative seed required")
    if not np.isfinite(learning_rate) or learning_rate <= 0 or not np.isfinite(ema_decay) or not 0 < ema_decay < 1:
        raise ValueError("invalid learning rate or EMA decay")
    source_before = _source_identity()
    dataset = load_teacher_dataset(Path(manifest_path))
    if any(s["split"] != "train" for s in dataset.samples):
        raise ValueError("from-scratch v training requires successful TRAIN data only")
    normalizer = TrainingNormalizer.fit(dataset)
    config = DiffusionConfig(condition_dim=dataset.conditions.shape[1], objective="v_mse", parameterization=parameterization)
    if parameterization not in ("epsilon_residual", "clean_x0"):
        raise ValueError("from-scratch v training requires explicit residual or clean_x0")
    schedule, parameterization = noise_schedule_identity(config), parameterization_identity(config)
    selected_device = _device(device)
    source_files = sorted(set(dataset.source_files) | {Path(p).resolve() for p in protected_inputs})
    inputs_before = _input_identity(source_files)
    output_dir = Path(output_dir).resolve()
    if any(p.is_relative_to(output_dir) for p in source_files) or Path(__file__).resolve().is_relative_to(output_dir):
        raise ValueError("v training output must not contain or overwrite frozen inputs")
    output_dir.mkdir(parents=True, exist_ok=False)
    count, effective_batch = len(dataset.samples), len(dataset.samples) * noise_repeats
    selection = {"schema": "v6_4_predeclared_weight_selection_v1", "weights": "last_step_ema",
        "last_optimizer_step": optimizer_steps, "ema_used": True, "ema_decay": ema_decay,
        "ema_initialization": "fresh_random_model_weights", "ema_warmup": True,
        "ema_decay_formula": "min(ema_decay,(1+optimizer_step)/(10+optimizer_step))",
        "ema_bias_correction": False, "selection_predeclared": True,
        "test_or_validation_checkpoint_selection": False}
    plan = {"schema": "v6_4_from_scratch_v_training_plan_v1", "model": config.to_dict(),
        **_objective_fields(config), "noise_schedule": schedule, "parameterization_identity": parameterization,
        "weight_selection": selection, "initialization": "from_scratch_no_parent_checkpoint",
        "manifest_path": str(dataset.manifest_path), "manifest_sha256": dataset.manifest_sha256,
        "optimizer_steps": optimizer_steps, "epochs": None, "distinct_successful_train_samples": count,
        "independent_train_groups": len({s["task"]["group_id"] for s in dataset.samples}),
        "noise_repeats_per_label_per_step": noise_repeats, "effective_noise_batch": effective_batch,
        "independent_noise_draws_total": optimizer_steps * effective_batch, "optimizer": "fresh AdamW",
        "learning_rate": learning_rate, "weight_decay": 0., "gradient_norm_clip": 1., "seed": seed,
        "training_noise_seed": seed + 1, "device": str(selected_device), "cpu_threads": cpu_threads,
        "tf32": False, "loss": "basic_unweighted_true_forward_v_MSE", "timestep_distribution": "independent_uniform_0_to99",
        "teacher_counts": {"train": count, "val": 0, "test": 0}, "normalizer_train_only": True,
        "model_contract_sha256": dataset.samples[0]["task"]["model_contract_sha256"],
        "condition_schema": CONDITION_SCHEMA, "representation": REPRESENTATION,
        "source_before": source_before, "input_files_before": inputs_before,
        "wall_deadline_required": False, "test_used_for_optimization": False, "candidate_repair_or_projection": False,
        "scope": ("finite clean_x0 parameterization mechanism, same v objective and declared TRAIN-only cohort; budget also changes so not an isolated superiority claim"
                  if config.parameterization == "clean_x0" else "new v-MSE objective and expanded TRAIN cohort both change; not an isolated objective ablation")}
    _write_json(output_dir / "config.json", plan)
    _write_json(output_dir / "split.json", dataset.split)
    _write_json(output_dir / "normalizer.json", normalizer.to_dict())
    _write_json(output_dir / "environment.json", {"python": sys.version, "numpy": np.__version__, "torch": str(torch.__version__),
        "cuda_runtime": torch.version.cuda, "device": str(selected_device), "tf32": False,
        "gpu": torch.cuda.get_device_name(selected_device) if selected_device.type == "cuda" else None})
    torch.set_num_threads(cpu_threads)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = ConditionalDDPM(config).to(selected_device)
    ema_model = copy.deepcopy(model).eval().requires_grad_(False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.)
    controls = torch.from_numpy(normalizer.normalize_controls(dataset.controls)).to(selected_device).repeat_interleave(noise_repeats, dim=0)
    conditions = torch.from_numpy(normalizer.normalize_conditions(dataset.conditions)).to(selected_device).repeat_interleave(noise_repeats, dim=0)
    generator = torch.Generator(device=selected_device).manual_seed(seed + 1)
    step, start, rows, initialization_coefficient = 0, time.perf_counter(), [], 1.
    try:
        with (output_dir / "training_log.jsonl").open("x", encoding="utf-8") as stream:
            for step in range(1, optimizer_steps + 1):
                model.train(); optimizer.zero_grad(set_to_none=True)
                loss = model.training_loss(controls, conditions, generator=generator)
                if not torch.isfinite(loss): raise ValueError("nonfinite v training loss")
                loss.backward()
                gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                if not torch.isfinite(gradient): raise ValueError("nonfinite v training gradient")
                optimizer.step()
                decay = min(ema_decay, (1 + step) / (10 + step))
                _update_ema(ema_model, model, decay)
                initialization_coefficient *= decay
                row = {"optimizer_step": step, "distinct_train_labels": count,
                    "independent_noise_examples": effective_batch, "train_v_mse": float(loss.detach()),
                    "ema_updates": step, "actual_ema_decay": decay,
                    "ema_initialization_coefficient": initialization_coefficient, "elapsed_s": time.perf_counter() - start}
                rows.append(row); stream.write(json.dumps(row, allow_nan=False) + "\n"); stream.flush()
        if selected_device.type == "cuda": torch.cuda.synchronize(selected_device)
        source_after, inputs_after = _source_identity(), _input_identity(source_files)
        if not _source_unchanged(source_before, source_after) or inputs_before != inputs_after:
            raise ValueError("v training source/HEAD or frozen teacher/history input changed")
        if ema_model.schedule_identity() != schedule or ema_model.parameterization_identity() != parameterization:
            raise ValueError("v EMA model interpretation changed")
        cpu_state = lambda m: {k: v.detach().cpu() for k, v in m.state_dict().items()}
        torch.save({"schema": "v6_4_terminal_optimizer_weights_v2", "model_config": config.to_dict(),
            **_objective_fields(config), "model_state_dict": cpu_state(model), "optimizer_steps": step,
            "canonical_sampling_checkpoint": False}, output_dir / "last_optimizer_weights.pt")
        torch.save(optimizer.state_dict(), output_dir / "optimizer_state.pt")
        checkpoint = {"schema": CHECKPOINT_SCHEMA, "model_config": config.to_dict(), "model_state_dict": cpu_state(ema_model),
            **_objective_fields(config), "weight_selection": selection, "weight_selection_sha256": object_sha(selection),
            "noise_schedule": schedule, "parameterization_identity": parameterization,
            "parameterization_sha256": parameterization["parameterization_sha256"], "schedule_sha256": schedule["schedule_sha256"],
            "terminal_alpha_bar": schedule["terminal_alpha_bar"], "normalizer": normalizer.to_dict(), "split": dataset.split,
            "manifest_sha256": dataset.manifest_sha256, "seed": seed, "optimizer_steps": step, "source_identity": source_before,
            "normalizer_sha256": sha256(output_dir / "normalizer.json"), "split_sha256": sha256(output_dir / "split.json"),
            "config_sha256": sha256(output_dir / "config.json"), "model_contract_sha256": plan["model_contract_sha256"],
            "representation": REPRESENTATION, "condition_schema": CONDITION_SCHEMA}
        torch.save(checkpoint, output_dir / "checkpoint.pt")
        report = {"schema": "v6_4_from_scratch_v_training_report_v1", "complete": True, "evidence_valid": True,
            **_objective_fields(config), "weight_selection": selection, "weight_selection_sha256": object_sha(selection),
            "initialization": plan["initialization"], "optimizer_steps": step, "epochs": None,
            "train_samples": count, "independent_train_groups": plan["independent_train_groups"], "val_samples": 0,
            "test_samples_used_for_optimization": 0, "noise_repeats_per_label_per_step": noise_repeats,
            "effective_noise_batch": effective_batch, "independent_noise_draws_total": step * effective_batch,
            "loss": plan["loss"], "first_train_v_mse": rows[0]["train_v_mse"], "last_train_v_mse": rows[-1]["train_v_mse"],
            "ema_initialization_coefficient_at_end": initialization_coefficient, "ema_updates": step,
            "elapsed_s": time.perf_counter() - start, "source_before": source_before, "source_after": source_after,
            "source_unchanged": True, "inputs_unchanged": True, "input_files_after": inputs_after,
            "checkpoint_sha256": sha256(output_dir / "checkpoint.pt"), "training_log_sha256": sha256(output_dir / "training_log.jsonl"),
            "last_optimizer_weights_sha256": sha256(output_dir / "last_optimizer_weights.pt"),
            "optimizer_state_sha256": sha256(output_dir / "optimizer_state.pt"),
            "normalizer_sha256": checkpoint["normalizer_sha256"], "split_sha256": checkpoint["split_sha256"],
            "config_sha256": checkpoint["config_sha256"], "noise_schedule": schedule,
            "parameterization_identity": parameterization, "parameterization_sha256": parameterization["parameterization_sha256"],
            "schedule_sha256": schedule["schedule_sha256"], "terminal_alpha_bar": schedule["terminal_alpha_bar"],
            "scope": plan["scope"] + "; loss alone is not raw or closed-loop success"}
        _write_json(output_dir / "training_report.json", report)
        _write_json(output_dir / "artifact_manifest.json", {"files": [{"path": p.name, "sha256": sha256(p), "bytes": p.stat().st_size}
            for p in sorted(output_dir.iterdir()) if p.is_file()]})
        return report
    except BaseException as exc:
        _write_json(output_dir / "training_failure.json", {"complete": False, "evidence_valid": False, "optimizer_step": step,
            "error": f"{type(exc).__name__}: {exc}", "partial_log_retained": (output_dir / "training_log.jsonl").exists(),
            "scope": "failed finite from-scratch v training; no qualified EMA checkpoint"})
        raise


def train_clean_x0_from_manifest(manifest_path: Path, output_dir: Path, *, optimizer_steps: int = 20000,
        noise_repeats: int = 32, seed: int = 69, device: str = "cuda", learning_rate: float = 1e-4,
        ema_decay: float = .999, cpu_threads: int = 4, protected_inputs: tuple[Path, ...] = ()) -> dict:
    """New from-scratch clean_x0 network, same explicit v-MSE objective."""
    return train_v_from_manifest(manifest_path, output_dir, optimizer_steps=optimizer_steps,
        noise_repeats=noise_repeats, seed=seed, device=device, learning_rate=learning_rate,
        ema_decay=ema_decay, cpu_threads=cpu_threads, protected_inputs=protected_inputs,
        parameterization="clean_x0")


def load_sampler(checkpoint: Path, *, device: str = "auto") -> DiffusionSampler:
    return DiffusionSampler(checkpoint, device=device)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=64)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--cpu-threads", type=int, default=4)
    args = parser.parse_args()
    try:
        report = train_from_manifest(args.manifest, args.output_dir, epochs=args.epochs,
            batch_size=args.batch_size, seed=args.seed, device=args.device,
            learning_rate=args.learning_rate, cpu_threads=args.cpu_threads)
        print(json.dumps({"complete": report["complete"], "optimizer_steps": report["optimizer_steps"],
                          "checkpoint": str(args.output_dir / "checkpoint.pt"),
                          "last_train_epsilon_mse": report["last_train_epsilon_mse"]}))
        return 0
    except Exception as exc:
        print(f"training failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
