"""Explicit Teacher-prior-assisted residual diffusion, separate from v1-v6.

The finite prior is constructed BEFORE diffusion.  This module only adds a raw
learned residual to that frozen center.  It never repairs, clips, projects,
selects by actual execution, emits torques, or treats a center as a certificate.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time

import numpy as np
import torch

from .contracts import TrajectoryProposal
from .dataset import (CONDITION_SCHEMA as ORIGINAL_CONDITION_SCHEMA,
                      encode_condition, load_teacher_dataset, object_sha, sha256)
from .diffusion_model import (ConditionalDDPM, DiffusionConfig,
                              noise_schedule_identity, objective_identity,
                              parameterization_identity)
from .task_protocol import TaskSpec
from .trajectory_codec import CubicBSplineCodec

REPRESENTATION = "teacher_prior_residual_32x17_v1"
CONDITION_SCHEMA = "initial517_plus_physical_prior510_v1"
CHECKPOINT_SCHEMA = "v6_4_teacher_prior_residual_checkpoint_v1"
METHOD = "teacher_prior_assisted_diffusion"
NORMALIZER_SCHEMA = "v6_4_train_only_teacher_prior_residual_normalizer_v1"
CENTER_SCHEMA = "teacher_prior_centers_v1"
REPO = Path(__file__).resolve().parents[1]


def _write(path: Path, value: dict):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def _read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _files(paths):
    return {str(p): {"sha256": sha256(p), "bytes": p.stat().st_size}
            for p in sorted({Path(p).resolve() for p in paths})}


def source_identity():
    """Executed learning semantics and the finite-prior/gate implementation."""
    names = ("v6_4/prior_diffusion.py", "v6_4/teacher_prior.py",
             "v6_4/dataset.py", "v6_4/diffusion_model.py", "v6_4/train_diffusion.py", "v6_4/contracts.py",
             "v6_4/task_protocol.py", "v6_4/trajectory_codec.py",
             "v6_4/teacher_planner.py", "v6_4/reference_adapter.py",
             "v6_4/proposal_gate.py", "v6_4/run_teacher_comparison.py",
             "v6_4/run_planning.py", "v6_lite/run_v6_lite.py",
             "v6_lite/hierarchical_qp.py", "v6_lite/runtime_command.py",
             "v6_lite/continuum_model_spec.py", "model_test/robot_model_spec_v5.py",
             "model_test/whole_body_verifier_v5.py", "requirements-learning.txt")
    hashes = {name: sha256(REPO / name) for name in names}
    def git(args):
        return subprocess.run(["git", *args], cwd=REPO, capture_output=True,
                              text=True, encoding="utf-8", errors="replace",
                              check=True).stdout.strip()
    return {"git_head": git(["rev-parse", "HEAD"]),
            "tracked_dirty": bool(git(["status", "--porcelain", "--untracked-files=no"])),
            "sources_sha256": hashes,
            "scope": "explicit residual learner, finite prior, original private predictor/gate; actual physics separate"}


def _same_source(before, after):
    return before["git_head"] == after["git_head"] and before["sources_sha256"] == after["sources_sha256"]


def _model_contract_and_assets():
    from v6_lite.run_v6_lite import default_v6_lite_robot_spec
    spec = default_v6_lite_robot_spec()
    return spec.runtime_contract_sha256(), spec._source_assets()


def semantic_identity():
    value = {"representation": REPRESENTATION, "condition_schema": CONDITION_SCHEMA,
             "original_condition_schema": ORIGINAL_CONDITION_SCHEMA,
             "condition_dim": 1027, "prior_condition": "510 row-major physical radians; no fitted per-feature scaling",
             "clean_training_variable": "free_absolute_teacher_controls - one frozen Task-conditioned center",
             "sampling_reconstruction": "center_free + residual_mean17 + residual_scale17 * raw_normalized_residual",
             "prior_only": "physical residual zero: center_free exactly; NOT zero normalized network output",
             "boundary": "codec eliminates original C0/C1; residual has only30x17 free coordinates",
             "objective": "v_mse", "network_parameterization": "clean_x0 of NORMALIZED RESIDUAL",
             "postprocessing": [], "method": METHOD,
             "center_not_execution_certificate": True, "legacy_weights_reinterpreted": False}
    return {**value, "semantic_sha256": object_sha(value)}


def _absolute_bound_file(row, path_key, hash_key):
    path = Path(row[path_key])
    if not path.is_absolute() or not path.is_file():
        raise ValueError("prior file must be an existing absolute path")
    path = path.resolve()
    if sha256(path) != row[hash_key]:
        raise ValueError("frozen prior file SHA mismatch")
    return path


def _verify_legacy_bootstrap(identity):
    baseline = REPO/"v6_lite/output/runs/research_acceptance_01"
    path = _absolute_bound_file(identity, "source", "sha256")
    manifest = baseline/"manifest.json"
    if (path != (baseline/"simulation/traces/v6_lite_scenario_04.npz").resolve()
            or sha256(manifest) != "379b9977a441c540a20cfc50a9e90614a1e0985fa2a5a73bbb7c9690f823eec0"):
        raise ValueError("fixed historical scene04 bootstrap provenance changed")
    records = {key.replace("\\", "/"): value for key, value in _read(manifest).items()}
    if records["simulation/traces/v6_lite_scenario_04.npz"] != identity["sha256"]:
        raise ValueError("fixed bootstrap bytes differ from historical manifest")
    return path, manifest.resolve()


@dataclass(frozen=True)
class PriorCenter:
    task: TaskSpec
    free: np.ndarray
    full: np.ndarray
    row: dict
    source_files: tuple[Path, ...]

    @property
    def identity(self):
        return {"task_id": self.task.task_id, "task_sha256": self.task.sha256(),
                "center_free_sha256": self.row["center_free_sha256"],
                "center_full_sha256": self.row["center_full_sha256"],
                "center_proposal_sha256": self.row["proposal_sha256"],
                "center_record_sha256": object_sha(self.row),
                "prior_budget": self.row["prior_budget"],
                "prior_source": self.row["source"],
                "bootstrap_identity": self.row["bootstrap_identity"],
                "costs": self.row["costs"],
                "prior_only_original_gate_raw_passed": self.row["prior_only_gate_raw_passed"],
                "prior_only_gate_and_acceleration_passed": self.row["prior_only_gate_passed"],
                "center_is_execution_action": False}


def load_center(row: dict, task: TaskSpec) -> PriorCenter:
    """A finite failed center remains eligible CONDITION DATA, never an action."""
    if (row["task_id"] != task.task_id or row["task_sha256"] != task.sha256()
            or row["split"] != task.split or row["group_id"] != task.group_id):
        raise ValueError("prior is not bound to this immutable Task/source group")
    if row.get("center_finite") is not True or row.get("start_boundary_exact") is not True:
        raise ValueError("prior has no finite exact-boundary center")
    if row.get("source_unchanged") is not True:
        raise ValueError("prior source changed or its qualification is missing")
    if not isinstance(row["chosen_attempt_index"], int) or isinstance(row["chosen_attempt_index"], bool) or not 0 <= row["chosen_attempt_index"] < 8:
        raise ValueError("prior chosen attempt must be one of the eight fixed starts")
    for key in ("prior_budget", "source", "bootstrap_identity", "costs"):
        if not isinstance(row[key], dict) or not row[key]:
            raise ValueError("prior budget/source/bootstrap/cost identities required")
    free_path = _absolute_bound_file(row, "center_free_path", "center_free_sha256")
    full_path = _absolute_bound_file(row, "center_full_path", "center_full_sha256")
    proposal_path = _absolute_bound_file(row, "center_proposal_path", "center_proposal_file_sha256")
    gate_path = _absolute_bound_file(row, "prior_only_gate_path", "prior_only_gate_sha256")
    free = np.asarray(np.load(free_path, allow_pickle=False), dtype=np.float64)
    full = np.asarray(np.load(full_path, allow_pickle=False), dtype=np.float64)
    codec = CubicBSplineCodec(task.initial_planner_q, task.initial_planner_dq)
    if (free.shape != (30, 17) or full.shape != (32, 17)
            or not np.all(np.isfinite(free)) or not np.all(np.isfinite(full))
            or not np.array_equal(codec.decode_free(free), full)):
        raise ValueError("prior CP shape/finiteness/exact eliminated C0/C1 binding failed")
    proposal = TrajectoryProposal.from_dict(_read(proposal_path))
    if (proposal.task_id != task.task_id or proposal.task_sha256 != task.sha256()
            or proposal.origin != "teacher" or proposal.postprocessing
            or proposal.sha256() != row["proposal_sha256"]
            or not np.array_equal(proposal.free_controls, free)):
        raise ValueError("prior proposal/CP/intrinsic identity mismatch")
    gate = _read(gate_path)
    if (gate["task_id"] != task.task_id or gate["task_sha256"] != task.sha256()
            or not isinstance(row["prior_only_gate_raw_passed"], bool)
            or gate["raw_passed"] is not row["prior_only_gate_raw_passed"]):
        raise ValueError("prior-only gate identity/status mismatch")
    acceleration_path = _absolute_bound_file(row, "prior_only_acceleration_guard_path", "prior_only_acceleration_guard_sha256")
    acceleration = _read(acceleration_path)
    if (acceleration.get("task_id") != task.task_id or acceleration.get("task_sha256") != task.sha256()
            or not isinstance(acceleration.get("passed"), bool)
            or acceleration.get("limits_rad_s2") != [2.5]*10+[4.]*7
            or row["prior_only_gate_passed"] is not (gate["raw_passed"] and acceleration["passed"])):
        raise ValueError("independent prior-only original acceleration status/identity mismatch")
    if (row["bootstrap_identity"].get("source_split") != "bootstrap"
            or row["bootstrap_identity"].get("source_group_id") != "research_acceptance_01:v6_lite_scenario_04"):
        raise ValueError("this prior version fixes disjoint legacy scene04 bootstrap; no Task future trace")
    bootstrap_path, bootstrap_manifest = _verify_legacy_bootstrap(row["bootstrap_identity"])
    free.setflags(write=False); full.setflags(write=False)
    return PriorCenter(task, free, full, copy.deepcopy(row), (free_path, full_path, proposal_path, gate_path, acceleration_path, bootstrap_path, bootstrap_manifest))


def load_centers(manifest_path: Path, tasks: list[TaskSpec]):
    manifest_path = Path(manifest_path).resolve()
    value = _read(manifest_path)
    if (value.get("schema") != CENTER_SCHEMA or value.get("complete") is not True
            or value.get("source_unchanged") is not True or value.get("all_tasks_attempted") is not True):
        raise ValueError("expected complete finite prior-centers manifest")
    rows = value["centers"]
    if len({row["task_id"] for row in rows}) != len(rows):
        raise ValueError("duplicate prior Task center")
    declared = {task.task_id: task for task in tasks}
    if set(declared) != {row["task_id"] for row in rows}:
        raise ValueError("prior centers must cover every declared Task exactly; no difficult Task dropped")
    centers = {row["task_id"]: load_center(row, declared[row["task_id"]]) for row in rows}
    paths = {manifest_path}
    for center in centers.values(): paths.update(center.source_files)
    return centers, paths


@dataclass(frozen=True)
class ResidualNormalizer:
    condition_mean: np.ndarray
    condition_scale: np.ndarray
    residual_mean: np.ndarray
    residual_scale: np.ndarray
    training_sample_ids: tuple[str, ...]
    center_sample_weight: dict

    @classmethod
    def fit(cls, dataset, centers):
        if not dataset.samples or any(s["split"] != "train" for s in dataset.samples):
            raise ValueError("residual normalization requires successful TRAIN-only labels")
        center_array = np.stack([centers[s["task_id"]].free for s in dataset.samples])
        residuals = dataset.controls - center_array
        return cls(dataset.conditions.mean(0), np.maximum(dataset.conditions.std(0), 1e-6),
                   residuals.mean((0, 1)), np.maximum(residuals.std((0, 1)), 1e-6),
                   tuple(sorted(s["sample_id"] for s in dataset.samples)),
                   {key: sum(s["task_id"] == key for s in dataset.samples) for key in centers})

    def condition(self, task, center):
        if task.sha256() != center.task.sha256(): raise ValueError("condition/center Task mismatch")
        original = encode_condition(task)
        if original.shape != (517,): raise ValueError("original immutable condition must have517 features")
        # The added510 features are physical radians; no near-zero510 std fit.
        value = np.r_[(original-self.condition_mean)/self.condition_scale, center.free.ravel(order="C")].astype(np.float32)
        if not np.all(np.isfinite(value)): raise ValueError("nonfinite float32 immutable/prior condition")
        return value

    def normalize_residuals(self, values):
        values = np.asarray(values, dtype=np.float64)
        if values.shape[-2:] != (30, 17) or not np.all(np.isfinite(values)): raise ValueError("invalid physical residual")
        return ((values-self.residual_mean)/self.residual_scale).astype(np.float32)

    def denormalize_residuals(self, values):
        values = np.asarray(values, dtype=np.float64)
        if values.shape[-2:] != (30, 17) or not np.all(np.isfinite(values)): raise ValueError("invalid raw normalized residual")
        return values*self.residual_scale+self.residual_mean

    def restore(self, values, center):
        return center.free + self.denormalize_residuals(values)

    def to_dict(self):
        return {"schema": NORMALIZER_SCHEMA, "representation": REPRESENTATION, "condition_schema": CONDITION_SCHEMA,
                "condition_mean": self.condition_mean.tolist(), "condition_scale": self.condition_scale.tolist(),
                "residual_mean": self.residual_mean.tolist(), "residual_scale": self.residual_scale.tolist(),
                "training_sample_ids": list(self.training_sample_ids), "center_sample_weight": self.center_sample_weight,
                "original517_fit": "13 successful TRAIN label rows, repeated centers weighted by label count",
                "center510_fit": "none; raw physical radians row-major",
                "residual_fit": "TRAIN labels minus each Task center; shared17D mean/std over samples and30 free rows; floor1e-6",
                "fit_validation_or_test": False, "zero_normalized_output_physical_residual": self.residual_mean.tolist(),
                "prior_only_is_zero_physical_residual": True, "postprocessing": []}

    @classmethod
    def from_dict(cls, value):
        if (value.get("schema") != NORMALIZER_SCHEMA or value.get("representation") != REPRESENTATION
                or value.get("condition_schema") != CONDITION_SCHEMA or value.get("fit_validation_or_test") is not False
                or value.get("center510_fit") != "none; raw physical radians row-major"
                or value.get("prior_only_is_zero_physical_residual") is not True or value.get("postprocessing") != []):
            raise ValueError("residual normalizer semantic identity mismatch")
        arrays = [np.asarray(value[k], dtype=np.float64) for k in ("condition_mean", "condition_scale", "residual_mean", "residual_scale")]
        if [a.shape for a in arrays] != [(517,), (517,), (17,), (17,)] or not all(np.all(np.isfinite(a)) for a in arrays) or np.any(arrays[1] <= 0) or np.any(arrays[3] <= 0):
            raise ValueError("residual normalizer dimensions/numerics invalid")
        ids = tuple(value["training_sample_ids"])
        if not ids or len(set(ids)) != len(ids) or sum(value["center_sample_weight"].values()) != len(ids):
            raise ValueError("residual normalizer TRAIN sample/center weights invalid")
        if not np.array_equal(arrays[2], value["zero_normalized_output_physical_residual"]):
            raise ValueError("zero normalized residual semantic mismatch")
        return cls(*arrays, ids, value["center_sample_weight"])


def _device(value):
    device = torch.device(value)
    if device.type not in ("cpu", "cuda") or device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("available CPU or CUDA required")
    return device


def train_prior_residual(teacher_manifest: Path, prior_manifest: Path, output: Path, *,
                         optimizer_steps=20000, noise_repeats=32, seed=70, device="cuda",
                         learning_rate=1e-4, ema_decay=.999, cpu_threads=4,
                         expected_samples=13, expected_groups=6, protected_inputs=()):
    """From scratch only; actual416 independent noises per step in formal budget."""
    integers = (optimizer_steps, noise_repeats, seed, cpu_threads, expected_samples, expected_groups)
    if any(isinstance(v, bool) or not isinstance(v, int) for v in integers) or min(optimizer_steps, noise_repeats, cpu_threads, expected_samples, expected_groups) <= 0 or seed < 0:
        raise ValueError("invalid finite residual training budget")
    if not np.isfinite(learning_rate) or learning_rate <= 0 or not np.isfinite(ema_decay) or not 0 < ema_decay < 1:
        raise ValueError("invalid optimizer/EMA configuration")
    before = source_identity()
    dataset = load_teacher_dataset(Path(teacher_manifest))
    if len(dataset.samples) != expected_samples or any(s["split"] != "train" for s in dataset.samples):
        raise ValueError("all original successful TRAIN labels must be retained")
    tasks = {}
    for sample in dataset.samples:
        task = TaskSpec.from_dict(sample["task"])
        if task.task_id in tasks and tasks[task.task_id].sha256() != task.sha256():
            raise ValueError("same Task ID must not alias different immutable conditions")
        tasks[task.task_id] = task
    if len(tasks) != expected_groups or len({t.group_id for t in tasks.values()}) != expected_groups:
        raise ValueError("all independent TRAIN source groups must be retained")
    centers, prior_files = load_centers(Path(prior_manifest), list(tasks.values()))
    contract, assets = _model_contract_and_assets()
    if any(t.model_contract_sha256 != contract for t in tasks.values()): raise ValueError("current nominal model assets differ")
    source_files = set(dataset.source_files) | prior_files | {Path(p).resolve() for p in protected_inputs} | set(assets)
    inputs_before = _files(source_files)
    output = Path(output).resolve()
    if any(p.is_relative_to(output) for p in source_files) or Path(__file__).resolve().is_relative_to(output):
        raise ValueError("output must not contain/overwrite frozen inputs or producer")
    output.mkdir(parents=True, exist_ok=False)
    normalizer = ResidualNormalizer.fit(dataset, centers)
    config = DiffusionConfig(condition_dim=1027, parameterization="clean_x0", objective="v_mse")
    schedule, parameterization, objective = noise_schedule_identity(config), parameterization_identity(config), objective_identity(config)
    selection = {"weights": "last_step_ema", "last_optimizer_step": optimizer_steps,
                 "ema_decay_cap": ema_decay, "ema_warmup": True, "ema_bias_correction": False,
                 "ema_decay_formula": "min(cap,(1+s)/(10+s)), s=optimizer_step starting at1",
                 "ema_initialization": "fresh_random_weights", "validation_or_test_selection": False}
    effective = len(dataset.samples)*noise_repeats
    selected_device = _device(device)
    plan = {"schema": "v6_4_teacher_prior_residual_training_plan_v1", "representation": REPRESENTATION,
            "condition_schema": CONDITION_SCHEMA, "semantics": semantic_identity(), "method": METHOD,
            "model": config.to_dict(), "noise_schedule": schedule, "parameterization_identity": parameterization,
            "objective_identity": objective, "weight_selection": selection,
            "teacher_manifest": str(dataset.manifest_path), "teacher_manifest_sha256": dataset.manifest_sha256,
            "prior_manifest": str(Path(prior_manifest).resolve()), "prior_manifest_sha256": sha256(Path(prior_manifest)),
            "train_samples": len(dataset.samples), "train_groups": len(tasks), "all_labels_retained": True,
            "centers": {key: center.identity for key, center in centers.items()},
            "center_failed_gate_excluded_from_training": False,
            "optimizer_steps": optimizer_steps, "noise_repeats_per_label": noise_repeats, "effective_noise_batch": effective,
            "noise_draws_total": optimizer_steps*effective, "seed": seed, "noise_seed": seed+1,
            "optimizer": "fresh AdamW", "learning_rate": learning_rate, "weight_decay": 0., "gradient_norm_clip": 1.,
            "device": str(selected_device), "cpu_threads": cpu_threads, "tf32": False,
            "initialization": "from_scratch_no_legacy_weights", "source_before": before, "inputs_before": inputs_before,
            "model_contract_sha256": contract, "test_or_validation_data_used": False, "postprocessing": [],
            "scope": "finite prior-assisted structural variant; prior costs/ability separate; not old absolute-network matrix or isolated superiority ablation"}
    _write(output/"config.json", plan); _write(output/"normalizer.json", normalizer.to_dict()); _write(output/"split.json", dataset.split)
    _write(output/"environment.json", {"python": sys.version, "numpy": np.__version__, "torch": str(torch.__version__),
           "cuda_runtime": torch.version.cuda, "device": str(selected_device), "tf32": False,
           "gpu": torch.cuda.get_device_name(selected_device) if selected_device.type == "cuda" else None})
    archive = output/"source_snapshot"
    for relative, expected in before["sources_sha256"].items():
        target = archive/relative; target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REPO/relative, target)
        if sha256(target) != expected: raise ValueError("source changed while archiving")
    _write(archive/"source_archive_receipt.json", {"source_identity": before, "copied_before_optimization": True})
    torch.set_num_threads(cpu_threads); random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    model = ConditionalDDPM(config).to(selected_device)
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.)
    center_array = np.stack([centers[s["task_id"]].free for s in dataset.samples])
    raw_residuals = dataset.controls-center_array
    np.savez_compressed(output/"derived_train_residuals.npz", residuals=raw_residuals, centers=center_array,
                        absolute_teacher_labels=dataset.controls, sample_ids=np.asarray([s["sample_id"] for s in dataset.samples]))
    controls = torch.from_numpy(normalizer.normalize_residuals(raw_residuals)).to(selected_device).repeat_interleave(noise_repeats, 0)
    conditions = torch.from_numpy(np.stack([normalizer.condition(tasks[s["task_id"]], centers[s["task_id"]]) for s in dataset.samples])).to(selected_device).repeat_interleave(noise_repeats, 0)
    generator = torch.Generator(device=selected_device).manual_seed(seed+1)
    start, step, initial_coefficient = time.perf_counter(), 0, 1.
    first_loss = last_loss = None
    try:
        with (output/"training_log.jsonl").open("x", encoding="utf-8") as stream:
            for step in range(1, optimizer_steps+1):
                model.train(); optimizer.zero_grad(set_to_none=True)
                loss = model.training_loss(controls, conditions, generator=generator)
                if not torch.isfinite(loss): raise ValueError("nonfinite residual v loss")
                loss.backward(); gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                if not torch.isfinite(gradient): raise ValueError("nonfinite residual training gradient")
                optimizer.step(); decay = min(ema_decay, (1+step)/(10+step))
                with torch.no_grad():
                    for dst, src in zip(ema.parameters(), model.parameters(), strict=True): dst.mul_(decay).add_(src, alpha=1-decay)
                    for dst, src in zip(ema.buffers(), model.buffers(), strict=True): dst.copy_(src)
                initial_coefficient *= decay; last_loss = float(loss.detach())
                if first_loss is None: first_loss = last_loss
                row = {"optimizer_step": step, "train_v_mse": last_loss, "effective_noise_batch": effective,
                       "independent_noise_draws_total": step*effective, "gradient_norm": float(gradient),
                       "ema_decay": decay, "ema_initialization_coefficient": initial_coefficient, "elapsed_s": time.perf_counter()-start}
                stream.write(json.dumps(row, allow_nan=False)+"\n"); stream.flush()
        if selected_device.type == "cuda": torch.cuda.synchronize(selected_device)
        after, inputs_after = source_identity(), _files(source_files)
        if not _same_source(before, after) or inputs_after != inputs_before or _model_contract_and_assets()[0] != contract:
            raise ValueError("source/HEAD/nominal assets/prior/teacher/history changed")
        cpu_state = lambda m: {k: v.detach().cpu() for k, v in m.state_dict().items()}
        torch.save({"schema": "v6_4_prior_residual_last_optimizer_v1", "model_state_dict": cpu_state(model),
                    "model_config": config.to_dict(), "canonical_sampling_checkpoint": False}, output/"last_optimizer_weights.pt")
        torch.save(optimizer.state_dict(), output/"optimizer_state.pt")
        checkpoint = {"schema": CHECKPOINT_SCHEMA, "representation": REPRESENTATION, "condition_schema": CONDITION_SCHEMA,
                      "semantics": semantic_identity(), "method": METHOD, "model_config": config.to_dict(),
                      "model_state_dict": cpu_state(ema), "objective_identity": objective,
                      "noise_schedule": schedule, "parameterization_identity": parameterization,
                      "weight_selection": selection, "weight_selection_sha256": object_sha(selection),
                      "normalizer": normalizer.to_dict(), "split": dataset.split, "source_identity": before,
                      "model_contract_sha256": contract, "optimizer_steps": step, "seed": seed,
                      "teacher_manifest_sha256": dataset.manifest_sha256, "prior_manifest_sha256": sha256(Path(prior_manifest)),
                      "centers": {key: center.identity for key, center in centers.items()},
                      "companion_hashes": {name: sha256(output/name) for name in ("config.json", "normalizer.json", "split.json", "derived_train_residuals.npz")}}
        torch.save(checkpoint, output/"checkpoint.pt")
        report = {"schema": "v6_4_prior_residual_training_report_v1", "complete": True, "evidence_valid": True,
                  "representation": REPRESENTATION, "condition_schema": CONDITION_SCHEMA, "method": METHOD,
                  "semantics": semantic_identity(), "train_samples": len(dataset.samples), "train_groups": len(tasks),
                  "all_labels_retained": True, "optimizer_steps": step, "effective_noise_batch": effective,
                  "noise_repeats_per_label": noise_repeats, "independent_noise_draws_total": step*effective,
                  "first_train_v_mse": first_loss, "last_train_v_mse": last_loss, "elapsed_s": time.perf_counter()-start,
                  "weight_selection": selection, "ema_initialization_coefficient_at_end": initial_coefficient,
                  "source_before": before, "source_after": after, "source_unchanged": True, "inputs_unchanged": True,
                  "checkpoint_sha256": sha256(output/"checkpoint.pt"), "training_log_sha256": sha256(output/"training_log.jsonl"),
                  "prior_manifest_sha256": checkpoint["prior_manifest_sha256"], "teacher_manifest_sha256": dataset.manifest_sha256,
                  "centers": checkpoint["centers"], "zero_normalized_output_raw_residual": normalizer.residual_mean.tolist(),
                  "prior_only": "center exactly, physical residual0", "physics_steps": 0, "test_used": False,
                  "scope": plan["scope"]+"; loss does not establish raw or actual completion"}
        _write(output/"training_report.json", report)
        paths = sorted(p for p in output.rglob("*") if p.is_file())
        _write(output/"artifact_manifest.json", {"files": [{"path": p.relative_to(output).as_posix(), "sha256": sha256(p), "bytes": p.stat().st_size} for p in paths]})
        return report
    except BaseException as exc:
        _write(output/"training_failure.json", {"complete": False, "evidence_valid": False, "optimizer_step": step,
               "error": f"{type(exc).__name__}: {exc}", "partial_log_retained": True, "physics_steps": 0, "qualified_checkpoint": False})
        raise


class PriorResidualSampler:
    def __init__(self, path, checkpoint, model, normalizer, device):
        self.path, self.checkpoint, self.model, self.normalizer, self.device = path, checkpoint, model, normalizer, device
        self.checkpoint_sha256 = sha256(path)

    @torch.no_grad()
    def sample(self, task: TaskSpec, center: PriorCenter, *, K=1, seed=64):
        if K not in (1, 8) or not isinstance(seed, int) or isinstance(seed, bool) or seed < 0: raise ValueError("fixed K1/K8 and nonnegative seed required")
        torch.set_num_threads(4)
        before = source_identity()
        if before["sources_sha256"] != self.checkpoint["source_identity"]["sources_sha256"]:
            raise ValueError("sampling source differs from frozen new residual checkpoint")
        if task.model_contract_sha256 != self.checkpoint["model_contract_sha256"] or _model_contract_and_assets()[0] != task.model_contract_sha256:
            raise ValueError("current/task/checkpoint nominal model differs")
        # Reverify immutable center artifacts at the use boundary.
        current_center = load_center(center.row, task)
        if not np.array_equal(current_center.free, center.free): raise ValueError("prior center changed")
        condition = torch.from_numpy(self.normalizer.condition(task, center)[None]).to(self.device)
        raw, residuals, rows = [], [], []
        start = time.perf_counter()
        for index in range(K):
            generator = torch.Generator(device=self.device).manual_seed(seed+index)
            normalized = self.model.sample_ddim(condition, steps=20, generator=generator).cpu().numpy()[0]
            residual = self.normalizer.denormalize_residuals(normalized)
            absolute = center.free+residual
            if not np.all(np.isfinite(absolute)): raise ValueError("nonfinite prior-assisted raw proposal")
            raw.append(absolute); residuals.append(residual)
            rows.append({"candidate_index": index, "seed": seed+index,
                         "raw_physical_residual_rms_rad": float(np.sqrt(np.mean(residual**2))),
                         "raw_physical_residual_max_abs_rad": float(np.max(abs(residual))),
                         "center_equivalent_exact": bool(np.array_equal(absolute, center.free)),
                         "zero_normalized_output_equivalent_exact": bool(np.array_equal(residual, np.broadcast_to(self.normalizer.residual_mean, (30, 17))))})
        after = source_identity()
        if not _same_source(before, after) or sha256(self.path) != self.checkpoint_sha256 or _model_contract_and_assets()[0] != task.model_contract_sha256:
            raise ValueError("source/HEAD/weights/assets changed during residual generation")
        load_center(center.row, task)
        metadata = {"method": METHOD, "representation": REPRESENTATION, "condition_schema": CONDITION_SCHEMA,
                    "semantics": semantic_identity(), "checkpoint_sha256": self.checkpoint_sha256,
                    "task_sha256": task.sha256(), "prior": center.identity,
                    "weight_selection": self.checkpoint["weight_selection"], "sampling_elapsed_s": time.perf_counter()-start,
                    "prior_construction_costs_separate": center.row["costs"], "postprocessing": [], "repair": False,
                    "source_before": before, "source_after": after, "candidates": rows,
                    "scope": "raw center+learned residual; original full gate and actual executor still required"}
        return np.stack(raw), np.stack(residuals), metadata


def load_prior_sampler(path: Path, *, device="cpu") -> PriorResidualSampler:
    path = Path(path).resolve(); checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("schema") != CHECKPOINT_SCHEMA:
        raise ValueError("new prior-residual loader must not reinterpret v1-v6 or foreign weights")
    if (checkpoint.get("representation") != REPRESENTATION or checkpoint.get("condition_schema") != CONDITION_SCHEMA
            or checkpoint.get("semantics") != semantic_identity() or checkpoint.get("method") != METHOD):
        raise ValueError("prior-residual semantic/checkpoint identity mismatch")
    config = DiffusionConfig.from_dict(checkpoint["model_config"])
    if config.condition_dim != 1027 or config.parameterization != "clean_x0" or config.objective != "v_mse" or config.noise_schedule != "cosine" or config.ddim_steps != 20:
        raise ValueError("prior residual architecture/variable/objective mismatch")
    if (checkpoint["noise_schedule"] != noise_schedule_identity(config)
            or checkpoint["parameterization_identity"] != parameterization_identity(config)
            or checkpoint["objective_identity"] != objective_identity(config)):
        raise ValueError("prior residual mathematical identity mismatch")
    selection = checkpoint["weight_selection"]
    if (object_sha(selection) != checkpoint["weight_selection_sha256"] or selection["weights"] != "last_step_ema"
            or selection["last_optimizer_step"] != checkpoint["optimizer_steps"] or selection["ema_warmup"] is not True
            or selection["validation_or_test_selection"] is not False):
        raise ValueError("sole predeclared final warmup EMA required")
    if set(checkpoint["companion_hashes"]) != {"config.json", "normalizer.json", "split.json", "derived_train_residuals.npz"}:
        raise ValueError("all checkpoint companions must remain bound")
    for name, expected in checkpoint["companion_hashes"].items():
        if Path(name).name != name or sha256(path.parent/name) != expected: raise ValueError("checkpoint companion SHA mismatch")
    plan = _read(path.parent/"config.json")
    normalizer_value = _read(path.parent/"normalizer.json")
    if (plan["model"] != config.to_dict() or plan["semantics"] != semantic_identity()
            or plan["weight_selection"] != selection or normalizer_value != checkpoint["normalizer"]
            or _read(path.parent/"split.json") != checkpoint["split"]):
        raise ValueError("checkpoint/declared config/normalizer/split mismatch")
    normalizer = ResidualNormalizer.from_dict(normalizer_value)
    selected_device = _device(device); model = ConditionalDDPM(config)
    expected = model.state_dict()
    for name, value in checkpoint["model_state_dict"].items():
        if not torch.isfinite(value).all(): raise ValueError("nonfinite checkpoint weights")
        if name in ("betas", "alphas", "alpha_bars") and not torch.equal(value, expected[name]): raise ValueError("saved diffusion buffers differ")
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(selected_device).eval()
    return PriorResidualSampler(path, checkpoint, model, normalizer, selected_device)


def reference_ddq_guard(prediction, spec):
    """Separate original acceleration guard; gate.py does not establish this."""
    times = np.asarray(prediction["time"], dtype=float)
    ddq = np.asarray(prediction["ddq"], dtype=float)
    limits = np.asarray(spec.planner_acceleration_limits, dtype=float)
    original = np.r_[np.full(10, 2.5), np.full(7, 4.)]
    finite = bool(times.shape == (1351,) and ddq.shape == (1351, 17)
                  and np.all(np.isfinite(times)) and np.all(np.isfinite(ddq)))
    grid = bool(finite and np.allclose(times, np.arange(1351)*.02, rtol=0., atol=1e-12))
    original_limits = bool(limits.shape == (17,) and np.array_equal(limits, original))
    within = bool(finite and original_limits and np.all(abs(ddq) <= limits+1e-12))
    return {"schema": "v6_4_original_reference_acceleration_guard_v1", "passed": finite and grid and original_limits and within,
            "checks": {"finite_complete_1351x17_ddq": finite, "original_20ms_grid": grid,
                       "original_2_5_continuum_4_rigid_limits": original_limits, "reference_acceleration_range": within},
            "limits_rad_s2": limits.tolist(),
            "time_array_sha256": hashlib.sha256(times.tobytes()).hexdigest() if finite else None,
            "ddq_array_sha256": hashlib.sha256(ddq.tobytes()).hexdigest() if finite else None,
            "max_abs_per_coordinate_rad_s2": np.max(abs(ddq), axis=0).tolist() if finite else None,
            "maximum_limit_ratio": float(np.max(abs(ddq)/original)) if finite else None,
            "physics_steps": 0, "scope": "original limits on saved nominal reference ddq, independent of gate.py; not continuous-time or actual-acceleration proof"}


def generate_prior_proposals(task, center, checkpoint, output, *, K=8, seed=64, device="cpu"):
    """Preserve each raw network residual BEFORE any gate or actual execution."""
    start = time.perf_counter()
    output = Path(output).resolve()
    if any(p.is_relative_to(output) for p in (*center.source_files, Path(checkpoint).resolve())):
        raise ValueError("generation output must not contain frozen inputs")
    output.mkdir(parents=True, exist_ok=False); torch.set_num_threads(4)
    sampler = load_prior_sampler(Path(checkpoint), device=device)
    raw, residuals, metadata = sampler.sample(task, center, K=K, seed=seed)
    proposals = []
    for index, controls in enumerate(raw):
        folder = output/f"c{index}"; folder.mkdir()
        proposal = TrajectoryProposal.from_controls(task, controls, origin="diffusion", seed=seed+index,
            postprocessing=(), metadata={**metadata, "candidate_index": index, "candidate_budget": K,
                                        "physical_residual": metadata["candidates"][index]})
        np.save(folder/"controls_free.npy", controls, allow_pickle=False)
        np.save(folder/"residual_free.npy", residuals[index], allow_pickle=False)
        _write(folder/"proposal.json", proposal.to_dict()); proposals.append(proposal)
    np.save(output/"center_free.npy", center.free, allow_pickle=False)
    metadata["total_generation_wall_s_including_load_validation_and_artifacts"] = time.perf_counter()-start
    _write(output/"generation.json", metadata)
    return proposals, metadata
