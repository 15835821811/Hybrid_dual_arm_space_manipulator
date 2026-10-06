"""Frozen, paired B.1 training and inference with shared reference semantics.

Training, overfit smoke, and validation never execute physics or query geometry.
Smoke weights are discarded; formal M0/M1 initialize afresh with the same seed.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import platform
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import torch

from v6_4.conditioned_controlpoint_denoiser import PilotDDPM
from v6_4.dataset import object_sha, sha256, task_dict
from v6_4.differentiable_spline_loss import DecodedReferenceLoss, paired_training_loss
from v6_4.reference_training_data import ControlNormalizer, load_reference_dataset
from v6_4.trajectory_codec import CubicBSplineCodec
from v6_4.typed_condition_encoder import (FLAT_DIM, GLOBAL_DIM, TOKEN_COUNT, TOKEN_DIM,
    TYPE_COUNT, encode_typed_condition, flatten_typed_condition, schema_config)

ROOT = Path(__file__).resolve().parents[1]
SOURCE_FILES = ("v6_4/architecture_pilot_training.py", "v6_4/conditioned_controlpoint_denoiser.py",
    "v6_4/differentiable_spline_loss.py", "v6_4/typed_condition_encoder.py",
    "v6_4/reference_training_data.py", "v6_4/diffusion_model.py", "v6_4/trajectory_codec.py",
    "v6_4/contracts.py", "v6_4/dataset.py", "v6_4/task_protocol.py",
    "model_test/robot_model_spec_v5.py", "v6_lite/hierarchical_qp.py")
CONDITION_NORMALIZER_SCHEMA = "v6_4_b1_train_only_typed_condition_normalizer_v1"
TRAINING_SCHEMA = "v6_4_b1_paired_reference_training_v1"
TRAIN_SEED, DRAW_SEED, VAL_SEED, SMOKE_SEED = 64101, 64102, 64103, 64104


def _write(path, value, *, exclusive=True):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x" if exclusive else "w", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+"\n")


def _sources():
    return {name: sha256(ROOT/name) for name in SOURCE_FILES}


def _synchronize(device):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


def configure_runtime(device="cuda"):
    device = torch.device(device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("frozen CUDA training requires the existing CUDA runtime")
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1) if torch.get_num_interop_threads() != 1 else None
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    if hasattr(torch.backends.cuda, "enable_cudnn_sdp"):
        torch.backends.cuda.enable_cudnn_sdp(False)
    return {"python": platform.python_version(), "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
        "device": str(device), "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "gpu_total_memory_bytes": torch.cuda.get_device_properties(device).total_memory if device.type == "cuda" else None,
        "cpu_threads": torch.get_num_threads(), "interop_threads": torch.get_num_interop_threads(),
        "tf32_matmul": torch.backends.cuda.matmul.allow_tf32, "tf32_cudnn": torch.backends.cudnn.allow_tf32,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "attention_backend": "math_SDPA; flash, memory_efficient and cudnn_SDPA disabled",
        "CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG")}


@dataclass(frozen=True)
class ConditionNormalizer:
    global_mean: np.ndarray
    global_scale: np.ndarray
    token_mean: np.ndarray
    token_scale: np.ndarray
    training_sample_ids: tuple[str, ...]
    std_floor: float = .01

    @classmethod
    def fit(cls, encoded, training_indices, training_sample_ids):
        indices = np.asarray(training_indices, dtype=int)
        if not len(indices) or len(indices) != len(training_sample_ids):
            raise ValueError("TRAIN identities and indices required for shared condition scaler")
        globals_ = np.stack([encoded[i]["global"] for i in indices])
        valid = np.concatenate([encoded[i]["token_features"][encoded[i]["token_mask"]] for i in indices])
        if not len(valid) or not np.all(np.isfinite(globals_)) or not np.all(np.isfinite(valid)):
            raise ValueError("finite TRAIN globals and valid token features required")
        return cls(globals_.mean(axis=0), np.maximum(globals_.std(axis=0), .01),
            valid.mean(axis=0), np.maximum(valid.std(axis=0), .01), tuple(training_sample_ids))

    def normalize(self, encoded):
        if encoded["global"].shape != (GLOBAL_DIM,) or encoded["token_features"].shape != (TOKEN_COUNT, TOKEN_DIM):
            raise ValueError("condition normalizer requires the frozen typed shape")
        mask = np.asarray(encoded["token_mask"], dtype=bool)
        features = np.zeros((TOKEN_COUNT, TOKEN_DIM), dtype=np.float64)
        features[mask] = (np.asarray(encoded["token_features"])[mask]-self.token_mean)/self.token_scale
        normalized = {"schema": encoded["schema"],
            "global": (np.asarray(encoded["global"])-self.global_mean)/self.global_scale,
            "token_features": features,
            "token_types": np.where(mask, encoded["token_types"], 0).astype(np.int64), "token_mask": mask.copy()}
        if not np.all(np.isfinite(normalized["global"])) or not np.all(np.isfinite(features)):
            raise ValueError("normalized typed conditions are nonfinite")
        return normalized

    def to_dict(self):
        return {"schema": CONDITION_NORMALIZER_SCHEMA, "global_mean": self.global_mean.tolist(),
            "global_scale": self.global_scale.tolist(), "token_mean": self.token_mean.tolist(),
            "token_scale": self.token_scale.tolist(), "training_sample_ids": list(self.training_sample_ids),
            "std_floor": self.std_floor, "fit_validation_or_test": False,
            "token_statistics": "only valid rows of TRAIN encoded initial-state conditions, pooled per feature",
            "invalid_token_features_after_normalization": "exact zero", "types_and_masks_normalized": False,
            "M0_flatten": "same normalized globals/features plus original types/masks, canonical physical order"}

    @classmethod
    def from_dict(cls, value):
        if (value.get("schema") != CONDITION_NORMALIZER_SCHEMA or value.get("fit_validation_or_test") is not False
                or value.get("types_and_masks_normalized") is not False or value.get("std_floor") != .01):
            raise ValueError("invalid TRAIN-only shared condition normalizer identity")
        arrays = [np.asarray(value[key], dtype=np.float64) for key in
            ("global_mean", "global_scale", "token_mean", "token_scale")]
        ids = tuple(value["training_sample_ids"])
        if (any(a.shape != shape for a, shape in zip(arrays, ((GLOBAL_DIM,), (GLOBAL_DIM,), (TOKEN_DIM,), (TOKEN_DIM,))))
                or not all(np.all(np.isfinite(a)) for a in arrays) or np.any(arrays[1]<.01)
                or np.any(arrays[3]<.01) or not ids or len(set(ids)) != len(ids)):
            raise ValueError("invalid condition scaler values or TRAIN IDs")
        return cls(*arrays, ids)


def _physical_scales():
    from v6_lite.run_v6_lite import default_v6_lite_robot_spec
    spec = default_v6_lite_robot_spec()
    return np.r_[np.full(10, 2.), (spec.planner_upper-spec.planner_lower)[10:]], spec.planner_velocity_limits.copy()


def make_conditions(tasks, normalizer, *, q_ranges=None, device="cpu", encoded=None):
    if isinstance(normalizer, dict): normalizer = ConditionNormalizer.from_dict(normalizer)
    if q_ranges is None: q_ranges, _ = _physical_scales()
    q_ranges = np.asarray(q_ranges, dtype=np.float64)
    if q_ranges.shape != (17,) or np.any(q_ranges<=0): raise ValueError("original positive work ranges required")
    task_values = [task_dict(task) for task in tasks]
    raw = [encode_typed_condition(task) for task in task_values] if encoded is None else encoded
    if len(raw) != len(task_values): raise ValueError("typed task batch length mismatch")
    conditions = [normalizer.normalize(item) for item in raw]
    fixed = np.stack([CubicBSplineCodec(task["initial_planner_q"], task["initial_planner_dq"]).fixed_controls
                      for task in task_values])
    result = {key: torch.as_tensor(np.stack([condition[key] for condition in conditions]), device=device,
        dtype=torch.long if key=="token_types" else torch.bool if key=="token_mask" else torch.float32)
        for key in ("global", "token_features", "token_types", "token_mask")}
    result["flat"] = torch.as_tensor(np.stack([flatten_typed_condition(item) for item in conditions]),
                                     dtype=torch.float32, device=device)
    result["fixed_controls"] = torch.as_tensor(fixed, dtype=torch.float32, device=device)
    result["fixed_controls_scaled"] = torch.as_tensor(fixed/q_ranges, dtype=torch.float32, device=device)
    return result


def _input_records(output):
    return {name: {"sha256": sha256(output/name), "bytes": (output/name).stat().st_size} for name in
        ("dataset/manifest.json", "dataset/normalizer.json", "condition_normalizer.json", "validation_draws.npz")}


def prepare_training(output, *, device="cuda"):
    """Freeze common data/scalers/validation draws and budget before either fit."""
    output = Path(output).resolve()
    if (output/"training_config.json").exists():
        return load_training_bundle(output, device="cpu")
    runtime = configure_runtime(device)
    dataset = load_reference_dataset(output/"dataset/manifest.json")
    control_normalizer = ControlNormalizer.from_dict(json.loads((output/"dataset/normalizer.json").read_text(encoding="utf-8")))
    train, val = dataset.indices("train"), dataset.indices("val")
    if (len(train), len(val)) != (7, 6): raise ValueError("pilot freezes the declared seven TRAIN / six VAL references")
    train_ids = tuple(dataset.samples[i]["sample_id"] for i in train)
    if set(control_normalizer.training_sample_ids) != set(train_ids): raise ValueError("control scaler is not fit to current TRAIN allocation")
    encoded = [encode_typed_condition(task) for task in dataset.tasks]
    condition_normalizer = ConditionNormalizer.fit(encoded, train, train_ids)
    _write(output/"condition_normalizer.json", condition_normalizer.to_dict())
    q_ranges, dq_scales = _physical_scales()
    # Fixed 16 validation draws per existing reference, drawn independently of
    # model weights; source tasks stay the three complete holdout groups.
    val_indices = np.repeat(val, 16)
    generator = torch.Generator(device="cpu").manual_seed(VAL_SEED)
    times = torch.randint(0, 100, (len(val_indices),), generator=generator)
    epsilon = torch.randn((len(val_indices),30,17),generator=generator)
    np.savez_compressed(output/"validation_draws.npz", indices=val_indices,
        timesteps=times.numpy(), epsilon=epsilon.numpy())
    config = {"schema": TRAINING_SCHEMA, "dataset_sha256": dataset.manifest_sha256,
        "models": ["M0", "M1"], "training_seed": TRAIN_SEED, "training_draw_seed": DRAW_SEED,
        "validation_draw_seed": VAL_SEED, "smoke_draw_seed": SMOKE_SEED,
        "optimizer_updates": 6000, "batch_size": 32, "effective_batch_size": 32,
        "optimizer": "AdamW", "learning_rate": .0001, "weight_decay": .01,
        "gradient_norm_clip": 1., "training_sampling": "uniform TRAIN reference with replacement",
        "paired_random_protocol": "separate CPU generator reset to training_draw_seed for each model; draw sample indices, t, epsilon in fixed order each update",
        "training_reference_slots_per_model": 6000*32, "distinct_train_references": 7,
        "independent_train_tasks": 3, "independent_validation_tasks": 3,
        "validation_every_updates": 250, "training_curve_every_updates": 50,
        "validation_draws_per_reference": 16, "validation_draw_count": len(val_indices),
        "checkpoint_selection": "minimum fixed VAL complete shared loss; ties retain earlier update",
        "early_stopping": False, "TEST_weight_selection": False, "smoke_optimizer_updates": 50,
        "formal_initialization": "fresh same seed; never smoke, previous checkpoint, or old optimizer",
        "noise_schedule": "cosine100", "prediction_type": "internal_v_residual",
        "training_target": "sqrt(alpha_bar)*epsilon-sqrt(1-alpha_bar)*x0", "sampler": "unclipped DDIM20",
        "decoded_loss": {"loss": "L_v + .1*L_decoded_q + .01*L_decoded_dq",
            "alpha_bar_minimum": .1, "q_ranges": q_ranges.tolist(), "dq_scales": dq_scales.tolist(),
            "q_range_source": "continuum work-coordinate range2; rigid original planner upper-lower",
            "dq_source": "original nominal planner_velocity_limits", "sample_count": 136,
            "time_variable": "physical time t from0to27 before terminal progress repair, applied zero times to both labels/predictions",
            "free_base_or_QP_backpropagation": False},
        "condition_schema": schema_config(), "runtime": runtime, "source_sha256": _sources(),
        "input_files": _input_records(output), "physics_steps": 0, "native_distance_queries": 0,
        "wall_deployment": "NOT_MET", "hardware_control": "NOT_ESTABLISHED"}
    _write(output/"training_config.json", config)
    return load_training_bundle(output, device="cpu")


def _guard(output, config):
    current = json.loads((output/"training_config.json").read_text(encoding="utf-8"))
    if object_sha(current) != object_sha(config):
        raise ValueError("paired training configuration changed during the run")
    if _sources() != config["source_sha256"]:
        raise ValueError("training source bytes changed after frozen config")
    if _input_records(output) != config["input_files"]:
        raise ValueError("training dataset/scalers/fixed validation draws changed after freeze")


def load_training_bundle(output, *, device="cpu"):
    output = Path(output).resolve()
    config = json.loads((output/"training_config.json").read_text(encoding="utf-8"))
    if config.get("schema") != TRAINING_SCHEMA or config.get("optimizer_updates") != 6000 or config.get("batch_size") != 32:
        raise ValueError("shared pilot training budget or schema changed")
    _guard(output, config)
    dataset = load_reference_dataset(output/"dataset/manifest.json")
    if dataset.manifest_sha256 != config["dataset_sha256"]: raise ValueError("frozen dataset mismatch")
    control = ControlNormalizer.from_dict(json.loads((output/"dataset/normalizer.json").read_text(encoding="utf-8")))
    condition = ConditionNormalizer.from_dict(json.loads((output/"condition_normalizer.json").read_text(encoding="utf-8")))
    expected_ids = {dataset.samples[i]["sample_id"] for i in dataset.indices("train")}
    if set(control.training_sample_ids) != expected_ids or set(condition.training_sample_ids) != expected_ids:
        raise ValueError("shared normalizers differ from frozen TRAIN-only allocation")
    q_ranges, dq_scales = np.asarray(config["decoded_loss"]["q_ranges"]), np.asarray(config["decoded_loss"]["dq_scales"])
    return {"output": output, "dataset": dataset, "control_normalizer": control,
        "condition_normalizer": condition, "q_ranges": q_ranges, "dq_scales": dq_scales,
        "config": config, "conditions": make_conditions(dataset.tasks, condition, q_ranges=q_ranges, device=device),
        "clean": torch.as_tensor(control.normalize(dataset.controls_free), dtype=torch.float32, device=device)}


def paired_draw(generator, train_indices, batch_size=32):
    indices = torch.as_tensor(train_indices, dtype=torch.long)
    chosen = indices[torch.randint(0,len(indices),(batch_size,),generator=generator)]
    times = torch.randint(0,100,(batch_size,),generator=generator)
    epsilon = torch.randn((batch_size,30,17),generator=generator)
    return chosen, times, epsilon


def _condition_subset(condition, indices):
    return {key: value[indices.to(value.device)] for key,value in condition.items()}


def _scalar_terms(loss, terms):
    return {"total": float(loss.detach()), **{key: float(value.detach()) for key,value in terms.items()}}


def _state_sha(model):
    digest = hashlib.sha256()
    for name,value in sorted(model.state_dict().items()):
        digest.update(name.encode()+b"\0"); digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _make_model(bundle, model_name, device):
    if model_name not in ("M0","M1"): raise ValueError("only declared M0/M1 are permitted")
    torch.manual_seed(bundle["config"]["training_seed"])
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(bundle["config"]["training_seed"])
    model = PilotDDPM(model_name,FLAT_DIM,global_dim=GLOBAL_DIM,token_dim=TOKEN_DIM,type_count=TYPE_COUNT).to(device)
    normalizer = bundle["control_normalizer"]
    decoder = DecodedReferenceLoss(normalizer.control_mean,normalizer.control_scale,bundle["q_ranges"],bundle["dq_scales"],
        sample_count=bundle["config"]["decoded_loss"]["sample_count"]).to(device)
    return model, decoder


def _optimizer(model, config):
    return torch.optim.AdamW(model.parameters(),lr=config["learning_rate"],weight_decay=config["weight_decay"])


def _step(model,decoder,optimizer,clean,condition,times,epsilon,config):
    optimizer.zero_grad(set_to_none=True)
    loss,terms = paired_training_loss(model,decoder,clean,condition,times,epsilon)
    if not torch.isfinite(loss): raise FloatingPointError("shared training loss is not finite")
    loss.backward()
    gradients = [p.grad for p in model.parameters() if p.grad is not None]
    if not gradients or not all(torch.all(torch.isfinite(g)) for g in gradients):
        raise FloatingPointError("model gradient missing or nonfinite")
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(),config["gradient_norm_clip"],error_if_nonfinite=True)
    optimizer.step()
    return _scalar_terms(loss,terms), float(norm)


def _condition_consumption(model,bundle,device):
    task = json.loads(json.dumps(bundle["dataset"].tasks[int(bundle["dataset"].indices("train")[0])]))
    target_variant = json.loads(json.dumps(task)); target_variant["requirements"][-1]["position_m"][0] += .01
    obstacle_variant = json.loads(json.dumps(task)); obstacle_variant["scenario"]["workspace_obstacles"][0]["center_w"][0] += .03
    condition = make_conditions([task,target_variant,obstacle_variant],bundle["condition_normalizer"],q_ranges=bundle["q_ranges"],device=device)
    generator = torch.Generator(device="cpu").manual_seed(SMOKE_SEED+1)
    noisy = torch.randn((1,30,17),generator=generator).repeat(3,1,1).to(device)
    times = torch.full((3,),25,dtype=torch.long,device=device)
    with torch.no_grad(): output = model.denoiser(noisy,times,condition)
    return {"fixed_noisy_latent_and_timestep": True,
        "task_goal_change_output_max_abs": float(torch.max(torch.abs(output[1]-output[0]))),
        "obstacle_change_output_max_abs": float(torch.max(torch.abs(output[2]-output[0]))),
        "output_finite": bool(torch.all(torch.isfinite(output))),
        "correct_obstacle_avoidance_established_by_output_change": False}


def smoke_training(output):
    """Each model gets exactly fifty fixed-batch overfit updates, then discarded."""
    bundle = load_training_bundle(output,device="cpu"); config=bundle["config"]
    device=config["runtime"]["device"]; runtime=configure_runtime(device)
    bundle = load_training_bundle(output,device=device)
    directory = bundle["output"]/"smoke"; directory.mkdir(exist_ok=False)
    results=[]
    for model_name in config["models"]:
        model,decoder = _make_model(bundle,model_name,device); model.train()
        optimizer = _optimizer(model,config)
        initial_state = {name:p.detach().cpu().clone() for name,p in model.named_parameters()}
        initial_sha = _state_sha(model)
        generator=torch.Generator(device="cpu").manual_seed(SMOKE_SEED)
        indices,times,epsilon=paired_draw(generator,bundle["dataset"].indices("train"))
        clean=bundle["clean"][indices.to(device)]; condition=_condition_subset(bundle["conditions"],indices)
        times,epsilon=times.to(device),epsilon.to(device)
        with torch.no_grad(): before,_=paired_training_loss(model,decoder,clean,condition,times,epsilon)
        _synchronize(device); started=time.perf_counter(); norms=[]; curve=[]
        for step in range(1,51):
            terms,norm=_step(model,decoder,optimizer,clean,condition,times,epsilon,config)
            norms.append(norm); curve.append({"update":step,**terms,"gradient_norm_before_clip":norm})
        _synchronize(device); elapsed=time.perf_counter()-started
        with torch.no_grad(): after,after_terms=paired_training_loss(model,decoder,clean,condition,times,epsilon)
        delta_sq=sum(float((p.detach().cpu()-initial_state[name]).double().square().sum()) for name,p in model.named_parameters())
        consumption=_condition_consumption(model,bundle,device)
        row={"schema":"v6_4_b1_fixed_batch_overfit_smoke_v1","model":model_name,"optimizer_updates":50,
            "batch_size":32,"initial_loss":float(before),"final_loss":float(after),"final_components":_scalar_terms(after,after_terms),
            "loss_decreased":bool(after<before),"gradient_finite_all_updates":True,"maximum_gradient_norm_before_clip":max(norms),
            "parameter_delta_l2":float(np.sqrt(delta_sq)),"initial_state_sha256":initial_sha,"final_state_sha256":_state_sha(model),
            "actual_weight_update_established":delta_sq>0,"condition_consumption":consumption,
            "elapsed_training_wall_s":elapsed,"mean_update_wall_s":elapsed/50,
            "estimated_6000_training_update_wall_s":elapsed*120,"runtime":runtime,
            "architecture_identity":model.architecture_identity(),"formal_uses_smoke_weights":False,
            "source_sha256_before":config["source_sha256"],"source_sha256_after":_sources(),
            "physics_steps":0,"native_distance_queries":0}
        _write(directory/f"{model_name}_report.json",row)
        with (directory/f"{model_name}_curve.jsonl").open("x",encoding="utf-8") as stream:
            for entry in curve: stream.write(json.dumps(entry,allow_nan=False)+"\n")
        _guard(bundle["output"],config)
        results.append(row)
        print(json.dumps({"model":model_name,"smoke_updates":50,"loss_before":float(before),"loss_after":float(after),
            "mean_update_wall_s":elapsed/50,"parameter_delta_l2":row["parameter_delta_l2"]}),flush=True)
    _write(directory/"summary.json",{"models":results,"formal_training_started":False,
        "smoke_passed":all(r["loss_decreased"] and r["actual_weight_update_established"] and r["condition_consumption"]["output_finite"]
            and r["condition_consumption"]["task_goal_change_output_max_abs"]>0
            and r["condition_consumption"]["obstacle_change_output_max_abs"]>0 for r in results),
        "no_learning_or_task_advantage_claim":True})
    return results


@torch.no_grad()
def fixed_validation(model,decoder,bundle):
    with np.load(bundle["output"]/"validation_draws.npz",allow_pickle=False) as saved:
        indices=torch.as_tensor(saved["indices"],dtype=torch.long)
        times=torch.as_tensor(saved["timesteps"],dtype=torch.long,device=bundle["clean"].device)
        epsilon=torch.as_tensor(saved["epsilon"],dtype=torch.float32,device=bundle["clean"].device)
    model.eval()
    # A single validation batch preserves the complete eligible-alpha reduction;
    # no biased average of per-chunk derived loss means is used.
    loss,terms=paired_training_loss(model,decoder,bundle["clean"][indices.to(bundle["clean"].device)],
        _condition_subset(bundle["conditions"],indices),times,epsilon)
    model.train()
    if not torch.isfinite(loss): raise FloatingPointError("fixed shared validation loss nonfinite")
    return _scalar_terms(loss,terms)


def _checkpoint(model,bundle,update,validation_loss,initial_sha,draw_sha,*,kind):
    return {"schema":"v6_4_b1_trained_pilot_checkpoint_v1","model_name":model.model_name,
        "state_dict":{k:v.detach().cpu() for k,v in model.state_dict().items()},
        "model_config":model.config.to_dict(),"training_config":bundle["config"],
        "architecture_identity":model.architecture_identity(),
        "condition_normalizer":bundle["condition_normalizer"].to_dict(),
        "control_normalizer":bundle["control_normalizer"].to_dict(),
        "dataset_sha256":bundle["dataset"].manifest_sha256,"selected_update":update,
        "best_validation_loss":validation_loss,"weight_selection":kind,
        "source_sha256":bundle["config"]["source_sha256"],"initial_state_sha256":initial_sha,
        "paired_training_draw_prefix_sha256":draw_sha,"seed":bundle["config"]["training_seed"],
        "smoke_weights_used":False,"TEST_weight_selection":False,"future_actual_trace_condition":False}


def train_model(output,model_name):
    """Exactly one fresh formal fit; caller owns launching the two main jobs."""
    bundle=load_training_bundle(output,device="cpu");config=bundle["config"]
    device=config["runtime"]["device"];runtime=configure_runtime(device)
    if runtime!=config["runtime"]:raise ValueError("runtime differs from frozen paired training configuration")
    bundle=load_training_bundle(output,device=device)
    smoke_path=bundle["output"]/"smoke/summary.json"
    if not smoke_path.is_file() or json.loads(smoke_path.read_text(encoding="utf-8")).get("smoke_passed") is not True:
        raise ValueError("both fixed-batch gradient/condition smoke diagnostics must pass before formal fit")
    directory=bundle["output"]/"models"/model_name;directory.mkdir(parents=True,exist_ok=False)
    model,decoder=_make_model(bundle,model_name,device);model.train();optimizer=_optimizer(model,config)
    initial_sha=_state_sha(model)
    generator=torch.Generator(device="cpu").manual_seed(config["training_draw_seed"])
    draw_digest=hashlib.sha256();exposures=np.zeros(len(bundle["dataset"].samples),dtype=np.int64)
    best=float("inf");selected_update=0;max_norm=0.;started=time.perf_counter();_synchronize(device)
    with (directory/"training_curve.jsonl").open("x",encoding="utf-8") as stream:
        for step in range(1,config["optimizer_updates"]+1):
            indices,times,epsilon=paired_draw(generator,bundle["dataset"].indices("train"),config["batch_size"])
            for value in (indices,times,epsilon):draw_digest.update(value.contiguous().numpy().tobytes())
            exposures+=np.bincount(indices.numpy(),minlength=len(exposures))
            condition=_condition_subset(bundle["conditions"],indices)
            terms,norm=_step(model,decoder,optimizer,bundle["clean"][indices.to(device)],condition,times.to(device),epsilon.to(device),config)
            max_norm=max(max_norm,norm)
            if step%config["training_curve_every_updates"]==0:
                _synchronize(device)
                stream.write(json.dumps({"kind":"train","update":step,**terms,"gradient_norm_before_clip":norm,
                    "elapsed_wall_s":time.perf_counter()-started},allow_nan=False)+"\n");stream.flush()
            if step%config["validation_every_updates"]==0:
                validation=fixed_validation(model,decoder,bundle)
                _guard(bundle["output"],config)
                if validation["total"]<best:
                    best=validation["total"];selected_update=step
                    checkpoint=_checkpoint(model,bundle,step,best,initial_sha,draw_digest.hexdigest(),kind="minimum_fixed_VAL_complete_shared_loss")
                    temporary=directory/"checkpoint.tmp";torch.save(checkpoint,temporary);temporary.replace(directory/"checkpoint.pt")
                stream.write(json.dumps({"kind":"validation","update":step,**validation,"best_validation_loss":best,
                    "selected_update":selected_update,"elapsed_wall_s":time.perf_counter()-started},allow_nan=False)+"\n");stream.flush()
                print(json.dumps({"model":model_name,"update":step,"validation_loss":validation["total"],
                    "best_validation_loss":best,"selected_update":selected_update}),flush=True)
    _synchronize(device);elapsed=time.perf_counter()-started;_guard(bundle["output"],config)
    torch.save(_checkpoint(model,bundle,config["optimizer_updates"],best,initial_sha,draw_digest.hexdigest(),kind="last_optimizer_update_not_selected_by_TEST"),directory/"last_checkpoint.pt")
    torch.save(optimizer.state_dict(),directory/"optimizer_state.pt")
    report={"schema":"v6_4_b1_actual_paired_training_report_v1","model":model_name,
        "optimizer_updates":config["optimizer_updates"],"effective_batch_size":32,
        "reference_slots_seen":int(exposures.sum()),"sample_exposure_counts":{s["sample_id"]:int(exposures[i]) for i,s in enumerate(bundle["dataset"].samples)},
        "selected_update":selected_update,"best_validation_loss":best,"checkpoint_path":str(directory/"checkpoint.pt"),
        "checkpoint_sha256":sha256(directory/"checkpoint.pt"),"last_checkpoint_sha256":sha256(directory/"last_checkpoint.pt"),
        "initial_state_sha256":initial_sha,"last_state_sha256":_state_sha(model),"paired_draws_sha256":draw_digest.hexdigest(),
        "architecture_identity":model.architecture_identity(),"maximum_gradient_norm_before_clip":max_norm,
        "elapsed_wall_s":elapsed,"runtime":runtime,"source_sha256_before":config["source_sha256"],"source_sha256_after":_sources(),
        "source_unchanged":True,"normalizers_fit_TRAIN_only":True,"smoke_weights_used":False,
        "complete_main_fit":True,"physics_steps":0,"native_distance_queries":0,
        "task_success_or_architecture_advantage_from_loss":False}
    _write(directory/"training_report.json",report)
    return report


def load_selected_model(output,model_name,*,device="cpu"):
    sampler=load_pilot_sampler(Path(output)/"models"/model_name/"checkpoint.pt",device=device)
    return sampler.model,sampler.checkpoint


class PilotSampler:
    def __init__(self,path,device="cpu"):
        self.path=Path(path).resolve();self.device=torch.device(device)
        self.runtime=configure_runtime(device)
        self.checkpoint=torch.load(self.path,map_location="cpu",weights_only=False)
        if self.checkpoint.get("schema")!="v6_4_b1_trained_pilot_checkpoint_v1" or self.checkpoint.get("smoke_weights_used") is not False:
            raise ValueError("sampler requires a real fresh formal pilot checkpoint")
        config=self.checkpoint["model_config"]
        self.model=PilotDDPM(self.checkpoint["model_name"],FLAT_DIM,global_dim=GLOBAL_DIM,token_dim=TOKEN_DIM,type_count=TYPE_COUNT)
        if config!=self.model.config.to_dict():raise ValueError("checkpoint differs from frozen pilot network/schedule")
        if _sources()!=self.checkpoint["source_sha256"]:raise ValueError("sampler source differs from trained pilot identity")
        self.model.load_state_dict(self.checkpoint["state_dict"],strict=True);self.model.to(device);self.model.eval()
        self.control_normalizer=ControlNormalizer.from_dict(self.checkpoint["control_normalizer"])
        self.condition_normalizer=ConditionNormalizer.from_dict(self.checkpoint["condition_normalizer"])
        self.q_ranges=np.asarray(self.checkpoint["training_config"]["decoded_loss"]["q_ranges"])
        self.checkpoint_sha256=sha256(self.path)

    def condition(self,task):
        return make_conditions([task],self.condition_normalizer,q_ranges=self.q_ranges,device=self.device)

    @torch.no_grad()
    def sample(self,task,initial_noise_np):
        noise=np.asarray(initial_noise_np,dtype=np.float32)
        if noise.shape!=(30,17) or not np.all(np.isfinite(noise)):raise ValueError("finite paired initial latent30x17 required")
        condition=self.condition(task);latent=torch.as_tensor(noise[None],device=self.device)
        _synchronize(self.device);started=time.perf_counter()
        normalized=self.model.sample_ddim(condition,initial_noise=latent)
        _synchronize(self.device);elapsed=time.perf_counter()-started
        free=self.control_normalizer.denormalize(normalized.cpu().numpy()[0])
        if not np.all(np.isfinite(free)):raise FloatingPointError("raw sampled reference controls nonfinite")
        metadata={"model":self.model.model_name,"checkpoint_path":str(self.path),"checkpoint_sha256":self.checkpoint_sha256,
            "model_config":self.checkpoint["model_config"],"condition_schema":schema_config()["schema"],
            "control_normalizer_sha256":object_sha(self.control_normalizer.to_dict()),
            "condition_normalizer_sha256":object_sha(self.condition_normalizer.to_dict()),
            "selected_update":self.checkpoint["selected_update"],"checkpoint_selection":self.checkpoint["weight_selection"],
            "best_validation_loss":self.checkpoint["best_validation_loss"],"TEST_weight_selection":False,
            "latent_float32_sha256":hashlib.sha256(noise.astype('<f4',copy=False).tobytes()).hexdigest(),
            "inference_wall_s":elapsed,"ddim_steps":20,"clipping_projection_repair_or_fallback":False,
            "inference_runtime":self.runtime,
            "physics_steps":0,"native_distance_queries":0,"future_actual_trace_condition":False}
        return free,metadata


def load_pilot_sampler(checkpoint,device="cpu"):
    return PilotSampler(checkpoint,device=device)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command",choices=("prepare","smoke","train"))
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--device",default="cuda",choices=("cpu","cuda"))
    parser.add_argument("--model",choices=("M0","M1"))
    args=parser.parse_args()
    if args.command=="prepare":
        bundle=prepare_training(args.output,device=args.device)
        print(json.dumps({"prepared":True,"training_config":str(bundle["output"]/"training_config.json"),
            "config_sha256":sha256(bundle["output"]/"training_config.json"),"formal_training_started":False}),flush=True)
    elif args.command=="smoke":smoke_training(args.output)
    else:
        if args.model is None:parser.error("train requires --model")
        print(json.dumps(train_model(args.output,args.model),ensure_ascii=False),flush=True)


if __name__=="__main__":main()
