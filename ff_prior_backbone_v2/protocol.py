from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONTROL = ROOT / "artifacts/v7_1/expand256_train_dev_001"
BASE = ROOT / "artifacts/v7_1/smoke_002"
DEFAULT_OUT = ROOT / "artifacts/v7_1/backbone_pair_smoke_002"
METHODS = ("mlp", "transformer")
P = {
    "version": "same_latent_backbone_pair_v2_storage_fix",
    "TRAIN": 14222, "VAL": 48, "latent_dim": 102, "condition_dim": 65,
    "updates": 6000, "batch": 64, "lr": 2e-4, "weight_decay": .01,
    "training_seed": 20263001, "initialization_seed": 20263401,
    "diagnostic_seed": 20263402, "diagnostic_steps_per_model": 2,
    "audit_updates": [1, 3000, 6000], "checkpoint_every": 20,
    "loss": "v_mse + 0.05 teacher_local_G", "local_G_weight": .05,
    "nonlinear_world_training_events": 0,
    "initialization": "both neural networks fresh; same frozen TRAIN-fitted codec/center",
    "parameter_ratio_max": 1.05,
    "transformer": {"width": 104, "heads": 4, "blocks": 2, "ff_width": 208,
                    "tokens": 6, "token_dimension": 17, "dropout": 0.0,
                    "token_meaning": "ordered temporal basis coefficients, not time samples"},
    "val_seeds": [20263111, 20263113],
    "sampling_seeds": [20262119, 20262123, 20262129],
    "ddim_steps": 16, "candidates_per_task": 5,
    "new_candidates": 360, "new_execution_units": 72,
    "reuse_methods": ["structured", "learned_mean"],
    "reused_candidates": 360, "reused_execution_units": 72,
    "candidate_coupled_cap": 113, "repair_cap": 110,
    "candidate_intervals_cap": 15872, "candidate_wall_s": 15.,
    "coupled_cap": 42000, "wall_budget_s": 21600.,
    "max_output_bytes": 20 * 1024**3, "cpu_threads": 2,
    "engineering_candidate_min": 144, "engineering_execution_min": 29,
    "gate_to_DEV": "complete finite training, exact identity and RNG, reload; no superiority gate",
    "VAL_checkpoint_selection": False, "VAL_gradients": False,
    "TEST": False, "expand": False, "pilot": False, "study": False,
    "strict_multimodal": False, "goal_relabel": False,
    "scientific_scope": "one paired training seed; architecture package comparison, not attention-only attribution",
}
