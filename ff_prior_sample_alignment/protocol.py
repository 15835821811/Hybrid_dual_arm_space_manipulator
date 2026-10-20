from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONTROL = ROOT / "artifacts/v7_1/expand256_train_dev_001"
BASE = ROOT / "artifacts/v7_1/smoke_002"
DEFAULT_OUT = ROOT / "artifacts/v7_1/sample_alignment_001"
METHODS = ("plain", "aligned")
P = {
    "version": "full16_sampling_endpoint_alignment_1",
    "TRAIN": 14222, "VAL": 48, "updates": 1200, "batch": 64,
    "lr": 2e-4, "weight_decay": .01, "training_seed": 20263001,
    "rollout_seed": 20263501, "rollout_batch": 8,
    "world_every": 20, "world_events": 60, "diagnostic_gradients": 2,
    "rollout_gradient_norm_cap": .5, "world_gradient_norm_cap": .5,
    "teacher_loss": "full16 raw normalized coefficients MSE + .05 teacher localG MSE",
    "world_loss": "full16 prepared output; current102 group Jacobian + smooth analytic chain",
    "world_gradient_intervals": 256, "world_fine_intervals": 512,
    "audit_updates": [1, 600, 1200],
    "val_seeds": [20263111, 20263113],
    "sampling_seeds": [20262119, 20262123, 20262129],
    "ddim_steps": 16, "reuse_methods": ["structured", "learned_mean"],
    "new_candidates": 360, "new_execution_units": 72,
    "reused_candidates": 360, "reused_execution_units": 72,
    "candidate_coupled_cap": 113, "candidate_intervals_cap": 15872,
    "candidate_wall_s": 15., "cpu_threads": 2,
    "coupled_cap": 75000, "wall_budget_s": 21600.,
    "max_output_bytes": 20 * 1024**3,
    "engineering_candidate_min": 144, "engineering_execution_min": 29,
    "gate_to_DEV": "finite complete training and exact identities; no advantage gate",
    "VAL_gradients": False, "VAL_checkpoint_selection": False,
    "TEST": False, "expand": False, "goal_relabel": False,
    "primary_physics_relaxation": False,
}
