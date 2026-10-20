from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "artifacts/v7_1/sample_alignment_001"
BASE = ROOT / "artifacts/v7_1/smoke_002"
OUT = ROOT / "artifacts/v7_1/capture_feasibility_001"
PACKAGE = ROOT / "ff_prior_capture_feasibility"
P = {
    "version": "fixed_capture_feasibility_1",
    "task": "capture_5_obstacles", "split": "APPLICATION_DEV_DIAGNOSTIC",
    "seeds": 12, "teacher_iterations": 10, "codec_repair_iterations": 10,
    "seed_offset_rad": .45, "seed_joints": [10, 11, 12, 14],
    "phase_intervals": 128, "difference_rad": .0001, "trust_rad": .15,
    "active_buffer_m": .004, "active_pair_cap": 240,
    "solver_goal_fraction": .65, "accept_goal_fraction": .8,
    "repair_clearance_m": .0002, "collision_scale_m": .003,
    "line_scales": [1., .5, .25, .125], "solver_max_iterations": 80,
    "validation_intervals": [256, 512, 1024, 2048],
    "teacher_execution_limit": 2, "execution_limit": 5,
    "coupled_cap": 45000, "wall_s": 14400, "cpu_threads": 2,
    "output_bytes_cap": 8 * 1024**3,
    "TRAIN_added": 0, "training": False, "TEST": False,
    "physics_relaxed": False, "goal_relabel": False,
    "legacy_repair_stages": [12, 13, 14, 15],
    "finite_failed_search_is_infeasibility_proof": False,
}
