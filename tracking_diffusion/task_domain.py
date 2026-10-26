"""Versioned local task family. Task, solver and model randomness are separate."""
from functools import lru_cache
import hashlib
import json
from pathlib import Path

import numpy as np

from v6_lite.irregular_waypoints import IrregularWaypointTarget

DOMAIN = "local10mm-v1"
CONFIG = Path(__file__).with_name("config") / "local10mm_v1.json"
STREAMS = {"curve": 11, "target_position": 23, "target_motion": 37,
           "solver": 53, "model": 71}


@lru_cache(maxsize=1)
def profile():
    value = json.loads(CONFIG.read_text(encoding="utf-8"))
    if value["domain"] != DOMAIN:
        raise ValueError("Unexpected task domain")
    return value


def profile_hash():
    return hashlib.sha256(CONFIG.read_bytes()).hexdigest()


def sample_task(seed, nominal=False, overrides=None):
    overrides = overrides or {}
    rng = {key: np.random.default_rng(np.random.SeedSequence([
        int(overrides.get(key, seed)), tag])) for key, tag in STREAMS.items()}
    p = profile()
    perturb = rng["curve"].uniform(-.010, .010, (7, 3))
    shift = rng["target_position"].uniform([-.012, -.008, -.010], [.012, .008, .010])
    velocity = rng["target_motion"].uniform([-.0025, .001, -.0015], [.0025, .004, .0015])
    omega = rng["target_motion"].uniform(-.0015, .0015, 3)
    if nominal:
        perturb[:] = 0.
        shift, velocity, omega = [np.array(p[k]) for k in (
            "target_shift", "target_velocity", "target_omega_local")]
    return {"curve_delta_m": perturb.tolist(), "target_shift": shift.tolist(),
            "target_velocity": velocity.tolist(), "target_omega_local": omega.tolist(),
            "solver_seed": int(rng["solver"].integers(0, 2**31-1)),
            "model_seed": int(rng["model"].integers(0, 2**31-1)),
            "stream_roots": {k: int(overrides.get(k, seed)) for k in STREAMS}}


def build_path(pc, rc, parameters):
    p = profile()
    points = np.asarray(p["waypoints_m"]) + np.asarray(parameters["curve_delta_m"])
    return IrregularWaypointTarget(np.array(pc), points, np.array(rc), 4.5, 21.,
        np.asarray(p["segment_durations_s"]),
        {"preset": DOMAIN, "nominal_seed": p["nominal_seed"],
         "waypoint_count": 7, "waypoint_points_m": points.tolist(),
         "perturbation_m": parameters["curve_delta_m"], "profile_sha256": profile_hash()})
