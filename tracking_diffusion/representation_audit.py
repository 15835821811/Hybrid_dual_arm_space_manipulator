"""Audit every deployment window; never silently drop difficult labels."""
import os
for _k in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ[_k] = "1"
import argparse
import json
from pathlib import Path

import numpy as np

from model_test.robot_model_spec_v5 import RobotModelSpecV5
from .pipeline import write_json
from .trajectory import JointChunk, fit_chunk
from .spline import SplineChunk, fit_spline


def bound_report(chunk, spec):
    q = chunk.controls
    v = 3*np.diff(q, axis=0)/chunk.horizon
    a = 2*np.diff(v, axis=0)/chunk.horizon
    return {"position": bool(np.all(q >= spec.planner_lower+.04) and np.all(q <= spec.planner_upper-.04)),
        "velocity": bool(np.all(np.abs(v) <= .70*spec.planner_velocity_limits+1e-8)),
        "acceleration": bool(np.all(np.abs(a) <= spec.planner_acceleration_limits+1e-8)),
        "max_velocity_ratio": float(np.max(np.abs(v)/(.70*spec.planner_velocity_limits))),
        "max_acceleration_ratio": float(np.max(np.abs(a)/spec.planner_acceleration_limits))}


def audit(folder, output, horizons=(1., .5), controls=4):
    spec = RobotModelSpecV5.from_urdf(Path("dual_arm_space_robot_2026/urdf/dual_arm_space_robot_2026.urdf"))
    reports = []
    for episode in sorted(Path(folder).glob("pilot_*/trace.npz")):
        with np.load(episode) as data:
            q, dq, times = data["q"], data["dq"], data["time"]
        for horizon in horizons:
            windows, failures = [], []
            for k in range(0, len(q)-1, 5):
                end = min(k+int(round(horizon/.02)), len(q)-1)
                actual_horizon = (end-k)*.02
                if controls == 4:
                    free, error = fit_chunk(q[k:end+1], dq[k])
                    chunk = JointChunk.decode(free, q[k], dq[k], actual_horizon)
                    bounds = bound_report(chunk, spec)
                else:
                    free, error = fit_spline(q[k:end+1], dq[k], actual_horizon, controls, spec)
                    chunk = SplineChunk.decode(free, q[k], dq[k], actual_horizon)
                    bounds = chunk.bound_report(spec)
                ok = error <= .002 and all(bounds[x] for x in ("position", "velocity", "acceleration"))
                window = {"time": float(times[k]), "horizon": actual_horizon,
                          "fit_error_rad": error, "bounds": bounds, "accepted": ok}
                windows.append(window)
                if not ok:
                    failures.append(window)
            reports.append({"episode": episode.parent.name, "horizon": horizon, "controls": controls,
                "total": len(windows), "accepted": len(windows)-len(failures),
                "initial_accepted": windows[0]["accepted"], "tail_accepted": windows[-1]["accepted"],
                "max_fit_error_rad": max(w["fit_error_rad"] for w in windows),
                "failures": failures})
    write_json(output, {"representation": "clamped cubic; fixed measured q,dq at start",
        "tail_rule": "shorten horizon to remaining episode time", "episodes": reports})
    print(json.dumps([{k:v for k,v in r.items() if k != "failures"} for r in reports],indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folder", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--controls", type=int, default=4)
    args = parser.parse_args()
    audit(args.folder, args.output, controls=args.controls)


if __name__ == "__main__":
    main()
