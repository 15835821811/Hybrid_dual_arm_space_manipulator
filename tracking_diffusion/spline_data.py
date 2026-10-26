"""All-window spline labels and an explicit teacher-label replay diagnostic."""
import os
for _k in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ[_k] = "1"
import argparse
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np

from model_test.robot_model_spec_v5 import RobotModelSpecV5
from .environment import Task
from .pipeline import run_episode, write_json
from .spline import SplineChunk, fit_spline, VELOCITY_FACTOR
from .task_domain import DOMAIN, profile_hash

HORIZON = .5
CONTROLS = 8


def metadata():
    return {"representation": "clamped_cubic_bspline", "controls": CONTROLS,
            "horizon": HORIZON, "tail": "remaining_task_time", "task_domain": DOMAIN,
            "profile_sha256": profile_hash(), "planning_velocity_factor": VELOCITY_FACTOR,
            "condition": "135D state/reference preview over 1s plus generated horizon (s)"}


def build_episode(path, spec):
    with np.load(path/"trace.npz") as data:
        arrays = {key: data[key] for key in data.files}
    condition, free, starts, horizons, errors, rejected = [], [], [], [], [], []
    q, dq = arrays["q"], arrays["dq"]
    for k in range(0, len(q)-1, 5):
        end = min(k+int(round(HORIZON/.02)), len(q)-1)
        h = (end-k)*.02
        x, error = fit_spline(q[k:end+1], dq[k], h, CONTROLS, spec)
        chunk = SplineChunk.decode(x, q[k], dq[k], h)
        if error > .002 or not chunk.bounds_ok(spec):
            rejected.append({"time": float(arrays["time"][k]), "error": error,
                             "bounds": chunk.bound_report(spec)})
        condition.append(np.r_[arrays["condition"][k],h])
        free.append(x)
        starts.append(arrays["time"][k])
        horizons.append(h)
        errors.append(error)
    return {"condition": np.array(condition,dtype=np.float32), "free": np.array(free,dtype=np.float32),
            "time":np.array(starts), "horizon":np.array(horizons), "fit_error":np.array(errors)}, rejected


def build(teacher_folder, output, train_ids, val_ids):
    if (output/"dataset.npz").exists():
        raise FileExistsError("Use a fresh dataset directory")
    if set(train_ids)&set(val_ids):
        raise ValueError("Task split overlap")
    spec = RobotModelSpecV5.from_urdf(Path("dual_arm_space_robot_2026/urdf/dual_arm_space_robot_2026.urdf"))
    records, report = [], []
    for split, names in ((0,train_ids),(1,val_ids)):
        for name in names:
            episode = teacher_folder/name
            result = json.loads((episode/"result.json").read_text(encoding="utf-8"))
            if not result["accepted_for_teacher_dataset"]:
                raise ValueError(f"Teacher admission failed: {name}")
            arrays, rejected = build_episode(episode,spec)
            report.append({"episode":name,"split":split,"windows":len(arrays["time"]),
                "max_fit_error_rad":float(arrays["fit_error"].max()),"rejected":rejected})
            write_json(output/"coverage.json", {"metadata":metadata(),"episodes":report})
            if rejected:
                raise RuntimeError(f"Incomplete coverage in {name}; no labels silently dropped")
            arrays["split"] = np.full(len(arrays["time"]),split)
            arrays["seed"] = np.full(len(arrays["time"]),result["task"]["seed"])
            records.append(arrays)
    merged = {k:np.concatenate([r[k] for r in records]) for k in records[0]}
    merged["metadata"] = np.array(json.dumps(metadata()))
    np.savez_compressed(output/"dataset.npz",**merged)
    print(json.dumps(report,indent=2),flush=True)


class LabelReplay:
    chunk_type = SplineChunk
    horizon = HORIZON
    task_domain = DOMAIN

    def __init__(self, teacher_episode):
        spec = RobotModelSpecV5.from_urdf(Path("dual_arm_space_robot_2026/urdf/dual_arm_space_robot_2026.urdf"))
        self.arrays,rejected = build_episode(teacher_episode,spec)
        if rejected:
            raise ValueError("Cannot replay incomplete labels")
        self.metadata = dict(metadata(), diagnostic="teacher labels; not a learned-policy success")

    def predict(self,condition,horizon,time,on_step=None):
        index = int(round(time/.1))
        return self.arrays["free"][index:index+1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage",choices=("build","replay"))
    parser.add_argument("--teacher",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--train",nargs="+",default=["pilot_00","pilot_01","pilot_02"])
    parser.add_argument("--validation",nargs="+",default=["pilot_03"])
    parser.add_argument("--episode",default="pilot_00")
    args = parser.parse_args()
    if args.stage == "build":
        build(args.teacher,args.output,args.train,args.validation)
    else:
        episode = args.teacher/args.episode
        result = json.loads((episode/"result.json").read_text(encoding="utf-8"))
        run_episode(Task(**result["task"]),args.output,LabelReplay(episode),"label-replay",progress=True)


if __name__ == "__main__":
    main()
