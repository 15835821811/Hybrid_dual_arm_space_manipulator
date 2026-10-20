"""Read-only paired statistics; no MuJoCo step or model fitting."""
from pathlib import Path
import numpy as np
from ff_prior.storage import read_json,atomic_json,file_digest,digest

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/"artifacts/v7_1/waypoint_domain_002"
OLD=ROOT/"artifacts/v7_1/waypoint_scene_001"
PKG=ROOT/"ff_prior_waypoint_domain"


def main():
    summary=read_json(OUT/"summary.json")
    old=read_json(OLD/"audited_summary.json")
    records={}
    for name,result in summary["results"].items():
        file=OUT/name/"native_trace.npz"
        assert file_digest(file)==result["native_trace_sha256"]
        with np.load(file) as a,np.load(OLD/name/"native_trace.npz") as b:
            n=len(a["time"])
            assert np.array_equal(a["desired"],b["desired"][:n])
            assert np.array_equal(a["qpos"][0],b["qpos"][0])
            assert np.array_equal(a["qvel"][0],b["qvel"][0])
            actual=a["q17"][:,:10];ref=a["qref"][:,:10]
            record=dict(initial_exactly_matches_old=True,reference_exactly_matches_old_prefix=True,
                actual_max_abs_rad=float(np.max(abs(actual))),reference_max_abs_rad=float(np.max(abs(ref))),
                actual_outside_frames=int(np.any(abs(actual)>1,axis=1).sum()),
                reference_outside_frames=int(np.any(abs(ref)>1,axis=1).sum()),
                actual_q17_exactly_matches_old=bool(np.array_equal(a["q17"],b["q17"])),
                max_normalized_raw_torque=float(np.max(abs(a["raw_torque"])/np.r_[np.full(60,4.),np.full(7,80.)])),
                total_absolute_torque_impulse_Nms=float(np.sum(abs(a["raw_torque"]))* .002),
                joint_path_length_rad=float(np.sum(np.linalg.norm(np.diff(a["q17"],axis=0),axis=1))),
                old_actual_max_abs_rad=float(np.max(abs(b["q17"][:,:10]))),
                old_status=old["results"][name]["status"],new_status=result["status"])
        records[name]=record
    paired={}
    for scene in ("stationary","drifting"):
        baseline=summary["results"][scene+"/task_controller"]
        prior=summary["results"][scene+"/diffusion_null_prior"]
        paired[scene]={}
        for metric in ("position_mm","rotation_deg"):
            paired[scene][metric]={}
            for stat in ("mean","p95","maximum","final"):
                a=np.array(baseline[metric][stat]);b=np.array(prior[metric][stat])
                paired[scene][metric][stat]=dict(baseline=a,prior=b,prior_minus_baseline=b-a,
                    percent_change=100*(b-a)/np.maximum(abs(a),1e-12))
    atomic_json(OUT/"paired_statistics.json",dict(records=records,paired=paired,
        interpretation="Two fixed development cases; trajectory time samples are not independent task replicates. No statistical superiority claim."))
    for name,r in records.items():
        print(name,r["new_status"],"domain",r["actual_max_abs_rad"],r["reference_max_abs_rad"],
              "old identical",r["actual_q17_exactly_matches_old"])
    print("paired",paired)


if __name__=="__main__":main()
