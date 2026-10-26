"""Saved-data startup analysis, with linear predictions distinguished from dynamics."""
import os
for _key in ("OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","OMP_NUM_THREADS"):
    os.environ[_key]="1"
import argparse
from pathlib import Path
import json
import numpy as np
from .pipeline import write_json
from .spline import SplineChunk


def analyze(folder):
    rows=[]
    for path in sorted(folder.glob("*/*/result.json")):
        result=json.loads(path.read_text(encoding="utf-8"))
        with np.load(path.parent/"trace.npz") as trace:
            t=trace["time"]
            error=np.linalg.norm(trace["error"][:,:3],axis=1)*1000
            keep=t<=1.+1e-8
            row={"task":path.parent.parent.name,"sampling_seed":path.parent.name,
                "flex_peak_first_second_mm":float(error[keep].max()),
                "flex_peak_time_s":float(t[keep][np.argmax(error[keep])]),"windows":[]}
            for planning in sorted(path.parent.glob("planning_*.npz")):
                with np.load(planning) as f:
                    h=float(f["horizon"])
                    times=f["preview_times"]
                    window={"time":float(f["time"]),"stages":{}}
                    for stage in ("before_late_guidance","before_final_gradient","after_gradient","after_projection","final_candidates"):
                        if stage not in f: continue
                        stats=[]
                        for free in f[stage]:
                            chunk=SplineChunk.decode(free,f["q"],f["dq"],h)
                            qp,_,ap=chunk.sample(times)
                            residual=(qp-f["q"])@f["jacobian"].T+times[:,None]*f["drift"]-f["target"]
                            stats.append({"predicted_flex_prefix_max_mm":float(1000*np.max(np.linalg.norm(residual[times<=.1+1e-8,:3],axis=1))),
                                "predicted_flex_window_max_mm":float(1000*np.max(np.linalg.norm(residual[:,:3],axis=1))),
                                "initial_acceleration_ratio":float(np.max(np.abs(ap[0])/f["acceleration_limits"])),
                                "acceleration_jump_rms_ratio":float(np.sqrt(np.mean(((ap[0]-f["previous_acceleration"])/f["acceleration_limits"])**2)))})
                        window["stages"][stage]=stats
                    selection=json.loads(planning.with_suffix(".json").read_text(encoding="utf-8"))
                    window["selection"]=selection
                    qpred,_,_=SplineChunk(f["selected_controls"],h).sample(times)
                    residual=(qpred-f["q"])@f["jacobian"].T+times[:,None]*f["drift"]-f["target"]
                    window["selected_predicted_prefix_max_mm"]=float(1000*np.max(np.linalg.norm(residual[times<=.1+1e-8,:3],axis=1)))
                    row["windows"].append(window)
        rows.append(row)
    write_json(folder/"startup_diagnostics.json",{"runs":rows,
        "prediction_note":"Local generalized-Jacobian predictions; actual execution error is separately measured",
        "raw_note":"before_late_guidance is the clean estimate at the beginning of guidance, not a separate unguided final sample"})
    print(json.dumps([{"task":r["task"],"seed":r["sampling_seed"],"peak_mm":r["flex_peak_first_second_mm"],
        "peak_time":r["flex_peak_time_s"],"first_windows":[{"t":w["time"],"index":w["selection"]["selected_index"],
        "selected_pred_mm":w["selected_predicted_prefix_max_mm"],"projected_candidates":w["stages"].get("after_projection")} for w in r["windows"][:3]]} for r in rows[:3]],indent=2))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("folder",type=Path)
    a=p.parse_args()
    analyze(a.folder)
