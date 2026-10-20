"""Read-only correction of unlimited-joint placeholders, retaining raw reports."""
import numpy as np
from ff_prior.robot_adapter import RobotAdapter
from ff_prior.storage import read_json,atomic_json,file_digest
from .experiment import OUT


def main():
    robot=RobotAdapter()
    summary=read_json(OUT/"summary.json")
    original_hash=file_digest(OUT/"summary.json")
    limited=robot.model.jnt_limited[robot.joints].astype(bool)
    ranges=robot.model.jnt_range[robot.joints][limited]
    evidence={}
    for name,result in summary["results"].items():
        path=OUT/name/"native_trace.npz"
        assert file_digest(path)==result["native_trace_sha256"]
        with np.load(path) as a:
            q=a["qpos"][:,robot.qpos_ids][:,limited]
            checks=dict(actual_planner_domain=bool(np.all(a["q17"]>=robot.lower) and np.all(a["q17"]<=robot.upper)),
                reference_planner_domain=bool(np.all(a["qref"]>=robot.lower) and np.all(a["qref"]<=robot.upper)),
                native_limited_joints=bool(np.all(q>=ranges[:,0]) and np.all(q<=ranges[:,1])))
            margins=dict(actual_planner_lower=float(np.min(a["q17"]-robot.lower)),
                actual_planner_upper=float(np.min(robot.upper-a["q17"])),
                native_lower=float(np.min(q-ranges[:,0])),native_upper=float(np.min(ranges[:,1]-q)))
        evidence[name]=dict(checks=checks,margins_rad=margins,trace_sha256=file_digest(path))
        result["gates"]["joint_limits"]=all(checks.values())
        result["status"]="EXECUTION_VERIFIED" if all(result["gates"].values()) else "EXECUTION_FAILED"
    summary["read_only_correction"]=dict(original_summary_sha256=original_hash,
        reason="Unrestricted MuJoCo joints have range [0,0] placeholders. Apply original ff_prior/physical_audit.py jnt_limited mask; planner domain still checked for all 17 coordinates.",
        limited_count=int(limited.sum()),unlimited_count=int((~limited).sum()),
        raw_reports_unchanged=True,new_physics_steps=0,evidence=evidence)
    atomic_json(OUT/"audited_summary.json",summary)
    assert file_digest(OUT/"summary.json")==original_hash
    print({k:v["status"] for k,v in summary["results"].items()})


if __name__=="__main__":main()
