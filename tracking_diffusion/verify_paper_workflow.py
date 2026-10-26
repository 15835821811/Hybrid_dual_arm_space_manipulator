"""Check real spline boundaries, inference gradients, data splits and state isolation."""
import os
for _name in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ[_name]="1"
import json
from pathlib import Path
import numpy as np
import torch
from .environment import Task,TrackingEnv
from .paper_model import PaperPolicy
from .guidance import ConstraintGuidance
from .pipeline import write_json
from .spline import SplineChunk


def main():
    folder=Path("tracking_output/local10mm_v1")
    manifest=json.loads((folder/"pilot_manifest.json").read_text(encoding="utf-8"))
    split={"train":[e["id"] for e in manifest["tasks"][:16]],
           "validation":[e["id"] for e in manifest["tasks"][16:18]],
           "internal_test":[e["id"] for e in manifest["tasks"][18:]]}
    write_json(folder/"paper_data_20/split_manifest.json",{
        "split":split,"task_manifest":"../pilot_manifest.json",
        "test_scope":"two development-domain internal tasks, not a large external benchmark",
        "training_selection":"IDs supplied to dataset builder before slicing and training"})
    with np.load(folder/"paper_data_20/dataset.npz") as d:
        assert len(d["time"])==4860
        assert d["free"].shape==(4860,6,17)
        assert d["condition"].shape==(4860,136)
        train_seeds=set(d["seed"][d["split"]==0])
        val_seeds=set(d["seed"][d["split"]==1])
        assert not train_seeds & val_seeds
        for entry in manifest["tasks"][18:]:
            assert entry["task"]["seed"] not in train_seeds | val_seeds
        assert d["fit_error"].max()<=.002
        for seed in set(d["seed"]):
            times=d["time"][d["seed"]==seed]
            np.testing.assert_allclose(times,np.arange(270)*.1,atol=1e-8)
    env=TrackingEnv(Task(**manifest["tasks"][0]["task"]))
    before=[env.data.qpos.copy(),env.data.qvel.copy(),env.data.ctrl.copy(),float(env.data.time)]
    policy=PaperPolicy(folder/"paper_train_a01/model.pt","cuda",73)
    policy.bind_environment(env)
    guide=ConstraintGuidance(env,.5,policy.xm,policy.xs,"cuda")
    z=np.random.default_rng(7).normal(size=(1,102))*.2
    _,gradient=guide.energy_gradient(z)
    errors=[]
    for index in (0,9,22,50,77,101):
        delta=np.zeros_like(z);delta[0,index]=1e-5
        numeric=(guide.energy_gradient(z+delta)[0][0]-guide.energy_gradient(z-delta)[0][0])/2e-5
        errors.append(abs(numeric-gradient[0,index])/max(1.,abs(numeric)))
    assert max(errors)<1e-6
    callbacks=[]
    free=policy.predict(env.condition(),horizon=.5,time=0.,
        on_step=lambda step,total,value:callbacks.append(step))
    assert callbacks==list(range(61))
    assert len(policy.last_guidance)==18
    assert np.isfinite(free).all()
    q,dq=env.state()
    bounds=[]
    for candidate in free:
        chunk=SplineChunk.decode(candidate,q,dq,.5)
        qq,vv,_=chunk.sample(0.)
        np.testing.assert_allclose(qq,q,atol=1e-12)
        np.testing.assert_allclose(vv,dq,atol=1e-12)
        bounds.append(chunk.bounds_ok(env.spec))
    assert any(bounds)
    for actual,expected in zip([env.data.qpos,env.data.qvel,env.data.ctrl,float(env.data.time)],before):
        np.testing.assert_array_equal(actual,expected)
    report={"passed":True,"train_tasks":16,"validation_tasks":2,"internal_test_tasks":2,
            "train_windows":4320,"validation_windows":540,"all_deployment_times_present":True,
            "guidance_gradient_relative_errors":errors,"callback_steps":len(callbacks),
            "late_guidance_steps":len(policy.last_guidance),"admitted_initial_candidates":sum(bounds),
            "physical_state_unchanged_by_generation":True,"guidance":policy.last_guidance}
    write_json(folder/"paper_workflow_checks.json",report)
    print(json.dumps({k:v for k,v in report.items() if k!="guidance"}),flush=True)


if __name__=="__main__":
    main()
