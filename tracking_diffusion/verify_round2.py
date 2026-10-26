"""Check split integrity, private FK and sampling acceleration against eager execution."""
import os
for _k in ("OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","OMP_NUM_THREADS"):
    os.environ[_k]="1"
from pathlib import Path
import numpy as np
import torch
from .candidate_fk import residuals
from .environment import Task,TrackingEnv
from .paper_model import PaperPolicy
from .pipeline import write_json
from .round2_data import BASE,freeze,read


def main():
    manifest=freeze(BASE)
    seeds={s:{e["task"]["seed"] for e in manifest["tasks"] if e["split"]==s} for s in ("train","validation","test","diagnostic")}
    assert all(not seeds[a]&seeds[b] for a in seeds for b in seeds if a!=b)
    coverage={}
    for size in (150,500):
        path=BASE/f"data_{size}/dataset.npz"
        if not path.exists(): continue
        with np.load(path) as data:
            t=set(data["seed"][data["split"]==0]);v=set(data["seed"][data["split"]==1])
            assert t<=seeds["train"] and v<=seeds["validation"] and not (t|v)&(seeds["test"]|seeds["diagnostic"])
            assert data["fit_error"].max()<=.002
            for seed in t|v:
                times=data["time"][data["seed"]==seed]
                assert len(times)==270 and np.allclose(times,np.arange(270)*.1)
            coverage[str(size)]={"train_tasks":len(t),"validation_tasks":len(v),"all_270_windows_present":True}
    env=TrackingEnv(Task(20261012,27.,2,"local10mm-v1",True))
    before=(env.data.qpos.copy(),env.data.qvel.copy(),env.data.ctrl.copy())
    q,_=env.state()
    g,d=env.generalized_map()
    r=residuals(env,q[None],np.array([0.]),g,d)
    assert np.allclose(r,-env.task_error()[None])
    assert all(np.array_equal(a,b) for a,b in zip(before,(env.data.qpos,env.data.qvel,env.data.ctrl)))
    checkpoint=Path("tracking_output/paper100_v1/train_a01/model.pt")
    policies=[PaperPolicy(checkpoint,"cuda",211,5,60,True,cuda_graph=graph) for graph in (False,True)]
    outputs=[]
    for p in policies:
        p.bind_environment(env)
        outputs.append(p.predict(env.condition(),.5,0.))
    graph_delta=float(np.max(np.abs(outputs[0]-outputs[1])))
    assert graph_delta<1e-7
    for p in policies:
        p.generator.manual_seed(211)
    policies[0].telemetry_stride=1
    policies[1].telemetry_stride=5
    counts=[[],[]]
    outputs=[p.predict(env.condition(),.5,0.,lambda k,n,x:counts[i].append(k)) for i,p in enumerate(policies)]
    assert counts[0]==list(range(61)) and counts[1]==list(range(0,61,5))
    assert np.max(np.abs(outputs[0]-outputs[1]))<1e-7
    assert all(np.array_equal(a,b) for a,b in zip(before,(env.data.qpos,env.data.qvel,env.data.ctrl)))
    report={"passed":True,"split_coverage":coverage,"new_test_tasks":len(seeds["test"]),
        "private_fk_origin_matches_native_pose":True,"physical_state_unchanged":True,
        "graph_vs_eager_max_control_delta":graph_delta,"callback_steps_full":counts[0],"callback_steps_light":counts[1]}
    write_json(BASE/"round2_checks.json",report)
    print(report)


if __name__=="__main__": main()
