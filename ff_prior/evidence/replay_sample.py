"""Replay the compact real control sample, comparing retained native checkpoints.

The full original 5,251-state verification is in the audit report. This extra
portable check compares 101 saved checkpoints and does not upgrade other data.
"""
from pathlib import Path
import time
import mujoco
import numpy as np
from ff_prior.pipeline import audit, config_from
from ff_prior.storage import read_json, atomic_json, digest
from ff_prior.initial_state import InitialState


def main():
    wall,cpu=time.perf_counter(),time.process_time()
    root=Path('ff_prior/evidence/smoke_002')
    _,robot,_,_=audit(config_from('configs/ff_prior/v7_1.proposed.yaml'))
    metadata=read_json(root/'portable_sample.json')
    with np.load(root/'portable_sample.npz') as a:
        initial=InitialState(a['initial_state'].copy(),metadata['model_hash'])
        assert initial.state_id==metadata['initial_state_id']
        ctrl=a['ctrl'].copy()
        assert digest(ctrl)==metadata['ctrl_hash']
        data=initial.reset(robot)
        steps=len(ctrl)
        checkpoints={k:a[k].copy() for k in ('qpos','qvel','qacc_warmstart','act','time')}
        states={int(k):j for j,k in enumerate(a['checkpoint_indices'])}
        maximum={k:0. for k in ('qpos','qvel','qacc_warmstart','act','time')}
        for i in range(steps+1):
            # Recorded checkpoints are after the tracker's forward evaluation.
            # Recompute derived/cache values without assigning physical state.
            robot.forward(data)
            if i in states:
                for k in maximum:
                    error=np.abs(np.asarray(getattr(data,k))-checkpoints[k][states[i]])
                    maximum[k]=max(maximum[k],float(error.max()) if error.size else 0.)
            if i<steps:
                data.ctrl[:]=ctrl[i]
                mujoco.mj_step(robot.model,data)
    passed=max(maximum.values())<=1e-10
    result={'status':'PASS' if passed else 'FAIL','scope':'one portable control replay, 101 native checkpoints',
            'physics_steps':steps,'maximum_errors':maximum,
            'wall_s':time.perf_counter()-wall,'cpu_s':time.process_time()-cpu}
    atomic_json(root/'portable_replay_result.json',result)
    print(result)
    assert passed


if __name__=='__main__':
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=1):
        main()
