import numpy as np
from ffgd.robot import Robot
from ffgd.contracts import Candidate
import ffgd.execution as execution


def test_budget_abort_saves_complete_input_ledger_and_can_replay(tmp_path,monkeypatch):
    r=Robot();d=r.new_data();snapshot=r.snapshot(d);q=r.Bplus@d.qpos[r.qpos_ids]
    candidate=Candidate(np.tile(q,(24,1)),.2,0);calls=[]
    def budget():
        calls.append(1)
        if len(calls)==2:raise RuntimeError('RESOURCE_WALL_BUDGET_EXHAUSTED')
    monkeypatch.setattr(execution,'check_budget',budget)
    bundle,summary=execution.track(r,snapshot,candidate,tmp_path,dict(physical_geometry_clearance_m=0.))
    assert summary['status']=='RESOURCE_BUDGET_ABORT' and summary['actual_steps']==64
    assert len(bundle['states'])==len(bundle['ctrl'])+1
    assert (tmp_path/'execution_partial.npz').exists() and (tmp_path/'execution.npz').exists()
    assert execution.replay(r,bundle,snapshot)['max_state_abs']<=1e-10
