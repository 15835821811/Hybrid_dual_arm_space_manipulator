"""Predeclared dt/dt2/dt4 native momentum and replay calibration."""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from .robot import Robot
from .trajectory import Boundary,BoundaryCodec
from .contracts import Candidate
from .execution import track,replay
from .orchestration import check_budget


def run(path):
    torch.set_num_threads(1)
    profile_path=Path(__file__).resolve().parents[1]/'configs/ffgd/numerical_profile_v1.json'
    profile=json.loads(profile_path.read_text());rows=[]
    for dt in (.002,.001,.0005):
        r=Robot(physics_dt=dt)
        for case in range(3):
            check_budget();out=path/f'dt_{dt}_case_{case}';out.mkdir()
            d=r.new_data();q0=r.spec.planner_zero;T=2.;v0=np.zeros(17)
            if case==2:
                v0=np.linspace(-.002,.002,17);d.qvel[r.dof_ids]=r.B@v0
                d.qvel[r.base_dofs]=np.array([.001,-.001,.0005,.0015,-.0007,.0003])
            snapshot=r.snapshot(d)
            bc=[Boundary(0,j,tuple(range(17)),tuple(q0) if j==0 else tuple(v0) if j==1 else (0.,)*17) for j in (0,1,2)]
            bc += [Boundary(T,j,tuple(range(17)),(0.,)*17) for j in (1,2)]
            codec=BoundaryCodec(8,17,T,bc)
            C=np.tile(q0,(8,1));amplitude=0 if case==0 else .02 if case==1 else .04
            delta=amplitude*np.sin(np.arange(17)+.3)
            C[3:]+=np.linspace(.2,1,5)[:,None]*delta
            C=codec.decode((C.ravel()-codec.offset)@codec.N)
            candidate=Candidate(C,T,512+case)
            bundle,report=track(r,snapshot,candidate,out,profile)
            rr=replay(r,bundle,snapshot)
            (out/'replay.json').write_text(json.dumps(rr,indent=2),encoding='utf-8')
            row=dict(case=case,dt=dt,execution=report,replay=rr,passed=report['status']=='FINISHED' and report['raw_torque_ratio']<=1
                     and report['linear_momentum_residual']<=profile['momentum_linear_kg_m_s']
                     and report['angular_momentum_residual']<=profile['momentum_angular_kg_m2_s'] and rr['max_state_abs']<=profile['replay_state_atol'])
            rows.append(row);(out/'result.json').write_text(json.dumps(row,indent=2),encoding='utf-8')
            print(json.dumps(row),flush=True)
        del r
    ratios=[]
    for case in range(3):
        values=[r for r in rows if r['case']==case]
        for key in ('linear_momentum_residual','angular_momentum_residual'):
            errors=[r['execution'][key] for r in values]
            ratios.append(dict(case=case,metric=key,errors=errors,nonincreasing=all(b<=a+1e-12 for a,b in zip(errors,errors[1:]))))
    return dict(passed=all(r['passed'] for r in rows) and all(r['nonincreasing'] for r in ratios),rows=rows,convergence=ratios,
                profile_hash=hashlib.sha256(profile_path.read_bytes()).hexdigest(),scope='stationary, two amplitudes, nonzero initial momentum; no contact/external wrench; 2 s calibration only')
