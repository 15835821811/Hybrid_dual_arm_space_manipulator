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


def recorded_input_sweep(path,config):
    """Isolate integrator error with identical zero-order-held input histories.

    Diagnostic only: no replanning, controller recomputation, state projection,
    or claim that an open-loop replay is a successful closed-loop execution.
    """
    import mujoco
    from .problem_io import load_problem
    from .contracts import Snapshot
    from .geometry import Geometry
    from .propagation import native_momentum,native_subtree_momentum
    from .orchestration import atomic_json
    torch.set_num_threads(1)
    root=Path(config['execution_directory'])
    provenance=json.loads((root/'provenance.json').read_text(encoding='utf-8'))
    original,initial,task,_=load_problem(Path(provenance['planning_directory'])/'problem.json')
    with np.load(root/'execution.npz') as z:source={k:z[k] for k in z.files}
    np.testing.assert_array_equal(source['initial_state'],initial.state)
    if len(source['states'])!=len(source['ctrl'])+1:raise ValueError('INCOMPLETE_INPUT_LEDGER')
    if np.any(source['qfrc_applied']) or np.any(source['xfrc_applied']):raise ValueError('EXTERNAL_WRENCH_LEDGER_REQUIRED')
    dt=float(original.model.opt.timestep);rows=[]
    for factor in (1,2,4):
        check_budget();out=path/f'substeps_{factor}';out.mkdir()
        r=Robot(physics_dt=dt/factor,scene=original.scene)
        # Prove that timestep is the sole compiled-model difference BEFORE binding.
        r.model.opt.timestep=dt
        if r.compiled_hash()!=original.model_hash:raise ValueError('MODEL_DIFFERENCE_BEYOND_TIMESTEP')
        r.model.opt.timestep=dt/factor
        snap=Snapshot(r.model_hash,initial.mask,initial.state.copy())
        np.testing.assert_array_equal(snap.state,initial.state)
        d=r.restore(snap);H=native_momentum(r,d);geometry=Geometry(r)
        inputs={k:np.repeat(source[k],factor,axis=0) for k in ('ctrl','qfrc_applied','xfrc_applied')}
        states=[];residual=[];clearance=[];status='FINISHED';source_error=0.
        for i in range(len(inputs['ctrl'])+1):
            if i%64==0:check_budget()
            mujoco.mj_forward(r.model,d)
            state=r.snapshot(d).state.copy();states.append(state)
            residual.append(native_subtree_momentum(r,d,forward=False)-H)
            scan=geometry.native_scan(d);clearance.append(scan['minimum'])
            if factor==1:source_error=max(source_error,float(np.max(abs(state-source['states'][i]))))
            low=d.qpos[r.qpos_ids];q=r.Bplus@low;limited=r.model.jnt_limited[r.joints].astype(bool);bounds=r.model.jnt_range[r.joints]
            if scan['minimum']<provenance['profile']['physical_geometry_clearance_m']:status='SAFETY_ABORT_GEOMETRY'
            if np.any(low[limited]<bounds[limited,0]) or np.any(low[limited]>bounds[limited,1]) or np.any(q<r.lower) or np.any(q>r.upper):status='SAFETY_ABORT_JOINT'
            if np.any(d.warning.number):status='SIMULATOR_WARNING'
            if status!='FINISHED' or i==len(inputs['ctrl']):break
            # Same observation schedule as track/replay; never recompute controls.
            mujoco.mj_forward(r.model,d);mujoco.mj_forward(r.model,d)
            for k,v in inputs.items():getattr(d,k)[:]=v[i]
            mujoco.mj_step(r.model,d)
        n=len(states)-1
        bundle=dict(states=np.asarray(states),initial_state=snap.state,mask=snap.mask,
                    momentum_residual=np.asarray(residual),clearance=np.asarray(clearance),
                    **{k:v[:n] for k,v in inputs.items()})
        np.savez_compressed(out/'execution.npz',**bundle)
        rr=replay(r,bundle,snap)
        peaks=[float(np.max(np.linalg.norm(bundle['momentum_residual'][:,j:j+3],axis=1))) for j in (0,3)]
        row=dict(factor=factor,physics_dt=dt/factor,held_control_period=dt,status=status,native_steps=n,
                 linear_momentum_residual=peaks[0],angular_momentum_residual=peaks[1],
                 min_clearance=min(clearance),replay=rr,model_hash=r.model_hash,
                 initial_state_bytes_unchanged=True,only_timestep_changed=True,
                 source_state_error=source_error if factor==1 else None)
        atomic_json(out/'result.json',row);rows.append(row);print(json.dumps(row),flush=True)
    convergence={k:all(b[k]<=a[k]+1e-12 for a,b in zip(rows,rows[1:])) for k in ('linear_momentum_residual','angular_momentum_residual')}
    return dict(passed=all(x['status']=='FINISHED' and x['replay']['max_state_abs']<=provenance['profile']['replay_state_atol'] for x in rows)
                and rows[0]['source_state_error']<=provenance['profile']['replay_state_atol'] and all(convergence.values()),
                rows=rows,convergence=convergence,source=str(root),candidate_hash=provenance['candidate_hash'],
                source_input_sha256=hashlib.sha256((root/'execution.npz').read_bytes()).hexdigest(),
                source_reports_modified=False,scope='Fixed recorded inputs; numerical convergence only, not closed-loop execution acceptance')


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
