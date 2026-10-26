"""Pure software, compiled-model identity and budget checks; no neural or physics run."""
import ast,time,hashlib,tempfile
from pathlib import Path
import numpy as np
import scipy.sparse as sp
import mujoco,torch
from scipy.spatial.transform import Rotation
from ff_prior.storage import read_json,file_digest,digest
from ff_prior_pilot_train.storage import atomic_json
from ff_prior.initial_state import STATE
from .models import load_static
from ff_prior_temporal_learning.models import SequenceNetwork
from .protocol import ROOT,PACKAGE,P
from .inputs import runtime,sources,versions
from .representation import Representation,solve_qp
from .predictor import world_momentum,generalized_momentum
from .budget import Budget,Limit


def main():
    tests=[]
    def check(name,passed,**evidence):
        tests.append(dict(name=name,passed=bool(passed),**evidence))
        if not passed:raise AssertionError(name)
    robot,initial,entry=runtime();m=robot.model
    from ff_prior.geometry import CollisionBackend
    check('whole_machine_pair_policy',len(CollisionBackend(robot).pairs)==3406)
    check('original_numeric_initial836',file_digest(ROOT/P['initial'])==P['initial_sha256'] and len(initial.values)==836)
    check('RK4_2ms',m.opt.integrator==mujoco.mjtIntegrator.mjINT_RK4 and m.opt.timestep==.002)
    arrays={k:hashlib.sha256(memoryview(np.ascontiguousarray(getattr(m,k)))).hexdigest() for k in dir(m)
        if isinstance(getattr(m,k,None),np.ndarray)}
    new_hash=initial.model_hash
    del robot._identity;m.opt.integrator=mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    baseline=robot.identity()['scene_model_hash']
    layout=read_json(ROOT/'ff_prior_balanced_scene/evidence/layout.json')
    check('old_compiled_model_identity',baseline==layout['new_scene_model_hash'])
    del robot._identity;m.opt.integrator=mujoco.mjtIntegrator.mjINT_RK4
    after={k:hashlib.sha256(memoryview(np.ascontiguousarray(getattr(m,k)))).hexdigest() for k in arrays}
    check('all_compiled_arrays_unchanged',arrays==after,array_count=len(arrays))
    check('new_integrator_identity',robot.identity()['scene_model_hash']==new_hash and baseline!=new_hash)
    data=mujoco.MjData(m);mujoco.mj_setState(m,data,initial.values,STATE)
    q0=robot.Bplus@data.qpos[robot.qpos_ids]
    for phase in ('A','B','C'):
        v=np.zeros(17) if phase=='A' else np.full(17,1e-5);acc=np.zeros(17)
        rep=Representation(robot,phase,q0,v,acc)
        control=np.tile(q0,(19 if phase=='A' else 273,1))
        x,report=rep.project(control,time.time()+60);ref=rep.ref(x)
        check(phase+'_reference_qp',rep.valid(x) and report['feasible'],variables=rep.dim,rows=rep.A.shape[0])
        check(phase+'_boundary',np.max(abs(ref.evaluate(0)-q0))<1e-10 and np.max(abs(ref.evaluate(0,1)-v))<1e-10)
        if phase!='A':check(phase+'_acceleration_memory',np.max(abs(ref.evaluate(0,2)-acc))<1e-10)
        check(phase+'_expected_dimensions',rep.dim==(323 if phase=='A' else 4590))
    x,qp=solve_qp(sp.eye(2),np.array([-2.,-3.]),sp.eye(2),np.zeros(2),np.ones(2),time.time()+10)
    check('QP_known_analytic_optimum',np.max(abs(x-1))<1e-8)
    R=Rotation.from_rotvec([.2,-.1,.4]).as_matrix();pos=np.array([2.,-1.,3.]);g=np.arange(6)*.1
    p,h=world_momentum(g,pos,R);back=generalized_momentum(p,h,pos,R)
    check('world_generalized_momentum_roundtrip',np.max(abs(back-g))<1e-14)
    moved=generalized_momentum(p,h,pos+[.1,.2,-.3],Rotation.from_rotvec([.1,.2,.3]).as_matrix())
    pp,hh=world_momentum(moved,pos+[.1,.2,-.3],Rotation.from_rotvec([.1,.2,.3]).as_matrix())
    check('inherited_world_momentum_transport',np.max(abs(pp-p))<1e-14 and np.max(abs(hh-h))<1e-14)
    static,codec=load_static(ROOT/P['static_model'])
    check('static_frozen_interface',codec.dim==102 and codec.basis.shape==(19,6) and static.calls==0)
    check('static_approved_checkpoint',file_digest(ROOT/P['static_model']/'model.pt')=='7915781e786b2ef8783ea0788f90d7a691312c97c033ee3cf88f169709099445')
    check('temporal_approved_checkpoint',file_digest(ROOT/P['temporal_model']/'model.pt')=='e3c4a188149b2d8c596ba432f31748ca39157810888775353a146b18146625d0')
    payload=torch.load(ROOT/P['temporal_model']/'model.pt',map_location='cpu',weights_only=False)
    temporal=SequenceNetwork(payload['condition_dim']);temporal.load_state_dict(payload['weights'])
    check('temporal_frozen_interface',payload['condition_dim']==1014 and tuple(temporal.position.shape)==(1,9,271) and temporal.calls==0)
    rng=np.random.default_rng(20261026);xt=rng.normal(size=102);clean=rng.normal(size=102);a=.37
    eps=(xt-np.sqrt(a)*clean)/np.sqrt(1-a)
    check('guided_DDIM_exact_recomposition',abs(np.sqrt(a)*clean+np.sqrt(1-a)*eps-xt).max()<1e-14)
    budget=Budget(PACKAGE/'evidence/software_budget',started=time.time())
    check('recorders_epoch_deadline',budget.physics_deadline==budget.deadline and budget.physics_deadline>time.time())
    budget.used['native']=P['native_cap']
    refused=False
    try:budget.reserve('native')
    except Limit:refused=True
    check('native_limit_refused_before_increment',refused and budget.used['native']==P['native_cap'] and budget.completed['native']==0)
    budget.used['coupled']=P['coupled_cap'];before=dict(budget.used);refused=False
    try:budget.propagation(512)
    except Limit:refused=True
    check('coupled_interval_atomic_reservation',refused and budget.used==before)
    check('zero_production_calls',all(robot.calls[k]==0 for k in ('physics_steps','coupled','distance','forward','fk','mass_matrix')))
    # Copy-gate equivalence is verified by removing only the explicit adapter lines.
    for new,old in [('dynamic_evaluation.py','ff_prior_balanced_brake/evaluation.py'),('satellite_evaluation.py','ff_prior_satellite_origin/evaluation.py')]:
        text=(PACKAGE/new).read_text(encoding='utf-8')
        package=old.split('/')[0]
        reverted=text.replace('replay=r._guided_replay_data\n    assert np.array_equal(integration(r,replay),env.initial.values)','replay=env.initial.reset(r)')
        reverted=reverted.replace('from '+package+'.protocol import P','from .protocol import P').replace('from '+package+'.ledger import','from .ledger import')
        check(new+'_original_gates_exact',reverted==(ROOT/old).read_text(encoding='utf-8-sig'))
    text=(PACKAGE/'static_audit.py').read_text(encoding='utf-8').replace('from .static_tracker import','from .tracker import').replace('from ff_prior_balanced_restore.native_budget import','from .native_budget import')
    text=text.replace('    data = robot._guided_replay_data\n    assert np.array_equal(integration_state(robot,data),initial.values)\n    recorded=','    data = initial.reset(robot)\n    recorded=')
    check('static_original_gates_exact',text==(ROOT/'ff_prior_balanced_restore/physical_audit.py').read_text(encoding='utf-8-sig'))
    imports=['sampler','qualification','execution','experiment','job','media']
    for name in imports:__import__('ff_prior_guided_chain.'+name)
    check('all_execution_modules_import',True)
    from inspect import signature
    from .static_audit import audit_execution
    check('static_execution_interface',list(signature(audit_execution).parameters)[:6]==['frozen_sample','profile','robot','initial','reference','output'])
    # Stored Jacobian and perturbation residual pairs: both full float64, no quantization.
    jac=4*323*(12+512+256)*16+8*4590*(3276+512+256)*16
    enumeration=4*(323+4590+4590)+72+192+288+192+220+96
    check('coupled_enumeration',enumeration==39072 and enumeration<=P['coupled_cap'])
    check('native_complete_chain_and_replay',2*sum(P['phase_intervals'].values())==P['native_cap']==83500)
    check('jacobian_reservation',jac+64*1024**2<=int(2.5*1024**3),bytes=jac)
    cost=dict(within_authorized_stop_envelope=True,coupled=enumeration,coupled_cap=P['coupled_cap'],native=P['native_cap'],
        neural=48,motion_projection_QPs=48,full_machine_guidance_QPs=12,maximum_iterations_each=180,
        model_interval_cap=P['interval_cap'],jacobian_and_perturb_float64_bytes=jac,
        storage_allocations_GiB=dict(native_replay=1.5,jacobians_sampler=2.5,model=1.,geometry=2.,media=.75,source_review=.25),
        reallocation='original .5GiB headroom assigned to complete Jacobian plus perturbation records; total8GiB unchanged',
        variable_near_pair_records='complete records until before-write capacity refusal; no dropped active/omitted records',
        dense_worst_single_prediction_bytes=(16385*(m.nq+m.nv+32+7+m.ngeom*12+67+12)*8),
        geometry_guard=P['geometry_cap'],mass_kinematic_guard=P['mass_kinematic_cap'],
        completing_all_iterations_not_guaranteed=True,stop_on_record_or_time_cap=True,physical_checks=0,network_calls=0)
    atomic_json(PACKAGE/'evidence/cost.json',cost)
    atomic_json(PACKAGE/'evidence/software_tests.json',dict(status='PASS',checks=tests,count=len(tests),
        source_hashes=sources(),versions=versions(),model_hash=new_hash,original_model_hash=baseline,
        compiled_array_hashes=arrays,physical_allowed_difference=['opt.integrator'],
        synthetic_math_checks_not_physical_evidence=True,production_coupled=0,production_native=0,neural_forward=0))
    print(dict(status='PASS',checks=len(tests),coupled=0,native=0,neural=0))


if __name__=='__main__':main()
