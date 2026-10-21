"""Saved algebra only; exact reuse and new execution are counted separately."""
import zipfile,hashlib
from types import SimpleNamespace
import numpy as np
from ff_prior.storage import read_json,digest,file_digest
from ff_prior_pilot_train.storage import atomic_json
from ff_prior.spline import Reference
from ff_prior.proposal_generator import quick_gates
from ff_prior.coupled_predictor import pose_errors
from ff_prior_delivery_guidance.audit_results import check_candidate_validation
from ff_prior_complex.tools.audit_completed import audit_physics
from .inputs import identity
from .protocol import ROOT,PACKAGE,P
from .representation import fixed_basis,project,prepare

def algebra_robot(limits):
    a=lambda k:np.asarray(limits[k])
    return SimpleNamespace(lower=a('lower'),upper=a('upper'),B=a('B'),joints=np.arange(len(limits['limited'])),
        model=SimpleNamespace(jnt_limited=a('limited'),jnt_range=a('ranges')),
        spec=SimpleNamespace(planner_zero=a('q0'),planner_velocity_limits=a('velocity'),planner_acceleration_limits=a('acceleration')))

def verify(out,ident):
    errors=[];fresh=historical=duplicates=executions=physics=0
    try:
        assert identity()==ident
        with zipfile.ZipFile(out/'executed_source.zip') as z:
            assert all(hashlib.sha256(z.read(p)).hexdigest()==h for p,h in ident['source_hashes'].items())
        task=ident['selection']['entry']['task'];assert read_json(out/'frozen_task.json')==task
        init=read_json(out/'initial_state.json')
        with np.load(out/'initial_state.npz') as a:
            assert digest({'values':a['values'],'model_hash':init['model_hash'],'state_mask':init['state_mask']})==init['state_id']==task['contract']['initial_state_id']
        robot=algebra_robot(read_json(out/'physical_limits.json'));c=task['condition']
        def quick(z):return all(quick_gates(robot,Reference(z,c['initial_features'],c['duration'])).values())
        with np.load(out/'representation_inputs.npz') as a:teacher,base,original=[a[k].copy() for k in ('teacher','base','original_basis')]
        assert np.array_equal(teacher,ident['selection']['teacher']['Z'])
        manifest=read_json(out/'candidate_manifest.json');rows=manifest['candidates']
        assert manifest['before_model_evaluation'] and manifest['digest']==digest(rows)
        assert [(r['rank'],r['mode']) for r in rows]==[(k,m) for k in P['ranks'] for m in P['modes']]
        owners={};expected=[]
        for row in rows:
            d=out/'candidates'/row['id'];b=fixed_basis(original,row['rank']);raw=project(teacher,base,b)
            value,details=prepare(raw,base,b,c['initial_features'],robot.lower,robot.upper,quick,row['mode'])
            with np.load(d/'representation.npz') as a:
                assert np.array_equal(a['basis'],b) and np.array_equal(a['raw'],raw) and np.array_equal(a['output'],value)
                for k,v in details.items():
                    if isinstance(v,np.ndarray):assert np.array_equal(v,a[k])
            assert np.array_equal(value,row['Z'])
            ref=Reference(value,c['initial_features'],c['duration']);assert ref.reference_hash==row['reference_hash']
            assert quick_gates(robot,ref)==row['quick_gates']
            r=read_json(d/'validation.json');check_candidate_validation(r,task)
            assert r['reference_hash']==ref.reference_hash and np.array_equal(r['Z'],value)
            origin=read_json(d/'validation_origin.json')
            if ref.reference_hash in owners:
                duplicates+=1;assert origin['kind']=='same_run_exact_duplicate' and origin['owner']==owners[ref.reference_hash]
                assert r==read_json(out/'candidates'/origin['owner']/'validation.json')
            else:
                owners[ref.reference_hash]=row['id']
                if r['eligible']:expected.append(row['id'])
                if origin['kind']=='historical_exact_identity':
                    historical+=1;assert r==read_json(ROOT/origin['source']/'validation.json')
                else:
                    assert origin['kind']=='new_model_validation';fresh+=1
                    if r['validation_grids']:
                        with np.load(d/'prediction.npz') as a:
                            pos,rot=pose_errors(a['ee'][-1],c['goals'])
                            assert np.allclose(pos,r['validation']['frozen_goal_position_error_m'],atol=1e-12,rtol=0)
                            assert np.allclose(rot,r['validation']['frozen_goal_rotation_error_rad'],atol=1e-12,rtol=0)
                        for grid in r['validation_grids']:
                            n=grid['intervals'];scans=read_json(d/f'grid_{n}_geometry_scans.json')
                            with np.load(d/f'grid_{n}_geometry_scans.npz') as a:
                                assert np.array_equal(a['minima'],[x['minimum_clearance_m'] for x in scans])
                                assert np.isclose(min(a['minima']),grid['geometry']['minimum_clearance_m'],atol=1e-12)
            exorigin=read_json(d/'execution_origin.json');assert exorigin['owner']==owners[ref.reference_hash]
        selection=read_json(out/'execution_selection.json');assert selection['ids']==expected and selection['before_execution']
        for name in owners.values():
            d=out/'candidates'/name;ex=d/'execution';r=read_json(d/'validation.json');o=read_json(ex/'outcome.json')
            if name not in expected:assert o['status']=='NO_CANDIDATE';continue
            result,steps=audit_physics(ex,task,r['reference_hash']);assert o['status']==result['level']
            executions+=1;physics+=steps
            assert read_json(ex/'initial_state.json')==init
            with np.load(out/'initial_state.npz') as a,np.load(ex/'initial_state.npz') as b:assert np.array_equal(a['values'],b['values'])
            with np.load(ex/'native_trace.npz') as a,np.load(ex/'replay_trace.npz') as b:
                assert np.array_equal(a['ctrl'],b['ctrl'])
                for key in ('qpos','qvel','act','time','qacc_warmstart'):
                    delta=float(np.max(abs(a[key]-b[key]),initial=0))
                    assert delta==result['replay']['maximum_errors'][key]
            scans=read_json(ex/'geometry_scans.json')
            with np.load(ex/'geometry_scans.npz') as a:
                assert len(a['qpos'])==result['checked_states']==len(scans)
                assert np.array_equal(a['minima'],[x['minimum_clearance_m'] for x in scans])
                assert min(a['minima'])==result['minimum_clearance_m']
        ledger=read_json(out/'ledger.json');budget=read_json(out/'budget.json')
        assert ledger['inflight'] is None and not ledger.get('unresolved_call_accounting')
        assert sum(v for k,v in ledger['calls'].items() if k.endswith('.physics_steps'))==physics
        assert sum(v for k,v in ledger['calls'].items() if k.endswith('.coupled'))==budget['actual_coupled']<=P['coupled_cap']
        assert budget['integration_intervals']<=P['integration_cap'] and executions<=P['execution_limit']
    except Exception as e:
        import traceback
        errors.append({'error':repr(e),'traceback':traceback.format_exc()})
    return {'status':'PASS' if not errors else 'FAIL','errors':errors,'fresh_model_checks':fresh,
        'historical_model_reuses':historical,'same_run_duplicates':duplicates,'fresh_executions':executions,
        'native_plus_replay_steps':physics,'scope':'saved arrays and frozen code, no new physics'}

def report(out,audit):
    rows=[]
    for c in read_json(out/'candidate_manifest.json')['candidates']:
        d=out/'candidates'/c['id'];r=read_json(d/'validation.json')
        rows.append({'id':c['id'],'rank':c['rank'],'dimensions':c['dimensions'],'mode':c['mode'],
            'eligible':r['eligible'],'validation':r['validation'],'origin':read_json(d/'validation_origin.json'),
            'execution':read_json(d/'execution_origin.json')})
    summary={'status':'FINISHED' if audit['status']=='PASS' else 'FINISHED_WITH_BLOCKED_ITEMS','raw_audit':audit,
        'rows':rows,'budget':read_json(out/'budget.json'),'TRAIN_added':0,'training':False,'TEST':False,'independent_review':'PENDING'}
    atomic_json(out/'summary.json',summary)
    (out/'REPORT.md').write_text('# 有界表示保真后台结果\n\n'+summary['status']+'；原始审计'+audit['status']+'。独立审查及中文分析待收尾。\n\n'+
        '\n'.join(f"{r['id']}: model={r['eligible']} execution={r['execution']}" for r in rows)+'\n',encoding='utf-8')
    return summary
