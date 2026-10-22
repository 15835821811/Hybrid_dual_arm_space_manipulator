import time
import subprocess
import zipfile
import numpy as np
from ff_prior.storage import read_json, digest, RunLock
from ff_prior_pilot_train.storage import atomic_json, atomic_npz, Ledger
from ff_prior.pipeline import audit, acceptance
from ff_prior.contracts import assert_frozen, assert_task_compatible
from ff_prior_origin_learn_bounded.runtime import TaskOriginSceneCache
from ff_prior.spline import Reference
from ff_prior.coupled_predictor import pose_errors
from ff_prior_fixed.correction import pose_vector
from ff_prior_static_goal_seed.physical_audit import audit_execution
from .protocol import ROOT, SOURCE, BASE, PACKAGE, OUT, P
from .inputs import identity
from .budget import Budget, SliceBudget, BudgetExhausted
from ff_prior_static_fullchain_v2.solver import Evaluator, optimize, metric, motion
from ff_prior_static_goal_seed.goal_seed import recover
from ff_prior_static_goal_seed.validation import validate
from ff_prior_static_goal_seed.proposals import load_bundle, public_setup, make, prepare
from .tree import search,fit323
from .m0 import verify as verify_propagation
from ff_prior.proposal_generator import quick_gates


def assert_resume_safe(out):
    if not (out/'ledger.json').exists():return
    ledger=read_json(out/'ledger.json')
    if ledger['inflight'] is not None or ledger.get('unresolved_call_accounting'):
        raise ValueError('UNKNOWN_WORK_NOT_REPLAYABLE')
    saved=read_json(out/'budget.json')
    actual=sum(v for k,v in ledger['calls'].items() if k.endswith('.coupled'))
    if actual!=saved['actual_coupled']:raise ValueError('UNRESOLVED_BUDGET_CHECKPOINT')


def screen(directory, z, ev, n, geometry):
    pred=ev.predict(z,n)
    pos,rot=pose_errors(pred['ee'][-1],ev.task['condition']['goals'])
    r={'position_error_m':pos,'rotation_error_rad':rot,
       'pose_pass_at_screen_grid':bool(max(pos)<=.005 and max(rot)<=np.deg2rad(1)),
       'goal':pose_vector(pred['ee'][-1],ev.task['condition']['goals']),
       'intervals':n,'qualification':'SCREEN_ONLY_NOT_MODEL_VALID'}
    minima=[];pairs=[]
    if geometry:
        for state in pred['qpos']:
            ev.budget.check();ev.scratch.qpos[:]=state;ev.robot.kinematics(ev.scratch)
            scan=ev.cc.scan(ev.scratch,P['active_buffer_m'])
            minima.append(scan['minimum_clearance_m']);pairs.append(scan['minimum_pair'])
        r.update(minima=np.asarray(minima),geometry_step_m=motion(pred,ev.robot))
        m=metric(r);r.update(metrics=m,rank=float(m['collision_violation']+.05*np.dot(r['goal'],r['goal'])))
    atomic_npz(directory/'trace.npz',Z=z,phase=pred['phase'],qpos=pred['qpos'],ee=pred['ee'],minima=np.asarray(minima))
    atomic_json(directory/'screen.json',r)
    if geometry:atomic_json(directory/'pairs.json',pairs)
    return r


def blocked_result(spec,reason):
    return {'status':'BLOCKED','reason':reason,'unit':spec,
        'candidates':[{'id':k,'eligible':False,'status':'DEPENDENCY_BLOCKED','reason':reason} for k in range(5)],
        'candidate_denominator':5,'execution_denominator':1,'execution_level':'NOT_EXECUTED','qualified':0}


def run_unit(directory,spec,entry,robot,initial,collision,profile,bundle,budget,progress):
    task=entry['task'];setup=public_setup(entry,robot,initial);base=setup[0];rows=[]
    atomic_json(directory/'public_setup.json',{'base':base,'jacobian':setup[1],
        'frame':setup[2],'bow_amplitude_m':setup[3],'task_hash':task['task_hash'],
        'public_fields_only':True,'private_solution_used':False})
    for k,seed in enumerate(spec['proposal_seeds']):
        pd=directory/'candidates'/f'{k:02d}'
        row={'id':k,'status':'NOT_COMPLETED','eligible':False,'selected_for_repair':False,'rank':None}
        raw,proposal,provenance=make(spec['method'],k,seed,task,setup,bundle)
        prepared,preparation=prepare(proposal,base,task,robot)
        atomic_npz(pd/'controls.npz',network_raw=raw,proposal=proposal,prepared=prepared)
        atomic_json(pd/'provenance.json',dict(provenance,preparation=preparation))
        try:
            candidate_deadline=min(time.time()+P['candidate_wall_s'],budget.unit_deadline-P['execution_reserve_s']-P['repair_wall_s'])
            if preparation['scale'] is None:
                row['status']='QUICK_REJECTED'
            else:
                quota=SliceBudget(budget,P['seed_coupled_cap'],P['seed_integration_cap'],candidate_deadline)
                seeded,seed_result=recover(pd/'goal_seed',prepared,Evaluator(task,robot,initial,collision,quota),progress)
                quota=SliceBudget(budget,P['search_coupled_cap'],P['search_integration_cap'],candidate_deadline)
                path,tree=search(pd/'tree',seeded,seed,task,robot,initial,collision,quota,progress)
                row.update(tree_result=tree,seed_result=seed_result,status='NO_COMPLETE_GLOBAL_ROUTE')
                if path is not None:
                    z,fit,times,desired,fitted=fit323(path,robot,task['condition']['duration'])
                    gates=quick_gates(robot,Reference(z,task['condition']['initial_features'],task['condition']['duration']))
                    atomic_npz(pd/'fit.npz',Z=z,times=times,desired=desired,fitted=fitted)
                    atomic_json(pd/'fit.json',dict(fit,quick_gates=gates,route_nodes=[n.edge_id for n in path]))
                    row.update(status='FIT_MOTION_REJECTED',fit=fit,fit_quick=gates)
                    if all(gates.values()):
                        quota=SliceBudget(budget,P['validation_coupled_cap'],P['validation_integration_cap'],candidate_deadline)
                        v=validate(pd/'before_repair',z,task,robot,initial,collision,profile,quota)
                        clearance=v['validation']['geometry']['minimum_clearance_m']
                        row.update(status='GLOBAL_ROUTE_CONVERTED_VALIDATED',eligible=bool(v['eligible']),
                            score=v['score'],rank=v['score']+1e6*min(0.,clearance)**2,
                            reference_hash=v['reference_hash'],validation_path='before_repair')
        except (BudgetExhausted,RuntimeError) as error:
            if not isinstance(error,BudgetExhausted) and str(error)!='BUDGET_EXHAUSTED':raise
            row.update(status='CANDIDATE_BUDGET_EXHAUSTED',reason=str(error),eligible=False)
        rows.append(row);atomic_json(pd/'outcome.json',row)
    needs=[r for r in rows if r['rank'] is not None and not r['eligible']]
    selected=sorted(needs,key=lambda r:(r['rank'],r['id']))[:1] if not any(r['eligible'] for r in rows) else []
    atomic_json(directory/'repair_selection.json',{'ids':[r['id'] for r in selected],
        'rule':'only if no candidate already qualified: lowest complete-route converted merit, then id',
        'before_repair':True,'screen_rows':rows,'quota':P['repair_coupled_cap']})
    for row in selected:
        pd=directory/'candidates'/f'{row["id"]:02d}';z=np.load(pd/'fit.npz')['Z']
        row['selected_for_repair']=True
        try:
            quota=SliceBudget(budget,P['repair_coupled_cap'],P['repair_integration_cap'],
                min(time.time()+P['repair_wall_s'],budget.unit_deadline-P['execution_reserve_s']))
            result=optimize(pd/'repair',z,'full323',None,Evaluator(task,robot,initial,collision,quota),progress,P['repair_iterations'])
            quota=SliceBudget(budget,P['validation_coupled_cap'],P['validation_integration_cap'],
                budget.unit_deadline-P['execution_reserve_s'])
            v=validate(pd/'validated',np.asarray(result['Z']),task,robot,initial,collision,profile,quota)
            row.update(status='REPAIRED_VALIDATED',eligible=bool(v['eligible']),score=v['score'],
                reference_hash=v['reference_hash'],validation_path='validated')
        except (BudgetExhausted,RuntimeError,ValueError) as error:
            if isinstance(error,BudgetExhausted) or str(error)=='BUDGET_EXHAUSTED':
                row.update(status='REPAIR_BUDGET_EXHAUSTED',reason=str(error),eligible=False)
            elif str(error)=='SPARSE_DENSE_NUMERICAL_DISAGREEMENT':
                row.update(status='NUMERICAL_UNRESOLVED',reason=str(error),eligible=False)
            else:raise
        atomic_json(pd/'outcome.json',row)
    eligible=sorted((r for r in rows if r['eligible']),key=lambda r:(r['score'],r['id']))
    chosen=eligible[0] if eligible else None
    atomic_json(directory/'execution_selection.json',{'candidate_id':chosen['id'] if chosen else None,
        'candidate_digest':digest(rows),'before_execution':True,'rule':'lowest qualified pose score then id; no replacement',
        'task_hash':task['task_hash']})
    executed={'level':'NOT_EXECUTED','reason':'NO_QUALIFIED_CANDIDATE'}
    if chosen is not None:
        budget.check();assert budget.native_steps+27500<=P['native_steps_cap']
        v=read_json(directory/'candidates'/f'{chosen["id"]:02d}'/chosen['validation_path']/'validation.json')
        progress('native67_execution_and_independent_full_replay',unit=spec['id'],candidate=chosen['id'])
        executed=audit_execution(task,profile,robot,initial,
            Reference(v['Z'],task['condition']['initial_features'],task['condition']['duration']),
            directory/'execution',min(budget.deadline,budget.unit_deadline))
    result={'status':'DONE','unit':spec,'candidates':rows,'candidate_denominator':5,
        'execution_denominator':1,'execution_level':executed['level'],'qualified':sum(r['eligible'] for r in rows),
        'no_replacement':True,'TRAIN_added':0}
    assert_frozen(task);atomic_json(directory/'result.json',result)
    return result

def run(out=OUT, progress=lambda *a,**k:None, previous_elapsed=0.):
    assert_resume_safe(out);ident=identity()
    with RunLock(out):
        assert read_json(PACKAGE/'evidence/software_tests.json')['source_hashes']==ident['source_hashes']
        if (out/'run.json').exists():assert read_json(out/'run.json')['identity']==ident
        else:
            atomic_json(out/'run.json',{'identity':ident,'producer_head':subprocess.check_output(
                ['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'TEST':False})
            with zipfile.ZipFile(out/'executed_source.zip','w',zipfile.ZIP_DEFLATED) as archive:
                for path in ident['source_hashes']:archive.write(ROOT/path,path)
        deadline=time.time()+max(0,P['wall_s']-previous_elapsed)
        config=read_json(BASE/'run.json')['config']
        physical,canonical,initial,_=audit(config);old=read_json(SOURCE/'physical_contract.json')
        assert physical['acceptance']==old['acceptance'] and digest(physical['identity'])==digest(old['identity'])
        assert physical['initial_state_id']==old['initial_state_id']
        profile=acceptance(config);atomic_json(out/'physical_contract.json',physical)
        bundle=load_bundle()
        atomic_json(out/'model_provenance.json',{'low_model':'sample_alignment_001/learning/aligned',
            'codec':'control_transfer_bounded_001/codec.json','new_training':False,
            'detail_head_zero':True,'network_condition_dimensions':65,'network_obstacle_conditioned':False,
            'complete_output_dimensions':323,'sampling_steps':16,'nonlinear_goal_seed':True,'forward_floating_tree':True})
        cache=TaskOriginSceneCache(canonical,initial,profile,max_entries=1,max_scene_models=1)
        ledger=Ledger(out,{'identity_hash':digest(ident)})
        saved=read_json(out/'budget.json') if (out/'budget.json').exists() else None
        budget=Budget(out,deadline,saved)
        entries={e['sample_id']:e for e in ident['selection']['entries']}
        first=ident['selection']['entries'][0]
        rr,ii,cc,proof=cache.get(first)
        budget.begin_unit();ledger.begin('m0','propagation_preflight',P['wall_s'],1);before=rr.calls.copy()
        quota=SliceBudget(budget,60,800000,min(budget.deadline,time.time()+360))
        preflight=verify_propagation(out/'m0',rr,ii,cc,quota)
        delta={k:rr.calls[k]-before[k] for k in before};budget.observe(delta)
        ledger.finish(preflight,delta);budget.save()
        if preflight['status']!='PASS':
            from .audit_results import report
            for spec in ident['selection']['units']:
                atomic_json(out/'units'/spec['id']/'result.json',blocked_result(spec,'M0_PROPAGATION_CHECK_FAILED'))
            checked={'status':'BLOCKED','reason':'M0_PROPAGATION_CHECK_FAILED'}
            atomic_json(out/'independent_verification.json',checked)
            return report(out,ident,checked)
        for spec in ident['selection']['units']:
            if spec['id'] in ledger.data['completed']:continue
            budget.begin_unit();directory=out/'units'/spec['id']
            entry=entries[spec['task_id']]
            # Build once per scene; setup is public contract work, not historical trajectory audit.
            rr,ii,cc,proof=cache.get(entry)
            assert_task_compatible(entry['task'],rr,ii,cc,profile)
            atomic_json(out/'scenes'/spec['task_id']/'initial_audit.json',proof)
            atomic_json(out/'scenes'/spec['task_id']/'limits.json',{'lower':rr.lower,'upper':rr.upper})
            ledger.begin(spec['id'],'global_bypass',P['wall_s'],6);before=rr.calls.copy()
            try:
                budget.check()
                result=run_unit(directory,spec,entry,rr,ii,cc,profile,bundle,budget,progress)
            except (BudgetExhausted,RuntimeError) as error:
                if not isinstance(error,BudgetExhausted) and str(error)!='BUDGET_EXHAUSTED':raise
                rows=[]
                for k in range(5):
                    path=directory/'candidates'/f'{k:02d}'/'outcome.json'
                    rows.append(read_json(path) if path.exists() else {'id':k,'status':'NOT_COMPLETED_BUDGET','eligible':False})
                result={'status':'BUDGET_EXHAUSTED','reason':str(error),'unit':spec,'candidates':rows,
                    'candidate_denominator':5,'execution_denominator':1,'execution_level':'NOT_COMPLETED',
                    'qualified':sum(r['eligible'] for r in rows)}
                atomic_json(directory/'result.json',result)
            delta={k:rr.calls[k]-before[k] for k in before}
            budget.observe(delta)
            atomic_json(directory/'calls.json',delta)
            ledger.finish({'status':result['status'],'execution_level':result['execution_level']},delta);budget.save()
        atomic_json(out/'runtime_cache.json',cache.stats())
        from .audit_results import verify, report
        progress('raw_evidence_accounting')
        assert identity()==ident,'FROZEN_SOURCE_OR_INPUT_CHANGED'
        checked=verify(out,ident);atomic_json(out/'independent_verification.json',checked)
        return report(out,ident,checked)
