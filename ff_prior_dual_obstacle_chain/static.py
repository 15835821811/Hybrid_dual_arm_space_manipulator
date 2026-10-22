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
from ff_prior_static_fullchain_v2.validation import validate
from ff_prior_static_fullchain_v2.proposals import load_bundle, public_setup, make, prepare


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


def run_unit(directory, spec, entry, robot, initial, collision, profile, bundle, budget, progress):
    task=entry['task'];setup=public_setup(entry,robot,initial);base=setup[0]
    atomic_json(directory/'public_setup.json',{'base':base,'jacobian':setup[1],
        'frame':setup[2],'bow_amplitude_m':setup[3],'task_hash':task['task_hash'],
        'public_fields_only':True,'private_solution_used':False})
    screen_budget=SliceBudget(budget,40,200000,min(budget.unit_deadline-P['execution_reserve_s'],time.time()+180))
    ev=Evaluator(task,robot,initial,collision,screen_budget)
    rows=[]
    for k,seed in enumerate(spec['proposal_seeds']):
        progress('public_proposals',unit=spec['id'],candidate=k)
        pd=directory/'candidates'/f'{k:02d}'
        raw,proposal,provenance=make(spec['method'],k,seed,task,setup,bundle)
        prepared,preparation=prepare(proposal,base,task,robot)
        atomic_npz(pd/'controls.npz',network_raw=raw,proposal=proposal,prepared=prepared)
        atomic_json(pd/'provenance.json',dict(provenance,preparation=preparation))
        screen(pd/'raw',raw,ev,P['raw_intervals'],False)
        row={'id':k,'status':'SCREENED','eligible':False,'selected_for_repair':False}
        if preparation['scale'] is None:
            row.update(status='QUICK_REJECTED',rank=None)
        else:
            s=screen(pd/'prepared',prepared,ev,P['screen_intervals'],True)
            row['rank']=s['rank']
        rows.append(row)
        atomic_json(pd/'outcome.json',row)
    selected=sorted((r for r in rows if r['rank'] is not None),key=lambda r:(r['rank'],r['id']))[:P['selected_repairs']]
    atomic_json(directory/'repair_selection.json',{'ids':[r['id'] for r in selected],
        'rule':P['screen_rank'],'before_search':True,'screen_rows':rows,'quota_per_selected':P['repair_coupled_cap']})
    for row in selected:
        k=row['id'];pd=directory/'candidates'/f'{k:02d}'
        z=np.load(pd/'controls.npz')['prepared']
        quota=SliceBudget(budget,P['repair_coupled_cap'],P['repair_integration_cap'],
            min(time.time()+P['repair_wall_s'],budget.unit_deadline-P['execution_reserve_s']))
        ev=Evaluator(task,robot,initial,collision,quota)
        row['selected_for_repair']=True
        try:
            result=optimize(pd/'repair',z,'full323',None,ev,progress,P['repair_iterations'])
            v=validate(pd/'validated',np.asarray(result['Z']),task,robot,initial,collision,profile,quota)
            row.update(status='VALIDATED',eligible=bool(v['eligible']),score=v.get('score'),reference_hash=v['reference_hash'])
        except (BudgetExhausted,RuntimeError,ValueError) as error:
            if isinstance(error,BudgetExhausted) or str(error)=='BUDGET_EXHAUSTED':
                row.update(status='BUDGET_EXHAUSTED',reason=str(error),eligible=False)
            elif str(error)=='SPARSE_DENSE_NUMERICAL_DISAGREEMENT':
                row.update(status='NUMERICAL_UNRESOLVED',reason=str(error),eligible=False)
            else:raise
        atomic_json(pd/'outcome.json',row)
    eligible=sorted((r for r in rows if r['eligible']),key=lambda r:(r['score'],r['id']))
    chosen=eligible[0]['id'] if eligible else None
    selection={'candidate_id':chosen,'candidate_digest':digest(rows),'before_execution':True,
        'rule':'lowest validated pose score then id; no replacement','task_hash':task['task_hash']}
    atomic_json(directory/'execution_selection.json',selection)
    executed={'level':'NOT_EXECUTED','reason':'NO_QUALIFIED_CANDIDATE'}
    if chosen is not None:
        budget.check()
        v=read_json(directory/'candidates'/f'{chosen:02d}'/'validated/validation.json')
        progress('native67_execution_and_independent_replay',unit=spec['id'],candidate=chosen)
        executed=audit_execution(task,profile,robot,initial,
            Reference(v['Z'],task['condition']['initial_features'],task['condition']['duration']),
            directory/'execution',min(budget.deadline,budget.unit_deadline))
        executed['goal_source']='frozen_user_static_application_world_goal'
        atomic_json(directory/'execution/report.json',executed)
    result={'status':'DONE','unit':spec,'candidates':rows,'candidate_denominator':5,
            'execution_denominator':1,'execution_level':executed['level'],
            'qualified':sum(r['eligible'] for r in rows),'no_replacement':True}
    assert_frozen(task);atomic_json(directory/'result.json',result)
    return result

