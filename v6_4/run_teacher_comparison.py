"""Same static gate/score for Teacher and raw Diffusion K1/K8 proposals.

Only statically selected proposals reach the original research torque runner.
No actual outcome participates in selection, repair or fallback. All attempts,
including absent/failed proposals and physical refusals, remain in the cohort.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time
import traceback

import numpy as np

from v6_lite.run_v6_lite import default_v6_lite_robot_spec
from v6_4.contracts import TrajectoryProposal
from v6_4.proposal_gate import gate_proposal
from v6_4.run_planning import load_task_suite, prepare_provider, run_attempt, save_reference
from v6_4.teacher_planner import TeacherPlanner, load_bootstrap
from v6_4.trajectory_codec import CubicBSplineCodec

ROOT=Path(__file__).resolve().parents[1]
SCORE_DEFINITION="sum(two predicted arm path lengths [m]) + maximum base translation [m] + 0.1*maximum base rotation [rad]"


def _sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def _write(path,value):
    with Path(path).open('x',encoding='utf-8') as f:json.dump(value,f,indent=2,allow_nan=False)
def _manifest(directory):
    return {p.relative_to(directory).as_posix():{'sha256':_sha(p),'bytes':p.stat().st_size}
            for p in sorted(directory.rglob('*')) if p.is_file()}
def _code_sources():
    paths=[ROOT/'v6_4'/name for name in ('run_teacher_comparison.py','contracts.py','task_protocol.py',
        'trajectory_codec.py','reference_adapter.py','teacher_planner.py','proposal_gate.py',
        'run_planning.py','evaluate_planning.py')]
    paths.extend(ROOT/p for p in ('v6_lite/run_v6_lite.py','v6_lite/hierarchical_qp.py',
        'model_test/robot_model_spec_v5.py','model_test/whole_body_verifier_v5.py'))
    return {p.relative_to(ROOT).as_posix():_sha(p) for p in sorted(paths)}


def static_score(task,prediction):
    """Declared world path/base cost, computed only from private prediction."""
    t=np.asarray(prediction['time'],dtype=float)
    if t.shape!=(1351,) or not np.allclose(t,np.arange(1351)*.02,atol=1e-12,rtol=0.):
        raise ValueError('static score needs the original complete 1351-state nominal grid')
    lengths={}
    for arm in ('rigid','continuum'):
        p=np.asarray(prediction[arm+'_position'],dtype=float)
        if p.shape!=(1351,3) or not np.all(np.isfinite(p)):raise ValueError('invalid predicted arm path')
        lengths[arm]=float(np.sum(np.linalg.norm(np.diff(p,axis=0),axis=1)))
    base=np.asarray(prediction['base_pose'],dtype=float)
    if base.shape!=(1351,7) or not np.all(np.isfinite(base)):raise ValueError('invalid predicted base pose')
    qnorm=np.linalg.norm(base[:,3:],axis=1)
    initial=np.asarray(task.base_pose,dtype=float)
    if np.max(abs(qnorm-1.))>1e-10 or abs(np.linalg.norm(initial[3:])-1.)>1e-10:
        raise ValueError('base rotation score needs normalized saved quaternions')
    translation=float(np.max(np.linalg.norm(base[:,:3]-initial[:3],axis=1)))
    # Quaternion sign is irrelevant. Clip cosine roundoff only, not proposals.
    rotation=float(np.max(2.*np.arccos(np.clip(abs(base[:,3:]@initial[3:]),0.,1.))))
    components={'arm_path_lengths_m':lengths,'base_translation_max_m':translation,
                'base_rotation_max_rad':rotation,'base_rotation_coefficient_m_per_rad':.1}
    return float(sum(lengths.values())+translation+.1*rotation),components


def _select_records(records):
    eligible=[r for r in records if r['eligible_for_selection']]
    chosen=min(eligible,key=lambda row:(row['score'],row['candidate_index'])) if eligible else None
    first=records[0] if records else None
    return (chosen['candidate_index'] if chosen else None,
            0 if first and first['eligible_for_selection'] else None)


def evaluate_candidate_set(task,proposals_or_None,output):
    """Return records and selectedIndex/K1 without running any actual physics.

    Input is an ordered list of at most eight immutable TrajectoryProposal or
    None values. None preserves a failed generation slot. selectedIndex is
    the lowest-cost raw-gate-passing proposal across the supplied slots;
    selectedIndexK1 is 0 only when candidate 0 passes the same gate. A shared
    1351-state native geometry gate applies independently to every proposal.
    """
    proposals=list(proposals_or_None)
    if not 1<=len(proposals)<=8:raise ValueError('candidate budget must be 1..8 with no dropped slots')
    if any(p is not None and not isinstance(p,TrajectoryProposal) for p in proposals):
        raise TypeError('candidate slots must contain immutable TrajectoryProposal or None')
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    sources=_code_sources();_write(output/'source_manifest.json',sources)
    _write(output/'task.json',task.to_dict())
    spec=default_v6_lite_robot_spec();codec=CubicBSplineCodec(task.initial_planner_q,task.initial_planner_dq)
    records=[];set_started=time.perf_counter()
    for index,proposal in enumerate(proposals):
        directory=output/f'candidate_{index:03d}';directory.mkdir()
        started=time.perf_counter()
        row={'candidate_index':index,'proposal_sha256':proposal.sha256() if proposal is not None else None,
            'origin':proposal.origin if proposal is not None else None,'gate_passed':False,'raw_proposal_passed':False,
            'eligible_for_selection':False,'score':None,'score_components':None,'failure':None,
            'postprocessing':list(proposal.postprocessing) if proposal is not None else [],
            'preparation_wall_s':0.,'gate_wall_s':0.,'scoring_wall_s':0.,'physics_steps':0}
        try:
            if proposal is None:
                gate={'schema':'v6_4_absent_proposal_gate_record_v1','passed':False,'raw_passed':False,
                      'status':'GENERATION_FAILED_NO_PROPOSAL','physics_steps_executed':0,
                      'nominal_grid_period_s':.02,'nominal_grid_state_count':1351,'nominal_screen':None}
                row['failure']={'phase':'generation','message':'No proposal; retain this candidate budget slot'}
                _write(directory/'gate.json',gate)
            else:
                if not isinstance(proposal,TrajectoryProposal):raise TypeError('only immutable TrajectoryProposal or None accepted')
                _write(directory/'proposal.json',proposal.to_dict())
                controls=codec.decode_free(proposal.free_controls)
                np.savez_compressed(directory/'control_points.npz',control_points=controls,controls_free=proposal.free_controls)
                pre=time.perf_counter();provider,_=prepare_provider(task,controls,spec)
                row['preparation_wall_s']=time.perf_counter()-pre
                save_reference(provider,directory/'nominal_prediction.npz')
                _write(directory/'reference_identity.json',provider.metadata)
                row['nominal_prediction_sha256']=_sha(directory/'nominal_prediction.npz')
                gate=gate_proposal(task,proposal,spec,provider=provider,output_path=directory/'gate.json')
                row['gate_wall_s']=float(gate['elapsed_wall_s'])
                row['gate_passed']=bool(gate['passed']);row['raw_proposal_passed']=bool(gate['raw_passed'])
                if row['gate_passed'] and row['raw_proposal_passed'] and not proposal.postprocessing and proposal.origin!='fallback':
                    pre=time.perf_counter();row['score'],row['score_components']=static_score(task,provider.prediction)
                    row['scoring_wall_s']=time.perf_counter()-pre;row['eligible_for_selection']=True
        except Exception as exc:
            row['failure']={'phase':'preparation_gate_or_score','type':type(exc).__name__,'message':str(exc),'traceback':traceback.format_exc()}
            if not (directory/'gate.json').exists():
                _write(directory/'gate.json',{'passed':False,'raw_passed':False,'status':'CANDIDATE_EVIDENCE_ERROR','failure':row['failure']})
        row['total_screen_wall_s']=time.perf_counter()-started
        _write(directory/'record.json',row);records.append(row)
        print(json.dumps({'task_id':task.task_id,'candidate':index,'raw_gate_passed':row['raw_proposal_passed'],
                          'score':row['score'],'screen_wall_s':row['total_screen_wall_s'],'failure':row['failure']}),flush=True)
    selected,first=_select_records(records)
    report={'schema':'v6_4_common_candidate_selection_v1','task_id':task.task_id,'task_sha256':task.sha256(),
        'candidate_count':len(proposals),'records':records,'selectedIndex':selected,'selectedIndexK1':first,
        'score_definition':SCORE_DEFINITION,'selection_tie_break':'lowest candidate_index',
        'selection_uses_actual_outcomes':False,'total_screen_wall_s':time.perf_counter()-set_started,
        'sources_unchanged':sources==_code_sources(),'gate_period_s':.02,'gate_state_count':1351,
        'gate_minimum_clearance_m':.005,'repair_used':False,'fallback_used':False,'physics_steps':0}
    report['prefix_K1_screen_wall_s']=float(records[0]['total_screen_wall_s'])
    report['cost_scope']='all supplied candidates actually screened; K1 prefix cost excludes shared setup and is not a separately timed single-candidate invocation'
    _write(output/'selection.json',report);_write(output/'artifact_manifest.json',_manifest(output))
    if not report['sources_unchanged']:raise RuntimeError('candidate-screening source freeze changed')
    return report


def _physical_steps(directory):
    traces=list((directory/'traces').glob('*.npz'))+list((directory/'failures').glob('*partial_trace.npz'))
    counts=[]
    for path in traces:
        with np.load(path,allow_pickle=False) as data:counts.append(len(data['torque']))
    return max(counts,default=0)


def _failed_trial_row(task,k,selected,proposals,task_output,error,*,method='teacher',phase='execution'):
    """Save one failed comparison cell without changing another K's result."""
    directory=Path(task_output)/f'K{k}';directory.mkdir(parents=True,exist_ok=True)
    failure={'phase':phase,'type':type(error).__name__,'message':str(error),'traceback':traceback.format_exc()}
    row={'task_id':task.task_id,'task_sha256':task.sha256(),'candidate_count':k,'selectedIndex':selected,
        'method':method,'task_success':False,'complete':False,'proposal_accepted':False,'raw_proposal_passed':False,
        'status':'COMPARISON_EVIDENCE_ERROR','new_physics_steps':0,'reused_actual_trial':False,'failure':failure}
    if selected is not None and 0<=selected<len(proposals) and proposals[selected] is not None:
        row['proposal_sha256']=proposals[selected].sha256()
    actual=directory/'actual'
    if actual.is_dir():
        row['actual_trial_path']=actual.as_posix()
        row['actual_artifacts']={p.relative_to(actual).as_posix():{'sha256':_sha(p),'bytes':p.stat().st_size}
                                 for p in sorted(actual.rglob('*')) if p.is_file()}
        try:row['new_physics_steps']=_physical_steps(actual)
        except Exception as count_error:
            row['new_physics_steps']=None
            row['physics_step_count_error']={'type':type(count_error).__name__,'message':str(count_error)}
    # No actual path is emitted when execution never created one. In particular
    # rejected reuse of another task's evidence is not attributed to this cell.
    _write(directory/'comparison_failure.json',failure)
    _write(directory/'comparison_row.json',row)
    return row


def _trial_row(task,k,selected,proposals,task_output,prior_trial=None,*,method='teacher'):
    """Return a retained cell, including partial evidence after pipeline errors."""
    if method not in ('teacher','diffusion'):raise ValueError('shared physical comparison method must be teacher or diffusion')
    directory=task_output/f'K{k}';directory.mkdir()
    try:
        return _trial_row_body(task,k,selected,proposals,task_output,prior_trial,method=method)
    except Exception as error:
        return _failed_trial_row(task,k,selected,proposals,task_output,error,method=method)


def _trial_row_body(task,k,selected,proposals,task_output,prior_trial=None,*,method='teacher'):
    directory=task_output/f'K{k}'
    if selected is None:
        row={'task_id':task.task_id,'task_sha256':task.sha256(),'candidate_count':k,'selectedIndex':None,
             'task_success':False,'complete':False,'proposal_accepted':False,'raw_proposal_passed':False,
             'status':'NO_ADMISSIBLE_RAW_PROPOSAL','new_physics_steps':0,'reused_actual_trial':False}
    elif prior_trial is not None and prior_trial['proposal_sha256']==proposals[selected].sha256():
        original=Path(prior_trial['actual_trial_path']);result=json.loads((original/'result.json').read_text(encoding='utf-8'))
        actual_proposal=TrajectoryProposal.from_dict(json.loads((original/'raw_proposal.json').read_text(encoding='utf-8')))
        if (result['task_sha256']!=task.sha256() or result['method']!=method
                or actual_proposal.sha256()!=proposals[selected].sha256()
                or actual_proposal.task_sha256!=task.sha256()):
            raise ValueError('reused actual trial task, method or raw proposal identity differs')
        files={p.relative_to(original).as_posix():{'sha256':_sha(p),'bytes':p.stat().st_size}
               for p in original.rglob('*') if p.is_file()}
        _write(directory/'reuse_lineage.json',{'original_trial':original.relative_to(ROOT).as_posix(),
            'proposal_sha256':proposals[selected].sha256(),'original_task_sha256':task.sha256(),
            'original_method':method,
            'same_selected_proposal':True,'original_files':files,'new_physics_steps':0,'actual_outcome_did_not_select_proposal':True})
        row={'task_id':task.task_id,'task_sha256':task.sha256(),'candidate_count':k,'selectedIndex':selected,
            'proposal_sha256':proposals[selected].sha256(),'task_success':bool(result['task_success']),
            'complete':bool(result['complete']),'proposal_accepted':bool(result['proposal_accepted']),
            'raw_proposal_passed':bool(result['raw_proposal_passed']),'status':result['status'],
            'actual_trial_path':original.as_posix(),'new_physics_steps':0,'reused_actual_trial':True}
    else:
        actual=directory/'actual'
        result=run_attempt(task,actual,method=method,proposal=proposals[selected],require_raw_gate=True,
            attribution={'comparison_candidate_count':k,'selected_candidate_index':selected,'static_selection':True,
                         'score_definition':SCORE_DEFINITION,'no_actual_outcome_selection':True})
        row={'task_id':task.task_id,'task_sha256':task.sha256(),'candidate_count':k,'selectedIndex':selected,
            'proposal_sha256':proposals[selected].sha256(),'task_success':bool(result['task_success']),
            'complete':bool(result['complete']),'proposal_accepted':bool(result['proposal_accepted']),
            'raw_proposal_passed':bool(result['raw_proposal_passed']),'status':result['status'],
            'actual_trial_path':actual.as_posix(),'new_physics_steps':_physical_steps(actual),'reused_actual_trial':False}
    row['method']=method
    _write(directory/'comparison_row.json',row)
    return row


def run_comparison(tasks_path,output,*,seed=64,rounds=2,geometry_rounds=8):
    """All three frozen test tasks, eight starts, one common old scene04 prior."""
    tasks=tuple(task for task in load_task_suite(tasks_path) if task.split=='test')
    if len(tasks)!=3 or len({task.family for task in tasks})!=3:
        raise ValueError('protocol-B comparison requires all three test families')
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    prior=ROOT/'v6_lite/output/runs/research_acceptance_01/simulation/traces/v6_lite_scenario_04.npz'
    times,q,identity=load_bootstrap(prior)
    source=_code_sources();started=time.perf_counter()
    _write(output/'protocol.json',{'task_suite_path':Path(tasks_path).resolve().relative_to(ROOT).as_posix(),
        'task_suite_sha256':_sha(tasks_path),'test_task_ids':[t.task_id for t in tasks],
        'task_hashes':{t.task_id:t.sha256() for t in tasks},'source_sha256':source,
        'prior':identity,'prior_is_test_future_trace':False,'seed':seed,'teacher_starts':8,
        'optimization_rounds':rounds,'ik_nfev_per_anchor':35,'teacher_geometry_stride':10,
        'geometry_optimization_rounds':geometry_rounds,'geometry_line_search_steps':4,
        'geometry_active_witnesses_per_round_budget':8,'geometry_native_gradient_queries_per_round_budget':272,
        'geometry_control_step_max_abs_rad':.25,'geometry_target_clearance_m':.007,
        'unchanged_geometry_acceptance_clearance_m':.005,
        'K1_definition':'candidate0 only','K8_definition':'minimum shared static score among eight raw-gatepassed candidates',
        'score_definition':SCORE_DEFINITION,'selection_before_actual_execution':True,
        'repair_used':False,'fallback_used':False,'duration_s':27.,'physics_period_s':.002,'task_period_s':.02})
    all_rows=[]
    for task in tasks:
        directory=output/task.task_id;directory.mkdir();planning=directory/'planning';planning.mkdir()
        planning_started=time.perf_counter();generation_failure=None
        try:
            attempts=TeacherPlanner(default_v6_lite_robot_spec(),seed=seed,max_starts=8,
                optimization_rounds=rounds,ik_max_evaluations=35,geometry_stride=10,
                geometry_optimization_rounds=geometry_rounds,geometry_line_search_steps=4).propose(
                    task,times,q,bootstrap_metadata=identity)
            if len(attempts)!=8:raise RuntimeError('teacher changed the eight-slot candidate budget')
        except Exception as exc:
            generation_failure={'type':type(exc).__name__,'message':str(exc),'traceback':traceback.format_exc()};attempts=[None]*8
        proposals=[]
        for index,attempt in enumerate(attempts):
            candidate=planning/f'candidate_{index:03d}';candidate.mkdir()
            _write(candidate/'planning_attempt.json',attempt.to_dict() if attempt is not None else {
                'attempt_index':index,'proposal':None,'failure':generation_failure,'closed_loop_success_established':False})
            proposals.append(attempt.proposal if attempt is not None else None)
            if attempt is not None and attempt.controls is not None:np.savez_compressed(candidate/'control_points.npz',control_points=attempt.controls)
            if attempt is not None and attempt.provider is not None:save_reference(attempt.provider,candidate/'nominal_prediction.npz')
            if attempt is not None and attempt.proposal is not None:_write(candidate/'proposal.json',attempt.proposal.to_dict())
        _write(planning/'generation_cost.json',{'generation_wall_s':time.perf_counter()-planning_started,
            'individual_planning_wall_s':[a.metadata['planning_wall_time_s'] if a is not None else None for a in attempts],
            'generation_failure':generation_failure,'physics_steps':0})
        try:selection=evaluate_candidate_set(task,proposals,directory/'screening')
        except Exception as error:
            selection=None
            first=_failed_trial_row(task,1,None,proposals,directory,error,phase='candidate_selection')
            eighth=_failed_trial_row(task,8,None,proposals,directory,error,phase='candidate_selection')
        if selection is not None:
            try:first=_trial_row(task,1,selection['selectedIndexK1'],proposals,directory)
            except Exception as error:
                first=_failed_trial_row(task,1,selection['selectedIndexK1'],proposals,directory,error)
            try:
                # An incomplete evidence-error cell cannot supply a valid
                # result/proposal lineage for reuse, even if it has a trace.
                prior=first if first.get('actual_trial_path') and first['status']!='COMPARISON_EVIDENCE_ERROR' else None
                eighth=_trial_row(task,8,selection['selectedIndex'],proposals,directory,prior_trial=prior)
            except Exception as error:
                eighth=_failed_trial_row(task,8,selection['selectedIndex'],proposals,directory,error)
        for row in (first,eighth):
            row['generation_scope']='eight-start invocation; K1 is candidate0 prefix comparator'
        all_rows.extend([first,eighth]);_write(directory/'task_result.json',{'rows':[first,eighth]})
        print(json.dumps({'task_id':task.task_id,'K1':first['status'],'K8':eighth['status'],
            'K1_index':first['selectedIndex'],'K8_index':eighth['selectedIndex']}),flush=True)
    summary={'schema':'v6_4_teacher_test_comparison_v1','test_task_count':len(tasks),'candidate_attempt_count':len(tasks)*8,
        'rows':all_rows,'by_K':{str(k):{'task_count':len(tasks),'accepted_count':sum(r['proposal_accepted'] for r in all_rows if r['candidate_count']==k),
            'task_success_count':sum(r['task_success'] for r in all_rows if r['candidate_count']==k)} for k in (1,8)},
        'new_physics_steps':sum(r['new_physics_steps'] for r in all_rows if r['new_physics_steps'] is not None),
        'new_physics_steps_exact_known':all(r['new_physics_steps'] is not None for r in all_rows),
        'elapsed_wall_s':time.perf_counter()-started,
        'all_sources_unchanged':source==_code_sources(),'all_three_tasks_attempted':len(all_rows)==6,
        'wall_20ms_is_acceptance_gate':False,'hardware':'NOT_ESTABLISHED','closed_loop_robustness':'NOT_ESTABLISHED'}
    _write(output/'report.json',summary);_write(output/'artifact_manifest.json',_manifest(output))
    return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tasks',type=Path,default=ROOT/'v6_4/output/tasks/protocol_b_01.json')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--seed',type=int,default=64)
    parser.add_argument('--rounds',type=int,default=2)
    parser.add_argument('--geometry-rounds',type=int,default=8)
    args=parser.parse_args();result=run_comparison(args.tasks,args.output,seed=args.seed,rounds=args.rounds,geometry_rounds=args.geometry_rounds)
    print(json.dumps({'by_K':result['by_K'],'all_three_tasks_attempted':result['all_three_tasks_attempted'],
        'all_sources_unchanged':result['all_sources_unchanged'],'new_physics_steps':result['new_physics_steps']}),flush=True)
