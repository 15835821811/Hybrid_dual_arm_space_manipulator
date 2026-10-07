"""Evidence/attribution counterexamples; all synthetic files stay temporary."""
from dataclasses import replace
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

import v6_4.build_teacher_cohort as cohort
import v6_4.report_comparison as report
from v6_4.contracts import TrajectoryProposal
from v6_4.run_teacher_comparison import SCORE_DEFINITION, _manifest
from v6_4.tests.test_task_protocol import fixture_task
from v6_4.trajectory_codec import CubicBSplineCodec


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,allow_nan=False),encoding='utf-8')


def absent_screen(path,task):
    path.mkdir(parents=True)
    write(path/'task.json',task.to_dict());write(path/'source_manifest.json',{'fixture':'a'*64})
    records=[]
    for index in range(8):
        record={'candidate_index':index,'proposal_sha256':None,'origin':None,'postprocessing':[],
            'raw_proposal_passed':False,'gate_passed':False,'eligible_for_selection':False,
            'score':None,'failure':{'phase':'generation'}}
        records.append(record)
        candidate=path/f'candidate_{index:03d}'
        write(candidate/'record.json',record)
        write(candidate/'gate.json',{'passed':False,'raw_passed':False})
    value={'task_id':task.task_id,'task_sha256':task.sha256(),'candidate_count':8,'records':records,
        'sources_unchanged':True,'selection_uses_actual_outcomes':False,'score_definition':SCORE_DEFINITION,
        'repair_used':False,'fallback_used':False,'selectedIndex':None,'selectedIndexK1':None}
    write(path/'selection.json',value);write(path/'artifact_manifest.json',_manifest(path))
    return value


def full_trial(path,task,method='fixed_reference',n=13500):
    """Tiny compressed source-binding fixture, never represented as a physical run."""
    path.mkdir(parents=True)
    result={'task_id':task.task_id,'task_sha256':task.sha256(),'method':method,'status':'TASK_COMPLETED',
        'proposal_accepted':True,'raw_proposal_passed':None if method=='fixed_reference' else True,
        'task_success':True,'complete':True,'fallback_used':False,'postprocessing':[]}
    write(path/'task.json',task.to_dict())
    write(path/'planning_identity.json',{'task_sha256':task.sha256(),'method':method})
    trace=path/'traces'/f'{task.task_id}.npz';trace.parent.mkdir()
    arrays={'time':.002*np.arange(1,n+1),'torque':np.zeros((n,67)),'planner_q':np.zeros((n,17)),
        'initial_qpos':np.asarray(task.initial_qpos),'initial_qvel':np.asarray(task.initial_qvel)}
    reference=None
    if method!='fixed_reference':
        proposal=TrajectoryProposal.from_controls(task,np.zeros((30,17)),origin=method)
        controls=CubicBSplineCodec(task.initial_planner_q,task.initial_planner_dq).decode_free(proposal.free_controls)
        write(path/'raw_proposal.json',proposal.to_dict())
        np.savez_compressed(path/'raw_proposal.npz',controls_free=proposal.free_controls,control_points=controls)
        write(path/'proposal_gate.json',{'task_id':task.task_id,'task_sha256':task.sha256(),'origin':method,
            'postprocessing':[],'passed':True,'raw_passed':True})
        reference=path/'selected_reference.npz';np.savez_compressed(reference,control_points=controls)
        arrays.update(generated_reference_time_s=arrays['time'],generated_reference_q=np.zeros((n,17)),
            generated_reference_dq=np.zeros((n,17)))
    np.savez_compressed(trace,**arrays)
    result.update(trace_path=str(trace.resolve()),reference_path=str(reference.resolve()) if reference else None)
    evaluation=path/'evaluation';evaluation.mkdir()
    fresh=evaluation/'fresh_replay.npz'
    np.savez_compressed(fresh,time=.002*np.arange(n+1),q=np.zeros((n+1,17)),
        qpos=np.tile(task.initial_qpos,(n+1,1)),qvel=np.tile(task.initial_qvel,(n+1,1)))
    (evaluation/'interval_boundaries.jsonl').write_text('{}\n')
    (evaluation/'interval_rows.jsonl').write_text('{}\n')
    evidence={'task_id':task.task_id,'task_sha256':task.sha256(),'model_contract_sha256':task.model_contract_sha256,
        'complete':True,'evidence_valid':True,'task_success':True,'trace_path':str(trace.resolve()),
        'trace_sha256':report._sha(trace),'fresh_replay_sha256':report._sha(fresh),'metrics':{'physics_steps':n},
        'reference_path':str(reference.resolve()) if reference else None,'reference_sha256':report._sha(reference) if reference else None,
        'task_requirements':{'passed':True},'execution_contract':{'passed':True},'native_geometry':{'passed':True},
        'rejection_count':0,'errors':[],
        'independent_interval':{'passed':True,'boundary_sha256':report._sha(evaluation/'interval_boundaries.jsonl'),
            'rows_sha256':report._sha(evaluation/'interval_rows.jsonl')}}
    write(evaluation/'report.json',evidence)
    write(evaluation/'manifest.json',{p.name:report._sha(p) for p in evaluation.iterdir()})
    write(path/'result.json',result)


class ComparisonEvidenceTests(unittest.TestCase):
    def test_all_absent_slots_are_retained_but_never_raw_success(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'screen';task=fixture_task();absent_screen(path,task)
            result=report.validate_selection(task,path,'teacher',[])
            self.assertEqual(len(result['records']),8)
            self.assertFalse(any(r['raw_proposal_passed'] for r in result['records']))
            self.assertIsNone(result['selectedIndex'])

    def test_duplicate_slots_and_missing_raw_files_fail_even_with_updated_manifest(self):
        for mutation in ('duplicate','missing','forged_pass'):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as temp:
                path=Path(temp)/'screen';task=fixture_task();value=absent_screen(path,task)
                if mutation=='duplicate':value['records'][1]=dict(value['records'][0])
                elif mutation=='missing':(path/'candidate_002/gate.json').unlink()
                else:
                    value['records'][0]['raw_proposal_passed']=True
                    write(path/'candidate_000/record.json',value['records'][0])
                    write(path/'candidate_000/gate.json',{'passed':False,'raw_passed':True})
                write(path/'selection.json',value)
                (path/'artifact_manifest.json').unlink();write(path/'artifact_manifest.json',_manifest(path))
                with self.assertRaises((ValueError,FileNotFoundError)):
                    report.validate_selection(task,path,'teacher',[])

    def test_full_bound_actual_fixture_passes_but_method_repair_acceptance_and_partial_fail(self):
        task=fixture_task()
        for mutation in (None,'method','repair','accepted','partial','trace_hash'):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as temp:
                path=Path(temp)/'actual';full_trial(path,task,'teacher',n=10 if mutation=='partial' else 13500)
                result=report._read(path/'result.json')
                if mutation=='method':result['method']='diffusion'
                elif mutation=='repair':result['postprocessing']=['repair'];result['proposal_accepted']=False
                elif mutation=='accepted':result['proposal_accepted']=False
                elif mutation=='trace_hash':
                    trace=Path(result['trace_path']);trace.write_bytes(trace.read_bytes()+b'changed')
                write(path/'result.json',result)
                if mutation is None:
                    verified=report.validate_actual_trial(task,path,'teacher');self.assertTrue(verified['task_success'])
                else:
                    with self.assertRaises(ValueError):report.validate_actual_trial(task,path,'teacher')

    def test_evaluation_manifest_tamper_and_task_substitution_fail(self):
        with tempfile.TemporaryDirectory() as temp:
            task=fixture_task();path=Path(temp)/'actual';full_trial(path,task)
            changed=replace(task,task_id='other_task',group_id='other_group')
            with self.assertRaises(ValueError):report.validate_actual_trial(changed,path,'fixed_reference')
            (path/'evaluation/interval_rows.jsonl').write_text('changed\n')
            with self.assertRaises(ValueError):report.validate_actual_trial(task,path,'fixed_reference')

    def test_denominator_cannot_repeat_one_fixed_task_three_times(self):
        row={'task_id':'same','method':'fixed_reference','candidate_count':1,'proposal_accepted':True,'task_success':True}
        with self.assertRaises(ValueError):report.summarize([row,row,row],'fixed_reference',1,None)

    def test_full_matrix_keeps_missing_candidates_as_zero_of_24_and_fixed_raw_na(self):
        tasks=[replace(fixture_task(),task_id=f'test_{i}',group_id=f'test_group_{i}',split='test',family=family)
            for i,family in enumerate(('end_effector_detour','mid_arm_detour','multiple_routes'))]
        with tempfile.TemporaryDirectory() as temp:
            base=Path(temp);suite=base/'suite.json';suite.write_text('{}')
            fixed=base/'fixed'
            for task in tasks:full_trial(fixed/task.task_id,task)
            for method in ('teacher','diffusion'):
                directory=base/method;directory.mkdir()
                rows=[]
                for task in tasks:
                    absent_screen(directory/task.task_id/'screening',task)
                    rows.extend({'method':method,'task_id':task.task_id,'task_sha256':task.sha256(),'candidate_count':k,
                        'status':'NO_ADMISSIBLE_RAW_PROPOSAL','selectedIndex':None,'proposal_accepted':False,
                        'raw_proposal_passed':False,'task_success':False,'complete':False,'new_physics_steps':0,
                        'reused_actual_trial':False} for k in (1,8))
                write(directory/'report.json',{'all_sources_unchanged':True,'checkpoint_unchanged':True,'rows':rows})
                protocol={'task_hashes':{t.task_id:t.sha256() for t in tasks},'task_suite_sha256':report._sha(suite),
                    'candidate_count':8,'repair_used':False,'fallback_used':False,'source_sha256':{'fixture':'a'*64}}
                if method=='diffusion':
                    checkpoint=directory/'checkpoint.pt';checkpoint.write_bytes(b'transient no-generated-candidates fixture')
                    protocol.update(checkpoint_path=str(checkpoint),checkpoint_sha256=report._sha(checkpoint))
                write(directory/'protocol.json',protocol);write(directory/'artifact_manifest.json',_manifest(directory))
            with patch.object(report,'load_task_suite',return_value=tuple(tasks)):
                result=report.build(suite,fixed,base/'teacher',base/'diffusion',base/'report')
            self.assertEqual(result['aggregates'][0]['total_completed_count'],3)
            self.assertIsNone(result['aggregates'][0]['raw_candidate_denominator'])
            for row in result['aggregates'][1:]:
                self.assertEqual(row['task_denominator'],3)
                self.assertEqual(row['raw_candidate_denominator'],3 if row['K']==1 else 24)
                self.assertEqual(row['total_completed_count'],0);self.assertEqual(row['raw_candidate_pass_count'],0)
            self.assertEqual(result['independent_diffusion_actual_execution_count'],0)
            # Copying one baseline result into another expected directory must
            # fail even though its JSON claims a valid completed task.
            write(fixed/tasks[1].task_id/'result.json',report._read(fixed/tasks[0].task_id/'result.json'))
            with patch.object(report,'load_task_suite',return_value=tuple(tasks)),self.assertRaises(ValueError):
                report.build(suite,fixed,base/'teacher',base/'diffusion',base/'invalid_report')


class CohortEvidenceTests(unittest.TestCase):
    def test_retry_preserves_original_failed_result_and_inputs_without_closed_loop_execution(self):
        task=fixture_task()
        with tempfile.TemporaryDirectory() as temp:
            source=Path(temp)/'source';destination=Path(temp)/'recovered';full_trial(source,task,'teacher')
            original=report._read(source/'result.json')
            original.update(complete=False,task_success=False,status='EXECUTION_OR_PIPELINE_FAILURE',failure={'type':'FileNotFoundError','phase':'monitor'})
            write(source/'result.json',original)
            write(source/'planning_identity.json',{'method':'teacher','task_sha256':task.sha256(),'qp_config':{}})
            before={p.relative_to(source).as_posix():report._sha(p) for p in source.rglob('*') if p.is_file()}
            proposal=TrajectoryProposal.from_dict(report._read(source/'raw_proposal.json'))
            with patch('v6_4.evaluate_planning.evaluate_trial',return_value={'complete':True,'task_success':False}) as evaluate,\
                 patch.object(cohort,'run_attempt',side_effect=AssertionError('new physical trial forbidden')):
                derived=cohort.recover_completed_trial(task,source,destination,proposal)
            self.assertEqual(evaluate.call_args.kwargs['scenario_result'],None)
            self.assertFalse(derived['task_success']);self.assertEqual(derived['new_closed_loop_physics_steps'],0)
            self.assertEqual(report._sha(destination/'prior_failed_result.json'),before['result.json'])
            self.assertEqual(report._sha(destination/'traces'/f'{task.task_id}.npz'),before[f'traces/{task.task_id}.npz'])
            self.assertEqual(before,{p.relative_to(source).as_posix():report._sha(p) for p in source.rglob('*') if p.is_file()})
            lineage=report._read(destination/'resume_lineage.json')
            self.assertEqual(lineage['new_closed_loop_physics_steps'],0);self.assertTrue(lineage['source_failure_preserved'])

    def test_output_absolute_and_real_short_windows_leaf_before_expensive_work(self):
        task=replace(fixture_task(),task_id='v6_4_end_effector_detour_train_001')
        with tempfile.TemporaryDirectory(prefix='v64_path_',dir=Path.cwd().anchor if os.name=='nt' else None) as temp:
            output,longest=cohort.check_output_paths(Path(temp)/'run',[task])
            self.assertTrue(output.is_absolute());self.assertLess(longest,250)
            leaf=output/'closed_loop'/task.task_id/'route_002'/'pcc_monitor'/task.task_id/'plots/minimum_clearance_comparison.png'
            leaf.parent.mkdir(parents=True);leaf.write_bytes(b'temporary path probe')
            self.assertEqual(leaf.read_bytes(),b'temporary path probe')
        if os.name=='nt':
            with self.assertRaises(ValueError):cohort.check_output_paths(Path('E:/')/('x'*150),[task])

    def test_duplicate_controls_cannot_be_a_second_route_or_be_registered_again(self):
        task=fixture_task();proposal=TrajectoryProposal.from_controls(task,np.zeros((30,17)),origin='teacher')
        proof={'task_success':True,'complete':True,'evidence_valid':True,'proposal':proposal,
            'trace_sha256':'a'*64,'actual_physics_steps':13500,'result':{'status':'TASK_COMPLETED'}}
        control_hash=report.hashlib.sha256(np.ascontiguousarray(proposal.free_controls,dtype='<f8').tobytes()).hexdigest()
        with patch.object(cohort,'validate_actual_trial',return_value=proof),patch.object(cohort,'collect',side_effect=AssertionError('duplicate registered')):
            sample,result=cohort.register_verified_route(task,Path('unused'),Path('unused'),{control_hash},set())
        self.assertIsNone(sample);self.assertEqual(result['status'],'DUPLICATE_ROUTE_NOT_COUNTED')

    def test_result_boolean_reuse_without_actual_teacher_evidence_cannot_meet_target(self):
        task=replace(fixture_task(),task_id='v6_4_end_effector_detour_train_000')
        old_cwd=Path.cwd()
        with tempfile.TemporaryDirectory() as temp:
            base=Path(temp)
            for index in (0,1):
                path=base/f'v6_4/output/teacher_closed_loop/ee_train_000_route{index}_01'
                write(path/'result.json',{'method':'fixed_reference','task_success':True,'task_sha256':task.sha256()})
            attempts=[SimpleNamespace(attempt_index=i,proposal=None,metadata={'failure':{'phase':'fixture'}}) for i in range(3)]
            try:
                os.chdir(base)
                with patch.object(cohort,'load_task_suite',return_value=(task,)),patch.object(cohort,'_sources',return_value={'fixture':'same'}),\
                     patch.object(cohort,'prepare',return_value=attempts),patch.object(cohort,'run_attempt',side_effect=AssertionError('physics forbidden')):
                    suite=base/'suite.json';suite.write_text('{}')
                    result=cohort.build(suite,base/'cohort')
            finally:os.chdir(old_cwd)
            self.assertFalse(result['all_success_targets_met']);self.assertEqual(result['dataset']['successful_teacher_count'],0)
            self.assertEqual(result['tasks'][0]['successful_routes'],0)

    def test_prepare_and_actual_errors_preserve_slots_and_continue_later_groups(self):
        tasks=[replace(fixture_task(),task_id=f'train_{i}',group_id=f'group_{i}') for i in range(3)]
        attempts=[SimpleNamespace(attempt_index=i,proposal=object(),metadata={'planning_wall_time_s':0.,'failure':None}) for i in range(3)]
        with tempfile.TemporaryDirectory() as temp:
            base=Path(temp);suite=base/'suite.json';suite.write_text('{}')
            with patch.object(cohort,'load_task_suite',return_value=tuple(tasks)),patch.object(cohort,'_sources',return_value={'fixture':'same'}),\
                 patch.object(cohort,'prepare',side_effect=[RuntimeError('fixture prepare failure'),attempts,attempts]),\
                 patch.object(cohort,'run_attempt',side_effect=RuntimeError('fixture actual failure')) as execute:
                result=cohort.build(suite,base/'cohort',reuse_first=False)
            self.assertTrue(result['all_tasks_attempted']);self.assertFalse(result['all_success_targets_met'])
            self.assertEqual(execute.call_count,6);self.assertEqual(result['dataset']['attempt_count'],9)
            self.assertTrue((base/'cohort/failures/train_0/prepare_failure.json').is_file())
            self.assertEqual(result['dataset']['successful_teacher_count'],0)


if __name__=='__main__':unittest.main()
