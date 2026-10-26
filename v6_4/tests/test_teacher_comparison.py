import json,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from scipy.spatial.transform import Rotation
from v6_4.contracts import TrajectoryProposal
from v6_4.run_teacher_comparison import static_score,_select_records,evaluate_candidate_set,_trial_row,run_comparison,ROOT


class CandidateComparisonTests(unittest.TestCase):
    def task(self):
        return SimpleNamespace(task_id='shared_selector',initial_planner_q=np.zeros(17),initial_planner_dq=np.zeros(17),
            base_pose=(0.,0.,0.,1.,0.,0.,0.),sha256=lambda:'a'*64,to_dict=lambda:{'task_id':'shared_selector'})

    def prediction(self):
        u=np.linspace(0.,1.,1351);base=np.zeros((1351,7));base[:,0]=.2*u
        base[:,3:]=Rotation.from_rotvec(np.c_[np.zeros((1351,2)),.3*u]).as_quat()[:,[3,0,1,2]]
        return {'time':np.arange(1351)*.02,'rigid_position':np.c_[.4*u,np.zeros((1351,2))],
            'continuum_position':np.c_[np.zeros(1351),.1*u,np.zeros(1351)],'base_pose':base}

    def test_frozen_units_and_paths_in_shared_score(self):
        score,terms=static_score(self.task(),self.prediction())
        self.assertAlmostEqual(score,.73,places=12)
        self.assertAlmostEqual(terms['base_rotation_max_rad'],.3,places=12)

    def test_invalid_grid_quaternion_and_nonfinite_rejected(self):
        for field,index,value in [('time',1,.1),('base_pose',(2,3),2.),('rigid_position',(2,0),float('nan'))]:
            p=self.prediction();p[field][index]=value
            with self.assertRaises(ValueError):static_score(self.task(),p)

    def test_K1_is_candidate0_and_K8_tie_uses_lowest_index(self):
        rows=[{'candidate_index':i,'eligible_for_selection':i>0,'score':1. if i>0 else 0.} for i in range(8)]
        self.assertEqual(_select_records(rows),(1,None))
        rows[0]['eligible_for_selection']=True
        self.assertEqual(_select_records(rows),(0,0))

    def test_all_eight_generation_failures_preserve_denominator_no_physics(self):
        with tempfile.TemporaryDirectory() as tmp,patch('v6_4.run_teacher_comparison.prepare_provider',side_effect=AssertionError('no prediction')),\
             patch('v6_4.run_teacher_comparison.run_attempt',side_effect=AssertionError('no physics')):
            report=evaluate_candidate_set(self.task(),[None]*8,Path(tmp)/'screen')
            self.assertEqual(len(report['records']),8);self.assertIsNone(report['selectedIndex'])
            self.assertIsNone(report['selectedIndexK1'])
            for i in range(8):
                self.assertTrue((Path(tmp)/'screen'/f'candidate_{i:03d}'/'gate.json').exists())

    def test_raw_gate_and_no_repair_are_required_even_for_low_score(self):
        task=self.task();raw=TrajectoryProposal.from_controls(task,np.zeros((30,17)),origin='diffusion')
        repaired=TrajectoryProposal.from_controls(task,np.zeros((30,17)),origin='diffusion',postprocessing=('repair',))
        provider=SimpleNamespace(prediction=self.prediction(),controls=np.zeros((32,17)),metadata={})
        def gate(task,proposal,spec,*,provider,output_path):
            value={'passed':True,'raw_passed':not proposal.postprocessing,'elapsed_wall_s':.001}
            Path(output_path).write_text(json.dumps(value));return value
        with tempfile.TemporaryDirectory() as tmp,patch('v6_4.run_teacher_comparison.prepare_provider',return_value=(provider,None)),\
             patch('v6_4.run_teacher_comparison.gate_proposal',side_effect=gate),\
             patch('v6_4.run_teacher_comparison.run_attempt',side_effect=AssertionError('no physics')):
            report=evaluate_candidate_set(task,[repaired,None,raw]+[None]*5,Path(tmp)/'screen')
            self.assertEqual(report['selectedIndex'],2);self.assertIsNone(report['selectedIndexK1'])
            self.assertFalse(report['records'][0]['eligible_for_selection'])
            self.assertTrue(report['records'][2]['raw_proposal_passed'])

    def test_same_selected_proposal_reuses_failed_actual_trial_without_physics(self):
        task=self.task();proposal=TrajectoryProposal.from_controls(task,np.zeros((30,17)),origin='teacher')
        with tempfile.TemporaryDirectory(dir=ROOT/'v6_4/output') as tmp:
            directory=Path(tmp);original=directory/'original';original.mkdir()
            result={'task_success':False,'complete':False,'proposal_accepted':True,'raw_proposal_passed':True,
                    'status':'EXECUTION_OR_PIPELINE_FAILURE','task_sha256':task.sha256(),'method':'teacher'}
            (original/'result.json').write_text(json.dumps(result))
            (original/'raw_proposal.json').write_text(json.dumps(proposal.to_dict()))
            previous={'proposal_sha256':proposal.sha256(),'actual_trial_path':str(original)}
            with patch('v6_4.run_teacher_comparison.run_attempt',side_effect=AssertionError('duplicate physics forbidden')):
                row=_trial_row(task,8,0,[proposal],directory,prior_trial=previous)
            self.assertTrue(row['reused_actual_trial']);self.assertFalse(row['task_success'])
            self.assertEqual(row['new_physics_steps'],0)
            lineage=json.loads((directory/'K8/reuse_lineage.json').read_text())
            self.assertEqual(lineage['proposal_sha256'],proposal.sha256())
            self.assertIn('result.json',lineage['original_files'])

    def test_reuse_rejects_another_method_or_task_identity(self):
        task=self.task();proposal=TrajectoryProposal.from_controls(task,np.zeros((30,17)),origin='diffusion')
        with tempfile.TemporaryDirectory(dir=ROOT/'v6_4/output') as tmp:
            directory=Path(tmp);original=directory/'original';original.mkdir()
            (original/'result.json').write_text(json.dumps({'task_sha256':task.sha256(),'method':'teacher'}))
            (original/'raw_proposal.json').write_text(json.dumps(proposal.to_dict()))
            previous={'proposal_sha256':proposal.sha256(),'actual_trial_path':str(original)}
            row=_trial_row(task,8,0,[proposal],directory,prior_trial=previous,method='diffusion')
            self.assertEqual(row['status'],'COMPARISON_EVIDENCE_ERROR')
            self.assertIn('identity differs',row['failure']['message'])
            self.assertNotIn('actual_trial_path',row)
            self.assertEqual(row['new_physics_steps'],0)

    def test_no_admissible_candidate_is_task_failure_without_physics(self):
        with tempfile.TemporaryDirectory() as tmp,patch('v6_4.run_teacher_comparison.run_attempt',side_effect=AssertionError('no physics')):
            row=_trial_row(self.task(),1,None,[None]*8,Path(tmp))
            self.assertEqual(row['status'],'NO_ADMISSIBLE_RAW_PROPOSAL')
            self.assertFalse(row['task_success']);self.assertEqual(row['new_physics_steps'],0)

    def test_shared_trial_retains_partial_path_steps_on_runner_exception(self):
        task=self.task();proposal=TrajectoryProposal.from_controls(task,np.zeros((30,17)),origin='diffusion')
        def interrupted(task,output,**kwargs):
            (output/'failures').mkdir(parents=True)
            np.savez(output/'failures/scene_partial_trace.npz',torque=np.zeros((7,67)))
            raise RuntimeError('failure after seven saved steps')
        with tempfile.TemporaryDirectory() as tmp,patch('v6_4.run_teacher_comparison.run_attempt',side_effect=interrupted):
            directory=Path(tmp);row=_trial_row(task,8,0,[proposal],directory,method='diffusion')
            self.assertEqual(row['status'],'COMPARISON_EVIDENCE_ERROR')
            self.assertEqual(row['method'],'diffusion');self.assertEqual(row['new_physics_steps'],7)
            self.assertEqual(row['actual_trial_path'],(directory/'K8/actual').as_posix())
            self.assertIn('failures/scene_partial_trace.npz',row['actual_artifacts'])
            self.assertTrue((directory/'K8/comparison_failure.json').exists())

    def test_shared_trial_preexecution_error_has_no_invented_actual_path(self):
        task=self.task();proposal=TrajectoryProposal.from_controls(task,np.zeros((30,17)),origin='teacher')
        with tempfile.TemporaryDirectory() as tmp,patch('v6_4.run_teacher_comparison.run_attempt',side_effect=RuntimeError('before mkdir')):
            row=_trial_row(task,1,0,[proposal],Path(tmp))
            self.assertNotIn('actual_trial_path',row);self.assertEqual(row['new_physics_steps'],0)

    def test_comparison_K8_exception_preserves_successful_K1_and_partial_K8(self):
        tasks=[SimpleNamespace(**{**vars(self.task()),'task_id':f'test_{i}','family':str(i),'split':'test'}) for i in range(3)]
        proposal=TrajectoryProposal.from_controls(tasks[0],np.zeros((30,17)),origin='teacher')
        attempt=SimpleNamespace(proposal=proposal,controls=None,provider=None,metadata={'planning_wall_time_s':.1},to_dict=lambda:{})
        def trials(task,k,selected,proposals,output,prior_trial=None):
            actual=output/f'K{k}/actual';(actual/'traces').mkdir(parents=True)
            np.savez(actual/'traces/scene.npz',torque=np.zeros((13500 if k==1 else 11,67)))
            if k==8:raise RuntimeError('K8 pipeline failed after partial physics')
            return {'task_id':task.task_id,'task_sha256':task.sha256(),'candidate_count':1,'selectedIndex':0,
                'proposal_sha256':proposal.sha256(),'status':'PASSED','task_success':True,'complete':True,
                'proposal_accepted':True,'raw_proposal_passed':True,'actual_trial_path':actual.as_posix(),
                'new_physics_steps':13500,'reused_actual_trial':False}
        with tempfile.TemporaryDirectory(dir=ROOT/'v6_4/output') as tmp:
            tasks_path=Path(tmp)/'tasks.json';tasks_path.write_text('{}')
            with patch('v6_4.run_teacher_comparison.load_task_suite',return_value=tasks),\
                 patch('v6_4.run_teacher_comparison.load_bootstrap',return_value=(None,None,{})),\
                 patch('v6_4.run_teacher_comparison.TeacherPlanner') as planner,\
                 patch('v6_4.run_teacher_comparison.evaluate_candidate_set',return_value={'selectedIndexK1':0,'selectedIndex':1}),\
                 patch('v6_4.run_teacher_comparison._trial_row',side_effect=trials):
                planner.return_value.propose.return_value=[attempt]*8
                result=run_comparison(tasks_path,Path(tmp)/'comparison',rounds=8)
            self.assertEqual(result['by_K']['1']['task_success_count'],3)
            self.assertEqual(result['by_K']['8']['task_success_count'],0)
            self.assertEqual(result['new_physics_steps'],3*(13500+11))
            for first,eighth in zip(result['rows'][::2],result['rows'][1::2]):
                self.assertEqual(first['status'],'PASSED');self.assertTrue(first['complete'])
                self.assertTrue(Path(first['actual_trial_path']).exists())
                self.assertEqual(eighth['status'],'COMPARISON_EVIDENCE_ERROR')
                self.assertEqual(eighth['new_physics_steps'],11)
                self.assertTrue(Path(eighth['actual_trial_path']).exists())


if __name__=='__main__':unittest.main()
