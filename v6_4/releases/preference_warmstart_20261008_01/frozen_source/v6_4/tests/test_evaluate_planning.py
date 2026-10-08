from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import tempfile
from unittest.mock import patch
import unittest

import numpy as np

from v6_4.evaluate_planning import evaluate_trial,summarize_trials,_execution_trace_checks
from v6_4.proposal_gate import requirement_results
from v6_4.task_protocol import TaskPoint
from v6_4.tests.test_task_protocol import fixture_task


def fresh_state(task,n=10):
    result={'time':np.arange(n+1)*.002,'q':np.zeros((n+1,17)),
            'qpos':np.tile(task.initial_qpos,(n+1,1)),'qvel':np.tile(task.initial_qvel,(n+1,1)),
            'base_pose':np.tile(task.base_pose,(n+1,1))}
    for name in ('rigid','continuum','target'):
        result[name+'_position']=np.zeros((n+1,3));result[name+'_rotation']=np.tile(np.eye(3),(n+1,1,1))
    return result


def tiny_trace(task):
    n=10
    trace={'torque':np.zeros((n,67)),'time':np.arange(1,n+1)*.002,
           'task_full_latency':np.array([.03]),'torque_latency':np.full(n,.003),
           'initial_qpos':np.array(task.initial_qpos),'initial_qvel':np.array(task.initial_qvel),
           'task_qpos':np.tile(task.initial_qpos,(2,1)), 'planner_q':np.zeros((n,17)),
           'command_velocity':np.zeros((n,17)),'task_selected_command':np.zeros((1,17)),
           'task_failure_reason':np.array(['none']),'task_execution_mode':np.array(['TRACK'])}
    for key in ('reference_q','reference_velocity','feedforward_acceleration','measured_planner_q_before_servo','measured_velocity_before_servo'):
        trace[key]=np.zeros((n,17))
    for key in ('reference_joint_limit_clip_count','reference_measured_window_clip_count','acceleration_clip_count','torque_saturation_count','reference_velocity_error_norm_rad_s'):
        trace[key]=np.zeros(n)
    return trace


class PlanningEvaluationTests(unittest.TestCase):
    def test_unconsumed_refusal_command_tail_does_not_invalidate_consumed_ramp(self):
        task=fixture_task();trace=tiny_trace(task)
        trace['task_selected_command']=np.vstack((trace['task_selected_command'],np.full((1,17),np.nan)))
        trace['task_failure_reason']=np.array(['none','iteration_limit'])
        trace['task_execution_mode']=np.array(['TRACK','UNCERTIFIED'])
        spec=SimpleNamespace(planner_lower=np.full(17,-np.pi),
            planner_upper=np.full(17,np.pi),torque_limits=np.full(67,4.))
        result=_execution_trace_checks(task,trace,spec,None)
        self.assertTrue(result['passed'])
        self.assertTrue(np.all(np.isnan(trace['task_selected_command'][-1])))

    def test_consumed_command_nan_short_and_wrong_shape_still_fail(self):
        task=fixture_task();spec=SimpleNamespace(planner_lower=np.full(17,-np.pi),
            planner_upper=np.full(17,np.pi),torque_limits=np.full(67,4.))
        for bad in (np.full((1,17),np.nan),np.empty((0,17)),np.zeros((1,16))):
            with self.subTest(shape=bad.shape),self.assertRaises(ValueError):
                trace=tiny_trace(task);trace['task_selected_command']=bad
                _execution_trace_checks(task,trace,spec,None)

    def test_point_position_and_rotation_must_coincide_terminal_cannot_be_early(self):
        task=fixture_task();state=fresh_state(task,1);state['time']=np.array([26.,27.])
        point=TaskPoint('via','continuum','world',(0.,0.,0.),tuple(np.eye(3).flat),26.5,(26.,27.),.01,.1)
        task=replace(task,requirements=(point,*task.requirements[1:]))
        state['continuum_rotation'][0]=np.diag([1.,-1.,-1.]);state['continuum_position'][1,0]=1.
        result=requirement_results(task,state)
        self.assertFalse(result['requirements'][0]['passed'])
        self.assertFalse(result['requirements'][1]['passed'])

    def test_partial_replay_even_with_good_safety_cannot_be_full_task_success(self):
        task=fixture_task();trace=tiny_trace(task);spec=SimpleNamespace(
            planner_lower=np.full(17,-np.pi),planner_upper=np.full(17,np.pi),torque_limits=np.full(67,4.))
        def replay(t,tr,cfg,out):
            (out/'fresh_replay.npz').write_bytes(b'mock fresh state')
            return fresh_state(t),spec,{'passed':True},{'passed':True}
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'trace.npz';np.savez(path,**trace)
            with patch('v6_4.evaluate_planning._replay',side_effect=replay):
                result=evaluate_trial(task,path,None,scenario_result={'passed':True},qp_config=None,output_dir=Path(temp)/'audit')
            # Successful frozen-control metadata must also match the reference
            # that the runtime consumed, not just a safe same-task replay.
            trace.update(generated_reference_time_s=trace['time'].copy(),
                generated_reference_q=np.zeros((10,17)),generated_reference_dq=np.zeros((10,17)),
                task_time=np.array([0.]),task_generated_reference_q=np.zeros((1,17)),
                task_generated_reference_dq=np.zeros((1,17)),actual_generated_q_error_norm=np.zeros(10),
                actual_full_qpos=np.tile(task.initial_qpos,(10,1)),actual_full_qvel=np.tile(task.initial_qvel,(10,1)))
            reference=Path(temp)/'reference.npz';np.savez(reference,control_points=np.zeros((32,17)))
            np.savez(path,**trace)
            with patch('v6_4.evaluate_planning._replay',side_effect=replay):
                bound=evaluate_trial(task,path,reference,scenario_result={'passed':True},qp_config=None,output_dir=Path(temp)/'bound')
            self.assertTrue(bound['evidence_valid'],bound['errors'])
            trace['generated_reference_q'][5,0]=.2;np.savez(path,**trace)
            with patch('v6_4.evaluate_planning._replay',side_effect=replay):
                mismatch=evaluate_trial(task,path,reference,scenario_result={'passed':True},qp_config=None,output_dir=Path(temp)/'mismatch')
            self.assertFalse(mismatch['evidence_valid'])
        self.assertTrue(result['evidence_valid'],result['errors'])
        self.assertFalse(result['complete']);self.assertFalse(result['task_success'])
        self.assertTrue(result['execution_contract']['passed'])

    def test_nonfinite_record_keeps_failure_and_fallback_stays_separate(self):
        task=fixture_task();trace=tiny_trace(task);trace['torque'][0,0]=np.nan
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'trace.npz';np.savez(path,**trace);out=Path(temp)/'audit'
            with patch('v6_4.evaluate_planning._replay') as replay:
                result=evaluate_trial(task,path,None,scenario_result={},qp_config=None,output_dir=out)
            self.assertFalse(result['evidence_valid']);replay.assert_not_called()
            original=(out/'report.json').read_bytes()
            with self.assertRaises(FileExistsError):evaluate_trial(task,path,None,scenario_result={},qp_config=None,output_dir=out)
            self.assertEqual(original,(out/'report.json').read_bytes())
        rows=[{'method':'diffusion','task_id':'a','candidate_count':1,'raw_proposal_passed':False,
               'proposal_accepted':True,'fallback_used':True,'evaluation':{'task_success':True}}]
        summary=summarize_trials(rows);entry=next(iter(summary.values()))
        self.assertEqual(entry['raw_proposal_pass_count'],0);self.assertEqual(entry['fallback_success_count'],1)


if __name__=='__main__':unittest.main()
