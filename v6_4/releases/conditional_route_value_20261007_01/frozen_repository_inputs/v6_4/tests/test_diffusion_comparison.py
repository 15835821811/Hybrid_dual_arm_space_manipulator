"""Negative coordinator tests; temporary mock traces are never delivery data."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from v6_4.contracts import TrajectoryProposal
from v6_4.run_diffusion_comparison import _failure_cell, run_comparison


class DiffusionCoordinatorTests(unittest.TestCase):
    def tasks(self):
        return [SimpleNamespace(task_id=f'test_{i}', family=f'family_{i}', split='test',
            initial_planner_q=np.zeros(17), initial_planner_dq=np.zeros(17),
            sha256=lambda: 'a' * 64, to_dict=lambda: {'test_only': True}) for i in range(3)]

    def test_K8_partial_failure_keeps_successful_K1_and_all_families(self):
        tasks = self.tasks()
        proposals = [TrajectoryProposal.from_controls(t, np.zeros((30, 17)), origin='diffusion') for t in tasks]
        by_id = dict(zip((t.task_id for t in tasks), proposals))
        def generate(task, checkpoint, output, **kwargs):
            return [by_id[task.task_id]] * 8, {}
        def trial(task, k, selected, candidates, output, **kwargs):
            self.assertEqual(kwargs['method'], 'diffusion')
            actual = output / f'K{k}' / 'actual'
            (actual / 'traces').mkdir(parents=True)
            np.savez(actual / 'traces' / 'test_only.npz', torque=np.zeros((13500 if k == 1 else 11, 67)))
            if k == 8:
                self.assertEqual(kwargs['prior_trial']['task_id'], task.task_id)
                raise RuntimeError('K8 failed after partial physics')
            return dict(task_id=task.task_id, task_sha256=task.sha256(), candidate_count=1,
                selectedIndex=0, proposal_sha256=by_id[task.task_id].sha256(), status='PASSED',
                task_success=True, complete=True, proposal_accepted=True, raw_proposal_passed=True,
                actual_trial_path=actual.as_posix(), new_physics_steps=13500, reused_actual_trial=False)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint = root / 'test_only.pt'; checkpoint.write_bytes(b'not a delivered checkpoint')
            tasks_path = root / 'tasks.json'; tasks_path.write_text('{}')
            with patch('v6_4.run_diffusion_comparison.load_task_suite', return_value=tasks), \
                 patch('v6_4.run_diffusion_comparison.generate', side_effect=generate), \
                 patch('v6_4.run_diffusion_comparison._executed_sources', return_value={'test': 'frozen'}), \
                 patch('v6_4.run_diffusion_comparison.evaluate_candidate_set', return_value={'selectedIndexK1':0, 'selectedIndex':1}), \
                 patch('v6_4.run_diffusion_comparison._trial_row', side_effect=trial):
                report = run_comparison(tasks_path, checkpoint, root / 'comparison')
            self.assertEqual(report['by_K']['1']['task_success_count'], 3)
            self.assertEqual(report['by_K']['8']['task_success_count'], 0)
            self.assertEqual(report['new_physics_steps'], 40533)
            self.assertTrue(report['all_tasks_attempted'])
            self.assertTrue(report['physical_step_count_exact_known'])
            for first, eighth in zip(report['rows'][::2], report['rows'][1::2]):
                self.assertEqual(first['status'], 'PASSED')
                self.assertTrue(first['complete'])
                self.assertEqual(eighth['status'], 'COMPARISON_EVIDENCE_ERROR')
                self.assertEqual(eighth['new_physics_steps'], 11)
                self.assertTrue(Path(eighth['actual_trial_path']).exists())

    def test_generation_failure_keeps_24_attempts_and_six_failure_cells(self):
        tasks = self.tasks()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint = root / 'test_only.pt'; checkpoint.write_bytes(b'not weights')
            tasks_path = root / 'tasks.json'; tasks_path.write_text('{}')
            with patch('v6_4.run_diffusion_comparison.load_task_suite', return_value=tasks), \
                 patch('v6_4.run_diffusion_comparison.generate', side_effect=RuntimeError('no generation')), \
                 patch('v6_4.run_diffusion_comparison._executed_sources', return_value={'test': 'frozen'}), \
                 patch('v6_4.run_teacher_comparison.run_attempt', side_effect=AssertionError('physics forbidden')):
                report = run_comparison(tasks_path, checkpoint, root / 'comparison')
            self.assertEqual(report['candidate_attempt_count'], 24)
            self.assertEqual(len(report['rows']), 6)
            self.assertEqual(report['new_physics_steps'], 0)
            for row in report['rows']:
                self.assertFalse(row['task_success'])
                self.assertEqual(row['status'], 'NO_ADMISSIBLE_RAW_PROPOSAL')
                self.assertEqual(row['generation_failure']['type'], 'RuntimeError')
                self.assertNotIn('actual_trial_path', row)

    def test_unreadable_trace_never_reports_exact_zero_physics(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            trace = root / 'K8' / 'actual' / 'traces' / 'broken.npz'
            trace.parent.mkdir(parents=True); trace.write_bytes(b'broken trace')
            row = _failure_cell(self.tasks()[0], 8, root, RuntimeError('pipeline error'))
            self.assertIsNone(row['new_physics_steps'])
            self.assertTrue(Path(row['actual_trial_path']).exists())
            self.assertIn('physical_step_count_error', row['failure'])


if __name__ == '__main__':
    unittest.main()
