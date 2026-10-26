"""Source-group mistakes must fail before a future trace can reach Teacher."""
import json
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from v6_4.prepare_teacher import load_declared_bootstrap
from v6_4.run_planning import load_task_suite


class BootstrapIdentityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        tasks = load_task_suite(Path(__file__).resolve().parents[1] / 'output/tasks/protocol_b_01.json')
        cls.source, cls.destination = [t for t in tasks if t.split == 'train'][:2]

    def saved_source(self, directory, task):
        root = Path(directory)
        (root / 'traces').mkdir()
        (root / 'task.json').write_text(json.dumps(task.to_dict()), encoding='utf-8')
        (root / 'result.json').write_text(json.dumps({'task_id': task.task_id,
            'task_sha256': task.sha256()}), encoding='utf-8')
        return root / 'traces/prior.npz'

    def test_different_train_group_requires_explicit_source(self):
        with TemporaryDirectory() as directory, patch('v6_4.prepare_teacher.load_bootstrap') as loader:
            prior = self.saved_source(directory, self.source)
            with self.assertRaisesRegex(ValueError, 'explicit'):
                load_declared_bootstrap(self.destination, prior)
            loader.assert_not_called()
            with self.assertRaises(FileNotFoundError):
                load_declared_bootstrap(self.destination, prior, bootstrap_task=self.source)
            loader.assert_not_called()

    def test_saved_test_trace_never_enters_bootstrap(self):
        test_source = replace(self.source, split='test')
        with TemporaryDirectory() as directory, patch('v6_4.prepare_teacher.load_bootstrap') as loader:
            prior = self.saved_source(directory, test_source)
            with self.assertRaisesRegex(ValueError, 'training TaskSpec'):
                load_declared_bootstrap(self.destination, prior, bootstrap_task=test_source)
            loader.assert_not_called()

    def test_changed_saved_result_identity_rejected(self):
        with TemporaryDirectory() as directory, patch('v6_4.prepare_teacher.load_bootstrap') as loader:
            prior = self.saved_source(directory, self.source)
            (Path(directory) / 'result.json').write_text(json.dumps({'task_id': self.source.task_id,
                'task_sha256': self.destination.sha256()}), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'training TaskSpec'):
                load_declared_bootstrap(self.destination, prior, bootstrap_task=self.source)
            loader.assert_not_called()

    def test_real_retry_prior_requires_explicit_bound_lineage(self):
        root = Path(__file__).resolve().parents[1] / 'output/teacher_bootstrap/fixed_ee_train_000_01'
        prior = root / 'traces' / f'{self.source.task_id}.npz'
        with self.assertRaisesRegex(ValueError, 'explicit source_evaluation'):
            load_declared_bootstrap(self.destination, prior, bootstrap_task=self.source)
        times, q, identity = load_declared_bootstrap(self.destination, prior, bootstrap_task=self.source,
            source_evaluation=root / 'evaluation_retry_01/report.json')
        self.assertEqual(times.shape, (13500,))
        self.assertEqual(q.shape, (13500,17))
        self.assertEqual(identity['source_group_id'], self.source.group_id)
        self.assertIn('source_evaluation_sha256', identity)

    def test_correct_task_metadata_cannot_substitute_another_real_trace(self):
        from shutil import copyfile
        root = Path(__file__).resolve().parents[1] / 'output/teacher_bootstrap'
        with TemporaryDirectory() as directory, patch('v6_4.prepare_teacher.load_bootstrap') as loader:
            prior = self.saved_source(directory, self.source)
            copyfile(root / 'fixed_ee_train_001_01/traces' / f'{self.destination.task_id}.npz', prior)
            evaluation = Path(directory) / 'evaluation'
            evaluation.mkdir()
            copyfile(root / 'fixed_ee_train_000_01/evaluation_retry_01/report.json', evaluation / 'report.json')
            with self.assertRaisesRegex(ValueError, 'raw SHA/path/task/model'):
                load_declared_bootstrap(self.destination, prior, bootstrap_task=self.source)
            loader.assert_not_called()


if __name__ == '__main__':
    unittest.main()
