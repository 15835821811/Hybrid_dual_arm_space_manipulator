"""Budget/slot regressions with temporary mock evidence, no simulation/training."""
from contextlib import contextmanager, ExitStack
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from v6_4 import task_anchored_residual as orchestration


class MockPlan:
    @classmethod
    def from_definition(cls, definition, z_m):
        values = np.asarray(z_m, dtype=float)
        mask = np.asarray(definition['interval_mask'], dtype=bool)
        if values.shape != (6,2) or not np.isfinite(values).all():
            raise ValueError('invalid generated residual shape/finite values')
        if np.any(values[~mask] != 0.) or np.any(np.linalg.norm(values, axis=1) > .020):
            raise ValueError('invalid generated mask or amplitude')
        return SimpleNamespace(to_dict=lambda: {'definition': definition, 'z_m': values.tolist()})


class ResidualOrchestrationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.output = Path(self.temporary.name)
        self.seeds = [2026100711, 2026100712, 2026100713, 2026100714]
        self.mask = [True, False, True, True, False, True]
        self.protocol = {'budget': {'total_actual_max': 37}, 'test_records': []}
        for i in range(4):
            tid = f'mock_TEST_{i}'
            task_path, definition_path = f'tasks/{tid}.json', f'definitions/{tid}.json'
            self.save(task_path, {'task_id': tid})
            self.save(definition_path, {'task_id': tid, 'interval_mask': self.mask})
            self.protocol['test_records'].append({'task_id': tid, 'task_path': task_path,
                'definition_path': definition_path, 'E2_sample_seeds': list(self.seeds)})
        for name in ('source_identity.json', 'dataset/manifest.json',
                     'training/training_config.json', 'training/condition_normalizer.json'):
            self.save(name, {'mock_identity': name})
        checkpoint = self.output/'training/model/selected.pt'
        checkpoint.parent.mkdir(parents=True)
        checkpoint.write_bytes(b'mock checkpoint; never deserialized')
        self.physics = patch('mujoco.mj_step', side_effect=AssertionError('physics forbidden')).start()
        self.optimizer = patch('torch.optim.AdamW.step', side_effect=AssertionError('optimizer forbidden')).start()
        self.addCleanup(patch.stopall)

    def tearDown(self):
        self.physics.assert_not_called()
        self.optimizer.assert_not_called()

    def save(self, relative, value):
        path = self.output/relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, allow_nan=False), encoding='utf8')

    def z(self):
        z = np.zeros((6,2))
        z[np.asarray(self.mask), 0] = .010
        return z

    @contextmanager
    def mock_sampling(self, side_effect=None):
        sampler = Mock()
        sampler.sample.side_effect = side_effect
        if side_effect is None:
            sampler.sample.return_value = (self.z(), {'mock': True})
        precheck = {'passed': True, 'reference_task_passed': True,
            'reference_velocity_passed': True, 'nonzero_reference': True,
            'reference_offset_analytic_peak_m': .010}
        with ExitStack() as stack:
            stack.enter_context(patch('v6_4.residual_dataset.load_residual_dataset', return_value=object()))
            stack.enter_context(patch('v6_4.residual_dataset.ConditionNormalizer.from_dict', return_value=object()))
            stack.enter_context(patch('v6_4.residual_dataset.retrieve_train_residual',
                side_effect=lambda *args: (self.z(), {'method': 'E1', 'z_modified': False})))
            stack.enter_context(patch('v6_4.residual_diffusion.load_residual_sampler', return_value=sampler))
            stack.enter_context(patch.object(orchestration.TaskSpec, 'from_dict',
                side_effect=lambda value: SimpleNamespace(task_id=value['task_id'])))
            stack.enter_context(patch.object(orchestration, 'TaskAnchoredResidualPlan', MockPlan))
            stack.enter_context(patch.object(orchestration, 'reference_precheck', return_value=precheck))
            yield sampler

    def test_nan_and_generation_errors_consume_exactly_sixteen_slots_without_fallback(self):
        def generate(task, definition, noise):
            call = generate.calls
            generate.calls += 1
            if call == 0:
                return np.full((6,2), np.nan), {'mock': 'NaN'}
            if call == 1:
                raise RuntimeError('mock generator failure')
            if call == 2:
                return np.full(12, np.nan), {'mock': 'wrong shape and NaN'}
            return self.z(), {'mock': True}
        generate.calls = 0
        with self.mock_sampling(generate) as sampler:
            manifest = orchestration.sample_candidates(self.output, self.protocol, 'cpu')
            self.assertEqual(sampler.sample.call_count, 16)
            slots = [s for r in manifest['records'] for s in r['E2']]
            self.assertEqual(len(slots), 16)
            self.assertEqual(sum(s['finite_format_amplitude_passed'] for s in slots), 13)
            self.assertEqual(manifest['K1_slot'], 0)
            self.assertEqual(manifest['K4_actual'], 'NOT_RUN')
            for r in manifest['records']:
                self.assertEqual([s['slot'] for s in r['E2']], [0,1,2,3])
                self.assertEqual([s['seed'] for s in r['E2']], self.seeds)
            for index, slot in enumerate(slots):
                directory = (self.output/slot['plan_path']).parent
                for name in ('started.json', 'raw.json', 'plan.json', 'reference_precheck.json'):
                    self.assertTrue((directory/name).exists())
                raw = orchestration.read(directory/'raw.json')
                self.assertFalse(raw['inference_repair'])
                noise = np.asarray(raw['initial_noise'])
                np.testing.assert_array_equal(noise[~np.repeat(self.mask,2)], 0.)
                if index < 3:
                    self.assertFalse(slot['precheck']['passed'])
                    self.assertEqual(orchestration.read(directory/'plan.json')['schema'], 'v64_b2_rejected_raw_residual_v1')
                    self.assertIn('generation_or_format_failure', raw)
                if index == 1:
                    self.assertIsNone(raw['raw_z_m'])
            # Explicit retained reuse neither regenerates failures nor samples replacements.
            self.assertEqual(orchestration.sample_candidates(self.output, self.protocol, 'cpu'), manifest)
            self.assertEqual(sampler.sample.call_count, 16)

    def test_retained_manifest_reordering_cannot_change_K1(self):
        with self.mock_sampling() as sampler:
            manifest = orchestration.sample_candidates(self.output, self.protocol, 'cpu')
            self.assertEqual(sampler.sample.call_count, 16)
            slots = manifest['records'][0]['E2']
            slots[0], slots[1] = slots[1], slots[0]
            self.save('test_candidate_manifest.json', manifest)
            with self.assertRaisesRegex(ValueError, 'slot/seed binding'):
                orchestration.sample_candidates(self.output, self.protocol, 'cpu')
            self.assertEqual(sampler.sample.call_count, 16)

    def test_retained_candidates_reject_changed_checkpoint(self):
        with self.mock_sampling() as sampler:
            orchestration.sample_candidates(self.output, self.protocol, 'cpu')
            (self.output/'training/model/selected.pt').write_bytes(b'different mock checkpoint')
            with self.assertRaisesRegex(ValueError, 'production identity'):
                orchestration.sample_candidates(self.output, self.protocol, 'cpu')
            self.assertEqual(sampler.sample.call_count, 16)

    def attempt(self, slot, task_id='mock_task'):
        value = {'slot_id': slot, 'task_id': task_id, 'status': 'ZERO_STEP_EXECUTION_REFUSAL',
            'actual_runner_started': True, 'entered_actual': False, 'actual_steps': 0,
            'full_task_success': False, 'evaluation': None}
        self.save(f'attempts/{slot}/attempt_result.json', value)

    def test_budget_allows_exact_declared_thirty_seven_slots_and_rejects_extra(self):
        slots = ['zero_interface']+[f'teacher_{i:02d}' for i in range(24)]
        slots += [f'TEST_{i:02d}_{m}' for i in range(4) for m in ('E0','E1','E2')]
        for slot in slots:
            self.attempt(slot)
        ledger = orchestration.validate_budget(self.output, self.protocol)
        self.assertEqual(ledger['terminal_slots'], 37)
        self.assertEqual(ledger['actual_runner_started'], 37)
        self.assertEqual(ledger['nonzero_step_actual_attempts'], 0)
        self.assertEqual(ledger['actual_physics_steps'], 0)
        self.assertEqual(ledger['replacements_or_extra_actual'], 0)
        self.attempt('teacher_24')
        with self.assertRaisesRegex(ValueError, 'undeclared actual slot'):
            orchestration.validate_budget(self.output, self.protocol)

    def test_final_table_counts_zero_step_runner_attempt_separately(self):
        self.attempt('zero_interface')
        self.attempt('TEST_00_E2', 'mock_TEST_0')
        self.save('teacher_manifest.json', {'records': []})
        self.save('dataset/manifest.json', {'data_status': 'DATA_LIMITED',
            'counts': {'train': {'references': 0, 'tasks': 0}, 'val': {'references': 0, 'tasks': 0}}})
        self.save('training/training_status.json', {'status': 'TRAINING_NOT_RUN'})
        self.save('plan.json', {'mock': 'only temporary report fixture'})
        summary = orchestration.finalize(self.output, self.protocol, reason='mock no successful labels')
        e2 = summary['table_B'][2]
        self.assertEqual(e2['denominator'], 4)
        self.assertEqual(e2['actual_attempt_count'], 1)
        self.assertEqual(e2['nonzero_step_actual_attempts'], 0)
        self.assertEqual(e2['full_task_success_count'], 0)
        self.assertEqual(e2['not_run_count'], 3)
        self.assertFalse(summary['verdict']['training_executed'])
        self.assertFalse(summary['verdict']['research_delivery_complete'])


if __name__ == '__main__':
    unittest.main()
