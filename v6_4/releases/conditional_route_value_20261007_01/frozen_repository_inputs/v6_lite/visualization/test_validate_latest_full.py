"""Small mocked-media tests of the independent visualization publication gates."""
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from v6_lite.visualization import validate_latest_full as validator


class LatestVisualizationGateTests(unittest.TestCase):
    def fixture(self, root):
        out = root / 'v6_lite/visualization/latest'
        out.mkdir(parents=True)
        for name in ('generation_plan.json', 'studies_metadata.json', 'nominal_metadata.json', 'result_data.json'):
            (out / name).write_text('{}\n', encoding='utf-8')
        (out / 'index.html').write_text('<html><body>Latest evidence</body></html>', encoding='utf-8')
        ramp = np.tile(np.arange(320, dtype=np.uint16) % 256, (180, 1)).astype(np.uint8)
        rgb = np.stack((ramp, np.roll(ramp, 40, axis=1), np.roll(ramp, 80, axis=1)), axis=2)
        Image.fromarray(rgb).save(out / 'plot.png')
        for name in ('error_curves.png', 'safety_clearance_summary.png', 'full_control_timing.png',
                     'execution_contract.png', 'actual_actions.png'):
            Image.fromarray(rgb).save(out / name)
        Image.fromarray(rgb).save(out / 'base_pose_drift.gif', save_all=True,
                                  append_images=[Image.fromarray(np.roll(rgb, 30, axis=1))], duration=100, loop=0)
        rows, sources = [], []
        generators = []
        for name in ('generate_latest_full.py', 'latest_nominal.py', 'latest_saved_renderer.py', 'latest_studies.py'):
            path = root / 'v6_lite/visualization' / name
            path.write_bytes(b'mock generator code')
            generators.append(self.record(root, path))
        for scene in sorted(validator.SCENES):
            source = root / f'v6_lite/output/runs/research_acceptance_01/simulation/traces/{scene}.npz'
            source.parent.mkdir(parents=True, exist_ok=True)
            source.write_bytes(('mock trace ' + scene).encode())
            sources.append(self.record(root, source))
            Image.fromarray(rgb).save(out / f'{scene}_tracking_paths_3d.png')
            for name in ('distance_comparison.png', 'gradient_comparison.png', 'minimum_clearance_comparison.png'):
                destination = out / 'pcc_monitor' / scene / 'plots' / name
                original = root / 'v6_lite/output/runs/research_acceptance_01/simulation/pcc_monitor' / scene / 'plots' / name
                destination.parent.mkdir(parents=True, exist_ok=True)
                original.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(rgb).save(original)
                destination.write_bytes(original.read_bytes())
                sources.append(self.record(root, original))
            trace_record = self.record(root, source)
            files = []
            for role in ('overview', 'front', 'side', 'top', 'iso', 'five_view_grid', 'continuum_focus'):
                video = out / f'{scene}_{role}.mp4'
                video.write_bytes(('mock compressed video ' + scene + role).encode())
                files.append(video.relative_to(root).as_posix())
            rows.append({'scenario_id': scene, 'frame_count': 811, 'fps': 30, 'duration_s': 27,
                         'source_trace': source.relative_to(root).as_posix(),
                         'source_trace_sha256': trace_record['sha256'], 'files': files})
        manifest = {'schema': 'v6_2_latest_visualization_bundle_v1',
                    'artifacts': [self.record(root, p) for p in sorted(out.rglob('*')) if p.is_file()],
                    'source_inputs': sources, 'generator_source': generators,
                    'inputs': {}, 'nominal': {'videos': rows}, 'complete': True, 'physics_recomputed': False,
                    'coverage': {'nominal_scenarios': 5, 'nominal_duration_s': 27, 'video_count': 35,
                                 'views': ['overview', 'front', 'side', 'top', 'iso'],
                                 'research_categories': sorted(validator.CATEGORIES),
                                 'all_pressure_scenarios_included': True, 'all_inertia_runs_included': 15,
                                 'old_b2_used_as_current_source': False},
                    'claim_scope': {'research_acceptance': 'PASSED', 'functional_checks': '25/25',
                                    'execution_checks': '11/11', 'wall_deployment': 'NOT_MET',
                                    'hardware': 'NOT_ESTABLISHED', 'shadow_closed_loop_robustness': False,
                                    'velocity_stress': 'FAILED_INCOMPLETE', 'hard_real_time': False,
                                    'continuous_time_collision_guarantee': False,
                                    'visualization_is_new_functional_acceptance': False}}
        self.save(out, manifest)
        frames = np.stack((rgb, np.roll(rgb, 20, axis=1), np.roll(rgb, 40, axis=1)))
        return out, manifest, frames

    @staticmethod
    def record(root, path):
        raw = path.read_bytes()
        return {'path': path.relative_to(root).as_posix(), 'bytes': len(raw),
                'sha256': hashlib.sha256(raw).hexdigest()}

    @staticmethod
    def save(out, manifest):
        (out / 'visualization_manifest.json').write_text(json.dumps(manifest), encoding='utf-8')

    def run_fixture(self, root, out, frames):
        # Full original-source verification is deliberately separate from media
        # gate tests; the real validator never bypasses those production gates.
        with ExitStack() as stack:
            stack.enter_context(patch.object(validator, 'ROOT', root))
            stack.enter_context(patch.object(validator, '_verify_source_pins', return_value={}))
            stack.enter_context(patch.object(validator, '_verify_failed_stress'))
            stack.enter_context(patch.object(validator, '_probe_video', return_value={
                'width': 320, 'height': 180, 'frame_count': 811, 'fps': 30, 'duration_s': 811 / 30}))
            decode = stack.enter_context(patch.object(validator, '_decode_frames', return_value=frames))
            result = validator.validate(out, write=False)
            return result, decode.call_count

    def test_full_mocked_bundle_decodes_all_35_and_preserves_claim_limits(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            out, _, frames = self.fixture(root)
            result, calls = self.run_fixture(root, out, frames)
            self.assertTrue(result['passed'], result['error'])
            self.assertEqual(calls, 35)
            self.assertEqual(len(result['videos']), 35)
            self.assertFalse(result['scope']['controller_acceptance_relabelled'])

    def test_tampered_artifact_rejects_before_any_media_decode(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            out, _, frames = self.fixture(root)
            (out / 'plot.png').write_bytes(b'changed artifact')
            result, calls = self.run_fixture(root, out, frames)
            self.assertFalse(result['passed'])
            self.assertEqual(calls, 0)
            self.assertTrue(any(not c['passed'] and c['name'].startswith('artifact:') for c in result['checks']))

    def test_partial_duration_missing_view_and_false_stress_claim_never_pass(self):
        for kind in ('partial', 'missing_view', 'missing_pcc', 'false_stress_claim'):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                out, manifest, frames = self.fixture(root)
                if kind == 'partial':
                    manifest['nominal']['videos'][0]['frame_count'] = 810
                elif kind == 'missing_view':
                    manifest['nominal']['videos'][0]['files'].pop()
                elif kind == 'missing_pcc':
                    path = out / 'pcc_monitor/v6_lite_scenario_04/plots/distance_comparison.png'
                    path.unlink()
                    manifest['artifacts'] = [r for r in manifest['artifacts']
                                             if r['path'] != path.relative_to(root).as_posix()]
                else:
                    manifest['claim_scope']['velocity_stress'] = 'PASSED'
                self.save(out, manifest)
                result, _ = self.run_fixture(root, out, frames)
                self.assertFalse(result['passed'])
                expected = {'partial': ':declared_full_source_time_grid', 'missing_view': ':seven_complete_unique_views',
                            'missing_pcc': 'fifteen_nominal_pcc_plots_exact',
                            'false_stress_claim': 'original_acceptance_and_deployment_claims'}[kind]
                self.assertTrue(any(not c['passed'] and expected in c['name'] for c in result['checks']))

    def test_absolute_traversing_and_duplicate_windows_paths_are_rejected(self):
        for kind in ('absolute', 'traversal', 'duplicate_normalized'):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                out, manifest, frames = self.fixture(root)
                if kind == 'absolute':
                    manifest['artifacts'][0]['path'] = 'C:\\private\\data.json'
                elif kind == 'traversal':
                    manifest['artifacts'][0]['path'] = '../data.json'
                else:
                    duplicate = dict(manifest['artifacts'][0])
                    duplicate['path'] = duplicate['path'].replace('/', '\\')
                    manifest['artifacts'].append(duplicate)
                self.save(out, manifest)
                result, calls = self.run_fixture(root, out, frames)
                self.assertFalse(result['passed'])
                self.assertEqual(calls, 0)
                self.assertEqual(result['error']['type'], 'ValueError')


if __name__ == '__main__':
    unittest.main()
