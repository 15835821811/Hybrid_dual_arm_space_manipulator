import copy
from dataclasses import replace
import unittest
from unittest.mock import patch

import mujoco
import numpy as np

from model_test.whole_body_verifier_v5 import WholeBodyCollisionVerifier, WholeBodyVerificationConfig
from v6_lite.run_v6_lite import default_v6_lite_robot_spec, build_scenarios, V6LiteRunConfig
from v6_lite.hierarchical_qp import free_joint_slices
from v6_4.task_protocol import task_from_scenario, canonical_json
from v6_4.reference_adapter import CartesianPassThroughReferenceProvider
from v6_4.task_anchored_reference import (
    build_reference_definition, TaskAnchoredResidualPlan,
    TaskAnchoredResidualReferenceProvider, reference_precheck,
    COEFFICIENT_NORM_BOUND_M, SUPPORT_CUTOFF_S,
)


class TaskAnchoredReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with patch.object(mujoco, 'mj_step', side_effect=AssertionError('physics forbidden')):
            cls.spec = default_v6_lite_robot_spec()
            cls.scene = build_scenarios(cls.spec, V6LiteRunConfig(scenario_count=3))[0]
            cls.task = task_from_scenario(cls.spec, cls.scene, task_id='b2_reference_unit',
                group_id='b2_reference_unit', family='end_effector_detour', split='train')
            cls.verifier = WholeBodyCollisionVerifier(cls.spec, cls.scene.obstacles,
                WholeBodyVerificationConfig(minimum_clearance=.005, query_distance_max=2.5,
                    adaptive_subdivisions=1, include_target_satellite_pairs=True))
            cls.model = cls.verifier.model
            cls.definition = build_reference_definition(cls.task)

    def data(self):
        data = mujoco.MjData(self.model)
        data.qpos[:] = self.task.initial_qpos
        data.qvel[:] = self.task.initial_qvel
        data.ctrl[:] = 0.
        mujoco.mj_forward(self.model, data)
        return data

    def plan(self, amplitude=.010):
        z = np.zeros((6, 2))
        z[np.asarray(self.definition['interval_mask']), 0] = amplitude
        return TaskAnchoredResidualPlan.from_definition(self.definition, z)

    def test_definition_is_deterministic_world_fixed_and_all_windows_protected(self):
        with patch.object(mujoco, 'MjData', side_effect=AssertionError('definition must be pure')):
            self.assertEqual(build_reference_definition(self.task), self.definition)
        self.assertEqual(self.definition['frame'], 'world')
        self.assertEqual(self.definition['time_mapping'], 'identity_physical_time')
        self.assertEqual(self.definition['base_reference_id'], self.definition['base_reference_version'])
        self.assertEqual(self.definition['support_cutoff_s'], 23.98)
        self.assertIsNone(self.definition['protected_time_contract']['additional_strict_path_intervals_s'])
        self.assertIsNone(self.definition['protected_time_contract']['additional_specified_velocity_requirements'])
        plan = self.plan()
        for point in self.task.requirements:
            grid = np.linspace(*point.time_window_s, 37)
            for offset in plan.offset_kinematics(grid):
                np.testing.assert_array_equal(offset, np.zeros((37, 3)))
        for offset in plan.offset_kinematics(np.linspace(SUPPORT_CUTOFF_S, 27., 31)):
            np.testing.assert_array_equal(offset, np.zeros((31, 3)))
        for interval, basis, active in zip(self.definition['intervals_s'],
                self.definition['transverse_bases'], self.definition['interval_mask']):
            if active:
                self.assertLessEqual(interval[1], 23.98)
                np.testing.assert_allclose(np.asarray(basis).T@basis, np.eye(2), atol=1e-14, rtol=0.)

    def test_zero_direct_passthrough_matches_all_ten_current_feedback_inputs(self):
        data = self.data()
        base = CartesianPassThroughReferenceProvider(self.task).prepare(self.spec, self.model, data, self.scene)
        provider = TaskAnchoredResidualReferenceProvider(self.task, self.plan(0.)).prepare(
            self.spec, self.model, data, self.scene)
        target_pos, target_vel = free_joint_slices(self.model, self.spec.target_free_joint_name)
        with patch.object(mujoco, 'mj_step', side_effect=AssertionError('physics forbidden')):
            for time in (0., 3., 8.08, 24.48, 27.):
                data.time = time
                # Private feedback snapshots, not execution or a target forecast.
                data.qpos[target_pos.start:target_pos.start+3] += [.002, -.001, .001]
                data.qvel[target_vel.start:target_vel.start+6] += [.001, 0., 0., 0., 0., .0001]
                mujoco.mj_forward(self.model, data)
                before = {key: np.asarray(getattr(data, key)).tobytes()
                          for key in ('qpos', 'qvel', 'ctrl', 'xpos', 'xmat')}
                expected, actual = base.sample(time), provider.sample(time)
                self.assertEqual(set(actual), set(expected))
                self.assertEqual(len(actual), 10)
                for key in expected:
                    self.assertEqual(np.asarray(actual[key]).tobytes(), np.asarray(expected[key]).tobytes())
                for key in before:
                    self.assertEqual(before[key], np.asarray(getattr(data, key)).tobytes())
                self.assertEqual(data.time, time)
        with self.assertRaises(ValueError):
            provider.sample(26.98)

    def test_nonzero_changes_consumed_position_velocity_only(self):
        data = self.data()
        base = CartesianPassThroughReferenceProvider(self.task).prepare(self.spec, self.model, data, self.scene)
        plan = self.plan()
        provider = TaskAnchoredResidualReferenceProvider(self.task, plan).prepare(self.spec, self.model, data, self.scene)
        index = np.flatnonzero(plan.interval_mask)[0]
        lower, upper = self.definition['intervals_s'][index]
        with patch.object(mujoco, 'mj_step', side_effect=AssertionError('physics forbidden')):
            for u in (.25, .5):
                time = lower+u*(upper-lower)
                data.time = time
                expected, actual = base.sample(time), provider.sample(time)
                dp, dv, _ = plan.offset_kinematics(time)
                np.testing.assert_array_equal(actual['continuum_target_position'], expected['continuum_target_position']+dp)
                np.testing.assert_array_equal(actual['continuum_target_velocity'], expected['continuum_target_velocity']+dv)
                self.assertGreater(np.linalg.norm(dp), .004)
                for key in set(expected)-{'continuum_target_position', 'continuum_target_velocity'}:
                    np.testing.assert_array_equal(actual[key], expected[key])
                if u == .25:
                    self.assertGreater(np.linalg.norm(dv), 0.)
                if u == .5:
                    self.assertAlmostEqual(np.linalg.norm(dp), .010, places=14)

    def test_phi_C2_endpoints_and_analytic_derivatives(self):
        plan = self.plan()
        provider = TaskAnchoredResidualReferenceProvider(self.task, plan)
        for (lower, upper), active in zip(self.definition['intervals_s'], plan.interval_mask):
            if not active:
                continue
            for time in (lower, upper):
                for offset in plan.offset_kinematics(time):
                    np.testing.assert_array_equal(offset, np.zeros(3))
                for eps in (1e-4, 1e-6):
                    toward = time+(eps if time == lower else -eps)
                    dp, dv, da = plan.offset_kinematics(toward)
                    self.assertLess(np.linalg.norm(dp), 1e-10)
                    self.assertLess(np.linalg.norm(dv), 1e-6)
                    self.assertLess(np.linalg.norm(da), 1e-3)
            for u in (.19, .37, .61, .83):
                time = lower+u*(upper-lower)
                h = (upper-lower)*1e-6
                dp, dv, da = plan.offset_kinematics(time)
                left, right = plan.offset_kinematics(time-h), plan.offset_kinematics(time+h)
                np.testing.assert_allclose((right[0]-left[0])/(2*h), dv, atol=1e-9, rtol=0.)
                np.testing.assert_allclose((right[1]-left[1])/(2*h), da, atol=1e-9, rtol=0.)
                left, right = provider.continuum_kinematics(time-h), provider.continuum_kinematics(time+h)
                p, v, a = provider.continuum_kinematics(time)
                np.testing.assert_allclose((right[0]-left[0])/(2*h), v, atol=1e-8, rtol=0.)
                np.testing.assert_allclose((right[1]-left[1])/(2*h), a, atol=1e-8, rtol=0.)

    def test_random_legal_coefficients_preserve_anchors_holds_and_bound(self):
        rng = np.random.default_rng(20261007)
        for _ in range(12):
            angle = rng.uniform(-np.pi, np.pi, size=6)
            radius = rng.uniform(0., COEFFICIENT_NORM_BOUND_M, size=6)
            z = np.c_[radius*np.cos(angle), radius*np.sin(angle)]
            z[~np.asarray(self.definition['interval_mask'])] = 0.
            plan = TaskAnchoredResidualPlan.from_definition(self.definition, z)
            for point in self.task.requirements:
                for offset in plan.offset_kinematics(np.linspace(*point.time_window_s, 13)):
                    np.testing.assert_array_equal(offset, np.zeros((13, 3)))
            for (a,b), active in zip(self.definition['intervals_s'], plan.interval_mask):
                if active:
                    p, _, _ = plan.offset_kinematics(np.linspace(a,b,101))
                    self.assertLessEqual(np.linalg.norm(p, axis=1).max(), .020+1e-15)

    def test_invalid_coefficients_mask_identity_and_plan_roundtrip(self):
        wide = replace(self.task.requirements[0], time_window_s=(0., 23.))
        task = replace(self.task, requirements=(wide,)+self.task.requirements[1:])
        definition = build_reference_definition(task)
        self.assertTrue(definition['applicable'])
        self.assertFalse(all(definition['interval_mask']))
        good = np.zeros((6,2))
        good[np.asarray(definition['interval_mask']), 1] = .010
        plan = TaskAnchoredResidualPlan.from_definition(definition, good)
        self.assertEqual(TaskAnchoredResidualPlan.from_dict(plan.to_dict()).sha256(), plan.sha256())
        before = canonical_json(plan.to_dict())
        copy_definition = plan.definition
        copy_definition['frame'] = 'changed'
        self.assertEqual(before, canonical_json(plan.to_dict()))
        with self.assertRaises(ValueError):
            plan.z_m[0,0] = 1.
        for invalid in (np.full((6,2), np.nan), np.ones((6,2)), np.zeros((12,)),
                        np.full((6,2), .001j), np.full((6,2), '0.001')):
            with self.assertRaises(ValueError):
                TaskAnchoredResidualPlan.from_definition(definition, invalid)
        inactive = np.flatnonzero(~np.asarray(definition['interval_mask']))[0]
        illegal = good.copy()
        illegal[inactive,0] = 1e-20
        with self.assertRaisesRegex(ValueError, 'mask-out'):
            TaskAnchoredResidualPlan.from_definition(definition, illegal)
        malformed = copy.deepcopy(definition)
        malformed['intervals_s'][0][1] += .1
        with self.assertRaises(ValueError):
            TaskAnchoredResidualPlan.from_definition(malformed, good)
        with self.assertRaises(ValueError):
            TaskAnchoredResidualReferenceProvider(self.task, plan)

    def test_no_intervals_strict_policy_and_degenerate_chords_are_not_applicable(self):
        strict = replace(self.task, path_freedom='strict_full_time_path')
        no_gaps = replace(self.task, requirements=(replace(self.task.requirements[0],
            time_window_s=(0.,27.)),)+self.task.requirements[1:])
        source = self.task.scenario
        initial = source['continuum_target']['initial_position_w']
        source['continuum_target']['waypoint_points_m'] = [initial]*len(source['continuum_target']['waypoint_points_m'])
        degenerate = replace(self.task, scenario_json=canonical_json(source))
        for task in (strict, no_gaps, degenerate):
            definition = build_reference_definition(task)
            self.assertEqual(definition['status'], 'NOT_APPLICABLE')
            self.assertFalse(any(definition['interval_mask']))
            with self.assertRaisesRegex(ValueError, 'NOT_APPLICABLE'):
                TaskAnchoredResidualPlan.from_definition(definition, np.zeros((6,2)))
        self.assertEqual(len(build_reference_definition(degenerate)['disabled_intervals']), 6)
        # A base hold beginning before the standard freeze also stays protected.
        source = self.task.scenario
        factor = 17.5/source['continuum_target']['path_duration_s']
        source['continuum_target']['path_duration_s'] = 17.5
        source['continuum_target']['segment_durations_s'] = [factor*t for t in source['continuum_target']['segment_durations_s']]
        shortened = replace(self.task, scenario_json=canonical_json(source))
        definition = build_reference_definition(shortened)
        self.assertTrue(any(a <= 22. and b >= 27. for a,b in definition['protected_time_intervals_s']))

    def test_precheck_is_reference_only_and_rejects_speed_limit_violation(self):
        with patch.object(mujoco, 'mj_step', side_effect=AssertionError('physics forbidden')):
            result = reference_precheck(self.task, self.plan())
            self.assertTrue(result['passed'], result)
            self.assertTrue(result['reference_task_passed'])
            self.assertTrue(result['nonzero_reference'])
            self.assertAlmostEqual(result['reference_offset_analytic_peak_m'], .010)
            self.assertEqual(result['whole_body_geometry'], 'NOT_RUN')
            self.assertEqual(result['nominal_joint_reference_geometry'], 'N/A_NO_JOINT_REFERENCE')
            self.assertEqual(result['physics_steps'], 0)
            self.assertEqual(result['native_distance_queries'], 0)
            self.assertGreaterEqual(result['reference_speed_upper_bound_m_s'], result['reference_speed_sampled_max_m_s'])
            rejected = reference_precheck(self.task, self.plan(), cartesian_speed_limit_m_s=.001)
            self.assertFalse(rejected['passed'])
            self.assertFalse(rejected['reference_velocity_passed'])


if __name__ == '__main__':
    unittest.main()
