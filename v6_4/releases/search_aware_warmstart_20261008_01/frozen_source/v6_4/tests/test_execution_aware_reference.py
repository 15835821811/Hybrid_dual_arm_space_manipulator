"""B.3.1's necessary pure-reference tests; no simulator state is created."""
import copy
import importlib.util
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import mujoco
import numpy as np

from v6_4.task_protocol import TaskSpec, canonical_json
from v6_4.task_anchored_reference import (
    REPRESENTATION_VERSION, PLATEAU_REPRESENTATION_VERSION, PLAN_SCHEMA_V2,
    SUPPORT_CUTOFF_S, TaskAnchoredResidualPlan, TaskAnchoredResidualReferenceProvider,
    build_reference_definition, reference_precheck,
)


ROOT = Path(__file__).resolve().parents[2]
OLD_RELEASE = ROOT/"v6_4/releases/conditional_route_value_20261007_01"


class ExecutionAwareReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.task = TaskSpec.from_dict(json.loads((OLD_RELEASE/
            "snapshot/tasks/b3_mother_00_c_plus/task.json").read_text(encoding="utf8")))
        cls.v1 = build_reference_definition(cls.task)
        cls.v2 = build_reference_definition(cls.task, version=PLATEAU_REPRESENTATION_VERSION)
        name = "_b31_frozen_v1_reference_for_compatibility"
        spec = importlib.util.spec_from_file_location(name,
            OLD_RELEASE/"frozen_repository_inputs/v6_4/task_anchored_reference.py")
        cls.old = importlib.util.module_from_spec(spec)
        sys.modules[name] = cls.old
        spec.loader.exec_module(cls.old)

    def setUp(self):
        for name in ("mj_step", "MjData", "mj_forward", "mj_geomDistance"):
            guard = patch.object(mujoco, name, side_effect=AssertionError("pure reference tests forbid physics/geometry"))
            guard.start()
            self.addCleanup(guard.stop)

    def plan(self, amplitude=.012, version=PLATEAU_REPRESENTATION_VERSION):
        definition = self.v2 if version == PLATEAU_REPRESENTATION_VERSION else self.v1
        z = np.zeros((6, 2))
        z[2, 0] = amplitude
        return TaskAnchoredResidualPlan.from_definition(definition, z)

    def test_default_v1_serialization_hash_and_numeric_paths_are_unchanged(self):
        old_definition = self.old.build_reference_definition(self.task)
        self.assertEqual(canonical_json(self.v1), canonical_json(old_definition))
        times = np.unique(np.r_[np.linspace(0., 27., 13501), np.asarray(self.v1["intervals_s"]).ravel()])
        for amplitude in (0., .012, -.012, .020):
            plan = self.plan(amplitude, REPRESENTATION_VERSION)
            old_plan = self.old.TaskAnchoredResidualPlan.from_definition(old_definition, plan.z_m)
            self.assertEqual(canonical_json(plan.to_dict()), canonical_json(old_plan.to_dict()))
            self.assertEqual(plan.sha256(), old_plan.sha256())
            for actual, expected in zip(plan.offset_kinematics(times), old_plan.offset_kinematics(times)):
                self.assertEqual(actual.tobytes(), expected.tobytes())
            provider = TaskAnchoredResidualReferenceProvider(self.task, plan)
            old_provider = self.old.TaskAnchoredResidualReferenceProvider(self.task, old_plan)
            self.assertEqual(provider.metadata, old_provider.metadata)
            for key in provider.prediction:
                self.assertEqual(provider.prediction[key].tobytes(), old_provider.prediction[key].tobytes())
        self.assertEqual(reference_precheck(self.task, self.plan(.012, REPRESENTATION_VERSION)),
            self.old.reference_precheck(self.task, self.old.TaskAnchoredResidualPlan.from_definition(
                old_definition, self.plan(.012, REPRESENTATION_VERSION).z_m)))

    def test_zero_v1_v2_use_the_exact_old_passthrough_sample(self):
        times = np.r_[np.linspace(0., 27., 41), np.asarray(self.v1["intervals_s"]).ravel()]
        for version in (REPRESENTATION_VERSION, PLATEAU_REPRESENTATION_VERSION):
            plan = self.plan(0., version)
            provider = TaskAnchoredResidualReferenceProvider(self.task, plan)
            self.assertTrue(provider.metadata["zero_residual_direct_passthrough"])
            for output in plan.offset_kinematics(times):
                np.testing.assert_array_equal(output, np.zeros((len(times), 3)))
            sentinel = {"unmodified_old_provider_sample": object()}
            with patch.object(provider._base, "sample", return_value=sentinel) as sample:
                for time in times:
                    self.assertIs(provider.sample(float(time)), sentinel)
                    sample.assert_called_with(float(time))
            v1_provider = TaskAnchoredResidualReferenceProvider(self.task, self.plan(0., REPRESENTATION_VERSION))
            for actual, expected in zip(provider.continuum_kinematics(times), v1_provider.continuum_kinematics(times)):
                self.assertEqual(actual.tobytes(), expected.tobytes())

    def test_plateau_endpoint_and_join_position_velocity_acceleration_are_C2(self):
        plan = self.plan()
        lower, upper = self.v2["intervals_s"][2]
        duration = upper-lower
        delta = np.asarray(self.v2["transverse_bases"][2])@plan.z_m[2]
        for u, value in ((0., 0.), (.25, 1.), (.75, 1.), (1., 0.)):
            time = lower+u*duration
            p, v, a = plan.offset_kinematics(time)
            np.testing.assert_allclose(p, value*delta, atol=1e-15, rtol=0.)
            np.testing.assert_allclose(v, np.zeros(3), atol=1e-14, rtol=0.)
            np.testing.assert_allclose(a, np.zeros(3), atol=1e-13, rtol=0.)
            for sign in (-1., 1.):
                p, v, a = plan.offset_kinematics(time+sign*duration*1e-7)
                np.testing.assert_allclose(p, value*delta, atol=1e-12, rtol=0.)
                self.assertLess(np.linalg.norm(v), 1e-10)
                self.assertLess(np.linalg.norm(a), 1e-6)
        for output in plan.offset_kinematics([lower-.01, upper+.01]):
            np.testing.assert_array_equal(output, np.zeros((2, 3)))
        plateau = plan.offset_kinematics(lower+duration*np.array([.25, .31, .5, .69, .75]))
        np.testing.assert_allclose(plateau[0], np.tile(delta, (5, 1)), atol=1e-15, rtol=0.)
        np.testing.assert_allclose(plateau[1], np.zeros((5, 3)), atol=1e-14, rtol=0.)
        np.testing.assert_allclose(plateau[2], np.zeros((5, 3)), atol=1e-13, rtol=0.)

    def test_analytic_derivatives_and_unchanged_speed_gate_with_conservative_bounds(self):
        plan = self.plan(.020)
        provider = TaskAnchoredResidualReferenceProvider(self.task, plan)
        lower, upper = self.v2["intervals_s"][2]
        duration, h = upper-lower, (upper-lower)*1e-6
        for u in (.071, .161, .31, .5, .69, .831, .953):
            time = lower+u*duration
            for function in (plan.offset_kinematics, provider.continuum_kinematics):
                p, v, a = function(time)
                left, right = function(time-h), function(time+h)
                np.testing.assert_allclose((right[0]-left[0])/(2*h), v, atol=2e-9, rtol=0.)
                np.testing.assert_allclose((right[1]-left[1])/(2*h), a, atol=2e-9, rtol=0.)
        # Exact stationary points of each transition's first/second derivative.
        v_extrema = np.array([.5, (3.-np.sqrt(3.))/6., (3.+np.sqrt(3.))/6.])
        times = lower+duration*np.r_[.25*v_extrema, 1.-.25*v_extrema]
        _, velocity, acceleration = plan.offset_kinematics(times)
        self.assertAlmostEqual(np.linalg.norm(velocity, axis=1).max(), .020*7.5/duration, places=14)
        self.assertAlmostEqual(np.linalg.norm(acceleration, axis=1).max(), .020*(160./np.sqrt(3.))/duration**2, places=13)
        result = reference_precheck(self.task, plan)
        self.assertTrue(result["passed"], result)
        self.assertEqual(result["cartesian_speed_limit_m_s"], .24)
        self.assertEqual(result["representation_version"], PLATEAU_REPRESENTATION_VERSION)
        self.assertGreaterEqual(result["reference_speed_upper_bound_m_s"], result["reference_speed_sampled_max_m_s"])
        self.assertGreaterEqual(result["reference_acceleration_upper_bound_m_s2"], result["reference_acceleration_sampled_max_m_s2"])
        self.assertFalse(reference_precheck(self.task, plan, cartesian_speed_limit_m_s=.001)["passed"])
        self.assertEqual(result["physics_steps"], 0)
        self.assertEqual(result["native_distance_queries"], 0)

    def test_protection_and_unchanged_two_dimensional_norm_limit(self):
        z = np.tile([.012, .016], (6, 1))
        z[~np.asarray(self.v2["interval_mask"])] = 0.
        plan = TaskAnchoredResidualPlan.from_definition(self.v2, z)
        for point in self.task.requirements:
            for output in plan.offset_kinematics(np.linspace(*point.time_window_s, 37)):
                np.testing.assert_array_equal(output, np.zeros((37, 3)))
        for output in plan.offset_kinematics(np.linspace(SUPPORT_CUTOFF_S, 27., 37)):
            np.testing.assert_array_equal(output, np.zeros((37, 3)))
        for (lower, upper), active in zip(self.v2["intervals_s"], plan.interval_mask):
            if active:
                position, _, _ = plan.offset_kinematics(np.linspace(lower, upper, 1001))
                self.assertLessEqual(np.linalg.norm(position, axis=1).max(), .020+1e-15)
        z[2] = [.012001, .016]
        with self.assertRaisesRegex(ValueError, "two-dimensional norm"):
            TaskAnchoredResidualPlan.from_definition(self.v2, z)

    def test_consumed_position_velocity_and_version_hash_binding(self):
        plan = self.plan()
        provider = TaskAnchoredResidualReferenceProvider(self.task, plan)
        version = PLATEAU_REPRESENTATION_VERSION
        self.assertEqual(plan.representation_version, version)
        self.assertEqual(plan.to_dict()["schema"], PLAN_SCHEMA_V2)
        self.assertEqual(str(provider.prediction["reference_mode"].item()), version)
        self.assertEqual(provider.metadata["reference_mode"], version)
        serialized = json.loads(str(provider.prediction["residual_plan_json"].item()))
        self.assertEqual(serialized, plan.to_dict())
        self.assertEqual(TaskAnchoredResidualPlan.from_dict(serialized).sha256(), provider.metadata["plan_sha256"])
        self.assertNotEqual(plan.sha256(), self.plan(.012, REPRESENTATION_VERSION).sha256())
        self.assertNotEqual(self.v1["definition_sha256"], self.v2["definition_sha256"])
        self.assertEqual(self.v1["intervals_s"], self.v2["intervals_s"])
        self.assertEqual(self.v1["transverse_bases"], self.v2["transverse_bases"])
        lower, upper = self.v2["intervals_s"][2]
        for u in (.125, .5, .875):
            time = lower+u*(upper-lower)
            original = {"continuum_target_position": np.array([1., 2., 3.]),
                "continuum_target_velocity": np.array([.1, .2, .3]),
                "continuum_target_rotation": np.eye(3), "rigid_target_position": np.array([4., 5., 6.]),
                "rigid_target_velocity": np.array([.4, .5, .6])}
            with patch.object(provider._base, "sample", return_value=copy.deepcopy(original)) as sample:
                actual = provider.sample(time)
                sample.assert_called_once_with(time)
            p, v, _ = plan.offset_kinematics(time)
            np.testing.assert_array_equal(actual["continuum_target_position"], original["continuum_target_position"]+p)
            np.testing.assert_array_equal(actual["continuum_target_velocity"], original["continuum_target_velocity"]+v)
            for key in set(original)-{"continuum_target_position", "continuum_target_velocity"}:
                np.testing.assert_array_equal(actual[key], original[key])
        mismatched = plan.to_dict(); mismatched["representation_version"] = REPRESENTATION_VERSION
        with self.assertRaises(ValueError):
            TaskAnchoredResidualPlan.from_dict(mismatched)
        malformed = plan.to_dict(); malformed["definition"]["basis_function"]["rise_ratio"] = .3
        with self.assertRaises(ValueError):
            TaskAnchoredResidualPlan.from_dict(malformed)
        with self.assertRaises(ValueError):
            build_reference_definition(self.task, version="v2_unknown")


if __name__ == "__main__":
    unittest.main()
