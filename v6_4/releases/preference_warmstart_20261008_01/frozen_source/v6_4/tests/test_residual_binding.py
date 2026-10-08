"""Exercise the real evidence binder and reuse guards without physics.

The model/providers below isolate binding from robot compilation. The actual
Cartesian representation and provider have separate analytic/robot tests.
References depend on feedback state, physical time and a nonzero residual, so
these fixtures detect time shifts, stale QP inputs and malformed consumed logs.
"""
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from v6_4 import residual_execution as execution
from v6_4.tests.test_task_protocol import fixture_task


def observed_data():
    return SimpleNamespace(qpos=np.zeros(81), qvel=np.zeros(79), time=0.)


class FeedbackBase:
    def __init__(self, task):
        self.observed = None

    def prepare(self, spec, model, observed, scene):
        self.observed = observed
        return self

    def sample(self, time_s):
        if abs(self.observed.time-time_s) > 1e-12:
            raise ValueError("test provider sampled at a different feedback time")
        angle = self.observed.qpos[3]*.05+time_s*.01
        c, s = np.cos(angle), np.sin(angle)
        return {
            "rigid_target_position": self.observed.qpos[:3].copy()+np.array([.1, -.05, .02])*time_s,
            "rigid_target_velocity": self.observed.qvel[:3].copy()+np.array([.1, -.05, .02]),
            "rigid_target_rotation": np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]]),
            "rigid_target_angular_velocity": np.array([0., 0., self.observed.qvel[3]*.05+.01]),
            "continuum_target_position": np.array([.02*time_s**2, .01*time_s, .3]),
            "continuum_target_velocity": np.array([.04*time_s, .01, 0.]),
            "continuum_target_rotation": np.eye(3),
            "continuum_target_angular_velocity": np.zeros(3),
            "posture_reference_q": np.arange(17, dtype=float)*.01,
            "posture_reference_dq": np.zeros(17),
        }


class ResidualFeedback(FeedbackBase):
    def __init__(self, task, plan):
        super().__init__(task)
        self.amplitude = float(plan.z_m[0, 0])

    def sample(self, time_s):
        value = super().sample(time_s)
        # C2 zero endpoints over the tiny evidence fixture. At its second QP
        # time (20 ms), this route has its declared ten-millimetre peak.
        u = np.pi*time_s/.04
        offset = self.amplitude*np.sin(u)**3
        rate = self.amplitude*3*np.sin(u)**2*np.cos(u)*np.pi/.04
        value["continuum_target_position"] += np.array([offset, 0., 0.])
        value["continuum_target_velocity"] += np.array([rate, 0., 0.])
        return value


def binding_fixture(amplitude=.010):
    n = 20
    clock = np.arange(n+1, dtype=float)*.002
    qpos = np.outer(clock, np.arange(1, 82, dtype=float)*.01)
    qvel = np.outer(clock, np.arange(1, 80, dtype=float)*.02)
    state = {"time": clock, "qpos": qpos, "qvel": qvel}
    task = SimpleNamespace(initial_qpos=qpos[0].copy(), initial_qvel=qvel[0].copy())
    z = np.zeros((6, 2)); z[0, 0] = amplitude
    plan = SimpleNamespace(z_m=z)
    observed = observed_data()
    provider = ResidualFeedback(task, plan).prepare(None, None, observed, None)
    samples = []
    for index, t in enumerate(clock):
        observed.qpos[:] = qpos[index]; observed.qvel[:] = qvel[index]; observed.time = t
        samples.append(provider.sample(t))
    trace = {"torque": np.zeros((n, 67)), "actual_full_qpos": qpos[1:].copy(),
        "actual_full_qvel": qvel[1:].copy(), "generated_reference_time_s": clock[1:].copy(),
        "task_time": clock[:-1:10].copy()}
    for key in execution.REFERENCE_KEYS:
        trace["task_input_reference_"+key] = np.asarray([samples[index][key] for index in (0, 10)])
    for arm in ("rigid", "continuum"):
        for suffix in ("position", "rotation"):
            key = arm+"_"+suffix
            trace["generated_"+key] = np.asarray([value[arm+"_target_"+suffix] for value in samples[1:]])
    for key in ("q", "dq"):
        trace["generated_reference_"+key] = np.asarray([value["posture_reference_"+key] for value in samples[1:]])
    return task, plan, trace, state


class ResidualBindingTests(unittest.TestCase):
    def bind(self, task, plan, trace, state):
        with patch("model_test.whole_body_verifier_v5.WholeBodyCollisionVerifier",
                return_value=SimpleNamespace(model=object())), \
                patch.object(execution, "scenario_from_task", return_value=SimpleNamespace(obstacles=[])), \
                patch.object(execution, "TaskAnchoredResidualReferenceProvider", ResidualFeedback), \
                patch.object(execution, "CartesianPassThroughReferenceProvider", FeedbackBase), \
                patch("mujoco.MjData", side_effect=lambda model: observed_data()), \
                patch("mujoco.mj_forward"), \
                patch("mujoco.mj_step", side_effect=AssertionError("physics forbidden")), \
                patch("mujoco.mj_geomDistance", side_effect=AssertionError("geometry queries forbidden")):
            return execution.reference_consumption_binding(task, plan, trace, state, object())

    def test_valid_feedback_dependent_nonzero_route_binds_all_QP_inputs(self):
        task, plan, trace, state = binding_fixture()
        result = self.bind(task, plan, trace, state)
        self.assertTrue(result["passed"])
        self.assertEqual(result["planning_inputs_bound"], 2)
        self.assertEqual(set(result["QP_reference_fields_bound"]), set(execution.REFERENCE_KEYS))
        self.assertEqual(result["generated_continuum_and_posture_poststep_samples_bound"], 20)
        self.assertTrue(result["nonzero_reference_consumed"])
        self.assertTrue(result["peak_at_least_5mm"])
        self.assertAlmostEqual(result["consumed_QP_reference_offset_peak_m"], .010)
        self.assertEqual(result["new_physics_steps_by_binding"], 0)

    def test_zero_route_has_exact_ten_field_parity_and_finite_cached_rigid_diagnostic(self):
        task, plan, trace, state = binding_fixture(0.)
        # The frozen executor's poststep rigid cache is not a current-state QP
        # input. A finite cache discrepancy must not falsify zero input parity.
        trace["generated_rigid_position"] += .00002
        trace["generated_rigid_rotation"] += .00001
        result = self.bind(task, plan, trace, state)
        self.assertTrue(result["zero_residual_all_10_fields_exact_base_parity"])
        self.assertFalse(result["nonzero_reference_consumed"])
        self.assertIn("finite cached", result["generated_rigid_poststep_pose_scope"])
        self.assertTrue(all(value == 0. for value in result["component_maximum_absolute_residual"].values()))

    def test_nonfinite_consumed_or_diagnostic_logs_are_rejected(self):
        task, plan, original, state = binding_fixture()
        keys = ["actual_full_qpos", "actual_full_qvel", "generated_rigid_position",
            "generated_rigid_rotation", "generated_continuum_position", "generated_continuum_rotation",
            "generated_reference_q", "generated_reference_dq", "generated_reference_time_s",
            "task_time", "task_input_reference_rigid_target_velocity"]
        for key in keys:
            with self.subTest(key=key):
                trace = copy.deepcopy(original)
                trace[key].flat[0] = np.nan
                with self.assertRaisesRegex(ValueError, "must be finite"):
                    self.bind(task, plan, trace, state)

    def test_wrong_physical_clocks_are_rejected(self):
        task, plan, original, state = binding_fixture()
        for key in ("task_time", "generated_reference_time_s"):
            with self.subTest(key=key):
                trace = copy.deepcopy(original)
                trace[key][0] += .001
                with self.assertRaisesRegex(ValueError, "clock differs"):
                    self.bind(task, plan, trace, state)

    def test_perturbing_any_consumed_QP_field_is_rejected(self):
        task, plan, original, state = binding_fixture()
        for key in execution.REFERENCE_KEYS:
            with self.subTest(key=key):
                trace = copy.deepcopy(original)
                trace["task_input_reference_"+key].flat[0] += 1e-5
                with self.assertRaisesRegex(ValueError, "consumed reference differs"):
                    self.bind(task, plan, trace, state)

    def test_mismatched_actual_feedback_and_generated_continuum_are_rejected(self):
        task, plan, original, state = binding_fixture()
        for key in ("actual_full_qpos", "actual_full_qvel", "generated_continuum_position",
                "generated_continuum_rotation", "generated_reference_q", "generated_reference_dq"):
            with self.subTest(key=key):
                trace = copy.deepcopy(original)
                trace[key].flat[0] += 1e-5
                with self.assertRaises(ValueError):
                    self.bind(task, plan, trace, state)

    def test_rejected_unconsumed_tail_is_excluded_from_QP_binding(self):
        task, plan, trace, state = binding_fixture()
        for key in ["task_time", *["task_input_reference_"+k for k in execution.REFERENCE_KEYS]]:
            shape = (1,)+trace[key].shape[1:]
            trace[key] = np.concatenate((trace[key], np.full(shape, np.nan)))
        result = self.bind(task, plan, trace, state)
        self.assertTrue(result["passed"])
        self.assertEqual(result["planning_inputs_bound"], 2)


class ResidualReuseTests(unittest.TestCase):
    def prepare_rejected_slot(self, root):
        task = fixture_task()
        plan = SimpleNamespace(to_dict=lambda: {"schema": "test-only-plan", "z_m": [[0., 0.]]*6})
        config = root/"frozen_execution_config.json"
        execution.write(config, {"test_only": "precheck_refuses_before_QP_construction"})
        identity = root/"source_identity.json"
        execution.write(identity, {"source_sha256": {}, "protected_artifacts": {
            str(config.resolve()): execution.sha(config)}})
        output = root/"slot"
        with patch.object(execution, "reference_precheck", return_value={"passed": False}), \
                patch.object(execution, "run_synchronous_scenario", side_effect=AssertionError("actual forbidden")), \
                patch("mujoco.mj_step", side_effect=AssertionError("physics forbidden")):
            result = execution.execute_residual_attempt(task, plan, output,
                qp_config_path=config, identity_path=identity, slot_id="declared_slot")
        self.assertEqual(result["status"], "REFERENCE_PRECHECK_REJECTED")
        self.assertEqual(result["actual_steps"], 0)
        return task, plan, output, config, identity

    def reuse(self, task, plan, output, config, identity, slot_id="declared_slot"):
        with patch.object(execution, "run_synchronous_scenario", side_effect=AssertionError("actual forbidden")), \
                patch("mujoco.mj_step", side_effect=AssertionError("physics forbidden")):
            return execution.execute_residual_attempt(task, plan, output,
                qp_config_path=config, identity_path=identity, slot_id=slot_id, reuse_completed=True)

    def test_valid_reuse_does_not_execute_and_wrong_slot_or_plan_hash_refuses(self):
        with tempfile.TemporaryDirectory() as temp:
            args = self.prepare_rejected_slot(Path(temp))
            self.assertEqual(self.reuse(*args)["actual_steps"], 0)
            with self.assertRaisesRegex(ValueError, "reuse identity mismatch"):
                self.reuse(*args, slot_id="different_slot")
            (args[2]/"plan.json").write_text("{}\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "retained plan file differs"):
                self.reuse(*args)

    def test_changed_or_unprotected_execution_config_refuses_before_actual(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            task, plan, output, config, identity = self.prepare_rejected_slot(root)
            alternative = root/"unprotected_config.json"
            alternative.write_bytes(config.read_bytes())
            with self.assertRaisesRegex(ValueError, "not frozen"):
                self.reuse(task, plan, output, alternative, identity)
            config.write_text("{}\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "protected artifact changed"):
                self.reuse(task, plan, output, config, identity)

    def test_reuse_checks_trace_evaluation_and_evaluation_manifest(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            args = self.prepare_rejected_slot(root)
            output = args[2]
            trace = output/"saved_trace.fixture"
            trace.write_bytes(b"retained fixture; no actual execution")
            evaluation = output/"evaluation"
            evaluation.mkdir()
            report = evaluation/"report.json"
            artifact = evaluation/"bound_evidence.fixture"
            execution.write(report, {"test_fixture": True})
            artifact.write_bytes(b"original evidence")
            execution.write(evaluation/"manifest.json", {"report.json": execution.sha(report),
                artifact.name: execution.sha(artifact)})
            result_path = output/"attempt_result.json"
            result = json.loads(result_path.read_text(encoding="utf-8"))
            result.update(trace_path=str(trace), trace_sha256=execution.sha(trace),
                evaluation_path=str(report), evaluation_sha256=execution.sha(report))
            result_path.write_text(json.dumps(result), encoding="utf-8")
            self.assertEqual(self.reuse(*args)["actual_steps"], 0)
            original_trace = trace.read_bytes()
            trace.write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "retained trace evidence differs"):
                self.reuse(*args)
            trace.write_bytes(original_trace)
            artifact.write_bytes(b"changed evidence")
            with self.assertRaisesRegex(ValueError, "retained evaluation artifact differs"):
                self.reuse(*args)


if __name__ == "__main__":
    unittest.main()
