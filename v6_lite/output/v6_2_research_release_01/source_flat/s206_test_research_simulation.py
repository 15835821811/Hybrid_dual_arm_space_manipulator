"""Simulation contracts keep physical consistency while compute time is diagnostic."""
from dataclasses import replace
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import mujoco
import numpy as np

from v6_lite.hierarchical_qp import HierarchicalQPConfig
from v6_lite.runtime_command import CommandCertificate, DispatchGate, model_id, state_id
from v6_lite.runtime_timing import runtime_identity
from v6_lite.run_evidence import start_run
from v6_lite.run_v6_lite import (
    V6LiteRunConfig, _parser, _select_acceptance_checks, run_scenario,
)


class ResearchSimulationTests(unittest.TestCase):
    def setUp(self):
        self.model = mujoco.MjModel.from_xml_string(
            '<mujoco><option timestep=".002" gravity="0 0 0"/>'
            '<worldbody><body><joint name="arm"/><geom size=".01"/></body>'
            '<body name="target" pos="1 0 0"><freejoint/>'
            '<geom size=".01"/></body></worldbody>'
            '<actuator><motor joint="arm"/></actuator></mujoco>')
        self.data = mujoco.MjData(self.model)
        self.data.qvel[1] = .1
        mujoco.mj_forward(self.model, self.data)
        self.command = np.zeros(17)
        self.identity = model_id(self.model, "fixed-source")
        self.source = state_id(self.model, self.data)
        self.certificate = CommandCertificate.issue(
            command_id=0, source_state_id=self.source, source_model_hash=self.identity,
            source_partition_id="partition", command=self.command,
            acquired=100., source_simulation_s=0., solve_finished=100.080,
            validated=100.090, policy="research_simulation")

    def check(self, gate=None, certificate=None, **overrides):
        arguments = {"now": 100.100, "model_hash": self.identity,
            "partition_hash": "partition", "command": self.command,
            "observed_state_id": state_id(self.model, self.data),
            "observed_simulation_s": float(self.data.time), "policy": "research_simulation"}
        arguments.update(overrides)
        return (gate or DispatchGate()).check(certificate or self.certificate, **arguments)

    def test_slow_compute_still_executes_ten_real_two_ms_steps_with_moving_target(self):
        # Declared monotonic timestamps represent 90 ms of compute, not a wall benchmark.
        initial_target = self.data.qpos[1:4].copy()
        gate = DispatchGate()
        preview = mujoco.MjData(self.model)
        mujoco.mj_setState(self.model, preview, self._integration(self.data),
                           mujoco.mjtState.mjSTATE_INTEGRATION)
        expected = []
        for _ in range(10):
            expected.append(self._integration(preview))
            mujoco.mj_step(self.model, preview)
        self.assertEqual(self.data.time, 0.)
        for step in range(10):
            observed = self._integration(self.data)
            decision = self.check(gate, substep=step, microstate_matches=np.array_equal(observed, expected[step]))
            self.assertTrue(decision["accepted"])
            self.assertFalse(decision["wall_clock_current"])
            self.assertFalse(decision["wall_deployment_certified"])
            self.assertEqual(decision["validity_clock"], "simulation_time")
            mujoco.mj_step(self.model, self.data)
        self.assertAlmostEqual(self.data.time, .020)
        self.assertGreater(np.linalg.norm(self.data.qpos[1:4]-initial_target), .001)
        self.assertEqual(self.check(gate, substep=10)["reason"], "EXPIRED_COMMAND_SIMULATION_TIME")

    def _integration(self, data):
        result = np.empty(mujoco.mj_stateSize(self.model, mujoco.mjtState.mjSTATE_INTEGRATION))
        mujoco.mj_getState(self.model, data, result, mujoco.mjtState.mjSTATE_INTEGRATION)
        return result

    def test_real_contract_faults_still_reject_without_a_physics_step(self):
        for change in ({"model_hash": "different"}, {"partition_hash": "different"},
                {"observed_state_id": "different"}, {"command": np.ones(17)},
                {"microstate_matches": False}, {"observed_simulation_s": .002},
                {"observed_simulation_s": .020}, {"observed_simulation_s": float("nan")}):
            with self.subTest(change=change):
                self.assertFalse(self.check(**change)["accepted"])
                self.assertEqual(self.data.time, 0.)

    def test_research_cannot_extend_simulation_coverage_or_use_nonfinite_start(self):
        for change in ({"simulation_valid_until": .040}, {"simulation_valid_until": float("nan")},
                {"simulation_valid_until": float("inf")}, {"maximum_supported_state_age": .040},
                {"planned_execution_start_simulation_s": float("nan")}):
            with self.subTest(change=change):
                self.assertFalse(self.check(certificate=replace(self.certificate, **change))["accepted"])

    def test_wall_path_rejects_same_slow_compute_and_research_certificate(self):
        wall = CommandCertificate.issue(command_id=0, source_state_id=self.source,
            source_model_hash=self.identity, source_partition_id="partition", command=self.command,
            acquired=100., source_simulation_s=0., solve_finished=100.080, validated=100.090)
        self.assertEqual(self.check(certificate=wall, policy="wall_deadline")["reason"],
                         "EXPIRED_COMMAND_WALL_CLOCK")
        self.assertEqual(self.check(policy="wall_deadline")["reason"], "CERTIFICATE_CLOCK_POLICY_MISMATCH")

    def test_research_does_not_disable_wall_time_order_or_packet_order(self):
        self.assertEqual(self.check(now=100.085)["reason"], "INVALID_TIME_ORDER")
        gate = DispatchGate()
        self.assertTrue(self.check(gate)["accepted"])
        self.assertEqual(self.check(gate)["reason"], "COMMAND_OUT_OF_ORDER")

    def test_p95_failure_remains_reported_without_rejecting_safe_research(self):
        checks = {"task_and_safety": True, "execution_consistent": True,
            "task_controller_runs_within_50hz_p95": False,
            "torque_controller_runs_within_500hz_p95": False}
        acceptance, performance = _select_acceptance_checks(checks, "research_simulation")
        self.assertTrue(all(acceptance.values()))
        self.assertFalse(all(performance.values()))
        self.assertEqual(performance["task_controller_runs_within_50hz_p95"], False)
        checks["execution_consistent"] = False
        self.assertFalse(all(_select_acceptance_checks(checks, "research_simulation")[0].values()))
        for policy in ("wall_deadline", "offline_replay"):
            self.assertEqual(_select_acceptance_checks(checks, policy)[0], checks)

    def test_default_research_routes_to_synchronous_path_with_no_native_actor(self):
        self.assertEqual(V6LiteRunConfig().dispatch_clock_policy, "research_simulation")
        self.assertEqual(_parser().parse_args([]).dispatch_clock_policy, "research_simulation")
        config = V6LiteRunConfig(pcc_mode="bounded_interval_pcc")
        config.validate()
        with patch("v6_lite.run_v6_lite.run_synchronous_scenario", return_value="research") as synchronous, \
                patch("v6_lite.runtime_native_executor.run_native_scenario", side_effect=AssertionError("native actor started")):
            self.assertEqual(run_scenario(None, config, None, None, None), "research")
        synchronous.assert_called_once()

    def test_research_provenance_names_simulation_scope_and_keeps_timing_diagnostic(self):
        spec = SimpleNamespace(runtime_contract_sha256=lambda: "runtime", source_bundle_sha256=lambda: "source",
            identity=lambda: SimpleNamespace(to_dict=lambda: {"fixture": True}))
        identity = runtime_identity("bounded_interval_pcc", spec, dispatch_clock_policy="research_simulation")
        self.assertEqual(identity["controller_version"], "v6_2_research_simulation_bounded_interval_pcc")
        self.assertEqual(identity["command_validity_clock"], "simulation_time")
        with tempfile.TemporaryDirectory(prefix="research_metadata_") as temporary, \
                patch("v6_lite.run_evidence.ROOT", Path(temporary)), \
                patch("v6_lite.run_evidence.SOURCE_FILES", ()), \
                patch("v6_lite.run_evidence._git", return_value=None):
            metadata = start_run(Path(temporary)/"trial", run_config=V6LiteRunConfig(),
                qp_config=HierarchicalQPConfig(), spec=spec, scenarios=())
        timing = metadata["timing_protocol"]
        self.assertEqual(timing["dispatch_clock_policy"], "research_simulation")
        self.assertEqual(timing["command_validity_clock"], "simulation_time")
        self.assertFalse(timing["performance_is_acceptance_gate"])
        self.assertFalse(timing["wall_deadline_enforced"])
        self.assertFalse(timing["wall_deployment_certified"])


if __name__ == "__main__":
    unittest.main()
