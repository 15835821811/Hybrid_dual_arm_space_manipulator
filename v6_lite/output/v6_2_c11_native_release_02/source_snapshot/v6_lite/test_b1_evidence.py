"""Tests for immutable runs, evidence bundles, and independent bad-action detection."""

from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

from model_test.whole_body_verifier_v5 import WholeBodyCollisionVerifier, WholeBodyVerificationConfig
from v6_lite.bundle_evidence import pack_evidence, verify_bundle
from v6_lite.hierarchical_qp import HierarchicalQPConfig
from v6_lite.recompute_execution_constraints import (
    ReplayConstraintBuilder, assess_tick, tick_residuals,
)
from v6_lite.run_evidence import fail_run, finish_run, start_run
from v6_lite.run_v6_lite import V6LiteRunConfig, build_scenarios, default_v6_lite_robot_spec


FROZEN = (
    Path(__file__).resolve().parent / "output" / "v6_2_a1" / "frozen_failure"
    / "traces" / "v6_lite_scenario_00_counterexample.json"
)


@dataclass(frozen=True)
class _Config:
    value: int = 1


class _Identity:
    def to_dict(self) -> dict:
        return {"model": "test"}


class _Spec:
    def identity(self) -> _Identity:
        return _Identity()


class _Scenario:
    def to_dict(self) -> dict:
        return {"scenario_id": "test"}


class B1EvidenceTests(unittest.TestCase):
    def test_failed_acceptance_with_nonfinite_metric_is_serialized(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "failed"
            metadata = start_run(output, run_config=_Config(), qp_config=_Config(),
                                 spec=_Spec(), scenarios=(_Scenario(),))
            finish_run(output, metadata, passed=False,
                       summary={"path_rmse": float("nan"),
                                "failed_scenarios": [{"scenario_id": "test"}]})
            record = json.loads((output / "run_failure.json").read_text())
            self.assertEqual(record["reason"], "acceptance_gate_failed")
            self.assertIsNone(record["summary"]["path_rmse"])
            self.assertEqual(json.loads((output / "run_metadata.json").read_text())["status"],
                             "FAILED")

    def test_run_directory_is_exclusive_and_failure_is_retained(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "runs" / "one"
            metadata = start_run(
                output, run_config=_Config(), qp_config=_Config(),
                spec=_Spec(), scenarios=(_Scenario(),),
            )
            self.assertEqual(metadata["status"], "RUNNING")
            self.assertEqual(metadata["run_id"], "one")
            with self.assertRaises(FileExistsError):
                start_run(output, run_config=_Config(), qp_config=_Config(),
                          spec=_Spec(), scenarios=(_Scenario(),))
            try:
                raise RuntimeError("injected failure")
            except RuntimeError as error:
                fail_run(output, metadata, error, scenario_id="test")
            failure = json.loads((output / "run_failure.json").read_text())
            self.assertEqual(failure["message"], "injected failure")
            self.assertEqual(json.loads((output / "run_metadata.json").read_text())["status"],
                             "FAILED")

    def test_bundle_hash_and_import_without_overwrite(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            evidence = root / "evidence"
            evidence.mkdir()
            (evidence / "trace.npz").write_bytes(b"small trace fixture")
            (evidence / "report.json").write_text('{"passed": false}\n', encoding="utf-8")
            bundle = root / "bundle.zip"
            pack_evidence({"run": evidence}, bundle)
            imported = root / "imported"
            result = verify_bundle(bundle, import_dir=imported)
            self.assertTrue(result["passed"])
            self.assertEqual((imported / "run" / "trace.npz").read_bytes(),
                             b"small trace fixture")
            with self.assertRaises(FileExistsError):
                verify_bundle(bundle, import_dir=imported)
            with self.assertRaises(FileExistsError):
                pack_evidence({"run": evidence}, bundle)
            sidecar = Path(str(bundle) + ".sha256.json")
            altered = json.loads(sidecar.read_text())
            altered["bundle_sha256"] = "0" * 64
            sidecar.write_text(json.dumps(altered), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "bundle hash"):
                verify_bundle(bundle)

    def test_positive_forged_residual_does_not_hide_bad_ramp(self):
        frozen = json.loads(FROZEN.read_text(encoding="utf-8"))
        snapshot = frozen["planning_snapshots"][-1]
        spec = default_v6_lite_robot_spec()
        config = V6LiteRunConfig(**frozen["run_config"])
        scenario = build_scenarios(spec, config)[0]
        verifier = WholeBodyCollisionVerifier(
            spec, scenario.obstacles,
            WholeBodyVerificationConfig(
                minimum_clearance=config.whole_body_minimum_clearance_m,
                query_distance_max=2.5,
                adaptive_subdivisions=config.verification_subdivisions,
                self_collision_ancestor_exclusion_depth=3,
                include_target_satellite_pairs=True,
            ),
        )
        model = verifier.model
        model.geom_contype[:] = 0
        model.geom_conaffinity[:] = 0
        data = mujoco.MjData(model)
        data.time = snapshot["time_s"]
        data.qpos[:] = snapshot["qpos"]
        data.qvel[:] = snapshot["qvel"]
        data.ctrl[:] = snapshot["ctrl_before_solve"]
        rows = ReplayConstraintBuilder(
            spec, model, verifier.pairs,
            HierarchicalQPConfig(**frozen["qp_config"]),
        ).build(data)
        recomputed = tick_residuals(
            rows, np.asarray(snapshot["old_command"]),
            np.asarray(snapshot["solver_candidate"]),
            HierarchicalQPConfig(**frozen["qp_config"]),
        )
        self.assertLess(recomputed["start_min_m_s"], -0.01)
        forged = assess_tick(recomputed, logged_ramp=1.0,
                             logged_lookahead=1.0, tolerance=1e-4)
        self.assertFalse(forged["feasible"])
        self.assertFalse(forged["log_agrees"])


if __name__ == "__main__":
    unittest.main()
