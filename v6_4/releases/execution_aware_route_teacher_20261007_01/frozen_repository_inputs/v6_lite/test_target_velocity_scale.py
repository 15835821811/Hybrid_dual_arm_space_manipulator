"""The velocity factor must preserve the five accepted seeded scenarios."""
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import unittest

from v6_lite.run_v6_lite import V6LiteRunConfig, build_scenarios, default_v6_lite_robot_spec

FIXTURE = Path(__file__).parent / "test_fixtures/target_velocity_baseline.json"
FIXTURE_SHA = "d9b282bc73dcfcba83d3ebfe187a89b5ee9d8eecae8d0ba8204feeece933f9b2"


class TargetVelocityScaleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        raw = FIXTURE.read_bytes()
        if hashlib.sha256(raw).hexdigest() != FIXTURE_SHA:
            raise AssertionError("frozen accepted scenario fixture changed")
        cls.original = json.loads(raw)["scenarios"]
        cls.spec = default_v6_lite_robot_spec()

    def test_default_scenarios_exactly_match_all_five_accepted_definitions(self):
        config = V6LiteRunConfig()
        config.validate()
        self.assertEqual(config.target_linear_velocity_scale, 1.0)
        self.assertEqual([s.to_dict() for s in build_scenarios(self.spec, config)], self.original)

    def test_double_velocity_changes_only_the_declared_translation_factor(self):
        config = V6LiteRunConfig(target_linear_velocity_scale=2.0)
        config.validate()
        changed = [s.to_dict() for s in build_scenarios(self.spec, config)]
        for old, new in zip(self.original, changed, strict=True):
            expected = dict(old)
            expected["target_satellite_linear_velocity_m_s"] = [2.0 * v for v in old["target_satellite_linear_velocity_m_s"]]
            self.assertEqual(new, expected)

    def test_nonpositive_or_nonfinite_factors_are_rejected_before_execution(self):
        for scale in (0.0, -1.0, float("nan"), float("inf"), float("-inf")):
            with self.subTest(scale=scale), self.assertRaises(ValueError):
                replace(V6LiteRunConfig(), target_linear_velocity_scale=scale).validate()


if __name__ == "__main__":
    unittest.main()
