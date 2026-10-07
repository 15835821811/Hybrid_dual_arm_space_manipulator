"""Protocol checks use temporary declarations and never execute physics."""
from dataclasses import dataclass, replace
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from v6_4 import residual_protocol as protocol
from v6_4.task_protocol import canonical_json
from v6_4.tests.test_task_protocol import fixture_task


def task_fixture(index, split):
    task = fixture_task()
    point = index * .003
    source = {"workspace_obstacles": [
        {"name": "rigid", "center_w": [1., 2., 3.], "radius_m": .035},
        {"name": "continuum", "center_w": [2., 3., 4.], "radius_m": .025}],
        "continuum_target": {"waypoint_points_m": [[point + k*.001, .1, .2] for k in range(7)]},
        "scenario_id": f"scene_{index}", "seed": index + 100}
    return replace(task, task_id=f"task_{split}_{index}", group_id=f"group_{index}",
        seed=index + 100, split=split, scenario_json=canonical_json(source))


def tasks_fixture():
    return tuple(task_fixture(i, "train" if i < 6 else "val" if i < 8 else "test") for i in range(12))


def definition_fixture(task):
    value = {"schema": "task_anchored_cartesian_residual_v1", "task_id": task.task_id,
        "task_sha256": task.sha256(), "interval_mask": [True, False, True, True, False, True],
        "coefficient_norm_bound_m": .020, "frame": "world", "time_mapping": "identity_physical_time",
        "support_cutoff_s": 23.98, "applicable": True}
    value["definition_sha256"] = protocol.object_sha256(value)
    return value


class FakePlan:
    def __init__(self, definition, z):
        self.value = {"schema": "task_anchored_residual_plan_v1", "definition": definition,
            "z_m": np.asarray(z).tolist()}

    @classmethod
    def from_definition(cls, definition, z):
        if definition.get("applicable") is not True:
            raise ValueError("NOT_APPLICABLE")
        return cls(definition, z)

    def to_dict(self):
        return self.value

    def sha256(self):
        return protocol.object_sha256(self.value)


@dataclass(frozen=True)
class FakeScene:
    scenario_id: str
    seed: int


class ResidualProtocolTests(unittest.TestCase):
    def test_three_fixed_nonzero_modes_use_only_active_mask(self):
        definition = definition_fixture(task_fixture(1, "train"))
        active = np.asarray(definition["interval_mask"])
        for index, expected in enumerate(([.010, 0.], [-.010, 0.], [0., .010])):
            z = protocol.teacher_coefficients(definition, index)
            np.testing.assert_array_equal(z[active], np.tile(expected, (active.sum(), 1)))
            np.testing.assert_array_equal(z[~active], np.zeros((2, 2)))
            self.assertLessEqual(float(np.max(np.linalg.norm(z, axis=1))), .020)
        for invalid in (-1, 3, True, 1.):
            with self.assertRaises(ValueError):
                protocol.teacher_coefficients(definition, invalid)

    def test_scene_generation_calls_one_twelve_scene_factory_without_actual(self):
        scenes = tuple(FakeScene(f"source_{i}", 2026100701+104729*i) for i in range(12))
        def construct(spec, scenario, *, task_id, group_id, family, split, layout_diagnostics):
            index = scenes.index(FakeScene(f"source_{layout_diagnostics['source_scene_index']}", scenario.seed))
            task = task_fixture(index, split)
            source = task.scenario
            source.update(scenario_id=scenario.scenario_id, seed=scenario.seed)
            return replace(task, task_id=task_id, group_id=group_id, family=family, seed=scenario.seed,
                scenario_json=canonical_json(source), layout_json=canonical_json(layout_diagnostics))
        with patch("v6_lite.run_v6_lite.default_v6_lite_robot_spec", return_value=object()), \
                patch("v6_lite.run_v6_lite.build_scenarios", return_value=scenes) as factory, \
                patch.object(protocol, "task_from_scenario", side_effect=construct), \
                patch("v6_4.reference_adapter.scenario_from_task", side_effect=lambda task: SimpleNamespace(to_dict=lambda: task.scenario)), \
                patch("mujoco.mj_step", side_effect=AssertionError("physics forbidden")):
            tasks = protocol.generate_residual_tasks()
        factory.assert_called_once()
        self.assertEqual(factory.call_args.args[1].scenario_count, 12)
        self.assertEqual([t.split for t in tasks], ["train"]*6+["val"]*2+["test"]*4)
        self.assertEqual(len({t.seed for t in tasks}), 12)
        self.assertEqual(len({t.group_id for t in tasks}), 12)
        self.assertTrue(all(t.family == "end_effector_detour" for t in tasks))
        self.assertTrue(all(not json.loads(t.layout_json)["trajectory_feasibility_established"] for t in tasks))

    def test_distance_diagnostic_cannot_relabel_old_template(self):
        old = task_fixture(1, "test")
        current = task_fixture(2, "test")
        result = protocol.legacy_distance_diagnostics([current], [("D0", old)])
        self.assertFalse(result["old_task_distances_used_to_select_or_replace_new_tasks"])
        self.assertAlmostEqual(result["records"][0]["waypoint_rms_distance_m"], .003)
        clone = replace(old, task_id="new_id", group_id="new_group", seed=9999)
        with self.assertRaisesRegex(ValueError, "numerical waypoint"):
            protocol.legacy_distance_diagnostics([clone], [("D0", old)])

    def _freeze_fixture(self, root, *, config_change=None):
        output = root / "new_output"
        output.mkdir()
        config = {"task_period_s": .020, "continuum_speed_limit_m_s": .24,
            "rigid_speed_limit_m_s": .24, "enable_pcc_braking_guard": True}
        config.update(config_change or {})
        protocol._write(output/"frozen_execution_config.json", config)
        protocol._write(output/"bootstrap_identity.json", {"base_git_head": "a"*40,
            "frozen_config_sha256": protocol.sha256_file(output/"frozen_execution_config.json")})
        positive = task_fixture(99, "test")
        old = root/"old_positive"
        old.mkdir()
        task_path, result_path, evaluation_path = [old/name for name in ("task.json", "result.json", "evaluation.json")]
        protocol._write(task_path, positive.to_dict())
        result = {"task_sha256": positive.sha256(), "task_success": True}
        evaluation = {**result, "complete": True, "evidence_valid": True,
            "metrics": {"physics_steps": 13500}, "execution_contract": {"passed": True},
            "independent_interval": {"passed": True}, "native_geometry": {"passed": True}}
        protocol._write(result_path, result)
        protocol._write(evaluation_path, evaluation)
        return output, (task_path, result_path, evaluation_path)

    def _patch_freeze(self, positive_paths, definitions=definition_fixture):
        from contextlib import ExitStack
        stack = ExitStack()
        stack.enter_context(patch.dict(sys.modules, {"v6_4.task_anchored_reference": SimpleNamespace(
            build_reference_definition=definitions, TaskAnchoredResidualPlan=FakePlan)}))
        stack.enter_context(patch.object(protocol, "generate_residual_tasks", return_value=tasks_fixture()))
        stack.enter_context(patch.object(protocol, "_legacy_tasks", return_value=([], [])))
        for name, path in zip(("POSITIVE_TASK", "POSITIVE_RESULT", "POSITIVE_EVALUATION"), positive_paths):
            stack.enter_context(patch.object(protocol, name, path))
        stack.enter_context(patch("mujoco.mj_step", side_effect=AssertionError("physics forbidden")))
        return stack

    def test_freeze_binds_all_slots_inputs_and_budget_and_refuses_regeneration(self):
        with tempfile.TemporaryDirectory() as tmp:
            output, paths = self._freeze_fixture(Path(tmp))
            with self._patch_freeze(paths):
                plan = protocol.freeze_residual_protocol(output)
                self.assertEqual(protocol.validate_frozen_protocol(output), plan)
                with self.assertRaises(FileExistsError):
                    protocol.freeze_residual_protocol(output)
            self.assertEqual(len(plan["records"]), 24)
            self.assertEqual(len(plan["test_records"]), 4)
            self.assertEqual(plan["budget"]["total_actual_max"], 37)
            self.assertEqual(plan["budget"]["TEST_E2_candidate_slots"], 16)
            self.assertEqual(plan["comparison"]["E2_K1_slot"], 0)
            self.assertEqual(plan["comparison"]["E2_K4_actual"], "NOT_RUN")
            self.assertEqual(plan["training"]["optimizer_updates"], 4000)
            self.assertEqual(plan["training"]["validation_draws_per_successful_VAL_reference"], 16)
            self.assertEqual(plan["training"]["optimizer"]["name"], "AdamW")
            self.assertEqual(plan["reference_precheck"]["old_joint_codec_range"], "N/A/new-representation")
            self.assertFalse(plan["zero_interface"]["included_in_new_TRAIN_VAL_TEST"])
            for row in plan["records"]:
                candidate = protocol._read(output/row["plan_path"])
                self.assertEqual(row["plan_sha256"], protocol.object_sha256(candidate))
            first = output/plan["records"][0]["plan_path"]
            first.write_text("{}\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "artifact differs"):
                protocol.validate_frozen_protocol(output)

    def test_execution_speed_change_refuses_before_any_task_generation(self):
        with tempfile.TemporaryDirectory() as tmp:
            output, paths = self._freeze_fixture(Path(tmp), config_change={"continuum_speed_limit_m_s": .25})
            with self._patch_freeze(paths), patch.object(protocol, "generate_residual_tasks") as generate:
                with self.assertRaisesRegex(ValueError, "frozen timing"):
                    protocol.freeze_residual_protocol(output)
                generate.assert_not_called()
            self.assertFalse((output/"plan.json").exists())
            self.assertFalse((output/"tasks.json").exists())

    def test_inapplicable_task_keeps_teacher_slots_without_fake_valid_plan(self):
        def definitions(task):
            value = definition_fixture(task)
            if task.task_id == "task_train_0":
                value.update(applicable=False, interval_mask=[False]*6, status="NOT_APPLICABLE")
                value["definition_sha256"] = protocol.object_sha256({k:v for k,v in value.items() if k!="definition_sha256"})
            return value
        with tempfile.TemporaryDirectory() as tmp:
            output, paths = self._freeze_fixture(Path(tmp))
            with self._patch_freeze(paths, definitions):
                plan = protocol.freeze_residual_protocol(output)
            rows = [row for row in plan["records"] if row["task_id"] == "task_train_0"]
            self.assertEqual(len(rows), 3)
            self.assertTrue(all(row["status"] == "NOT_APPLICABLE" for row in rows))
            self.assertEqual(len(plan["records"]), 24)
            self.assertTrue(all(protocol._read(output/r["plan_path"])["schema"] ==
                "v6_4_b2_inapplicable_candidate_slot_v1" for r in rows))


if __name__ == "__main__":
    unittest.main()
