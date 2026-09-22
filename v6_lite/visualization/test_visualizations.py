"""Regression checks for fixed V6-lite visualization artifacts."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from v6_lite.visualization.generate_visualizations import (
    SOURCE_CONTRACT_VERSION,
    VISUALIZATION_CONTRACT_VERSION,
    _clearance_summary_payload,
    _make_continuum_focus_camera,
    _frame_axis_endpoints,
    _safety_overlay_text,
    generate_continuum_focus,
    plot_clearance_summary,
)


ROOT = Path(__file__).resolve().parent / "output"


class V6LiteVisualizationTests(unittest.TestCase):
    @staticmethod
    def _v6_lite_6_metrics_fixture() -> dict:
        whole_body = [0.0054, 0.0061]
        continuum_target = [0.0252, 0.0264]
        scenarios = []
        for index, (whole, continuum) in enumerate(
            zip(whole_body, continuum_target)
        ):
            scenarios.append(
                {
                    "scenario": {"scenario_id": f"scenario_{index:02d}"},
                    "metrics": {
                        "whole_body_clearance": {
                            "minimum_clearance": whole,
                            "minimum_by_class": {
                                "continuum_target": continuum,
                            },
                        }
                    },
                }
            )
        return {
            "contract_version": SOURCE_CONTRACT_VERSION,
            "run_config": {"whole_body_minimum_clearance_m": 0.005},
            "qp_config": {
                "clearance_safe_m": 0.025,
                "rigid_target_clearance_safe_m": 0.005,
            },
            "aggregate_metrics": {
                "whole_body_minimum_clearance_m": min(whole_body),
                "continuum_target_minimum_clearance_m": min(continuum_target),
            },
            "scenarios": scenarios,
        }

    def test_coordinate_frame_axes_follow_rotation_columns(self) -> None:
        position = np.asarray([1.0, 2.0, 3.0])
        rotation = np.asarray(
            [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
        )
        endpoints = _frame_axis_endpoints(position, rotation, 0.1)
        np.testing.assert_allclose(endpoints[:, 0], np.repeat(position[None, :], 3, axis=0))
        np.testing.assert_allclose(endpoints[:, 1] - position, 0.1 * rotation.T)

    def test_v6_lite_6_clearance_summary_keeps_safety_classes_distinct(self) -> None:
        metrics = self._v6_lite_6_metrics_fixture()
        summary = _clearance_summary_payload(metrics)
        self.assertEqual(
            summary["whole_body_minimum_clearance_m"], [0.0054, 0.0061]
        )
        self.assertEqual(
            summary["continuum_target_minimum_clearance_m"], [0.0252, 0.0264]
        )
        self.assertEqual(summary["verification_gate_m"], 0.005)
        self.assertEqual(summary["continuum_target_qp_nominal_margin_m"], 0.025)
        self.assertEqual(summary["rigid_target_qp_nominal_margin_m"], 0.005)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "safety_clearance_summary.png"
            returned = plot_clearance_summary(metrics, output)
            self.assertEqual(returned, summary)
            self.assertTrue(output.is_file())
            self.assertGreater(output.stat().st_size, 10_000)

    def test_video_safety_text_names_both_clearance_quantities(self) -> None:
        text = _safety_overlay_text(0.005, 0.025)
        self.assertIn("online whole-body", text)
        self.assertIn("continuum-target min", text)
        self.assertIn("5.00 mm", text)
        self.assertIn("25.00 mm", text)

    def test_clearance_summary_rejects_pre_v6_lite_6_contract(self) -> None:
        metrics = self._v6_lite_6_metrics_fixture()
        metrics["contract_version"] = "v6_lite_5"
        with self.assertRaisesRegex(ValueError, SOURCE_CONTRACT_VERSION):
            _clearance_summary_payload(metrics)

    def test_continuum_focus_camera_uses_positive_y_side_view(self) -> None:
        minimum = np.asarray([0.40, -0.20, -0.20])
        maximum = np.asarray([1.95, 0.71, 0.20])
        camera, metadata = _make_continuum_focus_camera(
            np.stack([minimum, maximum])
        )
        np.testing.assert_allclose(
            camera.lookat, 0.5 * (minimum + maximum), rtol=0.0, atol=1e-12
        )
        self.assertEqual(camera.azimuth, -90.0)
        self.assertEqual(camera.elevation, -18.0)
        self.assertGreaterEqual(camera.distance, 1.45)
        self.assertEqual(metadata["azimuth_deg"], -90.0)
        self.assertEqual(metadata["elevation_deg"], -18.0)

    def test_continuum_focus_generator_never_calls_five_view_renderer(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            metrics_path = root / "metrics.json"
            output_dir = root / "focus"
            metrics_path.write_text(
                json.dumps(
                    {
                        "contract_version": SOURCE_CONTRACT_VERSION,
                        "passed": True,
                    }
                ),
                encoding="utf-8",
            )

            def fake_focus(_metrics: dict, path: Path) -> dict:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"focus-video-fixture")
                return {
                    "view": "continuum_focus",
                    "path": path.as_posix(),
                    "camera": {"azimuth_deg": -90.0, "elevation_deg": -18.0},
                }

            with mock.patch(
                "v6_lite.visualization.generate_visualizations."
                "render_continuum_focus_view",
                side_effect=fake_focus,
            ) as focus_renderer, mock.patch(
                "v6_lite.visualization.generate_visualizations.render_five_views"
            ) as five_view_renderer:
                manifest = generate_continuum_focus(metrics_path, output_dir)
            focus_renderer.assert_called_once()
            five_view_renderer.assert_not_called()
            self.assertEqual(manifest["video"]["view"], "continuum_focus")
            self.assertTrue((output_dir / "continuum_focus_manifest.json").is_file())

    def test_manifest_declares_plots_five_views_composite_and_preview(self) -> None:
        manifest = json.loads(
            (ROOT / "visualization_manifest.json").read_text(encoding="utf-8")
        )
        # Checked-in artifacts are intentionally not regenerated by unit
        # tests.  Accept the prior immutable bundle until a V6-lite 6 formal
        # run produces the new 11-artifact visualization contract.
        self.assertIn(
            manifest["contract_version"],
            {"v6_lite_visualization_5", VISUALIZATION_CONTRACT_VERSION},
        )
        self.assertEqual(
            manifest["video"]["views"], ["overview", "front", "side", "top", "iso"]
        )
        expected_artifacts = (
            11
            if manifest["contract_version"] == VISUALIZATION_CONTRACT_VERSION
            else 10
        )
        self.assertEqual(len(manifest["artifacts"]), expected_artifacts)
        if manifest["contract_version"] == VISUALIZATION_CONTRACT_VERSION:
            self.assertEqual(
                manifest["source_contract_version"], SOURCE_CONTRACT_VERSION
            )
            clearance = manifest["safety_clearance_plot"]
            self.assertEqual(
                clearance["quantities"],
                [
                    "whole_body_minimum_clearance_m",
                    "continuum_target_minimum_clearance_m",
                ],
            )
            self.assertIn(
                "online_whole_body_clearance_mm",
                manifest["video"]["overlay_metrics"],
            )
            self.assertIn(
                "dense_discrete_continuum_target_minimum_clearance_mm",
                manifest["video"]["overlay_metrics"],
            )
        self.assertEqual(
            manifest["video"]["coordinate_frames"]["frames_rendered"],
            [
                "rigid_grasp_target",
                "rigid_end_effector",
                "continuum_irregular_target",
                "continuum_end_effector",
                "free_base_initial_pose",
                "free_base_current_pose",
                "continuum_waypoint_frames_W1_to_W7",
            ],
        )
        self.assertEqual(
            manifest["base_pose_drift_gif"]["quantities"],
            ["base_translation_drift_mm", "base_attitude_drift_deg"],
        )
        self.assertEqual(
            manifest["video"]["coordinate_frames"]["waypoint_labels_rendered"],
            [f"W{index}" for index in range(1, 8)],
        )
        self.assertEqual(
            manifest["video"]["coordinate_frames"]["waypoint_label_view"],
            "front",
        )
        np.testing.assert_allclose(
            manifest["video"]["coordinate_frames"][
                "continuum_end_effector_local_offset_body_m"
            ],
            [0.0475, 0.0, 0.0],
            rtol=0.0,
            atol=1e-12,
        )
        self.assertIn(
            "base_translation_drift_mm", manifest["video"]["overlay_metrics"]
        )
        self.assertIn(
            "base_attitude_drift_deg", manifest["video"]["overlay_metrics"]
        )

    def test_zero_trust_visualization_audit_passes(self) -> None:
        audit = json.loads(
            (ROOT / "visualization_validation.json").read_text(encoding="utf-8")
        )
        self.assertTrue(audit["passed"])
        self.assertEqual(audit["passed_count"], audit["total_count"])
        self.assertTrue(all(audit["checks"].values()))
        self.assertIn(
            audit["contract_version"],
            {
                "v6_lite_visualization_audit_1",
                "v6_lite_visualization_audit_2",
            },
        )
        if audit["contract_version"] == "v6_lite_visualization_audit_2":
            self.assertTrue(
                audit["checks"]["distinct_clearance_summary_declared"]
            )
            self.assertTrue(
                audit["checks"]["distinct_clearance_video_overlays_declared"]
            )


if __name__ == "__main__":
    unittest.main()
