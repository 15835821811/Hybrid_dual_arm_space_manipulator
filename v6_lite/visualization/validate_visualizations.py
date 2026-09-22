"""Audit V6-lite plots and five-view videos without trusting their manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np
import mujoco
from PIL import Image

from model_test.whole_body_verifier_v5 import (
    WholeBodyCollisionVerifier,
    WholeBodyVerificationConfig,
)
from v6_lite.hierarchical_qp import free_joint_slices
from v6_lite.run_v6_lite import default_v6_lite_robot_spec
from v6_lite.visualization.generate_visualizations import (
    SOURCE_CONTRACT_VERSION,
    VISUALIZATION_CONTRACT_VERSION,
    _clearance_summary_payload,
)


ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = ROOT.parent
DEFAULT_OUTPUT = ROOT / "visualization" / "output"
VALIDATED_CONTINUUM_EE_OFFSET_M = np.asarray(
    [0.0475, 0.0, 0.0], dtype=np.float64
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _manifest_path(value: str) -> Path:
    """Resolve a portable manifest path against the repository root."""

    path = Path(value)
    return path if path.is_absolute() else REPOSITORY_ROOT / path


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _ffprobe(ffprobe: str, path: Path) -> dict[str, Any]:
    completed = subprocess.run(
        [
            ffprobe,
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=codec_name,width,height,r_frame_rate,nb_frames",
            "-show_entries",
            "format=duration",
            "-of",
            "json",
            str(path),
        ],
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"ffprobe failed for {path}: {completed.stderr}")
    return json.loads(completed.stdout)


def _decode_frame(
    ffmpeg: str, path: Path, time_s: float, width: int, height: int
) -> np.ndarray:
    completed = subprocess.run(
        [
            ffmpeg,
            "-v",
            "error",
            "-ss",
            f"{time_s:.6f}",
            "-i",
            str(path),
            "-frames:v",
            "1",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-",
        ],
        capture_output=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"frame decode failed for {path}")
    expected = width * height * 3
    if len(completed.stdout) != expected:
        raise ValueError(
            f"decoded frame has {len(completed.stdout)} bytes, expected {expected}"
        )
    return np.frombuffer(completed.stdout, dtype=np.uint8).reshape(height, width, 3)


def _quaternion_geodesic_angle_rad(
    reference_quaternion: np.ndarray, quaternions: np.ndarray
) -> np.ndarray:
    reference = np.asarray(reference_quaternion, dtype=np.float64).reshape(4)
    values = np.asarray(quaternions, dtype=np.float64)
    reference /= np.linalg.norm(reference)
    values = values / np.linalg.norm(values, axis=-1, keepdims=True)
    dots = np.abs(np.sum(values * reference, axis=-1))
    return 2.0 * np.arccos(np.clip(dots, 0.0, 1.0))


def validate(output_dir: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    manifest_path = output_dir / "visualization_manifest.json"
    manifest = _load_json(manifest_path)
    ffmpeg = shutil.which("ffmpeg")
    ffprobe = shutil.which("ffprobe")
    if not ffmpeg or not ffprobe:
        raise RuntimeError("ffmpeg and ffprobe are required for video audit")
    checks: dict[str, bool] = {}
    checks["contract_version"] = (
        manifest.get("contract_version") == VISUALIZATION_CONTRACT_VERSION
        and manifest.get("source_contract_version") == SOURCE_CONTRACT_VERSION
    )
    source_metrics = _manifest_path(manifest["source_metrics"])
    checks["source_metrics_hash"] = (
        source_metrics.is_file()
        and _sha256(source_metrics) == manifest["source_metrics_sha256"]
    )
    source_trace = _manifest_path(manifest["video"]["source_trace"])
    checks["source_trace_hash"] = (
        source_trace.is_file()
        and _sha256(source_trace) == manifest["video"]["source_trace_sha256"]
    )
    source_metrics_payload = _load_json(source_metrics)
    checks["source_contract_version"] = (
        source_metrics_payload.get("contract_version") == SOURCE_CONTRACT_VERSION
    )
    checks["irregular_waypoint_source"] = (
        source_metrics_payload["scenarios"][0]["scenario"]["continuum_target"][
            "mode"
        ]
        == "irregular_waypoints"
    )
    coordinate_frames = manifest["video"].get("coordinate_frames", {})
    checks["task_base_and_waypoint_frames_declared"] = (
        coordinate_frames.get("axis_color_order")
        == ["x_red", "y_green", "z_blue"]
        and coordinate_frames.get("frames_rendered")
        == [
            "rigid_grasp_target",
            "rigid_end_effector",
            "continuum_irregular_target",
            "continuum_end_effector",
            "free_base_initial_pose",
            "free_base_current_pose",
            "continuum_waypoint_frames_W1_to_W7",
        ]
        and float(coordinate_frames.get("target_axis_length_m", 0.0)) > float(
            coordinate_frames.get("end_effector_axis_length_m", 0.0)
        )
        and float(coordinate_frames.get("waypoint_axis_length_m", 0.0)) > 0.0
        and coordinate_frames.get("waypoint_labels_rendered")
        == [f"W{index}" for index in range(1, 8)]
        and coordinate_frames.get("waypoint_label_view") == "front"
        and coordinate_frames.get("continuum_end_effector_body_name") == "link_30"
        and np.allclose(
            np.asarray(
                coordinate_frames.get(
                    "continuum_end_effector_local_offset_body_m", []
                ),
                dtype=np.float64,
            ),
            VALIDATED_CONTINUUM_EE_OFFSET_M,
            rtol=0.0,
            atol=1e-12,
        )
        and source_metrics_payload["scenarios"][0]["scenario"][
            "continuum_target"
        ]["waypoint_count"]
        == 7
    )
    overlay_metrics = manifest["video"].get("overlay_metrics", [])
    checks["base_pose_drift_video_overlay_declared"] = (
        "base_translation_drift_mm" in overlay_metrics
        and "base_attitude_drift_deg" in overlay_metrics
    )
    checks["distinct_clearance_video_overlays_declared"] = (
        "online_whole_body_clearance_mm" in overlay_metrics
        and "dense_discrete_continuum_target_minimum_clearance_mm"
        in overlay_metrics
        and "whole_body_clearance_mm" not in overlay_metrics
    )

    expected_clearance = _clearance_summary_payload(source_metrics_payload)
    declared_clearance = manifest.get("safety_clearance_plot", {})
    declared_whole_body = np.asarray(
        declared_clearance.get("whole_body_minimum_clearance_m", []),
        dtype=np.float64,
    )
    expected_whole_body = np.asarray(
        expected_clearance["whole_body_minimum_clearance_m"],
        dtype=np.float64,
    )
    declared_continuum_target = np.asarray(
        declared_clearance.get("continuum_target_minimum_clearance_m", []),
        dtype=np.float64,
    )
    expected_continuum_target = np.asarray(
        expected_clearance["continuum_target_minimum_clearance_m"],
        dtype=np.float64,
    )
    checks["distinct_clearance_summary_declared"] = (
        declared_clearance.get("quantities")
        == [
            "whole_body_minimum_clearance_m",
            "continuum_target_minimum_clearance_m",
        ]
        and declared_clearance.get("time_scope")
        == "dense_discrete_replay_over_each_complete_run"
        and declared_clearance.get("scenario_ids")
        == expected_clearance["scenario_ids"]
        and declared_whole_body.shape == expected_whole_body.shape
        and np.allclose(
            declared_whole_body,
            expected_whole_body,
            rtol=0.0,
            atol=1e-12,
        )
        and declared_continuum_target.shape == expected_continuum_target.shape
        and np.allclose(
            declared_continuum_target,
            expected_continuum_target,
            rtol=0.0,
            atol=1e-12,
        )
        and abs(
            float(declared_clearance.get("verification_gate_m", float("nan")))
            - float(expected_clearance["verification_gate_m"])
        )
        <= 1e-12
        and abs(
            float(
                declared_clearance.get(
                    "continuum_target_qp_nominal_margin_m", float("nan")
                )
            )
            - float(expected_clearance["continuum_target_qp_nominal_margin_m"])
        )
        <= 1e-12
        and abs(
            float(
                declared_clearance.get(
                    "rigid_target_qp_nominal_margin_m", float("nan")
                )
            )
            - float(expected_clearance["rigid_target_qp_nominal_margin_m"])
        )
        <= 1e-12
    )
    aggregate = source_metrics_payload.get("aggregate_metrics", {})
    checks["clearance_aggregate_fields_recomputed"] = (
        abs(
            float(aggregate.get("whole_body_minimum_clearance_m", float("nan")))
            - float(np.min(expected_whole_body))
        )
        <= 1e-12
        and abs(
            float(
                aggregate.get(
                    "continuum_target_minimum_clearance_m", float("nan")
                )
            )
            - float(np.min(expected_continuum_target))
        )
        <= 1e-12
    )

    trace = np.load(source_trace, allow_pickle=False)
    continuum_tip_recomputed = (
        trace["continuum_tip_body_origin"]
        + np.einsum(
            "nij,j->ni",
            trace["continuum_rotation"],
            VALIDATED_CONTINUUM_EE_OFFSET_M,
        )
    )
    checks["continuum_end_effector_trace_uses_local_offset"] = (
        float(
            np.max(np.abs(continuum_tip_recomputed - trace["continuum_tip"]))
        )
        <= 1e-12
        and np.allclose(
            np.asarray(
                source_metrics_payload["kinematic_contract"][
                    "continuum_end_effector_initial_world_position_m"
                ],
                dtype=np.float64,
            ),
            np.asarray([1.930, 0.626, 0.0], dtype=np.float64),
            rtol=0.0,
            atol=1e-12,
        )
    )
    spec = default_v6_lite_robot_spec()
    verifier = WholeBodyCollisionVerifier(
        spec, (), WholeBodyVerificationConfig(adaptive_subdivisions=2)
    )
    base_qpos_slice, _base_dof_slice = free_joint_slices(
        verifier.model, spec.base_joint_name
    )
    initial_base_pose = np.asarray(trace["initial_qpos"][base_qpos_slice])
    translation_drift = np.linalg.norm(
        trace["base_qpos"][:, :3] - initial_base_pose[None, :3], axis=1
    )
    attitude_drift_deg = np.rad2deg(
        _quaternion_geodesic_angle_rad(
            initial_base_pose[3:7], trace["base_qpos"][:, 3:7]
        )
    )
    reported_base = source_metrics_payload["scenarios"][0]["metrics"][
        "free_floating_base"
    ]
    checks["base_pose_drift_source_recomputed"] = (
        float(
            np.max(
                np.abs(translation_drift - trace["base_translation_drift_m"])
            )
        )
        <= 1e-12
        and float(
            np.max(
                np.abs(
                    attitude_drift_deg - trace["base_orientation_drift_deg"]
                )
            )
        )
        <= 1e-10
        and abs(
            float(np.max(translation_drift))
            - float(reported_base["translation_drift_max_m"])
        )
        <= 1e-12
        and abs(
            float(np.max(attitude_drift_deg))
            - float(reported_base["orientation_drift_max_deg"])
        )
        <= 1e-10
    )
    artifacts = manifest.get("artifacts", [])
    checks["eleven_declared_artifacts"] = len(artifacts) == 11
    checks["all_artifact_hashes"] = all(
        _manifest_path(item["path"]).is_file()
        and _manifest_path(item["path"]).stat().st_size == int(item["bytes"])
        and _sha256(_manifest_path(item["path"])) == item["sha256"]
        for item in artifacts
    )

    error_plot = output_dir / "error_curves.png"
    path_plot = output_dir / "tracking_paths_3d.png"
    clearance_plot = output_dir / "safety_clearance_summary.png"
    preview = output_dir / "videos" / "v6_lite_scenario_00_five_view_preview.png"
    image_sizes = {}
    images_ok = True
    for path in (error_plot, path_plot, clearance_plot, preview):
        with Image.open(path) as image:
            image.load()
            image_sizes[path.name] = list(image.size)
            images_ok &= image.width >= 1200 and image.height >= 700
            array = np.asarray(image.convert("RGB"), dtype=np.float64)
            images_ok &= float(np.std(array)) >= 10.0
    checks["plots_and_preview_decode"] = images_ok

    gif_path = output_dir / "base_pose_drift.gif"
    gif_metadata = manifest.get("base_pose_drift_gif", {})
    with Image.open(gif_path) as gif:
        gif_frames = int(getattr(gif, "n_frames", 1))
        gif_size = list(gif.size)
        gif.seek(0)
        gif_first = np.asarray(gif.convert("RGB"), dtype=np.float64)
        gif.seek(gif_frames // 2)
        gif_middle = np.asarray(gif.convert("RGB"), dtype=np.float64)
        gif.seek(gif_frames - 1)
        gif_last = np.asarray(gif.convert("RGB"), dtype=np.float64)
    checks["base_pose_drift_gif_animated"] = (
        gif_frames == int(gif_metadata.get("frame_count", 0))
        and gif_frames >= 100
        and gif_size[0] >= 800
        and gif_size[1] >= 500
        and float(np.std(gif_middle)) >= 10.0
        and float(np.mean(np.abs(gif_last - gif_first))) >= 1.0
    )
    image_sizes[gif_path.name] = gif_size

    views = tuple(manifest["video"]["views"])
    checks["five_named_views"] = views == ("overview", "front", "side", "top", "iso")
    video_reports: dict[str, Any] = {}
    mid_frames = []
    videos_ok = True
    motion_ok = True
    expected_frame_count = int(manifest["video"]["frame_count"])
    expected_duration = expected_frame_count / float(manifest["video"]["fps"])
    for view in views:
        path = output_dir / "videos" / f"v6_lite_scenario_00_{view}.mp4"
        report = _ffprobe(ffprobe, path)
        stream = report["streams"][0]
        fps = float(Fraction(stream["r_frame_rate"]))
        duration = float(report["format"]["duration"])
        frame_count = int(stream["nb_frames"])
        width = int(stream["width"])
        height = int(stream["height"])
        videos_ok &= (
            stream["codec_name"] == "h264"
            and width == 640
            and height == 480
            and abs(fps - 30.0) <= 1e-9
            and frame_count == expected_frame_count
            and abs(duration - expected_duration) <= 0.02
        )
        first = _decode_frame(ffmpeg, path, 0.10, width, height)
        middle = _decode_frame(
            ffmpeg, path, 0.5 * float(manifest["video"]["duration_s"]), width, height
        )
        last = _decode_frame(
            ffmpeg,
            path,
            max(float(manifest["video"]["duration_s"]) - 0.1, 0.1),
            width,
            height,
        )
        motion_difference = float(
            np.mean(np.abs(last.astype(np.float64) - first.astype(np.float64)))
        )
        content_std = float(np.std(middle))
        motion_ok &= motion_difference >= 1.0 and content_std >= 10.0
        mid_frames.append(middle.astype(np.float64))
        video_reports[view] = {
            "codec": stream["codec_name"],
            "width": width,
            "height": height,
            "fps": fps,
            "frame_count": frame_count,
            "duration_s": duration,
            "middle_frame_std": content_std,
            "first_to_last_mean_absolute_difference": motion_difference,
        }
    checks["individual_video_encoding"] = videos_ok
    checks["individual_videos_show_motion"] = motion_ok
    pairwise_differences = []
    for first_index in range(len(mid_frames)):
        for second_index in range(first_index + 1, len(mid_frames)):
            pairwise_differences.append(
                float(np.mean(np.abs(mid_frames[first_index] - mid_frames[second_index])))
            )
    checks["camera_views_are_visually_distinct"] = min(pairwise_differences) >= 1.0

    composite = output_dir / "videos" / "v6_lite_scenario_00_five_view_grid.mp4"
    composite_report = _ffprobe(ffprobe, composite)
    composite_stream = composite_report["streams"][0]
    checks["five_view_composite_encoding"] = (
        composite_stream["codec_name"] == "h264"
        and int(composite_stream["width"]) == 1920
        and int(composite_stream["height"]) == 960
        and int(composite_stream["nb_frames"]) == expected_frame_count
        and abs(float(Fraction(composite_stream["r_frame_rate"])) - 30.0) <= 1e-9
    )
    preview_image = np.asarray(Image.open(preview).convert("RGB"), dtype=np.float64)
    checks["composite_preview_matches_video"] = (
        preview_image.shape == (960, 1920, 3)
        and float(np.std(preview_image)) >= 10.0
    )
    failures = [name for name, passed in checks.items() if not passed]
    result = {
        "contract_version": "v6_lite_visualization_audit_2",
        "passed": not failures,
        "passed_count": int(sum(checks.values())),
        "total_count": len(checks),
        "checks": checks,
        "failures": failures,
        "image_sizes": image_sizes,
        "video_reports": video_reports,
        "minimum_pairwise_view_difference": min(pairwise_differences),
        "composite": composite_report,
    }
    output_path = output_dir / "visualization_validation.json"
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2, allow_nan=False)
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main() -> None:
    args = _parser().parse_args()
    result = validate(args.output_dir.resolve())
    print(
        json.dumps(
            {
                "passed": result["passed"],
                "checks": f"{result['passed_count']}/{result['total_count']}",
                "failures": result["failures"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
