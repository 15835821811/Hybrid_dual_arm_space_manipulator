"""Generate and audit the complete B.2 visualization from the accepted five-scene run."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from v6_lite.visualization.generate_visualizations import (
    VIEWS,
    plot_base_pose_drift_gif,
    plot_clearance_summary,
    plot_error_curves,
    plot_tracking_paths,
    render_five_views,
)


ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "v6_lite/output/v6_2_b2/online_dense_slices_five_20260930"
OUTPUT = ROOT / "v6_lite/visualization/output_v6_2_b2_latest_20260930"
DOCUMENT = ROOT / "docs/V6_2_B2_LATEST_VISUALIZATION.md"
CONTRACT = "v6_2_b2_full_visualization_v1"


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT).as_posix()


def _record(path: Path) -> dict:
    return {"path": _relative(path), "sha256": _sha(path), "bytes": path.stat().st_size}


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _check_source(run: Path) -> tuple[dict, list[Path], dict]:
    source_manifest = _load(run / "artifact_manifest.json")
    metrics_path = ROOT / source_manifest["metrics"]["path"]
    if _sha(metrics_path) != source_manifest["metrics"]["sha256"]:
        raise ValueError("metrics do not match the immutable source manifest")
    metrics = _load(metrics_path)
    if (metrics.get("contract_version") != "v6_2_a1_ramp_aware_qp"
            or metrics["run_config"]["pcc_mode"] != "bounded_interval_pcc"
            or not metrics.get("passed") or len(metrics["scenarios"]) != 5):
        raise ValueError("source is not the passed bounded-interval five-scene run")
    trace_paths = []
    for scene, expected in zip(metrics["scenarios"], source_manifest["traces"], strict=True):
        path = ROOT / expected["path"]
        if (scene["trace"]["sha256"] != expected["sha256"]
                or _sha(path) != expected["sha256"]
                or scene["trace"]["path"] != expected["path"]):
            raise ValueError(f"trace mismatch: {path}")
        trace_paths.append(path)
    native = _load(run / "validation.json")
    execution = _load(run / "execution_validation.json")
    if (not native["passed"] or (native["passed_count"], native["total_count"]) != (26, 26)
            or not execution["passed"]
            or (execution["passed_count"], execution["total_count"]) != (11, 11)):
        raise ValueError("native replay or execution contract did not pass")
    return metrics, trace_paths, source_manifest


def _plot_timing(metrics: dict, output: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(12, 8.5), layout="constrained")
    names, p95, maximum, over = [], [], [], []
    for scene in metrics["scenarios"]:
        names.append(scene["scenario"]["scenario_id"][-2:])
        with np.load(ROOT / scene["trace"]["path"], allow_pickle=False) as trace:
            values = 1000.0 * trace["task_interval_full_control_latency_s"]
            p95.append(float(np.percentile(values, 95)))
            maximum.append(float(np.max(values)))
            over.append(int(np.count_nonzero(values > 20.0)))
            axes[0].plot(trace["task_time"], values, lw=0.55, alpha=0.65,
                         label=f"S{names[-1]}")
    axes[0].axhline(20.0, color="#dc2626", ls="--", label="20 ms task period")
    axes[0].set(xlabel="Simulation time (s)", ylabel="Full control call (ms)",
                title="Saved 50 Hz full-control timing samples; wall-clock measurements")
    axes[0].legend(ncol=6, fontsize=8)
    axes[0].grid(alpha=0.2)
    x = np.arange(5)
    axes[1].bar(x - .17, p95, width=.34, label="p95", color="#2563eb")
    axes[1].bar(x + .17, maximum, width=.34, label="maximum", color="#f97316")
    axes[1].axhline(20.0, color="#dc2626", ls="--", label="20 ms task period")
    axes[1].set(xticks=x, xticklabels=[f"S{n}" for n in names],
                ylabel="Full control call (ms)", title="Five complete scenarios")
    for i, count in enumerate(over):
        axes[1].text(i, max(maximum[i], p95[i]) + .6, f"{count} >20 ms",
                     ha="center", fontsize=8)
    axes[1].set_ylim(0, max(maximum) * 1.2)
    axes[1].legend(fontsize=8)
    axes[1].grid(axis="y", alpha=.2)
    fig.suptitle("B.2 latest accepted run | timing is measured, not a hard real-time proof")
    fig.savefig(output, dpi=170, facecolor="white")
    plt.close(fig)


def _plot_execution(metrics: dict, output: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), layout="constrained")
    for index, scene in enumerate(metrics["scenarios"]):
        with np.load(ROOT / scene["trace"]["path"], allow_pickle=False) as trace:
            t = trace["task_time"]
            delta = np.linalg.norm(trace["task_solver_candidate"]
                                   - trace["task_selected_command"], axis=1)
            if np.any(delta != 0):
                raise ValueError(f"candidate differs from executed command: {index}")
            axes[0].plot(t, np.linalg.norm(trace["task_selected_command"], axis=1),
                         lw=1, label=f"S{index:02d}")
            axes[1].plot(t, 1000 * trace["task_ramp_clearance_min_slack_m_s"],
                         lw=1, label=f"S{index:02d}")
            if (np.any(trace["task_execution_mode"] != "TRACK")
                    or np.any(trace["task_failure_reason"] != "none")):
                raise ValueError(f"successful scene has non-certified execution: {index}")
    axes[0].set(ylabel="Selected command norm (rad/s)",
                title="Solver candidate = selected command at all 6750 certified ticks")
    axes[1].set(xlabel="Simulation time (s)", ylabel="Minimum ramp clearance slack (mm/s)",
                title="Saved execution validation result across each ten-step servo ramp")
    axes[1].axhline(0, color="#dc2626", ls="--", label="feasibility boundary")
    for axis in axes:
        axis.grid(alpha=.2)
        axis.legend(ncol=6, fontsize=8)
    fig.suptitle("B.2 latest accepted run | 5 x 1350 certified task ticks")
    fig.savefig(output, dpi=170, facecolor="white")
    plt.close(fig)


def generate(run: Path = RUN, output: Path = OUTPUT, document: Path | None = DOCUMENT) -> dict:
    metrics, trace_paths, source_manifest = _check_source(run)
    metrics = json.loads(json.dumps(metrics))
    for scene, trace_path in zip(metrics["scenarios"], trace_paths, strict=True):
        scene["trace"]["path"] = str(trace_path)
    manifest_path = output / "visualization_manifest.json"
    if manifest_path.exists():
        raise FileExistsError(f"immutable visualization already exists: {manifest_path}")
    output.mkdir(parents=True, exist_ok=True)
    artifacts: list[Path] = []
    for filename, plotter in (
        ("error_curves.png", plot_error_curves),
        ("safety_clearance_summary.png", plot_clearance_summary),
        ("base_pose_drift.gif", plot_base_pose_drift_gif),
        ("full_control_timing.png", _plot_timing),
        ("execution_contract.png", _plot_execution),
    ):
        path = output / filename
        plotter(metrics, path)
        artifacts.append(path)
        print(f"generated {path.name}", flush=True)
    videos = []
    for index, scene in enumerate(metrics["scenarios"]):
        scenario_id = scene["scenario"]["scenario_id"]
        path = output / f"{scenario_id}_tracking_paths_3d.png"
        plot_tracking_paths(metrics, path, scenario_index=index)
        artifacts.append(path)
        print(f"generated {path.name}; rendering five views", flush=True)
        views, grid, preview, metadata = render_five_views(
            metrics, output / "videos", scenario_index=index)
        artifacts.extend([*views, grid, preview])
        videos.append({**metadata, "files": [_relative(p) for p in [*views, grid, preview]]})
        print(f"rendered {scenario_id}: {len(views)} views and grid", flush=True)
    monitor_plots = []
    for scene in metrics["scenarios"]:
        scenario_id = scene["scenario"]["scenario_id"]
        for name in ("distance_comparison.png", "gradient_comparison.png",
                     "minimum_clearance_comparison.png"):
            path = run / "pcc_monitor" / scenario_id / "plots" / name
            monitor_plots.append(_record(path))
    evidence_plot = ROOT / "v6_lite/output/v6_2_b2/online_complete_evidence_report_waiver_20260930/b2-online-evidence.png"
    repeated = ROOT / "v6_lite/output/v6_2_b2/online_dense_slices_repeated_timing_20260930/online_repeated_timing_summary.json"
    manifest = {
        "contract_version": CONTRACT,
        "source_run": _relative(run),
        "source_manifest": _record(run / "artifact_manifest.json"),
        "source_metrics": _record(run / "v6_lite_metrics.json"),
        "source_traces": [_record(path) for path in trace_paths],
        "acceptance": {"native_replay": _record(run / "validation.json"),
                       "execution_contract": _record(run / "execution_validation.json"),
                       "native_checks": "26/26", "execution_checks": "11/11"},
        "artifacts": [_record(path) for path in artifacts],
        "videos": videos,
        "latest_monitor_plots": monitor_plots,
        "summary_plot": _record(evidence_plot),
        "repeated_timing": _record(repeated),
        "historical_visualizations_are_excluded": True,
        "source_manifest_contract": source_manifest["contract_version"],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False)
                             + "\n", encoding="utf-8", newline="\n")
    if document is not None:
        _write_document(document, manifest)
    return manifest


def _write_document(path: Path, manifest: dict) -> None:
    output = ROOT / manifest["artifacts"][0]["path"]
    relative_output = output.parent.relative_to(ROOT).as_posix()
    lines = [
        "# V6.2-B.2 最新五场景可视化", "",
        "本页由 `python -m v6_lite.visualization.generate_b2_latest_full` 从已保存的最新正式运行自动生成。",
        f"数据源：`{manifest['source_run']}`；五个 27 秒场景；原生 MuJoCo 力矩重放 26/26、执行合同 11/11。",
        "图和视频使用该运行的五条 trace；墙钟计时取保存样本。历史 V6-lite/V6.1-B 图保留，仅作历史记录。", "",
        f"[完整清单](../{relative_output}/visualization_manifest.json) · [独立文件校验](../{relative_output}/visualization_validation.json) · [B.2 总证据](V6_2_B2_ONLINE_EVIDENCE.md)", "",
        "## 五场景总览", "",
        f"![五场景误差](../{relative_output}/error_curves.png)", "",
        f"![五场景安全间隙](../{relative_output}/safety_clearance_summary.png)", "",
        f"![完整控制计时](../{relative_output}/full_control_timing.png)", "",
        f"![求解候选与执行合同](../{relative_output}/execution_contract.png)", "",
        f"![基座漂移动画](../{relative_output}/base_pose_drift.gif)", "",
        "## 每个场景的路径与五视角回放", "",
    ]
    for video in manifest["videos"]:
        sid = video["scenario_id"]
        monitor = f"../{manifest['source_run']}/pcc_monitor/{sid}/plots"
        lines.extend([
            f"### {sid}", "",
            f"![{sid} 双臂路径](../{relative_output}/{sid}_tracking_paths_3d.png)", "",
            f"![{sid} 五视角预览](../{relative_output}/videos/{sid}_five_view_preview.png)", "",
            f"[五视角组合视频](../{relative_output}/videos/{sid}_five_view_grid.mp4) · "
            + " · ".join(f"[{view}](../{relative_output}/videos/{sid}_{view}.mp4)"
                            for view in VIEWS), "",
            f"最新 PCC 对照：[距离]({monitor}/distance_comparison.png) · "
            f"[梯度]({monitor}/gradient_comparison.png) · "
            f"[最小间隙]({monitor}/minimum_clearance_comparison.png)", "",
        ])
    lines.extend([
        "## 最新 PCC 对照及计时边界", "",
        "五个场景各自的距离、梯度、最小间隙图来自同一正式运行的 `pcc_monitor/`，逐图 SHA-256 见清单。",
        f"![B.2 在线证据与重复计时](../{manifest['summary_plot']['path']})", "",
        f"[三轮重复计时原始摘要](../{manifest['repeated_timing']['path']})", "",
        "完整五场景运行的 p95 最大值低于 20 ms；三轮 6 秒重复计时 p95 超过 20 ms。图中的计时是观测值，不构成稳定硬实时或连续时间安全证明。", "",
        "重建：", "", "```powershell",
        "python -m v6_lite.visualization.generate_b2_latest_full --output-dir v6_lite/visualization/output_v6_2_b2_rebuild",
        "python -m v6_lite.visualization.generate_b2_latest_full --output-dir v6_lite/visualization/output_v6_2_b2_rebuild --validate-only",
        "```", "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8", newline="\n")


def validate(output: Path = OUTPUT) -> dict:
    manifest = _load(output / "visualization_manifest.json")
    if manifest["contract_version"] != CONTRACT or len(manifest["videos"]) != 5:
        raise ValueError("unexpected visualization contract or scene count")
    records = ([manifest["source_manifest"], manifest["source_metrics"],
                *manifest["source_traces"], *manifest["acceptance"].values(),
                *manifest["artifacts"], *manifest["latest_monitor_plots"],
                manifest["summary_plot"], manifest["repeated_timing"]])
    records = [item for item in records if isinstance(item, dict) and "path" in item]
    failures = []
    for item in records:
        path = ROOT / item["path"]
        if not path.is_file() or path.stat().st_size != item["bytes"] or _sha(path) != item["sha256"]:
            failures.append(item["path"])
    for video in manifest["videos"]:
        if (video["source_trace_sha256"]
                != next(item["sha256"] for item in manifest["source_traces"]
                        if item["path"] == video["source_trace"])):
            failures.append(f"{video['scenario_id']}: trace binding")
        for relative in video["files"]:
            path = ROOT / relative
            if path.suffix == ".png":
                with Image.open(path) as image:
                    if image.size != (1920, 960):
                        failures.append(f"{relative}: preview dimensions")
                continue
            probe = subprocess.run(
                ["ffprobe", "-v", "error", "-select_streams", "v:0",
                 "-show_entries", "stream=nb_frames,width,height", "-of", "json", str(path)],
                capture_output=True, text=True, check=True)
            stream = json.loads(probe.stdout)["streams"][0]
            expected_size = (1920, 960) if "five_view_grid" in path.name else (640, 480)
            if (int(stream["nb_frames"]) != video["frame_count"]
                    or (stream["width"], stream["height"]) != expected_size):
                failures.append(f"{relative}: video frames or dimensions")
    result = {"contract_version": "v6_2_b2_full_visualization_audit_v1",
              "passed": not failures, "source_files_and_artifacts_checked": len(records),
              "scenario_count": len(manifest["videos"]),
              "video_count": sum(len(video["files"]) - 1 for video in manifest["videos"]),
              "failures": failures}
    (output / "visualization_validation.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8", newline="\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=RUN)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument("--document", type=Path)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    if not args.validate_only:
        document = (args.document.resolve() if args.document is not None
                    else DOCUMENT if args.output_dir.resolve() == OUTPUT else None)
        generate(args.run.resolve(), args.output_dir.resolve(), document)
    result = validate(args.output_dir.resolve())
    print(json.dumps(result, indent=2), flush=True)
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
