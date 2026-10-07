"""Check and seal the saved-state B.2 viewer without executing experiment code.

``--seal`` reruns validation and replaces only the two publication controls.
``--verify`` is read-only, rejects missing/unexpected files, and rechecks media.
An ordinary clone can verify source identities against the portable release;
omitted heavy source arrays are explicitly unavailable, never called decoded.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import subprocess

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
RUN = "task_anchored_residual_20261007_01"
SOURCE = ROOT / "v6_4/output" / RUN
OUTPUT = ROOT / "v6_4/visualization" / RUN
RELEASE = ROOT / "v6_4/releases" / RUN
MANIFEST_SHA = "95e23bfef59880d2061c6c33d0e00cfa88daef1cc8c58e4b66a7ef2c46cec2f7"
VERIFICATION_SHA = "3a7f9dc2c608d4353a64c5666a659c63ef4e885c32c4b0cc5fe166460aa7a176"
CONTROLS = {"visualization_manifest.json", "visualization_validation.json"}
SLOTS = tuple(f"TEST_{i:02d}_{m}" for i in range(4) for m in ("E0", "E1", "E2")) + ("teacher_00", "teacher_17")
VIEWS = {"overview", "front", "side", "top", "iso", "continuum_focus", "five_view_grid"}
BINDINGS = (
    "README.md", "docs/V6_4_B2_VISUALIZATION.md", "docs/V6_2_LATEST_VISUALIZATION.md",
    "v6_lite/README.md", "v6_lite/visualization/index.html", "v6_4/visualization/index.html",
    "v6_4/visualization/build_residual_figures.py", "v6_4/visualization/render_residual_replays.py",
    "v6_4/visualization/build_residual_dashboard.py", "v6_4/visualization/validate_residual_visualization.py",
    "v6_4/visualization/web/index.html", "v6_4/visualization/web/style.css", "v6_4/visualization/web/app.js",
    "v6_4/export_residual_release.py", f"v6_4/releases/{RUN}/release_manifest.json",
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def identity(path):
    path = Path(path)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"sha256": digest.hexdigest(), "bytes": path.stat().st_size}


def check(path, expected):
    require(Path(path).is_file(), f"Missing file: {path}")
    require(identity(path) == {key: expected[key] for key in ("sha256", "bytes")},
            f"Hash/size mismatch: {path}")


def relative_file(root, relative):
    pure = PurePosixPath(relative)
    require(bool(pure.parts) and not pure.is_absolute() and not any(p in ("..", ".") for p in pure.parts)
            and ":" not in relative and "\\" not in relative, f"Unsafe relative path: {relative}")
    path = Path(root).joinpath(*pure.parts)
    require(path.resolve().is_relative_to(Path(root).resolve()), f"Path escapes root: {relative}")
    return path


def inventory(output):
    return {p.relative_to(output).as_posix(): identity(p)
            for p in sorted(output.rglob("*")) if p.is_file() and p.relative_to(output).as_posix() not in CONTROLS}


class Sources:
    def __init__(self, source, release):
        from v6_4.export_residual_release import verify as verify_release
        self.release_result = verify_release(release)
        self.source, self.release = source, release
        self.release_manifest = read(release / "release_manifest.json")
        self.portable = read(release / "portable_paths.json")
        self.manifest_path = release / "snapshot/manifest.json"
        require(identity(self.manifest_path)["sha256"] == MANIFEST_SHA, "Unexpected frozen source manifest")
        require(identity(release / "snapshot/verification.json")["sha256"] == VERIFICATION_SHA,
                "Unexpected frozen source verification")
        self.manifest = read(self.manifest_path)
        self.checked, self.omitted, self.decoded = set(), set(), set()
        self.local_full_seal_checked = False
        if source.exists():
            payload = dict(self.manifest["payload"])
            payload["manifest.json"] = identity(self.manifest_path)
            payload["verification.json"] = identity(release / "snapshot/verification.json")
            actual = {p.relative_to(source).as_posix() for p in source.rglob("*") if p.is_file()}
            require(actual == set(payload), "Local sealed source has missing/unexpected files")
            for relative, expected in payload.items():
                check(relative_file(source, relative), expected)
            # These are original absolute paths, checked only on the original host.
            for original, expected in self.manifest["external"].items():
                check(Path(original), expected)
            self.local_full_seal_checked = True

    def resolve(self, relative, expected=None, needed=False):
        record = self.portable["artifacts"].get(relative)
        require(record is not None, f"Source absent from original seal: {relative}")
        if expected is not None:
            require(all(expected[k] == record[k] for k in ("sha256", "bytes")),
                    f"Source identity is not bound to original seal: {relative}")
        candidate = relative_file(self.source, relative)
        if candidate.is_file():
            check(candidate, record)
        elif record["release_relative_path"] is not None:
            candidate = relative_file(self.release, record["release_relative_path"])
            check(candidate, record)
        else:
            require(record["kind"] == "OMITTED_HEAVY", f"Invalid unavailable source: {relative}")
            self.omitted.add(relative)
            require(not needed, f"Required source is omitted from clone: {relative}")
            return None
        self.checked.add(relative)
        return candidate

    def json(self, relative):
        return read(self.resolve(relative, needed=True))

    def repository_record(self, record):
        prefix = f"v6_4/output/{RUN}/"
        require(record["path"].startswith(prefix), "Source record escaped frozen output")
        return self.resolve(record["path"][len(prefix):], record)


def check_image(path):
    with Image.open(path) as image:
        image.load()
        array = np.asarray(image.convert("RGB"))
        require(min(image.size) >= 100 and float(array.std()) > 2., f"Empty/invalid image: {path}")
        return {"width": image.width, "height": image.height, "pixel_std": float(array.std())}


def check_pdf(path):
    pdfinfo = shutil.which("pdfinfo")
    require(pdfinfo is not None, "pdfinfo is required to validate every PDF page")
    result = subprocess.run([pdfinfo, str(path)], capture_output=True, text=True, errors="replace", check=True)
    require(any(line.strip() == "Pages:           1" or (line.startswith("Pages:") and line.split()[-1] == "1")
                for line in result.stdout.splitlines()), f"PDF must contain one valid page: {path}")


def check_video(job):
    path, record, metadata, view = job
    check(path, record)
    command = [shutil.which("ffprobe"), "-v", "error", "-select_streams", "v:0", "-show_entries",
               "stream=width,height,nb_frames,avg_frame_rate,duration", "-of", "json", str(path)]
    stream = json.loads(subprocess.check_output(command, text=True))["streams"][0]
    width, height = metadata["width"], metadata["height"]
    if view == "five_view_grid":
        width, height = width * 3, height * 2
    count, fps = metadata["frame_count"], metadata["fps"]
    require(int(stream["width"]) == width and int(stream["height"]) == height
            and int(stream["nb_frames"]) == count and float(Fraction(stream["avg_frame_rate"])) == fps
            and abs(float(stream["duration"]) - count / fps) < .002, f"Encoded schedule mismatch: {path}")
    require(stream == record["probe"], f"Recorded ffprobe identity mismatch: {path}")
    picks = (0, count // 2, count - 1)
    selection = "+".join(f"eq(n\\,{n})" for n in picks)
    # Decode actual frame indices, then shrink only for bounded-memory pixel QA.
    decode = [shutil.which("ffmpeg"), "-v", "error", "-threads", "1", "-i", str(path),
              "-vf", f"select={selection},scale=160:120", "-vsync", "0", "-frames:v", "3",
              "-f", "rawvideo", "-pix_fmt", "rgb24", "-threads", "1", "-"]
    raw = subprocess.check_output(decode)
    require(len(raw) == 3 * 160 * 120 * 3, f"Cannot decode first/middle/last frames: {path}")
    frames = np.frombuffer(raw, np.uint8).reshape(3, 120, 160, 3).astype(float)
    std = [float(f.std()) for f in frames]
    require(min(std) > 2., f"Blank decoded frame: {path}")
    # Exclude the text strip. A tiny threshold rejects frozen/duplicated replays
    # without penalizing real small motions in a wide camera.
    strip = 18 if view == "five_view_grid" else 35
    motion = [float(np.mean(np.abs(frames[a, strip:] - frames[b, strip:]))) for a, b in ((0, 1), (1, 2), (0, 2))]
    require(max(motion) > .01, f"All sampled geometry frames are static: {path}")
    return {"path": record["relative_path"], "probe": stream, "decoded_frame_indices": list(picks),
            "decoded_frame_pixel_std": std, "geometry_mean_absolute_frame_differences": motion}


def validate(output, source, release, workers):
    require(shutil.which("ffmpeg") and shutil.which("ffprobe"), "ffmpeg and ffprobe are required")
    sources = Sources(source, release)
    plots, dashboard, videos = (read(output / name) for name in ("plot_data.json", "dashboard_data.json", "videos.json"))
    analysis = sources.json("delivery_analysis.json")
    summary = sources.json("summary.json")
    require(plots["source_manifest_sha256"] == MANIFEST_SHA, "Plot source manifest mismatch")
    require(len(plots["panels"]) == 30 and len(plots["source_files"]) == 120, "Unexpected plot/source counts")
    require(len(plots["test_runs"]) == len(analysis["TEST_rows"]) == 12, "Plot TEST row count changed")
    for plotted, original in zip(plots["test_runs"], analysis["TEST_rows"]):
        for key, value in original.items():
            if key == "execution_failure" and value is not None:
                # The public plot data intentionally omits the local-path traceback.
                value = {field: value[field] for field in ("type", "message")}
            require(plotted.get(key) == value, f"Plot TEST result field changed: {original['slot_id']}/{key}")
    require(plots["training"] == sources.json("training/model/training_report.json"), "Training plot data changed")
    require(plots["training"]["optimizer_updates"] == 4000 and plots["training"]["selected_update"] == 250
            and sum(plots["training"]["sample_exposures"].values()) == 128000
            and all(n == 0 for k, n in plots["training"]["sample_exposures"].items() if "_val_" in k),
            "Training/VAL exposure scope mismatch")
    original_table = sources.json("table_B.json")
    require(plots["table_B"] == [{key: row[key] for key in plots["table_B"][i]} for i, row in enumerate(original_table)],
            "Outcome table differs from original")
    require(len(dashboard["TEST_rows"]) == 12, "Dashboard must show all 12 fixed TEST attempts")
    for row, original in zip(dashboard["TEST_rows"], analysis["TEST_rows"]):
        require(row == {key: original.get(key) for key in row}, "Dashboard row changed from sealed result")
    require(dashboard["verdict"] == summary["verdict"] and plots["verdict"] == summary["verdict"], "Verdict changed")
    for field in ("additional_physics_steps", "additional_optimizer_updates", "additional_inference_calls", "additional_samples"):
        require(plots[field] == 0, f"Nonzero visualization work budget: {field}")
    require(len(plots["raw_candidates"]) == 16 and sum(row["accepted"] for row in plots["raw_candidates"]) == 14,
            "Raw candidate/rejection count mismatch")
    for row in plots["raw_candidates"]:
        raw = sources.json(row["source_path"])
        require(row["z_m"] == raw["raw_z_m"] and row["slot"] == raw["slot"] and not raw["inference_repair"],
                "Raw candidate changed or repaired")
        amplitude = float(np.max(np.linalg.norm(np.asarray(row["z_m"]), axis=1)))
        require(abs(amplitude - row["maximum_row_norm_m"]) < 1e-14 and row["accepted"] == (amplitude <= .02),
                "Raw amplitude/acceptance mismatch")
    for row in plots["source_files"]:
        sources.resolve(row["path"], row)
    image_results = {}
    panel_paths = set()
    for panel in plots["panels"]:
        require(panel["path"] not in panel_paths, "Duplicate plot panel")
        panel_paths.add(panel["path"])
        image_results[panel["path"]] = check_image(relative_file(output, panel["path"]))
        check_pdf(relative_file(output, panel["pdf_path"]))
        for relative in panel["source_paths"]:
            sources.resolve(relative)
    require(videos["slot_order"] == list(SLOTS) and videos["record_count"] == 14 and videos["video_count"] == 98,
            "Replay collection is not the complete fixed 14-slot set")
    for field in ("physics_steps_executed", "optimizer_updates", "new_samples"):
        require(videos[field] == 0, f"Nonzero replay work budget: {field}")
    require(not videos["historical_outputs_overwritten"] and not videos["smoke_truncated"], "Historical/smoke scope mismatch")
    check(ROOT / videos["generator"]["path"], videos["generator"])
    require(videos["source_manifest"]["sha256"] == MANIFEST_SHA, "Replay source manifest mismatch")
    jobs, slot_results = [], []
    for expected_slot, metadata in zip(SLOTS, videos["records"]):
        slot = metadata["slot_id"]
        require(slot == expected_slot and read(output / "replays" / slot / "replay_metadata.json") == metadata,
                "Collection metadata differs from per-slot metadata")
        result = sources.json(f"attempts/{slot}/attempt_result.json")
        for key in ("slot_id", "task_id", "task_sha256", "status", "full_task_success", "execution_failure"):
            require(metadata[key] == result.get(key), f"Replay outcome changed: {slot}/{key}")
        end = 16.62 if slot == "TEST_01_E2" else 14.9 if slot == "teacher_17" else 27.
        require(abs(metadata["source_end_s"] - end) < 1e-10 and metadata["source_start_s"] == 0,
                f"Incorrect saved replay endpoint: {slot}")
        require(result["actual_steps"] + 1 == metadata["source_state_count"], f"Source state count mismatch: {slot}")
        indices = np.asarray(metadata["selected_state_indices"])
        times = np.asarray(metadata["selected_saved_times_s"])
        presentation = np.asarray(metadata["presentation_times_s"])
        count, state_count = metadata["frame_count"], metadata["source_state_count"]
        require(len(indices) == len(times) == len(presentation) == count and np.issubdtype(indices.dtype, np.integer)
                and np.all(np.isfinite(times)) and np.all(np.isfinite(presentation))
                and indices[0] == 0 and indices[-1] == state_count - 1 and np.all(np.diff(indices) >= 0)
                and np.all((indices >= 0) & (indices < state_count)) and times[0] == presentation[0] == 0
                and abs(times[-1] - end) < 1e-10 and abs(presentation[-1] - end) < 1e-10
                and np.all(np.diff(times) >= 0) and np.all(np.diff(presentation) > 0)
                and np.all(times <= end + 1e-10) and np.all(presentation <= end + 1e-10)
                and np.all(times + 1e-10 >= presentation) and np.all(times - presentation <= .002 + 1e-9),
                f"Saved-state selection crosses its bound: {slot}")
        require(metadata["last_selected_state_index"] == state_count - 1 and abs(metadata["last_selected_time_s"] - end) < 1e-10
                and metadata["exact_source_endpoint_included"] and metadata["no_state_after_saved_endpoint"]
                and not metadata["smoke_truncated"], f"Missing/extrapolated exact endpoint: {slot}")
        require(metadata["deployment"] == "NOT_MET" and not metadata["visualization_is_new_acceptance"]
                and all(metadata[k] == 0 for k in ("physics_steps_executed", "optimizer_updates", "new_samples")),
                f"Replay research scope mismatch: {slot}")
        require(set(metadata["videos"]) == set(metadata["video_records"]) == VIEWS, f"Incomplete camera set: {slot}")
        for role, record in metadata["sourcepaths"].items():
            resolved = sources.repository_record(record)
            if role == "fresh_replay" and resolved is not None:
                with np.load(resolved, allow_pickle=False) as saved:
                    saved_times = saved["time"]
                    require(len(saved_times) == state_count and np.array_equal(saved_times[indices], times),
                            f"Selected clocks do not match actual saved array: {slot}")
                sources.decoded.add(record["path"])
        for key in ("preview", "focus_preview", "last_saved_state"):
            path = relative_file(output, metadata[key])
            check(path, metadata[f"{key}_record"])
            image_results[metadata[key]] = check_image(path)
        for gif in metadata["gifs"]:
            path = relative_file(output, gif["relative_path"])
            check(path, gif)
            image_results[gif["relative_path"]] = check_image(path)
            require(gif["source_start_s"] == 3. and gif["source_end_s"] == 9., "GIF scope changed")
        for view, record in metadata["video_records"].items():
            require(metadata["videos"][view] == record["relative_path"], "Video selector binding mismatch")
            jobs.append((relative_file(output, record["relative_path"]), record, metadata, view))
        slot_results.append({"slot_id": slot, "source_end_s": end, "frame_count": count,
                             "full_task_success": metadata["full_task_success"], "exact_endpoint_checked": True})
    with ThreadPoolExecutor(max_workers=workers) as pool:
        media_results = list(pool.map(check_video, jobs))
    return {"schema": "v64_b2_visualization_validation_v1", "status": "PASS", "checked_utc": datetime.now(timezone.utc).isoformat(),
            "source_manifest_sha256": MANIFEST_SHA, "source_verification_sha256": VERIFICATION_SHA,
            "local_full_payload_and_external_seal_rechecked": sources.local_full_seal_checked,
            "source_identity_files_read_and_hashed": len(sources.checked), "source_omitted_heavy_identity_only": sorted(sources.omitted),
            "source_saved_time_arrays_decoded": sorted(sources.decoded),
            "source_scope": "Omitted heavy source files, when absent, are ledger identities only; no decoded-source claim.",
            "png_panels_checked": 30, "pdf_pages_checked": 30, "plot_source_bindings_checked": 120,
            "fixed_TEST_rows_checked": 12, "replay_slots_checked": 14, "mp4_files_probed_and_decoded": len(media_results),
            "decoded_video_frames_checked": len(media_results) * 3, "slot_results": slot_results,
            "image_checks": image_results, "video_checks": media_results,
            "new_physics_steps": 0, "new_optimizer_updates": 0, "new_inference_calls": 0, "new_samples": 0,
            "deployment": "NOT_MET", "learning_advantage": "not_established", "visualization_is_new_acceptance": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--seal", action="store_true")
    mode.add_argument("--verify", action="store_true")
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--release", type=Path, default=RELEASE)
    parser.add_argument("--workers", type=int, choices=range(1, 5), default=2)
    args = parser.parse_args()
    source, output, release = args.source.resolve(), args.output.resolve(), args.release.resolve()
    require(output != source and not output.is_relative_to(source), "Viewer must be outside sealed source")
    manifest_path, validation_path = (output / name for name in ("visualization_manifest.json", "visualization_validation.json"))
    before = inventory(output)
    bindings = {relative: identity(ROOT / relative) for relative in BINDINGS}
    if args.verify:
        manifest = read(manifest_path)
        require(manifest["artifacts"] == before, "Viewer has missing, unexpected, or changed files")
        require(manifest["repository_bindings"] == bindings, "Publication generator/document bindings changed")
        prior = read(validation_path)
        require(prior["status"] == "PASS" and prior["visualization_manifest_sha256"] == identity(manifest_path)["sha256"],
                "Stored validation does not identify the current manifest")
    result = validate(output, source, release, args.workers)
    require(inventory(output) == before and {relative: identity(ROOT / relative) for relative in BINDINGS} == bindings,
            "Viewer or publication bindings changed during validation")
    if args.seal:
        manifest = {"schema": "v64_b2_visualization_manifest_v1", "source_manifest_sha256": MANIFEST_SHA,
                    "source_verification_sha256": VERIFICATION_SHA, "artifacts": before, "repository_bindings": bindings,
                    "excluded_self_referential_control_files": sorted(CONTROLS), "complete_source_replay_evidence_in_git": False,
                    "new_physics_steps": 0, "new_optimizer_updates": 0, "new_inference_calls": 0, "new_samples": 0}
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        result["visualization_manifest_sha256"] = identity(manifest_path)["sha256"]
        validation_path.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("status", "local_full_payload_and_external_seal_rechecked",
          "png_panels_checked", "pdf_pages_checked", "plot_source_bindings_checked", "replay_slots_checked",
          "mp4_files_probed_and_decoded", "decoded_video_frames_checked", "new_physics_steps")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
