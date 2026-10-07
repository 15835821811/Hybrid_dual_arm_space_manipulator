"""Independently validate the append-only B3.1 media supplement.

Reads saved arrays, CSVs, encoded media and source text only.  It never imports
the renderer, research modules, MuJoCo or a controller.  The scientific source
manifest is an immutable identity anchor; mutable current README paths in its
external inventory are deliberately not treated as current-seal evidence.
"""
from __future__ import annotations

import argparse
import ast
from concurrent.futures import ThreadPoolExecutor
import csv
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
from urllib.parse import unquote, urlsplit

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
RUN = "execution_aware_route_teacher_20261007_01"
SOURCE_MANIFEST_SHA = "6903dcc58ee793bb576d59b71732d6c73af2440e9e0d96b7950e6eeae8fbe9e5"
SOURCE_VERIFICATION_SHA = "8d0ae7b511bfc64728c2685b546b96f69cefac03ba395c5dfa0f400da8187330"
ACTUAL_PRODUCER = "d4464c8ae2aa7913a730ebbd775917e7a3b1af71"
SLOTS = [f"EA_{i:02d}" for i in range(28)]
VIEWS = {"overview", "front", "side", "top", "iso", "continuum_focus", "five_view_grid"}
CONTROLS = {"media_manifest.json", "validation.json"}


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


def match(path, expected):
    require(Path(path).is_file(), f"Missing file: {path}")
    actual = identity(path)
    require(actual == {k: expected[k] for k in ("sha256", "bytes")}, f"Hash/bytes mismatch: {path}")
    return actual


def relative_file(root, relative):
    require(isinstance(relative, str) and relative, "Empty/non-string relative path")
    pure = PurePosixPath(relative)
    require(not pure.is_absolute() and ":" not in relative and "\\" not in relative
            and all(p not in (".", "..") for p in pure.parts), f"Unsafe output path: {relative}")
    result = Path(root).joinpath(*pure.parts).resolve()
    require(result.is_relative_to(Path(root).resolve()), f"Path escapes output: {relative}")
    return result


def inventory(output):
    return {p.relative_to(output).as_posix(): identity(p) for p in sorted(output.rglob("*"))
            if p.is_file() and p.relative_to(output).as_posix() not in CONTROLS}


def numeric_equal(actual, expected, label, atol=1e-11):
    a, b = np.asarray(actual), np.asarray(expected)
    require(a.shape == b.shape, f"Shape mismatch: {label}: {a.shape} != {b.shape}")
    require(np.all(np.isfinite(a)) and np.all(np.isfinite(b)), f"Nonfinite numeric value: {label}")
    require(np.allclose(a, b, rtol=0., atol=atol), f"Numeric mismatch: {label}")
    return float(np.max(np.abs(a - b))) if a.size else 0.


class Sources:
    """Bind each used original to the old seal before any numeric decoding."""
    def __init__(self, source):
        self.source = Path(source).resolve()
        self.manifest_path = self.source / "manifest.json"
        require(identity(self.manifest_path)["sha256"] == SOURCE_MANIFEST_SHA, "Wrong scientific source seal")
        self.manifest = read(self.manifest_path)
        require(self.manifest["source_producer_commit"] == ACTUAL_PRODUCER, "Wrong actual producer")
        self.allowed = {str((self.source / p).resolve()).casefold(): v
                        for p, v in self.manifest["payload"].items()}
        self.allowed.update({str(Path(p).resolve()).casefold(): v for p, v in self.manifest["external"].items()})
        self.allowed[str(self.manifest_path).casefold()] = identity(self.manifest_path)
        verification = self.source / "verification.json"
        require(identity(verification)["sha256"] == SOURCE_VERIFICATION_SHA, "Wrong original seal verification")
        self.allowed[str(verification).casefold()] = identity(verification)
        self.checked = {}
        self.plan = self.json(self.source / "plan.json")
        require([r["slot_id"] for r in self.plan["slots"]] == SLOTS, "Source frozen slot order changed")

    def path(self, value):
        p = Path(value)
        return p.resolve() if p.is_absolute() else (ROOT / p).resolve()

    def bind(self, value, expected=None):
        path = self.path(value)
        key = str(path).casefold()
        require(key in self.allowed, f"Unsealed scientific source: {path}")
        bound = self.allowed[key]
        if expected is not None:
            require(all(expected[k] == bound[k] for k in ("sha256", "bytes")), f"Source identity not in old seal: {path}")
        if key not in self.checked:
            self.checked[key] = {"path": str(path), **match(path, bound)}
        return path

    def record(self, value):
        require(isinstance(value, dict) and all(k in value for k in ("path", "sha256", "bytes")), "Invalid source record")
        return self.bind(value["path"], value)

    def json(self, value):
        return read(self.bind(value))

    def records(self, values):
        require(isinstance(values, (list, dict)) and values, "Missing source records")
        result = {}
        iterable = enumerate(values) if isinstance(values, list) else values.items()
        for key, value in iterable:
            result[str(key)] = self.record(value)
        return result

    def unchanged(self):
        for row in self.checked.values():
            match(Path(row["path"]), row)


def generator(record, kind):
    path = Path(record["path"])
    path = path.resolve() if path.is_absolute() else (ROOT / path).resolve()
    match(path, record)
    tree = ast.parse(path.read_text(encoding="utf-8"))
    forbidden = {"mj_step", "mj_step1", "mj_step2", "mj_Euler", "mj_RungeKutta", "mj_geomDistance"}
    calls = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            name = node.func.attr if isinstance(node.func, ast.Attribute) else node.func.id if isinstance(node.func, ast.Name) else ""
            require(name not in forbidden, f"Forbidden integration/geometry call in {kind}: {name}")
            calls.append(name)
    return {"path": str(path), **identity(path), "direct_forbidden_calls": [],
            "mj_forward_call_sites": calls.count("mj_forward"),
            "guard_scope": "AST direct call check; generator is never imported or executed"}


def check_image(path):
    with Image.open(path) as image:
        image.load()
        pixels = np.asarray(image.convert("RGB"))
        require(min(image.size) >= 100 and float(pixels.std()) > 2., f"Empty/invalid PNG: {path}")
        return {"width": image.width, "height": image.height, "pixel_std": float(pixels.std())}


def check_pdf(path):
    require(shutil.which("pdfinfo"), "pdfinfo is required")
    result = subprocess.run([shutil.which("pdfinfo"), str(path)], capture_output=True, text=True, errors="replace", check=True)
    pages = [line for line in result.stdout.splitlines() if line.startswith("Pages:")]
    require(len(pages) == 1 and int(pages[0].split()[-1]) == 1, f"PDF is not one valid page: {path}")
    return {"pages": 1, "bytes": path.stat().st_size}


def check_video(job):
    output, metadata, view = job
    record = metadata["video_records"][view]
    relative = metadata["videos"][view]
    require(record["relative_path"] == relative, "Video path/selector disagreement")
    path = relative_file(output, relative)
    match(path, record)
    command = [shutil.which("ffprobe"), "-v", "error", "-show_entries",
               "stream=codec_type,codec_name,width,height,nb_frames,avg_frame_rate,duration", "-of", "json", str(path)]
    streams = json.loads(subprocess.check_output(command, text=True))["streams"]
    require(len(streams) == 1 and streams[0]["codec_type"] == "video", f"Unexpected audio/stream: {path}")
    stream = streams[0]
    count, fps = metadata["frame_count"], metadata["fps"]
    width, height = metadata["width"], metadata["height"]
    if view == "five_view_grid":
        width, height = width * 3, height * 2
    elif view == "continuum_focus":
        width, height = metadata.get("focus_width", 960), metadata.get("focus_height", 720)
    require(int(stream["width"]) == width and int(stream["height"]) == height
            and int(stream["nb_frames"]) == count and float(Fraction(stream["avg_frame_rate"])) == fps
            and stream["codec_name"] == "h264" and abs(float(stream["duration"]) - count / fps) < .002,
            f"Encoded dimensions/frame schedule mismatch: {path}")
    if "probe" in record:
        require(all(stream.get(k) == v for k, v in record["probe"].items()), f"Recorded ffprobe disagreement: {path}")
    picks = [0, count // 2, count - 1]
    selection = "+".join(f"eq(n\\,{n})" for n in picks)
    raw = subprocess.check_output([shutil.which("ffmpeg"), "-v", "error", "-threads", "1", "-i", str(path),
        "-vf", f"select={selection},scale=160:120", "-vsync", "0", "-frames:v", "3", "-f", "rawvideo",
        "-pix_fmt", "rgb24", "-threads", "1", "-"])
    require(len(raw) == 3 * 160 * 120 * 3, f"First/middle/last decode failed: {path}")
    frames = np.frombuffer(raw, np.uint8).reshape(3, 120, 160, 3).astype(float)
    std = [float(f.std()) for f in frames]
    require(min(std) > 2., f"Blank decoded frame: {path}")
    # Remove annotation strips in BOTH grid rows, so changing clock text cannot
    # make a frozen geometry video pass the motion test.
    geometry = np.concatenate((frames[:, 18:60], frames[:, 78:120]), axis=1) if view == "five_view_grid" else frames[:, 35:]
    differences = [float(np.mean(np.abs(geometry[a] - geometry[b]))) for a, b in ((0, 1), (1, 2), (0, 2))]
    require(max(differences) > .01, f"All sampled geometry frames are static: {path}")
    return {"path": relative, "probe": stream, "decoded_frame_indices": picks,
            "pixel_std": std, "geometry_mean_absolute_differences": differences}


def check_route_poster_pixels(job):
    output, poster = job
    path = relative_file(output, poster["path"])
    match(path, {"sha256": poster["png_sha256"], "bytes": poster["png_bytes"]})
    source = relative_file(output, poster["source_mp4"]["path"])
    width, height, frame = poster["width"], poster["height"], poster["frame_index"]
    raw = subprocess.check_output([shutil.which("ffmpeg"), "-v", "error", "-threads", "1", "-i", str(source),
        "-vf", f"select=eq(n\\,{frame})", "-vsync", "0", "-frames:v", "1", "-f", "rawvideo",
        "-pix_fmt", "rgb24", "-threads", "1", "-"])
    require(len(raw) == width*height*3, f"Cannot decode route poster source frame: {source}/{frame}")
    expected = np.frombuffer(raw, np.uint8).reshape(height, width, 3)
    with Image.open(path) as image:
        image.load()
        pixels = np.asarray(image.convert("RGB"))
        require(image.size == (width, height) and np.array_equal(pixels, expected), f"Route poster differs from exact encoded source frame: {path}")
    return {"path": poster["path"], "source_mp4": poster["source_mp4"]["path"], "frame_index": frame,
            "presentation_time_s": poster["presentation_time_s"], "actual_saved_time_s": poster["actual_saved_time_s"],
            "width": width, "height": height, "exact_decoded_RGB_pixel_parity": True}


def route_posters(output, videos, gallery, sources, workers, payload):
    path = output / "route_posters/poster_manifest.json"
    manifest = read(path)
    require(manifest["schema"] == "v64_b31_route_posters_v1" and manifest["record_count"] == 28
            and manifest["poster_count"] == 56 and [row["slot_id"] for row in manifest["records"]] == SLOTS,
            "Route poster manifest is not the fixed 28 x 2 cohort")
    require(all(manifest[field] == 0 for field in ("new_forward_calls", "new_physics_steps", "new_geometry_queries", "new_QP_solves", "new_model_calls"))
            and not manifest["existing_media_modified"] and manifest["source_MP4_and_metadata_unchanged"]
            and manifest["decoded_source_frames_extracted"] == 56, "Route poster extraction scope/cost changed")
    require(manifest["outputs"] == {p: row for p, row in payload.items() if p.startswith("route_posters/") and p != "route_posters/poster_manifest.json"},
            "Route poster manifest outputs do not exhaust their sealed directory")
    sources.record(manifest["source_plan"])
    require(sources.path(manifest["source_plan"]["path"]) == sources.source / "plan.json", "Route poster source plan is not frozen B3.1 plan")
    producer = generator(manifest["generator"], "route poster extractor")
    require(producer["mj_forward_call_sites"] == 0, "Poster extractor must decode encoded media without forwarding a model")
    match(relative_file(output, manifest["generator"]["source_copy"]), manifest["generator"])
    require(all(gallery["source_route_posters"][key] == identity(path)[key] for key in ("sha256", "bytes")),
            "Gallery does not bind the current route poster manifest bytes")
    expected_mapping, jobs, used = {}, [], set()
    for row, video, frozen in zip(manifest["records"], videos["records"], sources.plan["slots"]):
        sid = frozen["slot_id"]
        slot_receipt = read(output / "route_posters" / (sid+"_poster_receipt.json"))
        require(slot_receipt["schema"] == "v64_b31_route_poster_slot_v1" and slot_receipt["generator_sha256"] == producer["sha256"]
                and slot_receipt["record"] == row and all(slot_receipt[field] == 0 for field in ("new_forward_calls", "new_physics_steps", "new_geometry_queries")),
                "Route poster slot receipt differs from its sealed collection row")
        require(row["task_id"] == frozen["task_id"] and row["mode"] == frozen["mode"] and row["source_role"] == frozen["source_role"]
                and row["source_slot"] == (frozen["old_source_slot"] or sid), "Route poster task/mode/source alias changed")
        declaration = next(t for t in sources.plan["tasks"] if t["task_id"] == frozen["task_id"])
        interval = declaration["route_interval_s"]
        midpoint = (interval[0]+interval[1])/2.
        require(row["route_interval_s"] == interval and abs(row["route_midpoint_s"]-midpoint) < 1e-12,
                "Route poster window/midpoint differs from frozen task")
        metadata_path = relative_file(output, row["source_metadata"]["path"])
        require(metadata_path == output / "replays" / sid / "replay_metadata.json", "Route poster metadata belongs to another slot")
        match(metadata_path, row["source_metadata"])
        require(read(metadata_path) == video, "Route poster metadata differs from video collection")
        schedule = video["schedule"]
        frame = int(np.argmin(np.abs(np.asarray(schedule["presentation_times_s"])-midpoint)))
        require(frame == 153 and abs(schedule["presentation_times_s"][frame]-10.2) < 1e-12, "Unexpected nearest route presentation frame")
        require(set(row["posters"]) == {"five_view_grid", "continuum_focus"}, "Route poster view coverage changed")
        expected_mapping[sid] = {}
        for view, poster in row["posters"].items():
            record = video["video_records"][view]
            require(poster["path"] not in used, "Route poster path is shared across slot/views")
            used.add(poster["path"])
            require(poster["source_mp4"]["path"] == video["videos"][view]
                    and all(poster["source_mp4"][key] == record[key] for key in ("sha256", "bytes")), "Route poster source video identity changed")
            match(relative_file(output, poster["source_mp4"]["path"]), poster["source_mp4"])
            require(poster["frame_index"] == frame and poster["presentation_time_s"] == schedule["presentation_times_s"][frame]
                    and poster["selected_saved_index"] == schedule["selected_saved_indices"][frame]
                    and poster["actual_saved_time_s"] == schedule["selected_saved_times_s"][frame]
                    and interval[0] <= poster["actual_saved_time_s"] <= interval[1]
                    and poster["width"] == record["width"] and poster["height"] == record["height"], "Route poster encoded/saved clock or dimensions changed")
            expected_mapping[sid][view] = poster["path"]
            jobs.append((output, poster))
    require(used == {p.relative_to(output).as_posix() for p in (output / "route_posters").rglob("*.png")}, "Route poster PNG exact membership changed")
    require(manifest["route_posters"] == gallery["route_posters"] == expected_mapping, "Gallery/manifest route poster mapping differs")
    parser = Links()
    parser.feed((output / "index.html").read_text(encoding="utf-8"))
    require(expected_mapping["EA_00"]["five_view_grid"] in parser.poster_attributes
            and expected_mapping["EA_00"]["continuum_focus"] in parser.poster_attributes, "Initial gallery video posters do not use the authenticated route posters")
    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(check_route_poster_pixels, jobs))
    match(Path(producer["path"]), producer)
    return {"manifest": {"path": "route_posters/poster_manifest.json", **identity(path)}, "producer": producer,
            "record_count": 28, "poster_count": 56, "decoded_source_frames": 56, "checks": results}


class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []
        self.ids = []
        self.poster_attributes = []
        self.embedded_gallery = []
        self.in_gallery = False
    def handle_starttag(self, tag, attrs):
        self.links.extend(value for key, value in attrs if key in ("href", "src", "poster") and value)
        self.ids.extend(value for key, value in attrs if key == "id")
        self.poster_attributes.extend(value for key, value in attrs if key == "poster")
        if tag == "script" and dict(attrs).get("id") == "gallery-data":
            self.in_gallery = True
    def handle_endtag(self, tag):
        if tag == "script":
            self.in_gallery = False
    def handle_data(self, data):
        if self.in_gallery:
            self.embedded_gallery.append(data)


def html_links(output, receipt):
    results = []
    for path in sorted(output.rglob("*.html")):
        parser = Links()
        text = path.read_text(encoding="utf-8")
        parser.feed(text)
        require(len(parser.ids) == len(set(parser.ids)), f"Duplicate HTML element IDs: {path}")
        for element in re.findall(r"getElementById\(['\"]([^'\"]+)['\"]\)", text):
            require(element in parser.ids, f"Missing static JS element: {element}")
        require(not re.search(r"\b(fetch\s*\(|XMLHttpRequest|WebSocket\s*\()", text), "Unexpected gallery network dependency")
        if path == output / "index.html":
            require(json.loads("".join(parser.embedded_gallery)) == read(output / "gallery_data.json"), "Inline gallery JSON differs from bound gallery_data.json")
        for value in parser.links:
            url = urlsplit(value)
            require(not url.scheme and not url.netloc, f"External HTML dependency: {value}")
            if not url.path:
                continue
            target = (path.parent / unquote(url.path)).resolve()
            require(target.is_relative_to(ROOT), f"HTML link escapes repository: {value}")
            require(target.is_file() or target == output / "validation.json", f"Broken local HTML link: {value}")
            results.append({"html": path.relative_to(output).as_posix(), "reference": value,
                            "target": target.relative_to(ROOT).as_posix(), "deferred_receipt": not target.exists()})
    require(results, "No HTML links checked")
    return results


def load_arrays(path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def actual_parity(metadata, sources, original):
    paths = sources.records(metadata["sourcepaths"])
    for key in ("actual_trace", "fresh_replay", "task", "plan", "evaluation"):
        require(key in paths, f"Missing required original source: {key}")
    result = original["source_result"]
    require(identity(paths["actual_trace"])["sha256"] == result["trace_sha256"], "Trace not bound to original result")
    require(identity(paths["evaluation"])["sha256"] == result["evaluation_sha256"], "Evaluation not bound to original result")
    evaluation = read(paths["evaluation"])
    require(evaluation == result["evaluation"], "Embedded evaluation differs from original")
    require(identity(paths["fresh_replay"])["sha256"] == evaluation["fresh_replay_sha256"], "Fresh not bound to evaluation")
    require(identity(paths["plan"])["sha256"] == result["plan_file_sha256"], "Plan differs from actual consumed source")
    require(identity(paths["task"]) == identity(sources.bind(original["frozen_slot"]["task_path"])), "Actual task bytes differ from frozen canonical Task")
    task = read(paths["task"])
    plan = read(paths["plan"])
    require(metadata["reference_version"] == plan["representation_version"], "Displayed reference representation differs from original plan")
    trace, fresh = load_arrays(paths["actual_trace"]), load_arrays(paths["fresh_replay"])
    qpos = np.concatenate((trace["initial_qpos"].reshape(1, -1), trace["actual_full_qpos"]), axis=0)
    qvel = np.concatenate((trace["initial_qvel"].reshape(1, -1), trace["actual_full_qvel"]), axis=0)
    times = np.r_[0., trace["time"]]
    require(qpos.shape == (13501, 81) and qvel.shape == (13501, 79), "Unexpected saved actual state dimensions")
    require(np.array_equal(qpos, fresh["qpos"]) and np.array_equal(qvel, fresh["qvel"]), "Actual/fresh state parity fails")
    require(np.array_equal(qpos[0], task["initial_qpos"]) and np.array_equal(qvel[0], task["initial_qvel"]), "Actual initial state differs from frozen Task")
    time_error = numeric_equal(times, fresh["time"], "actual/fresh clock", 1e-9)
    require(np.all(np.diff(times) > 0.) and abs(times[-1] - 27.) < 1e-9, "Actual clock is not full 27 seconds")
    require(metadata["render_state_kind"] == "ACTUAL_TRACE_SAVED_STATES", "Render source is not actual trace")
    parity = metadata["actual_fresh_parity"]
    require(parity["qpos_bit_exact"] and parity["qvel_bit_exact"] and parity["state_count"] == 13501
            and parity["qpos_max_abs_difference"] == parity["qvel_max_abs_difference"] == 0.
            and parity["initial_state_matches_Task"] and not parity["fresh_replay_is_actual_state_source"]
            and abs(parity["time_max_abs_difference_s"]-time_error) < 1e-12 and parity["time_tolerance_s"] == 1e-9,
            "Metadata denies/misstates exact actual/fresh parity")
    for field, role in (("actual_saved_state_source", "actual_trace"), ("independent_replay_state_source", "fresh_replay")):
        require(all(metadata[field][k] == metadata["sourcepaths"][role][k] for k in ("path", "sha256", "bytes")), "Render-source provenance roles changed")
    cost = metadata["render_cost"]
    require(all(cost[key] == 0 for key in ("physics_steps", "geometry_queries", "QP_solves", "model_samples"))
            and cost["forbidden_call_attempts"] == [] and cost["mj_forward_calls"] >= metadata["frame_count"]
            and cost["native_view_frames"] == metadata["frame_count"]*6
            and cost["encoded_video_frames"] == metadata["frame_count"]*7, "Render-only guard/cost receipt mismatch")
    generated = independent_base(task, times) + independent_offset(plan, times)
    c_difference = numeric_equal(generated[1:], trace["generated_continuum_position"], "Generated C execution reference phase", 1e-12)
    grasp = fresh["target_position"] + np.einsum("nij,j->ni", fresh["target_rotation"], np.asarray(task["scenario"]["grasp_point_target_frame_m"]))
    r_difference = numeric_equal(grasp[:-1], trace["generated_rigid_position"], "Generated R cached PRESTEP phase", 1e-12)
    binding = metadata["reference_execution_binding"]
    require(abs(binding["continuum_poststep_generated_reference_max_abs_m"]-c_difference) < 1e-12
            and abs(binding["rigid_poststep_generated_cached_prestep_reference_max_abs_m"]-r_difference) < 1e-12
            and binding["poststep_saved_reference_samples"] == 13500 and binding["reference_tolerance_m"] == 1e-12,
            "Generated-reference binding numbers changed")
    require(metadata["model_contract_sha256"] == evaluation["native_geometry"]["model_contract_sha256"]
            and metadata["model_source_bundle_sha256"] == evaluation["native_geometry"]["model_source_bundle_sha256"], "Render model identity differs from independent original evaluation")
    require("historical_model_observation" in paths and metadata["source_compiled_model_sha256"]
            == read(paths["historical_model_observation"])["execution_contract"]["source_compiled_model_sha256"], "Render compiled model differs from original execution contract")
    return trace, fresh, times, {"state_count": len(times), "qpos_bit_exact": True, "qvel_bit_exact": True,
                               "time_max_abs_s": time_error, "sourcepaths": {k: str(v) for k, v in paths.items()}}


def frame_schedule(metadata, times):
    require(metadata["fps"] == 15 and metadata["frame_count"] == 406, "Expected 15 fps / 406 frames")
    presentation = np.arange(406, dtype=float) / 15.
    presentation[-1] = float(times[-1])
    indices = np.searchsorted(times, presentation, side="left")
    indices = np.minimum(indices, len(times) - 1)
    indices[0], indices[-1] = 0, len(times) - 1
    saved_schedule = metadata["schedule"]
    actual_indices = np.asarray(saved_schedule["selected_saved_indices"])
    require(np.issubdtype(actual_indices.dtype, np.integer) and np.array_equal(indices, actual_indices), "Saved frame selection differs from independent 15fps schedule")
    numeric_equal(saved_schedule["presentation_times_s"], presentation, "presentation clock", 1e-9)
    numeric_equal(saved_schedule["selected_saved_times_s"], times[indices], "selected actual clock", 1e-9)
    require(not metadata["smoke_truncated"] and metadata.get("includes_exact_saved_endpoint", metadata.get("exact_source_endpoint_included"))
            and not saved_schedule["smoke_truncated"] and saved_schedule["includes_exact_saved_endpoint"]
            and saved_schedule["source_state_count"] == 13501 and abs(metadata["source_end_s"] - 27.) < 1e-9
            and abs(metadata["displayed_saved_end_s"] - times[-1]) < 1e-12, "Truncated/missing exact endpoint")
    return {"frame_count": 406, "fps": 15, "first_state_index": 0, "last_state_index": 13500,
            "encoded_duration_s": 406 / 15., "saved_duration_s": float(times[-1])}


def check_csv(path):
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames
        require(fields and len(fields) == len(set(fields)), f"Invalid CSV headers: {path}")
        rows = list(reader)
    require(rows, f"Empty CSV: {path}")
    columns = {}
    for field in fields:
        try:
            values = np.asarray([float(row[field]) if row[field] != "" else np.nan for row in rows], dtype=float)
        except (ValueError, TypeError):
            raise ValueError(f"Non-numeric CSV field: {path}/{field}") from None
        require(not np.isinf(values).any() and (np.isfinite(values).all() or field.startswith("task_")),
                f"Unexpected unavailable/nonfinite CSV field: {path}/{field}")
        columns[field] = values
    return columns


def angle_deg(actual, reference):
    product = np.einsum("...ji,...jk->...ik", reference, actual)
    return np.rad2deg(np.arccos(np.clip((np.trace(product, axis1=-2, axis2=-1)-1.)/2., -1., 1.)))


def independent_base(task, times):
    target = task["scenario"]["continuum_target"]
    require(target["mode"] == "irregular_waypoints" and target["reference_profile"] == "minimum_jerk_c2", "Unknown Task/base definition")
    points = np.asarray(target["waypoint_points_m"])
    durations = np.asarray(target["segment_durations_s"])
    transition = float(target["transition_duration_s"])
    knots = np.r_[0., np.cumsum(durations)] + transition
    result = np.empty((len(times), 3), dtype=float)
    def segment(selected, start, duration, first, last):
        u = np.clip((times[selected]-start)/duration, 0., 1.)
        w = u*u*u*(10.+u*(-15.+6.*u))
        result[selected] = first + w[:, None]*(last-first)
    segment(times < transition, 0., transition, np.asarray(target["initial_position_w"]), points[0])
    for index, duration in enumerate(durations):
        selected = (times >= knots[index]) & ((times < knots[index+1]) if index < len(durations)-1 else True)
        segment(selected, knots[index], duration, points[index], points[index+1])
    return result


def independent_offset(plan, times):
    definition = plan["definition"]
    result = np.zeros((len(times), 3))
    for interval, enabled, basis, z in zip(definition["intervals_s"], definition["interval_mask"], definition["transverse_bases"], plan["z_m"]):
        if not enabled:
            continue
        start, end = interval
        selected = (times >= start) & (times <= end)
        u = np.clip((times[selected]-start)/(end-start), 0., 1.)
        if plan["representation_version"] == "task_anchored_cartesian_residual_v1":
            w = 64.*u**3*(1.-u)**3
        else:
            require(plan["representation_version"] == "task_anchored_cartesian_residual_v2", "Unknown residual version")
            rise = definition["basis_function"]["rise_ratio"]
            fall = definition["basis_function"]["fall_start_ratio"]
            v = np.where(u < rise, u/rise, np.where(u > fall, (1.-u)/(1.-fall), 1.))
            w = v**3*(10.-15.*v+6.*v*v)
        result[selected] += w[:, None]*(np.asarray(basis)@np.asarray(z))[None, :]
    return result


def statistics(values):
    values = np.asarray(values)
    return {"samples": int(values.size), "rms": float(np.sqrt(np.mean(values**2))), "mean": float(np.mean(values)),
            "maximum": float(values.max()), "minimum": float(values.min()), "last": float(values[-1])}


def check_stats(saved, values, label):
    expected = statistics(values)
    require(set(saved) == set(expected) and saved["samples"] == expected["samples"], f"Summary sample denominator changed: {label}")
    for key in expected:
        require(abs(saved[key]-expected[key]) < 1e-8, f"Summary statistic mismatch: {label}/{key}")


def compare_columns(actual, expected, picks, label):
    require(set(actual) == set(expected), f"CSV column/unit contract changed: {label}: {set(actual)^set(expected)}")
    maximum = 0.
    for key, values in expected.items():
        a, b = actual[key], np.asarray(values)[picks]
        require(a.shape == b.shape and np.array_equal(np.isfinite(a), np.isfinite(b)), f"Unavailable cell mask mismatch: {label}/{key}")
        finite = np.isfinite(b)
        maximum = max(maximum, numeric_equal(a[finite], b[finite], f"{label}/{key}", 1e-8))
    return maximum


def plot_csv_contract(output, plotted, trace, fresh, times, sources, frozen, original):
    """Recompute every exported numeric column and all full-density summaries."""
    norm = lambda value: np.linalg.norm(value, axis=-1)
    task = sources.json(frozen["task_path"])
    records = sources.records(plotted["source_records"])
    routes = [p for p in records.values() if p.name == "route_clearance.npz"]
    require(len(routes) == 1, "Missing/ambiguous saved route-clearance source")
    clearance = load_arrays(routes[0])
    t, tt = fresh["time"], trace["task_time"][:1350]
    numeric_equal(clearance["time"], t, "clearance native clock", 1e-9)
    indices = np.rint(tt/.002).astype(int)
    numeric_equal(t[indices], tt, "consumed/fresh exact tick alignment", 1e-9)
    base = independent_base(task, t)
    q, q0 = fresh["base_pose"][:, 3:], np.asarray(task["base_pose"][3:])
    dots = np.abs((q@q0)/(norm(q)*norm(q0)))
    scenario = task["scenario"]
    grasp = fresh["target_position"] + np.einsum("nij,j->ni", fresh["target_rotation"], np.asarray(scenario["grasp_point_target_frame_m"]))
    grasp_R = fresh["target_rotation"] @ np.asarray(scenario["grasp_rotation_target_frame"])
    native = {"time_s": t, "fresh_state_index": np.arange(13501),
        "base_translation_drift_m": norm(fresh["base_pose"][:, :3]-task["base_pose"][:3]),
        "base_orientation_drift_deg": np.rad2deg(2*np.arccos(np.clip(dots, 0., 1.))),
        "robot_target_minimum_m": fresh["target_minimum_m"], "robot_target_minimum_censored": fresh["target_minimum_censored"],
        "continuum_sphere_minimum_m": clearance["continuum_obstacle_minimum_m"],
        "continuum_actual_minus_task_base_m": norm(fresh["continuum_position"]-base),
        "continuum_orientation_to_Task_deg": angle_deg(fresh["continuum_rotation"], np.broadcast_to(np.asarray(scenario["continuum_target_rotation_world"]), (13501, 3, 3))),
        "rigid_position_to_terminal_Task_grasp_m": norm(fresh["rigid_position"]-grasp),
        "rigid_orientation_to_terminal_Task_grasp_deg": angle_deg(fresh["rigid_rotation"], grasp_R)}
    consumed = {"time_s": tt, "fresh_state_index": indices,
                "original_17D_intervention_rad_s": trace["task_avoidance_intervention"][:1350]}
    for arm in ("continuum", "rigid"):
        reference = trace["task_input_reference_"+arm+"_target_position"][:1350]
        rotation = trace["task_input_reference_"+arm+"_target_rotation"][:1350]
        consumed[arm+"_position_tracking_error_m"] = norm(fresh[arm+"_position"][indices]-reference)
        consumed[arm+"_orientation_tracking_error_deg"] = angle_deg(fresh[arm+"_rotation"][indices], rotation)
        for axis, axis_name in enumerate("xyz"):
            native[arm+"_actual_"+axis_name+"_m"] = fresh[arm+"_position"][:, axis]
            consumed[arm+"_actual_"+axis_name+"_m"] = fresh[arm+"_position"][indices, axis]
            consumed[arm+"_consumed_reference_"+axis_name+"_m"] = reference[:, axis]
    for axis, axis_name in enumerate("xyz"):
        native["continuum_Task_base_"+axis_name+"_m"] = base[:, axis]
    vector_available = all(k in trace for k in ("task_qp_box_nominal_velocity", "task_qp_selected_velocity"))
    require(vector_available == (frozen["old_source_slot"] is None), "Old/new QP-vector availability scope changed")
    if vector_available:
        nominal, selected = trace["task_qp_box_nominal_velocity"][:1350], trace["task_qp_selected_velocity"][:1350]
        require(nominal.shape == selected.shape == (1350, 17), "Saved vector dimensions are not original 17D")
        numeric_equal(norm(selected-nominal), consumed["original_17D_intervention_rad_s"], "Original scalar/17D parity", 1e-9)
        consumed["box_nominal_17D_norm_rad_s"], consumed["selected_17D_norm_rad_s"] = norm(nominal), norm(selected)
    for key in ("task_pcc_clearance", "task_capsule_clearance", "task_interval_current_envelope_margin_m", "task_interval_ramp_minimum_envelope_margin_m"):
        if key in trace:
            consumed[key] = trace[key][:1350]
    torque = trace["torque"]
    require(torque.shape == (13500, 67), "Unexpected torque channel/interval denominator")
    control = {"applied_interval_end_time_s": trace["time"], "applied_interval_start_time_s": trace["time"]-.002,
        "actual_torque_L2_67channels_Nm": norm(torque), "actual_torque_max_abs_67channels_Nm": np.max(np.abs(torque), axis=1),
        "actual_torque_saturation_count": trace["torque_saturation_count"]}
    native_rows = np.r_[np.arange(0, 13501, 10)]
    errors = {}
    for role, expected, picks in (("native", native, native_rows), ("consumed", consumed, np.arange(1350)), ("control", control, np.arange(9, 13500, 10))):
        errors[role] = compare_columns(check_csv(relative_file(output, plotted["csv"][role])), expected, picks, plotted["slot_id"]+"/"+role)
    summary = plotted["summary"]
    quality = original["quality"]
    declaration = next(row for row in sources.plan["tasks"] if row["task_id"] == frozen["task_id"])
    interval = declaration["route_interval_s"]
    require(summary["original_full_quality_vector"] == quality["full_metrics"] and summary["original_independent_Task_requirement_results"] == original["source_result"]["evaluation"]["task_requirements"], "Original metric/Task requirement object changed")
    require(summary["route_interval_s"] == interval and summary["task_horizon_s"] == [0., 27.] and summary["fresh_states"] == 13501
            and summary["consumed_reference_samples"] == 1350 and summary["actual_steps"] == 13500, "Summary scope changed")
    route_t, route_tt = ((clock >= interval[0]-1e-9) & (clock <= interval[1]+1e-9) for clock in (t, tt))
    expected_tracking = {k for k in consumed if "tracking_error" in k}
    require(set(summary["consumed_reference_tracking"]) == expected_tracking, "Tracking summary field set changed")
    for key in expected_tracking:
        check_stats(summary["consumed_reference_tracking"][key]["whole_task"], consumed[key], key+"/whole")
        check_stats(summary["consumed_reference_tracking"][key]["route"], consumed[key][route_tt], key+"/route")
    expected_native = {k for k in native if "drift" in k or "minimum_m" in k or "minus_task_base" in k}
    require(set(summary["native_diagnostics"]) == expected_native, "Native summary field set changed")
    for key in expected_native:
        check_stats(summary["native_diagnostics"][key]["whole_task"], native[key], key+"/whole")
        check_stats(summary["native_diagnostics"][key]["route"], native[key][route_t], key+"/route")
    diagnostics = summary["control_diagnostics"]
    require(diagnostics["QP_vectors"] == ("AVAILABLE_SAVED_17D_VECTORS" if vector_available else "NOT_MEASURED_OLD_EVIDENCE"), "QP vector missingness label changed")
    scalar = consumed["original_17D_intervention_rad_s"]
    check_stats(diagnostics["original_scalar_17D_intervention_rad_s"]["whole_task"], scalar, "scalar/whole")
    check_stats(diagnostics["original_scalar_17D_intervention_rad_s"]["route"], scalar[route_tt], "scalar/route")
    for key, values in control.items():
        if key.startswith("actual_torque"):
            check_stats(diagnostics["native_applied_torque"][key], values, key+"/all13500")
    for key in consumed:
        if key.startswith("task_"):
            record = diagnostics["existing_interval_PCC_fields"][key]
            require(record["saved_samples"] == 1350 and record["finite_samples"] == int(np.isfinite(consumed[key]).sum())
                    and record["nonfinite_CSV_cells"] == "empty, not zero", "Existing diagnostic missingness changed")
    return {"native_rows": 1351, "consumed_rows": 1350, "control_rows": 1350, "summary_native_states": 13501,
            "summary_torque_intervals": 13500, "all_exported_columns_independently_recomputed": True,
            "maximum_CSV_abs_difference": errors, "route_native_samples": int(route_t.sum()), "route_consumed_samples": int(route_tt.sum()),
            "units": "m, seconds, degrees, rad/s, Nm and explicit counts; no scaling of exported CSV", "QP_vectors_available": vector_available}


def validate(output, source, receipt, workers):
    require(shutil.which("ffmpeg") and shutil.which("ffprobe"), "ffmpeg and ffprobe are required")
    require(output != source and not output.is_relative_to(source), "Media output must be outside scientific source")
    manifest = read(output / "media_manifest.json")
    before = inventory(output)
    require(manifest["payload"] == before, "Media manifest payload hash/bytes or exact membership mismatch")
    require(set(manifest["excluded_self_referential_files"]) == CONTROLS, "Unexpected media control exclusions")
    sources = Sources(source)
    sources.record(manifest["source_manifest"])
    require(not manifest["source_research_artifacts_modified"], "Media claims scientific source modifications")
    for field in ("physics_steps_executed", "QP_solves", "training_updates", "model_samples"):
        require(manifest[field] == 0, f"Nonzero media scientific budget: {field}")
    plot_relative = "figures/plots_final.json" if (output / "figures/plots_final.json").exists() else "figures/plots.json"
    videos, plots, gallery = (read(output / p) for p in ("videos.json", plot_relative, "gallery_data.json"))
    require(videos["record_count"] == 28 and videos["video_count"] == 196 and videos["slot_order"] == SLOTS
            and not videos["smoke_truncated"] and not videos["historical_outputs_overwritten"]
            and not videos["independent_replay_states_used_for_render"] and videos["actual_saved_state_source_verified"], "Video collection scope changed")
    sources.record(videos["source_manifest"])
    sources.record(videos["source_verification"])
    sources.records(plots["source_records"])
    require(plots["source_manifest_sha256"] == SOURCE_MANIFEST_SHA and plots["source_producer_commit"] == ACTUAL_PRODUCER
            and plots["source_report_sha256"] == identity(sources.bind(source / "report.json"))["sha256"]
            and plots["source_files_unchanged"] and not plots["existing_scientific_outputs_modified"], "Plot scientific source/scope changed")
    for key in ("new_physics_steps", "new_geometry_queries", "new_solver_calls", "new_models_or_sampling", "new_training_runs"):
        require(plots[key] == 0, f"Nonzero plot scientific budget: {key}")
    require([r["slot_id"] for r in videos["records"]] == [r["slot_id"] for r in plots["records"]] == SLOTS, "Missing/reordered/duplicate 28-slot coverage")
    require(manifest["record_count"] == 28 and manifest["video_count"] == 196, "Manifest coverage is not 28 x 7")
    generators = {"gallery": generator(manifest["generator"], "gallery"),
                  "renderer": generator(videos["generator"], "renderer")}
    plot_script = relative_file(output, plots["plot_producer_script"])
    require(identity(plot_script)["sha256"] == plots["plot_producer_source_sha256"], "Frozen plot producer SHA differs")
    generators["plots"] = generator({"path": str(plot_script), **identity(plot_script)}, "plots")
    current_plot = ROOT / "v6_4/plot_execution_aware_media.py"
    require(identity(current_plot) == identity(plot_script), "Current plot producer differs from its final frozen copy")
    generators["current_plot_producer"] = {"path": str(current_plot), **identity(current_plot)}
    for index, record in enumerate(videos["generator_records"]):
        path = Path(record["path"])
        path = path.resolve() if path.is_absolute() else (ROOT / path).resolve()
        match(path, record)
        generators[f"renderer_dependency_{index}"] = {"path": str(path), **identity(path)}
    require(plots["outputs"] == {p: row for p, row in before.items() if p.startswith("figures/") and p not in {plot_relative, "figures/visual_qa.json"}},
            "Plot outputs do not exhaust the retained figure directory")
    if "figures/visual_qa.json" in before:
        qa = read(output / "figures/visual_qa.json")
        require(qa["selected_manifest"] == plot_relative and qa["selected_manifest_sha256"] == identity(output / plot_relative)["sha256"], "Sampled visual QA is stale")
    if plot_relative.endswith("plots_final.json"):
        base_path = output / "figures/plots.json"
        base_plot = read(base_path)
        require(identity(base_path)["sha256"] == plots["base_plot_manifest_sha256"], "Final plot lineage does not bind original manifest")
        base_script = relative_file(output, base_plot["plot_producer_script"])
        require(identity(base_script)["sha256"] == base_plot["plot_producer_source_sha256"], "Original plot producer byte snapshot differs")
        require(all(before[p] == value for p, value in base_plot["outputs"].items()), "Original retained plot outputs changed")
        require(all(final["summary"] == prior["summary"] and final["csv"] == prior["csv"] and final["source_records"] == prior["source_records"]
                    for final, prior in zip(plots["records"], base_plot["records"])), "Layout correction changed scientific numbers/CSV/source provenance")
    jobs, results = [], []
    used_media = set()
    selected_figures, selected_pdfs, selected_csvs = set(), set(), set()
    for frozen, metadata, plotted in zip(sources.plan["slots"], videos["records"], plots["records"]):
        sid = frozen["slot_id"]
        original = sources.json(source / "slots" / sid / "slot_result.json")
        for field in ("slot_id", "task_id", "mode", "source_role"):
            for row in (metadata, plotted):
                require(row[field] == frozen[field], f"Slot identity changed: {sid}/{field}")
        source_slot = frozen["old_source_slot"] or sid
        require(metadata["source_slot"] == plotted["source_slot"] == source_slot, f"Source alias changed: {sid}")
        require(metadata["status"] == original["status"] == "TASK_COMPLETED" and metadata["full_task_success"]
                and metadata["quality_label_eligible"] == original["quality"]["quality_label_eligible"], f"Outcome changed: {sid}")
        require(read(output / "replays" / sid / "replay_metadata.json") == metadata, f"Collection/per-slot metadata changed: {sid}")
        require(metadata["renderer_producer_bindings"] == videos["generator_records"], "Per-slot renderer producer identities differ from collection")
        require(set(metadata["videos"]) == set(metadata["video_records"]) == VIEWS, f"Incomplete seven views: {sid}")
        require(metadata["reference_family"] == frozen["reference_version"] and metadata["amplitude_m"] == frozen["amplitude_m"]
                and metadata["deployment"] == "NOT_MET" and not metadata["visualization_is_new_acceptance"]
                and not metadata["new_safety_evaluation"], f"Reference/deployment meaning changed: {sid}")
        require(metadata["reference_execution_binding"]["binding_scope"] == "post-step generated references only; not held/consumed QP task inputs", "Generated/consumed reference phases conflated")
        require(all(metadata.get(k) == 0 for k in ("physics_steps_executed", "optimizer_updates", "new_samples")), f"Nonzero render scientific budget: {sid}")
        sources.records(plotted["source_records"])
        trace, fresh, times, parity = actual_parity(metadata, sources, original)
        schedule = frame_schedule(metadata, times)
        csv_result = plot_csv_contract(output, plotted, trace, fresh, times, sources, frozen, original)
        for key in ("preview", "focus_preview", "last_saved_state"):
            match(relative_file(output, metadata[key]), metadata[key+"_record"])
        require(set(plotted["figures"]) == set(plotted["pdf"]) == {"trajectory", "tracking", "safety_base", "control_diagnostics"}
                and set(plotted["csv"]) == {"native", "consumed", "control"}, "Missing complete plot/PDF/CSV roles")
        selected_figures.update(plotted["figures"].values())
        selected_pdfs.update(plotted["pdf"].values())
        selected_csvs.update(plotted["csv"].values())
        for relative in metadata["videos"].values():
            require(relative not in used_media, f"Duplicate/shared video across slots: {relative}")
            used_media.add(relative)
        for view in sorted(VIEWS):
            jobs.append((output, metadata, view))
        results.append({"slot_id": sid, "actual_fresh_parity": parity, "schedule": schedule, "csv": csv_result})
    require(len(used_media) == 196 and used_media == {p for p in before if p.endswith(".mp4")}, "MP4 exact membership differs from 196 records")
    require(len(plots["task_figures"]) == 4 and len({r["task_id"] for r in plots["task_figures"]}) == 4, "Task figure coverage is not four tasks")
    for task, declaration in zip(plots["task_figures"], sources.plan["tasks"]):
        require(task["task_id"] == declaration["task_id"] and task["source_slots"] == [r["slot_id"] for r in sources.plan["slots"] if r["task_id"] == declaration["task_id"]], "Seven-mode task figure cohort changed")
        selected_figures.update(task["figures"].values())
        selected_pdfs.update(task["pdf"].values())
    selected_figures.update(plots["all_tasks_figures"].values())
    selected_pdfs.update(plots["all_tasks_pdf"].values())
    require(len(selected_figures) == len(selected_pdfs) == 117 and len(selected_csvs) == 84, "Selected figure/PDF/CSV coverage or uniqueness mismatch")
    for relative in selected_figures | selected_pdfs | selected_csvs:
        require(relative in before and relative_file(output, relative).is_file(), f"Unbound plot link: {relative}")
    retained = set(plots.get("unselected_retained_outputs", []))
    require(retained.isdisjoint(selected_figures | selected_pdfs | selected_csvs), "Unselected legacy artifact is still selected")
    require({p for p in before if p.startswith("figures/") and p.endswith((".png", ".pdf", ".csv"))}
            == selected_figures | selected_pdfs | selected_csvs | retained, "Figure directory has undeclared/unselected media")
    require(gallery["records"] == [{**{k: p[k] for k in ("slot_id", "task_id", "mode", "source_slot", "source_role", "figures", "csv", "summary")},
            **{k: v[k] for k in ("videos", "preview", "focus_preview", "reference_version")}}
            for v, p in zip(videos["records"], plots["records"])], "Gallery projection differs from source metadata")
    require(gallery["task_figures"] == plots["task_figures"] and gallery["all_tasks_figures"] == plots.get("all_tasks_figures", {}), "Gallery task figures changed")
    require(gallery["source_plots"]["sha256"] == identity(output / plot_relative)["sha256"], "Gallery does not bind selected final plots manifest")
    require(all(gallery["source_videos"][key] == identity(output / "videos.json")[key] for key in ("sha256", "bytes")),
            "Gallery does not bind current video collection bytes")
    images = {p: check_image(relative_file(output, p)) for p in before if p.endswith(".png")}
    pdfs = {p: check_pdf(relative_file(output, p)) for p in before if p.endswith(".pdf")}
    links = html_links(output, receipt)
    poster_checks = route_posters(output, videos, gallery, sources, workers, before)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        media = list(pool.map(check_video, jobs))
    require(inventory(output) == before, "Media payload changed during validation")
    sources.unchanged()
    for row in generators.values():
        match(Path(row["path"]), row)
    return {"schema": "v64_b31_media_validation_v1", "status": "PASS", "checked_utc": datetime.now(timezone.utc).isoformat(),
            "media_manifest_sha256": identity(output / "media_manifest.json")["sha256"],
            "validator_source": {"path": str(Path(__file__).resolve()), **identity(__file__)},
            "source_manifest_sha256": SOURCE_MANIFEST_SHA, "actual_algorithm_producer_commit": ACTUAL_PRODUCER,
            "payload_files": len(before), "fixed_slots_checked": 28, "mp4_files_probed_and_decoded": len(media),
            "selected_png_figures": len(selected_figures), "selected_pdf_figures": len(selected_pdfs), "selected_CSV_files": len(selected_csvs),
            "unselected_retained_outputs_checked": sorted(retained),
            "route_poster_count": poster_checks["poster_count"], "decoded_route_poster_frames_checked": poster_checks["decoded_source_frames"],
            "route_poster_checks": poster_checks,
            "decoded_frames_checked": len(media) * 3, "png_checks": images, "pdf_checks": pdfs,
            "local_html_link_checks": links, "slot_checks": results, "video_checks": media,
            "source_files_read_and_hashed": list(sources.checked.values()), "generator_source_checks": generators,
            "old_source_full_external_seal_rechecked": False,
            "source_scope": "Only used original evidence reauthenticated against immutable seal; mutable current docs excluded. Portable old release verification is separate.",
            "browser_render": "NOT_CHECKED_STATIC_LINKS_ONLY", "new_physics_steps": 0, "new_geometry_queries": 0,
            "new_QP_solves": 0, "new_model_calls": 0, "deployment": "NOT_MET", "visualization_is_new_acceptance": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--workers", type=int, choices=range(1, 9), default=2)
    args = parser.parse_args()
    output, source, receipt = args.output.resolve(), args.source.resolve(), args.receipt.resolve()
    require(not receipt.exists(), f"Receipt is exclusive-create; already exists: {receipt}")
    require(not receipt.is_relative_to(source), "Receipt must not be inside scientific source")
    require(not receipt.is_relative_to(output) or receipt == output / "validation.json", "Only validation.json control may be written inside media output")
    try:
        result = validate(output, source, receipt, args.workers)
    except Exception as error:
        result = {"schema": "v64_b31_media_validation_v1", "status": "FAIL", "checked_utc": datetime.now(timezone.utc).isoformat(),
                  "error_type": type(error).__name__, "error": str(error), "new_physics_steps": 0, "new_geometry_queries": 0,
                  "new_QP_solves": 0, "new_model_calls": 0}
    receipt.parent.mkdir(parents=True, exist_ok=True)
    with receipt.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: result[k] for k in ("status", "media_manifest_sha256", "fixed_slots_checked", "mp4_files_probed_and_decoded", "error") if k in result}, ensure_ascii=False))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
