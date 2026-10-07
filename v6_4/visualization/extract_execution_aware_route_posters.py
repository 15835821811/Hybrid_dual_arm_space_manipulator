"""Extract exact route-midpoint presentation frames from completed B.3.1 MP4s.

Append-only PNGs and receipts. FFmpeg decode only: no forward, simulation,
geometry, solver, reference evaluation, model, interpolation, or video rewrite.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import struct
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]
SLOTS = tuple(f"EA_{index:02d}" for index in range(28))
VIEWS = ("five_view_grid", "continuum_focus")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def identity(path, relative_to):
    path = Path(path).resolve()
    require(path.is_relative_to(relative_to), "source path escapes its declared root")
    return {"path": path.relative_to(relative_to).as_posix(), "sha256": sha(path), "bytes": path.stat().st_size}


def local_file(root, relative):
    path = (root/relative).resolve()
    require(path.is_relative_to(root) and path.is_file(), "missing or escaped local media source: "+str(relative))
    return path


def write_json(path, value):
    with path.open("x", encoding="utf8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def decode_png(ffmpeg, path, index):
    command = [ffmpeg, "-v", "error", "-threads", "1", "-i", str(path),
        "-vf", f"select=eq(n\\,{index})", "-vsync", "0", "-frames:v", "1",
        "-pix_fmt", "rgb24", "-c:v", "png", "-f", "image2pipe", "pipe:1"]
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False, timeout=60)
    require(result.returncode == 0, "FFmpeg decode failed: "+result.stderr.decode("utf8", errors="replace")[-2000:])
    data = result.stdout
    require(len(data) > 24 and data[:8] == b"\x89PNG\r\n\x1a\n" and data[12:16] == b"IHDR", "FFmpeg did not return a PNG frame")
    width, height = struct.unpack(">II", data[16:24])
    return data, width, height


def extract_slot(media, folder, declared, task, plan_sha, producer_sha, ffmpeg):
    sid = declared["slot_id"]
    metadata_path = media/"replays"/sid/"replay_metadata.json"
    try:
        metadata_bytes = metadata_path.read_bytes()
        metadata = json.loads(metadata_bytes.decode("utf8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return None
    metadata_digest = hashlib.sha256(metadata_bytes).hexdigest()
    for key in ("slot_id", "task_id", "mode", "source_role"):
        require(metadata[key] == declared[key], "fixed slot metadata identity differs: "+sid+"/"+key)
    require(metadata["sourcepaths"]["frozen_study_plan"]["sha256"] == plan_sha, "renderer metadata does not bind the frozen study plan")
    require(metadata["status"] == "TASK_COMPLETED" and metadata["full_task_success"] and metadata["quality_label_eligible"], "completed media metadata has invalid original outcome")
    require(metadata["fps"] == 15. and metadata["frame_count"] == 406 and not metadata["smoke_truncated"], "not the declared full 15fps video")
    schedule = metadata["schedule"]
    require(schedule["includes_exact_saved_endpoint"] and not schedule["smoke_truncated"], "incomplete saved frame schedule")
    times = schedule["presentation_times_s"]
    require(len(times) == len(schedule["selected_saved_indices"]) == len(schedule["selected_saved_times_s"]) == 406, "frame schedule lengths differ")
    interval = task["route_interval_s"]
    midpoint = sum(interval)/2.
    index = min(range(len(times)), key=lambda i: (abs(times[i]-midpoint), i))
    presentation, actual = times[index], schedule["selected_saved_times_s"][index]
    require(index == 153 and abs(presentation-10.2) < 1e-9, "unexpected B.3.1 route-midpoint frame")
    require(interval[0] <= presentation <= interval[1] and interval[0] <= actual <= interval[1], "poster frame lies outside frozen route interval")
    record = {"slot_id": sid, "task_id": metadata["task_id"], "mode": metadata["mode"],
        "source_slot": metadata["source_slot"], "source_role": metadata["source_role"],
        "source_metadata": {"path": metadata_path.relative_to(media).as_posix(), "sha256": metadata_digest, "bytes": len(metadata_bytes)},
        "route_interval_s": interval, "route_midpoint_s": midpoint, "posters": {}}
    for view in VIEWS:
        source_path = local_file(media, metadata["videos"][view])
        source = identity(source_path, media)
        original_source = metadata["video_records"][view]
        require(source["sha256"] == original_source["sha256"] and source["bytes"] == original_source["bytes"], "MP4 differs from completed metadata: "+sid+"/"+view)
        png, width, height = decode_png(ffmpeg, source_path, index)
        require(width == original_source["width"] and height == original_source["height"], "decoded poster dimensions differ from source MP4")
        target = folder/(sid+"_"+view+"_route.png")
        digest = hashlib.sha256(png).hexdigest()
        if target.exists():
            # Resume after interruption without replacing an already created PNG.
            require(sha(target) == digest and target.stat().st_size == len(png), "existing append-only poster differs from exact source frame")
        else:
            with target.open("xb") as stream:
                stream.write(png)
        require(sha(source_path) == source["sha256"], "MP4 changed during extraction")
        record["posters"][view] = {"path": target.relative_to(media).as_posix(), "source_mp4": source,
            "frame_index": index, "presentation_time_s": presentation,
            "selected_saved_index": schedule["selected_saved_indices"][index], "actual_saved_time_s": actual,
            "width": width, "height": height, "png_sha256": digest, "png_bytes": len(png)}
    require(sha(metadata_path) == metadata_digest, "completed metadata changed during extraction")
    receipt = {"schema": "v64_b31_route_poster_slot_v1", "generator_sha256": producer_sha,
        "record": record, "new_forward_calls": 0, "new_physics_steps": 0, "new_geometry_queries": 0}
    receipt_path = folder/(sid+"_poster_receipt.json")
    if receipt_path.exists():
        require(json.loads(receipt_path.read_text(encoding="utf8")) == receipt, "existing slot poster receipt differs")
    else:
        write_json(receipt_path, receipt)
    return record


def verify_record(media, record):
    metadata = record["source_metadata"]
    path = local_file(media, metadata["path"])
    require(sha(path) == metadata["sha256"] and path.stat().st_size == metadata["bytes"], "poster metadata source changed")
    for poster in record["posters"].values():
        source = poster["source_mp4"]
        video = local_file(media, source["path"])
        image = local_file(media, poster["path"])
        require(sha(video) == source["sha256"] and video.stat().st_size == source["bytes"], "poster MP4 source changed")
        require(sha(image) == poster["png_sha256"] and image.stat().st_size == poster["png_bytes"], "route poster changed")


def run(source, media, watch, poll_seconds):
    ffmpeg = shutil.which("ffmpeg")
    require(ffmpeg is not None, "FFmpeg is required")
    plan_path = source/"plan.json"
    plan_bytes = plan_path.read_bytes()
    plan_sha = hashlib.sha256(plan_bytes).hexdigest()
    plan = json.loads(plan_bytes.decode("utf8"))
    require(tuple(s["slot_id"] for s in plan["slots"]) == SLOTS, "requires fixed B.3.1 28-slot plan")
    tasks = {task["task_id"]: task for task in plan["tasks"]}
    folder = media/"route_posters"
    folder.mkdir(parents=True, exist_ok=True)
    final = folder/"poster_manifest.json"
    require(not final.exists(), "poster manifest already finalized; refuse overwrite")
    script = Path(__file__).resolve(); script_bytes = script.read_bytes()
    producer_sha = hashlib.sha256(script_bytes).hexdigest()
    copy = folder/"source"/script.name
    copy.parent.mkdir(exist_ok=True)
    if copy.exists():
        require(copy.read_bytes() == script_bytes, "append-only poster source snapshot differs")
    else:
        with copy.open("xb") as stream:
            stream.write(script_bytes)
    completed = {}
    for sid in SLOTS:
        prior = folder/(sid+"_poster_receipt.json")
        if prior.exists():
            receipt = json.loads(prior.read_text(encoding="utf8"))
            require(receipt["generator_sha256"] == producer_sha and receipt["record"]["slot_id"] == sid, "prior poster receipt producer/slot differs")
            verify_record(media, receipt["record"])
            completed[sid] = receipt["record"]
    while len(completed) < 28:
        for slot in plan["slots"]:
            sid = slot["slot_id"]
            if sid in completed:
                continue
            record = extract_slot(media, folder, slot, tasks[slot["task_id"]], plan_sha, producer_sha, ffmpeg)
            if record is not None:
                completed[sid] = record
                print(json.dumps({"status": "SLOT_POSTERS_COMPLETE", "slot_id": sid, "completed_slots": len(completed), "posters": 2,
                    "frame_index": 153, "presentation_time_s": 10.2}), flush=True)
        if len(completed) == 28:
            break
        print(json.dumps({"status": "WAITING_COMPLETED_VIDEO_METADATA", "completed_slots": len(completed),
            "remaining_slots": [sid for sid in SLOTS if sid not in completed], "next_check_s": poll_seconds if watch else None}), flush=True)
        if not watch:
            return {"status": "PENDING", "completed_slots": len(completed), "poster_count": 2*len(completed)}, 2
        time.sleep(poll_seconds)
    records = [completed[sid] for sid in SLOTS]
    for record in records:
        verify_record(media, record)
    require(sha(plan_path) == plan_sha and script.read_bytes() == script_bytes, "frozen plan or poster producer changed during watch")
    paths = [p for p in sorted(folder.rglob("*")) if p.is_file()]
    expected = {sid+"_"+view+"_route.png" for sid in SLOTS for view in VIEWS}
    require({p.name for p in folder.glob("*.png")} == expected, "route-poster PNG membership differs from 28x2")
    manifest = {"schema": "v64_b31_route_posters_v1", "status": "PASS", "created_utc": datetime.now(timezone.utc).isoformat(),
        "record_count": 28, "poster_count": 56, "records": records,
        "route_posters": {r["slot_id"]: {view: r["posters"][view]["path"] for view in VIEWS} for r in records},
        "generator": {**identity(script, ROOT), "source_copy": copy.relative_to(media).as_posix()},
        "source_plan": identity(plan_path, ROOT),
        "selection_rule": "Nearest presentation time to frozen route midpoint, earlier index for ties; exact FFmpeg select by frame index; no seek or interpolation",
        "decode_command_contract": "ffmpeg -i source.mp4 -vf select=eq(n\\,153) -vsync 0 -frames:v 1 -pix_fmt rgb24 -c:v png -f image2pipe pipe:1",
        "outputs": {p.relative_to(media).as_posix(): {"sha256": sha(p), "bytes": p.stat().st_size} for p in paths},
        "source_MP4_and_metadata_unchanged": True, "existing_media_modified": False,
        "decoded_source_frames_extracted": 56, "new_forward_calls": 0, "new_physics_steps": 0,
        "new_geometry_queries": 0, "new_QP_solves": 0, "new_model_calls": 0}
    write_json(final, manifest)
    return {"status": "PASS", "completed_slots": 28, "poster_count": 56, "manifest": str(final), "manifest_sha256": sha(final)}, 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--media", type=Path, required=True)
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--poll-seconds", type=float, default=30.)
    args = parser.parse_args()
    require(1. <= args.poll_seconds <= 60., "poll interval must be between 1 and 60 seconds")
    result, code = run(args.source.resolve(), args.media.resolve(), args.watch, args.poll_seconds)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
