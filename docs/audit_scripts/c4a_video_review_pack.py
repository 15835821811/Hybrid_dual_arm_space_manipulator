"""Post-render ffprobe checks and first/middle/last frame packs for human review.

Reads finished C4-A media only; writes to a fresh external QA directory. A PASS
is machine verification, never a claim that the generated images were viewed.
No experiment modules, physics, inference, or model training are invoked.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

from PIL import Image, ImageDraw, ImageOps

VIEWS = ("overview", "front", "side", "top", "iso", "five_view_grid", "continuum_focus")


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for data in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            h.update(data)
    return h.hexdigest()


def check(condition, message):
    if not condition:
        raise ValueError(message)


def command(args, receipt):
    started = time.perf_counter()
    result = subprocess.run(args, shell=False, capture_output=True, text=True, encoding="utf-8")
    receipt["commands"].append({"argv": args, "exit_code": result.returncode,
                                "elapsed_s": time.perf_counter() - started,
                                "stderr": result.stderr})
    check(result.returncode == 0, "media tool failed: " + result.stderr[-2000:])
    return result.stdout


def build(media, output, receipt):
    ffprobe, ffmpeg = shutil.which("ffprobe"), shutil.which("ffmpeg")
    check(bool(ffprobe and ffmpeg), "ffprobe and ffmpeg are required")
    receipt["tools"] = {"ffprobe": command([ffprobe, "-version"], receipt).splitlines()[0],
                        "ffmpeg": command([ffmpeg, "-version"], receipt).splitlines()[0]}
    payload = read(media / "dashboard_data.json")
    manifest = read(media / "visualization_manifest.json")
    request = read(media / "build_request.json")
    receipt["media_manifest_sha256"] = sha(media / "visualization_manifest.json")
    check(len(payload["slots"]) == 12, "expected twelve logical media slots")
    check(set(payload["views"]) == set(VIEWS), "missing or extra view")
    unique = [row for row in payload["slots"] if not row.get("alias_of_slot") and row["actual_steps"] > 0]
    check(manifest["unique_actual_media"] == len(unique) and manifest["video_count"] == 7 * len(unique),
          "video inventory differs from unique saved actual count")
    for slot in unique:
        sid = slot["task_id"] + "_" + slot["method"]
        meta = slot["media"]
        check(set(meta["videos"]) == set(VIEWS), "incomplete views: " + sid)
        check(meta["includes_exact_saved_endpoint"] is True
              and meta["selected_saved_indices"][-1] == slot["actual_steps"], "missing exact saved endpoint")
        frame_count = meta["frame_count"]
        indices = [0, frame_count // 2, frame_count - 1]
        unique_indices = sorted(set(indices))
        check(frame_count == len(meta["selected_saved_indices"]) == len(meta["selected_saved_times_s"]),
              "frame metadata length differs")
        check(abs(meta["encoded_duration_s"] - frame_count / meta["fps"]) < 1e-9, "encoded duration metadata differs")
        sheet = Image.new("RGB", (1440, 7 * 390 + 44), "#f1f4f8")
        drawing = ImageDraw.Draw(sheet)
        drawing.text((12, 12), sid + " | first / middle / last | VISUAL REVIEW PENDING", fill="black")
        for row_index, view in enumerate(VIEWS):
            relative = meta["videos"][view]
            video = (media / relative).resolve()
            check(video.is_relative_to(media), "video path escapes media")
            check(sha(video) == meta["video_records"][view]["sha256"] == manifest["files"][relative]["sha256"],
                  "video bytes differ: " + relative)
            probe = json.loads(command([ffprobe, "-v", "error", "-count_frames", "-select_streams", "v:0",
                "-show_entries", "stream=width,height,r_frame_rate,nb_read_frames:format=duration", "-of", "json", str(video)], receipt))
            stream = probe["streams"][0]
            dimensions = ((3 * request["width"], 2 * request["height"]) if view == "five_view_grid" else
                          (request["focus_width"], request["focus_height"]) if view == "continuum_focus" else
                          (request["width"], request["height"]))
            check((stream["width"], stream["height"]) == dimensions, "video dimensions differ: " + relative)
            check(int(stream["nb_read_frames"]) == frame_count, "decoded frame count differs: " + relative)
            check(abs(float(Fraction(stream["r_frame_rate"])) - meta["fps"]) < 1e-9, "video FPS differs")
            check(abs(float(probe["format"]["duration"]) - frame_count / meta["fps"]) < 0.002,
                  "video encoded duration differs: " + relative)
            folder = output / "frames" / sid / view
            folder.mkdir(parents=True, exist_ok=False)
            select = "+".join("eq(n\\," + str(n) + ")" for n in unique_indices)
            command([ffmpeg, "-v", "error", "-nostdin", "-i", str(video), "-vf", "select=" + select,
                     "-vsync", "0", "-frames:v", str(len(unique_indices)), str(folder / "%02d.png")], receipt)
            extracted = sorted(folder.glob("*.png"))
            check(len(extracted) == len(unique_indices), "preview extraction incomplete")
            frames = dict(zip(unique_indices, extracted))
            for column, n in enumerate(indices):
                y = 44 + row_index * 390
                with Image.open(frames[n]) as frame:
                    thumb = ImageOps.contain(frame.convert("RGB"), (470, 350))
                    sheet.paste(thumb, (column * 480 + (480 - thumb.width) // 2, y + 24))
                drawing.text((column * 480 + 8, y + 4),
                             view + " | frame " + str(n) + " | t=" + format(meta["selected_saved_times_s"][n], ".3f"), fill="black")
            receipt["videos"].append({"slot": sid, "view": view, "video": relative,
                "sha256": meta["video_records"][view]["sha256"], "probe": probe,
                "review_frame_indices": indices,
                "review_frames": [str(frames[n].relative_to(output)) for n in indices],
                "visual_review": "NOT_RUN"})
        sheet_path = output / (sid + "_seven_view_contact_sheet.jpg")
        sheet.save(sheet_path, quality=94)
        receipt["contact_sheets"].append({"slot": sid, "path": sheet_path.name, "sha256": sha(sheet_path),
                                           "visual_review": "NOT_RUN"})
        print(json.dumps({"event": "C4A_VIDEO_MACHINE_QA", "slot": sid, "views": 7,
                          "contact_sheet": str(sheet_path), "visual_review": "NOT_RUN"}), flush=True)
    receipt["unique_saved_actuals"] = len(unique)
    receipt["video_count"] = len(receipt["videos"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--media", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    media, output = args.media.resolve(), args.output.resolve()
    check(not output.exists() and not output.is_relative_to(media) and not media.is_relative_to(output),
          "use a fresh QA directory separate from media")
    output.mkdir(parents=True, exist_ok=False)
    receipt = {"schema": "c4a_video_review_pack_v1", "started_utc": datetime.now(timezone.utc).isoformat(),
               "script_sha256": sha(Path(__file__)), "argv": [sys.executable, *sys.argv], "media": str(media),
               "status": "FAIL", "commands": [], "videos": [], "contact_sheets": [],
               "visual_review": "NOT_RUN", "physics_steps": 0, "model_samples": 0, "training_updates": 0}
    start = time.perf_counter()
    try:
        build(media, output, receipt)
        receipt["status"] = "MACHINE_PASS_VISUAL_REVIEW_PENDING"
    except Exception as error:
        receipt["error"] = {"type": type(error).__name__, "message": str(error)}
    receipt.update(ended_utc=datetime.now(timezone.utc).isoformat(), elapsed_wall_s=time.perf_counter() - start)
    with (output / "receipt.json").open("x", encoding="utf-8") as handle:
        json.dump(receipt, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps({"status": receipt["status"], "videos_checked": len(receipt["videos"]),
                      "error": receipt.get("error"), "receipt": str(output / "receipt.json")}))
    return 0 if receipt["status"].startswith("MACHINE_PASS") else 1


if __name__ == "__main__":
    raise SystemExit(main())
