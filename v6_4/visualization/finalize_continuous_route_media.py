"""Build portable Markdown navigation and independently verify C.1 derived media."""
from __future__ import annotations

import argparse
import csv
from fractions import Fraction
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import shutil
import subprocess
from urllib.parse import unquote

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def read(path):
    return json.loads(Path(path).read_text(encoding="utf8"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf8",newline="\n")


class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.paths = []

    def handle_starttag(self,tag,attrs):
        self.paths += [v for k,v in attrs if k in ("href","src","poster") and v and not v.startswith(("#","http:","https:","data:"))]


def finalize(source, output):
    collection = read(output/"collection.json")
    manifest = read(output/"media_manifest.json")
    assert sha(source/"manifest.json") == collection["experiment_manifest_sha256"] == manifest["source_manifest_sha256"]
    for rel, record in manifest["files"].items():
        assert sha(output/rel) == record["sha256"] and (output/rel).stat().st_size == record["bytes"], rel
    raw_manifest = read(source/"manifest.json")
    for rel, digest in raw_manifest.items():
        assert sha(source/rel) == digest, rel
    assert len(collection["slots"]) == 16 and len(collection["media"]) == 12 and len(collection["plots"]) == 12
    assert len(list(output.rglob("*.mp4"))) == 84
    assert len(list((output/"figures").rglob("*.png"))) == 48
    assert len(list((output/"figures").rglob("*.csv"))) == 36
    actual = {r["slot_id"]:r for r in collection["media"]}
    for m in actual.values():
        assert m["render_cost"]["physics_steps"] == m["render_cost"]["geometry_queries"] == m["render_cost"]["QP_solves"] == 0
        assert not m["render_cost"]["forbidden_call_attempts"]
        assert m["schedule"]["selected_saved_indices"][-1] == m["schedule"]["source_state_count"]-1
        for video in m["video_records"].values():
            probe = video["probe"]
            assert float(Fraction(probe["avg_frame_rate"])) == m["fps"] == 15
            assert int(probe["nb_frames"]) == m["frame_count"]
            assert (output/video["relative_path"]).stat().st_size < 100_000_000
    posters = []
    for m in collection["media"]:
        prefs = read(source/"frozen_tasks"/m["task_id"]/"preferences.json")[0]
        low, high = prefs["W_key"][0]
        desired = (low+high)/2
        saved = np.asarray(m["schedule"]["selected_saved_times_s"])
        frame = min(int(np.searchsorted(saved,desired)),len(saved)-1)
        poster = output/"replays"/m["slot_id"]/"route_focus_preview.png"
        if not poster.exists():
            subprocess.run([shutil.which("ffmpeg"),"-y","-loglevel","error","-i",str(output/m["videos"]["continuum_focus"]),
                            "-vf",f"select=eq(n\\,{frame})","-frames:v","1",str(poster)],check=True)
        record = {"path":poster.relative_to(output).as_posix(),"sha256":sha(poster),"frame_index":frame,
                  "video_sha256":m["video_records"]["continuum_focus"]["sha256"],
                  "saved_state_index":m["schedule"]["selected_saved_indices"][frame],"saved_time_s":float(saved[frame]),
                  "frozen_W_key":prefs["W_key"],"desired_time_s":desired}
        posters.append(record)
        # Feature the frozen key interval, while retaining the standard midpoint preview.
        old, new = m["focus_preview"], record["path"]
        for page in output.rglob("*.html"):
            content = page.read_text(encoding="utf8")
            if old in content:
                page.write_text(content.replace(old,new),encoding="utf8",newline="\n")
        m["route_focus_preview"] = record
    rows = []
    aliases = no_plan = prefix = complete = 0
    for row in collection["slots"]:
        slot = read(source/"actual"/row["task_id"]/row["method"]/"slot.json")
        assert row["status"] == slot["status"] and row["alias_of_method"] == slot.get("alias_of_method")
        if slot["status"] == "NO_PLAN":
            no_plan += 1
            assert row["evidence_slot"] is None and row["media_available"] is False
            content = (output/row["page"]).read_text(encoding="utf8")
            assert "<video" not in content and "<img" not in content
            video_cell = "无计划，无视频"
        else:
            evidence = row["task_id"]+"_"+slot["evidence_method"]
            assert row["evidence_slot"] == evidence
            m = actual[evidence]
            assert m["task_sha256"] == slot["task_sha256"] and m["reference_plan_sha256"] == slot["plan_sha256"]
            if slot.get("alias_of_method"): aliases += 1
            elif slot["full_task_success"]: complete += 1
            else: prefix += 1
            video_cell = f"[五视角]({m['videos']['five_view_grid']}) · [连续体侧视]({m['videos']['continuum_focus']})"
        label = f"复用{row['alias_of_method']}" if row["alias_of_method"] else row["status"]
        rows.append(f"|{row['task_id']}|{row['method']}|{label}|[视频、图与CSV]({row['page']})|{video_cell}|")
    assert (complete,prefix,aliases,no_plan) == (10,2,3,1)
    verified_ticks = {}
    # Recalculate errors independently from original arrays, not plotted summaries.
    for p in collection["plots"]:
        task, method = p["slot_id"].rsplit("_",1)
        result = read(source/"actual"/task/method/"attempt/attempt_result.json")
        n = result["actual_steps"]//10
        with np.load(result["trace_path"],allow_pickle=False) as tr, np.load(Path(result["evaluation_path"]).parent/"fresh_replay.npz",allow_pickle=False) as f:
            tt = tr["task_time"][:n]
            idx = np.rint(tt/.002).astype(int)
            csv_path = output/"figures"/p["slot_id"]/"consumed_tracking.csv"
            with csv_path.open(encoding="utf8",newline="") as stream:
                records = list(csv.DictReader(stream))
            assert len(records) == n
            assert np.array_equal(np.asarray([float(r["time_s"]) for r in records]),tt)
            assert np.array_equal(np.asarray([int(r["actual_saved_state_index"]) for r in records]),idx)
            for arm in ("rigid","continuum"):
                expected = np.linalg.norm(f[arm+"_position"][idx]-tr["task_input_reference_"+arm+"_target_position"][:n],axis=1)
                observed = np.asarray([float(r[arm+"_position_error_m"]) for r in records])
                assert np.max(np.abs(expected-observed)) < 1e-14
                assert abs(np.sqrt(np.mean(expected**2))-p["tracking"][arm]["position_rms_m"]) < 1e-14
                a, b = f[arm+"_rotation"][idx], tr["task_input_reference_"+arm+"_target_rotation"][:n]
                rotation = np.rad2deg(np.arccos(np.clip((np.trace(np.matmul(a.transpose(0,2,1),b),axis1=1,axis2=2)-1)/2,-1,1)))
                observed = np.asarray([float(r[arm+"_orientation_error_deg"]) for r in records])
                assert np.max(np.abs(rotation-observed)) < 2e-6
            verified_ticks[p["slot_id"]] = n
    for page in output.rglob("*.html"):
        links = Links(); links.feed(page.read_text(encoding="utf8"))
        for rel in links.paths:
            target = (page.parent/unquote(rel.split("#")[0])).resolve()
            assert target.is_relative_to(ROOT) and target.is_file(), (page,rel)
    text = "# V6.4-C.1 当前媒体目录\n\n84段视频、48张诊断图、36个CSV，覆盖全部16逻辑槽。12条独立actual包括10条完整轨迹和2条失败前缀；3个别名共享资产，1个NO_PLAN无轨迹。\n\n[HTML总目录](index.html) · [C.1结果总览](../continuous_route_optimizer_20261007_01/index.html) · [来源与验证说明](../../../docs/V6_4_C1_MEDIA_SUPPLEMENT.md) · [完整帧时序与机器验证](collection.json)\n\n|任务|方法|状态/别名|完整诊断|视频|\n|---|---|---|---|---|\n"+"\n".join(rows)+"\n\n"
    for m in collection["media"]:
        p = next(p for p in collection["plots"] if p["slot_id"] == m["slot_id"])
        text += f"## {m['slot_id']}\n\n保存终点 **{m['source_end_s']:.3f}s**，{'完整27秒' if m['full_task_success'] else '失败前缀，后续未执行'}。\n\n[五视角合成]({m['videos']['five_view_grid']}) · [连续体侧视]({m['videos']['continuum_focus']}) · "+" · ".join(f"[{key}]({path})" for key,path in m["videos"].items() if key not in ("five_view_grid","continuum_focus"))+"\n\n"
        text += f"![连续体侧视关键区间预览]({m['route_focus_preview']['path']})\n\n[末端轨迹]({p['figures']['trajectory']}) · [跟踪误差]({p['figures']['tracking']}) · [净空与基座]({p['figures']['safety_base']}) · [控制诊断]({p['figures']['control']}) · "+" · ".join(f"[{Path(path).name}]({path})" for path in p["csv"])+"\n\n"
    text += "视频为原始actual保存状态重绘；新增物理步/距离查询/QP求解均为0。跟踪误差使用实际消耗的QP输入，Task路线偏差单列；失败前缀不补齐。部署NOT_MET。\n"
    (output/"README.md").write_text(text,encoding="utf8",newline="\n")
    write(output/"collection.json",collection)
    receipt = {"schema":"v64_c1_media_independent_verification_v1","status":"PASS",
               "sealed_raw_files_verified":len(raw_manifest),"logical_slots":16,"unique_complete":complete,"unique_prefix":prefix,
               "alias_slots":aliases,"no_plan_slots":no_plan,"video_files":84,"figure_files":48,"csv_files":36,
               "tracking_csv_recomputed_from_raw":"PASS","consumed_tick_counts":verified_ticks,
               "orientation_tolerance_deg":2e-6,"HTML_relative_links":"PASS","browser_live_preview":"NOT_VERIFIED",
               "route_poster_bindings":posters,"verifier_sha256":sha(Path(__file__)),"new_physics_steps":0}
    write(output/"verification.json",receipt)
    manifest["files"] = {p.relative_to(output).as_posix():{"sha256":sha(p),"bytes":p.stat().st_size} for p in sorted(output.rglob("*")) if p.is_file() and p.name!="media_manifest.json"}
    write(output/"media_manifest.json",manifest)
    print(json.dumps(receipt),flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source",required=True,type=Path)
    parser.add_argument("--output",required=True,type=Path)
    args = parser.parse_args()
    finalize(args.source.resolve(),args.output.resolve())
