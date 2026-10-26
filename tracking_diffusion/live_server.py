"""Loopback-only live experiment UI. Run with python -m tracking_diffusion.live_server."""
from __future__ import annotations

import os
for _key in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ[_key] = "1"

import argparse
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import threading
import time
import traceback
from urllib.parse import urlparse
import uuid
import numpy as np

from .environment import ROOT, Task
from .pipeline import run_episode, write_json
from .telemetry import frame, scene, joint_candidates, rounded

WEB = Path(__file__).with_name("web")


class RunManager:
    def __init__(self, output, checkpoint, device="cpu", workflow="paper"):
        self.output, self.checkpoint, self.device = Path(output), Path(checkpoint), device
        self.workflow = workflow
        self.output.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.condition = threading.Condition(self.lock)
        self.worker = None
        self.cancelled, self.paused = False, False
        self.events_file = None
        self.state = {"status": "idle", "stage": "ready", "sequence": 0, "run_id": None,
                      "scene": None, "frame": None, "history": [], "denoising": [],
                      "logs": [], "result": None, "failure": None, "mode": "teacher"}
        self.sim_time, self.clock_start, self.paused_seconds = 0., time.perf_counter(), 0.
        self.speed = 1.

    def snapshot(self):
        # Checkpoints are atomically published after training. Switch only while
        # idle; an active run always keeps its explicitly loaded model.
        if self.workflow=="paper" and not (self.worker and self.worker.is_alive()):
            deployed=ROOT/"tracking_output/paper500_v2/deployment.json"
            candidates=[ROOT/"tracking_output/paper500_v2/train_t2/model.pt",ROOT/"tracking_output/paper500_v2/train_t1/model.pt",ROOT/"tracking_output/paper500_v2/train_mid/model.pt"]
            if deployed.is_file():
                candidates.insert(0,ROOT/json.loads(deployed.read_text(encoding="utf-8"))["checkpoint"])
            for candidate in candidates:
                if candidate.is_file():
                    self.checkpoint=candidate
                    break
        with self.lock:
            value = json.loads(json.dumps(self.state, allow_nan=False))
        value["checkpoint_available"] = self.checkpoint.is_file()
        value["workflow"] = self.workflow
        value["checkpoint_path"] = str(self.checkpoint)
        configuration={"steps":60,"candidates":5,"scoring":"legacy","continuity":0.,"cuda_graph":False}
        deployed=ROOT/"tracking_output/paper500_v2/deployment.json"
        if self.workflow=="paper" and deployed.is_file():
            candidate=json.loads(deployed.read_text(encoding="utf-8"))
            if (ROOT/candidate["checkpoint"]).resolve()==self.checkpoint.resolve():
                configuration.update(candidate)
        value["inference_config"]=configuration
        steps=configuration["steps"]
        value["checkpoint_label"] = (f"条件 Transformer · {steps} 步 DDIM" if self.workflow == "paper" else "旧试验权重") if value["checkpoint_available"] else "尚无训练权重"
        training = self.checkpoint.parent/"training_report.json"
        if training.is_file():
            report = json.loads(training.read_text(encoding="utf-8"))
            value["training"] = {k:v for k,v in report.items() if k != "history"}
            if "train_tasks" in report:
                value["checkpoint_label"] = f"条件 Transformer · {report['train_tasks']} 个训练任务 · {steps} 步 DDIM"
        stages={}
        for name in ("training","screen","refine","validation","final"):
            path=ROOT/f"tracking_output/paper500_v2/{name}_stage.json"
            if path.is_file():
                stages[name]=json.loads(path.read_text(encoding="utf-8"))
        value["experiment_stages"]=stages
        batch_file = ROOT/"tracking_output/paper500_v2/teachers/summary.json"
        if not batch_file.is_file():
            batch_file=ROOT/"tracking_output/paper100_v1/teachers/summary.json"
        if batch_file.is_file():
            try:
                batch = json.loads(batch_file.read_text(encoding="utf-8"))
                value["batch"] = {k:v for k,v in batch.items() if k != "episodes"}
                value["batch"].setdefault("total",batch.get("total_primary_tasks"))
            except json.JSONDecodeError:
                pass  # A worker may be replacing the progress file at this instant.
        value["server_time"] = time.time()
        value["runs"] = [p.parent.name for p in sorted(self.output.glob("*/result.json"),
                        key=lambda p: p.stat().st_mtime, reverse=True)[:20]]
        return value

    def log(self, text):
        with self.lock:
            self.state["logs"].append({"at": datetime.now().strftime("%H:%M:%S"), "text": text})
            self.state["logs"] = self.state["logs"][-40:]

    def publish(self, event):
        # JSON conversion also rejects NaNs before they can break the client.
        encoded = json.dumps(event, ensure_ascii=False, allow_nan=False)
        if self.events_file:
            self.events_file.write(encoded + "\n")
            self.events_file.flush()
        with self.lock:
            self.state["sequence"] += 1
            self.state["updated_at"] = time.time()

    def observe(self, kind, env, payload):
        if kind == "metrics":
            # Keep every 50 Hz error sample, even though the skeleton is sent
            # at 10 Hz. Otherwise the 0.26/0.28 s startup peak is invisible.
            with self.lock:
                self.state.setdefault("error_history",[]).append(payload)
            self.publish({"kind":"metrics","value":payload})
        elif kind == "initialized":
            value = scene(env)
            with self.lock:
                self.state["scene"] = value
                self.state["stage"] = "execution"
                self.clock_start = time.perf_counter()
                self.paused_seconds = 0.
            self.publish({"kind": "scene", "value": value})
            self.log(f"模型已就绪：17 维规划 / 67 路力矩 / {len(env.pairs)} 个碰撞检查对")
        elif kind == "state":
            value = frame(env)
            with self.lock:
                self.sim_time = value["time"]
                self.state["frame"] = value
                self.state["stage"] = "execution"
                self.state["history"].append({k: value[k] for k in (
                    "time", "actual_c", "actual_r", "error_c_mm", "error_r_mm", "margin_mm")})
            self.publish({"kind": "frame", "value": value})
        elif kind == "denoising":
            value = joint_candidates(env, payload["free"])
            value.update({"step": payload["step"], "total": payload["total"], "time": float(env.data.time)})
            value["guidance_active"] = self.workflow == "paper" and self.state.get("guidance_enabled",True) and payload["step"] > .7*payload["total"]
            with self.lock:
                if payload["step"] == 0:
                    self.state["denoising"] = []
                self.state["denoising"].append(value)
                self.state["stage"] = "denoising"
            self.publish({"kind": "denoising", "value": value})
        elif kind == "selected":
            value = {"time":float(env.data.time), "controls": rounded(payload["controls"]), "accepted": payload["accepted"],
                     "selected_index":payload.get("selected_index"),
                     "guidance": payload.get("guidance",[])}
            if payload.get("candidate_stages"):
                value["candidate_stages"]={name:joint_candidates(env,free) for name,free in payload["candidate_stages"].items()}
            with self.lock:
                self.state["selection"] = value
                self.state["stage"] = "selection"
            self.publish({"kind": "selection", "value": value})
        elif kind == "finished":
            result = payload["result"]
            with self.lock:
                self.state["result"] = result
                self.state["failure"] = result["failure"]
                self.state["status"] = ("cancelled" if result["failure"] == "USER_CANCELLED" else
                                        "completed" if result["completed"] else "failed")
                self.state["stage"] = "verification"
            self.publish({"kind": "result", "value": result})
            label = "合格教师样本" if result["accepted_for_teacher_dataset"] else "未通过教师样本筛选"
            self.log(label if result["mode"] == "teacher" else "扩散执行结束，请查看完整误差与约束结果")
            if result["failure"]:
                self.log(result["failure"])

    def control(self):
        with self.condition:
            if self.paused:
                begin = time.perf_counter()
                while self.paused and not self.cancelled:
                    self.condition.wait(.1)
                self.paused_seconds += time.perf_counter()-begin
            if self.cancelled:
                return False
            if self.speed:
                delay = self.clock_start + self.paused_seconds + self.sim_time/self.speed - time.perf_counter()
                if delay > 0:
                    self.condition.wait(min(delay, .1))
            return not self.cancelled

    def start(self, options):
        if not isinstance(options, dict):
            raise ValueError("请求必须是 JSON 对象")
        mode = options.get("mode", "teacher")
        if mode not in ("teacher", "diffusion", "feedback-only"):
            raise ValueError("未知生成模式")
        seed = int(options.get("seed", 20261012))
        duration = float(options.get("duration", 27.))
        speed = float(options.get("speed", 1.))
        if speed not in (0., 1.):
            raise ValueError("播放速度必须是 0（最快）或 1（按仿真时间）")
        if not 0 <= seed <= 2**31-1 or duration not in (3., 27.):
            raise ValueError("随机种子或时长不合法")
        if mode == "diffusion" and not self.checkpoint.is_file():
            raise ValueError("尚无训练权重，请先完成数据生成和训练。教师轨迹不等于扩散结果。")
        with self.lock:
            if self.worker and self.worker.is_alive():
                raise ValueError("已有计算正在运行，请先终止或等待完成")
            run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]
            self.state = {"status": "running", "stage": "initializing", "sequence": 0,
                          "run_id": run_id, "scene": None, "frame": None, "history": [],
                          "denoising": [], "logs": [], "result": None, "failure": None,
                          "mode": mode, "speed": speed}
            self.cancelled, self.paused = False, False
            self.speed = self.state["speed"]
            nominal = bool(options.get("nominal", seed == 20261012))
            sampling_seed = int(options.get("sampling_seed",73))
            guidance = options.get("guidance",True) is not False
            self.state["guidance_enabled"] = guidance
            self.state["recording_mode"] = "full" if options.get("recording")=="full" else "light"
            domain = "local10mm-v1" if self.workflow == "paper" else "legacy-wide-v1"
            self.worker = threading.Thread(target=self._run, args=(Task(seed, duration, 2, domain, nominal), run_id, mode, sampling_seed, guidance), daemon=True)
            self.worker.start()
        return run_id

    def _run(self, task, run_id, mode, sampling_seed=73, guidance=True):
        folder = self.output/run_id
        folder.mkdir()
        try:
            self.events_file = (folder/"events.jsonl").open("w", encoding="utf-8")
            self.publish({"kind": "run", "value": {"mode": mode, "seed": task.seed, "duration": task.duration}})
            self.log("正在构建真实 MuJoCo 模型和期望曲线")
            policy = None
            if mode == "diffusion":
                if self.workflow == "paper":
                    from .paper_model import PaperPolicy
                    deployed=ROOT/"tracking_output/paper500_v2/deployment.json"
                    configuration={}
                    if deployed.is_file():
                        candidate=json.loads(deployed.read_text(encoding="utf-8"))
                        if (ROOT/candidate["checkpoint"]).resolve()==self.checkpoint.resolve():
                            configuration=candidate
                    policy=PaperPolicy(self.checkpoint,self.device,sampling_seed,configuration.get("candidates",5),
                        configuration.get("steps",60),guidance,configuration.get("projection",True),
                        configuration.get("scoring","legacy"),configuration.get("continuity",0.),True,
                        configuration.get("cuda_graph",False))
                    policy.diagnostic=True
                    policy.telemetry_stride=1 if self.state["recording_mode"]=="full" else 5
                    self.log(f"{policy.steps} 步条件去噪；末段注入约束；记录模式 {self.state['recording_mode']}；候选评分 {policy.scoring}")
                else:
                    from .model import DiffusionPolicy
                    policy = DiffusionPolicy(self.checkpoint, self.device, sampling_seed)
                split = "TRAIN" if task.seed in policy.train_seeds else "VAL" if task.seed in policy.val_seeds else "OTHER / 非本权重训练集"
                manifest_path = policy.metadata.get("task_manifest")
                if manifest_path and Path(manifest_path).is_file():
                    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
                    if any(e["task"]["seed"]==task.seed and e.get("split")=="test" for e in manifest["tasks"]):
                        split = "TEST / 预登记测试任务"
                self.state["task_split"] = split
                self.log(f"任务来源：{split}；试验权重演示不代表泛化性能")
            run_episode(task, folder, policy=policy, mode=mode, observer=self.observe, control=self.control)
        except Exception as exc:
            with self.lock:
                self.state["status"], self.state["failure"] = "failed", str(exc)
                self.state["stage"] = "verification"
            failure = {"completed": False, "failure": str(exc), "traceback": traceback.format_exc()}
            write_json(folder/"server_failure.json", failure)
            self.publish({"kind": "error", "value": {"failure": str(exc)}})
            self.log(f"计算失败：{exc}")
        finally:
            if self.events_file:
                self.events_file.close()
                self.events_file = None

    def action(self, action):
        with self.condition:
            if not self.worker or not self.worker.is_alive():
                raise ValueError("当前没有正在运行的任务")
            if action == "pause":
                self.paused = True
                self.state["status"] = "paused"
            elif action == "resume":
                self.paused = False
                self.state["status"] = "running"
            elif action == "cancel":
                self.cancelled = True
                self.state["status"] = "stopping"
            else:
                raise ValueError("未知控制操作")
            self.condition.notify_all()

    def replay(self, run_id):
        if not run_id or any(c not in "0123456789_abcdef" for c in run_id):
            raise ValueError("无效的运行编号")
        path = self.output/run_id/"events.jsonl"
        if not path.is_file():
            raise ValueError("找不到该运行的记录")
        # Bound replay transfer: earlier chunks keep their true first/last
        # samples; the final chunk keeps every denoising step. Raw log is intact.
        events, pending = [], []
        with path.open(encoding="utf-8") as source:
            for line in source:
                event = json.loads(line)
                if event["kind"] == "denoising":
                    if event["value"]["step"] == 0 and pending:
                        events.extend([pending[0], pending[-1]])
                        pending = []
                    pending.append(event)
                else:
                    events.append(event)
        events.extend(pending)
        errors = [event["value"] for event in events if event["kind"] == "metrics"]
        if not errors and (path.parent/"trace.npz").is_file():
            with np.load(path.parent/"trace.npz") as trace:
                errors = [{"time":float(t),"error_c_mm":float(1000*np.linalg.norm(e[:3])),
                           "error_r_mm":float(1000*np.linalg.norm(e[6:9]))}
                          for t,e in zip(trace["time"],trace["error"])]
        return {"run_id": run_id, "events": events,
                "error_history":errors,
                "denoising_resolution": "earlier chunks: first/last; final chunk: all steps"}


def make_handler(manager):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            pass

        def send_json(self, code, value):
            body = json.dumps(value, ensure_ascii=False, allow_nan=False).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            path = urlparse(self.path).path
            if path == "/api/state":
                return self.send_json(200, manager.snapshot())
            if path.startswith("/api/replay/"):
                try:
                    return self.send_json(200, manager.replay(path.rsplit("/", 1)[1]))
                except ValueError as exc:
                    return self.send_json(400, {"error": str(exc)})
            files = {"/": "index.html", "/app.js": "app.js", "/style.css": "style.css"}
            if path not in files:
                return self.send_error(404)
            file = WEB/files[path]
            body = file.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", {".html": "text/html", ".js": "text/javascript", ".css": "text/css"}[file.suffix]+"; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self'; style-src 'self'; connect-src 'self'; img-src 'self' data:")
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            origin = self.headers.get("Origin")
            if (origin and origin != "http://"+self.headers.get("Host", "")) or self.headers.get("X-Tracking-Client") != "1":
                return self.send_json(403, {"error": "只接受本地界面发起的操作"})
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 < size < 8192:
                    raise ValueError("请求大小不合法")
                options = json.loads(self.rfile.read(size))
                if not isinstance(options, dict):
                    raise ValueError("请求必须是 JSON 对象")
                if self.path == "/api/start":
                    return self.send_json(202, {"run_id": manager.start(options)})
                if self.path == "/api/control":
                    manager.action(options.get("action"))
                    return self.send_json(200, {"ok": True})
                return self.send_json(404, {"error": "Unknown endpoint"})
            except (ValueError, TypeError, KeyError) as exc:
                return self.send_json(400, {"error": str(exc)})
    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--output", type=Path, default=ROOT/"tracking_output/live")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--workflow", choices=("paper","legacy"), default="paper")
    args = parser.parse_args()
    os.chdir(ROOT)
    if args.checkpoint is None:
        latest = ROOT/"tracking_output/paper100_v1/train_a01/model.pt"
        args.checkpoint = (ROOT/"tracking_output/ui_pilot/model.pt" if args.workflow == "legacy" else
                           latest if latest.is_file() else ROOT/"tracking_output/local10mm_v1/paper_train_a01/model.pt")
    manager = RunManager(args.output, args.checkpoint, args.device, args.workflow)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(manager))
    print(f"Tracking UI: http://127.0.0.1:{args.port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        with manager.condition:
            manager.cancelled = True
            manager.condition.notify_all()
        server.server_close()


if __name__ == "__main__":
    main()
