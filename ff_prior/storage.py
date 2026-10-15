"""Atomic immutable records, exclusive run ownership and cumulative accounting."""
from __future__ import annotations
import hashlib
import json
import os
import time
from pathlib import Path
import numpy as np


def plain(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def digest(value):
    return hashlib.sha256(json.dumps(value, default=plain, sort_keys=True,
        separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(value, f, default=plain, sort_keys=True, indent=2, allow_nan=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def atomic_npz(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    with tmp.open("wb") as f:
        np.savez_compressed(f, **arrays)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def file_digest(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda:f.read(1024*1024),b""): h.update(block)
    return h.hexdigest()


class RunLock:
    def __init__(self, directory):
        self.path = Path(directory) / "run.lock"

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Never remove a lock based only on age/PID heuristics.
        self.fd = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
        os.write(self.fd, json.dumps({"pid": os.getpid(), "created": time.time()}).encode())
        return self

    def __exit__(self, *_):
        os.close(self.fd)
        self.path.unlink()


class Ledger:
    def __init__(self, directory, identity):
        self.directory = Path(directory)
        self.path = self.directory / "ledger.json"
        self.identity = identity
        self.data = read_json(self.path) if self.path.exists() else {
            "identity": identity, "completed": {}, "inflight": None,
            "wall_s": {}, "cpu_s": {}, "calls": {}, "attempts": {}}
        if self.data["identity"] != identity:
            raise ValueError("output identity conflict")
        # A crash consumes all reserved work; it never refunds a proposal.
        active = self.data.get("inflight")
        if active:
            elapsed = max(0, time.time() - active["started"])
            stage = active["stage"]
            self.data["wall_s"][stage] = self.data["wall_s"].get(stage, 0) + elapsed
            self.data["completed"][active["id"]] = {"status": "INTERRUPTED", "stage": stage}
            self.data.setdefault("unresolved_call_accounting",[]).append(active["id"])
            self.data["inflight"] = None
            self.save()

    def save(self):
        atomic_json(self.path, self.data)

    def begin(self, proposal_id, stage, wall_budget, attempt_budget):
        if proposal_id in self.data["completed"]:
            return False
        if self.data["wall_s"].get(stage, 0) >= wall_budget or self.data["attempts"].get(stage, 0) >= attempt_budget:
            raise RuntimeError("BUDGET_EXHAUSTED")
        if self.data["inflight"]:
            raise RuntimeError("one owner/one in-flight work item required")
        self.data["attempts"][stage] = self.data["attempts"].get(stage, 0) + 1
        self.data["inflight"] = {"id": proposal_id, "stage": stage, "started": time.time()}
        self.wall, self.cpu = time.perf_counter(), time.process_time()
        self.save()
        return True

    def finish(self, result, calls=None):
        active = self.data["inflight"]
        stage = active["stage"]
        self.data["wall_s"][stage] = self.data["wall_s"].get(stage, 0) + time.perf_counter() - self.wall
        self.data["cpu_s"][stage] = self.data["cpu_s"].get(stage, 0) + time.process_time() - self.cpu
        self.data["completed"][active["id"]] = dict(result, stage=stage)
        for k, v in (calls or {}).items():
            key = f"{stage}.{k}"
            self.data["calls"][key] = self.data["calls"].get(key, 0) + int(v)
        self.data["inflight"] = None
        self.save()

    def index(self):
        # JSON records are authoritative. JSONL is a rebuildable sorted view.
        rows = []
        for file in sorted((self.directory / "records").glob("*.json")):
            rows.append(read_json(file))
        path = self.directory / "index.jsonl"
        tmp = path.with_suffix(".tmp")
        tmp.write_text("".join(json.dumps(r, default=plain, sort_keys=True, allow_nan=False) + "\n" for r in rows), encoding="utf-8")
        os.replace(tmp, path)
        failures = [dict(proposal_id=k, **v) for k, v in sorted(self.data["completed"].items())
                    if v.get("status") in ("REJECTED","ERROR","INTERRUPTED","EXECUTION_FAILED",
                        "NO_MODEL_VALID_FROZEN_GOAL_CANDIDATE","BUDGET_EXHAUSTED")]
        atomic_json(self.directory / "failures.json", failures)
        return rows
