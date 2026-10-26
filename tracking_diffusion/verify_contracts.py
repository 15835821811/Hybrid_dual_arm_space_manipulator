"""Focused checks for the live application's physics and API boundaries.

Run from project root after start_live.ps1:
  python -m tracking_diffusion.verify_contracts
"""
import json
from pathlib import Path
import tempfile
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import numpy as np

from .live_server import RunManager
from .environment import Task, TrackingEnv
from .telemetry import scene, frame, joint_candidates
from .trajectory import JointChunk


def main():
    results = {}
    url = "http://127.0.0.1:8765"
    def request(path, body=None, origin=None):
        headers = {"X-Tracking-Client": "1", "Content-Type": "application/json"}
        if origin:
            headers["Origin"] = origin
        req = Request(url+path, data=None if body is None else json.dumps(body).encode(), headers=headers)
        try:
            with urlopen(req, timeout=10) as response:
                return response.status, json.load(response)
        except HTTPError as error:
            return error.code, json.load(error)
    before = request("/api/state")[1]["run_id"]
    for name, body in (("invalid_duration", {"duration": 4}),
                       ("invalid_speed", {"speed": -1}), ("array_payload", [])):
        code, response = request("/api/start", body)
        assert code == 400 and "error" in response, (code, response)
        assert request("/api/state")[1]["run_id"] == before
        results[name] = "rejected without replacing current run"
    assert request("/api/start", {}, "https://example.invalid")[0] == 403
    results["cross_origin_request"] = "rejected"
    assert request("/api/replay/..%2Foutside")[0] == 400
    results["replay_path_escape"] = "rejected"
    with tempfile.TemporaryDirectory(prefix="tracking_contract_") as folder:
        manager = RunManager(folder, Path(folder)/"missing.pt")
        try:
            manager.start({"mode": "diffusion"})
        except ValueError:
            pass
        else:
            raise AssertionError("Missing weights must not fall back to teacher")
        assert manager.state["status"] == "idle" and manager.worker is None
    results["missing_checkpoint"] = "rejected; no teacher fallback"

    env = TrackingEnv(Task(20261012, 3., 2))
    before_state = {name: getattr(env.data, name).copy() for name in ("qpos", "qvel", "ctrl")}
    before_time = float(env.data.time)
    s, f = scene(env), frame(env)
    q, dq = env.state()
    candidate = np.zeros((4, 2, 17))
    curves = joint_candidates(env, candidate)
    for name, value in before_state.items():
        np.testing.assert_array_equal(getattr(env.data, name), value)
    assert float(env.data.time) == before_time
    assert len(s["geometry"]["continuum"]) == 32 and len(s["geometry"]["rigid"]) == 9
    assert len(curves["joint_curves"]) == 4
    q0, dq0, _ = JointChunk.decode(candidate[0], q, dq).sample(0.)
    np.testing.assert_allclose(q0, q)
    np.testing.assert_allclose(dq0, dq, atol=1e-12)
    results["telemetry_does_not_move_robot"] = "qpos, qvel, ctrl and time unchanged"
    results["generated_boundary"] = "starts at measured position and velocity"
    # Cancelling before the first physics step must retain USER_CANCELLED,
    # rather than fail JSON serialization of unmeasured infinite minima.
    diagnostics = env.diagnostics()
    assert diagnostics["minimum_pair_margin_m"] is None
    json.dumps(diagnostics, allow_nan=False)
    results["zero_step_diagnostics"] = "unmeasured clearances are null, not infinity or zero"
    output = Path("tracking_output/ui_verification/contracts.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
