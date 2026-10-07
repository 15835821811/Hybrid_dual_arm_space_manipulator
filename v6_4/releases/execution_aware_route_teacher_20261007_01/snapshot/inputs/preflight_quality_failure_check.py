"""One necessary mocked failure-path check; no real physics/query/solve."""
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import Mock, patch

ROOT = Path(r"E:\v64b31work_20261007_01")
STAGING = Path(r"E:\v64b31_staging_20261007_01")
sys.path.insert(0, str(ROOT))
import mujoco
import numpy as np


def forbidden(*args, **kwargs):
    raise AssertionError("Real simulator, model allocation, geometry, or solver forbidden")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf8")


with patch.object(mujoco, "MjData", side_effect=forbidden), \
     patch.object(mujoco, "mj_step", side_effect=forbidden), \
     patch.object(mujoco, "mj_step1", side_effect=forbidden), \
     patch.object(mujoco, "mj_step2", side_effect=forbidden), \
     patch.object(mujoco, "mj_forward", side_effect=forbidden), \
     patch.object(mujoco, "mj_geomDistance", Mock(return_value=.02)) as query_sentinel:
    from v6_4 import execution_aware_route_teacher as runner
    from v6_lite.hierarchical_qp import HierarchicalVelocityQP

    fixture = STAGING/"preflight_quality_failure_fixture"
    if fixture.exists():
        raise FileExistsError("One-shot fixture already exists; do not silently rerun")
    fixture.mkdir()
    output = fixture/"synthetic_output"
    attempt = output/"attempts"/"MOCK_ONLY"
    directory = output/"slots"/"MOCK_ONLY"
    directory.mkdir(parents=True)
    attempt.mkdir(parents=True)
    task = SimpleNamespace(task_id="MOCK_TASK_NOT_AN_ACTUAL", scenario={
        "workspace_obstacles": [{"name": "irrelevant"}, {"name": "mock_route_sphere"}]},
        sha256=lambda: "MOCK_TASK_SHA")
    residual = SimpleNamespace(definition={"intervals_s": [[0., 1.], [2., 3.], [4., 5.]]},
        z_m=np.zeros((6, 2)))
    slot = {"slot_id": "MOCK_ONLY", "mode": "v2_plus12"}
    evaluation = {"complete": True, "evidence_valid": True, "full_task_success": True,
        **{key: {"passed": True} for key in runner.GATE_NAMES}}
    save(attempt/"task.json", {"mock_only": True})
    save(attempt/"plan.json", {"mock_only": True})
    save(attempt/"evaluation.json", evaluation)
    (attempt/"fake_trace.txt").write_text("mock sentinel, never parsed as a trajectory\n", encoding="utf8")
    result = {"status": "TASK_COMPLETED", "full_task_success": True,
        "full_27s_success": True, "actual_steps": 13500, "fallback_used": False,
        "evaluation": evaluation, "evaluation_path": str(attempt/"evaluation.json"),
        "evaluation_sha256": digest(attempt/"evaluation.json"),
        "trace_path": str(attempt/"fake_trace.txt"),
        "trace_sha256": digest(attempt/"fake_trace.txt"), "synthetic_fixture_only": True}
    save(attempt/"attempt_result.json", result)
    original_result = json.loads(json.dumps(result))
    protected = {str(path): digest(path) for path in attempt.iterdir()}
    save(output/"source_identity.json", {"source_sha256": {}, "protected_artifacts": protected,
        "synthetic_fixture_only": True})

    def fake_quality(*args, **kwargs):
        for _ in range(3):
            mujoco.mj_geomDistance(None, None, 0, 1, 1., np.zeros(6))
        raise RuntimeError("DECLARED_MOCK_FAILURE_AFTER_THREE_SENTINELS")

    with patch.object(HierarchicalVelocityQP, "solve", side_effect=forbidden), \
         patch.object(runner, "build_route_quality", side_effect=fake_quality) as builder, \
         patch.object(runner, "event", Mock()):
        quality, quality_path, costs = runner._quality_with_retained_failure(
            output, directory, attempt, task, residual, slot, result)
    assert builder.call_count == 1, "No retry or replacement"
    assert query_sentinel.call_count == 3
    assert costs["native_geometry_query_calls"] == 3
    row = costs["phase_counts"]["route_quality"]["mj_geomDistance"]
    assert row == {"started": 3, "returned": 3, "raised": 0}
    assert all(costs[key] == 0 for key in ("actual_physics_steps", "private_preview_physics_steps",
        "independent_saved_torque_replay_steps", "qp_solve_calls"))
    assert result == original_result and quality["status"] == result["status"]
    assert quality["quality_label_eligible"] is False
    assert quality["full_metrics"] is None and quality["failed_prefix_metrics"] is None
    assert quality["quality_stage_status"] == "QUALITY_POSTPROCESS_FAILED"
    assert quality["safety"]["full_task_and_original_safety_passed"] is True
    assert quality["safety"]["path_quality_comparison_eligible"] is False
    assert quality["costs"]["additional_route_quality_geometry_queries"] == 3
    assert quality["metric_unavailable"]["error"]["retry_or_replacement"] is False
    assert quality_path.name == "route_quality_failure.json" and quality_path.is_file()
    assert all(digest(path) == sha for path, sha in protected.items())
    receipt = {"schema": "v64_b31_mock_quality_failure_check_v1", "passed": True,
        "tests_run": 1, "invocations": 1, "mocked_build_route_quality_calls": builder.call_count,
        "query_sentinel_calls": query_sentinel.call_count,
        "query_sentinel_started_returned_raised": row,
        "quality_label_eligible": quality["quality_label_eligible"],
        "full_metrics": quality["full_metrics"], "failed_prefix_metrics": quality["failed_prefix_metrics"],
        "original_actual_status_preserved": result == original_result,
        "original_evidence_unchanged": True, "retry_or_replacement": False,
        "scope": "Synthetic fixture tests exception handling and transparent counting; no scene, trajectory, Task, or safety result is measured.",
        "cost": {"actual_physics_steps": 0, "private_preview_physics_steps": 0,
            "independent_saved_torque_replay_steps": 0, "real_geometry_queries": 0,
            "real_qp_solves": 0, "model_allocations": 0, "DDIM_calls": 0,
            "optimizer_updates": 0, "mock_native_geometry_sentinels": 3},
        "source_sha256": {"v6_4/execution_aware_route_teacher.py": digest(ROOT/"v6_4/execution_aware_route_teacher.py"),
            "v6_4/conditional_execution.py": digest(ROOT/"v6_4/conditional_execution.py")},
        "harness_sha256": digest(__file__), "fixture_root": str(fixture),
        "receipt_artifact_sha256": {str(path.relative_to(fixture)): digest(path)
            for path in fixture.rglob("*") if path.is_file()}}
    save(STAGING/"preflight_quality_failure_check.json", receipt)
    print(json.dumps({"passed": receipt["passed"], "cost": receipt["cost"],
        "source_sha256": receipt["source_sha256"]}, indent=2))
