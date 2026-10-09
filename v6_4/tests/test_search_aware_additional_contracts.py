"""Post-freeze C.3 verification additions; authored after producer 9f39b42.

These six contracts are not experiment producers and are not a test receipt.
Run only after formal request timing. All work uses temporary small fixtures.
No historical archive, real DDIM, optimizer update, physics or native geometry
is required. The QP history check mocks its solver and validator; it does not
claim that a complete physical runner was constructed.
"""
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import mujoco
import numpy as np
import pytest
import torch

from v6_4 import closed_loop_warmstart_validation as validation
from v6_4 import route_candidate_evaluator as candidate
from v6_4 import search_aware_warmstart_experiment as c3
from v6_4 import simple_warmstart_regression as models
from v6_4.residual_diffusion import ResidualDDPM
from v6_4.route_initializers import raw_seed_plan
from v6_4.route_optimizer_protocol import digest, read, write
from v6_4.task_protocol import canonical_json
from v6_4.tests.test_execution_diagnostics import fixed_qp_fixture
from v6_4.tests.test_preference_diffusion_warmstart import fixture as dataset_fixture
from v6_4.tests.test_task_protocol import fixture_task
from v6_lite.hierarchical_qp import HierarchicalVelocityQP


def forbidden(*args, **kwargs):
    raise AssertionError("post-freeze contract forbids scientific execution")


@pytest.fixture(autouse=True)
def no_scientific_execution(monkeypatch):
    for name in ("mj_step", "mj_step1", "mj_step2", "mj_forward",
                 "mj_geomDistance", "MjData", "MjModel"):
        monkeypatch.setattr(mujoco, name, forbidden)
    monkeypatch.setattr(ResidualDDPM, "sample_ddim", forbidden)
    monkeypatch.setattr(torch.optim.AdamW, "step", forbidden)
    monkeypatch.setattr(c3.subprocess, "run", forbidden)


@dataclass(frozen=True)
class SceneObstacle:
    radius: float
    center: object


@dataclass(frozen=True)
class Scene:
    scenario_id: str
    obstacles: tuple

    def to_dict(self):
        return {"scenario_id": self.scenario_id,
                "obstacles": [{"radius": o.radius, "center": list(o.center)}
                              for o in self.obstacles]}


class SmallTask:
    model_contract_sha256 = "model"
    initial_qpos = [0.] * 81
    initial_qvel = [0.] * 79
    initial_planner_q = [0.] * 17
    initial_planner_dq = [0.] * 17

    def __init__(self, task_id):
        self.task_id = task_id

    def sha256(self):
        return "sha-" + self.task_id


class SmallPlan:
    def __init__(self, value):
        self.value = value

    def sha256(self):
        return digest(self.value)

    def to_dict(self):
        return self.value


def validation_fixture(root, *, poison=False, execute=True):
    """Ten sealed selections and twenty mock endpoints; no physical executor."""
    run = root / "run"
    phase = root / "val"
    for name in ("source_identity.json", "frozen_execution_config.json",
                 "frozen_run_config.json"):
        write(run / name, {"identity": name})
    checkpoints = {}
    for endpoint in validation.CHECKPOINTS:
        path = root / (endpoint + ".pt")
        path.write_bytes(endpoint.encode())
        checkpoints[endpoint] = path
    entries = []
    for tid in ("val_a", "val_b"):
        for endpoint in validation.VAL_ENDPOINTS:
            directory = root / "search" / tid / endpoint
            metrics = {"I_support": .01, "L_full": .3, "d_support": .031}
            rows = [{"candidate_id": "C00", "prediction_admissible": True,
                     "prediction_metrics": metrics, "status": "PREDICTED_COMPLETE",
                     "costs": {k: 10 for k in validation.COST_ORDER[:-1]}}]
            selection = {
                "task_id": tid, "task_sha256": SmallTask(tid).sha256(),
                "selection_reads_final_actual": False,
                "preferences": {p: {
                    "selected_plan": {"endpoint": endpoint, "preference": p},
                    "source_candidate_id": "C00", "prediction_metrics": metrics}
                    for p in ("A", "B")},
                "budget": {"candidate_budget": 12 if endpoint == "R12" else 8,
                           "slots_consumed": 1, "stop_reason": "MINIMUM_STEP_POLL_COMPLETE"},
                "registry_content_sha256": digest(rows)}
            if poison:
                rows[0]["TEST"] = {"preferred_checkpoint": "D4000", "score": -1e30}
                selection["test_score_override"] = {"D4000": 1e30, "S4000": 1e30}
                selection["test_results"] = {"path": str(root / "test_search/poison.json")}
                selection["registry_content_sha256"] = digest(rows)
            paths = {"selection_path": directory / "selection.json",
                     "candidate_registry_path": directory / "candidate_registry.json",
                     "proposals_path": directory / "proposals.json",
                     "planning_cost_path": directory / "planning_cost.json"}
            for key, value in (
                    ("selection_path", selection), ("candidate_registry_path", rows),
                    ("proposals_path", []),
                    ("planning_cost_path", {"end_to_end_cold_planning_s": 2.})):
                write(paths[key], value)
            entries.append({"task_id": tid, "endpoint": endpoint, **paths})
    validation.freeze_phase_selections(phase, entries, checkpoint_files=checkpoints)

    def runner(task, plan, directory, **kwargs):
        return {"status": "TASK_COMPLETED", "actual_runner_started": True,
                "entered_actual": True, "actual_steps": 13500,
                "full_task_success": True,
                "evaluation": {"evidence_valid": True,
                               **{g: {"passed": True} for g in validation.FIVE_GATES}},
                "quality": {"I_support": .01, "L_full": .3, "d_support": .031},
                "costs": {}, "elapsed_wall_s": 1.}

    if execute:
        for tid in ("val_a", "val_b"):
            validation.execute_frozen_task(phase, SmallTask(tid), {},
                execution_run=run, executor=runner, plan_loader=SmallPlan)
    return run, phase


def test_c3_prepare_preserves_historical_TaskSpec_bytes_and_external_roles(
        tmp_path, monkeypatch):
    # C3 prepare is exercised, including its real copy/write/split logic.
    # Only scene/model/geometry production and identity acquisition are mocked.
    import v6_4.preference_warmstart_protocol as old_protocol
    import v6_4.route_pair_protocol as pair_protocol
    import v6_lite.run_v6_lite as runner

    base = fixture_task()
    old_root = tmp_path / "c2"
    rows = []
    before = {}
    for group in range(3):
        for side in ("plus", "minus"):
            tid = f"old_{group}_{side}"
            task = replace(base, task_id=tid, group_id=f"old_{group}", split="test")
            path = old_root / "frozen_tasks" / tid / "task.json"
            write(path, task.to_dict())
            write(path.parent / "preferences.json", [])
            write(path.parent / "geometry_precheck.json", {"passed": True})
            before[path] = path.read_bytes()
            rows.append({"task_id": tid, "task_sha256": task.sha256(),
                         "mother_id": f"old_{group}", "mother_source_sha256": f"oldsha{group}",
                         "group_id": f"old_{group}", "split": "train",
                         "geometry_precheck_passed": True})
    write(old_root / "learning_split_manifest.json", {"tasks": rows})
    for name in ("frozen_execution_config.json", "frozen_run_config.json"):
        write(old_root / name, {})
    monkeypatch.setattr(c3, "C2", old_root)
    monkeypatch.setattr(c3, "ROOT", tmp_path)
    monkeypatch.setattr(c3, "_used_seeds", lambda: (set(), []))
    choices = Mock(return_value=[(1, 111), (2, 222), (3, 333)])
    monkeypatch.setattr(old_protocol, "next_unused_seeds", choices)

    def git(*args):
        return {"status": "", "merge-base": c3.BASE,
                "branch": c3.BRANCH}.get(args[0], "mock-head")
    monkeypatch.setattr(c3, "git", git)
    spec = SimpleNamespace(_source_assets=lambda: [])
    monkeypatch.setattr(runner, "default_v6_lite_robot_spec", lambda: spec)
    scenes = Mock(side_effect=lambda spec, cfg: [
        Scene(f"mother_{cfg.seed}",
              (SceneObstacle(.1, (0., 0., 0.)), SceneObstacle(.025, (0., 0., 0.))))])
    monkeypatch.setattr(runner, "build_scenarios", scenes)

    def task_from_scene(spec, scene, **kwargs):
        scenario = {"workspace_obstacles": [{"name": "base"}, {"name": "sphere"}],
                    "source_scene": scene.to_dict()}
        return replace(base, task_id=kwargs["task_id"], group_id=kwargs["group_id"],
                       split=kwargs["split"], scenario_json=canonical_json(scenario),
                       requirements=kwargs.get("requirements", base.requirements))
    monkeypatch.setattr(c3, "task_from_scenario", task_from_scene)
    definition = {"intervals_s": [[float(i), float(i + 1)] for i in range(6)],
                  "transverse_bases": [np.eye(3)[:, :2].tolist()] * 6,
                  "interval_mask": [False, True, True, False, False, False]}
    monkeypatch.setattr(c3, "build_reference_definition", lambda task: definition)
    monkeypatch.setattr(c3, "active_intervals", lambda value: [1, 2])
    monkeypatch.setattr(c3, "_target", lambda task: SimpleNamespace(
        sample=lambda t: (np.zeros(3), np.zeros(3))))
    monkeypatch.setattr(pair_protocol, "_geometry_check", lambda *args: {
        "passed": False, "native_distance_queries": 0})
    monkeypatch.setattr(pair_protocol, "validate_pair_contract", lambda *args: None)
    monkeypatch.setattr(c3, "related_pairs", lambda *args: (
        SimpleNamespace(_pair_policy_sha256="policy"),
        [SimpleNamespace(geom_a_name="body", geom_b_name="sphere")]))

    run = tmp_path / "prepared"
    plan = c3.prepare(run)
    assert len(plan["tasks"]) == 12
    choices.assert_called_once_with(set(), count=3)
    assert scenes.call_count == 3  # Failed prechecks never redraw a mother.
    for source, original in before.items():
        copy = run / source.relative_to(old_root)
        assert source.read_bytes() == original == copy.read_bytes()
        assert read(copy)["split"] == "test"
    old_rows = [r for r in plan["tasks"] if r["historical"]]
    assert len(old_rows) == 6
    assert all(r["split"] == "train" and r["original_task_declared_split"] == "test"
               for r in old_rows)
    assert all(not r["geometry_precheck_passed"] for r in plan["tasks"]
               if not r["historical"])
    assert len(c3.validate_splits(read(run / "split_manifest.json"))) == 12


def test_c3_two_proposals_raw_rejection_and_retained_initializers_never_regenerate(
        tmp_path, monkeypatch):
    ds = dataset_fixture()  # Small task JSON/arrays only; no archive trace loads.
    task = ds.tasks["v0"]
    calls = []
    def sample(task, preference, family, noise):
        calls.append((preference, family, noise.copy()))
        raw = np.zeros((6, 2))
        raw[np.flatnonzero(~ds.search_masks[0])[0], 0] = 1e-30
        return raw, {"source": "diffusion", "preference": preference, "family": family}

    sampler = object.__new__(models.SearchAwareSampler)
    sampler.model_name = "D"
    sampler.sample = sample
    proposals = sampler.initializer_proposals(task, noise_seed=models.TRAINING_DEFAULTS["val_noise_seed"])
    assert list(proposals) == [1, 3]
    assert [(p, f) for p, f, n in calls] == [("A", "v1"), ("B", "v2")]
    for slot, proposal in proposals.items():
        plan, diag = raw_seed_plan(task, proposal, slot)
        assert plan is None and not diag["raw_repaired"] and not diag["resampled"]
    # Unsupported returns still occupy their declared proposal; no extra sample.
    calls.clear()
    def unsupported(task, preference, family, noise):
        calls.append((preference, family))
        return None, {"source": "diffusion", "preference": preference, "family": family,
                      "initializer_rejection": "UNSUPPORTED_TRAINING_CONDITION"}
    sampler.sample = unsupported
    absent = sampler.initializer_proposals(task, noise_seed=models.TRAINING_DEFAULTS["val_noise_seed"])
    assert list(absent) == [1, 3] and len(calls) == 2
    assert all(p["raw_z_m"] is None for p in absent.values())

    run = tmp_path / "run"
    write(run / "dataset/manifest.json", {"small_fixture": True})
    checkpoint = run / "models/D/checkpoint_0250.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"mock checkpoint; never loaded")
    frozen = {"task_id": task.task_id, "task_sha256": task.sha256()}
    monkeypatch.setattr(c3, "load_task", lambda *args: (task, []))
    load = Mock(return_value=SimpleNamespace(
        initializer_proposals=Mock(return_value=proposals)))
    monkeypatch.setattr(models, "SearchAwareSampler", load)
    first = c3._initializers(run, frozen, "D250", "val")
    path = c3.stream_path(run, frozen, "D250", "val") / "initializer_proposals.json"
    original = path.read_bytes()
    assert c3._initializers(run, frozen, "D250", "val") == first
    assert path.read_bytes() == original
    load.assert_called_once()
    load.return_value.initializer_proposals.assert_called_once()
    assert c3.BudgetLedger(run).totals()["val_ddim_samples"] == 2


def test_val_scoring_forbids_real_TEST_paths_and_ignores_TEST_fields(tmp_path):
    clean = tmp_path / "clean"
    clean.mkdir()
    _, clean_phase = validation_fixture(clean)
    expected = validation.score_checkpoints(clean_phase)
    poison = tmp_path / "poison"
    poison.mkdir()
    _, phase = validation_fixture(poison, poison=True)
    forbidden_roots = [poison / "test_search", poison / "frozen_test"]
    for root in forbidden_roots:
        root.mkdir()
        (root / "poison.json").write_text("malformed TEST must never be opened", encoding="utf8")
    original_open = Path.open
    attempted = []

    def guarded_open(path, *args, **kwargs):
        resolved = path.resolve()
        if any(resolved == root or root in resolved.parents for root in forbidden_roots):
            attempted.append(str(resolved))
            raise AssertionError("VAL touched a TEST input")
        return original_open(path, *args, **kwargs)

    with patch.object(Path, "open", guarded_open):
        observed = validation.score_checkpoints(phase)
    assert not attempted
    assert observed["scores"] == expected["scores"]
    assert observed["selected_checkpoint_D"] == expected["selected_checkpoint_D"] == "D250"
    assert observed["selected_checkpoint_S"] == expected["selected_checkpoint_S"] == "S250"


def test_same_candidate_retains_qp_history_and_explicit_reset_clears_it(monkeypatch):
    # Exercise real history plumbing, while both numerical solver and action
    # validator are mocked. This does not construct a full physical runner.
    import v6_lite.hierarchical_qp as hierarchy
    qp, data, arguments = fixed_qp_fixture()
    qp._unconstrained_solve = Mock(return_value=np.zeros(17))
    velocity = np.arange(17, dtype=float) * .001
    solver_inputs = []
    duals = []

    def solver(hessian, linear, matrix, lower, upper, initial, initial_dual):
        solver_inputs.append((initial.copy(), initial_dual.copy()))
        dual = np.arange(len(lower), dtype=float) + 1.
        duals.append(dual)
        return velocity.copy(), True, "mock_solved", 1, dual

    qp._solve_qp_admm = Mock(side_effect=solver)
    selected = SimpleNamespace(selected_command=velocity.copy(), mode="TRACK",
                               failure_reason="none")
    monkeypatch.setattr(hierarchy, "validate_action", Mock(return_value=selected))
    HierarchicalVelocityQP.solve(qp, data, **arguments)
    HierarchicalVelocityQP.solve(qp, data, **arguments)
    np.testing.assert_array_equal(solver_inputs[0][0], np.zeros(17))
    np.testing.assert_array_equal(solver_inputs[0][1], np.zeros(len(duals[0])))
    np.testing.assert_array_equal(solver_inputs[1][0], velocity)
    np.testing.assert_array_equal(solver_inputs[1][1], duals[0])
    assert qp.solve_count == 2
    qp.reset()
    assert qp.solve_count == 0 and qp._previous_constraint_dual == {}
    HierarchicalVelocityQP.solve(qp, data, **arguments)
    np.testing.assert_array_equal(solver_inputs[2][0], np.zeros(17))
    np.testing.assert_array_equal(solver_inputs[2][1], np.zeros(len(duals[0])))


def test_preview_refusal_keeps_original_stage_reason_and_is_not_NO_PLAN_or_actual_failure(
        tmp_path, monkeypatch):
    import v6_4.evaluate_planning as planning
    import v6_4.reference_adapter as adapter
    import v6_4.residual_execution as execution
    import v6_4.route_optimizer_protocol as protocol
    import v6_4.task_anchored_reference as reference
    import v6_lite.hierarchical_qp as hierarchy
    import v6_lite.run_v6_lite as runner

    task, plan = SmallTask("mock"), SmallPlan({"z": [0., 0.]})
    run = tmp_path / "run"
    for name in ("source_identity.json", "frozen_execution_config.json",
                 "frozen_run_config.json"):
        write(run / name, {})
    monkeypatch.setattr(protocol, "verify_frozen", lambda *args: None)
    monkeypatch.setattr(reference, "reference_precheck", lambda *args, **kwargs: {"passed": True})
    monkeypatch.setattr(reference, "TaskAnchoredResidualReferenceProvider", lambda *args: object())
    monkeypatch.setattr(adapter, "scenario_from_task",
                        lambda *args: SimpleNamespace(scenario_id=task.task_id))
    monkeypatch.setattr(runner, "V6LiteRunConfig",
                        lambda **kwargs: SimpleNamespace(validate=lambda: None))
    monkeypatch.setattr(hierarchy, "HierarchicalQPConfig", lambda **kwargs: object())
    monkeypatch.setattr(runner, "default_v6_lite_robot_spec", lambda: object())
    monkeypatch.setattr(execution, "start_run", lambda *args, **kwargs: {})
    monkeypatch.setattr(execution, "finish_run", lambda *args, **kwargs: None)
    monkeypatch.setattr(planning, "_execution_trace_checks", lambda *args: {"passed": False})
    monkeypatch.setattr(candidate, "fresh_quality", lambda *args, **kwargs: (
        {"I_support": .01, "L_full": .1, "d_support": .03}, {"passed": False}))
    include_receipt = [True]

    def refuse(spec, cfg, qp, scene, traces, **kwargs):
        failures = traces.parent / "failures"
        failures.mkdir(parents=True)
        np.savez(failures / "mock_partial_trace.npz", torque=np.zeros((10, 67)))
        if include_receipt[0]:
            write(failures / "preview_refusal.json", {
                "stage": "private_preview", "failure_reason": "PREVIEW_REJECTED",
                "attempted_private_preview_steps": 4})
        raise RuntimeError("private preview rejected: PREVIEW_REJECTED")

    monkeypatch.setattr(runner, "run_synchronous_scenario", refuse)
    evaluator = candidate.NominalCandidateEvaluator(run, task, {"geometry_precheck_passed": True,
                                                               "obstacle_name": "sphere"})
    refused = evaluator(plan, "C00")
    assert refused["status"] == "EXECUTION_REFUSED"
    assert refused["prediction_steps"] == 10
    assert not refused["prediction_admissible"] and "tool_error" not in refused
    assert "PREVIEW_REJECTED" in refused["execution_failure"]["message"]
    assert len(refused["failure_receipts"]) == 1
    receipt = run / "planning/mock/predictions/C00" / refused["failure_receipts"][0]
    assert read(receipt)["stage"] == "private_preview"
    assert read(receipt)["attempted_private_preview_steps"] == 4
    assert not (run / "actual").exists()
    include_receipt[0] = False
    unbound = evaluator(plan, "C01")
    assert unbound["status"] == "TOOL_ERROR"
    assert unbound["prediction_steps"] == 10 and unbound["tool_error"]
    candidate.verify_seal(run / "planning/mock/predictions/C00")


def test_incomplete_consumed_units_and_missing_original_timers_never_retry(
        tmp_path, monkeypatch):
    import v6_4.route_optimizer_protocol as protocol

    monkeypatch.setattr(c3, "verify_run", lambda *args: {})
    monkeypatch.setattr(protocol, "verify_frozen", lambda *args: None)
    task, plan = SmallTask("mock"), SmallPlan({"z": [0., 0.]})
    frozen = {"task_id": task.task_id, "task_sha256": task.sha256()}
    run = tmp_path / "run"
    monkeypatch.setattr(c3, "load_task", lambda *args: (task, []))
    monkeypatch.setattr(models, "SearchAwareSampler", forbidden)

    initial = c3.stream_path(run, frozen, "D250", "val")
    write(initial / "initializer_started.json", {"consumed": True})
    started_bytes = (initial / "initializer_started.json").read_bytes()
    with pytest.raises(RuntimeError, match="unfinished initializer"):
        c3._initializers(run, frozen, "D250", "val")
    assert (initial / "initializer_started.json").read_bytes() == started_bytes
    assert not (initial / "initializer_proposals.json").exists()

    evaluator = object.__new__(candidate.NominalCandidateEvaluator)
    evaluator.run, evaluator.task, evaluator.frozen = run, task, {}
    evaluator.execution_identity = {}
    prediction = run / "planning/mock/predictions/C00"
    write(prediction / "started.json", {"consumed": True})
    original = (prediction / "started.json").read_bytes()
    with pytest.raises(RuntimeError, match="unfinished consumed candidate"):
        evaluator(plan, "C00")
    assert (prediction / "started.json").read_bytes() == original
    assert not (prediction / "result.json").exists()

    phase_root = tmp_path / "actual_case"
    phase_root.mkdir()
    execution_run, phase = validation_fixture(phase_root, execute=False)
    actual = phase / "actual/val_a/R12_A"
    write(actual / "started.json", {"consumed": True})
    original = (actual / "started.json").read_bytes()
    with pytest.raises(RuntimeError, match="unfinished consumed actual"):
        validation.execute_frozen_task(phase, SmallTask("val_a"), {},
            execution_run=execution_run, executor=forbidden, plan_loader=SmallPlan)
    assert (actual / "started.json").read_bytes() == original
    assert not (actual / "slot.json").exists()

    for name in ("plan.json", "source_identity.json", "frozen_execution_config.json",
                 "frozen_run_config.json"):
        write(run / name, {})
    monkeypatch.setattr(c3, "BudgetLedger", lambda *args: SimpleNamespace(reserve=Mock()))
    monkeypatch.setattr(candidate, "NominalCandidateEvaluator", forbidden)
    selected = c3.stream_path(run, frozen, "R", "test")
    write(selected / "planning/mock/selection.json", {"consumed": True})
    original = (selected / "planning/mock/selection.json").read_bytes()
    with pytest.raises(RuntimeError, match="without original planning timer"):
        c3.run_stream(run, frozen, "R", 12, stage="test")
    assert (selected / "planning/mock/selection.json").read_bytes() == original
    assert not (selected / "planning_cost.json").exists()
    write(selected / "planning_cost.json", {"original_elapsed_s": 2.})
    timer_bytes = (selected / "planning_cost.json").read_bytes()
    with pytest.raises(RuntimeError, match="lacks outer timer"):
        c3.run_formal_stream(run, frozen, "R", "test")
    assert (selected / "planning_cost.json").read_bytes() == timer_bytes
    assert not (selected / "outer_process.json").exists()
    write(selected / "outer_process.json", {"exit_code": 1, "elapsed_wall_s": 3.})
    receipt_bytes = (selected / "outer_process.json").read_bytes()
    with pytest.raises(RuntimeError, match="retained worker failure"):
        c3.run_formal_stream(run, frozen, "R", "test")
    assert (selected / "outer_process.json").read_bytes() == receipt_bytes

