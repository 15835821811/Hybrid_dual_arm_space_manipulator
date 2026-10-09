"""Callable budgeted Cartesian direct search and resumable C.1 command line."""
from __future__ import annotations

import argparse
import copy
from dataclasses import asdict
import json
from pathlib import Path
import sys
import time

import numpy as np

from .route_optimizer_protocol import (PreferenceSpec, SearchSpec, active_intervals, parameter_plan,
    initial_candidates, project_disks, build_reference_definition, digest, read, write)


def admissible(row):
    m = row.get("prediction_metrics") or {}
    return bool(row.get("prediction_admissible") is True and
        all(m.get(k) is not None and np.isfinite(m[k]) for k in ("I_support", "L_full")))


def norm(row):
    return float(np.linalg.norm(row["x_m"]))


def rank_candidates(rows, tie_band=.001, clearance=.030):
    """Strict A, anchored engineering tie A, exact constrained B, diagnostic B."""
    valid = [r for r in rows if admissible(r)]
    strict = min(valid, key=lambda r: (r["prediction_metrics"]["I_support"], r["candidate_id"]), default=None)
    ties = [r for r in valid if r["prediction_metrics"]["I_support"] <= strict["prediction_metrics"]["I_support"] + tie_band] if strict else []
    tie = min(ties, key=lambda r: (r["prediction_metrics"]["L_full"], norm(r),
                  ("v1", "v2").index(r["family"]), r["candidate_id"]), default=None)
    available_b = [r for r in valid if r["prediction_metrics"].get("d_support") is not None
                   and np.isfinite(r["prediction_metrics"]["d_support"])
                   and r["prediction_metrics"]["d_support"] >= clearance]
    b = min(available_b, key=lambda r: (r["prediction_metrics"]["L_full"],
              r["prediction_metrics"]["I_support"], norm(r), r["candidate_id"]), default=None)
    diagnostic = min(valid, key=lambda r: (
        max(0., clearance - r["prediction_metrics"]["d_support"])
            if r["prediction_metrics"].get("d_support") is not None else float("inf"),
        r["prediction_metrics"]["L_full"], r["prediction_metrics"]["I_support"], norm(r), r["candidate_id"]), default=None)
    return {"strict": strict, "tie_selected": tie, "B": b, "best_safe_diagnostic": diagnostic,
            "tie_group_ids": [r["candidate_id"] for r in ties], "B_available_ids": [r["candidate_id"] for r in available_b]}


def non_dominated(rows):
    valid = [r for r in rows if admissible(r) and r["prediction_metrics"].get("d_support") is not None
             and not r["prediction_metrics"].get("clearance_is_censored_lower_bound", False)]
    def vector(r):
        m = r["prediction_metrics"]
        return np.array([m["I_support"], m["L_full"], -m["d_support"]])
    return [r["candidate_id"] for r in valid if not any(
        np.all(vector(other) <= vector(r)) and np.any(vector(other) < vector(r)) for other in valid)]


def _historical_mode(plan, active):
    z = plan.z_m
    return bool((len(active) == 1 or np.all(z[active[:-1]] == 0.))
        and abs(z[2, 1]) <= 1e-14
        and any(abs(z[2, 0] - v) <= 1e-14 for v in (0., -.012, .012, -.020, .020)))


def _preference_selection(rows, active, reason, tie_band=.001):
    ranks = rank_candidates(rows, tie_band)
    result = {}
    for name, winner in (("A", ranks["tie_selected"]), ("B", ranks["B"])):
        status = ("NOT_APPLICABLE" if not active else "BUDGET_EXHAUSTED_WITH_INCUMBENT" if winner and reason == "CANDIDATE_BUDGET_EXHAUSTED" else
            "PLAN_FOUND" if winner else "NO_ADMISSIBLE_PLAN_WITHIN_BUDGET" if not ranks["strict"] else "PREFERENCE_UNMET_WITHIN_BUDGET")
        result[name] = {"status": status, "selected_plan": winner["plan"] if winner else None,
            "source_candidate_id": winner["candidate_id"] if winner else None,
            "prediction_task_passed": winner["prediction_task_passed"] if winner else None,
            "online_guards_passed": winner["online_guards_passed"] if winner else None,
            "preference_met": bool(winner), "prediction_metrics": winner["prediction_metrics"] if winner else None,
            "formal_actual_validation": "NOT_RUN",
            "selected_origin_source": winner.get("origin_source") if winner else None,
            "selected_lineage": winner.get("proposal_lineage") if winner else None,
            "source_attribution": ("direct_learning_seed" if winner.get("source") in ("diffusion", "retrieval", "regression") else
                "learning_seed_descendant" if winner.get("origin_source") in ("diffusion", "retrieval", "regression") else
                "rule_seed_or_descendant") if winner else "no_plan",
            "best_safe_diagnostic": ranks["best_safe_diagnostic"]["candidate_id"] if name == "B" and not winner and ranks["best_safe_diagnostic"] else None}
    return result


def optimize(task, preferences, execution_identity, evaluator, run_root, *, search_spec=None, initializer=None):
    """Optimize a new frozen TaskSpec using an injected nominal evaluator.

    No task-ID lookup, historic oracle or final actual input exists here.
    Both preferences share exactly the same pool and <=12 consumed slots.
    Evaluator receives (immutable plan, candidate_id), and owns fresh rollout
    state plus retained physical receipts. Mock evaluators require no physics.
    """
    started = time.perf_counter()
    spec = search_spec or SearchSpec()
    if not isinstance(spec, SearchSpec):
        raise TypeError("frozen SearchSpec required")
    if not execution_identity or not all(k in execution_identity for k in
            ("source_identity_sha256", "config_sha256", "task_sha256", "model_contract_sha256")):
        raise ValueError("explicit frozen execution/config identity required")
    if execution_identity["task_sha256"] != task.sha256() or execution_identity["model_contract_sha256"] != task.model_contract_sha256:
        raise ValueError("Task and execution identity mismatch")
    active = active_intervals(build_reference_definition(task))
    root = Path(run_root); root.mkdir(parents=True, exist_ok=True)
    inputs = {"task_sha256": task.sha256(), "preferences": [p.to_dict() for p in preferences],
              "execution_identity": execution_identity, "search_spec": asdict(spec)}
    if initializer is not None:
        from .route_initializers import FrozenSeedInitializer, json_raw
        if isinstance(initializer, dict):
            initializer = FrozenSeedInitializer(initializer)
        if not callable(initializer) or not isinstance(getattr(initializer, "identity", None), dict):
            raise TypeError("initializer requires a callable and frozen identity dictionary")
        inputs["initializer_identity"] = json_raw(initializer.identity)
    if (root / "optimizer_inputs.json").exists():
        if digest(read(root / "optimizer_inputs.json")) != digest(inputs):
            raise ValueError("optimizer cache inputs differ")
    else:
        write(root / "optimizer_inputs.json", inputs)
    if (root / "selection.json").exists():
        from .route_candidate_evaluator import verify_seal
        for directory in sorted((root / "predictions").glob("*")):
            verify_seal(directory)
        selection = read(root / "selection.json")
        if selection["input_sha256"] != digest(inputs):
            raise ValueError("retained selection inputs differ")
        if digest(read(root / "candidate_registry.json")) != selection["registry_content_sha256"]:
            raise ValueError("retained candidate registry differs")
        registry = read(root / "candidate_registry.json")
        saved_proposals = read(root / "proposals.json")
        for path in sorted(root.glob("prefix_*.json")):
            snapshot = read(path)
            seal = snapshot.pop("snapshot_content_sha256")
            if digest(snapshot) != seal or digest(registry[:snapshot["budget"]["slots_consumed"]]) != snapshot["registry_content_sha256"]:
                raise ValueError("retained prefix seal or candidate pool differs")
            if digest(saved_proposals[:snapshot["budget"]["proposal_attempts"]]) != snapshot["proposals_content_sha256"]:
                raise ValueError("retained prefix proposal pool differs")
        return selection
    if active:
        if len(preferences) != 2 or {p.name for p in preferences} != {"A", "B"}:
            raise ValueError("A and B must share this single optimizer invocation")
        for p in preferences:
            if p.task_sha256 != task.sha256() or [list(w) for w in p.W_support] != [build_reference_definition(task)["intervals_s"][i] for i in active]:
                raise ValueError("preference Task/window identity mismatch")
    rows = []; cache = {}; proposals = []; hits = 0; cursor = 0; radius_index = 0
    adaptive_slots = 0; sweep_improved = False; sweeps = []
    reason = "NOT_APPLICABLE" if not active else None
    registry_path = root / "candidate_registry.jsonl"
    retained = [json.loads(line) for line in registry_path.read_text(encoding="utf8").splitlines()] if registry_path.exists() else []

    def retain(row):
        rows.append(row)
        position = len(rows) - 1
        if position < len(retained):
            if digest(retained[position]) != digest(row):
                raise ValueError("resumed deterministic sequence differs from retained registry")
        else:
            with registry_path.open("a", encoding="utf8") as stream:
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        print(json.dumps({"event": "C1_CANDIDATE_TERMINAL", "task": task.task_id, "candidate": row["candidate_id"],
            "source": row["source"], "status": row["status"], "slots": len(rows),
            "I_support": (row.get("prediction_metrics") or {}).get("I_support"),
            "d_support": (row.get("prediction_metrics") or {}).get("d_support")}), flush=True)

    def evaluate(proposal):
        nonlocal hits
        if "initializer_slot" in proposal:
            from .route_initializers import raw_seed_plan, json_raw
            plan, diagnostics = raw_seed_plan(task, proposal, proposal["initializer_slot"])
            proposal = {**json_raw(proposal), "raw_seed_diagnostics": diagnostics,
                "proposal_index": len(proposals)}
            if plan is None:
                # No synthetic x or plan: an illegal raw seed consumes a slot,
                # but never reaches the nominal evaluator or exact plan cache.
                proposal["x_m"] = None
                proposals.append(proposal)
                row = {**proposal, "candidate_id": f"C{len(rows):02d}",
                    "status": "INITIALIZER_RAW_REJECTED", "prediction_rollout_started": False,
                    "prediction_steps": 0, "prediction_admissible": False,
                    "prediction_task_passed": False, "online_guards_passed": False,
                    "prediction_metrics": None, "plan": None, "plan_sha256": None,
                    "active_intervals": active, "coefficient_norm_m": None,
                    "non_historical_continuous_point": False,
                    "formal_actual_validation": "NOT_RUN", "costs": {"prediction_physics_steps": 0}}
                retain(row)
                return row
            proposal["x_m"] = plan.z_m[active].reshape(-1).tolist()
        else:
            plan = parameter_plan(task, proposal["family"], proposal["x_m"])
        family = "v1" if not np.any(plan.z_m) else proposal["family"]
        key = digest({"plan_sha256": plan.sha256(), "execution_identity": execution_identity})
        proposal = {**proposal, "family": family, "content_key": key, "proposal_index": len(proposals)}
        proposals.append(proposal)
        if key in cache:
            hits += 1; proposal["cache_hit_candidate_id"] = cache[key]["candidate_id"]
            return None
        cid = f"C{len(rows):02d}"
        evidence = evaluator(plan, cid)
        row = {**proposal, **evidence, "candidate_id": cid, "family": family,
            "plan": plan.to_dict(), "plan_sha256": plan.sha256(), "search_content_key": key,
            "active_intervals": active, "coefficient_norm_m": float(np.linalg.norm(plan.z_m)),
            "non_historical_continuous_point": not _historical_mode(plan, active),
            "formal_actual_validation": "NOT_RUN"}
        retain(row); cache[key] = row
        return row

    seeds = initial_candidates(task) if active else []
    if initializer is not None and active:
        overrides = initializer(task)
        if not isinstance(overrides, dict) or set(overrides) != {1, 3}:
            raise ValueError("initializer must override exactly slots 1 and 3 once")
        seeds = [({**overrides[i], "initializer_slot": i} if i in overrides else seed)
                 for i, seed in enumerate(seeds)]
    seed_count = 0

    def snapshot_prefix(budget, stop_reason="CANDIDATE_BUDGET_EXHAUSTED"):
        # This is called before constructing the next candidate, never from
        # the completed final pool. Disk exclusivity makes prefix seals final.
        path = root / f"prefix_{budget:02d}.json"
        pool = copy.deepcopy(rows)
        ranks = rank_candidates(pool, spec.tie_band_rad_s)
        snapshot = {"schema": "v64_c2_sealed_search_prefix_v1", "task_id": task.task_id,
            "task_sha256": task.sha256(), "input_sha256": digest(inputs),
            "execution_identity": execution_identity, "active_intervals": active,
            "active_dim": 2 * len(active), "prefix_budget": budget,
            "candidate_ids": [r["candidate_id"] for r in pool],
            "registry_content_sha256": digest(pool), "proposals_content_sha256": digest(proposals),
            "preferences": _preference_selection(pool, active, stop_reason, spec.tie_band_rad_s),
            "strict_winner": ranks["strict"]["candidate_id"] if ranks["strict"] else None,
            "tie_selected_winner": ranks["tie_selected"]["candidate_id"] if ranks["tie_selected"] else None,
            "tie_group_ids": ranks["tie_group_ids"], "B_available_ids": ranks["B_available_ids"],
            "finite_non_dominated_ids": non_dominated(pool),
            "comparable_candidate_ids": [r["candidate_id"] for r in pool if admissible(r)],
            "budget": {"candidate_budget": budget, "slots_consumed": len(pool),
                "proposal_attempts": len(proposals), "cache_hits": hits,
                "prediction_rollouts_started": sum(r.get("prediction_rollout_started", False) for r in pool),
                "stop_reason": stop_reason},
            "candidate_costs": {r["candidate_id"]: r.get("costs") for r in pool},
            "elapsed_wall_s": time.perf_counter() - started,
            "protocol_completed": stop_reason != "TOOL_ERROR",
            "formal_actual_validation": "NOT_RUN", "selection_reads_final_actual": False,
            "later_slots_read": False}
        if path.exists():
            saved = read(path); seal = saved.pop("snapshot_content_sha256")
            if digest(saved) != seal or saved["registry_content_sha256"] != snapshot["registry_content_sha256"] or saved["proposals_content_sha256"] != snapshot["proposals_content_sha256"]:
                raise ValueError("sealed prefix changed during resume")
            return
        snapshot["snapshot_content_sha256"] = digest(snapshot)
        write(path, snapshot)

    for i, seed in enumerate(seeds):
        if len(proposals) >= spec.proposal_limit:
            reason = "PROPOSAL_LIMIT"; break
        source = seed.get("source", "initializer_unknown")
        origin = source if source in ("diffusion", "retrieval", "regression") else "zero" if i == 0 else "geometry" if "initializer_slot" not in seed else "initializer_unknown"
        row = evaluate({**seed, "source": source, "parent_candidate_id": None, "preference_center": None,
            "origin_source": origin, "proposal_lineage": [{"initial_position": i, "source": origin}]})
        if row and row.get("tool_error"):
            reason = "TOOL_ERROR"; break
    seed_count = len(rows)
    if len(rows) == 4:
        snapshot_prefix(4, reason or "CANDIDATE_BUDGET_EXHAUSTED")
    moves = [(coordinate, sign) for coordinate, sign in spec.coordinate_sign_order if coordinate < 2 * len(active)] + [("family", 0)]
    while active and reason is None and len(rows) < spec.candidate_budget and len(proposals) < spec.proposal_limit:
        ranks = rank_candidates(rows, spec.tie_band_rad_s)
        preference = "A" if adaptive_slots % 2 == 0 else "B"
        center = ranks["tie_selected"] if preference == "A" else ranks["B"] or ranks["best_safe_diagnostic"]
        # Failed seeds still define a declared fallback center, not infeasibility.
        center = center or next((r for r in rows if r.get("x_m") is not None), None)
        if center is None:
            reason = "NO_VALID_PARAMETER_CENTER"; break
        coordinate, sign = moves[cursor]
        raw = np.asarray(center["x_m"]).copy(); family = center["family"]
        if coordinate == "family":
            family = "v2" if family == "v1" else "v1"
        else:
            raw[coordinate] += sign * spec.step_ladder_m[radius_index]
        proposal = {"family": family, "x_m": project_disks(raw).tolist(), "source": "adaptive_poll",
            "parent_candidate_id": center["candidate_id"], "preference_center": preference,
            "coordinate": coordinate, "sign": sign, "step_m": spec.step_ladder_m[radius_index],
            "poll_cursor": cursor, "pre_projection_x_m": raw.tolist(),
            "projected": not np.array_equal(raw, project_disks(raw)),
            "projection_scope": "original_nonlearning_adaptive_poll_only",
            "origin_source": center.get("origin_source"),
            "proposal_lineage": [*center.get("proposal_lineage", []), {"parent_candidate_id": center["candidate_id"],
                "source": "adaptive_poll", "coordinate": coordinate, "sign": sign}]}
        before = (ranks["strict"]["candidate_id"] if ranks["strict"] else None,
                  ranks["tie_selected"]["candidate_id"] if ranks["tie_selected"] else None,
                  ranks["B"]["candidate_id"] if ranks["B"] else None,
                  ranks["best_safe_diagnostic"]["candidate_id"] if ranks["best_safe_diagnostic"] else None)
        row = evaluate(proposal)
        if row is not None:
            adaptive_slots += 1
            after = rank_candidates(rows, spec.tie_band_rad_s)
            current = tuple(after[k]["candidate_id"] if after[k] else None for k in ("strict", "tie_selected", "B", "best_safe_diagnostic"))
            sweep_improved |= current != before
            if row.get("tool_error"):
                reason = "TOOL_ERROR"
        cursor += 1
        if cursor == len(moves):
            sweeps.append({"step_m": spec.step_ladder_m[radius_index], "improved": sweep_improved,
                           "slots_after": len(rows), "complete": True})
            cursor = 0
            if not sweep_improved:
                radius_index += 1
                if radius_index == len(spec.step_ladder_m):
                    reason = "MINIMUM_STEP_POLL_COMPLETE"
            sweep_improved = False
        if row is not None and len(rows) in (4, 8, 12) and len(rows) <= spec.candidate_budget:
            snapshot_prefix(len(rows), reason or "CANDIDATE_BUDGET_EXHAUSTED")
    if reason is None:
        reason = "CANDIDATE_BUDGET_EXHAUSTED" if len(rows) >= spec.candidate_budget else "PROPOSAL_LIMIT"
    ranks = rank_candidates(rows, spec.tie_band_rad_s)
    frontier = non_dominated(rows)
    selection = {"schema": "v64_c1_selection_v1", "task_id": task.task_id, "task_sha256": task.sha256(),
        "input_sha256": digest(inputs), "execution_identity": execution_identity,
        "active_intervals": active, "active_dim": 2 * len(active), "preferences": {},
        "strict_winner": ranks["strict"]["candidate_id"] if ranks["strict"] else None,
        "tie_selected_winner": ranks["tie_selected"]["candidate_id"] if ranks["tie_selected"] else None,
        "tie_group_ids": ranks["tie_group_ids"], "B_available_ids": ranks["B_available_ids"],
        "finite_non_dominated_ids": frontier, "comparable_candidate_ids": [r["candidate_id"] for r in rows if admissible(r)],
        "seed_only_selection": {k: (v["candidate_id"] if isinstance(v, dict) else v)
                               for k, v in rank_candidates(rows[:seed_count]).items()},
        "budget": {"candidate_budget": spec.candidate_budget, "slots_consumed": len(rows),
            "proposal_attempts": len(proposals), "proposal_limit": spec.proposal_limit, "cache_hits": hits,
            "prediction_rollouts_started": sum(r.get("prediction_rollout_started", False) for r in rows),
            "continuous_candidates_constructed": sum(p["source"] == "adaptive_poll" for p in proposals),
            "non_historical_continuous_candidates_evaluated": sum(r["non_historical_continuous_point"] and r.get("prediction_rollout_started", False) for r in rows),
            "shared_A_B_pool": True, "stop_reason": reason, "poll_truncated": cursor != 0,
            "completed_sweeps": sweeps, "current_step_m": spec.step_ladder_m[min(radius_index, 2)]},
        "formal_actual_validation": "NOT_RUN", "selection_reads_final_actual": False}
    selection["preferences"] = _preference_selection(rows, active, reason, spec.tie_band_rad_s)
    selection["elapsed_wall_s"] = time.perf_counter() - started
    # Early normal termination covers larger requested endpoints with its
    # actual pool and cost; a tool failure keeps protocol_completed false.
    for budget in (4, 8, 12):
        if budget <= spec.candidate_budget and not (root / f"prefix_{budget:02d}.json").exists():
            snapshot_prefix(budget, reason)
    selection["sealed_prefixes"] = {str(budget): f"prefix_{budget:02d}.json" for budget in (4, 8, 12) if budget <= spec.candidate_budget}
    selection["registry_content_sha256"] = digest(rows)
    write(root / "candidate_registry.json", rows)
    write(root / "proposals.json", proposals)
    write(root / "cost_ledger.json", {"budget": selection["budget"], "candidate_costs": {r["candidate_id"]: r.get("costs") for r in rows}})
    write(root / "selection.json", selection)
    return selection


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "analyze-history", "optimize", "execute-selected", "validate", "report", "run-all"))
    parser.add_argument("--output")
    parser.add_argument("--run")
    args = parser.parse_args()
    from . import evaluate_route_optimizer as pipeline
    try:
        if args.command == "prepare":
            if not args.output:
                parser.error("prepare requires --output")
            from .route_optimizer_protocol import prepare
            result = prepare(args.output)
            print(json.dumps({"event": "C1_PREPARED", "run": args.output, "tasks": len(result["tasks"])}), flush=True)
        else:
            if not args.run:
                parser.error("this command requires --run")
            pipeline.dispatch(args.command, Path(args.run).resolve())
    except Exception:
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
