"""Fixed-noise obstacle-only diagnostics for the sealed B.2 checkpoint.

``inspect`` reads provenance and condition wiring without DDIM or physics.
``run`` requires a pre-frozen plan byte hash and performs exactly its 32 DDIM
calls.  There is no execution, model fitting, clipping, projection or retry.
The old public TEST tasks are diagnostics, never a new independent test set.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import itertools
import json
from pathlib import Path
import time

import numpy as np

from .dataset import object_sha, sha256
from .residual_dataset import (ConditionNormalizer, _read, _write,
                               encode_residual_condition, validate_z)
from .task_protocol import TaskSpec

EXPECTED_CHECKPOINT_SHA256 = "9ace53f482253216cea7baabed7479d84f37a9e80e71231219c7e2f9d3c94ca1"
EXPECTED_STATE_SHA256 = "4fcfbfadec5541d91d0d0c4829f5eecf19bfb71495c25432e0eb0621354d7473"
SCHEMA = "v6_4_b3_old_checkpoint_obstacle_response_v1"


def _record(path):
    p = Path(path).resolve()
    return {"path": str(p), "sha256": sha256(p), "bytes": p.stat().st_size}


def _canonical_newline_sha(path):
    data = Path(path).read_bytes().replace(b"\r\n", b"\n")
    return hashlib.sha256(data).hexdigest()


def inspect_source(old_output, *, checkpoint=None):
    """Read old bytes and verify imported algorithm differs at most in CRLF."""
    import torch

    out = Path(old_output).resolve()
    path = Path(checkpoint).resolve() if checkpoint else out / "training/model/selected.pt"
    if sha256(path) != EXPECTED_CHECKPOINT_SHA256:
        raise ValueError("P0 must use the sealed B.2 selected update250 checkpoint")
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if (ckpt["schema"] != "v6_4_residual_ddpm_checkpoint_v1"
            or ckpt["update"] != 250 or ckpt["state_sha256"] != EXPECTED_STATE_SHA256
            or ckpt["model_config"]["condition_dim"] != 878
            or sum(ckpt["sample_exposures"].values()) != 8000):
        raise ValueError("checkpoint identity/update/dim/exposure differs")
    source_bindings = []
    repo = Path(__file__).resolve().parents[1]
    for old_name, expected in ckpt["training_config"]["source_files"].items():
        old_path = Path(old_name)
        if sha256(old_path) != expected:
            raise ValueError("sealed checkpoint producer source changed: " + old_name)
        # B.2 and B.3 worktrees have identical repository-relative layout.
        relative = Path(*old_path.parts[-2:])
        current = repo / relative
        if _canonical_newline_sha(old_path) != _canonical_newline_sha(current):
            raise ValueError("imported algorithm substantively differs from checkpoint producer: " + str(relative))
        source_bindings.append({"old": _record(old_path), "imported": _record(current),
                                "difference_at_most_crlf_lf": True,
                                "canonical_newline_sha256": _canonical_newline_sha(current)})
    for name in ("condition_normalizer", "residual_normalizer"):
        if _read(out / ("training/" + name + ".json")) != ckpt[name]:
            raise ValueError("TRAIN scaler file differs from embedded checkpoint scaler")
    cn = ckpt["condition_normalizer"]
    names = cn["feature_names_with_units"]
    obstacle_indices = [i for i, n in enumerate(names) if n.startswith("obstacle.")]
    if len(names) != 878 or obstacle_indices != list(range(582, 774)):
        raise ValueError("unexpected old B.2 condition layout")
    if any("relative" in n.lower() for n in names):
        raise ValueError("relative feature layout now needs a declared recomputation rule")
    sources = {n: _record(out / n) for n in (
        "manifest.json", "verification.json", "tasks.json", "definitions.json",
        "training/training_config.json", "training/condition_normalizer.json",
        "training/residual_normalizer.json", "training/model/training_report.json")}
    return {"schema": SCHEMA, "checkpoint": _record(path), "selected_update": ckpt["update"],
            "state_sha256": ckpt["state_sha256"], "selected_exposures": 8000,
            "validation_loss": ckpt["validation_loss"], "condition_dim": len(names),
            "condition_schema": cn["schema"], "condition_frame": "world",
            "obstacle_feature_indices_zero_based": obstacle_indices,
            "obstacle_feature_names": [names[i] for i in obstacle_indices],
            "obstacle_layout": "eight canonical physical rows: present(1), types(2), center(3), rotation(9), size(3), linear/angular velocities(3+3)",
            "relative_obstacle_derived_feature_names": [],
            "relative_feature_handling": "NONE_PRESENT_IN_FROZEN_878_SCHEMA; obstacle primitive fields re-encoded canonically",
            "train_scaler_task_ids": cn["fit_unique_task_ids"],
            "condition_scaler_object_sha256": object_sha(cn),
            "residual_scaler_object_sha256": object_sha(ckpt["residual_normalizer"]),
            "sources": sources, "producer_source_bindings": source_bindings,
            "ddim_calls": 0, "physics_steps": 0, "optimizer_updates": 0,
            "old_public_test_role": "DIAGNOSTIC_ONLY_NOT_NEW_INDEPENDENT_TEST"}


def obstacle_condition_view(task, definition, donor_task, normalizer, swap):
    """Encode a counterfactual obstacle view without changing the actual task.

    The encoder checks a task SHA binding.  A temporary, condition-only copy of
    that one binding reflects the synthetic obstacle view; the original full
    definition, anchors, interval/basis/mask and actual TaskSpec stay untouched.
    The returned vectors are the only objects passed to the neural network.
    """
    task = task.to_dict() if isinstance(task, TaskSpec) else copy.deepcopy(task)
    donor = donor_task.to_dict() if isinstance(donor_task, TaskSpec) else copy.deepcopy(donor_task)
    original_task_sha, original_definition_sha = object_sha(task), object_sha(definition)
    if definition["task_sha256"] != original_task_sha or definition["task_id"] != task["task_id"]:
        raise ValueError("actual task/definition mismatch")
    hybrid = copy.deepcopy(task)
    actual_obstacles, donor_obstacles = task["scenario"]["workspace_obstacles"], donor["scenario"]["workspace_obstacles"]
    mode = swap["mode"]
    if mode == "all_workspace_obstacles":
        hybrid["scenario"]["workspace_obstacles"] = copy.deepcopy(donor_obstacles)
        swapped_names = [o.get("name") for o in donor_obstacles]
    elif mode == "matching_obstacle_fields":
        prefix = swap["name_prefix"]
        fields = swap["fields"]
        allowed = {"center_w", "radius_m", "size_m", "rotation_world", "linear_velocity_w_m_s", "angular_velocity_w_rad_s"}
        if not prefix or not fields or len(set(fields)) != len(fields) or not set(fields) <= allowed:
            raise ValueError("invalid obstacle-only swap fields")
        selected = [o for o in actual_obstacles if str(o.get("name", "")).startswith(prefix)]
        donors = [o for o in donor_obstacles if str(o.get("name", "")).startswith(prefix)]
        if len(selected) != 1 or len(donors) != 1:
            raise ValueError("prefix must select exactly one obstacle in each task")
        for obstacle in hybrid["scenario"]["workspace_obstacles"]:
            if obstacle.get("name") == selected[0].get("name"):
                for field in fields:
                    if field not in donors[0]:
                        raise ValueError("declared donor obstacle field missing: " + field)
                    obstacle[field] = copy.deepcopy(donors[0][field])
        swapped_names = [selected[0].get("name")]
    else:
        raise ValueError("unknown frozen obstacle swap mode")
    remainder_original, remainder_hybrid = copy.deepcopy(task), copy.deepcopy(hybrid)
    remainder_original["scenario"].pop("workspace_obstacles")
    remainder_hybrid["scenario"].pop("workspace_obstacles")
    if remainder_original != remainder_hybrid:
        raise AssertionError("counterfactual changed a non-obstacle declaration")
    condition_definition = copy.deepcopy(definition)
    condition_definition["task_sha256"] = object_sha(TaskSpec.from_dict(hybrid).to_dict())
    correct = encode_residual_condition(task, definition)
    counterfactual = encode_residual_condition(hybrid, condition_definition)
    if (correct["names"] != counterfactual["names"] or
            not np.array_equal(correct["valid"], counterfactual["valid"]) or
            not np.array_equal(correct["literal"], counterfactual["literal"])):
        raise ValueError("obstacle swap changed masks/types/capacity")
    changed = np.flatnonzero(correct["values"] != counterfactual["values"])
    if any(not correct["names"][i].startswith("obstacle.") for i in changed):
        raise ValueError("counterfactual changed non-obstacle numerical input")
    x_correct = normalizer.transform(task, definition)
    x_counterfactual = normalizer.transform(hybrid, condition_definition)
    # A second construction uses the untouched real-task encoding and replaces
    # only the canonical obstacle block.  It must equal the condition-view
    # result, including TRAIN normalization.  No reference identity enters it.
    obstacle_indices = np.asarray([i for i,n in enumerate(correct["names"]) if n.startswith("obstacle.")])
    block_replacement = correct["values"].copy()
    block_replacement[obstacle_indices] = counterfactual["values"][obstacle_indices]
    if not np.array_equal(block_replacement, counterfactual["values"]):
        raise AssertionError("obstacle-only block replacement disagrees with synthetic view")
    block_normalized = np.where(correct["valid"], (block_replacement-normalizer.mean)/normalizer.std, 0.).astype(np.float32)
    if not np.array_equal(block_normalized, x_counterfactual):
        raise AssertionError("obstacle-only normalized vector cross-check failed")
    if np.any(x_correct[np.setdiff1d(np.arange(len(x_correct)), changed)] !=
              x_counterfactual[np.setdiff1d(np.arange(len(x_correct)), changed)]):
        raise AssertionError("normalization changed a frozen non-obstacle feature")
    if object_sha(task) != original_task_sha or object_sha(definition) != original_definition_sha:
        raise AssertionError("condition construction mutated original task or definition")
    return {"raw_correct": correct["values"], "raw_swap": counterfactual["values"],
            "correct": x_correct, "swap": x_counterfactual,
            "ledger": {"actual_task_id": task["task_id"], "actual_task_sha256": original_task_sha,
                       "donor_task_id": donor["task_id"], "donor_task_sha256": object_sha(donor),
                       "synthetic_condition_task_sha256": condition_definition["task_sha256"],
                       "actual_definition_content_sha256": original_definition_sha,
                       "original_definition_unchanged": True,
                       "temporary_encoder_binding_only": "task_sha256 rebound in condition-only copy; never a provider/execution plan",
                       "swap": copy.deepcopy(swap), "swapped_obstacle_names": swapped_names,
                       "changed_feature_indices": changed.tolist(),
                       "changed_feature_names": [correct["names"][i] for i in changed],
                       "all_non_obstacle_features_bitwise_equal": True,
                       "independent_obstacle_block_replacement_exact_match": True,
                       "relative_derived_features": "none in frozen 878 schema",
                       "condition_l2_difference": float(np.linalg.norm(x_counterfactual.astype(float)-x_correct))}}


def raw_reference_offsets(definition, z_m, times):
    """Same analytic residual, also describe illegal raw amplitudes unchanged."""
    z, mask = validate_z(z_m, np.asarray(definition["interval_mask"], dtype=bool))
    times = np.asarray(times, dtype=np.float64)
    if times.ndim != 1 or not np.isfinite(times).all() or np.any(times < 0) or np.any(times > 27):
        raise ValueError("finite physical reference times within [0,27] required")
    offsets = np.zeros((len(times), 3))
    for interval, basis, coefficients, active in zip(definition["intervals_s"], definition["transverse_bases"], z, mask):
        if not active:
            continue
        lower, upper = interval
        inside = (times > lower) & (times < upper)
        u = np.where(inside, (times-lower)/(upper-lower), 0.)
        weight = 64*u**3*(1-u)**3
        offsets += np.where(inside, weight, 0.)[:, None]*(np.asarray(basis)@coefficients)
    return offsets


def _distance_summary(a, b):
    difference = np.asarray(a)-np.asarray(b)
    return {"l2_m": float(np.linalg.norm(difference)),
            "coefficient_rms_m": float(np.sqrt(np.mean(difference**2))),
            "max_absolute_coefficient_delta_m": float(np.max(np.abs(difference)))}


def self_check(old_output):
    """Numerical invariants, with no network forward, DDIM or physics call."""
    import torch
    from .task_anchored_reference import TaskAnchoredResidualPlan
    from .residual_protocol import PAIRED_SAMPLE_SEEDS

    out = Path(old_output)
    before = inspect_source(out)
    tasks = sorted([t for t in _read(out/"tasks.json")["tasks"] if t["split"] == "test"],key=lambda t:t["task_id"])
    definitions = _read(out/"definitions.json")
    ckpt = torch.load(before["checkpoint"]["path"],map_location="cpu",weights_only=False)
    normalizer = ConditionNormalizer.from_dict(ckpt["condition_normalizer"])
    checks = []
    swap = {"mode":"matching_obstacle_fields","name_prefix":"continuum_side","fields":["center_w"]}
    for i, t in enumerate(tasks):
        d = definitions[t["task_id"]]
        view = obstacle_condition_view(t,d,tasks[i^1],normalizer,swap)
        # The fixed B.2 obstacle pair differs only in its world y center.
        if view["ledger"]["changed_feature_names"] != ["obstacle.1.center.1:m"]:
            raise AssertionError("old continuum-only pair changed unexpected fields")
        z = np.zeros((6,2)); z[np.asarray(d["interval_mask"],dtype=bool)] = [.012,-.01]
        accepted = TaskAnchoredResidualPlan.from_definition(d,z)
        grid = np.unique(np.r_[np.arange(1351)*.02,np.asarray(d["intervals_s"]).ravel()])
        actual = raw_reference_offsets(d,z,grid)
        expected = accepted.offset_kinematics(grid)[0]
        error = float(np.max(np.abs(actual-expected)))
        if error != 0.:
            raise AssertionError("diagnostic offset formula differs from inherited provider")
        illegal = np.zeros((6,2)); illegal[np.flatnonzero(d["interval_mask"])[0],0] = .020001
        try:
            TaskAnchoredResidualPlan.from_definition(d,illegal)
        except ValueError:
            pass
        else:
            raise AssertionError("inherited amplitude guard unexpectedly allowed illegal raw z")
        midpoint = np.mean(d["intervals_s"][np.flatnonzero(d["interval_mask"])[0]])
        raw_peak = float(np.linalg.norm(raw_reference_offsets(d,illegal,[midpoint])[0]))
        if abs(raw_peak-.020001)>1e-15:
            raise AssertionError("illegal raw diagnostic was clipped/projected")
        for seed in PAIRED_SAMPLE_SEEDS:
            noise=np.random.default_rng(seed).standard_normal(12).astype(np.float32)
            if not np.array_equal(noise,np.random.default_rng(seed).standard_normal(12).astype(np.float32)):
                raise AssertionError("same seed differs")
        checks.append({"task_id":t["task_id"],"condition_features_changed":view["ledger"]["changed_feature_names"],
                       "maximum_provider_offset_difference_m":error,"illegal_raw_peak_preserved_m":raw_peak})
    if inspect_source(out) != before:
        raise AssertionError("old source bytes changed during invariant check")
    return {"status":"PASSED","schema":SCHEMA,"checks":checks,
            "network_forward_calls":0,"ddim_calls":0,"physics_steps":0,"optimizer_updates":0}


def run_probe(plan_path, plan_sha256, old_output, output, *, device="cpu"):
    """Only explicit root-authorized invocation after the outer plan is frozen."""
    import torch
    from .residual_diffusion import load_residual_sampler

    plan_path, old_output, output = Path(plan_path), Path(old_output), Path(output)
    if sha256(plan_path) != plan_sha256:
        raise ValueError("plan byte hash mismatch")
    plan = _read(plan_path)
    p = plan["old_checkpoint_probe"]
    if p.get("enabled") is not True or p["checkpoint_sha256"] != EXPECTED_CHECKPOINT_SHA256:
        raise ValueError("probe disabled or checkpoint changed")
    if (p["noise_pairs_per_task"], p["maximum_ddim_calls"], p["selected_update"]) != (4, 32, 250):
        raise ValueError("old checkpoint probe fixed at 4x4x2=32 DDIM calls, update250")
    pairs = p["task_pairs"]
    if len(pairs) != 2 or any(len(pair) != 2 for pair in pairs):
        raise ValueError("exactly two disjoint task pairs required")
    task_ids = [tid for pair in pairs for tid in pair]
    if len(set(task_ids)) != 4:
        raise ValueError("four unique diagnostic tasks required")
    probe = p["signed_probe"]
    if probe != {"interval_selection":"first_active", "basis_axis":0, "time_fraction":.5,
                 "world_direction_rule":"selected_task_world_transverse_basis_column_0"}:
        raise ValueError("signed readout fixed at first active interval midpoint, world basis axis0")
    threshold = float(p["observed_response_threshold_m"])
    if not np.isfinite(threshold) or threshold <= 0:
        raise ValueError("predeclared positive response threshold required")
    grid = p["reference_time_grid_s"]
    if (grid["start"], grid["stop"], grid["period"]) != (0., 27., .02):
        raise ValueError("reference diagnostic grid fixed at 20ms across [0,27]")
    # Claim all inputs and the output destination before invoking any sampler.
    identity = inspect_source(old_output)
    if output.exists():
        raise FileExistsError("probe output already exists; no implicit retry/resume")
    suite, definitions = _read(old_output/"tasks.json"), _read(old_output/"definitions.json")
    tasks = {t["task_id"]: t for t in suite["tasks"]}
    if set(task_ids) != {tid for tid, t in tasks.items() if t["split"] == "test"}:
        raise ValueError("P0 must use all four old public B.2 diagnostic tasks")
    donor = {a:b for pair in pairs for a,b in (pair, pair[::-1])}
    ckpt = torch.load(identity["checkpoint"]["path"], map_location="cpu", weights_only=False)
    normalizer = ConditionNormalizer.from_dict(ckpt["condition_normalizer"])
    views = {tid: obstacle_condition_view(tasks[tid], definitions[tid], tasks[donor[tid]], normalizer,
                                           p["obstacle_swap"]) for tid in task_ids}
    from .residual_protocol import PAIRED_SAMPLE_SEEDS
    if p["noise_seeds"] != list(PAIRED_SAMPLE_SEEDS):
        raise ValueError("P0 reuses four original B.2 paired sample seeds")
    noises = np.stack([[np.random.default_rng(seed).standard_normal(12).astype(np.float32)
                        for seed in p["noise_seeds"]] for _ in task_ids])
    for ti, tid in enumerate(task_ids):
        noises[ti,:,~np.repeat(np.asarray(definitions[tid]["interval_mask"],dtype=bool),2)] = 0.
    # DDIM is deterministic: sample_ddim has no random draw after initial_noise.
    sampler = load_residual_sampler(identity["checkpoint"]["path"], device)
    output.mkdir(parents=True)
    _write(output/"source_identity.json", identity)
    _write(output/"probe_config.json", {"schema": SCHEMA, "plan": _record(plan_path), "protocol": p,
        "task_order": task_ids, "inference_device": str(device), "sampler": "inherited deterministic DDIM20, eta=0",
        "random_stream_rule": "original B.2 np.random.default_rng(seed).standard_normal(12).astype(float32) for four fixed seeds; same values for each task; inactive dims zero per frozen mask; each true/swap pair shares exact noise",
        "sampler_random_terms_after_initial_noise": 0,
        "postprocessing": "inverse frozen TRAIN residual scaler only; no clamp/project/repair/fallback",
        "additional_physics_steps": 0, "optimizer_updates": 0})
    _write(output/"condition_wiring.json", {"tasks": [views[tid]["ledger"] for tid in task_ids]})
    times = np.arange(1351, dtype=float)*.02
    z_values = np.empty((4,4,2,6,2), dtype=float)
    normalized_values = np.empty_like(z_values)
    offsets = np.empty((4,4,2,len(times),3), dtype=float)
    rows, calls = [], 0
    journal = output/"ddim_call_ledger.jsonl"
    started = time.perf_counter()
    for ti, tid in enumerate(task_ids):
        definition = definitions[tid]
        mask = np.asarray(definition["interval_mask"], dtype=bool)
        slot = int(np.flatnonzero(mask)[0])
        direction = np.asarray(definition["transverse_bases"][slot],dtype=float)[:,0]
        midpoint = float(np.mean(definition["intervals_s"][slot]))
        for ni in range(4):
            for ci, name in enumerate(("correct", "swap")):
                calls += 1
                if calls > 32:
                    raise RuntimeError("DDIM budget exceeded")
                entry = {"call_index": calls, "task_id": tid, "noise_index": ni, "condition": name,
                         "initial_noise_sha256": hashlib.sha256(noises[ti,ni].tobytes()).hexdigest(),
                         "status": "STARTED"}
                with journal.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(entry, sort_keys=True)+"\n")
                condition = torch.as_tensor(views[tid][name][None], device=sampler.device)
                if sampler.device.type == "cuda": torch.cuda.synchronize(sampler.device)
                begin = time.perf_counter()
                with torch.no_grad():
                    normalized = sampler.model.sample_ddim(condition, torch.as_tensor(mask[None], device=sampler.device),
                        initial_noise=noises[ti,ni].reshape(1,12))
                if sampler.device.type == "cuda": torch.cuda.synchronize(sampler.device)
                elapsed = time.perf_counter()-begin
                latent = normalized.cpu().numpy().astype(np.float64).reshape(6,2)
                z = sampler.residual_normalizer.inverse(latent, mask)
                if not np.isfinite(z).all():
                    raise FloatingPointError("nonfinite raw DDIM output; no replacement")
                normalized_values[ti,ni,ci], z_values[ti,ni,ci] = latent, z
                offsets[ti,ni,ci] = raw_reference_offsets(definition, z, times)
                norms = np.linalg.norm(z, axis=1)
                signed_offset = float(raw_reference_offsets(definition, z, [midpoint])[0]@direction)
                row = {**entry, "status": "COMPLETED", "raw_z_m": z.tolist(),
                    "raw_normalized_latent": latent.tolist(), "interval_mask": mask.tolist(),
                    "per_interval_coefficient_norm_m": norms.tolist(), "maximum_coefficient_norm_m": float(norms.max()),
                    "within_original_20mm_amplitude": bool(np.all(norms <= .02)),
                    "raw_reference_offset_peak_sampled_m": float(np.linalg.norm(offsets[ti,ni,ci],axis=1).max()),
                    "raw_reference_offset_analytic_peak_m": float(norms.max()),
                    "signed_interval_slot": slot, "signed_probe_time_s": midpoint,
                    "signed_world_direction": direction.tolist(), "signed_midpoint_displacement_m": signed_offset,
                    "inference_wall_s": elapsed, "output_modified": False,
                    "actual_or_reference_precheck_executed": False}
                rows.append(row)
                with journal.open("a", encoding="utf-8") as f:
                    f.write(json.dumps({k:v for k,v in row.items() if k not in ("raw_z_m","raw_normalized_latent")}, sort_keys=True)+"\n")
    comparisons, noise_variation = [], []
    for ti, tid in enumerate(task_ids):
        for ni in range(4):
            delta_offset = offsets[ti,ni,1]-offsets[ti,ni,0]
            comparisons.append({"task_id": tid, "noise_index": ni,
                **_distance_summary(z_values[ti,ni,1], z_values[ti,ni,0]),
                "reference_world_offset_rms_m": float(np.sqrt(np.mean(np.sum(delta_offset**2,axis=1)))),
                "reference_world_offset_peak_delta_m": float(np.linalg.norm(delta_offset,axis=1).max()),
                "condition_response_observed": bool(np.linalg.norm(delta_offset,axis=1).max() > threshold),
                "signed_midpoint_displacement_delta_m": rows[ti*8+ni*2+1]["signed_midpoint_displacement_m"]-rows[ti*8+ni*2]["signed_midpoint_displacement_m"]})
        for ci, name in enumerate(("correct", "swap")):
            for na, nb in itertools.combinations(range(4), 2):
                delta_offset = offsets[ti,na,ci]-offsets[ti,nb,ci]
                noise_variation.append({"task_id": tid, "condition": name, "noise_indices": [na,nb],
                    **_distance_summary(z_values[ti,na,ci], z_values[ti,nb,ci]),
                    "reference_world_offset_rms_m": float(np.sqrt(np.mean(np.sum(delta_offset**2,axis=1)))),
                    "reference_world_offset_peak_delta_m": float(np.linalg.norm(delta_offset,axis=1).max())})
    np.savez_compressed(output/"raw_outputs.npz", initial_noise=noises,
        normalized_latents=normalized_values, raw_z_m=z_values, reference_times_s=times,
        raw_reference_world_offsets_m=offsets,
        raw_conditions=np.stack([[views[t]["raw_correct"],views[t]["raw_swap"]] for t in task_ids]),
        normalized_conditions=np.stack([[views[t]["correct"],views[t]["swap"]] for t in task_ids]))
    final_identity = inspect_source(old_output)
    if final_identity != identity or sha256(plan_path) != plan_sha256:
        raise ValueError("old source or frozen plan changed while running")
    observed = sum(c["condition_response_observed"] for c in comparisons)
    report = {"schema": SCHEMA, "status": "COMPLETED", "ddim_calls": calls,
        "budget_maximum_ddim_calls": 32, "physics_steps": 0, "actual_attempts": 0,
        "private_preview_steps": 0, "independent_physics_replay_steps": 0, "optimizer_updates": 0,
        "old_public_test_role": "DIAGNOSTIC_ONLY_NOT_NEW_INDEPENDENT_TEST",
        "checkpoint_sha256": identity["checkpoint"]["sha256"], "selected_update": 250,
        "selected_exposures": 8000, "sampler": "deterministic DDIM20 eta=0",
        "condition_response_status": "CONDITION_RESPONSE_OBSERVED" if observed else "CONDITION_RESPONSE_NOT_OBSERVED",
        "same_noise_pairs_observed": observed, "same_noise_pairs_total": 16,
        "response_threshold_m": threshold, "signed_probe": probe,
        "amplitude_legal_raw_outputs": sum(r["within_original_20mm_amplitude"] for r in rows),
        "raw_outputs_total": len(rows), "correct_adaptation_claim": "NOT_EVALUATED_NO_ROUTE_PREFERENCE_LABELS",
        "conditional_value_supported": "NOT_ESTABLISHED_BY_P0",
        "scope": "Same-noise obstacle response is a diagnostic; numerical diversity and changes do not establish useful route decisions or new TEST generalization.",
        "elapsed_s": time.perf_counter()-started, "records": rows,
        "same_noise_between_obstacle_conditions": comparisons,
        "different_noise_within_obstacle_condition": noise_variation,
        "input_sources_unchanged": True, "postprocessing": "inverse TRAIN scaler only; no clip/project/repair/retry"}
    _write(output/"report.json", report)
    _write(output/"manifest.json", {"schema": SCHEMA, "plan": _record(plan_path),
        "files": [_record(x) for x in sorted(output.iterdir()) if x.is_file()]})
    return report


def main():
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=("inspect", "self-check", "run"))
    p.add_argument("--old-output", required=True)
    p.add_argument("--output")
    p.add_argument("--plan")
    p.add_argument("--plan-sha256")
    p.add_argument("--device", default="cpu")
    a = p.parse_args()
    if a.action == "self-check":
        result = self_check(a.old_output)
        if a.output:
            if Path(a.output).exists(): p.error("self-check output already exists")
            _write(a.output,result)
        print(json.dumps(result,sort_keys=True))
    elif a.action == "inspect":
        result = inspect_source(a.old_output)
        if a.output:
            if Path(a.output).exists(): p.error("inspect output already exists")
            _write(a.output, result)
        print(json.dumps({k:v for k,v in result.items() if k not in ("sources","producer_source_bindings","obstacle_feature_names","obstacle_feature_indices_zero_based")}, sort_keys=True))
    else:
        if not all((a.output,a.plan,a.plan_sha256)): p.error("run requires --output --plan --plan-sha256")
        result = run_probe(a.plan,a.plan_sha256,a.old_output,a.output,device=a.device)
        print(json.dumps({k:v for k,v in result.items() if k not in ("records","same_noise_between_obstacle_conditions","different_noise_within_obstacle_condition")}, sort_keys=True))


if __name__ == "__main__":
    main()
