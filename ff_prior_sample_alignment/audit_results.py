"""Saved-evidence audit; no new optimizer steps or coupled/torque simulations."""
import hashlib
import zipfile
import numpy as np
import torch
from ff_prior.storage import read_json, digest
from ff_prior.diffusion import weight_hash
from ff_prior_latent.learning import LatentDenoiser
from ff_prior_delivery_guidance.audit_results import check_candidate_validation
from ff_prior_complex.tools.audit_completed import audit_physics
from ff_prior_endpoint_jacobian.state_audit import reconstruct_adamw, state_close
from ff_prior_paired_objectives.objectives import vector_gradient, get_vector
from ff_prior_endpoint_jacobian.comparison import unguided_sample
from ff_prior_endpoint_jacobian.endpoint import group_chain, attach_gradient, loss_from_ee
from ff_prior_backbone_v2.learning import loss
from .objectives import rollout, teacher_loss, prepared_fixed, combine
from .learning import load
from .inputs import identity
from .protocol import METHODS, CONTROL, P


def saved_world_gradient(model, pd, codec, arrays, rows, data):
    r = read_json(pd / "gradient.json")
    i = r["index"]
    with np.load(pd / "gradient.npz") as raw:
        g, q, perturb, J, w = group_chain(r["central"]["ee"], raw["perturbation_ee"],
                                         rows[i]["task"]["condition"]["goals"], r["epsilon"])
        for actual, key in ((g, "gradient"), (q, "central_groups"), (perturb, "perturbation_groups"),
                            (J, "group_jacobian"), (w, "analytic_smooth_weights")):
            assert np.allclose(actual, raw[key], atol=1e-12, rtol=1e-10)
        clean = rollout(model, data["c"][i:i+1], [r["seed"]])[0]
        assert np.array_equal(clean.detach().numpy(), r["raw_clean"])
        p = r["preparation"]
        prepared, z = prepared_fixed(clean, codec, data["centers"][i], arrays[1][i], rows[i]["task"],
                                     p["lower"], p["upper"], p["coefficient_shrink"])
        assert np.max(abs(z.detach().numpy()-p["prepared_Z"])) < 1e-12
        assert np.max(abs(prepared.detach().numpy()-raw["x"])) < 1e-12
        assert r["status"] == "PASS" and r["calls"]["coupled"] == 212 and r["full_sampling_steps"] == 16
        goals = rows[i]["task"]["condition"]["goals"]
        for h in r["half_checks"]:
            slopes = [loss_from_ee(s["ee"], goals)[0] for s in h["samples"]]
            slope = (slopes[1]-slopes[0])/r["epsilon"]
            err = abs(slope-g[h["dimension"]])/max(abs(slope), abs(g[h["dimension"]]), 1e-6)
            assert err <= .02 and abs(err-h["relative_error"]) < 1e-9
        vals = [loss_from_ee(s["ee"], goals)[0] for s in r["direction_samples"]]
        slope = (vals[1]-vals[0])/(2*r["epsilon"])
        expected = g@np.asarray(r["direction"])
        err = abs(slope-expected)/max(abs(slope), abs(expected), 1e-6)
        assert err <= .05 and r["convergence"]["passed"]
        return vector_gradient(attach_gradient(prepared, r["true_loss"], g), model).detach()


def audit_pool(directory, records, task):
    chosen = min((r for r in records if r["eligible"]), key=lambda r: (r["score"], r["candidate_id"]), default=None)
    selected = read_json(directory / "selection.json")
    assert selected["selection_before_execution"] and selected["task_hash"] == task["task_hash"]
    assert selected["candidate_pool_digest"] == digest(records)
    assert selected["selected_id"] == (chosen["candidate_id"] if chosen else None)
    outcome = read_json(directory / "outcome.json")
    if chosen is None:
        assert outcome["status"] == "NO_CANDIDATE"
        return 0, 0
    if (directory / "report.json").exists():
        p, steps = audit_physics(directory, task, chosen["reference_hash"])
        assert outcome["status"] == p["level"]
        return int(p["level"] == "EXECUTION_VERIFIED"), steps
    assert outcome["status"] == "ERROR"
    return 0, 0


def verify(out, ident, arrays, rows, G, data):
    errors = []
    updates = world = candidate_count = units = native = steps = app_candidates = app_units = 0
    try:
        assert identity() == ident
        with zipfile.ZipFile(out / "executed_source.zip") as z:
            assert all(hashlib.sha256(z.read(n)).hexdigest() == h for n, h in ident["source_hashes"].items())
        reports, bundles = [], {}
        for kind in METHODS:
            directory = out / "learning" / kind
            bundles[kind] = load(directory)
            report = read_json(directory / "training.json")
            assert report["updates"] == 1200 and report["TRAIN_rows_seen"] == 14222
            assert report["actual_TRAIN_draws"] == 76800 and report["VAL_gradient_updates"] == 0
            progress = torch.load(directory / "progress.pt", weights_only=False, map_location="cpu")
            assert progress["updates"] == 1200
            assert all(float(s["step"]) == 1200 for s in progress["optimizer"]["state"].values())
            assert report["segments"] == read_json(CONTROL / "learning/goal_loss/training.json")["segments"][:60]
            assert len(report["world_events"]) == (60 if kind == "aligned" else 0)
            model = LatentDenoiser(102, 65)
            codec = bundles[kind][1]
            initial = torch.load(directory / "initial.pt", weights_only=False, map_location="cpu")
            model.load_state_dict(initial["weights"])
            assert weight_hash(model) == report["initial_weight_hash"] != report["trained_weight_hash"]
            for pd in sorted((directory / "updates").iterdir()):
                before = torch.load(pd / "before.pt", weights_only=False, map_location="cpu")
                after = torch.load(pd / "after.pt", weights_only=False, map_location="cpu")
                model.load_state_dict(before["weights"])
                bv, _, _ = loss(model, data, G, before["indices"], before["t"], before["noise"])
                bg = vector_gradient(bv, model).detach()
                tg = wg = None
                if kind == "aligned":
                    tv, _ = teacher_loss(model, data, G, before["full_indices"], before["full_seeds"])
                    tg = vector_gradient(tv, model).detach()
                if (pd / "gradient.json").exists():
                    wg = saved_world_gradient(model, pd, codec, arrays, rows, data)
                    world += 1
                mixed, scales = combine(bg, tg, wg)
                with np.load(pd / "gradients.npz") as raw:
                    assert np.array_equal(mixed.numpy(), raw["mixed"])
                assert scales == read_json(pd / "update.json")["scales"]
                expected, st = reconstruct_adamw(model, mixed, before["optimizer"])
                model.load_state_dict(after["weights"])
                assert torch.allclose(expected, get_vector(model), rtol=2e-6, atol=1e-8)
                state_close(st, after["optimizer"])
                updates += 1
            reports.append(report)
        assert reports[0]["initial_weight_hash"] == reports[1]["initial_weight_hash"]
        assert reports[0]["segments"] == reports[1]["segments"]
        assert reports[0]["train_draw_counts"] == reports[1]["train_draw_counts"]
        assert world == 60
        for sj, seed in enumerate(P["val_seeds"]):
            for j, i in enumerate(data["va"]):
                for kind in METHODS:
                    r = read_json(out / "unguided" / str(sj) / kind / f"{j:02d}.json")
                    raw, sequence = unguided_sample(bundles[kind], arrays[1][i], rows[i]["task"], seed+j*997)
                    assert np.array_equal(raw, r["raw_Z"]) and sequence == r["steps"]
                    assert r["physics_guidance_calls"] == 0
                    check_candidate_validation(dict(r, Z=r["prepared_Z"]), rows[i]["task"])
        for sj, seed in enumerate(P["sampling_seeds"]):
            for ti, e in enumerate(ident["selection"]["entries"]):
                for kind in METHODS:
                    records = []
                    for k in range(5):
                        r = read_json(out / "candidates" / str(sj) / e["sample_id"] / kind / f"{k:02d}.json")
                        assert r["task_hash"] == e["task"]["task_hash"] and r["seed"] == seed+ti*101+k*997
                        assert r["model_calls"]["coupled"] <= 113
                        if r["status"] == "DONE":
                            check_candidate_validation(r, e["task"])
                            assert r["actual_network_calls"] == 16 and r["integration_intervals"] <= 15872
                        else:
                            assert not r["eligible"]
                        records.append(r)
                        candidate_count += 1
                    a, b = audit_pool(out / "executions" / str(sj) / e["sample_id"] / kind, records, e["task"])
                    native += a
                    steps += b
                    units += 1
        from .application import APP_METHODS
        for e in ident["selection"]["application"]["entries"]:
            if e["initial_status"] != "PASS":
                assert (out / "application" / e["sample_id"] / "BLOCKED.json").exists()
                continue
            for kind in APP_METHODS:
                root = out / "application" / e["sample_id"] / kind
                records = [read_json(root / "candidates" / f"{k:02d}.json") for k in range(5)]
                for k, r in enumerate(records):
                    assert r["task_hash"] == e["task"]["task_hash"] and r["seed"] == e["seed"]+k*997
                    if r["status"] == "DONE":
                        check_candidate_validation(r, e["task"])
                    if kind == "aligned_unguided":
                        assert r["physics_guidance_calls"] == 0
                    assert r["model_calls"]["coupled"] <= 113
                a, b = audit_pool(root / "execution", records, e["task"])
                app_candidates += 5
                app_units += 1
                native += a
                steps += b
        ledger = read_json(out / "ledger.json")
        assert ledger["inflight"] is None and not ledger.get("unresolved_call_accounting")
        assert sum(v for k, v in ledger["calls"].items() if k.endswith(".physics_steps")) == steps
        assert sum(v for k, v in ledger["calls"].items() if k.endswith(".coupled")) <= read_json(out / "query_budget.json")["reserved_coupled_upper"] <= P["coupled_cap"]
        assert candidate_count == 360 and units == 72
    except Exception as error:
        import traceback
        errors.append({"reason": repr(error), "traceback": traceback.format_exc()})
    return {"status": "PASS" if not errors else "FAIL", "errors": errors,
            "saved_update_reconstructions": updates, "full16_world_VJP_reconstructions": world,
            "new_DEV_candidates": candidate_count, "DEV_execution_units": units,
            "application_candidates": app_candidates, "application_execution_units": app_units,
            "native_verified": native, "native_plus_replay_steps": steps,
            "scope": "saved evidence, neural gradients and algebra; no new simulation or optimizer.step"}
