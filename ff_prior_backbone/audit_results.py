import hashlib
import zipfile
import numpy as np
import torch
from ff_prior.storage import read_json, digest
from ff_prior.diffusion import weight_hash
from ff_prior_delivery_guidance.audit_results import check_candidate_validation
from ff_prior_delivery_guidance.protocol import ALLOCATIONS
from ff_prior_complex.tools.audit_completed import audit_physics
from ff_prior_endpoint_jacobian.state_audit import reconstruct_adamw, state_close
from ff_prior_paired_objectives.objectives import vector_gradient, get_vector
from ff_prior_endpoint_jacobian.comparison import unguided_sample
from .protocol import P, METHODS, CONTROL
from .models import make, parameter_counts
from .inputs import identity
from .learning import load, loss


def verify(out, ident, arrays, rows, G, data):
    errors = []
    candidates = execution_units = physics_steps = native_verified = update_proofs = 0
    try:
        assert identity() == ident and read_json(out / "run.json")["identity"] == ident
        with zipfile.ZipFile(out / "executed_source.zip") as archive:
            assert all(hashlib.sha256(archive.read(n)).hexdigest() == h for n, h in ident["source_hashes"].items())
        ledger = read_json(out / "ledger.json")
        assert ledger["inflight"] is None and not ledger.get("unresolved_call_accounting")
        counts = parameter_counts()
        assert max(counts.values()) / min(counts.values()) <= 1.05
        reports = []
        bundles = {}
        for kind in METHODS:
            directory = out / "learning" / kind
            r = read_json(directory / "training.json")
            bundles[kind] = load(directory)
            assert r["updates"] == 6000 and r["TRAIN_rows_seen"] == 14222
            assert r["actual_TRAIN_draws"] == 384000 and not r["VAL_gradient_updates"]
            assert not r["VAL_checkpoint_selection"] and r["world_events"] == 0
            assert r["metadata"]["parameter_count"] == counts[kind]
            initial = torch.load(directory / "initial.pt", map_location="cpu", weights_only=False)
            model = make(kind)
            assert weight_hash(model) == r["initial_weight_hash"]
            model.load_state_dict(initial["weights"])
            assert weight_hash(model) == r["initial_weight_hash"] != r["trained_weight_hash"]
            final = torch.load(directory / "progress.pt", map_location="cpu", weights_only=False)
            assert final["updates"] == 6000
            assert all(float(v["step"]) == 6000 for v in final["optimizer"]["state"].values())
            for step in P["audit_updates"]:
                pd = directory / "updates" / f"{step:04d}"
                before = torch.load(pd / "before.pt", map_location="cpu", weights_only=False)
                after = torch.load(pd / "after.pt", map_location="cpu", weights_only=False)
                model.load_state_dict(before["weights"])
                value, _, _ = loss(model, data, G, before["indices"], before["t"], before["noise"])
                gradient = vector_gradient(value, model).detach()
                vector, state = reconstruct_adamw(model, gradient, before["optimizer"])
                model.load_state_dict(after["weights"])
                assert torch.allclose(vector, get_vector(model), rtol=2e-6, atol=1e-8)
                state_close(state, after["optimizer"])
                update_proofs += 1
            reports.append(r)
        old = read_json(CONTROL / "learning/goal_loss/training.json")
        assert reports[0]["segments"] == reports[1]["segments"] == old["segments"]
        assert reports[0]["train_draw_counts"] == reports[1]["train_draw_counts"] == old["train_draw_counts"]
        # Reconstruct every new unguided neural sample; no propagation/optimizer step.
        for sj, seed in enumerate(P["val_seeds"]):
            for j, i in enumerate(data["va"]):
                for kind in METHODS:
                    r = read_json(out / "unguided" / str(sj) / kind / f"{j:02d}.json")
                    raw, steps = unguided_sample(bundles[kind], arrays[1][i], rows[i]["task"], seed+j*997)
                    assert np.array_equal(raw, np.asarray(r["raw_Z"]))
                    assert steps == r["steps"] and r["physics_guidance_calls"] == 0
                    assert r["task_hash"] == rows[i]["task"]["task_hash"]
                    check_candidate_validation(dict(r, Z=r["prepared_Z"]), rows[i]["task"])
        entries = ident["selection"]["entries"]
        for sj, seed in enumerate(P["sampling_seeds"]):
            for ti, e in enumerate(entries):
                sid, task = e["sample_id"], e["task"]
                for kind in METHODS:
                    records = []
                    for k in range(5):
                        r = read_json(out / "candidates" / str(sj) / sid / kind / f"{k:02d}.json")
                        records.append(r)
                        candidates += 1
                        assert r["method"] == kind and r["seed"] == seed+ti*101+k*997
                        assert r["task_hash"] == task["task_hash"] and not r["private_witness_used"]
                        assert r["model_calls"]["coupled"] <= 113
                        if r["status"] != "DONE":
                            assert not r["eligible"]
                            continue
                        check_candidate_validation(r, task)
                        assert r["actual_network_calls"] == 16 and len(r["sampling"]["steps"]) == 16
                        assert r["repair_coupled_calls"] <= 110 and r["validation_coupled_calls"] <= 3
                        assert r["model_calls"]["coupled"] == r["repair_coupled_calls"]+r["validation_coupled_calls"]
                        assert r["integration_intervals"] <= 15872 and r["wall_s"] <= 15.
                        guides = r["sampling"]["guidance"]
                        assert [g["iteration"] for g in guides] == list(ALLOCATIONS)
                        assert all(g["coupled_calls"] <= ALLOCATIONS[g["iteration"]] and g["latent_feedback_max_rad"] <= 1e-12 for g in guides)
                        assert sum(g["coupled_calls"] for g in guides) == r["repair_coupled_calls"]
                        assert max(s["decomposition_residual_after"] for s in r["sampling"]["steps"]) <= 1e-10
                        assert np.max(abs(np.asarray(r["Z"])-guides[-1]["output_Z"])) <= 1e-12
                    chosen = min((r for r in records if r["eligible"]), key=lambda r: (r["score"], r["candidate_id"]), default=None)
                    directory = out / "executions" / str(sj) / sid / kind
                    selection = read_json(directory / "selection.json")
                    assert selection["selection_before_execution"] and selection["task_hash"] == task["task_hash"]
                    assert selection["candidate_pool_digest"] == digest(records)
                    assert selection["selected_id"] == (chosen["candidate_id"] if chosen else None)
                    outcome = read_json(directory / "outcome.json")
                    execution_units += 1
                    if chosen is None:
                        assert outcome["status"] == "NO_CANDIDATE"
                    elif (directory / "report.json").exists():
                        p, steps = audit_physics(directory, task, chosen["reference_hash"])
                        physics_steps += steps
                        native_verified += p["level"] == "EXECUTION_VERIFIED"
                        assert outcome["status"] == p["level"]
                    else:
                        assert outcome["status"] == "ERROR"
        assert candidates == 360 and execution_units == 72
        assert sum(v for k, v in ledger["calls"].items() if k.endswith(".physics_steps")) == physics_steps
        coupled = sum(v for k, v in ledger["calls"].items() if k.endswith(".coupled"))
        assert coupled <= read_json(out / "query_budget.json")["reserved_coupled_upper"] <= P["coupled_cap"]
    except Exception as error:
        import traceback
        errors.append({"reason": repr(error), "traceback": traceback.format_exc()})
    return {"status": "PASS" if not errors else "FAIL", "errors": errors,
            "new_candidates": candidates, "new_execution_units": execution_units,
            "native_verified": native_verified, "native_plus_replay_steps": physics_steps,
            "read_only_AdamW_reconstructions": update_proofs, "TEST": False,
            "scope": "saved evidence and neural/AdamW algebra; no new simulation or optimizer.step; old controls hash-only"}
