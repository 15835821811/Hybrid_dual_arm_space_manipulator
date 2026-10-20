import time
import subprocess
import zipfile
import numpy as np
from ff_prior.storage import read_json, digest, RunLock
from ff_prior_pilot_train.storage import atomic_json, atomic_npz, Ledger
from ff_prior.pipeline import audit, acceptance
from ff_prior.contracts import assert_frozen, assert_task_compatible
from ff_prior_origin_learn_bounded.runtime import TaskOriginSceneCache
from ff_prior_latent.learning import Codec
from ff_prior.guidance import public_goal_initializer
from ff_prior.spline import Reference
from ff_prior.proposal_generator import quick_gates
from ff_prior.physical_audit import audit_execution
from ff_prior_delivery_guidance.sampling import LatentProjector
from .protocol import ROOT, SOURCE, BASE, PACKAGE, OUT, P
from .inputs import identity
from .solver import Evaluator, optimize, basis, solve_step, metric, accept
from .validation import validate

class Budget:
    def __init__(self, out, robot, deadline):
        self.out, self.robot, self.deadline = out, robot, deadline
        self.used = read_json(out/"ledger.json").get("calls", {}) if (out/"ledger.json").exists() else {}
        self.previous = sum(v for k, v in self.used.items() if k.endswith(".coupled"))
        self.start = robot.calls["coupled"]
        self.last_size_check = 0
    def check(self, upcoming=0):
        if time.time() >= self.deadline:
            raise RuntimeError("WALL_BUDGET_EXHAUSTED")
        if self.previous + self.robot.calls["coupled"]-self.start + upcoming > P["coupled_cap"]:
            raise RuntimeError("COUPLED_BUDGET_EXHAUSTED")
        if time.time()-self.last_size_check > 30:
            if sum(p.stat().st_size for p in self.out.rglob("*") if p.is_file()) > P["output_bytes_cap"]:
                raise RuntimeError("OUTPUT_BUDGET_EXHAUSTED")
            self.last_size_check = time.time()

def assert_resume_safe(out):
    if (out/"ledger.json").exists():
        r = read_json(out/"ledger.json")
        if r["inflight"] is not None or r.get("unresolved_call_accounting"):
            raise ValueError("UNKNOWN_WORK_NOT_REPLAYABLE")

def local_diagnosis(directory, source_directory, codec, ev):
    pd = source_directory/"iterations/00"
    if not (pd/"linearization.npz").exists():
        atomic_json(directory/"result.json", {"status": "BLOCKED_NO_LINEARIZATION"})
        return
    row = read_json(pd/"record.json")
    with np.load(pd/"linearization.npz") as saved:
        z, e, d, fulljg, fulljd = [saved[k].copy() for k in ("Z", "goal", "active_distances", "jg", "jd")]
    result = {}
    for kind in ("legacy24", "codec102", "full323"):
        directions = basis(kind, codec)
        transform = directions.reshape(len(directions), -1).T
        jg, jd = fulljg@transform, fulljd@transform
        ev.robot.calls["sqp"] += 1
        step, solver = solve_step(z, directions, e, d, jg, jd, ev.robot.lower, ev.robot.upper)
        delta = np.einsum("i,ijk->jk", step, directions)
        trials = []
        for scale in P["line_scales"]:
            trial = z+scale*delta
            c = ev.task["condition"]
            gates = quick_gates(ev.robot, Reference(trial, c["initial_features"], c["duration"]))
            if not all(gates.values()):
                trials.append({"scale": scale, "status": "QUICK_REJECTED", "gates": gates})
                continue
            value = ev.evaluate(trial)
            trials.append(dict(metric(value), scale=scale, status="COARSE_DIAGNOSTIC_NOT_MODEL_VALID"))
        singular = np.linalg.svd(jg, compute_uv=False)
        result[kind] = {"solver": solver, "trials": trials, "goal_singular_values": singular,
                        "direction_rank": int(np.linalg.matrix_rank(transform))}
        atomic_npz(directory/f"{kind}.npz", jg=jg, jd=jd, step=step, direction_map=transform)
    atomic_json(directory/"result.json", {"status": "DONE", "variants": result, "source": str(pd),
        "same_linearization_and_solver": True, "rank_alone_is_not_global_feasibility": True})

def run(out=OUT, progress=lambda *a, **k: None, previous_elapsed=0.):
    out.mkdir(parents=True, exist_ok=True)
    assert_resume_safe(out)
    ident = identity()
    from .audit_results import verify, report
    with RunLock(out):
        assert read_json(PACKAGE/"evidence/software_tests.json")["source_hashes"] == ident["source_hashes"]
        if (out/"run.json").exists():
            assert read_json(out/"run.json")["identity"] == ident
        else:
            atomic_json(out/"run.json", {"identity": ident, "producer_head": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(), "TEST": False})
            with zipfile.ZipFile(out/"executed_source.zip", "w", zipfile.ZIP_DEFLATED) as archive:
                for p in ident["source_hashes"]: archive.write(ROOT/p, p)
        config = read_json(BASE/"run.json")["config"]
        physical, canonical, initial, _ = audit(config)
        old = read_json(SOURCE/"physical_contract.json")
        assert physical["acceptance"] == old["acceptance"]
        assert digest(physical["identity"]) == digest(old["identity"]) and physical["initial_state_id"] == old["initial_state_id"]
        profile = acceptance(config)
        entry = ident["selection"]["entry"]; task = entry["task"]
        cache = TaskOriginSceneCache(canonical, initial, profile, max_entries=1, max_scene_models=1)
        rr, ii, cc, proof = cache.get(entry)
        assert_task_compatible(task, rr, ii, cc, profile); assert_frozen(task)
        atomic_json(out/"frozen_task.json", task)
        atomic_json(out/"physical_contract.json", physical)
        atomic_json(out/"scene_initial_audit.json", proof)
        codec = Codec(read_json(SOURCE/"learning/aligned/codec.json"))
        base = public_goal_initializer(task, rr, ii)
        deadline = time.time()+max(0, P["wall_s"]-previous_elapsed)
        ledger = Ledger(out, {"identity_hash": digest(ident)})
        budget = Budget(out, rr, deadline)
        ev = Evaluator(task, rr, ii, cc, budget)

        def unit(key, stage, action):
            if key in ledger.data["completed"]:
                return
            ledger.begin(key, stage, P["wall_s"], 64)
            before = rr.calls.copy()
            budget.check()
            action()
            budget.check()
            ledger.finish({"status": "DONE"}, {k: rr.calls[k]-before[k] for k in before})

        for seed in ident["selection"]["seeds"]:
            directory = out/"teachers"/seed["id"]
            def teacher(seed=seed, directory=directory):
                start = np.clip(np.asarray(seed["Z"]), rr.lower, rr.upper)
                atomic_json(directory/"seed.json", {"frozen_seed": seed, "actual_start_Z": start,
                    "clipped_to_original_domain": True, "task_hash": task["task_hash"]})
                result = optimize(directory, start, "full323", codec, ev, progress, P["teacher_iterations"])
                validate(directory, np.asarray(result["Z"]), task, rr, ii, cc, profile, budget)
            unit("teacher_"+seed["id"], "teacher_search_and_validation", teacher)

        validations = [(s["id"], read_json(out/"teachers"/s["id"]/"validation.json"))
                       for s in ident["selection"]["seeds"]]
        eligible = sorted(((sid, r) for sid, r in validations if r["eligible"]),
                          key=lambda item: (item[1]["score"], item[0]))
        selected = [sid for sid, r in eligible[:P["teacher_execution_limit"]]]
        selection = {"ids": selected, "task_hash": task["task_hash"], "before_execution": True,
                     "candidate_digest": digest(validations), "policy": "lowest_pose_score_then_seed_id; no replacement"}
        if (out/"teacher_execution_selection.json").exists():
            assert read_json(out/"teacher_execution_selection.json") == selection
        else:
            atomic_json(out/"teacher_execution_selection.json", selection)

        def execute_record(key, directory, candidate):
            def action():
                atomic_json(directory/"selection.json", {"reference_hash": candidate["reference_hash"],
                    "task_hash": task["task_hash"], "before_execution": True, "eligible": candidate["eligible"]})
                if not candidate["eligible"]:
                    atomic_json(directory/"outcome.json", {"status": "NO_CANDIDATE"})
                    return
                progress("native_torque_and_replay", unit=key)
                r = audit_execution(task, profile, rr, ii,
                    Reference(candidate["Z"], task["condition"]["initial_features"], task["condition"]["duration"]),
                    directory, budget.deadline)
                r["goal_source"] = "frozen_user_static_application_world_goal"
                atomic_json(directory/"report.json", r)
                atomic_json(directory/"outcome.json", {"status": r["level"]})
            unit(key, "selected_execution", action)
        for sid in selected:
            execute_record("execute_teacher_"+sid, out/"executions"/sid,
                           read_json(out/"teachers"/sid/"validation.json"))
        if eligible:
            sid, original = eligible[0]  # Pre-execution order; never select using execution outcomes.
            z = np.asarray(original["Z"])
            projector = LatentProjector(task, rr, ii, cc, profile, deadline, codec=codec, base=base)
            projected = codec.decode(codec.encode(z, base), base)
            prepared, preparation = projector.prepare(z)
            atomic_json(out/"representation_source.json", {"teacher_id": sid, "before_execution_rank": 0,
                "projection_max_rad": float(abs(projected-z).max()),
                "prepare_max_rad": float(abs(prepared-z).max()), "preparation": preparation})
            for name, value in (("pure_codec", projected), ("deployed_prepare", prepared)):
                directory = out/"representation"/name
                unit("validate_"+name, "representation", lambda d=directory, v=value:
                     validate(d, v, task, rr, ii, cc, profile, budget))
                execute_record("execute_"+name, directory/"execution", read_json(directory/"validation.json"))
            directory = out/"representation/codec_repair"
            def codec_repair():
                result = optimize(directory, prepared, "codec102", codec, ev, progress, P["codec_repair_iterations"])
                validate(directory, result["Z"], task, rr, ii, cc, profile, budget)
            unit("codec_repair", "representation", codec_repair)
            execute_record("execute_codec_repair", directory/"execution", read_json(directory/"validation.json"))
        else:
            atomic_json(out/"representation_BLOCKED.json", {"status": "BLOCKED_NO_MODEL_VALID_WITNESS",
                "no_conclusion_of_representation_impossibility": True})
        first = ident["selection"]["seeds"][0]
        unit("local_subspaces", "local_diagnosis", lambda: local_diagnosis(
            out/"local_subspaces", out/"teachers"/first["id"], codec, ev))
        def legacy():
            budget.check(110)
            z = np.asarray(first["Z"])
            projector = LatentProjector(task, rr, ii, cc, profile, deadline, codec=codec, base=base)
            records = []
            for it in P["legacy_repair_stages"]:
                z, r = projector(z, it)
                records.append(r)
            atomic_json(out/"legacy_local/repair.json", {"stages": records, "Z": z,
                "task_hash": task["task_hash"], "standalone_repair_not_DDIM": True,
                "used": projector.used})
            validate(out/"legacy_local", z, task, rr, ii, cc, profile, budget)
        unit("legacy_local", "local_diagnosis", legacy)
        assert_frozen(task)
        atomic_json(out/"runtime_cache.json", cache.stats())
        progress("saved_evidence_audit")
        checked = verify(out, ident)
        atomic_json(out/"independent_verification.json", checked)
        return report(out, checked)
