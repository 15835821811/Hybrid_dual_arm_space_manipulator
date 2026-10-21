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
from .solver import Evaluator, optimize
from .propagation import predict_phases
from ff_prior.coupled_predictor import predict_coupled
from .validation import validate

class Budget:
    def __init__(self, out, robot, deadline):
        self.out, self.robot, self.deadline = out, robot, deadline
        self.used = read_json(out/"ledger.json").get("calls", {}) if (out/"ledger.json").exists() else {}
        self.previous = sum(v for k, v in self.used.items() if k.endswith(".coupled"))
        self.start = robot.calls["coupled"]
        self.last_size_check = 0
        self.integrations = read_json(out/"budget.json")["integration_intervals"] if (out/"budget.json").exists() else 0
    def check(self, upcoming=0):
        if time.time() >= self.deadline:
            raise RuntimeError("WALL_BUDGET_EXHAUSTED")
        if self.previous + self.robot.calls["coupled"]-self.start + upcoming > P["coupled_cap"]:
            raise RuntimeError("COUPLED_BUDGET_EXHAUSTED")
        if time.time()-self.last_size_check > 30:
            if sum(p.stat().st_size for p in self.out.rglob("*") if p.is_file()) > P["output_bytes_cap"]:
                raise RuntimeError("OUTPUT_BUDGET_EXHAUSTED")
            self.last_size_check = time.time()

    def consume(self, intervals):
        self.check(1)
        if self.integrations+intervals>P['integration_cap']:raise RuntimeError('INTEGRATION_BUDGET_EXHAUSTED')
        self.integrations+=intervals
    def can_fit(self, queries, intervals):
        return (self.previous+self.robot.calls['coupled']-self.start+queries<=P['coupled_cap']
            and self.integrations+intervals<=P['integration_cap'] and time.time()+60<self.deadline)
    def save(self):
        atomic_json(self.out/'budget.json',{'integration_intervals':self.integrations,
            'actual_coupled':self.previous+self.robot.calls['coupled']-self.start,'limits':P})

def assert_resume_safe(out):
    if (out/"ledger.json").exists():
        r = read_json(out/"ledger.json")
        if r["inflight"] is not None or r.get("unresolved_call_accounting"):
            raise ValueError("UNKNOWN_WORK_NOT_REPLAYABLE")

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
            budget.save()

        atomic_json(out/'physical_limits.json',{'lower':rr.lower,'upper':rr.upper,'geom_rbound':rr.model.geom_rbound})
        def sanity():
            z=np.asarray(ident['selection']['seeds'][0]['Z'])
            ref=Reference(z,task['condition']['initial_features'],task['condition']['duration'])
            budget.consume(512); original=predict_coupled(ii,ref,rr,512,budget.deadline)
            budget.consume(512); current=predict_phases(ii,ref,rr,np.linspace(0,1,513),budget.deadline)
            errors={k:float(abs(original[k]-current[k]).max()) for k in ('qpos','ee','geom','geom_rotation','velocity')}
            atomic_npz(out/'sanity_uniform.npz',old_qpos=original['qpos'],new_qpos=current['qpos'],old_ee=original['ee'],new_ee=current['ee'])
            atomic_json(out/'sanity_uniform.json',{'status':'PASS' if max(errors.values())<1e-12 else 'FAIL','errors':errors})
            assert max(errors.values())<1e-12,'nonuniform integrator uniform equivalence failed'
        unit('sanity_uniform','implementation_check',sanity)

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
                if not all(quick_gates(rr, Reference(prepared, task['condition']['initial_features'], task['condition']['duration'])).values()):
                    atomic_json(directory/'result.json', {'status':'BLOCKED_PREPARE_QUICK_GATE','Z':prepared})
                    validate(directory, prepared, task, rr, ii, cc, profile, budget)
                    return
                result = optimize(directory, prepared, "codec102", codec, ev, progress, P["codec_repair_iterations"])
                validate(directory, result["Z"], task, rr, ii, cc, profile, budget)
            unit("codec_repair", "representation", codec_repair)
            execute_record("execute_codec_repair", directory/"execution", read_json(directory/"validation.json"))
        else:
            atomic_json(out/"representation_BLOCKED.json", {"status": "BLOCKED_NO_MODEL_VALID_WITNESS",
                "no_conclusion_of_representation_impossibility": True})
        assert_frozen(task)
        atomic_json(out/"runtime_cache.json", cache.stats())
        progress("saved_evidence_audit")
        checked = verify(out, ident)
        atomic_json(out/"independent_verification.json", checked)
        return report(out, checked)
