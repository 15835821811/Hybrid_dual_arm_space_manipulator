import time
import subprocess
import zipfile
from pathlib import Path
import torch
from ff_prior.storage import read_json, digest, RunLock
from ff_prior_pilot_train.storage import atomic_json, Ledger
from ff_prior.pipeline import audit, acceptance
from ff_prior_origin_learn_bounded.runtime import TaskOriginSceneCache
from ff_prior_latent.learning import Codec
from ff_prior_endpoint_jacobian.data import load as load_data
from .protocol import ROOT, CONTROL, BASE, P, METHODS
from .inputs import identity
from . import learning, comparison, application


class Budget:
    def __init__(self, out, deadline):
        self.out, self.deadline = out, deadline
        path = out / "query_budget.json"
        self.total = read_json(path)["reserved_coupled_upper"] if path.exists() else 0
        self.last_size_check = 0.

    def check(self):
        if time.time() >= self.deadline:
            raise RuntimeError("WALL_BUDGET_EXHAUSTED")
        if time.time()-self.last_size_check > 30:
            if sum(p.stat().st_size for p in self.out.rglob("*") if p.is_file()) >= P["max_output_bytes"]:
                raise RuntimeError("STORAGE_LIMIT")
            self.last_size_check = time.time()

    def reserve(self, n):
        self.check()
        if self.total+n > P["coupled_cap"]:
            raise RuntimeError("QUERY_LIMIT")
        self.total += n
        atomic_json(self.out / "query_budget.json", {"reserved_coupled_upper": self.total,
                    "limit": P["coupled_cap"], "unknown_work_not_refunded": True})


def assert_resume_safe(out):
    path = out / "ledger.json"
    if path.exists():
        l = read_json(path)
        if l["inflight"] is not None or l.get("unresolved_call_accounting"):
            raise ValueError("unknown work: do not replay")
    for p in out.glob("learning/*/journal.json"):
        if read_json(p)["status"] != "COMMITTED":
            raise ValueError("unknown training segment")


def run(out, progress, previous_elapsed=0.):
    from .audit_results import verify
    from .reporting import report
    out = Path(out)
    ident = identity()
    assert_resume_safe(out)
    deadline = time.time()+max(0., P["wall_budget_s"]-previous_elapsed)
    torch.set_num_threads(2)
    with RunLock(out):
        tests = read_json(ROOT / "ff_prior_sample_alignment/evidence/software_tests.json")
        if tests["status"] != "PASS" or tests["source_hashes"] != ident["source_hashes"]:
            raise ValueError("stale software verification")
        if (out / "run.json").exists():
            assert read_json(out / "run.json")["identity"] == ident
        else:
            atomic_json(out / "run.json", {"identity": ident, "TEST": False,
                        "producer_head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()})
            with zipfile.ZipFile(out / "executed_source.zip", "w", zipfile.ZIP_DEFLATED) as archive:
                for name in ident["source_hashes"]:
                    archive.write(ROOT / name, name)
        config = read_json(BASE / "run.json")["config"]
        physical, canonical, initial, _ = audit(config)
        assert physical["acceptance"] == read_json(CONTROL / "physical_contract.json")["acceptance"]
        profile = acceptance(config)
        atomic_json(out / "physical_contract.json", physical)
        cache = TaskOriginSceneCache(canonical, initial, profile, max_entries=16, max_scene_models=3)
        codec = Codec(read_json(CONTROL / "learning/goal_loss/codec.json"))
        progress("M0_load_existing_TRAIN")
        arrays, rows, G, data = load_data(codec)
        atomic_json(out / "training_input.json", {"TRAIN": 14222, "VAL": 48, "new_data": 0,
                    "codec_hash": digest(codec.state), "TRAIN_ids": [arrays[3][i] for i in data["tr"]],
                    "VAL_ids": [arrays[3][i] for i in data["va"]]})
        atomic_json(out / "frozen_tasks.json", ident["selection"])
        atomic_json(out / "reuse_manifest.json", {"control": str(CONTROL),
                    "methods": P["reuse_methods"], "candidates": 360, "execution_units": 72, "new_physics": False})
        ledger = Ledger(out, {"identity_hash": digest(ident)})
        budget = Budget(out, deadline)
        if not (out / "numerical_sanity.json").exists():
            learning.sanity(out, ident["selection"], codec, arrays, rows, data, cache, profile, ledger, budget, progress)
        assert read_json(out / "numerical_sanity.json")["status"] == "PASS"
        for kind in METHODS:
            learning.train(out, kind, ident["selection"], codec, arrays, rows, G, data,
                           cache, profile, ledger, budget, progress)
        a, b = [read_json(out / "learning" / k / "training.json") for k in METHODS]
        assert a["segments"] == b["segments"] and a["train_draw_counts"] == b["train_draw_counts"]
        assert a["initial_weight_hash"] == b["initial_weight_hash"]
        bundles = {k: learning.load(out / "learning" / k) for k in METHODS}
        comparison.unguided(out, bundles, arrays, rows, data, cache, profile, ledger, budget, progress)
        comparison.dev(out, bundles, ident["selection"]["entries"], cache, profile, ledger, budget, progress)
        application.run(out, bundles, ident["selection"]["application"], cache, profile, ledger, budget, progress)
        atomic_json(out / "runtime_cache.json", cache.stats())
        progress("M5_saved_evidence_audit")
        if (out / "independent_verification.json").exists():
            checked = read_json(out / "independent_verification.json")
        else:
            checked = verify(out, ident, arrays, rows, G, data)
            atomic_json(out / "independent_verification.json", checked)
        return report(out, ident["selection"]["entries"], checked)
