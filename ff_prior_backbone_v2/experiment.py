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
from . import learning, comparison


class Budget:
    def __init__(self, out, deadline):
        self.out, self.deadline = out, deadline
        p = out / "query_budget.json"
        self.total = read_json(p)["reserved_coupled_upper"] if p.exists() else 0
        self.last_size_check = 0.

    def check(self):
        if time.time() >= self.deadline:
            raise RuntimeError("WALL_BUDGET_EXHAUSTED")
        if time.time() - self.last_size_check > 30:
            if sum(p.stat().st_size for p in self.out.rglob("*") if p.is_file()) >= P["max_output_bytes"]:
                raise RuntimeError("STORAGE_LIMIT")
            self.last_size_check = time.time()

    def reserve(self, n):
        self.check()
        if self.total + n > P["coupled_cap"]:
            raise RuntimeError("QUERY_LIMIT")
        self.total += n
        atomic_json(self.out / "query_budget.json", {"reserved_coupled_upper": self.total,
                    "limit": P["coupled_cap"], "unknown_work_not_refunded": True})


def assert_resume_safe(out):
    p = out / "ledger.json"
    if p.exists():
        ledger = read_json(p)
        if ledger["inflight"] is not None or ledger.get("unresolved_call_accounting"):
            raise ValueError("unknown physics work: do not replay")
    for path in out.glob("learning/*/journal.json"):
        if read_json(path)["status"] != "COMMITTED":
            raise ValueError("unknown training work: do not replay")


def run(out, progress, previous_elapsed=0.):
    from .audit_results import verify
    from .reporting import report
    out = Path(out)
    ident = identity()
    assert_resume_safe(out)
    deadline = time.time() + max(0., P["wall_budget_s"]-previous_elapsed)
    torch.set_num_threads(2)
    with RunLock(out):
        proof = read_json(ROOT / "ff_prior_backbone_v2/evidence/software_tests.json")
        if proof["status"] != "PASS" or proof["source_hashes"] != ident["source_hashes"]:
            raise ValueError("software proof stale")
        if (out / "run.json").exists():
            if read_json(out / "run.json")["identity"] != ident:
                raise ValueError("run identity changed")
        else:
            atomic_json(out / "run.json", {"identity": ident, "TEST": False,
                "producer_head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()})
            with zipfile.ZipFile(out / "executed_source.zip", "w", zipfile.ZIP_DEFLATED) as archive:
                for name in ident["source_hashes"]:
                    archive.write(ROOT / name, name)
        config = read_json(BASE / "run.json")["config"]
        physical, canonical, initial, _ = audit(config)
        if physical["acceptance"] != read_json(CONTROL / "physical_contract.json")["acceptance"]:
            raise ValueError("physical contract changed")
        profile = acceptance(config)
        atomic_json(out / "physical_contract.json", physical)
        cache = TaskOriginSceneCache(canonical, initial, profile, max_entries=16, max_scene_models=4)
        codec = Codec(read_json(CONTROL / "learning/goal_loss/codec.json"))
        progress("M0_load_exact_TRAIN")
        arrays, rows, G, data = load_data(codec)
        atomic_json(out / "training_input.json", {"TRAIN": 14222, "VAL": 48, "new_data": 0,
                    "data_hash": read_json(CONTROL / "learning/goal_loss/training.json")["metadata"]["data_hash"],
                    "codec_hash": digest(codec.state), "TRAIN_ids": [arrays[3][i] for i in data["tr"]],
                    "VAL_ids": [arrays[3][i] for i in data["va"]]})
        atomic_json(out / "frozen_tasks.json", ident["selection"])
        atomic_json(out / "reuse_manifest.json", {"control": str(CONTROL), "methods": P["reuse_methods"],
                    "candidates": 360, "execution_units": 72, "new_physics": False})
        if not (out / "numerical_sanity.json").exists():
            if (out / "sanity_started.json").exists():
                raise ValueError("interrupted numerical sanity is unknown work; not replayed")
            atomic_json(out / "sanity_started.json", {"started": time.time(), "temporary_steps_per_model": 2})
            progress("M0_TRAIN_numerical_sanity")
            learning.sanity(out, data, G)
        if read_json(out / "numerical_sanity.json")["status"] != "PASS":
            raise ValueError("numerical sanity incomplete")
        for kind in METHODS:
            learning.train(out, kind, codec, arrays, G, data, deadline, progress)
        bundles = {k: learning.load(out / "learning" / k) for k in METHODS}
        a, b = [read_json(out / "learning" / k / "training.json") for k in METHODS]
        if a["segments"] != b["segments"] or a["train_draw_counts"] != b["train_draw_counts"]:
            raise ValueError("paired randomness/consumption differs")
        ledger = Ledger(out, {"identity_hash": digest(ident)})
        budget = Budget(out, deadline)
        comparison.unguided(out, bundles, arrays, rows, data, cache, profile, ledger, budget, progress)
        comparison.dev(out, bundles, ident["selection"]["entries"], cache, profile, ledger, budget, progress)
        atomic_json(out / "runtime_cache.json", cache.stats())
        progress("M4_read_only_raw_audit")
        if (out / "independent_verification.json").exists():
            checked = read_json(out / "independent_verification.json")
        else:
            checked = verify(out, ident, arrays, rows, G, data)
            atomic_json(out / "independent_verification.json", checked)
        return report(out, ident["selection"]["entries"], checked)
