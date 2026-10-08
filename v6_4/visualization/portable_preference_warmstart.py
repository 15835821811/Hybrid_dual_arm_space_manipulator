"""Relocated C.2 model loading with exact-byte guards and unchanged sampler math.

API: load_sampler(release, device="cpu") -> original PreferenceSampler instance.
CLI (no inference): python -m v6_4.visualization.portable_preference_warmstart
    --release v6_4/releases/preference_warmstart_20261008_01 --action inspect
CLI (two explicit samples): append --action sample --task-json <frozen Task JSON>.

Requires a repository clone and its Python dependencies, but never accesses an
original E: path. Original checkpoint/freeze dictionaries remain unchanged.
Raw native execution archives are deliberately not required for model loading.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

from .publish_preference_warmstart import ROOT, read, sha


def _canonical_sha(value):
    data = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(data.encode("utf8")).hexdigest()


def _normalized(path):
    return str(path).replace("\\", "/").rstrip("/")


class PortableResolver:
    """Resolve original identity strings solely through the release mapping."""
    def __init__(self, release):
        self.release = Path(release).resolve()
        final = self.release / "release_manifest.json"
        self.manifest_path = final if final.is_file() else self.release / "release_manifest_train.json"
        self.manifest = read(self.manifest_path)
        if self.manifest.get("schema") != "v64_c2_portable_release_v1":
            raise ValueError("not a C.2 portable release")
        mapping = self.release / ("portable_paths.json" if final.is_file() else "portable_paths_train.json")
        self.mapping_path = mapping
        self.mapping = {}
        for original, row in read(mapping).items():
            key = _normalized(original)
            if key in self.mapping and self.mapping[key] != row:
                raise ValueError("ambiguous portable path identity")
            self.mapping[key] = row
        self.checked = {}
        self.source_manifest = read(self.release / "frozen_source_manifest.json")
        if self.source_manifest["algorithm_producer_commit"] != self.manifest["algorithm_producer_commit"]:
            raise ValueError("portable source/model producer differs")

    def _verified_path(self, relative, expected, size=None):
        path = (self.release / relative).resolve()
        if not path.is_relative_to(self.release):
            raise ValueError("portable path escapes the release")
        if size is not None and path.stat().st_size != size:
            raise ValueError("portable artifact size differs: " + relative)
        if path not in self.checked:
            self.checked[path] = sha(path)
        if self.checked[path] != expected:
            raise ValueError("portable artifact SHA differs: " + relative)
        return path

    def resolve(self, original, expected=None):
        row = self.mapping.get(_normalized(original))
        if row is None or row.get("available") is not True or not row.get("path"):
            raise ValueError("required original identity has no portable bytes: " + str(original))
        if expected is not None and row["sha256"] != expected:
            raise ValueError("portable mapping differs from original frozen SHA")
        return self._verified_path(row["path"], row["sha256"], row.get("size"))

    def snapshot(self, relative, expected=None):
        matches = [r for r in self.manifest["inventory"] if r["path"] == relative]
        if len(matches) != 1 or not matches[0]["copied"]:
            raise ValueError("required scientific artifact omitted: " + relative)
        row = matches[0]
        if expected is not None and row["sha256"] != expected:
            raise ValueError("inventory differs from checkpoint SHA: " + relative)
        path = self.resolve(row["source_path"], row["sha256"])
        if path != (self.release / "snapshot" / relative).resolve():
            raise ValueError("mapping points to a different snapshot artifact")
        return path

    def verify_scientific_inputs(self):
        for row in self.manifest["inventory"]:
            relative = row["path"]
            if row["copied"] and (relative.split("/")[0] in ("model", "dataset", "frozen_tasks")
                or relative in ("model_freeze.json", "learning_split_manifest.json", "source_identity.json")):
                self.snapshot(relative)
        for relative, expected in self.source_manifest["source_sha256"].items():
            self._verified_path("frozen_source/" + relative, expected)
            current = ROOT / relative
            if not current.resolve().is_relative_to(ROOT) or sha(current) != expected:
                raise ValueError("loaded repository producer source differs: " + relative)


def load_dataset(release):
    """Load and verify portable TRAIN/VAL labels/scalers without original paths."""
    resolver = PortableResolver(release)
    resolver.verify_scientific_inputs()
    from ..preference_teacher_dataset import load_preference_dataset
    dataset = load_preference_dataset(resolver.snapshot("dataset/dataset_manifest.json"))
    return dataset


def load_sampler(release, device="cpu"):
    """Construct the unchanged sampler using verified relocated bytes; no forward."""
    started = time.perf_counter()
    resolver = PortableResolver(release)
    resolver.verify_scientific_inputs()
    frozen = read(resolver.snapshot("model_freeze.json"))
    freeze_body = {k: v for k, v in frozen.items() if k != "freeze_sha256"}
    if _canonical_sha(freeze_body) != frozen["freeze_sha256"]:
        raise ValueError("original model freeze content identity differs")
    if _canonical_sha(frozen["retrieval_identity"]) != frozen["retrieval_identity_sha256"]:
        raise ValueError("frozen retrieval rule identity differs")
    for original, expected in frozen["artifacts"].items():
        resolver.resolve(original, expected)
    checkpoint = resolver.resolve(frozen["selected_checkpoint"])
    if checkpoint != resolver.snapshot("model/model/selected.pt"):
        raise ValueError("copied checkpoint is not the original pre-TEST selected model")
    import torch
    from ..dataset import object_sha
    from ..preference_diffusion_warmstart import CHECKPOINT_SCHEMA, PreferenceSampler
    from ..preference_teacher_dataset import ConditionNormalizer, load_preference_dataset
    from ..residual_dataset import ResidualNormalizer
    from ..residual_diffusion import ResidualDDPM, ResidualDiffusionConfig, _configure, _state_sha
    c = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if c.get("schema") != CHECKPOINT_SCHEMA:
        raise ValueError("copied model is not the trained C.2 checkpoint")
    for original, expected in c["training_config"]["source_files"].items():
        path = resolver.resolve(original, expected)
        relative = path.relative_to(resolver.release / "frozen_source").as_posix()
        if resolver.source_manifest["source_sha256"].get(relative) != expected:
            raise ValueError("checkpoint source is not a frozen producer input")
    config_path = resolver.snapshot("model/training_config.json", c["training_config_sha256"])
    if read(config_path) != c["training_config"]:
        raise ValueError("checkpoint training protocol differs from sealed file")
    dataset_manifest = resolver.resolve(c["training_config"]["dataset_manifest_path"],
                                        c["training_config"]["dataset_manifest_sha256"])
    if dataset_manifest != resolver.snapshot("dataset/dataset_manifest.json"):
        raise ValueError("checkpoint refers to a different portable dataset")
    dataset = load_preference_dataset(dataset_manifest)
    for name, key in (("condition_normalizer.json", "condition_normalizer"),
                      ("residual_normalizer.json", "residual_normalizer"),
                      ("condition_schema.json", "condition_schema")):
        if read(resolver.snapshot("model/" + name)) != c[key]:
            raise ValueError("checkpoint scaler/schema differs from frozen artifact: " + name)
    if dataset.condition_scaler.to_dict() != c["condition_normalizer"] or dataset.residual_scaler.to_dict() != c["residual_normalizer"]:
        raise ValueError("model scalers differ from the portable TRAIN label dataset")
    training = read(resolver.snapshot("model/training_summary.json"))
    if training["selected_checkpoint_update"] != c["update"] or training["optimizer_updates_total"] != 4000:
        raise ValueError("selected update or total training updates differ")
    if frozen["supported_conditions"] != c["supported_conditions"]:
        raise ValueError("portable model training support differs from pre-TEST freeze")
    # Populate precisely the attributes set by the original constructor. Its
    # numerical sample/initializer methods are reused without modification.
    _configure()
    sampler = PreferenceSampler.__new__(PreferenceSampler)
    sampler.checkpoint_path = checkpoint
    sampler.checkpoint_sha256 = sha(checkpoint)
    sampler.device = torch.device(device)
    sampler.model = ResidualDDPM(ResidualDiffusionConfig.from_dict(c["model_config"])).to(sampler.device)
    sampler.model.load_state_dict(c["state_dict"], strict=True)
    sampler.model.eval()
    if _state_sha(sampler.model) != c["state_sha256"]:
        raise ValueError("relocated model tensor identity differs")
    sampler.checkpoint = c
    sampler.condition_normalizer = ConditionNormalizer.from_dict(c["condition_normalizer"])
    sampler.residual_normalizer = ResidualNormalizer.from_dict(c["residual_normalizer"])
    sampler.identity = {"checkpoint_sha256": sampler.checkpoint_sha256,
        "training_config_sha256": c["training_config_sha256"],
        "condition_scaler_sha256": object_sha(c["condition_normalizer"]),
        "residual_scaler_sha256": object_sha(c["residual_normalizer"]),
        "condition_schema_sha256": object_sha(c["condition_schema"]),
        "selected_checkpoint_update": c["update"], "supported_conditions": c["supported_conditions"]}
    sampler.model_load_wall_s = time.perf_counter() - started
    sampler.sample_units = 0
    sampler.portable_verification = {"release": str(resolver.release),
        "stage": resolver.manifest["stage"], "algorithm_producer_commit": resolver.manifest["algorithm_producer_commit"],
        "verified_portable_artifacts": len(resolver.checked),
        "checkpoint_sha256": sampler.checkpoint_sha256,
        "original_paths_accessed": False, "checkpoint_contents_modified": False,
        "model_forward_calls": 0, "ddim_sample_units": 0, "optimizer_updates": 0, "physics_steps": 0}
    return sampler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--action", choices=("inspect", "sample"), default="inspect")
    parser.add_argument("--task-json", type=Path)
    args = parser.parse_args()
    sampler = load_sampler(args.release, args.device)
    if args.action == "inspect":
        result = sampler.portable_verification
    else:
        if not args.task_json:
            parser.error("--task-json required for two explicit DDIM samples")
        from ..route_initializers import json_raw
        from ..task_protocol import TaskSpec
        task = TaskSpec.from_dict(read(args.task_json))
        result = {"proposals": json_raw(sampler.initializer_proposals(task, noise_seed=64224)),
                  "ddim_sample_units": sampler.sample_units, "physics_steps": 0}
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
