import ast
from pathlib import Path
from ff_prior.storage import read_json, file_digest, digest
from ff_prior_pilot_train.storage import atomic_json
from .protocol import ROOT, CONTROL, P


def sources():
    pending = list((ROOT / "ff_prior_backbone").glob("*.py"))
    found = set()
    def add(module):
        parts = module.split(".")
        for i in range(1, len(parts) + 1):
            p = ROOT.joinpath(*parts[:i])
            for path in (p.with_suffix(".py"), p / "__init__.py"):
                if path.is_file() and path not in found:
                    pending.append(path)
    while pending:
        path = pending.pop()
        if path in found:
            continue
        found.add(path)
        package = path.relative_to(ROOT).with_suffix("").parts[:-1]
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8-sig"))):
            if isinstance(node, ast.Import):
                for name in node.names:
                    add(name.name)
            elif isinstance(node, ast.ImportFrom):
                prefix = list(package[:len(package)-node.level+1]) if node.level else []
                module = ".".join(prefix + ([node.module] if node.module else []))
                add(module)
                for name in node.names:
                    if name.name != "*":
                        add(module + "." + name.name)
    found.add(ROOT / "ff_prior_backbone/job.ps1")
    return {p.relative_to(ROOT).as_posix(): file_digest(p) for p in sorted(found)}


def freeze():
    from ff_prior_expand256_train.inputs import identity_for
    if identity_for()[0] != read_json(CONTROL / "run.json")["identity"]:
        raise ValueError("historical source identity changed")
    if read_json(CONTROL / "independent_verification.json")["status"] != "PASS":
        raise ValueError("historical control not audited")
    files = dict(read_json(ROOT / "ff_prior_endpoint_jacobian/evidence/binding.json")["files"])
    # Bind every consumed row and low-cost sensitivity cache; do not copy data.
    for r in read_json(CONTROL / "training_input_audit.json")["records"]:
        files[r["record"]] = r["record_sha256"]
    for b in read_json(CONTROL / "sensitivity_summary.json")["bindings"].values():
        files[b["array"]] = b["array_sha256"]
    entries = read_json(CONTROL / "frozen_tasks.json")["entries"]
    if len(entries) != 12:
        raise ValueError("frozen DEV identity changed")
    selection = {"entries": entries, "frozen_before_training": True,
                 "task_digest": digest(entries), "heldout_TEST": False,
                 "val_seeds": P["val_seeds"], "sampling_seeds": P["sampling_seeds"]}
    dest = ROOT / "ff_prior_backbone/evidence"
    if (dest / "binding.json").exists():
        raise ValueError("freeze is once-only")
    atomic_json(dest / "binding.json", {"files": files, "source": str(CONTROL),
                                     "reuse_methods": P["reuse_methods"]})
    atomic_json(dest / "selection.json", selection)
    return {"bound_files": len(files), "DEV_tasks": len(entries)}


def identity():
    binding = read_json(ROOT / "ff_prior_backbone/evidence/binding.json")
    for name, h in binding["files"].items():
        if file_digest(ROOT / name) != h:
            raise ValueError("bound input changed: " + name)
    return {"protocol": P, "source_hashes": sources(), "binding": binding,
            "selection": read_json(ROOT / "ff_prior_backbone/evidence/selection.json")}
