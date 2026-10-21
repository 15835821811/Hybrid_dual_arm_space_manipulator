import ast
import numpy as np
from ff_prior.storage import read_json, file_digest, digest
from ff_prior_pilot_train.storage import atomic_json
from .protocol import ROOT, SOURCE, BASE, PACKAGE, P, PREVIOUS

def sources():
    pending = list(PACKAGE.glob("*.py")); found = set()
    def add(module):
        parts = module.split(".")
        for i in range(1, len(parts)+1):
            p = ROOT.joinpath(*parts[:i])
            for path in (p.with_suffix(".py"), p/"__init__.py"):
                if path.is_file() and path not in found:
                    pending.append(path)
    while pending:
        p = pending.pop()
        if p in found:
            continue
        found.add(p)
        pkg = p.relative_to(ROOT).with_suffix("").parts[:-1]
        for node in ast.walk(ast.parse(p.read_text(encoding="utf-8-sig"))):
            if isinstance(node, ast.Import):
                for name in node.names: add(name.name)
            elif isinstance(node, ast.ImportFrom):
                prefix = list(pkg[:len(pkg)-node.level+1]) if node.level else []
                module = ".".join(prefix+([node.module] if node.module else []))
                add(module)
                for name in node.names:
                    if name.name != "*": add(module+"."+name.name)
    found.add(PACKAGE/"job.ps1")
    return {p.relative_to(ROOT).as_posix(): file_digest(p) for p in sorted(found)}

def freeze():
    destination = PACKAGE/"evidence/selection.json"
    if destination.exists(): raise ValueError("freeze is once-only")
    files = {}
    def get(p):
        files[p.relative_to(ROOT).as_posix()] = file_digest(p)
        return read_json(p)
    previous = get(PREVIOUS/'run.json')['identity']
    target = previous['selection']['entry']
    assert target['task']['task_hash'] == get(PREVIOUS/'frozen_task.json')['task_hash']
    seeds=[]
    for sid in P['seed_ids']:
        r=get(PREVIOUS/'teachers'/sid/'validation.json')
        assert r['task_hash']==target['task']['task_hash'] and not r['eligible']
        seeds.append({'id':sid,'Z':r['Z'],'source_reference_hash':r['reference_hash'],
                      'source_qualified_for_five':False})
    for path in (SOURCE/'physical_contract.json', SOURCE/'learning/aligned/codec.json',
                 BASE/'run.json', PREVIOUS/'independent_verification.json',
                 PREVIOUS/'teachers/00_original_failed/iterations/00/linearization.npz'):
        files[path.relative_to(ROOT).as_posix()]=file_digest(path)
    old_goal = ROOT/"DIFFUSION_GOAL_CURRENT.md"
    (PACKAGE/"evidence").mkdir(parents=True, exist_ok=True)
    (PACKAGE/"evidence/previous_goal.md").write_bytes(old_goal.read_bytes())
    selection = {"entry": target, "seeds": seeds, "task_hash": target["task"]["task_hash"],
                 "before_any_new_physics": True, "previous_final_paths_not_qualified": True}
    atomic_json(PACKAGE/"evidence/binding.json", {"files": files,
        "previous_source_hashes": dict(previous["source_hashes"], **previous["binding"]["previous_source_hashes"])})
    atomic_json(destination, selection)
    return selection

def identity():
    b = read_json(PACKAGE/"evidence/binding.json")
    for p, h in dict(b["files"], **b["previous_source_hashes"]).items():
        if file_digest(ROOT/p) != h: raise ValueError("changed frozen input: "+p)
    s = read_json(PACKAGE/"evidence/selection.json")
    assert s["task_hash"] == s["entry"]["task"]["task_hash"]
    return {"protocol": P, "source_hashes": sources(), "binding": b, "selection": s}
