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
    destination=PACKAGE/'evidence/selection.json'
    if destination.exists():raise ValueError('freeze is once-only')
    files={}
    def get(p):
        files[p.relative_to(ROOT).as_posix()]=file_digest(p)
        return read_json(p)
    old=get(PREVIOUS/'run.json')['identity'];entry=old['selection']['entry']
    assert entry['task']==get(PREVIOUS/'frozen_task.json')
    teacher=get(PREVIOUS/'teachers'/P['teacher']/'validation.json')
    execution=get(PREVIOUS/'executions'/P['teacher']/'report.json')
    assert teacher['eligible'] and execution['level']=='EXECUTION_VERIFIED'
    assert teacher['reference_hash']==execution['metadata']['reference_hash']
    assert get(ROOT/'ff_prior_capture_trust/evidence/completed/saved_audit_v2.json')['status']=='PASS'
    for folder in [PREVIOUS/'teachers'/P['teacher'],PREVIOUS/'representation/pure_codec',PREVIOUS/'executions'/P['teacher']]:
        for path in folder.iterdir():
            if path.is_file() and (path.name.startswith('grid_') or path.name in ('validation.json','prediction.npz','native_trace.npz','report.json','selection.json','outcome.json')):
                files[path.relative_to(ROOT).as_posix()]=file_digest(path)
    for path in (SOURCE/'physical_contract.json',SOURCE/'learning/aligned/codec.json',BASE/'run.json'):
        files[path.relative_to(ROOT).as_posix()]=file_digest(path)
    (PACKAGE/'evidence/previous_goal.md').write_bytes((ROOT/'DIFFUSION_GOAL_CURRENT.md').read_bytes())
    selection={'entry':entry,'task_hash':entry['task']['task_hash'],'teacher':teacher,
               'teacher_execution':execution,'ranks':P['ranks'],'modes':P['modes'],
               'no_DEV_fitting':True,'before_any_new_physics':True}
    atomic_json(PACKAGE/'evidence/binding.json',{'files':files,'previous_source_hashes':dict(old['source_hashes'],**old['binding']['previous_source_hashes'])})
    atomic_json(destination,selection)
    return selection


def identity():
    b = read_json(PACKAGE/"evidence/binding.json")
    for p, h in dict(b["files"], **b["previous_source_hashes"]).items():
        if file_digest(ROOT/p) != h: raise ValueError("changed frozen input: "+p)
    s = read_json(PACKAGE/"evidence/selection.json")
    assert s["task_hash"] == s["entry"]["task"]["task_hash"]
    return {"protocol": P, "source_hashes": sources(), "binding": b, "selection": s}
