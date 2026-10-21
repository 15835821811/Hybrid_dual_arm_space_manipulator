import ast
import numpy as np
from ff_prior.storage import read_json, file_digest, digest
from ff_prior_pilot_train.storage import atomic_json
from .protocol import ROOT, SOURCE, BASE, PACKAGE, P, DATA

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
    from .protocol import REUSE
    old_package=ROOT/'ff_prior_control_transfer'
    if (PACKAGE/'evidence/selection.json').exists():raise ValueError('once-only freeze')
    b=read_json(old_package/'evidence/binding.json');files=dict(b['files'])
    for p,h in files.items():assert file_digest(ROOT/p)==h
    assert read_json(REUSE/'status.json')['status']=='FINISHED_WITH_BLOCKED_ITEMS'
    assert read_json(REUSE/'independent_verification.json')['status']=='PASS'
    assert read_json(REUSE/'reconstruction_gate.json')['status']=='PASS'
    assert read_json(REUSE/'TRAIN_diagnostic/migration.json')['status']=='PASS'
    # Explicit reuse of completed evidence, not re-execution or copying validation grades.
    for p in REUSE.rglob('*'):
        if p.is_file():files[p.relative_to(ROOT).as_posix()]=file_digest(p)
    for p in (old_package/'evidence/selection.json',old_package/'docs/ANALYSIS.md',
              old_package/'docs/CLAIM_VERDICT.md',ROOT/'paper/review-traces/experiment-result-to-claim/20261021_control_transfer_run01/verdict.json'):
        files[p.relative_to(ROOT).as_posix()]=file_digest(p)
    s=read_json(old_package/'evidence/selection.json')
    s.update(reused_evidence_root=REUSE.relative_to(ROOT).as_posix(),
             new_reconstruction_executions=0,
             original_protected=[j for j in range(16) if read_json(REUSE/'TRAIN_diagnostic/before'/f'{j:02d}'/'validation.json')['eligible']])
    assert s['original_protected']==[12]
    atomic_json(PACKAGE/'evidence/selection.json',s)
    atomic_json(PACKAGE/'evidence/binding.json',{'files':files})
    (PACKAGE/'evidence/previous_goal.md').write_bytes((ROOT/'DIFFUSION_GOAL_CURRENT.md').read_bytes())
    return {'files':len(files),'TRAIN':14222,'reused_reconstruction':6,'reused_executions':3,'DEV':12}

def identity():
    b=read_json(PACKAGE/'evidence/binding.json')
    for p,h in b['files'].items():
        if file_digest(ROOT/p)!=h:raise ValueError('frozen input changed: '+p)
    return {'protocol':P,'binding':b,'selection':read_json(PACKAGE/'evidence/selection.json'),'source_hashes':sources()}
