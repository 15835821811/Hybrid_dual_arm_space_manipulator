import sys
import ast
from ff_prior.storage import read_json, file_digest, digest
from ff_prior_pilot_train.storage import atomic_json
from .protocol import ROOT, PACKAGE, SOURCE, P, PREVIOUS

def sources():
    pending = list(PACKAGE.glob('*.py')); found = set()
    def add(module):
        parts=module.split('.')
        for i in range(1,len(parts)+1):
            p=ROOT.joinpath(*parts[:i])
            for path in (p.with_suffix('.py'),p/'__init__.py'):
                if path.is_file() and path not in found: pending.append(path)
    while pending:
        p=pending.pop()
        if p in found: continue
        found.add(p);pkg=p.relative_to(ROOT).with_suffix('').parts[:-1]
        for node in ast.walk(ast.parse(p.read_text(encoding='utf-8-sig'))):
            if isinstance(node,ast.Import):
                for name in node.names: add(name.name)
            elif isinstance(node,ast.ImportFrom):
                prefix=list(pkg[:len(pkg)-node.level+1]) if node.level else []
                module='.'.join(prefix+([node.module] if node.module else []));add(module)
                for name in node.names:
                    if name.name!='*':add(module+'.'+name.name)
    found.add(PACKAGE/'job.ps1')
    found.add(PACKAGE/'docs/EXPERIMENT_PLAN.md')
    return {p.relative_to(ROOT).as_posix():file_digest(p) for p in sorted(found)}


def freeze():
    destination=PACKAGE/'evidence/selection.json'
    if destination.exists():raise ValueError('once-only freeze')
    old=read_json(SOURCE);rows=old['certification']
    assert len(rows)==4 and [r['seed'] for r in rows]==[31000000,31104729,31209458,31314187]
    files=[ROOT/'ff_prior_dynamic_reference/evidence/completed_statistics.json',SOURCE,ROOT/'tracking_diffusion/config/local10mm_v1.json',
           ROOT/'dual_arm_space_robot_2026/urdf/dual_arm_space_robot_2026.urdf']
    for row in rows:
        assert row['split']=='train' and row['target_motion']=='moving'
        for name in ['trace.npz','result.json']:files.append(ROOT/row['path']/name)
        assert file_digest(ROOT/row['path']/'trace.npz')==row['trace_sha256']
    files.extend(p for p in PREVIOUS.rglob('*') if p.is_file())
    # Complete local mesh identity, not only URDF text.
    files.extend(p for p in (ROOT/'dual_arm_space_robot_2026/meshes').rglob('*') if p.is_file())
    atomic_json(PACKAGE/'evidence/binding.json',dict(files={p.relative_to(ROOT).as_posix():file_digest(p) for p in files}))
    atomic_json(destination,dict(rows=rows,denominator=4,split='TRAIN',new_dataset_entries=0,
        source_selection=str(SOURCE.relative_to(ROOT)),before_new_fit_or_physics=True,new_execution_case=0,reuse_cases=[1,2,3],
        margin_source='ff_prior_dynamic_reference/evidence/completed_statistics.json'))
    return dict(frozen=4,bindings=len(files))

def identity():
    binding=read_json(PACKAGE/'evidence/binding.json')
    for path,h in binding['files'].items():
        if file_digest(ROOT/path)!=h:raise ValueError('input changed: '+path)
    return dict(protocol=P,selection=read_json(PACKAGE/'evidence/selection.json'),
                binding=binding,source_hashes=sources())

if __name__=='__main__':
    import json
    print(json.dumps(freeze() if sys.argv[-1]=='freeze' else identity()))
