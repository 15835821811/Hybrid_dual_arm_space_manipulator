import sys
import ast
from ff_prior.storage import read_json,file_digest,digest
from ff_prior_pilot_train.storage import atomic_json
from .protocol import ROOT,PACKAGE,SOURCE,P,OLD_REFERENCE,OLD_MARGIN

def sources():
    pending=list(PACKAGE.glob('*.py'));found=set()
    def add(module):
        parts=module.split('.')
        for i in range(1,len(parts)+1):
            p=ROOT.joinpath(*parts[:i])
            for path in (p.with_suffix('.py'),p/'__init__.py'):
                if path.is_file() and path not in found:pending.append(path)
    while pending:
        p=pending.pop()
        if p in found:continue
        found.add(p);pkg=p.relative_to(ROOT).with_suffix('').parts[:-1]
        for node in ast.walk(ast.parse(p.read_text(encoding='utf-8-sig'))):
            if isinstance(node,ast.Import):
                for name in node.names:add(name.name)
            elif isinstance(node,ast.ImportFrom):
                prefix=list(pkg[:len(pkg)-node.level+1]) if node.level else []
                module='.'.join(prefix+([node.module] if node.module else []));add(module)
                for name in node.names:
                    if name.name!='*':add(module+'.'+name.name)
    found.update([PACKAGE/'job.ps1',PACKAGE/'docs/EXPERIMENT_PLAN.md'])
    return {p.relative_to(ROOT).as_posix():file_digest(p) for p in sorted(found)}

def selection(inventory):
    """Prefix selection is frozen before new fits or executions, never outcome-based."""
    assert len(inventory['train'])==32 and len(inventory['validation'])==8
    keys=[];rows=[]
    for split,limit in [('train',8),('validation',4)]:
        for i,row in enumerate(inventory[split]):
            assert row['split']==split and row['target_motion']=='moving'
            assert row['old_admitted'] and row['available'] and row['finite']
            key=(row['path'],row['seed']);assert key not in keys;keys.append(key)
            if i<limit:
                rows.append(dict(id=('t' if split=='train' else 'v')+f'{i:02}',
                    split=split,index=i,source=row,
                    reuse=(('artifacts/v7_1/dynamic_margin_001/t00' if i==0 else
                            f'artifacts/v7_1/dynamic_reference_001/t{i:02}')
                           if split=='train' and i<4 else None)))
    assert not ({r['seed'] for r in inventory['train']} & {r['seed'] for r in inventory['validation']})
    return rows

def freeze():
    destination=PACKAGE/'evidence/selection.json'
    if destination.exists():raise ValueError('ONCE_ONLY_FREEZE')
    old=read_json(SOURCE);inventory=old['selected'];rows=selection(inventory)
    assert [r['source'] for r in rows[:4]]==old['certification']
    files={SOURCE,ROOT/'tracking_diffusion/config/local10mm_v1.json',
           ROOT/'dual_arm_space_robot_2026/urdf/dual_arm_space_robot_2026.urdf',
           ROOT/'ff_prior_fullchain/evidence/dynamic_handoff_closeout.json',
           ROOT/'ff_prior_fullchain/evidence/dynamic_margin_closeout.json'}
    for split,items in inventory.items():
        for row in items:
            for name,key in [('trace.npz','trace_sha256'),('result.json','result_sha256')]:
                p=ROOT/row['path']/name;assert file_digest(p)==row[key];files.add(p)
    for previous in (OLD_REFERENCE,OLD_MARGIN):
        files.add(previous/'run.json')
    for row in rows[:4]:
        files.update(p for p in (ROOT/row['reuse']).rglob('*') if p.is_file())
    files.update(p for p in (ROOT/'dual_arm_space_robot_2026/meshes').rglob('*') if p.is_file())
    atomic_json(PACKAGE/'evidence/binding.json',dict(files={p.relative_to(ROOT).as_posix():file_digest(p) for p in sorted(files)}))
    atomic_json(destination,dict(rows=rows,inventory=inventory,denominator=12,
        selected_before_fitting=True,new_tasks=0,new_library_entries=0,TEST=False))
    return dict(selected=12,new_execution_max=8,bindings=len(files))

def identity():
    binding=read_json(PACKAGE/'evidence/binding.json')
    for path,h in binding['files'].items():
        if file_digest(ROOT/path)!=h:raise ValueError('INPUT_CHANGED: '+path)
    return dict(protocol=P,selection=read_json(PACKAGE/'evidence/selection.json'),
                binding=binding,source_hashes=sources())

if __name__=='__main__':
    import json
    print(json.dumps(freeze() if sys.argv[-1]=='freeze' else identity()))
