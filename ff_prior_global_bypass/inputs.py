"""Freeze task inputs and model only. No historical candidate/answer data."""
import ast
import sys
from ff_prior.storage import read_json, file_digest, digest
from ff_prior.contracts import assert_frozen
from ff_prior_pilot_train.storage import atomic_json
from .protocol import ROOT, SOURCE, BASE, CODEC, PACKAGE, P


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
    if destination.exists():raise ValueError('freeze is once-only')
    old_path=ROOT/'ff_prior_static_fullchain_v2/evidence/selection.json'
    original=read_json(old_path)
    entries=[e for e in original['entries'] if e['sample_id']=='capture_5_obstacles']
    assert len(entries)==1 and entries[0]['split']=='APPLICATION_DEV'
    units=[]
    for old in original['units']:
        if old['task_id']!=entries[0]['sample_id']:continue
        u=dict(old,source_unit=old['id'],id=f'u{len(units):02d}');units.append(u)
    receipt=ROOT/'ff_prior_fullchain/evidence/fullchain_acceptance_closeout.json'
    review=ROOT/'paper/review-traces/experiment-result-to-claim/20261022_fullchain_acceptance_run01/verdict.json'
    assert read_json(receipt)['closed']
    authorization=ROOT/'ff_prior_global_bypass/evidence/authorization.json'
    assert read_json(authorization)['accepted_scope'] is True
    inputs=[old_path,SOURCE/'frozen_tasks.json',SOURCE/'physical_contract.json',BASE/'run.json',CODEC,receipt,review,authorization]
    inputs += [SOURCE/'learning/aligned'/n for n in ('model.pt','codec.json','training.json')]
    # Exact URDF and referenced mesh assets, without constructing or running physics.
    import xml.etree.ElementTree as ET
    urdf=ROOT/'dual_arm_space_robot_2026/urdf/dual_arm_space_robot_2026.urdf'
    tree=ET.parse(urdf).getroot();compiler=tree.find('.//compiler')
    meshdir=compiler.attrib.get('meshdir','') if compiler is not None else ''
    inputs.append(urdf)
    inputs += [(urdf.parent/meshdir/x.attrib['filename']).resolve() for x in tree.findall('.//mesh') if 'filename' in x.attrib]
    selection=dict(entries=entries,units=units,candidate_denominator=30,execution_denominator=6,
        task_split='APPLICATION_DEV',before_new_physics=True,private_answer_paths_used=False,
        global_forward_tree=True,nonlinear_goal_seed=True,original_tasks_and_seeds_unchanged=True)
    for e in entries:assert_frozen(e['task'])
    binding={'files':{p.relative_to(ROOT).as_posix():file_digest(p) for p in sorted(set(inputs))}}
    atomic_json(PACKAGE/'evidence/binding.json',binding);atomic_json(destination,selection)
    return dict(tasks=1,units=6,proposals=30,input_files=len(binding['files']),selection_hash=digest(selection))


def identity():
    binding=read_json(PACKAGE/'evidence/binding.json')
    for path,h in binding['files'].items():
        if file_digest(ROOT/path)!=h:raise ValueError('frozen input changed: '+path)
    selection=read_json(PACKAGE/'evidence/selection.json')
    for e in selection['entries']:assert_frozen(e['task'])
    assert len(selection['units'])==6 and sum(len(u['proposal_seeds']) for u in selection['units'])==30
    return {'protocol':P,'source_hashes':sources(),'binding':binding,'selection':selection}


if __name__=='__main__':
    import json
    print(json.dumps(freeze() if sys.argv[-1]=='freeze' else identity(),ensure_ascii=True))
