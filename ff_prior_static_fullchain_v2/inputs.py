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
    inputs=[SOURCE/'frozen_tasks.json',SOURCE/'physical_contract.json',BASE/'run.json',CODEC]
    inputs += [SOURCE/'learning/aligned'/n for n in ('model.pt','codec.json','training.json')]
    original=read_json(inputs[0])['application']['entries']
    entries=[]
    for name in P['tasks']:
        e=next(e for e in original if e['sample_id']==name)
        assert e['initial_status']=='PASS' and e['split']=='APPLICATION_DEV'
        assert_frozen(e['task'])
        entries.append({k:e[k] for k in ('sample_id','task','scene','split','initial_status')})
    units=[]
    for ti,entry in enumerate(entries):
        for si,seed in enumerate(P['seeds']):
            methods=P['methods'] if (ti+si)%2==0 else list(reversed(P['methods']))
            for method in methods:
                units.append({'id':f'u{len(units):02d}',
                    'task_id':entry['sample_id'],'task_hash':entry['task']['task_hash'],
                    'seed':seed,'method':method,'proposal_seeds':[seed+k for k in range(5)]})
    selection={'entries':entries,'units':units,'candidate_denominator':60,'execution_denominator':12,
               'task_split':'APPLICATION_DEV','before_new_physics':True,'private_answer_paths_used':False}
    binding={'files':{p.relative_to(ROOT).as_posix():file_digest(p) for p in inputs}}
    atomic_json(PACKAGE/'evidence/binding.json',binding)
    atomic_json(destination,selection)
    return {'tasks':2,'units':12,'proposals':60,'selection_hash':digest(selection)}


def identity():
    binding=read_json(PACKAGE/'evidence/binding.json')
    for path,h in binding['files'].items():
        if file_digest(ROOT/path)!=h:raise ValueError('frozen input changed: '+path)
    selection=read_json(PACKAGE/'evidence/selection.json')
    for e in selection['entries']:assert_frozen(e['task'])
    assert len(selection['units'])==12 and sum(len(u['proposal_seeds']) for u in selection['units'])==60
    return {'protocol':P,'source_hashes':sources(),'binding':binding,'selection':selection}


if __name__=='__main__':
    import json
    print(json.dumps(freeze() if sys.argv[-1]=='freeze' else identity(),ensure_ascii=True))
