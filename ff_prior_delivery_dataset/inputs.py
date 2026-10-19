"""Exact selected inputs and imported source closure for the M2 worker."""
import ast
from pathlib import Path
from ff_prior.storage import read_json, file_digest, digest
from ff_prior_loop.data import assert_disjoint
from ff_prior_delivery_guidance.inputs import identity_for as m1_identity
from .protocol import ROOT, SOURCE, M1, OLD, BASE, SELECTION, POPULATION, PROTOCOL


def sources():
    pending = list((ROOT/'ff_prior_delivery_dataset').glob('*.py'))
    found = set()
    def add(module):
        parts = module.split('.')
        for i in range(1, len(parts)+1):
            p = ROOT.joinpath(*parts[:i])
            for path in (p.with_suffix('.py'), p/'__init__.py'):
                if path.is_file() and path not in found:
                    pending.append(path)
    while pending:
        path = pending.pop()
        if path in found:
            continue
        found.add(path)
        parts = path.relative_to(ROOT).with_suffix('').parts
        package = parts[:-1]
        for node in ast.walk(ast.parse(path.read_text(encoding='utf-8-sig'))):
            if isinstance(node, ast.Import):
                for name in node.names:
                    add(name.name)
            elif isinstance(node, ast.ImportFrom):
                prefix = list(package[:len(package)-node.level+1]) if node.level else []
                module = '.'.join(prefix+([node.module] if node.module else []))
                add(module)
                for name in node.names:
                    if name.name != '*':
                        add(module+'.'+name.name)
    # Non-Python launcher is also frozen, but mutable goal/tracker are not code.
    found.add(ROOT/'ff_prior_delivery_dataset/job.ps1')
    return {p.relative_to(ROOT).as_posix(): file_digest(p) for p in sorted(found)}


def identity_for(source=SOURCE):
    if Path(source).resolve()!=SOURCE.resolve():
        raise ValueError('alternate population refused')
    selection=read_json(SELECTION)
    if selection['protocol']!=PROTOCOL or selection['population_sha256']!=file_digest(POPULATION):
        raise ValueError('frozen selection changed')
    if selection['source_index_sha256']!=file_digest(SOURCE/'training_index.json'):
        raise ValueError('source index changed')
    entries=selection['entries']
    if digest(entries)!=selection['entries_digest'] or not selection['selection_before_generation']:
        raise ValueError('selection identity invalid')
    frozen_m1=read_json(M1/'run.json')['identity']
    if m1_identity()[0]!=frozen_m1:
        raise ValueError('M1 implementation or input changed')
    summary=read_json(M1/'summary.json')
    if not summary['selection']['M2_ready'] or summary['audit']!='PASS' or summary['selection']['method']!=selection['sampler']:
        raise ValueError('M1 selection proof missing')
    index=read_json(SOURCE/'training_index.json')['records']
    assert_disjoint(index)
    by_id={e['sample_id']:e for e in index}
    files={}
    for p in (SELECTION,POPULATION,SOURCE/'training_index.json',M1/'summary.json',M1/'run.json',
              M1/'independent_verification.json',BASE/'run.json',BASE/'baseline_audit.json',
              OLD/'learning/goal_loss/model.pt',OLD/'learning/goal_loss/codec.json',OLD/'learning/goal_loss/training.json'):
        files[p.relative_to(ROOT).as_posix()]=file_digest(p)
    for e in entries:
        entry=e['source_entry']
        if by_id[e['parent_sample_id']]!=entry or entry['split']!='TRAIN':
            raise ValueError('parent is not inherited TRAIN')
        for k in ('record','arrays'):
            if file_digest(ROOT/entry[k])!=entry[k+'_sha256']:
                raise ValueError('selected parent changed')
            files[entry[k]]=entry[k+'_sha256']
        row=read_json(ROOT/entry['record'])
        if row['task']!=e['task'] or row.get('scene',{'objects':[]})!=e['scene']:
            raise ValueError('frozen parent task/scene changed')
        if 'complete_initial_state' in row:
            p=(ROOT/entry['record']).parent.parent/row['complete_initial_state']
            if file_digest(p)!=row['initial_state_sha256']:
                raise ValueError('complete source state changed')
            files[p.relative_to(ROOT).as_posix()]=file_digest(p)
    identity={'protocol':PROTOCOL,'source':str(SOURCE),'source_hashes':sources(),'source_files':files,
              'tasks_hash':digest(entries),'sampler':selection['sampler'],'selection_sha256':file_digest(SELECTION)}
    return identity,(read_json(BASE/'run.json')['config'],selection,index,read_json(POPULATION)['existing_reference_hashes'])
