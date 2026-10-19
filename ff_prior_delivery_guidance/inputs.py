"""Bind only necessary data and the local Python import closure; never read TEST."""
import ast
import hashlib
from pathlib import Path
from ff_prior.storage import read_json, file_digest, digest
from ff_prior.contracts import assert_frozen
from .protocol import ROOT, SOURCE, OLD, BASE, PROTOCOL


def sources():
    pending = list((ROOT/'ff_prior_delivery_guidance').glob('*.py'))
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
    found.add(ROOT/'ff_prior_delivery_guidance/job.ps1')
    return {p.relative_to(ROOT).as_posix(): file_digest(p) for p in sorted(found)}


def identity_for(source=SOURCE):
    source = Path(source).resolve()
    if source != SOURCE.resolve():
        raise ValueError('M1 input population is frozen; alternate source refused')
    index_path = source/'training_index.json'
    index = read_json(index_path)['records']
    if len(index) != 13886 or sum(e['split']=='TRAIN' for e in index) != 13838:
        raise ValueError('current TRAIN/VAL population changed')
    train_families = {e['family_id'] for e in index if e['split']=='TRAIN'}
    other_families = {e['family_id'] for e in index if e['split']!='TRAIN'}
    if train_families & other_families or {e['split'] for e in index} != {'TRAIN', 'VAL'}:
        raise ValueError('TRAIN/VAL family isolation failed')
    named = {f'endpoint_mode_{i:02d}_v0' for i in range(16)}
    pool = [e for e in index if e['sample_id'] in named]
    if len(pool) != 16 or any(e['split']!='TRAIN' for e in pool):
        raise ValueError('missing fixed TRAIN source tasks')
    files = {index_path.relative_to(ROOT).as_posix(): file_digest(index_path)}
    grouped = {False: [], True: []}
    for e in pool:
        p = ROOT/e['record']
        if file_digest(p) != e['record_sha256']:
            raise ValueError('source record changed: '+str(p))
        files[e['record']] = e['record_sha256']
        row = read_json(p)
        assert_frozen(row['task'])
        grouped[bool(row['scene']['objects'])].append((e, row))
    entries = []
    for obstacle in (False, True):
        chosen = sorted(grouped[obstacle], key=lambda pair: hashlib.sha256(pair[0]['sample_id'].encode()).hexdigest())[:4]
        if len(chosen) != 4:
            raise ValueError('target stratum too small')
        for e, row in chosen:
            if file_digest(ROOT/e['arrays']) != e['arrays_sha256']:
                raise ValueError('selected teacher arrays changed')
            files[e['arrays']] = e['arrays_sha256']
            entries.append({'slot': len(entries), 'sample_id': e['sample_id'], 'family_id': e['family_id'],
                            'split': 'TRAIN', 'scene': row['scene'], 'task': row['task'],
                            'arrays': e['arrays'], 'reference_hash': row['reference_hash'],
                            'selection_used_method_results': False, 'goal_relabelled': False})
    for p in [OLD/'learning/goal_loss'/name for name in ('model.pt','codec.json','training.json')]+[
        OLD/'learning/weighted_mean.json', OLD/'summary.json', OLD/'independent_verification.json',
        BASE/'run.json', BASE/'baseline_audit.json']:
        files[p.relative_to(ROOT).as_posix()] = file_digest(p)
    if file_digest(OLD/'learning/goal_loss/model.pt') != PROTOCOL['model_sha256']:
        raise ValueError('stable model checkpoint changed')
    identity = {'protocol': PROTOCOL, 'source': str(source), 'source_files': files,
                'source_hashes': sources(), 'tasks_hash': digest(entries)}
    return identity, (read_json(BASE/'run.json')['config'], entries)
