"""Minimal immutable lineage; avoid re-auditing all ancestral experiment trees."""
import ast
from pathlib import Path
from ff_prior.storage import read_json,file_digest,digest
from ff_prior.recovery import process_is_alive
from ff_prior_loop.data import assert_disjoint
from ff_prior_delivery_dataset.inputs import identity_for as parent_identity
from .protocol import ROOT,SOURCE,CACHE,OLD,BASE,CACHE_HASHES,PROTOCOL


def sources():
    pending = list((ROOT/'ff_prior_delivery_train').glob('*.py'))
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
    found.add(ROOT/'ff_prior_delivery_train/job.ps1')
    return {p.relative_to(ROOT).as_posix(): file_digest(p) for p in sorted(found)}


def identity_for(source=SOURCE):
    if Path(source).resolve()!=SOURCE.resolve():raise ValueError('exact M2 data required')
    state=read_json(SOURCE/'status.json');summary=read_json(SOURCE/'summary.json')
    if state['status']!='FINISHED' or process_is_alive(state['pid']) or not summary['M3_ready']:
        raise ValueError('M2 must finish and pass before M3')
    if read_json(SOURCE/'independent_verification.json')['status']!='PASS':raise ValueError('M2 raw audit missing')
    if parent_identity()[0]!=read_json(SOURCE/'run.json')['identity']:raise ValueError('M2 frozen identity changed')
    index=read_json(SOURCE/'training_index.json')['records'];assert_disjoint(index)
    new=read_json(SOURCE/'qualified_increment.json')['records']
    if sum(e['split']=='TRAIN' for e in index)!=13966 or sum(e['split']=='VAL' for e in index)!=48 or len(new)!=128:
        raise ValueError('training population changed')
    if any(e['execution_status'] not in ('NOT_RUN','PASS') for e in index):raise ValueError('failed/unknown execution in training')
    entries=read_json(OLD/'frozen_tasks.json')['entries']
    if len(entries)!=12 or [e['split'] for e in entries]!=['DEV']*4+['NEW_VAL']*8:
        raise ValueError('original frozen development tasks changed')
    families={e['family_id'] for e in index if e['split']=='TRAIN'}
    if any(e.get('family_id') in families for e in entries):raise ValueError('development family in TRAIN')
    files={}
    for name,expected in CACHE_HASHES.items():
        p=CACHE/name
        if file_digest(p)!=expected:raise ValueError('verified parent training cache changed')
        files[p.relative_to(ROOT).as_posix()]=expected
    for p in [SOURCE/name for name in ('run.json','summary.json','independent_verification.json','training_index.json','qualified_increment.json')]+[
            OLD/'frozen_tasks.json',OLD/'sensitivity_summary.json',OLD/'summary.json',OLD/'independent_verification.json',
            OLD/'learning/goal_loss/model.pt',OLD/'learning/goal_loss/codec.json',OLD/'learning/goal_loss/training.json',
            BASE/'run.json',BASE/'baseline_audit.json']:
        files[p.relative_to(ROOT).as_posix()]=file_digest(p)
    for e in new:
        for k in ('record','arrays'):
            if file_digest(ROOT/e[k])!=e[k+'_sha256']:raise ValueError('new qualified bytes changed')
            files[e[k]]=e[k+'_sha256']
        row=read_json(ROOT/e['record']);p=SOURCE/row['complete_initial_state']
        if file_digest(p)!=row['initial_state_sha256']:raise ValueError('new full state changed')
        files[p.relative_to(ROOT).as_posix()]=file_digest(p)
    return {'protocol':PROTOCOL,'source':str(SOURCE),'source_hashes':sources(),'source_files':files,
            'comparison_tasks_hash':digest(entries),'new_ids':[e['sample_id'] for e in new]},(
                read_json(BASE/'run.json')['config'],index,entries,[e['sample_id'] for e in new])
