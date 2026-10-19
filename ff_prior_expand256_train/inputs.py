"""Minimal immutable lineage; avoid re-auditing all ancestral experiment trees."""
import ast
from pathlib import Path
from ff_prior.storage import read_json,file_digest,digest
from ff_prior.recovery import process_is_alive
from ff_prior_loop.data import assert_disjoint
from ff_prior_expand256.inputs import identity_for as source_identity
from .protocol import ROOT,SOURCE,CACHE,OLD,BASE,CACHE_HASHES,PROTOCOL


def sources():
    pending = list((ROOT/'ff_prior_expand256_train').glob('*.py'))
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
    found.add(ROOT/'ff_prior_expand256_train/job.ps1')
    return {p.relative_to(ROOT).as_posix(): file_digest(p) for p in sorted(found)}


def require_expansion_gate(state,summary,audit,alive=False):
    if state['status']!='FINISHED' or alive:raise ValueError('expansion not finished and exited')
    checks=(summary.get('next_stage_ready'),summary.get('data_status')=='PASS',audit.get('status')=='PASS',
        summary.get('qualified_increment',0)>=128,summary.get('fixed_targets_covered',0)>=32,
        summary.get('execution_verified',0)>=26,summary.get('execution_units')==32,
        summary.get('slots_accounted')==256,summary.get('admission_rate_actual_attempts',0)>=.5,
        not summary.get('errors'),not summary.get('unknown_or_pending_attempts'))
    if not all(checks):raise ValueError('expansion gate BLOCKED; no training')


def identity_for(source=SOURCE):
    from .protocol import BINDING
    if Path(source).resolve()!=SOURCE.resolve():raise ValueError('exact expanded dataset required')
    state=read_json(SOURCE/'status.json')
    if state['status']!='FINISHED':raise ValueError('expansion dependency not complete')
    summary=read_json(SOURCE/'summary.json');audit=read_json(SOURCE/'independent_verification.json')
    require_expansion_gate(state,summary,audit,process_is_alive(state['pid']))
    if source_identity()[0]!=read_json(SOURCE/'run.json')['identity']:raise ValueError('expansion source or inputs changed')
    index=read_json(SOURCE/'training_index.json')['records'];assert_disjoint(index)
    new=read_json(SOURCE/'qualified_increment.json')['records']
    if not 128<=len(new)<=256 or len(new)!=summary['qualified_increment']:raise ValueError('qualified increment count mismatch')
    if sum(e['split']=='TRAIN' for e in index)!=13966+len(new) or sum(e['split']=='VAL' for e in index)!=48:
        raise ValueError('merged TRAIN/VAL count changed')
    if any(e['execution_status'] not in ('NOT_RUN','PASS') for e in index):raise ValueError('failed execution in training')
    if {e['sample_id'] for e in index if e.get('new_in_this_run')}!={e['sample_id'] for e in new}:raise ValueError('new IDs mismatch')
    entries=read_json(OLD/'frozen_tasks.json')['entries']
    if len(entries)!=12 or [e['split'] for e in entries]!=['DEV']*4+['NEW_VAL']*8:raise ValueError('original task protocol changed')
    families={e['family_id'] for e in index if e['split']=='TRAIN'}
    if any(e.get('family_id') in families for e in entries):raise ValueError('development family leakage')
    files={};warm=read_json(BINDING)
    for name,h in CACHE_HASHES.items():
        p=CACHE/name
        if file_digest(p)!=h:raise ValueError('bound warm cache changed')
        files[p.relative_to(ROOT).as_posix()]=h
    for name,h in warm['model_hashes'].items():
        p=OLD/'learning/goal_loss'/name
        if file_digest(p)!=h:raise ValueError('warm weights or codec changed')
        files[p.relative_to(ROOT).as_posix()]=h
    for p in [BINDING]+[SOURCE/n for n in ['run.json','summary.json','independent_verification.json','training_index.json','qualified_increment.json']]+[
        OLD/'frozen_tasks.json',OLD/'sensitivity_summary.json',OLD/'independent_verification.json',BASE/'run.json',BASE/'baseline_audit.json']:
        files[p.relative_to(ROOT).as_posix()]=file_digest(p)
    for e in new:
        if e['split']!='TRAIN':raise ValueError('nonTRAIN increment')
        for k in ('record','arrays'):
            if file_digest(ROOT/e[k])!=e[k+'_sha256']:raise ValueError('increment bytes changed')
            files[e[k]]=e[k+'_sha256']
        row=read_json(ROOT/e['record']);p=SOURCE/row['complete_initial_state']
        if file_digest(p)!=row['initial_state_sha256']:raise ValueError('full state changed')
        files[p.relative_to(ROOT).as_posix()]=file_digest(p)
    return {'protocol':PROTOCOL,'source':str(SOURCE),'source_hashes':sources(),'source_files':files,
        'actual_TRAIN':13966+len(new),'actual_increment':len(new),'comparison_tasks_hash':digest(entries),
        'new_ids':[e['sample_id'] for e in new]},(read_json(BASE/'run.json')['config'],index,entries,[e['sample_id'] for e in new])
