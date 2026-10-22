"""Freeze final component/reference identities, historical reuse, and missing units."""
import ast
import sys
from ff_prior.storage import read_json,file_digest,digest
from ff_prior_pilot_train.storage import atomic_json
from ff_prior.contracts import assert_frozen
from .protocol import ROOT,PACKAGE,STATIC,DYNAMIC,FIVE,BASE,P


def sources():
    pending=list(PACKAGE.glob('*.py'));found=set()
    def add(module):
        parts=module.split('.')
        for i in range(1,len(parts)+1):
            p=ROOT.joinpath(*parts[:i])
            for f in (p.with_suffix('.py'),p/'__init__.py'):
                if f.is_file() and f not in found:pending.append(f)
    while pending:
        p=pending.pop()
        if p in found:continue
        found.add(p);pkg=p.relative_to(ROOT).with_suffix('').parts[:-1]
        for n in ast.walk(ast.parse(p.read_text(encoding='utf-8-sig'))):
            if isinstance(n,ast.Import):
                for v in n.names:add(v.name)
            elif isinstance(n,ast.ImportFrom):
                prefix=list(pkg[:len(pkg)-n.level+1]) if n.level else []
                module='.'.join(prefix+([n.module] if n.module else []));add(module)
                for v in n.names:
                    if v.name!='*':add(module+'.'+v.name)
    found.update([PACKAGE/'job.ps1',PACKAGE/'docs/EXPERIMENT_PLAN.md'])
    return {p.relative_to(ROOT).as_posix():file_digest(p) for p in sorted(found)}


def matching_source(run):
    old=read_json(run/'run.json')['identity']['source_hashes']
    mismatch=[p for p,h in old.items() if file_digest(ROOT/p)!=h]
    if mismatch:raise ValueError('OLD_COMPONENT_CHANGED: '+repr(mismatch))
    return old


def freeze():
    dest=PACKAGE/'evidence/selection.json'
    if dest.exists():raise ValueError('ONCE_ONLY_FREEZE')
    files={BASE/'run.json',STATIC/'physical_contract.json'};units=[]
    old_static=read_json(STATIC/'run.json')['identity'];old_dynamic=read_json(DYNAMIC/'summary.json')
    static_sources=matching_source(STATIC);dynamic_sources=matching_source(DYNAMIC)
    for source in (STATIC,DYNAMIC,FIVE):files.update([source/'run.json',source/'executed_source.zip'])
    files.update(ROOT/p for p in old_static['binding']['files'])
    files.update(ROOT/p for p in read_json(DYNAMIC/'run.json')['identity']['binding']['files'])
    entries=old_static['selection']['entries']
    for e in entries:assert_frozen(e['task'])
    for spec in old_static['selection']['units']:
        d=STATIC/'units'/spec['id'];result=read_json(d/'result.json')
        group=0 if spec['method']=='structured_geometry' else 1
        row=dict(id=f's{len(units):02d}',scene=spec['task_id'],method=P['methods'][group],
            component_method=spec['method'],seed=spec['seed'],task_hash=spec['task_hash'],source=d.relative_to(ROOT).as_posix())
        files.update([d/'result.json',d/'execution_selection.json'])
        if spec['task_id']=='capture_3_obstacles':
            chosen=read_json(d/'execution_selection.json')['candidate_id'];assert chosen is not None
            v=d/'candidates'/f'{chosen:02d}'/'validated/validation.json'
            assert read_json(v)['eligible'] and result['execution_level']=='EXECUTION_VERIFIED'
            row.update(action='NEW_STATIC_EXECUTION',candidate=chosen,reference=v.relative_to(ROOT).as_posix())
            files.update([v,d/'execution/report.json',d/'execution/frozen_task.json'])
        else:
            row.update(action='BLOCKED',reason='NO_PUBLIC_MODEL_VALID_FIVE_OBSTACLE_PATH',evidence=FIVE.relative_to(ROOT).as_posix())
        units.append(row)
    for r in old_dynamic['results']:
        if r['split']!='DEV':continue
        d=DYNAMIC/r['unit'];report=r['result'];origin=ROOT/report['reused_from'] if report.get('reused_from') else d
        source_result=read_json(origin/'result.json')
        assert report['status']=='EXECUTION_VERIFIED' and source_result['status']=='EXECUTION_VERIFIED'
        assert source_result['replay_full_integration_error']<=P['full_integration_atol']
        files.update(p for p in d.glob('*') if p.is_file())
        files.update(p for p in origin.glob('*') if p.is_file())
        units.append(dict(id=r['unit'],scene=r['scene'],seed_group=r['seed_group'],
            method=P['methods'][0 if r['method']=='deterministic_sequence' else 1],component_method=r['method'],
            source=d.relative_to(ROOT).as_posix(),execution_source=origin.relative_to(ROOT).as_posix(),
            action='EXACT_HISTORICAL_REUSE',original_new_execution=report['new_execution'],
            native_trace_sha256=source_result['native_trace_sha256'],
            task_hash=digest(read_json(d/'task.json'))))
    for i,seed in enumerate([20264611,20264617,20264623]):
        for m,method in enumerate(P['methods']):
            units.append(dict(id=f'c{i}{m}',scene='five_obstacle_to_drifting',method=method,seed=seed,
                action='BLOCKED',reason='NO_QUALIFIED_FIVE_OBSTACLE_END_STATE',
                source=FIVE.relative_to(ROOT).as_posix(),
                task_status='DEPENDENCY_BLOCKED_NOT_FROZEN_FOR_EXECUTION',
                task_roles=dict(approach='original_capture_5_obstacles',tracking='original_drifting_seven_waypoints'),
                task_hash=None,not_a_three_obstacle_substitute=True))
    assert len(units)==30 and sum(u['action']=='NEW_STATIC_EXECUTION' for u in units)==6
    assert sum(u['action']=='EXACT_HISTORICAL_REUSE' for u in units)==12
    assert sum(u['action']=='BLOCKED' for u in units)==12
    for name in ('deterministic_sequence','temporal_diffusion'):
        files.update(DYNAMIC/'learning'/name/n for n in ('model.pt','training.json'))
    files.update([DYNAMIC/'codec.json',DYNAMIC/'summary.json',FIVE/'ledger.json',FIVE/'budget.json'])
    for name in ('static_fullchain_v2','temporal_learning','static_goal_seed'):
        p=ROOT/f'ff_prior_fullchain/evidence/{name}_closeout.json';receipt=read_json(p);assert receipt['closed']
        files.update([p,ROOT/receipt['analysis'],ROOT/receipt['independent_review'],ROOT/receipt['completed_resume']])
    atomic_json(PACKAGE/'evidence/binding.json',dict(files={p.relative_to(ROOT).as_posix():file_digest(p) for p in sorted(files)}))
    selection=dict(units=units,entries=entries,old_static_sources=static_sources,old_dynamic_sources=dynamic_sources,
        physical_contract=old_static['binding'],logical_denominator=30,new_static_executions_max=6,
        reused_dynamic_logical=12,reused_dynamic_unique=8,blocked=12,
        new_proposals=0,new_training=0,development_only=True,source_paths_not_new_planning=True,
        combo_execution_frozen=False,combo_dependency_required=True)
    atomic_json(dest,selection)
    return dict(files=len(files),units=30,new_executions_max=6,exact_reuse_logical=12,blocked=12)


def identity():
    binding=read_json(PACKAGE/'evidence/binding.json')
    for path,h in binding['files'].items():
        if file_digest(ROOT/path)!=h:raise ValueError('INPUT_CHANGED: '+path)
    selection=read_json(PACKAGE/'evidence/selection.json')
    for scope in ('old_static_sources','old_dynamic_sources'):
        for path,h in selection[scope].items():
            if file_digest(ROOT/path)!=h:raise ValueError('REUSE_COMPONENT_CHANGED: '+path)
    return dict(protocol=P,selection=selection,binding=binding,source_hashes=sources())

if __name__=='__main__':
    import json
    print(json.dumps(freeze() if sys.argv[-1]=='freeze' else identity()))
