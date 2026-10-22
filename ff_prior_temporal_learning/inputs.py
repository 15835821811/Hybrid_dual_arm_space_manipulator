import sys
import ast
from ff_prior.storage import read_json,file_digest,digest
from ff_prior_pilot_train.storage import atomic_json
from .protocol import ROOT,PACKAGE,DATA,DEV,P

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

def freeze():
    destination=PACKAGE/'evidence/selection.json'
    if destination.exists():raise ValueError('ONCE_ONLY_FREEZE')
    view=read_json(DATA/'qualified_view.json');assert view['gate']['passed']
    review_path=ROOT/'paper/review-traces/experiment-result-to-claim/20261022_temporal_data_run01/verdict.json'
    review=read_json(review_path);assert review['can_proceed_existing_single_temporal_candidate']
    receipt=read_json(ROOT/'ff_prior_fullchain/evidence/temporal_data_closeout.json');assert receipt['closed']
    assert receipt['elapsed_s']+P['wall_s']<=21600 and receipt['output_bytes']+P['max_bytes']<=20*1024**3
    files={p for p in DATA.rglob('*') if p.is_file()}
    files.update(ROOT/p for p in read_json(DATA/'run.json')['identity']['binding']['files'])
    files.update([review_path,ROOT/'ff_prior_fullchain/evidence/temporal_data_closeout.json'])
    for unit in ['u00','u06']:
        files.update(DEV/unit/name for name in ['task.json','initial.npz','identity.json'])
    files.add(DEV/'run.json')
    atomic_json(PACKAGE/'evidence/binding.json',dict(files={p.relative_to(ROOT).as_posix():file_digest(p) for p in sorted(files)}))
    atomic_json(destination,dict(data=view['rows'],train=8,validation=4,diagnostic_seeds=[20264200+i for i in range(8)],
        VAL_sample_seeds=[20266000+i for i in range(4)],
        DEV_task_seeds={'stationary':20261020,'drifting':20261021},
        DEV_noise_seeds={'stationary':[20261020,20262020,20263020],'drifting':[20261021,20262021,20263021]},
        logical_DEV_units=12,unique_DEV_executions_max=8,unique_VAL_executions_max=8,
        data_selected_before_training=True,DEV_never_in_gradient=True,TEST=False,
        P3_previous_consumption=receipt))
    return dict(files=len(files),selected_train=8,selected_validation=4,wall_s=P['wall_s'])

def identity():
    binding=read_json(PACKAGE/'evidence/binding.json')
    for path,h in binding['files'].items():
        if file_digest(ROOT/path)!=h:raise ValueError('INPUT_CHANGED: '+path)
    return dict(protocol=P,selection=read_json(PACKAGE/'evidence/selection.json'),
                binding=binding,source_hashes=sources())

if __name__=='__main__':
    import json
    print(json.dumps(freeze() if sys.argv[-1]=='freeze' else identity()))
