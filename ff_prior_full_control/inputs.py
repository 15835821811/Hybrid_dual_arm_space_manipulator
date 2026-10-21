import ast
import numpy as np
from ff_prior.storage import read_json, file_digest, digest
from ff_prior_pilot_train.storage import atomic_json
from .protocol import ROOT, SOURCE, BASE, PACKAGE, P, DATA

def sources():
    pending = list(PACKAGE.glob("*.py")); found = set()
    def add(module):
        parts = module.split(".")
        for i in range(1, len(parts)+1):
            p = ROOT.joinpath(*parts[:i])
            for path in (p.with_suffix(".py"), p/"__init__.py"):
                if path.is_file() and path not in found:
                    pending.append(path)
    while pending:
        p = pending.pop()
        if p in found:
            continue
        found.add(p)
        pkg = p.relative_to(ROOT).with_suffix("").parts[:-1]
        for node in ast.walk(ast.parse(p.read_text(encoding="utf-8-sig"))):
            if isinstance(node, ast.Import):
                for name in node.names: add(name.name)
            elif isinstance(node, ast.ImportFrom):
                prefix = list(pkg[:len(pkg)-node.level+1]) if node.level else []
                module = ".".join(prefix+([node.module] if node.module else []))
                add(module)
                for name in node.names:
                    if name.name != "*": add(module+"."+name.name)
    found.add(PACKAGE/"job.ps1")
    return {p.relative_to(ROOT).as_posix(): file_digest(p) for p in sorted(found)}

def freeze():
    if (PACKAGE/'evidence/selection.json').exists():raise ValueError('once-only freeze')
    files={}
    def bind(path):
        files[path.relative_to(ROOT).as_posix()]=file_digest(path)
        return read_json(path) if path.suffix=='.json' else None
    for root in (SOURCE,DATA):
        assert bind(root/'status.json')['status']=='FINISHED'
        assert bind(root/'independent_verification.json')['status']=='PASS'
        bind(root/'run.json');bind(root/'physical_contract.json')
    index=bind(DATA/'training_input_audit.json')['records']
    bind(DATA/'loaded_training_arrays.npz');bind(DATA/'learning/goal_loss/training.json')
    rows=[bind(ROOT/e['record']) for e in index]
    s=bind(SOURCE/'frozen_tasks.json')
    entries=s['entries'];apps=s['application']['entries']
    assert len(entries)==12 and len(apps)==2 and s['sampling_seeds']==P['sampling_seeds']
    forbidden={e['task']['task_hash'] for e in entries+apps}
    tr=[r for r in rows if r['split']=='TRAIN'];va=[r for r in rows if r['split']=='VAL']
    assert len(tr)==14222 and len(va)==48
    assert not {r['task']['task_hash'] for r in tr}&forbidden
    assert not {r['family_id'] for r in tr}&{r['family_id'] for r in va}
    selected=sorted([r for r in tr if r['sample_id'].startswith('complex_train_')],key=lambda r:r['sample_id'])
    assert len(selected)==6 and all(r['validation_level']=='EXECUTION_VERIFIED' and r['training_eligible'] for r in selected)
    for r in selected:
        directory=ROOT/'artifacts/v7_1/complex_train_val_001'
        for p in (directory/'tasks'/r['sample_id']/'witness.npz', directory/'witness_executions'/r['sample_id']/'report.json'):
            bind(p)
    reuse=[]
    for sj,seed in enumerate(P['sampling_seeds']):
        for ti,e in enumerate(entries):
            sid=e['sample_id']
            for method,root,old in [('old_aligned',SOURCE,'aligned'),('structured',DATA,'structured'),('learned_mean',DATA,'learned_mean')]:
                directory=root/'candidates'/str(sj)/sid/old
                paths=[]
                for k in range(5):
                    p=directory/f'{k:02d}.json';r=bind(p)
                    assert r['task_hash']==e['task']['task_hash'] and r['seed']==seed+ti*101+k*997
                    paths.append(p.relative_to(ROOT).as_posix())
                execution=root/'executions'/str(sj)/sid/old
                for p in execution.glob('*.json'):bind(p)
                reuse.append({'scope':'DEV','seed_group':sj,'sample_id':sid,'method':method,'candidates':paths,'execution':execution.relative_to(ROOT).as_posix()})
    for e in apps:
        for method in ('aligned','structured','learned_mean'):
            directory=SOURCE/'application'/e['sample_id']/method;paths=[]
            for k in range(5):
                p=directory/'candidates'/f'{k:02d}.json';r=bind(p)
                assert r['task_hash']==e['task']['task_hash'] and r['seed']==e['seed']+997*k
                paths.append(p.relative_to(ROOT).as_posix())
            for p in (directory/'execution').glob('*.json'):bind(p)
            reuse.append({'scope':'APPLICATION_DEV','sample_id':e['sample_id'],'method':'old_aligned' if method=='aligned' else method,'candidates':paths,'execution':(directory/'execution').relative_to(ROOT).as_posix()})
    for sj in range(2):
        for j in range(48):bind(SOURCE/'unguided'/str(sj)/'aligned'/f'{j:02d}.json')
    bind(SOURCE/'learning/aligned/training.json');bind(SOURCE/'learning/aligned/codec.json')
    bind(BASE/'run.json')
    selection={'entries':entries,'application':apps,'reconstruction':selected,
        'reconstruction_execution_ids':[r['sample_id'] for r in selected if 'static_detour' in r['sample_id']],
        'reuse':reuse,'forbidden_TRAIN_task_hashes':sorted(forbidden),
        'frozen_before_training':True,'no_new_data':True,'TEST':False}
    atomic_json(PACKAGE/'evidence/selection.json',selection)
    atomic_json(PACKAGE/'evidence/binding.json',{'files':files})
    (PACKAGE/'evidence/previous_goal.md').write_bytes((ROOT/'DIFFUSION_GOAL_CURRENT.md').read_bytes())
    return {'files':len(files),'TRAIN':len(tr),'reconstruction':len(selected),'DEV':len(entries)}

def identity():
    b=read_json(PACKAGE/'evidence/binding.json')
    for p,h in b['files'].items():
        if file_digest(ROOT/p)!=h:raise ValueError('frozen input changed: '+p)
    return {'protocol':P,'binding':b,'selection':read_json(PACKAGE/'evidence/selection.json'),'source_hashes':sources()}

