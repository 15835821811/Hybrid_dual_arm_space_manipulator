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
    from .protocol import PREVIOUS
    from .data import load
    if (PACKAGE/'evidence/selection.json').exists():raise ValueError('once-only freeze')
    old_package=ROOT/'ff_prior_rollout_world'
    files=dict(read_json(old_package/'evidence/binding.json')['files'])
    assert read_json(PREVIOUS/'status.json')['status']=='FINISHED'
    assert read_json(PREVIOUS/'independent_verification.json')['status']=='PASS'
    for p,h in files.items():assert file_digest(ROOT/p)==h
    for p in PREVIOUS.rglob('*'):
        if p.is_file():files[p.relative_to(ROOT).as_posix()]=file_digest(p)
    for p in [old_package/'docs/ANALYSIS.md',old_package/'docs/CLAIM_VERDICT.md',old_package/'evidence/completed/completed_resume.json']:
        files[p.relative_to(ROOT).as_posix()]=file_digest(p)
    s=read_json(old_package/'evidence/selection.json')
    codec,arrays,rows,data=load()
    forbidden=set(s['forbidden_TRAIN_task_hashes'])
    unique={}
    for i in data['tr']:
        r=rows[i];h=r['task']['task_hash']
        if h not in forbidden and h not in unique:unique[h]=int(i)
    complex_ids=[data['ids'][r['sample_id']] for r in s['diagnostic'] if r['sample_id'].startswith('complex_train_')]
    assert len(complex_ids)==6 and len({rows[i]['task']['task_hash'] for i in complex_ids})==6
    complex_hashes={rows[i]['task']['task_hash'] for i in complex_ids}
    ordinary=[i for h,i in sorted(unique.items()) if h not in complex_hashes]
    ordinary.sort(key=lambda i:(float(np.sqrt(np.mean((arrays[0][i]-arrays[1][i])**2))),rows[i]['task']['task_hash']))
    bins=np.array_split(ordinary,3);rng=np.random.default_rng(P['selection_seed'])
    groups=[('complex_existing',complex_ids,3),('motion_small',bins[0].tolist(),4),('motion_medium',bins[1].tolist(),4),('motion_large',bins[2].tolist(),5)]
    fit=[];check=[]
    for name,pool,n in groups:
        order=rng.permutation(len(pool));assert len(pool)>=2*n
        for dest,slots in [(fit,order[:n]),(check,order[n:2*n])]:
            for slot in slots:
                r=dict(rows[pool[int(slot)]]);r['frozen_stratum']=name
                r['teacher_excursion_rms_rad']=float(np.sqrt(np.mean((arrays[0][pool[int(slot)]]-arrays[1][pool[int(slot)]])**2)))
                dest.append(r)
    fit_hashes={r['task']['task_hash'] for r in fit};check_hashes={r['task']['task_hash'] for r in check}
    assert len(fit)==len(check)==16 and not fit_hashes&check_hashes
    excluded=[i for i in data['tr'] if rows[i]['task']['task_hash'] in check_hashes]
    s.update(fit_pool=fit,check_pool=check,check_task_hashes=sorted(check_hashes),
        fit_task_hashes=sorted(fit_hashes),adaptation_excluded_ids=[arrays[3][i] for i in excluded],
        adaptation_TRAIN_count=len(data['tr'])-len(excluded),check_scope='held out from this round adaptation only; old model and immutable codec previously saw TRAIN',
        fit_seed_pairs=[[P['diagnostic_seeds'][k]+j*997 for k in range(2)] for j in range(16)],
        check_seed_pairs=[[P['diagnostic_seeds'][k]+500003+j*997 for k in range(2)] for j in range(16)],
        frozen_before_any_new_predictions=True,original_protected=None)
    atomic_json(PACKAGE/'evidence/selection.json',s)
    atomic_json(PACKAGE/'evidence/binding.json',{'files':files})
    (PACKAGE/'evidence/previous_goal.md').write_bytes((ROOT/'DIFFUSION_GOAL_CURRENT.md').read_bytes())
    return {'files':len(files),'TRAIN_adaptation':s['adaptation_TRAIN_count'],'check_excluded':len(excluded),'fit_outputs':32,'check_outputs':32,'new_data':0}


def identity():
    b=read_json(PACKAGE/'evidence/binding.json')
    for p,h in b['files'].items():
        if file_digest(ROOT/p)!=h:raise ValueError('frozen input changed: '+p)
    return {'protocol':P,'binding':b,'selection':read_json(PACKAGE/'evidence/selection.json'),'source_hashes':sources()}
