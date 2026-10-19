"""Exact selected inputs and imported source closure for the M2 worker."""
import ast
from pathlib import Path
from ff_prior.storage import read_json, file_digest, digest
from ff_prior_loop.data import assert_disjoint
from ff_prior_delivery_train.inputs import identity_for as parent_identity
from .protocol import ROOT, SOURCE, MODEL_RUN, MODEL_DIR, BASE, SELECTION, POPULATION, PROTOCOL, PREPARED


def sources():
    pending = list((ROOT/'ff_prior_expand256').glob('*.py'))
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
    found.add(ROOT/'ff_prior_expand256/job.ps1')
    return {p.relative_to(ROOT).as_posix(): file_digest(p) for p in sorted(found)}


def identity_for(source=SOURCE):
    if Path(source).resolve()!=SOURCE.resolve():raise ValueError('alternate source refused')
    selection=read_json(SELECTION);population=read_json(POPULATION)
    if selection['protocol']!=PROTOCOL or selection['population_sha256']!=file_digest(POPULATION):raise ValueError('selection changed')
    if selection['source_index_sha256']!=file_digest(SOURCE/'training_index.json'):raise ValueError('source index changed')
    if digest(selection['entries'])!=selection['entries_digest'] or not selection['selection_before_generation']:raise ValueError('task freeze changed')
    if parent_identity()[0]!=read_json(MODEL_RUN/'run.json')['identity']:raise ValueError('delivered training source or inputs changed')
    summary=read_json(MODEL_RUN/'summary.json')
    if summary['selected_method']!='expanded_diffusion' or summary['audit']!='PASS':raise ValueError('delivered model not qualified')
    prepared=read_json(PREPARED)
    if not prepared['engineering_prerequisites_pass'] or prepared['slots']!=256:raise ValueError('prepared protocol unavailable')
    model=read_json(ROOT/'ff_prior_delivery_release/evidence/model_manifest.json')
    files={}
    for name,b in model['files'].items():
        p=MODEL_DIR/name
        if file_digest(p)!=b['sha256']:raise ValueError('delivered model or codec changed')
        files[p.relative_to(ROOT).as_posix()]=b['sha256']
    if file_digest(MODEL_DIR/'model.pt')!=selection['model_sha256']:raise ValueError('selection checkpoint changed')
    index=read_json(SOURCE/'training_index.json')['records'];assert_disjoint(index)
    if sum(e['split']=='TRAIN' for e in index)!=13966 or sum(e['split']=='VAL' for e in index)!=48:raise ValueError('population changed')
    by_id={e['sample_id']:e for e in index};entries=selection['entries']
    if len(entries)!=64 or sum(bool(e['scene']['objects']) for e in entries)!=32:raise ValueError('strata not frozen')
    if len(set(selection['execution_target_ids']))!=32:raise ValueError('execution quota mismatch')
    forbidden_families=set(population['excluded_heldout_families']);forbidden_tasks=set(population['excluded_task_hashes'])
    from .protocol import proposal_id,proposal_seed
    for e in entries:
        entry=e['source_entry']
        if by_id[e['parent_sample_id']]!=entry or entry['split']!='TRAIN':raise ValueError('not inherited TRAIN')
        if e['family_id'] in forbidden_families or e['task']['task_hash'] in forbidden_tasks:raise ValueError('heldout or previous target reused')
        expected=[{'slot':s,'attempt':a,'sample_id':proposal_id(e['target_slot'],s,a),'seed':proposal_seed(e['target_slot'],s,a)} for s in range(4) for a in range(2)]
        if e['proposals']!=expected:raise ValueError('frozen seeds changed')
        for k in ('record','arrays'):
            if file_digest(ROOT/entry[k])!=entry[k+'_sha256']:raise ValueError('parent bytes changed')
            files[entry[k]]=entry[k+'_sha256']
        row=read_json(ROOT/entry['record'])
        if row['task']!=e['task'] or row.get('scene',{'objects':[]})!=e['scene']:raise ValueError('parent task/scene changed')
        if 'complete_initial_state' in row:
            p=(ROOT/entry['record']).parent.parent/row['complete_initial_state']
            if file_digest(p)!=row['initial_state_sha256']:raise ValueError('full initial state changed')
            files[p.relative_to(ROOT).as_posix()]=file_digest(p)
    for name,h in population['heldout_metadata_bindings'].items():
        if file_digest(ROOT/name)!=h:raise ValueError('heldout exclusion metadata changed')
        files[name]=h
    for p in [SELECTION,POPULATION,PREPARED,SOURCE/'training_index.json',SOURCE/'frozen_tasks.json',MODEL_RUN/'run.json',
              MODEL_RUN/'summary.json',MODEL_RUN/'independent_verification.json',BASE/'run.json',BASE/'baseline_audit.json',
              ROOT/'ff_prior_delivery_release/evidence/model_manifest.json']:
        files[p.relative_to(ROOT).as_posix()]=file_digest(p)
    return {'protocol':PROTOCOL,'source':str(SOURCE),'source_hashes':sources(),'source_files':files,
            'tasks_hash':digest(entries),'sampler':selection['sampler'],'selection_sha256':file_digest(SELECTION)},(
                read_json(BASE/'run.json')['config'],selection,index,population['existing_reference_hashes'])


