from pathlib import Path
from ff_prior.storage import read_json,file_digest,digest
from ff_prior.recovery import process_is_alive
from ff_prior_world_endpoint.identity import sources as prior_sources,identity_for as prior_identity
from .protocol import ROOT,SOURCE,PROTOCOL as P

def sources():
    return dict(prior_sources(),**{p.relative_to(ROOT).as_posix():file_digest(p)
        for p in sorted((ROOT/'ff_prior_paired_objectives').rglob('*.py'))})

def identity_for(source):
    source=Path(source).resolve()
    if source!=SOURCE.resolve():raise ValueError('exact audited 13838-row parent required')
    state=read_json(source/'status.json');summary=read_json(source/'summary.json')
    if state['status']!='FINISHED' or process_is_alive(state['pid']):raise ValueError('parent not complete/exited')
    if not summary['independent_audit_passed']:raise ValueError('parent raw audit required')
    old=read_json(source/'run.json')['identity'];expected,context=prior_identity(old['source'])
    if old!=expected:raise ValueError('parent source identity changed')
    config,_,entries,targets=context
    index=read_json(source/'training_index.json')['records']
    if sum(e['split']=='TRAIN' for e in index)!=13838:raise ValueError('TRAIN count changed')
    files=dict(old['source_files'])
    for p in sorted(source.rglob('*')):
        if p.is_file() and p.name not in ('run.lock','launch.guard') and p.suffix!='.tmp':
            files[p.relative_to(ROOT).as_posix()]=file_digest(p)
    identity={'protocol':P,'source':str(source),'source_files':files,'source_hashes':sources(),
        'targets_hash':digest(targets),'comparison_entries_hash':digest(entries),
        'parent_C1':True,'parent_C2':False,'used_validation_role':'REUSED_DEV'}
    return identity,(config,index,entries,targets)
