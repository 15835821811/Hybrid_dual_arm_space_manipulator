"""Bind the completed training closure and preserve its negative quality result."""
from pathlib import Path
import numpy as np
from ff_prior.storage import read_json,file_digest,digest
from ff_prior.recovery import process_is_alive
from ff_prior_origin_learn_bounded.experiment import identity_for as parent_identity,sources as parent_sources
from ff_prior_generalization.data import guard_population
from .protocol import ROOT,SOURCE,PROTOCOL as P


def sources():
    return dict(parent_sources(),**{p.relative_to(ROOT).as_posix():file_digest(p)
                                  for p in sorted((ROOT/'ff_prior_origin_batch_next').rglob('*.py'))})


def require_parent(source):
    source=Path(source).resolve()
    if source!=SOURCE.resolve():raise ValueError('exact completed 13774 training parent required')
    state=read_json(source/'status.json');summary=read_json(source/'summary.json')
    if state['status']!='FINISHED' or process_is_alive(state['pid']):
        raise ValueError('parent training must complete and exit')
    if not summary['origin_training_closure_passed'] or not summary['independent_audit_passed']:
        raise ValueError('C1 closure and raw audit required')
    if read_json(source/'independent_verification.json')['status']!='PASS':
        raise ValueError('raw audit required')
    if summary['errors'] or summary['unresolved_calls']:raise ValueError('unresolved parent work')
    # C2 is evidence to retain, not an expansion permission gate.
    return summary


def identity_for(source):
    source=Path(source).resolve();summary=require_parent(source)
    frozen=read_json(source/'run.json')['identity']
    expected,context=parent_identity(frozen['source'])
    if frozen!=expected:raise ValueError('parent frozen identity changed')
    config,index,_,used_validation,entries=context
    if sum(e['split']=='TRAIN' for e in index)!=P['old_TRAIN'] or sum(e['split']=='VAL' for e in index)!=48:
        raise ValueError('complete inherited population required')
    files=dict(frozen['source_files'])
    for p in sorted(source.rglob('*')):
        if p.is_file() and p.name not in ('run.lock','launch.guard') and p.suffix!='.tmp':
            files[p.relative_to(ROOT).as_posix()]=file_digest(p)
    dev=entries[:4]
    guard=guard_population(index,dev)
    protected=read_json(ROOT/'artifacts/v7_1/independent_new_validation_recovery_001/frozen_tasks.json')['entries']
    protected+=read_json(ROOT/'artifacts/v7_1/qualified_origin_train_001/frozen_tasks.json')['entries'][4:]
    protected+=entries[4:]
    protected=list({e['task']['task_hash']:e for e in protected}.values())
    heldout_initials=[]
    original_origins=set()
    for e in index:
        row=read_json(ROOT/e['record']);q=np.asarray(row['condition']['initial_features'])
        original_origins.add(digest(q))
        if e['split']!='TRAIN':heldout_initials.append(q)
    for j,e in enumerate(dev+protected):
        q=np.asarray(e['task']['condition']['initial_features'])
        original_origins.add(digest(q));heldout_initials.append(q)
        if j>=4:
            guard.append({'sample_id':f'PROTECTED_USED_DEV_{j}', 'split':'DEV',
                'family_id':e.get('family_id',e['sample_id']), 'goals':e['task']['condition']['goals']})
    heldout=[e for e in guard if e['split']!='TRAIN']
    preservation=read_json(ROOT/'ff_prior_expand/evidence/preservation_before.json')
    identity={'protocol':P,'source':str(source),'source_files':files,'source_hashes':sources(),
        'parent_C1':summary['origin_training_closure_passed'],
        'parent_C2':summary['pre_registered_quality_claim_passed'],
        'original_DEV_hash':digest(dev),'guard_hash':digest(guard),
        'used_validation_hash':digest(used_validation),'preservation':preservation}
    return identity,(config,dev,index,guard,heldout,original_origins,heldout_initials)
