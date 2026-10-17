from pathlib import Path
import numpy as np
from ff_prior.storage import read_json,file_digest,digest
from ff_prior.recovery import process_is_alive
from ff_prior_origin_learn_bounded.experiment import identity_for as parent_identity,sources as parent_sources
from .protocol import ROOT,SOURCE,PROTOCOL as P

def sources():
    return dict(parent_sources(),**{p.relative_to(ROOT).as_posix():file_digest(p)
        for p in sorted((ROOT/'ff_prior_world_endpoint').rglob('*.py'))})

def select_targets(index):
    pools={False:[],True:[]}
    for e in index:
        if e['split']!='TRAIN' or e.get('producer_run_id')!='qualified_origin_batch_001':continue
        row=read_json(ROOT/e['record'])
        if row.get('data_role')=='INDEPENDENT_ORIGIN_TRAIN':pools[bool(row['scene']['objects'])].append((e,row))
    chosen=[]
    for obstacle in (False,True):
        chosen+=sorted(pools[obstacle],key=lambda pair:digest({'endpoint_target':pair[0]['sample_id']}))[:8]
    if len(chosen)!=16:raise ValueError('sixteen qualified TRAIN targets required')
    result=[]
    for slot,(e,r) in enumerate(chosen):
        with np.load(ROOT/e['arrays']) as a:z=a['Z'].copy()
        result.append({'sample_id':f'endpoint_mode_{slot:02d}','slot':slot,'split':'TRAIN',
            'parent_sample_id':r['sample_id'],'source_record':e['record'],'source_arrays':e['arrays'],
            'family_id':r['family_id'],'task':r['task'],'scene':r['scene'],
            'context_id':r['context_id'],'origin_hash':r['origin_hash'],'Z':z,
            'frozen_before_mode_generation':True})
    return result

def identity_for(source):
    source=Path(source).resolve()
    if source!=SOURCE.resolve():raise ValueError('exact13774 completed parent required')
    s=read_json(source/'status.json');summary=read_json(source/'summary.json')
    if s['status']!='FINISHED' or process_is_alive(s['pid']):raise ValueError('parent must complete and exit')
    if not summary['origin_training_closure_passed'] or not summary['independent_audit_passed']:
        raise ValueError('verified parent C1 required')
    old=read_json(source/'run.json')['identity'];expected,context=parent_identity(old['source'])
    if old!=expected:raise ValueError('parent frozen identity changed')
    config,index,_,_,entries=context
    if sum(e['split']=='TRAIN' for e in index)!=13774:raise ValueError('13774 TRAIN required')
    targets=select_targets(index)
    if {e['task']['task_hash'] for e in targets}&{e['task']['task_hash'] for e in entries}:raise ValueError('TRAIN/DEV overlap')
    files=dict(old['source_files'])
    for p in sorted(source.rglob('*')):
        if p.is_file() and p.name not in ('run.lock','launch.guard') and p.suffix!='.tmp':
            files[p.relative_to(ROOT).as_posix()]=file_digest(p)
    identity={'protocol':P,'source':str(source),'source_files':files,'source_hashes':sources(),
              'targets_hash':digest(targets),'comparison_entries_hash':digest(entries),
              'parent_C1':True,'parent_C2':summary['pre_registered_quality_claim_passed'],
              'used_validation_role':'REUSED_DEV; no unseen-generalization claim'}
    return identity,(config,index,entries,targets)
