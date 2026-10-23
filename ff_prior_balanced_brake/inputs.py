"""Frozen public task plus the verified same-scene actual terminal binding."""
import ast
import numpy as np
from ff_prior.storage import read_json,file_digest,digest
from ff_prior_pilot_train.storage import atomic_json,atomic_npz
from .protocol import ROOT,PACKAGE,SOURCE,DYNAMIC,P


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
    found.add(PACKAGE/'job.ps1');found.add(PACKAGE/'docs/EXPERIMENT_PLAN.md')
    return {p.relative_to(ROOT).as_posix():file_digest(p) for p in sorted(found)}


def freeze():
    import shutil
    assert not (PACKAGE/'evidence/contract.json').exists(),'FREEZE_ONLY_ONCE'
    control=read_json(ROOT/'ff_prior_fullchain/control.json')
    assert control['next_phase']=='D3_AUTHORIZED_BALANCED_BRAKE_FREEZE','REPAIR_SCOPE_NOT_READY'
    authorization=PACKAGE/'evidence/authorization.json';auth=read_json(authorization)
    assert auth['physical_trial_authorized'] and auth['run']=='balanced_brake_001'
    assert auth['candidate_denominator']==1 and not any(auth[k] for k in ('training','expansion','TEST','advantage_comparison'))
    static_sources=read_json(SOURCE/'run.json')['identity']['source_hashes']
    for path,expected in static_sources.items():
        assert file_digest(ROOT/path)==expected,'STATIC_FROZEN_SOURCE_CHANGED: '+path
    failed=ROOT/'artifacts/v7_1/balanced_tracking_001'
    for path,expected in read_json(failed/'run.json')['identity']['source_hashes'].items():
        assert file_digest(ROOT/path)==expected,'FAILED_FROZEN_SOURCE_CHANGED: '+path
    closed_path=ROOT/'ff_prior_fullchain/evidence/balanced_restore_closeout.json';closed=read_json(closed_path)
    assert closed['closed'] and (closed['model_valid'],closed['executed'],closed['replay_verified'])==(1,1,1)
    verdict=read_json(ROOT/closed['independent_review'])
    assert verdict['claims']['C1']['claim_supported']=='yes','STATIC_REVIEW_NOT_CONFIRMED'
    failed_closed_path=ROOT/'ff_prior_fullchain/evidence/balanced_tracking_closeout.json'
    failed_closed=read_json(failed_closed_path)
    assert failed_closed['closed'] and failed_closed['execution_status']=='EXECUTION_FAILED','FAILED_RUN_NOT_CLOSED'
    assert failed_closed['completed_resume_calls']==1
    old_package=ROOT/'ff_prior_balanced_tracking'
    old_binding=read_json(old_package/'evidence/binding.json')['files']
    # Hash the already-bound files; do not repeat closed-run arithmetic analysis.
    for path,expected in old_binding.items():
        assert file_digest(ROOT/path)==expected,'PREDECESSOR_BINDING_CHANGED: '+path
    contract=read_json(old_package/'evidence/contract.json')
    raw=dict(contract);assert digest({k:v for k,v in raw.items() if k!='contract_hash'})==raw['contract_hash']
    terminal_path=old_package/'evidence/terminal.npz'
    with np.load(terminal_path,allow_pickle=False) as a:
        assert a['integration'].shape==(836,)
        assert digest(a['integration'])==closed['terminal_state_digest']==contract['source_terminal_digest'],'STATIC_FULL_REPLAY_MISMATCH'
        assert abs(float(a['time'])-27.5)<1e-8
    layout=read_json(ROOT/'ff_prior_balanced_scene/evidence/layout.json')
    assert digest(contract['scene'])==digest(layout['scene']),'SEVEN_SCENE_IDENTITY_MISMATCH'
    assert contract['source_static_closeout']==closed_path.relative_to(ROOT).as_posix()
    original_contract_hash=contract.pop('contract_hash')
    contract.update(schema='balanced_same_scene_discrete_brake_repair_v1',
        controller='original_whole_body_feedback_intersected_with_discrete_braking_viability',
        original_dynamic_contract_hash=original_contract_hash,
        both_stages_sources_unchanged_since_static_run=False,
        dynamic_source_bundle_hash=digest(sources()),
        original_static_sources_unchanged=True,failed_dynamic_sources_unchanged=True,
        repair_authorization=authorization.relative_to(ROOT).as_posix(),
        failed_run_closeout=failed_closed_path.relative_to(ROOT).as_posix(),
        repair='new isolated controller stopping envelope; same actual static terminal, scene, seven waypoint timing and physical gates')
    contract['contract_hash']=digest(contract)
    shutil.copyfile(terminal_path,PACKAGE/'evidence/terminal.npz')
    assert file_digest(terminal_path)==file_digest(PACKAGE/'evidence/terminal.npz')
    atomic_json(PACKAGE/'evidence/contract.json',contract)
    files=dict(old_binding)
    for path in (PACKAGE/'evidence/terminal.npz',PACKAGE/'evidence/contract.json',authorization,
                 PACKAGE/'evidence/budget_derivation.json',failed_closed_path,ROOT/failed_closed['independent_review'],
                 ROOT/failed_closed['analysis'],failed/'run.json',failed/'execution/result.json'):
        files[path.relative_to(ROOT).as_posix()]=file_digest(path)
    atomic_json(PACKAGE/'evidence/binding.json',dict(files=files))
    return dict(inputs=len(files),contract_hash=contract['contract_hash'],extra_coupled=0,extra_native_steps=0)


def identity():
    binding=read_json(PACKAGE/'evidence/binding.json')
    for path,h in binding['files'].items():
        if file_digest(ROOT/path)!=h:raise ValueError('FROZEN_INPUT_CHANGED: '+path)
    c=read_json(PACKAGE/'evidence/contract.json');raw=dict(c);given=raw.pop('contract_hash')
    assert digest(raw)==given
    return dict(protocol=P,source_hashes=sources(),binding=binding,contract=c)


if __name__=='__main__':
    import sys,json
    print(json.dumps(freeze() if sys.argv[-1]=='freeze' else identity(),ensure_ascii=True))
