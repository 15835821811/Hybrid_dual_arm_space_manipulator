"""Bind only the sealed predecessor identity and its final state; no old audit."""
import ast
import numpy as np
from ff_prior.storage import read_json,file_digest,digest
from ff_prior_pilot_train.storage import atomic_json,atomic_npz
from .protocol import ROOT,PACKAGE,SOURCE,P

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
    assert not (PACKAGE/'evidence/contract.json').exists(),'FREEZE_ONCE'
    control=read_json(ROOT/'ff_prior_fullchain/control.json')
    assert control['next_phase']=='D3_AUTHORIZED_SATELLITE_ORIGIN_FREEZE'
    auth=read_json(PACKAGE/'evidence/authorization.json')
    assert auth['physical_trial_authorized'] and auth['physical_origin_mechanism_approved']
    assert auth['run']==P_RUN and auth['candidate_denominator']==1
    receipt_path=ROOT/'ff_prior_fullchain/evidence/balanced_brake_closeout.json'
    receipt=read_json(receipt_path)
    assert receipt['closed'] and receipt['full_dynamic_pass'] and receipt['full836_replay_exact']
    assert receipt['waypoint_arrivals']==7 and receipt['final_hold_s']==1.5
    old=read_json(ROOT/'ff_prior_balanced_brake/evidence/contract.json')
    raw=dict(old);h=raw.pop('contract_hash');assert digest(raw)==h
    native=SOURCE/'execution/native_trace.npz';result=read_json(SOURCE/'execution/result.json')
    assert result['status']=='EXECUTION_VERIFIED' and file_digest(native)==result['native_trace_sha256']
    for path,expected in read_json(SOURCE/'run.json')['identity']['source_hashes'].items():
        assert file_digest(ROOT/path)==expected,'PREDECESSOR_SOURCE_CHANGED: '+path
    with np.load(native,allow_pickle=False) as a:
        terminal={k:a[k][-1].copy() for k in ('integration','qpos','qvel','time')}
        terminal.update(reference_q=a['qref'][-1].copy(),reference_qd=a['reference_qd'][-1].copy())
        desired=a['desired'][-1].copy();index=len(a['time'])-1
    assert terminal['integration'].shape==(836,) and abs(float(terminal['time'])-54.5)<1e-8
    assert index==13500
    np.testing.assert_allclose(desired[0,:3,3],old['waypoints']['waypoint_points_m'][-1],atol=1e-14)
    c={k:old[k] for k in ('scene','scene_model_hash','waypoints','static_goals','grasp_local',
                         'grasp_rotation_local','original_waypoints_sha256','target_velocity','network_role')}
    c.update(schema='physical_satellite_origin_v1',split='APP_DEV',
        predecessor_contract_hash=h,source_native_sha256=file_digest(native),source_terminal_index=index,
        source_terminal_digest=digest(terminal['integration']),absolute_start_time=float(terminal['time']),
        hold_continuum_pose=desired[0],initial_desired=desired,source_closeout=receipt_path.relative_to(ROOT).as_posix(),
        no_new_neural_sampling=True,moving_satellite_claim='SUBJECT_TO_NEW_EXECUTION_AND_REPLAY',
        force_contract=P,continuum_reference='hold already-completed W7; never restart original seven points',
        target_velocity_changed_by_initialization=False,source_state_mask='mjSTATE_INTEGRATION')
    c['contract_hash']=digest(c)
    atomic_npz(PACKAGE/'evidence/terminal.npz',**terminal)
    atomic_json(PACKAGE/'evidence/contract.json',c)
    # Existing immutable input hashes are used as provenance, not reanalysis.
    files=dict(read_json(ROOT/'ff_prior_balanced_brake/evidence/binding.json')['files'])
    paths=[native,SOURCE/'execution/result.json',SOURCE/'run.json',receipt_path,
        ROOT/'ff_prior_balanced_brake/evidence/contract.json',
        ROOT/'ff_prior_moving_origin/evidence/remaining_budget.json']
    paths += list((PACKAGE/'evidence').glob('*.json'))+[PACKAGE/'evidence/terminal.npz']
    for path in paths:files[path.relative_to(ROOT).as_posix()]=file_digest(path)
    atomic_json(PACKAGE/'evidence/binding.json',dict(files=files))
    return dict(inputs=len(files),source_files=len(sources()),contract_hash=c['contract_hash'],
                terminal_digest=c['source_terminal_digest'],absolute_time=c['absolute_start_time'],
                new_native_steps=0,new_coupled=0)

P_RUN='moving_satellite_001'
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
