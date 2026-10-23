"""Record the user's explicit repair scope and prepare isolated binding code."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from ff_prior.storage import read_json
from ff_prior_pilot_train.storage import atomic_json
p=ROOT/'ff_prior_balanced_brake'
auth=p/'evidence/authorization.json'
assert not auth.exists(),'AUTH_ALREADY_PREPARED'
atomic_json(auth,dict(date='2026-10-23',user_request='后两点及最终保持未完成检查原因并修正',
    physical_trial_authorized=True,run='balanced_brake_001',candidate_denominator=1,
    predecessor='balanced_restore_001',failed_run_preserved='balanced_tracking_001',
    mechanism='proactive discrete stopping viability intersected with every original bound',
    original_scene_waypoints_timing_gates_preserved=True,training=False,expansion=False,TEST=False,
    advantage_comparison=False,new_neural_sampling=False,repair_authorization_resets_blocking_audit=True))
atomic_json(p/'evidence/budget_derivation.json',dict(
    previous_D3=dict(coupled=47127,native_including_replay=74660,wall_s=3315.796,core_bytes=2221774427),
    previous_failed_dynamic=dict(native_including_replay=19660,remaining_old_allowance=7340,
        old_allowance_reused=False),
    new_repair=dict(coupled_limit=2000,planned_coupled=0,native_including_replay=27000,
        native=13500,replay=13500,wall_s=1200,bytes=2*1024**3,executions=1),
    readonly_final_media=dict(wall_s=600,bytes=512*1024**2,native=0,coupled=0),
    D3_after_caps=dict(coupled=49127,native_including_replay=101660,wall_s=5115.796,
        bytes=2221774427+int(2.5*1024**3)),
    original_D3_caps=dict(coupled=60000,native_including_replay=165000,wall_s=21540,bytes=20*1024**3),
    conservative_fullcycle_after=dict(coupled=69127,native_including_replay=249160,
        wall_s=26715.796+30.160764217+4.895535469+48.343,
        bytes=20*1024**3+2221774427+int(2.5*1024**3)+15*1024**3),
    storage_note='Conservative ledger includes D1/D2 20GiB, all current D3 core bytes, new2.5GiB and additional15GiB prior reserves; below40GiB.',
    original_fullcycle_caps=dict(coupled=80000,native_including_replay=312500,wall_s=43200,bytes=40*1024**3),
    scope='New explicit repair allocation from original cycle remainder, not restoration of spent dynamic allowance'))
path=p/'inputs.py';s=path.read_text(encoding='utf-8')
start=s.index('def freeze():');end=s.index('\ndef identity():')
freeze='''def freeze():
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

'''
path.write_text(s[:start]+freeze+s[end:],encoding='utf-8')
test=p/'test_workflow.py';s=test.read_text(encoding='utf-8')
s=s.replace("'STATIC_FROZEN_SOURCE_CHANGED','DYNAMIC_SOURCE_NOT_IN_STATIC_BUNDLE',\n                         'DYNAMIC_SOURCE_DIFFERS_FROM_STATIC_BUNDLE',\n                         'D3_AUTHORIZED_BALANCED_TERMINAL_TRACKING_FREEZE'", "'STATIC_FROZEN_SOURCE_CHANGED','FAILED_FROZEN_SOURCE_CHANGED',\n                         'FAILED_RUN_NOT_CLOSED',\n                         'D3_AUTHORIZED_BALANCED_BRAKE_FREEZE'")
test.write_text(s,encoding='utf-8')
control=ROOT/'ff_prior_fullchain/control.json';c=read_json(control)
c.update(current_phase='D3_balanced_2R5C_brake_implementation',next_phase='D3_AUTHORIZED_BALANCED_BRAKE_FREEZE',
    next_phase_action='Finish failed-run closeout, implement exact discrete braking, freeze one isolated repair and launch one worker.',
    goal_status='ACTIVE_NEW_EXPLICIT_BRAKE_REPAIR_AUTHORIZED',authorization_evidence=auth.relative_to(ROOT).as_posix(),
    current_user_steering=auth.relative_to(ROOT).as_posix(),additional_local_repair_allowed=False,
    app_goal_transition='NEW_EXPLICIT_BRAKE_REPAIR_RESETS_BLOCKING_AUDIT_20261023')
atomic_json(control,c)
goal=ROOT/'DIFFUSION_GOAL_CURRENT.md';old=goal.read_text(encoding='utf-8-sig')
goal.write_text('# 2026-10-23 最新明确授权：修正后两点与最终保持\n\n用户要求“后两点及最终保持未完成检查原因并修正”。失败balanced_tracking_001保留并收尾；新独立balanced_brake_001修复离散制动可行性，从已验证同场景真实静态末态起步，保持原7点/时序/全部门。冻结一轮1200秒/2GiB/27000原生含完整重放，计划0耦合，单worker。旧等待或禁止额外运行文字不能覆盖本次明确修复授权。授权证据ff_prior_balanced_brake/evidence/authorization.json；计划ff_prior_balanced_brake/docs/EXPERIMENT_PLAN.md。长期目标不标complete/paused。\n\n'+old,encoding='utf-8')
print('Prepared isolated repair scope and binding code; no physics.')
