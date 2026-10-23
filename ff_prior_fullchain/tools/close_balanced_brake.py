"""Seal the independently reviewed braking repair; no further physical work."""
from pathlib import Path
import hashlib,subprocess,sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from ff_prior.storage import read_json,file_digest
from ff_prior_pilot_train.storage import atomic_json
out=ROOT/'ff_prior_fullchain/evidence/balanced_brake_closeout.json'
assert not out.exists(),'CLOSE_ONLY_ONCE'
review='paper/review-traces/experiment-result-to-claim/20261023_balanced_brake/verdict.json'
v=read_json(ROOT/review)
assert v['claims']['C1']['claim_supported']=='yes','REVIEW_MUST_CONFIRM_ENGINEERING_SCOPE'
resume='ff_prior_balanced_brake/evidence/completed_001/receipt.json';r=read_json(ROOT/resume)
assert r['status']=='PASS' and r['complete_manifest_hash_size_unchanged'] and r['exactly_one_invocation']
assert r['response']['resume']=='NO_WORK_ALREADY_COMPLETE'
result=read_json(ROOT/'artifacts/v7_1/balanced_brake_001/execution/result.json')
assert result['status']=='EXECUTION_VERIFIED' and all(result['gates'].values())
assert result['steps']==13500 and result['replay_completed_steps']==13500
assert result['replay_full_integration_error']==0
media='ff_prior_balanced_delivery/evidence/brake_media_check.json';m=read_json(ROOT/media)
assert m['status']=='PASS_LOCAL_ARTIFACT_CHECKS' and m['listed_source_files_unchanged']
assert m['videos_frames']==[819,819] and m['video_duration_s']==[54.5,54.5]
analysis='ff_prior_balanced_brake/docs/ANALYSIS.md'
commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
assert subprocess.check_output(['git','show',commit+':'+analysis],cwd=ROOT).decode('utf-8').replace('\r\n','\n')==(ROOT/analysis).read_text(encoding='utf-8').replace('\r\n','\n')
for name in ('stdout','stderr'):
    path=ROOT/'ff_prior_balanced_brake/evidence/completed_001'/f'{name}.bin'
    assert file_digest(path)==r[name]['sha256']
    assert hashlib.sha256(subprocess.check_output(['git','show',commit+':'+path.relative_to(ROOT).as_posix()],cwd=ROOT)).hexdigest()==r[name]['sha256']
atomic_json(out,dict(closed=True,run='artifacts/v7_1/balanced_brake_001',commit=commit,
    producer_git_head='0def32650ed74718dbff43476f257dcd70c9ff49',source_files=189,bound_inputs=265,
    analysis=analysis,independent_review=review,execution_status='EXECUTION_VERIFIED',
    completed_resume=resume,completed_resume_calls=1,core_files=r['core_files'],core_bytes=r['core_bytes'],
    native_steps=13500,replay_steps=13500,coupled_calls=0,elapsed_s=103.563,
    actual_handoff=True,full836_replay_exact=True,full_dynamic_pass=True,
    waypoint_arrivals=7,waypoint_denominator=7,final_hold_s=1.5,
    final_media='ff_prior_balanced_delivery/final_brake_001/index.html',media_verification=media,
    full_chain_duration_s=54.5,media_url='http://127.0.0.1:8771/index.html',
    browser_ui_verified=False,browser_ui_limitation='Origin approval gate did not authorize automatic opening; local artifact checks completed without bypass.',
    paper_writing_provisional=True,new_physics_during_closeout=False,long_term_goal_complete=False,
    physical_satellite_drift=False,network_advantage_claim=False,
    D3_cumulative=dict(coupled=47127,native_including_replay=101660,worker_wall_s=3419.359,core_bytes=2843483028)))
control=ROOT/'ff_prior_fullchain/control.json';c=read_json(control)
c.update(active_worker=False,active_run_role='most_recent_closed_run',
    current_phase='D3_balanced_2R5C_brake_closed_verified',
    next_phase='BALANCED_2R5C_FULLCHAIN_DELIVERED_NO_AUTOMATIC_PHYSICS',
    next_phase_action='Current repair scope delivered and permanently closed. No repeated audits/resume/physical work; long-term physical moving-satellite origin remains unresolved.',
    goal_status='CURRENT_REPAIR_DELIVERED_LONG_TERM_INCOMPLETE')
c['phases'][-1]['status']='STATIC_AND_REPAIRED_DYNAMIC_EXECUTION_REPLAY_VERIFIED_MEDIA_DELIVERED'
atomic_json(control,c)
goal=ROOT/'DIFFUSION_GOAL_CURRENT.md';s=goal.read_text(encoding='utf-8-sig')
s='# 2026-10-23 修复单元已通过并永久收尾\n\nbalanced_brake_001已完成原2R5C同场景真实静态末态接27秒动态，七点7/7及最后1.5秒保持，19门通过，13500执行+13500独立重放，13501×836差0。新独立审查确认，收据ff_prior_fullchain/evidence/balanced_brake_closeout.json。一次完成resume已PASS，81文件621708601bytes不变，禁止再resume/审查/追加物理。完整54.5秒视频/三维/坐标系/末端轨迹/臂身曲线交付http://127.0.0.1:8771/index.html；原失败run保留。实体卫星零速继承，长期物理一致移动起源仍未完成；不把长期goal设complete/paused，不自动新轮次。\n\n'+s
goal.write_text(s,encoding='utf-8')
print(out)
