"""Persist bounded static findings and route latest scene steering; no physics."""
import json,datetime
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
assert not (ROOT/'ff_prior_horizon_restore/evidence/closeout_statistics.json').exists(),'CLOSEOUT_METADATA_ALREADY_WRITTEN'
def read(p):return json.loads((ROOT/p).read_text(encoding='utf-8-sig'))
def put(p,v):
    path=ROOT/p;path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
v=read('paper/review-traces/experiment-result-to-claim/20261023_horizon_restore/verdict.json')
assert v['claims']['C1']['claim_supported']=='yes' and v['claims']['Caux']['claim_supported']=='yes'
resume=read('ff_prior_horizon_restore/evidence/completed_resume.json')
assert resume['status']=='PASS' and resume['completed_invocations']==1
stats=dict(run='horizon_restore_001',candidate_denominator=1,model_valid=1,execution_verified=1,replay_verified=1,
    actual_coupled=2681,reserved_model_intervals=1987216,native_steps=13750,replay_steps=13750,
    worker_elapsed_s=468.375,iterations=8,core_files=127,core_bytes=458379716,
    checks=v['checks'],claims=v['claims'],D3_used=dict(coupled=46454,worker_s=2458.484,native_including_replay=27500,core_bytes=1224791218),
    actual_terminal_digest=v['routing']['terminal_state_digest'],new_neural_samples=0,training_updates=0,TEST=False,
    media='http://127.0.0.1:8769/index.html',once_only_resume=True,
    producer_git_head='bfc4ea9c436a32bb5fc9aa49db4e66857e11283f',source_identity='142 actual hashes plus executed_source.zip; one pre-existing archived working-tree difference')
put('ff_prior_horizon_restore/evidence/closeout_statistics.json',stats)
put('ff_prior_balanced_scene/evidence/user_layout_selection.json',dict(date='2026-10-23',
    request='你刚性臂这一侧没必要放这么多障碍物吧，而且太紧凑了，连续体一侧障碍物又太少，且为啥不是那个不规则航点，且障碍物不止关注机械臂末端，也要涉及对臂形的影响吧',
    question='下一场景的障碍布局采用哪种？两种都保留原七个不规则航点，并覆盖臂身避障；选择后我先给出具体布局和预算，现有结果单独保留。',
    answer='重新分布：刚性侧2个疏散障碍，连续体侧5个覆盖末端路径与中段臂身（推荐）',
    scope='concrete scene design and budget; not a hidden second run under the completed original-five unit',
    new_actual_run_scope_pending=True))
put('ff_prior_balanced_scene/evidence/budget_proposal.json',dict(status='PROPOSAL_NOT_RUN_PERMISSION',
    D3_used=stats['D3_used'],restore_iterations=28,static_coupled_cap=11000,dynamic_coupled_cap=2000,
    restore_plus_validation_calls_worst=28*366+6+7,model_intervals_worst=28*946016+32256+32512,
    model_intervals_cap=29000000,static_wall_s=15000,restore_wall_s=13200,dynamic_wall_s=1200,
    static_native_replay=27500,dynamic_native_replay=27000,static_bytes_cap=12*1024**3,dynamic_bytes_cap=2*1024**3,
    closeout_reserve_bytes=1024**3,D3_conservative=dict(coupled=59454,wall_s=18658.484,native_replay=82000,
    bytes=1224791218+15*1024**3),cycle_conservative=dict(coupled=79454,wall_s=40258.484,native_replay=229500,bytes=1224791218+35*1024**3),
    native_steps_executed_for_proposal=0,coupled_executed_for_proposal=0,fixed_initial_geometry_checks=1))
stamp=datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
track='''# horizon_restore_001 已收尾 / 双侧布局待确认新运行范围

|单元|固定分母|状态|证据|
|---|---:|---|---|
|horizon_restore_001/u00 原五障碍|1|CLOSED_EXECUTION_VERIFIED|MODEL_VALID、67力矩执行、完整836重放全通过；C1/Caux yes/high；一次完成resume PASS|
|原五障碍动态 horizon_tracking_001|未冻结|NOT_STARTED_SUPERSEDED_BY_LAYOUT_STEERING|实现草稿保留；未启动物理，不偷换场景|
|刚性2 / 连续体5新布局|布局选择1|DESIGN_READY_NEW_RUN_SCOPE_PENDING|原7航点、臂身约束、一次固定初态几何检查；新物理0|

准确分析：ff_prior_horizon_restore/docs/ANALYSIS.md。新布局与预算：ff_prior_balanced_scene/docs/LAYOUT_AND_BUDGET.md。
原五障碍Git基点bfc4ea9加142实际源码hash/zip及233输入为身份；保留一个用户已有hierarchical_qp归档差异，非纯clean-tree提交。
2681耦合、1987216模型区间、27500原生含重放、468.375秒。2249检查中2248直接通过，1项Git差异解释，0未决。
旧run禁止再审查/resume/物理；新布局不能在旧末态中变更障碍后拼接。长期目标未完成，无训练/扩库/对照/TEST。
'''
for name in ('paper/EXPERIMENT_TRACKER_'+stamp+'_horizon_closeout.md','paper/EXPERIMENT_TRACKER_20261023_horizon_restore.md'):
    (ROOT/name).write_text(track,encoding='utf-8')
findings='''\n\n## Research Findings — horizon_restore_001 / 2026-10-23\n\nC1=yes/high and Caux=yes/high from fresh independent reviewer; paper writing provisional without paper claim audit. One original-five APP_DEV candidate MODEL_VALID/executed/replayed. 13751×836 full-state replay exact zero; 2681 coupled,1987216 reserved model intervals,27500 native including replay,468.375s. Sole completed resume PASS,127 core files/458379716bytes unchanged. Source identity is Git base bfc4ea9 plus142 actual hashes/zip and233 inputs; the prior hierarchical_qp working-tree difference is pre-frozen and not invoked.2249 checks=2248 pass+1 explained HEAD difference,0 unresolved.\n\nBounded static system claim only, not bare network/advantage/generalization/dynamic/physical satellite drift. Missing physical intermediate arrays/source-only boundaries are retained in the fresh review. Routing=confirm static; latest user selects rigid2/continuum5 scene with original7 waypoints and body-shape coverage. New complete-scene run scope pending concrete layout/cost review; no hidden obstacle swap in actual old terminal state. Old static run permanently closed; no re-audit/resume/physics.\n'''
with (ROOT/'findings.md').open('a',encoding='utf-8') as f:f.write(findings)
with (ROOT/'MANIFEST.md').open('a',encoding='utf-8') as f:
    for skill,path,desc in [('experiment-result-to-claim','ff_prior_horizon_restore/docs/ANALYSIS.md','原五障碍单元收尾，C1/Caux yes/high，保留证据边界'),('experiment-result-to-claim','paper/EXPERIMENT_TRACKER_'+stamp+'_horizon_closeout.md','当前收尾与新布局路由'),('experiment-plan','ff_prior_balanced_scene/docs/LAYOUT_AND_BUDGET.md','2+5布局与新增完整初态单元预算草案')]:
        f.write(f'\n| {stamp} | /{skill} | {path} | experiment | {desc} |\n')
control=read('ff_prior_fullchain/control.json')
control.update(current_phase='D3_horizon_restore_closed_verified',active_run_role='most_recent_closed_run',active_worker=False,
    next_phase='BALANCED_2R5C_LAYOUT_READY_NEW_FULLCHAIN_RUN_SCOPE_PENDING',
    decision_document='ff_prior_balanced_scene/docs/LAYOUT_AND_BUDGET.md',
    implementation_handoff='ff_prior_balanced_scene/docs/LAYOUT_AND_BUDGET.md',
    next_phase_action='Original-five static unit closed. User selected new 2R5C layout; concrete proposal ready, no automatic second static unit or old-scene dynamic launch.',
    current_user_steering='ff_prior_balanced_scene/evidence/user_layout_selection.json')
for p in control['phases']:
    if p['id']=='D3_original_five_obstacles':p['status']='STATIC_CLOSED_EXECUTION_VERIFIED_DYNAMIC_NOT_RUN'
put('ff_prior_fullchain/control.json',control)
print(json.dumps(dict(tracker='paper/EXPERIMENT_TRACKER_'+stamp+'_horizon_closeout.md',new_physics=0),ensure_ascii=False))
