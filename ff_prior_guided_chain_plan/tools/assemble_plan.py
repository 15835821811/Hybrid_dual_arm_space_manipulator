"""Assemble this planning delivery only. No model forward or physics imports."""
from pathlib import Path
from datetime import datetime
import hashlib, json, shutil

ROOT=Path(__file__).resolve().parents[2]
PKG=ROOT/'ff_prior_guided_chain_plan'
STAMP='20261026_113732'
def read(rel): return json.loads((ROOT/rel).read_text(encoding='utf-8-sig'))
def write(rel,obj):
    p=ROOT/rel
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()

b=read('ff_prior_guided_chain_plan/evidence/budget_derivation.json')
scope=read('ff_prior_guided_chain_plan/evidence/scope.json')
plan=(PKG/'docs/EXPERIMENT_PLAN.md').read_text(encoding='utf-8')
tracker="""# 冻结权重 Diffusion 连续链工程跟踪

更新时间：2026-10-26。当前无新物理 worker；长期目标未完成。

|阶段|状态|交付/依赖|
|---|---|---|
|旧 2R5C 静态＋七航点修复|封存，原阶段执行和重放已验证|不复审、不 resume，原报告保留|
|旧实体卫星起源＋追踪|封存，原机器人动量门失败，重放一致|重放相同不能推翻物理失败|
|动量根因|既有诊断完成，RK4 修复未实测|原0秒重建；不扣初值/注速/重置|
|G0 冻结模型复用检查|完成，11项结构/身份检查通过|只读元数据，0新采样/耦合/原生|
|G0 具体方案和预算|完成；新增资源未获批|建议追加30000耦合、50000原生、11小时、7.5GiB|
|G1 生产接口与引导适配|未实现|完整初态预测器、边界QP、DDIM回注、RK4与记录|
|G2 核算与冻结|未冻结|新额度落实、软件检查、源/输入hash与单worker锁|
|G3 新A段|未运行|27.5秒静态全门＋完整重放＋必要新审查|
|G4 新B段|依赖G3|新实际末态→原七点/保持27秒|
|G5 新C段|依赖G4|新实际54.5秒→2秒公开推力＋关机27秒|
|G6 可视化与收尾|依赖新实际数据|坐标系/轨迹/臂身/动量/推力/全状态重放，失败段标BLOCKED|

本方案替代固定旧Z的动量重建提案，不追加两次重建。唯一完整链候选，固定三个阶段seed；无训练、扩库、优势对照或TEST。冻结神经先验、物理引导、控制反馈贡献分开，不设置网络优势门。现有媒体不冒充新方案结果。旧定时任务保持删除，App旧goal卡不伪造complete/paused替换。

详细方案：ff_prior_guided_chain_plan/docs/EXPERIMENT_PLAN.md
复用证据：ff_prior_guided_chain_plan/evidence/model_reuse.json
完整预算：ff_prior_guided_chain_plan/evidence/budget_derivation.json
"""
# Preserve prior latest documents before replacing, then write timestamped versions first.
outputs=[]
for name,content in [('EXPERIMENT_PLAN',plan),('EXPERIMENT_TRACKER',tracker)]:
    latest=ROOT/f'paper/{name}.md'
    backup=PKG/f'history/{name}_before_{STAMP}.md'
    backup.parent.mkdir(parents=True,exist_ok=True)
    if latest.exists() and not backup.exists():
        shutil.copyfile(latest,backup); outputs.append(backup.relative_to(ROOT).as_posix())
    version=ROOT/f'paper/{name}_{STAMP}.md'
    version.write_text(content,encoding='utf-8')
    shutil.copyfile(version,latest)
    outputs += [version.relative_to(ROOT).as_posix(),latest.relative_to(ROOT).as_posix()]
for name,content in [('EXPERIMENT_PLAN',plan),('EXPERIMENT_TRACKER',tracker)]:
    version=PKG/f'docs/{name}_{STAMP}.md'
    version.write_text(content,encoding='utf-8')
    shutil.copyfile(version,PKG/f'docs/{name}.md')
    outputs += [version.relative_to(ROOT).as_posix(),(PKG/f'docs/{name}.md').relative_to(ROOT).as_posix()]

spec={
 'status':'PROPOSAL_READY_PRODUCTION_NOT_IMPLEMENTED_BUDGET_EXTENSION_PENDING',
 'unit':'guided_diffusion_chain_001','module_proposed':'ff_prior_guided_chain.job',
 'source_base_commit':scope['source_base_commit'],'plan':'ff_prior_guided_chain_plan/docs/EXPERIMENT_PLAN.md',
 'existing_model_reuse':'frozen static102 prior plus deterministic221 details; frozen temporal271x17 prior',
 'production_implemented':False,'frozen':False,'worker_started':False,
 'candidate_chains':1,'candidates_per_phase':1,'retries':0,
 'phases':b['phases'],'ddim_steps':16,'guidance_steps_zero_based':[12,13,14,15],
 'reference_contracts':['ff_prior_balanced_scene/evidence/layout.json','ff_prior_balanced_brake/evidence/contract.json','ff_prior_satellite_origin/evidence/contract.json'],
 'original_initial':'ff_prior_balanced_restore/evidence/initial_state.npz',
 'original_initial_sha256':'5ad6c16207374d7368f2cff96d7e4ad7bd6c4aa92cc35e20aa2db451c28e91e6',
 'integrator_proposed':'mjINT_RK4','native_dt':0.002,
 'physical_model_allowed_difference':['opt.integrator'],'physics_gate_changes':[],
 'reference_changes':'New neural-guided joint trajectories; original public world goals, seven points, timing, holds and C2 reference unchanged.',
 'actual_initializations':1,'replay_initializations':1,'actual_phase_resets':0,
 'state_injection':False,'subtract_initial_momentum':False,'force_exception':'Only existing public2s satelliteCOM thrust, thenzero27s; robot external0',
 'sampler':'NN clean trajectory -> motion/boundary QP -> floating propagation -> full-machine guidance -> recode -> DDIM transition.',
 'predictor':'New inherited-momentum full-state propagation; no mj_step in planner. All native calls would be counted.',
 'fd_dimensions':[323,4590,4590],'fd_h_rad':0.0001,
 'max_constraint_rows':4044,'geometry_selected_cap':512,'torque_selected_cap':256,
 'solver_maxiter':180,'line_search':[1.0,0.5,0.25,0.125],
 'all_pair_gate_required':True,'initial_phase_windows_unchanged':True,
 'stage_release':'Model gates, execution gates, whole836 replay and necessary new review before next phase; same actual/replay objects.',
 'freeze_requires':['additional_budget_resolved','production_implementation','software_and_counter_checks','source_zip_and_input_hashes','identical_original_initial_values','equal_physical_arrays_except_integrator','full_cost_and_record_envelope','single_worker_lock'],
 'budget':'ff_prior_guided_chain_plan/evidence/budget_derivation.json',
 'success_unknown':True,'training':False,'expansion':False,'TEST':False,'advantage_comparison':False,
 'preserve_old_sealed_runs':True
}
write('ff_prior_guided_chain_plan/evidence/implementation_spec.json',spec)
goal={
 'status':'EXECUTION_PLAN_ONLY_BUDGET_EXTENSION_PENDING','objective':'One new frozen-weight Diffusion-guided full trajectory chain from original0s initial through2R5C seven irregular waypoints to physical satellite2s thrust and27s coast tracking, unchanged gates and836 replay.',
 'steps':[{'id':f'G{i}','state':('COMPLETE_PLAN_ONLY' if i==0 else 'NOT_RUN'),'details':x} for i,x in enumerate(['Reuse identity and plan/budget','Implement adapters/predictor/guidance','Cost checks and freeze after budget resolution','A static stage','B seven-waypoint stage','C physical satellite origin/coast','Evidence/review/all visuals/closeout'])],
 'plan':spec['plan'],'budget':spec['budget'],'old_goal_card_not_replaced':True,'automation_recreated':False
}
write('ff_prior_guided_chain_plan/evidence/execution_goal.json',goal)
outputs += ['ff_prior_guided_chain_plan/evidence/implementation_spec.json','ff_prior_guided_chain_plan/evidence/execution_goal.json']

prefix="""# 2026-10-26 当前：冻结权重 Diffusion 全轨迹引导方案就绪，追加预算未决

最新用户要求先核对现有模型、给出自由漂浮双臂/原七航点/真实移动卫星的具体适配与预算，并合并动量修复。模型结构和身份11项检查通过；静态102维低频权重及时序271×17整轨迹权重可作冻结先验，不能直接复用旧零初速/零动量预测接口。0新神经前向、采样、优化、耦合、几何和原生；尚未获得新场景物理资格。

具体方案ff_prior_guided_chain_plan/docs/EXPERIMENT_PLAN.md：三段各在开始前生成完整时域，DDIM后四步回注全机物理约束；完整初态/非零继承动量预测、原2R5C/七点/目标/时序/全部物理门、2ms RK4重建0–83.5秒连续链。旧动量根因结论保留，RK4效果尚未实际验证；不扣初值、不注速、不重置，不把旧成功路径当新网络输出。新提案替代此前固定旧Z的动量重建提案，不另加重复单元。

本轮计划预留1800秒/32MiB后保守余额10873耦合、34340原生、4536.641秒、587514890bytes。拟定一个完整单元上限40000耦合、83500原生含独立重放、12小时、8GiB；建议追加30000耦合、50000原生、11小时、7.5GiB，同时覆盖D3和全周期上限。完整推导ff_prior_guided_chain_plan/evidence/budget_derivation.json。仅提案，原上限未改；未生产实现、冻结或启动worker。

旧run仍为封存动量失败；原真实媒体继续保留。无训练/扩库/优势对照/TEST。长期目标未完成，旧定时任务保持删除；不伪造complete/paused替换App旧goal卡。后文为历史，不覆盖本节、最新用户指示及control。

"""
for rel in ['DIFFUSION_GOAL_CURRENT.md','ff_prior_fullchain/docs/CONTINUOUS_DELIVERY_GOAL.md']:
    p=ROOT/rel
    old=p.read_text(encoding='utf-8-sig')
    if not old.startswith('# 2026-10-26 当前：冻结权重'):
        p.write_text(prefix+old,encoding='utf-8')
    outputs.append(rel)
control_path=ROOT/'ff_prior_fullchain/control.json'
control_before=control_path.read_bytes()
backup=PKG/f'history/control_before_{STAMP}.json'
if not backup.exists(): backup.write_bytes(control_before)
c=json.loads(control_before.decode('utf-8-sig'))
c.update(
 current_phase='D3_frozen_diffusion_guidance_plan_ready',
 current_user_steering='ff_prior_guided_chain_plan/evidence/scope.json',
 decision_document=spec['plan'],
 implementation_handoff='ff_prior_guided_chain_plan/evidence/implementation_spec.json',
 next_phase='D3_DIFFUSION_GUIDED_CHAIN_PLAN_READY_BUDGET_EXTENSION_REQUIRED',
 next_phase_action='New integrated guided chain plan: resolve additional30000coupled/50000native/39600s/7.5GiB, implement and freeze one candidate chain, then single worker. No new physics or freeze now.',
 autonomous_scope='Latest user directs frozen model reuse inspection and concrete whole-trajectory guidance plus momentum-repair plan. Existing physical scope persists; new resource ceilings unresolved.',
 latest_planning_goal='ff_prior_guided_chain_plan/evidence/execution_goal.json',
 latest_planning_budget=spec['budget'],
 new_momentum_failure_impasse_turns=0
)
if not any(x['id']=='D3_guided_diffusion_chain' for x in c['phases']):
    c['phases'].append({'id':'D3_guided_diffusion_chain','purpose':'New neural whole-trajectory generation plus full-machine guidance and unchanged-gate momentum rebuild, original0s to83.5s','status':'PLAN_READY_BUDGET_EXTENSION_UNRESOLVED_NOT_IMPLEMENTED'})
write('ff_prior_fullchain/control.json',c)
outputs += ['ff_prior_fullchain/control.json',backup.relative_to(ROOT).as_posix()]
outputs += [p.relative_to(ROOT).as_posix() for p in PKG.rglob('*') if p.is_file()]
outputs=list(dict.fromkeys(outputs))
manifest=ROOT/'MANIFEST.md'
with manifest.open('a',encoding='utf-8') as f:
    f.write('\n')
    for rel in outputs:
        f.write(f'| {STAMP} | /experiment-plan | {rel} | engineering | 冻结权重复用、整轨迹引导与原门动量修复方案；0新采样/物理；预算未批准 |\n')
write('ff_prior_guided_chain_plan/evidence/assembly_receipt.json',{
 'created_at':datetime.now().astimezone().isoformat(),'status':'PLANNING_FILES_ASSEMBLED',
 'source_base':scope['source_base_commit'],'outputs':outputs,
 'prior_latest_preserved':True,'old_closed_run_identity_retained':all(c[k]==json.loads(control_before.decode('utf-8-sig'))[k] for k in ['active_run','active_run_role','active_module','active_source_commit','active_report','active_closeout_receipt']),
 'budget_extension_approved':False,'production_implemented':False,'new_native':0,'new_coupled':0,'new_samples':0
})
print(json.dumps({'status':'PLAN_ASSEMBLED','files':len(outputs),'native':0,'coupled':0,'samples':0},ensure_ascii=False))

