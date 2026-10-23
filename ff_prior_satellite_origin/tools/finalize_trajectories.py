"""Seal only this turn's media/plan metadata; old run remains closed and unchanged."""
from pathlib import Path
import hashlib,json,datetime

ROOT=Path(__file__).resolve().parents[2]
PKG=ROOT/'ff_prior_satellite_origin'
DEST=PKG/'delivery_002'
PLAN=PKG/'planning'
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

def main():
    now=datetime.datetime.now(datetime.timezone.utc).isoformat()
    old=PKG/'delivery_001/manifest.json'
    assert sha(old)=='c8c75f4f5ef96ae42585489dbc57cda994139a015052a5a42e5a82884c0cc9ff'
    for stem in ('EXPERIMENT_PLAN','EXPERIMENT_TRACKER'):
        (PLAN/(stem+'.md')).write_bytes((PLAN/(stem+'_20261023_204207.md')).read_bytes())
    pagepath=DEST/'index.html';page=pagepath.read_text(encoding='utf-8')
    label='<p><a href="manifest.json">本次显示更新清单</a> · <a href="../planning/EXPERIMENT_PLAN.md">后续详细执行目标</a></p>'
    if label not in page:page=page.replace('<section><h2>颜色与末端轨迹</h2>','<section><h2>颜色与末端轨迹</h2>'+label,1)
    pagepath.write_text(page,encoding='utf-8')
    manifest=read(DEST/'manifest.json')
    manifest['presentation_revision']='trajectory panels and clearer static figure; no new physics'
    manifest['finalized_at_utc']=now
    manifest['old_delivery_manifest_sha256']=sha(old)
    manifest['qa']='../evidence/trajectory_update_qa.json'
    manifest['files']={p.name:{'bytes':p.stat().st_size,'sha256':sha(p)} for p in DEST.iterdir() if p.is_file() and p.name!='manifest.json'}
    manifest['output_bytes']=sum(v['bytes'] for v in manifest['files'].values())
    assert manifest['output_bytes']<64*1024**2
    save(DEST/'manifest.json',manifest)
    controlpath=ROOT/'ff_prior_fullchain/control.json';control=read(controlpath)
    assert control['current_phase']=='D3_physical_satellite_origin_closed_momentum_failure'
    assert control['active_run_role']=='most_recent_closed_run' and not control['active_worker']
    control.update(decision_document='ff_prior_satellite_origin/planning/EXPERIMENT_PLAN.md',implementation_handoff='ff_prior_satellite_origin/planning/EXPERIMENT_PLAN.md',next_phase='D3_MOMENTUM_CONTINUATION_PLAN_READY_NO_AUTOMATIC_PHYSICS',next_phase_action='Read-only momentum contract and admissible repair feasibility first. Same continuation remains failed; full 83.5s rebuild exceeds remaining native budget. No new worker or automatic retry.',latest_visualization='ff_prior_satellite_origin/delivery_002/index.html',latest_planning_goal='ff_prior_satellite_origin/planning/execution_goal.json',latest_planning_budget='ff_prior_satellite_origin/planning/remaining_budget.json')
    save(controlpath,control)
    note='# 2026-10-23 显示澄清与下一阶段详细执行目标\n\n用户本轮要求解释视频变红、绘制末端轨迹并制定下一步目标。新版实际0–83.5秒媒体位于ff_prior_satellite_origin/delivery_002/index.html：双臂累计轨迹加粗、独立XY/XZ轨迹窗、完整XYZ导出及颜色图例。红色是小于10mm的显示提示，不代表穿透或因果避障修正；物理门仍为不穿透。旧媒体和已封存run不变，本轮0新原生/重放/耦合/几何查询。\n\n详细目标与预算：ff_prior_satellite_origin/planning/EXPERIMENT_PLAN.md、EXPERIMENT_TRACKER.md、remaining_budget.json。先明确连续交接动量合同，再冻结有效修复；相同54.5秒初态的盲重跑不能消除已知失败，0–83.5秒完整重建至少83500原生步，当前余额34340，故不启动新物理单元。本轮80MiB/1500秒只读预留后主账余额10873耦合/34340原生/8136.641秒/688178186bytes。保持原门、不扣初值追认、不训练/扩库/对照/TEST。长期目标未完成，旧定时任务保持删除。下文为历史记录。\n\n'
    for rel in ('DIFFUSION_GOAL_CURRENT.md','ff_prior_fullchain/docs/CONTINUOUS_DELIVERY_GOAL.md'):
        p=ROOT/rel;oldtext=p.read_text(encoding='utf-8-sig')
        if not oldtext.startswith(note.split('\n')[0]):p.write_text(note+oldtext,encoding='utf-8')
    outputs=['ff_prior_satellite_origin/planning/'+n for n in ('EXPERIMENT_PLAN_20261023_204207.md','EXPERIMENT_PLAN.md','EXPERIMENT_TRACKER_20261023_204207.md','EXPERIMENT_TRACKER.md','execution_goal.json','remaining_budget.json')]+['ff_prior_satellite_origin/delivery_002/index.html']
    with (ROOT/'MANIFEST.md').open('a',encoding='utf-8') as f:
        for path in outputs:f.write(f'\n| 20261023_204207 | /experiment-plan | {path} | engineering | 轨迹显示澄清、连续动量约束及保守预算；0新物理 |\n')
    record={'status':'COMPLETE_READONLY_MEDIA_AND_PLAN','created_at_utc':now,'user_request':'为啥视频中一开始机械臂会变红，这个是避障规划？可视化把末端轨迹也绘制出来，其次分析下一步进程该如何进行，并制定详细执行目标','media':'ff_prior_satellite_origin/delivery_002/index.html','media_manifest_sha256':sha(DEST/'manifest.json'),'old_delivery_manifest_unchanged':True,'old_delivery_manifest_sha256':sha(old),'scientific_status_unchanged':'EXECUTION_FAILED_MOMENTUM','new_native_steps':0,'new_replay_steps':0,'new_coupled_calls':0,'new_geometry_queries':0,'new_neural_samples':0,'old_resume_calls':0,'new_scientific_reaudit':False,'plan':'ff_prior_satellite_origin/planning/EXPERIMENT_PLAN.md','plan_sha256':sha(PLAN/'EXPERIMENT_PLAN.md'),'qa':'ff_prior_satellite_origin/evidence/trajectory_update_qa.json','budget':'ff_prior_satellite_origin/planning/remaining_budget.json','measured_initial_media_render_s':manifest['wall_s'],'new_media_bytes_excluding_manifest':manifest['output_bytes'],'turn_wall_policy':'Reserve the entire 1500s planning/media envelope; no complete turn-start wall measurement claimed.','storage_policy':'Reserve 64MiB media plus 16MiB new metadata/Git overhead; old sealed artifacts not duplicated.','app_goal_status_not_mutated':True,'automation':'DELETED_NOT_RECREATED','physics_worker_started':False}
    save(PKG/'evidence/trajectory_and_plan_closeout.json',record)
    print(json.dumps(record,ensure_ascii=False))

if __name__=='__main__':main()
