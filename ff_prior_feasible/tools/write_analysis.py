"""Write the Chinese report after verified completion; derive all numbers from files."""
from pathlib import Path
from datetime import datetime
from ff_prior.storage import read_json,atomic_json,digest
from ff_prior_feasible.experiment import METHODS,PREVIOUS


def main():
    out=Path('artifacts/v7_1/goal_preserving_smoke_001'); s=read_json(out/'summary.json')
    a=read_json(out/'independent_verification.json'); state=read_json(out/'status.json')
    assert a['status'] in ('PASS','PASS_WITH_PRESERVATION_EXCEPTION') and state['status'] in ('FINISHED','FINISHED_WITH_BLOCKED_ITEMS')
    previous=read_json(PREVIOUS/'summary.json'); targets=read_json(out/'frozen_tasks.json')['evaluation_entries']
    m=s['methods']; old_names=['structured_joint','frozen_diffusion_joint','trained_diffusion_joint','trained_diffusion_goal_only']
    names=['强样条＋约束碰撞修正','旧冻结 Diffusion＋约束碰撞修正','新 Diffusion＋约束碰撞修正','新 Diffusion＋只修正目标']
    n,b=m[METHODS[2]],m[METHODS[0]]
    ratio=lambda key: 100*(n[key]/b[key]-1) if b[key] else 0
    stability='通过' if s['stability_gate_passed'] else '未通过'
    advantage='通过' if s['exploratory_advantage_gate_passed'] else '未通过'
    lines=['# 保持世界目标的碰撞修正：实际分析报告','',
        f'本轮已完成，后台耗时 {state["elapsed_s"]:.1f} 秒。98/98 软件测试通过。预注册候选稳定门槛{stability}，探索性优势门槛{advantage}。强制核心目标“验证可复现的 Diffusion 优越性，并最终实际扩库”仍保持进行中，详见CORE_GOAL.md；两项必须同时完成。',
        '', '## 实测结果','',
        '| 方法 | 上轮合格/50 | 本轮MODEL_VALID/50 | 本轮目标合格/50 | 执行重放/10 | 耦合查询 | 积分区间 | 候选秒 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for method,old,name in zip(METHODS,old_names,names):
        r=m[method]; lines.append(f'| {name} | {previous["methods"][old]["eligible"]} | {r["MODEL_VALID"]} | {r["eligible"]} | {r["executed_verified"]} | {r["coupled_queries"]} | {r["integration_intervals"]} | {r["candidate_wall_s"]:.3f} |')
    lines+=['',f'新 Diffusion 相对样条：耦合查询 {ratio("coupled_queries"):+.2f}%，积分区间 {ratio("integration_intervals"):+.2f}%，实际候选耗时 {ratio("candidate_wall_s"):+.2f}%。相同上限不代表实际成本相同。',
        '', '本轮没有新训练。新权重仍是上一轮600步检查点，旧权重仍是200步检查点；改善首先归于修正/验证机制，不能写成新增训练收益。',
        '', '## 逐任务与配对结果','', '| 任务 | 样条 | 旧Diffusion | 新Diffusion | 新Diffusion仅目标 |','|---|---:|---:|---:|---:|']
    for e in targets:
        lines.append('| '+e['sample_id']+' | '+' | '.join(str(m[x]['by_task'][e['sample_id']])+'/5' for x in METHODS)+' |')
    pairs={}; raw_hashes=0; adaptive_required=0
    for other in (METHODS[0],METHODS[1],METHODS[3]):
        wins=losses=both=neither=0; task_wins=[]; task_losses=[]
        for e in targets:
            sid=e['sample_id']; av=bv=0
            for k in range(5):
                ra=read_json(out/'candidates'/sid/METHODS[2]/f'{k:02d}.json')
                rb=read_json(out/'candidates'/sid/other/f'{k:02d}.json')
                x,y=ra['eligible'],rb['eligible']; av+=x; bv+=y
                wins+=x and not y; losses+=y and not x; both+=x and y; neither+=not x and not y
            if av and not bv: task_wins.append(sid)
            if bv and not av: task_losses.append(sid)
        pairs[other]={'candidate_wins':wins,'candidate_losses':losses,'both':both,'neither':neither,'task_wins':task_wins,'task_losses':task_losses}
    for e in targets:
        for k in range(5):
            r=read_json(out/'candidates'/e['sample_id']/METHODS[2]/f'{k:02d}.json')
            old=read_json(PREVIOUS/'candidates'/e['sample_id']/'trained_diffusion_joint'/f'{k:02d}.json')
            raw_hashes+=r.get('sampling',{}).get('guidance',[{}])[-1].get('input_hash')==old.get('sampling',{}).get('guidance',[{}])[-1].get('input_hash')
            adaptive_required+=r['eligible'] and len(r.get('validation_grids',[]))==2 and not r['validation_grids'][0]['gates']['geometry_refinement']
    for other,p in pairs.items():
        lines.append(f'\n新 Diffusion 对 {other}：同种子候选独赢 {p["candidate_wins"]}、独输 {p["candidate_losses"]}；任务独赢 {p["task_wins"] or "无"}，独输 {p["task_losses"] or "无"}。')
    lines+=['',f'与上轮新模型相比，最终修正前输入哈希一致 {raw_hashes}/50；其中本轮 {adaptive_required} 条合格新模型候选需要1024网格才能满足原2mm几何采样门槛。两处改造共同作用，不能把所有增益独归于碰撞约束。',
        '', '## 失败与机制','',
        '原方案把世界目标、3mm碰撞裕量和路径几何位移放入一个加权目标，允许一种误差改善补偿另一种误差退化。本轮在局部求解中约束双末端位置/姿态范数，再用完整自由漂浮传播回溯；已满足目标内层门槛的候选不能因碰撞改善而被推出。',
        '', '优化内层为3.25mm/0.65°，回溯保护为4mm/0.8°；碰撞修正缓冲从原优化代价的3mm改为0.2mm。它们是提案算法的参数，实际物理准入仍为5mm/1°和0mm不穿透。候选通过率改善不等于终端精度或安全裕量全面提高，实际误差和最小间距需一起看。',
        '', '验证器按需细化256→512→1024；原2mm几何变化、1mm对齐收敛、全机碰撞对均保持。之前检测到的穿透不会被后续网格覆盖。更细网格是更充分的模型验证，不是降低物理约束，也不等于连续时间形式证明。']
    for method,name in zip(METHODS,names): lines.append(f'\n{name}失败计数（原因可重叠）：{m[method]["failure_reasons"] or "无"}。')
    cond=read_json('ff_prior_feasible/evidence/conditioning_audit.json')
    lines+=['', '## 数据与学习侧仍有的短板','',
        '实际TRAIN为208条、30个家族，只有3条含障碍（1.44%）；VAL为48条、6家族，也只有3条含障碍。网络48维条件包括初始q17、双末端世界目标、时长和姿态掩码，没有障碍几何。障碍信息由物理修正器读取，因此当前学习到的是运动先验，尚不是充分的场景条件绕行策略。这个结构限制是源码和数据审计事实；其对成功率的因果贡献仍需消融验证。',
        '', '13种TRAIN初始q17仍共享既定基座姿态、静止零动量假设；不是任意初速/基座状态数据集。少量同家族变体增加条数，不能等同扩大独立任务覆盖。不能从当前结果推断微重力导致Diffusion失效。',
        '', '## 真实物理与保留审计','',
        f'独立核验 {a["audited_candidates"]} 条候选、{a["network_calls"]} 次网络调用、{a["actual_torque_executions"]} 次67力矩执行和独立重放，共 {a["physics_steps_including_replay"]} 个动力学步。执行最小间距 {a["minimum_execution_clearance_m"]*1000:.6f} mm，保持段最大位置误差 {a["maximum_hold_position_m"]*1000:.6f} mm，最大旋转误差 {a["maximum_hold_rotation_rad"]:.9f} rad。',
        '', f'候选最大查询 {a["max_candidate_queries"]}/113，最大积分区间 {a["max_candidate_integration_intervals"]}/15872，最大耗时 {a["max_candidate_wall_s"]:.3f}/15秒；全方法共同预算。独立审计没有再运行模型，也未把未执行候选升级为EXECUTION_VERIFIED。',
        '', f'{a["old_files_unchanged"]}/{a["old_files_total"]} 个旧文件未改变；四项用户未提交修改、main及旧分支保持。旧实验结果均保留，原冻结DEV仅核验、未执行；没有pilot、study、TEST、扩库或生产规模作业。',
        '', '保留审计有一项明确例外：旧测试test_real_ctrl_execution_independent_replay_and_reaction使用固定输出路径，覆盖了artifacts/v7_1/test_physics/report.json。原哈希已保存，原报告完整字节未备份，无法从哈希恢复，不能宣称所有旧文件未改。该测试的native_trace.npz与其他旧文件哈希未变；本轮提供隔离测试输出插件，防止再次覆盖。',
        '', '## 下一步门控','',
        '首先保留本轮结果并按真实成败判断修正器。若候选稳定但Diffusion成本仍高或无独赢，则优先增加少量固定TRAIN目标的场景成对修正标签、显式障碍条件及训练难例平衡；不凭增加近重复无障碍轨迹宣称数据完善。若本轮优势成立，则用预先冻结的新种子和TRAIN生成/VAL保留规则做有界复核，再讨论扩库。',
        '', '下一轮需区分：修正器贡献、网络场景条件贡献、新数据贡献。持续保留强样条、旧冻结模型和同候选/同预算，固定目标失败不重标；所有物理阈值不变。当前探索VAL已用于开发，且只涉及6个家族；完整目标还需要重复性证据，不能由一轮smoke自动结案。',
        '', '按用户本轮追加要求，最终实际扩库是强制交付。长期按1千→1万→10万条合格先验推进；参考HTDCR的轻量生成、精修与学习/引导分工。其10万构型对×5模式是源码名义配置，不是本项目应直接复制的物理合格数量。扩库阶段另行冻结覆盖、质量、资源与授权门槛，本轮未启动生产规模。',
        '', '## 终端','', '```powershell', '.\\ff_prior_feasible\\job.ps1 status','.\\ff_prior_feasible\\job.ps1 report','.\\ff_prior_feasible\\job.ps1 resume','```',
        '', '原始证据：artifacts/v7_1/goal_preserving_smoke_001；逐候选JSON、真实native_trace.npz、执行报告、账本、源码归档均保留。']
    details={'paired':pairs,'previous_raw_proposal_hashes_equal':raw_hashes,'new_eligible_needing_1024':adaptive_required,
        'relative_new_vs_structured_percent':{k:ratio(k) for k in ('coupled_queries','integration_intervals','candidate_wall_s')}}
    atomic_json(out/'analysis_details.json',details)
    docs=Path('ff_prior_feasible/docs'); stamp=datetime.now().strftime('%Y%m%d_%H%M%S')
    version=docs/f'ANALYSIS_{stamp}.md'; version.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    (docs/'ANALYSIS.md').write_bytes(version.read_bytes())
    with Path('MANIFEST.md').open('a',encoding='utf-8') as f: f.write(f'\n- {stamp} | ANALYSIS | {version.as_posix()} | 保持目标的碰撞修正smoke真实报告\n')
    print(details)


if __name__=='__main__': main()
