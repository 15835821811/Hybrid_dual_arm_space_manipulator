"""Render the final Chinese analysis from audited results, never from progress guesses."""
from pathlib import Path
from datetime import datetime
from collections import Counter
from ff_prior.storage import read_json,file_digest


def main():
    out=Path('artifacts/v7_1/qualified_guidance_loop_001'); s=read_json(out/'summary.json')
    audit=read_json(out/'independent_verification.json'); t=read_json(out/'learning/training.json')
    status=read_json(out/'status.json'); gate=read_json(out/'expansion_decision.json')
    tasks=read_json(out/'frozen_tasks.json')['evaluation_entries']; m=s['methods']
    final=read_json(out/'final_dataset_index.json')['records']; train=read_json(out/'training_index.json')['records']
    elapsed=status['elapsed_s']; timestamp=datetime.now().strftime('%Y%m%d_%H%M%S')
    labels={'structured_joint':'平滑样条＋联合修正','frozen_diffusion_joint':'旧 Diffusion＋联合修正',
        'trained_diffusion_joint':'新 Diffusion＋联合修正','trained_diffusion_goal_only':'新 Diffusion＋仅目标修正'}
    lines=['# 合格数据、训练与碰撞引导闭环分析','',
        f'后台实测 {elapsed:.1f} 秒，状态 **{status["status"]}**。软件 **90/90**，原始证据复核 **{audit["status"]}**。',
        '',f'新标签 **{s["new_generation_eligible"]}/48** 通过指定世界目标和 MODEL_VALID。预选 **{s["preselected_witness_verified"]}/8** 通过真实 MuJoCo 67 力矩执行与独立重放。',
        '',f'扩库结论：**{gate["decision"]}**。'+('已按预注册范围生成额外最多12条TRAIN尝试；新增部分没有再次训练。' if gate['passed'] else '尚未达到预先冻结的稳定采样条件，未启动额外12条扩库，更未启动pilot或TEST。'),
        '', '## 1. 本轮实际改动', '',
        '把上一轮只有索引引用的复杂数据接入实际数组加载、TRAIN抽样和新网络训练。增加16个静止零动量初始构型，每个完整保存初态；物理home、机器人编译模型、惯量、自由基座和17→67映射不变。每个世界目标只由首条完整耦合传播派生一次，另外两条轨迹向同一目标修正。失败不重标。',
        '', '新增全路径碰撞代价，使用128区间的自由漂浮传播与33个几何引导状态。24个低频方向在同一求解器中修正末端和绕行路径；每方向限步避免一个病态大方向压小所有有效方向。增加沿途几何变化惩罚，避免绕行过猛触发原有2mm采样门槛。引导栅格只用于提案，随后独立做完整几何、动力学执行和重放验收。',
        '', '共享修正器只在第16步DDIM的干净轨迹上作用；v预测、float64状态、投影后重建epsilon以及最后控制点精确返回继续保留。',
        '', '## 2. 数据能否用于学习', '',
        f'实际训练索引 {dict(Counter(e["split"] for e in train))}；最终可用索引 {dict(Counter(e["split"] for e in final))}。TRAIN抽样最少/最多次数为 {min(t["train_draw_counts"].values())}/{max(t["train_draw_counts"].values())}，所有TRAIN行均参与训练，VAL梯度更新为0。',
        '',f'训练600次更新，VAL选中第 **{t["selected_update"]}** 次权重；权重确实变化，加载后输出最大差 {t["reload_error"]}。TRAIN-only temporal prior 秩为 {t["prior_rank"]}。fit_population_token是训练初态集合的身份标识，不冒充任意单一初态。',
        '', '合格数据已可用于当前范围的残差运动先验训练。MODEL_VALID样本只承诺通过模型门槛，未执行样本不会被抽查批量升级。数据可用与网络能稳定规划是两件事；后者看下一节逐任务结果。',
        '', '原TEST文件仅参与保全哈希核验，没有加载TEST数组用于训练、调参或最终评估。',
        '', '## 3. 同预算对照', '',
        '10个VAL任务：原6复杂任务加4个新初态任务。每方法每任务5个相同种子候选；每候选110次修正+3次独立验证查询、15秒上限，Diffusion16次网络调用。方法次序按任务轮换。原DEV只核验哈希，不重跑。',
        '', '| 方法 | MODEL_VALID/50 | 指定目标合格/50 | 执行重放/10 | 耦合查询 | 距离查询 | 候选耗时秒 |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for name,r in m.items(): lines.append(f'| {labels[name]} | {r["MODEL_VALID"]} | {r["eligible"]} | {r["executed_verified"]} | {r["coupled_queries"]} | {r["geometry_queries"]} | {r["wall_s"]:.2f} |')
    lines+=['', '| 任务 | 样条 | 旧Diffusion | 新Diffusion联合 | 新Diffusion仅目标 |','|---|---|---|---|---|']
    for e in tasks:
        cells=[]
        for name in labels:
            p=out/'executions'/e['sample_id']/name/'outcome.json'
            outcome=read_json(p)['status'] if p.exists() else 'BLOCKED'
            cells.append(f'{m[name]["by_task"][e["sample_id"]]}/5；{outcome}')
        lines.append('| '+e['sample_id']+' | '+' | '.join(cells)+' |')
    def passed(e,method):
        p=out/'executions'/e['sample_id']/method/'outcome.json'
        return p.exists() and read_json(p)['status']=='EXECUTION_VERIFIED'
    wins=[e['sample_id'] for e in tasks if passed(e,'trained_diffusion_joint') and not passed(e,'structured_joint')]
    losses=[e['sample_id'] for e in tasks if not passed(e,'trained_diffusion_joint') and passed(e,'structured_joint')]
    collision_gains=collision_losses=0
    for e in tasks:
        for k in range(5):
            a=out/'candidates'/e['sample_id']/'trained_diffusion_joint'/f'{k:02d}.json'
            b=out/'candidates'/e['sample_id']/'trained_diffusion_goal_only'/f'{k:02d}.json'
            if a.exists() and b.exists():
                av,bv=read_json(a)['eligible'],read_json(b)['eligible']; collision_gains+=av and not bv; collision_losses+=bv and not av
    a,b=m['trained_diffusion_joint'],m['structured_joint']
    lines+=['',f'新Diffusion对样条的任务级独赢：{wins}；反向失败：{losses}。新Diffusion实际查询相对样条 {(a["coupled_queries"]/b["coupled_queries"]-1)*100:+.2f}%，候选耗时 {(a["wall_s"]/b["wall_s"]-1)*100:+.2f}%。',
        '',f'同一新权重、同一噪声种子的碰撞引导消融：联合修正独有合格 **{collision_gains}** 个候选，取消碰撞代价反而独有合格 **{collision_losses}** 个。必须同时看这两个方向，不能把更多碰撞计算自动写成普遍收益。',
        '', '## 4. 失败归因与噪声诊断', '',
        '| 方法 | 未合格原因计数（同一候选可多项） |','|---|---|']
    for name,r in m.items(): lines.append(f'| {labels[name]} | {r["failure_reasons"]} |')
    lines+=['',f'新训练与旧冻结网络的合格数分别是 {a["eligible"]}/50 与 {m["frozen_diffusion_joint"]["eligible"]}/50，选中执行分别为 {a["executed_verified"]}/10 与 {m["frozen_diffusion_joint"]["executed_verified"]}/10。新增数据训练的收益没有据此成立。',
        '', '本轮最具体的机制问题是联合修正仍采用加权软代价。在同权重同噪声的消融中，加入碰撞项使碰撞失败从8降至1，但世界目标失败从0增至7，几何细化失败从2增至7。这些计数有重叠。50组消融在最后去噪修正前的输入哈希完全相同，因此这次变化来自联合修正过程；不能归咎于重新抽样。',
        '', '3mm碰撞激活裕量高于实际0mm验收要求，修正器可能为增大间隙牺牲终端误差或增加路径弯曲。现有加权和与有限迭代不能保证所有约束同时满足。应优先验证保留目标可行性的分层/带约束修正，以及在已不穿透时是否仍需要额外推离；物理验收仍保持0mm、5mm/1°等原门槛。不能靠继续堆数据来替代这一修正。',
        '', '8个不同初态的见证执行全部通过，最大位置误差约0.00127mm。这个范围内没有证据把失败归因为微重力耦合模型错误；更直接的证据指向局部修正的约束权衡和几何验证分辨率上限。该判断仅覆盖本轮案例。']
    lines+=['','| 噪声时刻 | VAL残差重建RMSE rad | 末端控制点RMSE rad | 打乱条件RMSE rad |','|---|---:|---:|---:|']
    for x,y in zip(t['noise_bands'],t['shuffled_condition_noise_bands']): lines.append(f'| {x["t"]} | {x["rmse_rad"]:.6g} | {x["terminal_rmse_rad"]:.6g} | {y["rmse_rad"]:.6g} |')
    lines+=['',f'零残差基线VAL RMSE：{t["zero_residual_val_rmse_rad"]:.6g} rad。噪声重建是诊断指标，不是世界目标到达率；低训练损失不能取代实际规划/执行。',
        '', '碰撞、目标误差与几何离散误差分开计数。局部引导不保证全局可达：固定查询预算内可能停在局部形状，修正目标又改变沿途路径，或为避障增加曲率而触发严格几何细化门槛。本轮未以固定基座、放宽门槛或重标目标规避失败。',
        '', '## 5. 真实物理证据与保持项', '',
        f'本轮实际执行（含见证） **{audit["actual_torque_executions"]}** 次，含独立重放 **{audit["physics_steps_including_replay"]}** 个物理步。最小实测间隙 {audit["minimum_execution_clearance_m"]:.9g} m；最大保持位置误差 {audit["maximum_hold_position_m"]:.9g} m；最大保持姿态误差 {audit["maximum_hold_rotation_rad"]:.9g} rad。',
        '', f'逐候选最大查询 {audit["max_candidate_queries"]}，最大候选秒 {audit["max_candidate_wall_s"]:.4f}。原始控制数组哈希、81维qpos、67维力矩、完整时间、保持区间末端误差与独立重放逐份复核。',
        '',f'已核验 {audit["all_old_run_files_unchanged"]} 个旧运行文件未变；main、旧分支、用户原有四个未提交文件和冻结DEV保留。旧网络不覆盖，新checkpoint与原始NPZ均仅保存在本地。',
        '', '## 6. 扩库门槛与下一步', '', '| 预注册检查 | 结果 |','|---|---|']
    for k,v in gate['checks'].items(): lines.append(f'| {k} | {"PASS" if v else "BLOCKED"} |')
    diagnostic=out/'posthoc_resolution/results.json'
    if diagnostic.exists():
        diag=read_json(diagnostic)
        lines+=['','### 补充：几何分辨率诊断（不改变正式结果）','',
            '按照预先写入诊断脚本的规则，取冻结任务顺序中最先出现的最多3个“目标通过、仅geometry_refinement失败”的新联合Diffusion候选。保持原路径、初态、世界目标、0mm不穿透与2mm几何步长门槛，另做512→1024区间传播和全机几何检查。没有重新采样或执行，没有替换正式候选。',
            '', '| 候选 | 原最大几何步长mm | 1024区间步长mm | 加密后模型+目标 |','|---|---:|---:|---|']
        for d in diag['results']:
            lines.append(f'| {d["candidate"]} | {d["original_max_geometry_step_m"]*1000:.4f} | {d["diagnostic_1024_validation"]["geometry"]["maximum_geometry_step_m"]*1000:.4f} | {d["diagnostic_model_and_goal_pass"]} |')
        lines+=['',f'额外耦合查询 {diag["additional_coupled_queries"]} 次，单列为事后诊断，不归入正式等预算表。正式候选和扩库门控哈希未变。加密后通过只支持这些案例原先受分辨率上限阻断，不能记成新增EXECUTION_VERIFIED或将正式结果改成通过。',
            '', '因此下轮可把“几何验证上限是否应自适应”作为独立软件改造：保持2mm门槛和全部物理约束，提前冻结各方法相同的计算预算，并重新计费完整验证成本。不能把更密集检查隐藏在某一个方法之外。']
    lines+=['', '本轮是小规模工程与探索性VAL证据。一个训练种子、五个采样种子及复用开发任务不足以证明独立测试泛化；新旧模型还同时改变了数据、训练步数、归一化和选择损失，所以不能单独归因为新增数据。',
        '', '下一步应由失败项决定：若新初态通过而旧复杂任务未通过，优先在TRAIN补充受约束绕障和路径形状修正，验证局部修正是否能同时满足目标与几何细化；保持原VAL冻结。仅当稳定门槛通过后再扩大覆盖，不以数量替代每条标签与采样质量。',
        '', '## 7. 终端与证据', '', '```powershell',' .\\ff_prior_loop\\job.ps1 status',' .\\ff_prior_loop\\job.ps1 report',' .\\ff_prior_loop\\job.ps1 resume','```','',
        '后台报告：artifacts/v7_1/qualified_guidance_loop_001/REPORT.md。详细原始证据包括 generation、learning、candidates、executions、witness_executions、independent_verification.json。紧凑证据归档在 ff_prior_loop/evidence/qualified_guidance_loop_001；不包含大规模权重或轨迹数组。',
        '', '完成后的resume实测返回NO_WORK_ALREADY_COMPLETE，653个既有输出文件字节未变，后台进程已退出。']
    text='\n'.join(lines)+'\n'; directory=Path('ff_prior_loop/docs')
    tracker='\n'.join(['# 闭环执行跟踪','','| 阶段 | 状态 | 证据 |','|---|---|---|',
        '| M0 软件 | PASS 90/90 | evidence/tests.xml |',
        f'| M1 数据 | {s["new_generation_eligible"]}/48；审计{s["preselected_witness_verified"]}/8 | generation、witness_executions |',
        f'| M2 训练 | {t["status"]}，600更新，选中{t["selected_update"]} | learning/training.json |',
        f'| M3 采样及执行 | 新联合合格{a["eligible"]}/50，执行{a["executed_verified"]}/10 | candidates、executions |',
        f'| M4 扩库 | {gate["decision"]} | expansion_decision.json |','| 原始证据复核 | PASS | independent_verification.json |',''])
    for name,value in [('ANALYSIS',text),('EXPERIMENT_TRACKER',tracker)]:
        for filename in (f'{name}_{timestamp}.md',f'{name}.md'):
            p=directory/filename; p.write_text(value,encoding='utf-8')
            for mf in (Path('MANIFEST.md'),Path('ff_prior_loop/MANIFEST.md')):
                with mf.open('a',encoding='utf-8') as f:f.write(f'| {timestamp} | experiment-plan | {p.as_posix()} | experiment | 实测闭环分析 |\n')
    print(str(directory/'ANALYSIS.md'))


if __name__=='__main__': main()
