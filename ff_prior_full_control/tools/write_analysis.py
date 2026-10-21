"""Render the Chinese report from compact saved-result statistics (no experiments)."""
from ff_prior.storage import read_json
from ff_prior_full_control.protocol import PACKAGE


def main():
    s=read_json(PACKAGE/'evidence/completed/statistics.json')
    v=s['VAL'];d=s['groups']['DEV'];b=d['full323'];old=d['old_aligned']
    lines=['# 完整19×17控制输出与资格保持prepare：有界闭环分析',
        '', '运行 `full_control_bounded_001`；2026-10-21；冻结源码 `f788a689`，启动记录 `aa3a0d2e`。',
        '', '**本轮预定流程已实际跑完，但新模型没有通过DEV工程门，也没有建立Diffusion收益。** 完整表示和prepare通过独立TRAIN重建/执行门；从头训练的323维候选，无引导命中和有引导整体资格均退化。保留旧模型作为当前基线，本轮不扩库、不启动TEST，也不自动追加试验。',
        '', f'后台耗时 {s["status"]["elapsed_s"]:.2f} 秒（约 {s["status"]["elapsed_s"]/60:.2f} 分钟）。原始运行状态FINISHED仅表示预定工作完成，不能解读为性能验收通过。原始summary的独立审查字段保持运行结束时的PENDING快照，收尾审查另存在本包和review-traces。',
        '', '## 1. 本轮实际实现与执行边界',
        '', '完整19×17控制残差展开为323维，时间基为19×19单位阵；无低秩压缩。65维公开条件、标准化、ridge中心和逐维尺度只用14222条TRAIN拟合。48条VAL及原DEV12、三/五障碍APPLICATION_DEV不进入训练/归一化/模型选择，07已执行路径也没有作为TRAIN教师。没有新增训练数据。',
        '', 'prepare在原快门满足时精确保留原控制，否则仅沿公开初始化方向按预冻结比例收缩；快门不代表全物理资格。最后引导后的精确控制直接返回，避免归一化往返造成末位越界。所有完整物理门仍随后执行。',
        '', '一个3×256 SiLU MLP从固定种子初始化，6000次AdamW更新、batch64、lr2e-4。每5步使用批前8个TRAIN条件完整展开16步DDIM，与既有教师比较完整控制MSE，教师梯度范数最多普通梯度的50%，合成后每步只更新一次。**这是完整采样教师控制监督，仍不是非线性世界终点监督。**',
        '', '对照旧模型已经过预训练及后续训练，新模型从头初始化；输出维数、codec、prepare、损失也同时变化。因此本轮是方法组合的比较，不能把退化单独归因于323维、MLP、prepare或训练初始化。相同种子在不同维数下也不等于同形状噪声。',
        '', '## 2. 独立TRAIN资格重建门：通过',
        '', '|样本|最大双臂位置误差 mm|最大姿态误差 °|控制重建最大误差 rad|全门|新执行/重放|', '|---|---:|---:|---:|---|---|']
    for r in s['reconstruction']:
        lines.append(f'|{r["sample_id"]}|{r["position_mm"]:.8f}|{r["rotation_deg"]:.8g}|{r["preparation"]["prepared_error_rad"]:.3g}|通过|'+('通过' if 'static_detour' in r['sample_id'] else '未选执行')+'|')
    lines += ['', '6/6重建使用实际float32训练目标解码，并通过完整模型与冻结世界目标门；prepare均精确保持解码结果。预选3条static_detour实际原生67维力矩执行及独立重放全部通过，未失败补位。全缓存float64纯表示往返最大误差1.6653e-16 rad，仅为软件精度检查，不能把全库升级为执行合格。',
        '', f'3条TRAIN执行的终点最大位置误差 {s["TRAIN_reconstruction"]["execution"]["final_position_mm"]["worst"]:.6f} mm，保持段最大位置误差 {s["TRAIN_reconstruction"]["execution"]["hold_position_mm"]["worst"]:.6f} mm。本轮证明已有合格轨迹可以被表示并保持，不等于网络已能生成它们。',
        '', '## 3. 训练：一次真实完成',
        '', '14222/14222 TRAIN均参与，累计384000次样本抽取，每条最少13次、最多136次；6000实际更新、300个已提交训练段、1200个完整16步教师批次（9600条计算图路径）。VAL梯度更新为0，不选checkpoint。教师图路径不是新增合格数据。权重已重载核对，初始与最终hash不同。',
        '', f'记录的第100步v损失 {s["training_history"][0]["v_loss"]:.4f}、完整教师损失 {s["training_history"][0]["full16_loss"]:.4f}；第6000步分别 {s["training_history"][-1]["v_loss"]:.4f}、{s["training_history"][-1]["full16_loss"]:.4f}。它们来自不同随机批次，只是训练日志，不能当成固定诊断集的改善曲线。正式是否学好由下面完整采样结果判断。',
        '', '## 4. 完整无引导采样：0/96，误差尾部明显变坏',
        '', 'VAL48×2种子；完整16步DDIM，无物理目标修正。以下误差每条取双臂较大值，包含全部失败样本，缺失分母为0。',
        '', '|模型|全门命中|模型门通过（不含指定目标）|位置均值/中位/P95/最大 mm|姿态均值/中位/P95/最大 °|', '|---|---:|---:|---|---|']
    for m,label in [('old_aligned','旧aligned（历史复用）'),('full323','新full323（本轮）')]:
        r=v[m];f=lambda x:'/'.join(f'{x[k]:.3f}' for k in ('mean','median','p95','worst'))
        lines.append(f'|{label}|{r["eligible"]}/96|{r["model_valid"]}/96|{f(r["position_mm"])}|{f(r["rotation_deg"])}|')
    lines += ['', '|种子组|旧命中|新命中|旧/新位置均值 mm|旧/新位置P95 mm|', '|---|---:|---:|---:|---:|']
    for sj,group in s['VAL_by_seed'].items():
        a=group['old_aligned'];c=group['full323'];lines.append(f'|{sj}|{a["eligible"]}/48|{c["eligible"]}/48|{a["position_mm"]["mean"]:.3f}/{c["position_mm"]["mean"]:.3f}|{a["position_mm"]["p95"]:.3f}/{c["position_mm"]["p95"]:.3f}|')
    lines += ['', '96条配对中只有1条新位置误差低于旧模型，平均差为+67.975 mm。新模型96条全部不满足指定世界目标；其中88条还未达到几何密度门，4条存在已检测穿透（包含在88条内），另8条模型门通过但目标不通过。',
        '', '新模型96条raw与prepare控制逐元素完全相同，因此这里的终点退化不能归咎于prepare修改。旧模型prepare有改动，比较对象是各自完整预处理后的输出；不拿旧prepare后指标冒充旧raw指标。',
        '', '## 5. 原冻结DEV同预算：15/180候选、13/36执行单元成功',
        '', '原DEV4＋已用8任务，3种子，每任务5候选。新候选180；旧aligned/强样条/条件均值540候选准确复用。逻辑720候选/144执行单元不等于本轮重新执行144条。',
        '', '|方法|合格候选|执行/重放成功（全单元分母）|候选位置均值/P95/最大 mm|候选姿态均值/P95/最大 °|', '|---|---:|---:|---|---|']
    for m,label in [('full323','新full323'),('old_aligned','旧aligned'),('structured','强样条（复用）'),('learned_mean','条件均值（复用）')]:
        r=d[m];f=lambda x:'/'.join(f'{x[k]:.3f}' for k in ('mean','p95','worst'))
        lines.append(f'|{label}|{r["eligible"]}/180|{r["execution"]["verified"]}/36|{f(r["position_mm"])}|{f(r["rotation_deg"])}|')
    lines += ['', '**新模型180/180候选都到达5 mm/1°目标，165条失败全部且仅因geometry_refinement。** 固定最高1024网格下几何运动步长超过2 mm：全180条均值2.555 mm、P95 3.490 mm、最大4.194 mm；旧模型最大1.290 mm。新DEV检查点的最小间距最低为+0.776 mm，没有观测到穿透；密度不足意味着资格未被证明，不能改记合格，更不能宣称连续时间无碰撞。',
        '', '原DEV4：3/60候选、3/12执行单元；已用8任务：12/120候选、10/24执行单元。三种子分别6/60、6/60、3/60候选，执行分别5/12、5/12、3/12。23个DEV单元没有合格候选，均未进入力矩执行。13个实际执行全部通过，故条件执行成功率13/13，但全流程成功率必须保留13/36。',
        '', '180条对应候选中，新位置误差有56条改善，124条变差；平均新减旧为+0.447 mm。13个双方均实际执行的同任务/种子单元，只有3个新位置误差更小，平均差+0.472 mm；另外23个单元不能因新模型未执行而被排除成功率分母。',
        '', '|实际执行误差（仅已执行样本）|样本数|位置均值/中位/P95/最大 mm|姿态均值/中位/P95/最大 °|', '|---|---:|---|---|']
    for m,label in [('full323','新full323'),('old_aligned','旧aligned')]:
        r=d[m]['execution'];f=lambda x:'/'.join(f'{x[k]:.3f}' for k in ('mean','median','p95','worst'))
        lines.append(f'|{label}|{r["attempted"]}|{f(r["final_position_mm"])}|{f(r["final_rotation_deg"])}|')
    lines += ['', '新配置未达到144/180候选及29/36执行单元的工程参考门；没有因失败追加训练或增加查询预算。逐任务×种子数据和保持段误差见 `evidence/completed/statistics.json` 的DEV_task_seed、DEV_common_execution_pairs和groups。',
        '', '## 6. 固定三/五障碍抓捕接近场景',
        '', '|场景|新候选|新目标门|新检测穿透|新密度门失败|旧aligned/强样条/条件均值合格候选|', '|---|---:|---:|---:|---:|---|']
    for scope in ['capture_3_obstacles','capture_5_obstacles']:
        g=s['groups'][scope];r=g['full323'];lines.append(f'|{scope}|{r["eligible"]}/5|{r["goal_pass"]}/5|{r["failed_gates"].get("geometry",0)}/5|{r["failed_gates"].get("geometry_refinement",0)}/5|'+ '/'.join(str(g[m]['eligible']) for m in ('old_aligned','structured','learned_mean'))+'（各5候选）|')
    lines += ['', '新模型两个场景均无执行候选。三障碍4条穿透；五障碍5条全部穿透，最小间距范围−82.983至−41.770 mm。因此障碍场景并非仅验证网格不足，实际避障也失败。虽然10/10终点门通过，无法替代整条路径避障资格。旧aligned三障碍5/5并1次执行成功；五障碍所有历史方法均0/5。',
        '', '网络条件仍不直接包含障碍，局部引导仍只有原24个修正方向；开放323维控制并没有同时扩大绕障优化空间。此前07存在一条执行合格路径，仅证明该特定场景可行，不证明当前采样器能找到它。本轮未使用07教师训练，也未运行动态不规则航点或实际接触抓取。',
        '', '## 7. 分级证据、预算和审计',
        '', '本轮新物理执行共16条＝3条TRAIN重建＋13条DEV，均通过原生67维力矩执行与独立重放。原生91500步，独立重放91500步，合计183000步；保存两套轨迹、完整初态、原生及中点几何。对照历史复用共114执行单元，其中110条实际执行成功（108 DEV＋2三障碍），另4个NO_CANDIDATE；不记为本轮新执行。',
        '', '全新的41个预选执行单元中25个NO_CANDIDATE（23 DEV＋2应用），分母完整。未执行合格候选仍只能是MODEL_VALID；VAL里仅模型门通过但目标失败的8条不能作为指定目标合格轨迹。无新数据入库，TRAIN_added=0。执行是在MuJoCo原生动力学中的真实力矩仿真，不是硬件验证。',
        '', '只读原始审计PASS、候选ERROR=0、633个账本单元已提交、无inflight/未知调用。实际新增耦合16321次，预留上界21800/30000，运行6小时上限内。7项软件测试仅证明实现检查，不代替物理实验。完成后resume仅核一次，返回NO_WORK_ALREADY_COMPLETE；596个运行文件清单和hash保持不变，4个原用户修改保持不变。',
        '', '冻结源码与原始运行数据保持不变，收尾辅助工具位于本包tools，证据、报告和审查分别写入evidence、docs和paper/review-traces。没有重新运行旧模型训练、旧物理审计或旧resume。新的独立审查复现全部300段批次/t/noise摘要，并重建保存的第1/5/3000/6000步神经VJP、合成梯度和手写AdamW；梯度、权重与优化器矩差异均为0。16套新原生/重放数组和保存门值另作只读核对；不新增optimizer.step或物理仿真。审查并非重新独立推导整套物理模型，也不是连续时间碰撞证明。',
        '', '独立审查结论见 [CLAIM_VERDICT.md](CLAIM_VERDICT.md) 和 `paper/review-traces/experiment-result-to-claim/20261021_full_control_run01/`。软件完成、表示/数据资格、实际执行和科学收益分别判定。',
        '', '## 8. 瓶颈与下一阶段判断',
        '', '1. **表示保真问题已在本轮独立TRAIN范围内解决，生成质量没有随之解决。** 已有教师重建6/6且3/3执行，证明323维足够承载这些轨迹；新网络无引导0/96且误差变大，不能以重建成功推断学会生成。',
        '', f'2. **完整输出失去了旧低维时间表示带来的平滑约束。** DEV控制点二阶差分RMS均值从旧 {old["control_second_difference_rms_rad"]["mean"]:.6f} rad增加到新 {b["control_second_difference_rms_rad"]["mean"]:.6f} rad，约 {b["control_second_difference_rms_rad"]["mean"]/old["control_second_difference_rms_rad"]["mean"]:.1f} 倍。该描述量不是实际加速度，但与大量2 mm密度失败一致。prepare保留可行控制也会保留网络生成的不必要时间起伏，不能指望它自动学习轨迹规律。不能通过重新强制平滑解决，因为已知这可能毁坏必要绕行。',
        '', '3. **当前控制MSE与世界终点仍未完全对齐，且从头初始化引入学习难度混杂。** 教师控制接近不保证自由漂浮后的世界端点接近；本轮没有直接世界终点损失。不能由这一次混合变化认定完整维数本身不可学，也不能无证据把问题归为微重力或MLP容量错误。',
        '', '4. **带引导完整方法满足端点门，全路径避障仍不足。** DEV180/180和应用10/10满足目标门，但本轮未单独识别引导的因果贡献。本轮24方向局部修正未使全部路径资格通过；完整输出中的多余形状变化和障碍绕行仍需解决。采样密度失败与实际穿透应分别解决。',
        '', '**建议下一轮只做一次有界、TRAIN先行的学习稳定性验证，不扩库、不TEST：** 保留完整323输出和资格保持prepare；优先评估函数保持的旧低频模型迁移＋全维细节残差分支，或完整可逆的多尺度时间表示，让新增细节从小幅开始，而不是让所有自由度从头同尺度采样。任何迁移先核对旧功能在TRAIN上的数值等价，不能直接宣称等价。',
        '', '在预先冻结、独立于DEV的少量TRAIN教师上，检查全维重建、完整采样教师拟合和时间差分误差是否共同下降；平滑/差分约束针对不必要抖动，不删除表示自由度、不修改已合格教师，不将旧07 DEV路径作为训练标签。先证明TRAIN可学且目标/几何资格被保留，再训练一个预定候选，用相同完整无引导和原DEV预算评估。不要仅因单一去噪loss下降就放行。',
        '', '障碍部分需要后续独立任务验证障碍条件或全路径敏感方向，但不与本轮退化自动混合调参。下一轮范围和数值门应在运行前冻结；本报告只是建议，未自动启动。当前应保留旧aligned用于已验证范围，新full323保留为负结果模型，不替换旧模型。',
        '', '## 9. 文件与终端',
        '', '- 冻结协议：[EXPERIMENT_PLAN.md](EXPERIMENT_PLAN.md)。',
        '- 紧凑统计：`ff_prior_full_control/evidence/completed/statistics.json`；逐条路径、hash、门和误差：`compact_rows.json`。',
        '- 原始权重、轨迹、执行/重放：`artifacts/v7_1/full_control_bounded_001`；保留本地，不上传大数据。',
        '- 独立审查与完成检查另行版本化；原始run没有改写为性能PASS。',
        '', '```powershell', r'.\ff_prior_full_control\job.ps1 status', r'.\ff_prior_full_control\job.ps1 report', r'.\ff_prior_full_control\job.ps1 resume', '```',
        '', 'resume已完成幂等核验，正常情况下无需反复执行；完成状态会直接返回不重复训练或物理工作。']
    (PACKAGE/'docs/ANALYSIS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


if __name__=='__main__':main()
