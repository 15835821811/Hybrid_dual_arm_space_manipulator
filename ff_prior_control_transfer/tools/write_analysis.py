"""Chinese bounded negative-result report from saved M1 statistics."""
from ff_prior.storage import read_json
from ff_prior_control_transfer.protocol import PACKAGE


def main():
    s=read_json(PACKAGE/'evidence/completed/statistics.json');g=s['gate'];a=s['diagnostic']['before'];b=s['diagnostic']['after']
    lines=['# 完整323控制功能保持迁移与TRAIN可学性：有界结果',
        '', '2026-10-21；run `control_transfer_bounded_001`；源码 `df13c4e1`，启动记录 `7d32e4e2`。',
        '', f'**后台按冻结协议结束，耗时{s["status"]["elapsed_s"]:.2f}秒。功能保持迁移、教师重建和时间规律控制得到支持，TRAIN可学性综合门未通过，因此6000步候选训练、VAL无引导诊断与冻结DEV对照均没有启动。** 本轮不是新候选失败的DEV实验，也不能给出新模型的DEV成功率或相对收益。',
        '', '原始状态FINISHED_WITH_BLOCKED_ITEMS，原始审计PASS表示证据和账本一致，不代表可学性门通过。本轮未调门或追加试验；旧模型和全部历史结果保留。',
        '', '## 1. 实际机制与数据范围',
        '', '完整323控制仍为19×17。保留旧aligned低102维网络参数、6列时间基、条件中心和尺度，用固定正交补齐剩余13列；221维细节clean头初始输出为0。低102噪声按旧种子独立抽取，细节采用另一个固定随机流。DDIM对低维沿用v预测，细节直接预测clean，epsilon统一从原noisy与clean重新计算。',
        '', '细节头使用有界tanh，尺度及幅度界只来自既有14222 TRAIN。实际所有细节尺度为1e-4 rad，每维归一化界为3.000—5.557；完整codec仍可表示323个方向，网络输出幅度受当前TRAIN支持限制，不能宣称覆盖任意未见大绕行或07路径。没有用DEV调幅度。',
        '', '保留资格保持prepare：快门满足则原控制精确保留；不强制平滑、逐元素裁剪或低秩压缩。快门之后仍需完整物理和目标验收。网络条件仍为65维，不直接编码障碍，后续预定局部引导仍24方向。',
        '', '所有14222 TRAIN/48 VAL只复用旧数据，DEV不进入拟合/诊断/训练。M1固定8个TRAIN任务（6条complex_train教师＋按样本ID hash预选的coverage_train_00288、pilot_train_07198），每任务2种子，共16条。名称含pilot的记录是既有TRAIN数据，本轮未启动pilot生产。',
        '', '## 2. 功能保持与重建：通过',
        '', f'- 16条完整16步无引导raw控制与旧模型最大差 **{s["migration"]["max_control_error_rad"]:.3g} rad**，小于预冻结1e-12 rad。仅证明迁移初态、这些固定TRAIN样本的raw功能保持；不是任意输入的全局数值证明，也不是新旧带引导完整方法等价。',
        '- 6/6独立TRAIN教师经实际float32训练目标编码、解码与prepare后，完整模型及5mm/1°目标门通过。',
        '- 预选3条static_detour的新原生67维力矩执行和独立重放全部通过；没有失败补位，也没有把网络诊断输出冒充教师重建执行。',
        '', f'这3条新执行终点最大位置误差 {s["reconstruction"]["execution"]["final_position_mm"]["worst"]:.6f} mm，保持段最大位置误差 {s["reconstruction"]["execution"]["hold_position_mm"]["worst"]:.6f} mm。它们仍是同一批既有教师的本轮重建验证，TRAIN_added=0。',
        '', '## 3. 256步临时TRAIN可学性诊断',
        '', '固定16个条件/种子，每步完整展开16步DDIM，共256个真实AdamW更新。目标为完整物理控制教师MSE＋0.25倍教师误差二阶时间差分MSE；没有真实非线性世界终点梯度。最终权重保存为临时诊断证据，未转作正式候选，正式训练目录不存在。',
        '', '|指标|更新前|256步后|', '|---|---:|---:|',
        f'|固定教师损失|{g["before_loss"]:.6f}|{g["after_loss"]:.6f}|',
        f'|完整模型门通过|{a["model_valid"]}/16|{b["model_valid"]}/16|',
        f'|双臂位置≤5 mm|{a["position_pass"]}/16|{b["position_pass"]}/16|',
        f'|双臂姿态≤1°|{a["rotation_pass"]}/16|{b["rotation_pass"]}/16|',
        f'|模型及指定目标全部通过|{a["eligible"]}/16|{b["eligible"]}/16|',
        f'|归一化终点Q中位数|{g["before"]["median_Q"]:.6f}|{g["after"]["median_Q"]:.6f}|',
        f'|控制点二阶差分RMS rad|{g["before"]["roughness_rms_rad"]:.8f}|{g["after"]["roughness_rms_rad"]:.8f}|',
        '', f'教师损失下降 **{100*s["loss_relative_reduction"]:.2f}%**，Q中位下降 **{100*s["Q_median_relative_reduction"]:.2f}%**；15/16条Q改善，1/16退化。Q=max(双臂位置误差/5mm、双臂姿态误差/1°)，Q≤1才满足原目标门。',
        '', '|世界误差（每条取双臂较大值，全16条含失败）|均值|中位|P95|最大|', '|---|---:|---:|---:|---:|']
    for phase,label in [('before','更新前'),('after','更新后')]:
        for k,unit in [('position_mm','位置mm'),('rotation_deg','姿态°')]:
            r=s['diagnostic'][phase][k];lines.append(f'|{label}{unit}|'+ '|'.join(f'{r[v]:.4f}' for v in ('mean','median','p95','worst'))+'|')
    lines += ['', '前后32条世界误差均有实际传播结果，无缺失。所有32条raw与prepare逐元素相同，prepare本轮没有修改这些提案。更新前3条存在检测穿透；更新后16/16全部模型门通过，最大几何运动步长1.5424 mm，未出现上一轮的大量密度失败。本轮后验模型通过仍是离散检查，不是连续时间/硬件安全证明。',
        '', '## 4. 为什么没有进入正式候选训练',
        '', '|预冻结门|结果|判定|', '|---|---|---|',
        '|初态功能迁移、6重建、3执行|16/16等价；6/6；3/3|通过|',
        '|固定教师损失下降至少10%|下降99.78%|通过|',
        '|Q中位下降至少10%|下降79.63%|通过|',
        '|时间差分RMS≤1.10×max(原输出,教师)|0.001372≤0.001759 rad|通过|',
        '|至少12/16全门合格|仅6/16|**未通过**|',
        '|合格数不下降|1→6|通过|',
        '|原本合格的逐条仍合格|唯一原合格样本退化|**未通过**|',
        '|权重真实变化、初态可恢复|通过|通过|',
        '', '**直接瓶颈是世界位置精度。** 更新后10条不合格都超出5mm位置门，其中1条同时超出1°姿态门；不是模型数值发散，也不是时间密度门卡住。不能把学习平均改善改写为已满足所有预定放行条件。',
        '', '关键退化案例：`coverage_train_00288`，seed20270213。',
        '', '|指标|之前|之后|', '|---|---:|---:|',
        '|世界位置误差 mm|3.1883|12.3691|', '|世界姿态误差 °|0.7245|1.1185|',
        '|教师控制RMSE rad|0.030300|0.001936|',
        '', '**更接近同一教师控制轨迹，并不保证世界终点误差单调下降。** 这个样本的控制RMSE约降低93.6%，却从原目标合格变成不合格。该实测现象直接说明当前平均控制误差目标不能替代逐条世界目标验收。不能仅由这点确定是某个物理模型错误；映射敏感性、轨迹误差方向和多样本共享更新都可能参与，单次实验尚未分离其因果。',
        '', '## 5. 全16条配对结果',
        '', '|TRAIN目标|种子|位置前→后 mm|姿态前→后 °|全门前→后|', '|---|---:|---:|---:|---|']
    for p in s['pairs']:
        lines.append(f'|{p["sample_id"]}|{p["seed"]}|{p["before_position_mm"]:.3f}→{p["after_position_mm"]:.3f}|{p["before_rotation_deg"]:.3f}→{p["after_rotation_deg"]:.3f}|{int(p["before_eligible"])}→{int(p["after_eligible"])}|')
    lines += ['', '这是同一16条TRAIN条件/种子的有界过拟合诊断，不是新的VAL/DEV实验。它支持有限TRAIN内的可优化性改善，不支持泛化、Diffusion相对样条收益或新模型最终可用。',
        '', '## 6. 本轮实际运行与未运行',
        '', '|内容|本轮状态|', '|---|---|',
        '|软件测试|8/8通过，仅软件证据|',
        '|教师重建/新执行|6/6模型与目标，3/3执行及独立重放|',
        '|迁移与临时TRAIN诊断|16条功能检查，256次更新，前后32次物理验证|',
        '|正式14222 TRAIN、6000步候选训练|**BLOCKED，未运行**|',
        '|VAL48×2完整无引导|**BLOCKED，未运行**|',
        '|原DEV180候选/36执行单元与应用10/2|**BLOCKED，未运行**|',
        '|新增数据/扩库/pilot/TEST|未运行|',
        '', '本轮真实执行只来自3条教师重建，原生与重放合计46500步；诊断后6条目标合格提案未执行，不能升级为EXECUTION_VERIFIED，也没有加入TRAIN。旧540/108等对照只被绑定备用，本轮下游未启动，不能报告成新旧DEV对照已完成。',
        '', '原始审计PASS，无候选ERROR、无inflight或未知账本工作；42个已完成单元。新增实际耦合76次，预留266/33000，壁钟135秒，未触发资源上限。旧结果、用户4个未提交修改与作业保留；完成resume仅执行一次，NO_WORK_ALREADY_COMPLETE，452个运行文件清单与hash不变。',
        '', '## 7. 审查与证据边界',
        '', '新的独立experiment-result-to-claim审查读取170源码/15412输入绑定、旧权重/codec、原始迁移控制、临时256更新状态、首末VJP/手写AdamW、前后32条物理预测和3套执行重放证据。详见 [CLAIM_VERDICT.md](CLAIM_VERDICT.md) 和 `paper/review-traces/experiment-result-to-claim/20261021_control_transfer_run01/`。不把只读审计描述为重新跑了一次物理实验。',
        '', '统计与逐条来源/hash见 `evidence/completed/statistics.json`、`diagnostic_rows.json`；原始权重和完整预测/执行数组保留在独立run。原始summary仍保留结束时的独立审查PENDING快照，后续审查单独版本化，未改写冻结运行数据。论文claim audit如不可用，写作结论保持provisional。',
        '', '## 8. 下一阶段建议（未自动执行）',
        '', '保留本轮已验证的功能保持迁移和受控细节；先不换架构、不增维、不扩库，也不通过增加步数或放宽物理门追认本轮通过。下一步应集中修正**完整采样训练目标与逐条世界终点验收的偏差**。',
        '', '建议冻结同一TRAIN问题上的一次有界方向验证：在完整16步最终输出上，对超出5mm/1°的世界误差分项给予明确监督或约束，并对原已合格样本加入保资格约束；核对训练参数更新是否同时降低违规样本误差、避免原合格样本越门。需要直接核实际世界终点效果，不仅看教师MSE。实施前应冻结导数路径、数值检查、幅度和预算，避免又做多轮未解释的调参。',
        '', '本轮前后已经揭示控制MSE与世界精度的差异，后续不需要重新重复证明323重建和初态迁移；可按身份复用合格证据。若新的TRAIN方向验证通过，再训练一个预定候选、完成无引导及原DEV同预算对照。当前没有依据启动TEST或宣称扩库会解决终点精度。以上是下一协议建议，本轮未授权范围内不自动追加。',
        '', '## 9. 终端与文件',
        '', '```powershell',r'.\ff_prior_control_transfer\job.ps1 status',r'.\ff_prior_control_transfer\job.ps1 report',r'.\ff_prior_control_transfer\job.ps1 resume','```',
        '', '冻结计划：[EXPERIMENT_PLAN.md](EXPERIMENT_PLAN.md)。本轮已结束，正常无需重复resume；已验证它不会再次启动临时训练或物理工作。长期静态避障抓捕与动态航点科学目标仍未完成。']
    (PACKAGE/'docs/ANALYSIS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


if __name__=='__main__':main()
