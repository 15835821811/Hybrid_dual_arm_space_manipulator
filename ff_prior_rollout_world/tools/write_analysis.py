"""Render bounded experiment report from saved compact statistics."""
from pathlib import Path
import json
from ff_prior_rollout_world.protocol import PACKAGE

def main():
    s=json.loads((PACKAGE/'evidence/completed/statistics.json').read_text(encoding='utf-8'))
    lines=['''# 完整323采样世界监督与原合格保护：实测分析

2026-10-21；run `rollout_world_bounded_001`；冻结源码 `744c4f2d`，启动登记 `a888eb9e`。

## 结论

本轮有界流程已实际跑完，约 **3599秒（60分钟）**。固定TRAIN世界监督可学性得到支持，但**新模型无引导泛化退化，稳定Diffusion相对优势未建立**。原DEV的完整方法仍合格；固定五障碍抓捕仍未解决。暂不替换旧生产候选、不扩库、不启动TEST。

- M1：同8个TRAIN目标×2种子，全门从 **1/16→15/16**，原合格样本保留；64份完整12×323 Jacobian数值检查通过。达到原冻结训练门。
- 正式候选：14222条TRAIN全部参与，6000次普通更新、384000次样本抽取、1200批完整16步教师损失。随后世界细化在锚点上 **0/16→14/16**；32份完整Jacobian检查通过。只训练一个候选，M1临时权重没有用于该候选。
- 无引导VAL：旧 **24/96（25%）**，新 **4/96（4.17%）**。不是稳定改善。
- 原DEV：新旧Diffusion均 **180/180模型全门合格、36/36原生力矩执行及独立重放通过**。成功率没有提高，候选位姿尾部整体未改善。
- 应用DEV：三障碍 **5/5合格、1/1执行重放通过**；五障碍 **0/5合格、无候选可执行**，分母仍计为0/1执行单元。旧Diffusion同为5/10候选、1/2单元。

## 世界监督到底改善了什么

每个刷新点真实自由漂浮传播得到双臂世界残差，计算全部323方向Jacobian；再通过完整16步DDIM反传到网络。内层64步使用局部线性世界代理，**并非每个内层步都计算真实非线性传播**。每个提议随后用真实传播、全机几何与原门检查，再决定保留或回退。

M1共4轮256次提议更新，接受前三轮比例1、0.25、0.25；第四轮三种比例均不能保持资格，已拒绝并恢复。正式候选细化共2轮128次提议，第一轮比例1接受，第二轮均被保护/模型资格条件拒绝；最终保存第一轮权重。没有因结果不好追加训练。

|检查集|全门前→后|位置中位前→后 mm|位置最坏前→后 mm|姿态中位前→后 °|姿态最坏前→后 °|
|---|---:|---:|---:|---:|---:|''']
    for key,label in [('TRAIN_world','M1临时诊断'),('candidate_world','正式候选细化')]:
        d=s['TRAIN'][key];a=d['before'];b=d['after']
        lines.append(f"|{label}|{a['eligible']}/16→{b['eligible']}/16|{a['position_mm']['median']:.3f}→{b['position_mm']['median']:.3f}|{a['position_mm']['worst']:.3f}→{b['position_mm']['worst']:.3f}|{a['rotation_deg']['median']:.3f}→{b['rotation_deg']['median']:.3f}|{a['rotation_deg']['worst']:.3f}→{b['rotation_deg']['worst']:.3f}|")
    lines.append('''
Q定义为双臂位置/5mm、姿态/1°的最大值。M1 Q中位9.1097→0.3212，正式细化8.7953→0.6255。M1全部16条都满足终点位姿，但有1条几何失败；正式最终15条MODEL_VALID，14条同时达到指定目标，几何/姿态失败保留分母。

原保护索引12：初始3.188mm/0.724°，M1结束0.684mm/0.106°。正式普通6000步后退化为6.535mm/1.390°，世界细化恢复到0.837mm/0.205°。所以保护在**世界细化接受点、固定锚点**上成立；不能声称普通6000步或全部任务始终保资格。共96份完整Jacobian，未触发数值失败。

## 无引导泛化与尾部

每个种子48个VAL任务，完整16步无物理引导。新96条raw与prepare完全相同，网络失败没有被prepare修改掩盖。旧证据的prepare会修改raw，旧指标取其已保存验收输出。因此这是各自完整采样/prepare系统的对比，不能孤立归因世界监督。

|种子组|旧合格|新合格|旧位置中位/P95/最坏 mm|新位置中位/P95/最坏 mm|旧姿态中位/P95/最坏 °|新姿态中位/P95/最坏 °|
|---|---:|---:|---|---|---|---|''')
    def triple(d):return '/'.join(f'{d[k]:.3f}' for k in ['median','p95','worst'])
    for key,label in [('0','0'),('1','1'),('combined','合并')]:
        d=s['VAL'][key];a=d['old'];b=d['new'];n=a['total']
        lines.append(f"|{label}|{a['eligible']}/{n}|{b['eligible']}/{n}|{triple(a['position_mm'])}|{triple(b['position_mm'])}|{triple(a['rotation_deg'])}|{triple(b['rotation_deg'])}|")
    lines.append('''
合并位置均值13.769→16.046mm，姿态均值2.304→2.500°；位置P95/最大值有所下降，但位置中位和姿态最坏值恶化，不能选择有利尾部指标宣称整体改善。配对96条：旧合格24条中22条失去资格，仅保留2条，另新增2条合格。位置改善24/96、姿态改善32/96。模型资格93/96→94/96，说明主要障碍仍是指定目标精度，而非大规模数值发散或运动域失败。全部96有端点记录，没有剔除失败。

## 同预算完整方法与真实执行

原DEV4＋已用8任务，3种子，每方法每任务5候选；共同late4，113耦合/15872积分/15秒上限。以下候选位姿取每条双臂较差者；执行表取各5候选中按冻结规则选择的一条，不能把候选尾部与选择后执行尾部混为同一指标。

|方法|候选合格|执行重放成功|候选位置均值/中位/P95/最坏 mm|候选姿态均值/中位/P95/最坏 °|
|---|---:|---:|---|---|''')
    def quad(d):return '/'.join(f'{d[k]:.3f}' for k in ['mean','median','p95','worst'])
    for method in ['old_aligned','world323','structured','learned_mean']:
        d=s['DEV'][method];a=d['candidates'];e=d['execution']
        lines.append(f"|{method}|{a['eligible']}/{a['total']}|{e['success']}/{e['units']}|{quad(a['position_mm'])}|{quad(a['rotation_deg'])}|")
    lines.append('''
新模型位置配对改善102/180、姿态97/180，但少数恶化使均值/P95/最坏值增加；这种混合结果不支持稳定优势。强样条、条件均值也全部成功，不能把共同物理引导的成功归给Diffusion网络。

|方法|实际执行位置均值/中位/P95/最坏 mm|实际执行姿态均值/中位/P95/最坏 °|保持段位置最坏 mm|保持段姿态最坏 °|
|---|---|---|---:|---:|''')
    for method in ['old_aligned','world323','structured','learned_mean']:
        e=s['DEV'][method]['execution']
        lines.append(f"|{method}|{quad(e['position_mm'])}|{quad(e['rotation_deg'])}|{e['hold_position_mm']['worst']:.3f}|{e['hold_rotation_deg']['worst']:.3f}|")
    lines.append('''
新模型在选择后执行位置上有小幅改善（均值0.718→0.701mm），姿态均值略差。上述全部仍在原门内；36个任务种子单元并非36个独立任务，候选也共享任务，不能用180独立样本的推断夸大显著性。仅一轮候选训练，尚无训练种子重复，科学优势未成立。

### 静态避障抓捕的单独判断

三障碍新旧均5/5；新执行终点0.0344mm/0.00449°，旧0.2223mm/0.02951°。五障碍新旧均0/5：新终点最大仅0.383mm/0.0472°附近，但全部几何失败，最小间距约−30.25至−48.57mm，主要为前臂/腕部与`v7_capture_3`障碍相交。因此“更准地到目标”并没有产生合格绕行路径。这不是微小浮点越界，更不能通过放宽仅不穿透门解决。

本轮新执行共37条（DEV36＋三障碍1），独立重放均通过，共405500原生与重放步。五障碍单元没有合格候选，未进行力矩执行；不能写37/37等于全部38单元成功。旧对照DEV540候选/108执行单元及应用30候选/6单元只复用，其中应用实际2条成功执行；旧6教师重建和3教师执行亦仅复用。全部执行均为原生67维力矩MuJoCo物理仿真与独立重放，非硬件实测、非连续时间无碰撞证明。

## 具体瓶颈与下一阶段

1. **世界目标监督已能修正固定锚点，泛化保护仍不足。** 8任务16输出上的真实验收能防止这些输出退化，却没有约束其他条件/噪声。VAL旧合格22/24丢失符合分布外于锚点的能力退化现象，但本轮没有分别评价ordinary模型的VAL，不能把全部退化因果归于最后世界细化。
2. **普通训练与世界细化仍分段，目标不一致。** 6000步后锚点0/16，说明大规模普通去噪/教师误差优化没有稳定解决同一批世界精度。世界细化恢复14/16，却只覆盖很少条件。共享网络参数更新可同时改变低频旧功能与高频细节；可行初始化不保证训练后的旧功能保持。
3. **世界终点约束无法决定绕行拓扑。** 当前条件仍65维，不显式输入障碍几何；五障碍失败发生在整条路径中部。323表示容量/资格保持prepare已有可行性证据，但“表示能装下合格路径”不等于网络能根据场景生成它。现有24方向局部引导也未解决该场景。
4. **本轮设计不足以隔离因果。** 新旧同时存在表示、prepare、普通训练和世界细化差别；未运行同323正式ordinary模型的独立VAL对照。不凭该结果断言MLP不如Transformer或自由漂浮建模错误。

下一轮建议先做一次有界**功能保持与泛化保护**实验，继续使用现有TRAIN，不扩库：固定旧低102主干/或用旧完整采样函数回放约束其漂移；受控细节和世界残差分支训练；世界锚点按既有TRAIN分层轮换，设置独立TRAIN检查池，原合格保护覆盖不同目标与噪声，记录ordinary与世界阶段的独立输出。先确认检查池命中与尾部不退化，再一个候选、一次原DEV复核。阈值/训练预算须新协议事前冻结，不能用当前VAL选择温度/权重并重测。

静态五障碍应作为独立应用瓶颈：后续若授权，在独立TRAIN障碍任务里验证障碍条件编码及整轨迹碰撞损失/可行绕行引导；保留原07已执行DEV路径仅作评价，不混入TRAIN。不要再次只压终点误差就宣称避障抓捕闭环完成。现在不启动这些新实验、不扩库、不TEST。长期静态避障抓捕及动态航点目标仍未完成。

## 完整性、等级与证据

- 软件：7项预先冻结测试；正式流程M0—M4完成。软件通过不等于科学收益。
- 数据：无新增TRAIN，无DEV/VAL进入梯度或checkpoint选择；14222 TRAIN/48 VAL保持。模型合格记录不批量升级为执行认证，也未把新DEV轨迹加入训练库。
- 物理：仅不穿透、5mm/1°、2mm几何步长、全机几何、完整初态、自由漂浮、17→67、运动/力矩/重放门均保持。37条通过原生力矩执行及重放的记录各自是EXECUTION_VERIFIED；其余模型通过记录只算MODEL_VALID。
- 预算：实际耦合79107，预留耦合上界86222/112000，运行3599秒/6小时，输出约4.136GB/20GiB，2数学线程。没有候选未知错误或未决ledger。
- 原始只读审计PASS；新的独立审查已完成：C1在固定锚点范围yes，C2稳定相对收益no，总体partial；论文审查状态provisional（无PAPER_CLAIM_AUDIT）。见[CLAIM_VERDICT.md](CLAIM_VERDICT.md)。冻结运行的summary仍保留当时`independent_review=PENDING`，不回写原始文件；交付审查状态以本包审查与版本tracker为准。
- 完成resume仅核一次：`NO_WORK_ALREADY_COMPLETE`，3351文件集合/hash不变，冻结源码/输入身份不变，用户4项修改hash不变。保存[完成证明](../evidence/completed/completed_resume.json)。
- 全部逐任务/种子配对和缺失分母见[紧凑原始统计](../evidence/completed/statistics.json)，可用只读`tools/summarize_completed.py`复算。原始运行保留；本地提交不上传大数据。

## 原DEV逐任务/种子配对

每行5候选；终点分布包含全部候选。执行位置为冻结选择候选的真实末端误差。完整姿态分布、保持段及重放数据见statistics.json。

|任务|种子|旧/新合格|旧位置均值/中位/P95/最坏 mm|新位置均值/中位/P95/最坏 mm|旧/新执行位置 mm|
|---|---:|---|---|---|---|''')
    for u in s['DEV']['units']:
        lines.append(f"|{u['task']}|{u['seed_group']}|{u['old']['eligible']}/5；{u['new']['eligible']}/5|{quad(u['old']['position_mm'])}|{quad(u['new']['position_mm'])}|{u['old_execution']['position_mm']:.3f}/{u['new_execution']['position_mm']:.3f}|")
    lines.append('''
终端查询（在仓库根目录）：
```powershell
.\\ff_prior_rollout_world\\job.ps1 status
.\\ff_prior_rollout_world\\job.ps1 report
.\\ff_prior_rollout_world\\job.ps1 resume
```
resume已验证完成时不重复工作；无需再用聊天轮询。
''')
    (PACKAGE/'docs/ANALYSIS.md').write_text('\n'.join(lines),encoding='utf-8')
    print('ANALYSIS.md written from saved evidence')

if __name__=='__main__':main()
