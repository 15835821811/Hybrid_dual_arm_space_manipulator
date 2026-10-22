"""Write closeout routing outside the immutable run; no experimental computation."""
from pathlib import Path
import json
ROOT=Path(__file__).resolve().parents[2]
package=ROOT/'ff_prior_static_goal_seed'
review='paper/review-traces/experiment-result-to-claim/20261022_static_goal_seed_run01'
p=package/'evidence/statistics.json'
s=json.loads(p.read_text(encoding='utf-8'));s['fresh_review']=review+'/verdict.json'
p.write_text(json.dumps(s,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
note='''
## P1唯一五障碍机制收尾，转P4（2026-10-22）

static_goal_seed_001生产源码2d0a4d4b；30公开提案、6完整验证、24仅初筛。30/30种子位姿达门，6/6完整验证仍穿透16.7–35.4mm且只几何失败；0模型合格、0实际执行。91种子迭代及84修正/76接受，29202耦合、18298744积分、2585.422秒。原worker末尾JSON类型审计错误保留FAILED，独立辅助只读审计141/141通过，未重跑物理。独立review整体/C1/C2 no/high，初始化子结论yes；5326/5327谓词为真，唯一额外Git树一致性为否，实际176源码归档/86输入和数值证据全一致。复现必须archive+manifest，用户旧QP修改不动。失败resume拒绝只测一次1254文件hash不变。

报告ff_prior_static_goal_seed/docs/ANALYSIS.md；review paper/review-traces/experiment-result-to-claim/20261022_static_goal_seed_run01/verdict.json。唯一同瓶颈机制机会已用；不再局部调参、不训练、不扩库、不TEST。P4继续最终组件/连续交接接口与固定开发套件，五障碍及其接近→动态组合依赖明确BLOCKED，不用三障碍替换。原工程全链路未完成，唯一接续保持active。当前阶段以control为准。
'''
with (ROOT/'ff_prior_fullchain/docs/ROADMAP.md').open('a',encoding='utf-8') as f:f.write(note)
(ROOT/'paper/EXPERIMENT_TRACKER_20261022_static_goal_seed_closeout.md').write_text('# P1机制收尾与P4路由\n'+note,encoding='utf-8')
findings=(ROOT/review/'findings.md').read_text(encoding='utf-8')
with (ROOT/'findings.md').open('a',encoding='utf-8') as f:f.write('\n'+findings+'\n')
goal='''# 当前有效执行目标：自由漂浮双臂工程全链路

更新：2026-10-22。长期目标ACTIVE；应用卡片内的旧“当前P1_001”是历史文字，真实阶段由`ff_prior_fullchain/control.json`与status决定。

从公开完整初态、障碍、世界目标和时序出发，生成并执行路径，完成刚性臂三/五障碍非接触预抓捕接近，再接移动卫星与连续体七不规则航点追踪。完整323或经认证时序表示、自由漂浮全机避障、原生67力矩、完整状态独立重放、学习/采样/共同修正/回退对照均须有真实证据。工程可行与Diffusion科学收益分开，后者不作为工程推进门；不保证正结论。

## 当前证据与缺口

|阶段|真实状态|后续边界|
|---|---|---|
|P0 preserve323|负结果已收尾；32梯度、64临时更新全部回退|不重跑，未有新模型|
|P1 静态三障碍|static_fullchain_002中6/6执行重放通过|公开输入路径来源保留；最终组件变化须另验身份|
|P1 静态五障碍|static_goal_seed_001：30/30种子达目标；6完整验证全部几何失败，24仅初筛，0执行|唯一机制机会已用；原worker审计故障独立只读收尾，禁止再resume或局部扫参|
|P2 动态表示/反馈|4/4既有TRAIN经认证；原静止/漂移局部工程门通过，失败保留|不能把分段成功写成五障碍连续组合成功|
|P3 时序学习|8TRAIN/4VAL；两模型各3000更新；无反馈VAL各0/4；共同反馈DEV各6/6逻辑|成功依赖共同反馈；未建立稳定Diffusion优势；不追加训练|
|P4 最终固定开发套件|正在实施/冻结组件与连续交接接口|三/五障碍、静止/漂移、五障碍接近→移动追踪五场景；缺依赖保留BLOCKED，不换任务|

最新报告：`ff_prior_static_goal_seed/docs/ANALYSIS.md`。独立审查C1/C2 no/high，目标初始化子结论yes；允许继续独立P4。末尾审计恢复141/141通过不等于碰撞通过。原FAILED及1254原文件保持；仅一次失败resume拒绝核验已完成。

## 接下来按部就班实施

1. P4先冻结最终组件、公开路径来源、准确任务/种子/物理合同和身份复用矩阵。保留五场景×三种子×两方法30逻辑执行机会，缺失必须在分母中。
2. 有准确同源码/模型/任务/控制的记录才复用；不同接口的新执行使用独立输出。完整INTEGRATION重放与2mm几何步长显式验收。
3. 实现实际状态连续交接，禁止拼接两个重置场景冒充一条轨迹；若五障碍无合格路径，则该组合BLOCKED，继续可独立验证部分。
4. 完成一次新证据审查、一次resume安全核验、报告/可视化/本地提交。路线P4上限2线程/6h/20GiB，不追加训练、局部同瓶颈修复或大规模搜索。

所有场景完整工程证据和连续组合真正通过后才完成长期工程目标。若固定范围仍有缺口，应交付部分结果及明确阻塞，不能用目标卡complete或科学优势代替工程证据。

## 始终保留的约束

仅不穿透、双臂世界5mm/1°、2mm几何步长、完整初态、17→67、自由漂浮、全机几何及原运动/力矩/重放门。无实际状态裁剪/重置，无失败重标，无mock，无抽查升级。DEV与07不入TRAIN；旧模型数据结果、用户修改和旧作业保留。

当前不扩库、不1024/30000生产、不pilot/study/strict_multimodal、不最终TEST、不硬件/接触捕获。查询与时间只记录，不要求胜出。“不惜代价”指持续机制改进，不是无限预算、放宽物理或保证正结论。

唯一heartbeat持续串行推进，健康worker只查status/PID创建时间/心跳后静默；终态先收尾再切换。源冻结后不可原地修补冒用身份。工程完成或用户停止才管理接续；当前未完成。

```powershell
.\\ff_prior_fullchain\\job.ps1 status
.\\ff_prior_fullchain\\job.ps1 next
.\\ff_prior_fullchain\\job.ps1 report
```

详细阶段协议和历史证据见`ff_prior_fullchain/docs/ROADMAP.md`；最新实际run、报告、下一步见`control.json`。
'''
(ROOT/'DIFFUSION_GOAL_CURRENT.md').write_text(goal,encoding='utf-8')
