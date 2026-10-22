# Experiment Tracker：持续工程全链路

更新2026-10-22。P2a第一版结束、独立审查和完成resume已收尾；唯一响应余量修复已实际完成。长期工程目标ACTIVE；P1五障碍、P2b和学习链仍未完成。

|Run/阶段|真实状态|证据与限制|
|---|---|---|
|P0 preserve323|CLOSED_NEGATIVE|3e659ce4，禁止重复resume/审计|
|static_fullchain_001|CLOSED_DEPLOYMENT_FAILURE|未知调用保留|
|static_fullchain_002|CLOSED_PARTIAL|三障碍6/6单元执行重放，五障碍0/6；60提案、23完整验证、1预算未决、36仅筛选，12合格6实际执行|
|dynamic_reference_001|CLOSED_PARTIAL|4/4参考、3/4真实执行，四次完整重放差0；t00仅实际加速度失败。108000步，未新增TRAIN或训练|
|dynamic_margin_001|CLOSED_LOCAL_PASS|1新执行通过+3精确历史复用，组合4/4；实际加速度0.970613倍原限；旧失败保留|
|P2b动态DEV交接|CLOSED_LOCAL_PASS|6/8独特、10/12逻辑；先验4/6、基线2/2；8完整重放差0，收益不支持|
|P3时序候选|P3B_RUNNING|temporal_learning_001；先TRAIN诊断，条件正式训练/VAL/原DEV；结果待实际运行|
|P4整链开发验收|QUEUED|最多30执行，五障碍与完整连续组合仍缺|

新独立review：paper/review-traces/experiment-result-to-claim/20261022_dynamic_reference_run01；C1局部partial/high，C2 NOT_EVALUATED；缺少PAPER_CLAIM_AUDIT，论文结论provisional。166源码/86输入和完整状态/控制/几何/原目标核对通过；不能把表示过门当执行通过。完成resume仅一次，64核心文件hash完全保持，UTF-8完整输出已存。

报告ff_prior_dynamic_reference/docs/ANALYSIS.md。修复经验rho来自已保存TRAIN实际加速度与参考之差，须新执行；旧失败永久保留。P2总步数432000不增，修复27000从未启动P2b转移；只运行一次，不调门、不扩库、不启动TEST。控制文件与run/status为实时事实。

修复源码d5ba2106，启动回执ff_prior_dynamic_margin/evidence/launch_receipt.json（PID创建时间匹配）；唯一接续已更新，原P2a收尾提交95735ab0。实时结果以统一status为准。

修复实际76.985秒/27000新步，完整状态重放差0，完成resume仅一次46文件不变。零训练/扩库/DEV/TEST；五障碍与长期学习目标仍未完成。

新review已完成：20261022_dynamic_margin_run01，局部P2a yes/PASS，整体partial/high，C2未评估；不预判P2b。

P2b dynamic_handoff_001源码549c4a2e已实际后台启动；168源码/86输入，两原任务身份精确。12逻辑单位、最多8独特执行+4精确共享，1线程/4h/12GiB/297000步上限；网络6×16完整DDIM，无训练。新baseline观测补齐完整中间状态，不把共享算独立重复。P2a收尾a10cb8dd。

## P2b实际完成与P3资格交接

dynamic_handoff_001已完成新独立审查：6/8独特实际执行、10/12逻辑单位通过；基线2/2独立、先验4/6，另外4基线逻辑记录共享。8份完整836维重放差0。两漂移先验连续体最坏6.753/7.626mm，保留失败；C2收益不支持。局部“每任务×方法至少一条”工程门PASS，整体partial/high。216000新原生/重放步，P2累计351000≤432000；完成resume仅一次135文件不变。报告ff_prior_dynamic_handoff/docs/ANALYSIS.md。

转P3a：固定既有32TRAIN/8VAL来源，预选前8TRAIN/前4VAL；前4TRAIN精确复用，余8条至多一次新执行重放，失败不替换。只认证现有来源，不扩库；其余28条不升级。至少8/4合格后才另冻一个时序候选与序列基线。学习门失败保留BLOCKED并继续原P1五障碍独立修复。新阶段是否已启动仅以control/status为准。

P3a temporal_data_001已实际后台启动，源码292021c2；29项软件检查通过，214输入文件绑定。固定8TRAIN/4VAL，4精确历史复用+最多8新执行重放；剩余28来源不升级。单线程/3000秒/5GiB/216000新步，上限来自P3总6h/20GiB。原学习门8/4不变，训练尚未启动。当前真实阶段/PID心跳以control/status为准。

## P3a当前资格完成

temporal_data_001已完成：固定8TRAIN/4VAL全部EXECUTION_VERIFIED，4精确旧复用+8新执行重放，剩余28来源不升级。新最大位置2.112068mm/姿态0.021865度，完整836维重放差0。独立review局部PASS、整体partial/high，C2未评估。耗时377.453秒/216000新步/1152681165字节；完成resume仅一次146文件hash不变。报告ff_prior_temporal_data/docs/ANALYSIS.md。进入一个完整时序候选及确定性序列基线的实施/冻结，训练尚未开始；后续预算扣已用，当前run以control为准。

P3b temporal_learning_001已实际后台启动，源码681965a0，40软件检查通过（零真实优化器/原生步）。新公共条件1014维+271×36局部时序，完整273×17参考中271×17自由输出；两个175361参数时序卷积网络。先各32步临时TRAIN完整采样诊断且回退，通过才各3000正式更新。仅8TRAIN/4VAL，VAL无梯度。完整采样系数监督不冒充当前非线性世界损失。随后8VAL无世界修正执行上限、原DEV12逻辑/8独特执行上限，全部物理门不变；当前正式训练进度以status为准。新预算18000秒/16GiB/432000步，与P3a实耗合计低于P3原6h/20GiB。

## P3b时序学习完成，转P1五障碍机制修复

2026-10-22，temporal_learning_001（源码681965a0）真实结束，worker820.125秒。两方法均完成32次临时诊断并恢复，再各3000正式更新；固定8TRAIN/4VAL，新增数据0。无世界反馈VAL两法各0/4，原DEV共同反馈两法各6/6逻辑，实际det2/2+Diffusion6/6，另外4det共享；全部16实际轨迹完整836重放差0，失败仍8/16不升级。新原生/重放349934步，完成resume仅一次278文件hash不变。

独立review overall partial/high，局部动态工程C1 yes、跨场景统一C2 no；1334项新证据核对无不一致，缺PAPER_CLAIM_AUDIT，论文主张provisional。VAL Diffusion原加速度凸包12.85–19.85倍导致prepare缩到5.04%–7.78%，目标运动损坏；确定性VAL无prepare变化但有世界误差和穿透。DEV漂移尾部改善、静止位置退化，成功率相同；两个任务不构成未见泛化/稳定优势，反馈贡献不能归给网络。TRAIN/VAL有2球、DEV无外加障碍且抓捕局部y相差3cm，不把跨域VAL/DEV作为控制消融。

报告ff_prior_temporal_learning/docs/ANALYSIS.md。下一步读ff_prior_fullchain/docs/P1_REMAINING_HANDOFF.md，以公开非线性目标初始化→几何绕行→323修正补原五障碍唯一机制机会；具体代码/任务/预算另冻后启动。新网络不追加训练，P1剩余64566耦合/14021.469秒等原余额不增。之后进入P4真实连续组合验收；工程全链路未达成，goal/唯一接续继续active。不扩库、不放宽原物理、不TEST。

未来新执行版本应把完整836 INTEGRATION重放误差显式纳入接受门；P3b旧evaluate仅对qpos/qvel门控，虽本run全部fullerr确为0，冻结源码不原地改写或重跑。
