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
|P3时序候选|P3A_RUNNING|预选8TRAIN/4VAL；4历史复用+8新执行上限；正式训练未开始|
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
