# 当前目标：原00/07可行步恢复与自适应信赖域

本轮保持原静态避障抓捕→后续动态航点的长期目标。上一轮目标归档ff_prior_capture_trust/evidence/previous_goal.md。
当前阶段是五障碍路径求解，不是新Diffusion训练。冻结协议ff_prior_capture_trust/docs/EXPERIMENT_PLAN.md。

|阶段|本轮内容|状态|
|---|---|---|
|M0|10软件测试、133源码/原输入冻结、后台传播一致性|软件PASS；物理一致性待运行|
|M1|原00/07上一轮最终Z，零步射线恢复、实际自适应信赖域、最多各12次|PREPARED|
|M2|严格合格后原生67执行重放及codec/prepare/102诊断|条件未满足前BLOCKED|
|M3|原始审计、新独立review、一次resume、报告与提交|待完成|

初始0.0375rad、最小0.001171875、最大0.15；同活动集实际/预测ratio≥0.1才可接受，仍需原全路径改善与物理分项目标门。
12次含拒绝重线性化，不盲目加预算或换seed。仅不穿透、5mm/1度、2mm、完整初态/自由漂浮/17→67/全机几何/力矩重放保持。
2数学库线程/4小时/12GiB输出/12000耦合/2400万积分区间。不训练、扩库、pilot、study、strict_multimodal或TEST。
无见证只报告本次有界未找到，不证明不可达，不归因102维或Diffusion架构；不自动追加下一实验。
输出artifacts/v7_1/capture_trust_001；终端 .\ff_prior_capture_trust\job.ps1 status|report|resume。
