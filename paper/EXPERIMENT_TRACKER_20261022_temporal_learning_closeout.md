# P3b新证据收尾与下一阶段

## P3b时序学习完成，转P1五障碍机制修复

2026-10-22，temporal_learning_001（源码681965a0）真实结束，worker820.125秒。两方法均完成32次临时诊断并恢复，再各3000正式更新；固定8TRAIN/4VAL，新增数据0。无世界反馈VAL两法各0/4，原DEV共同反馈两法各6/6逻辑，实际det2/2+Diffusion6/6，另外4det共享；全部16实际轨迹完整836重放差0，失败仍8/16不升级。新原生/重放349934步，完成resume仅一次278文件hash不变。

独立review overall partial/high，局部动态工程C1 yes、跨场景统一C2 no；1334项新证据核对无不一致，缺PAPER_CLAIM_AUDIT，论文主张provisional。VAL Diffusion原加速度凸包12.85–19.85倍导致prepare缩到5.04%–7.78%，目标运动损坏；确定性VAL无prepare变化但有世界误差和穿透。DEV漂移尾部改善、静止位置退化，成功率相同；两个任务不构成未见泛化/稳定优势，反馈贡献不能归给网络。TRAIN/VAL有2球、DEV无外加障碍且抓捕局部y相差3cm，不把跨域VAL/DEV作为控制消融。

报告ff_prior_temporal_learning/docs/ANALYSIS.md。下一步读ff_prior_fullchain/docs/P1_REMAINING_HANDOFF.md，以公开非线性目标初始化→几何绕行→323修正补原五障碍唯一机制机会；具体代码/任务/预算另冻后启动。新网络不追加训练，P1剩余64566耦合/14021.469秒等原余额不增。之后进入P4真实连续组合验收；工程全链路未达成，goal/唯一接续继续active。不扩库、不放宽原物理、不TEST。

未来新执行版本应把完整836 INTEGRATION重放误差显式纳入接受门；P3b旧evaluate仅对qpos/qvel门控，虽本run全部fullerr确为0，冻结源码不原地改写或重跑。
