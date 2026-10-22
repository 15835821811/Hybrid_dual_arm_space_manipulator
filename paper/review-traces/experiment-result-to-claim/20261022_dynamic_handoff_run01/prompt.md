# 独立 result-to-claim 审查任务

2026-10-22；角色：独立 paper architect/reviewer。按 `C:/Users/admin/.codex/skills/experiment-result-to-claim/SKILL.md` 与 `shared-references/review-tracing.md` 执行。此目录为本次新审查，不沿用旧物理审查结论代替新证据。

## 收到的原始有界委派

> 按 C:/Users/admin/.codex/skills/experiment-result-to-claim/SKILL.md 做全新独立只读result-to-claim审查，用户/skill明确委派。工作区E:/桌面/mujoco_irregular_waypoints_diffusion_adapter_bundle_20260902-main/Hybrid_dual_arm_space_manipulator。新run artifacts/v7_1/dynamic_handoff_001，源码549c4a2e，协议ff_prior_dynamic_handoff/docs/EXPERIMENT_PLAN.md，168源/86输入。只审本次新原始结果，不重审P1/P2a/旧domain物理结果。原始汇总：8真实新执行+8独立重放，每13500步，12逻辑单位含4同run确定性基线共享；原两冻结任务/初态、每场景三采样噪声。基线2独特执行合格（6逻辑），终点先验6独特执行4通过，漂移u09/u11仅dynamic_position失败，连续体峰值6.7532/7.6264mm，其他原门过。网络6×16=96真实DDIM，零训练/扩库/TEST。终态345.547s、216000步骤、约1.124GB、无inflight。检查raw来源任务/initial/166?实际168源码zip与86输入，6份DDIM原始输出可只读重建model/seed/update，完整836INTEGRATION/native67力矩/世界时序目标/全机几何与边界4.5/25.5无重置、独立controls replay、全部失败分母、共享基线身份不当独立重复。world position5mm/1deg、仅不穿透和所有原门保持。共同反馈控制器+零空间终点先验不是完整时序网络，原始Z未经动态验收不能MODEL_VALID。工程局部门按事前协议“两场景×两方法各至少1完整成功+原始审计”与整体C1/C2分开，不能用FINISHED_WITH_BLOCKED_ITEMS推断无任何推进资格，也不能写12/12。分析u09/u11保存数据中失败时段与control_filter关系可做代数，但无反事实不能断言唯一因果。仅保存数据/神经预测/代数，不新增仿真/optimizer/resume，不改源码。输出新trace paper/review-traces/experiment-result-to-claim/20261022_dynamic_handoff_run01，仅写该目录，包含prompt/inputs/response/verdict/routing/handoff和必要audit。缺PAPER_CLAIM_AUDIT论文provisional；长期五障碍/P3/P4仍未完成。完成告知，关键发现可先发。

## 实际判定问题

1. 原始证据是否满足事前 P2b 工程局部门，允许推进后续已授权依赖工作？
2. 本轮对完整工程 C1 和 Diffusion 贡献 C2 各支持什么、不支持什么？
3. 是否存在来源、分母、物理门、连续状态、采样或归因的不一致？
4. 提供 `claim_supported`、支持/不支持范围、缺口、建议改写、下一步和置信度；缺论文 claim audit 时标为 provisional。

## 输入

- `ff_prior_dynamic_handoff/docs/EXPERIMENT_PLAN.md` 与 `paper/EXPERIMENT_PLAN.md` 的当前主张、预注册推进门。
- `artifacts/v7_1/dynamic_handoff_001/run.json`、源码 zip、status/budget/model_loaded、全部 u00–u11 任务/初态/身份/结果与四个复用记录。
- 八条唯一执行的 native/replay/geometry NPZ、几何最小对记录、control_filter、两切换边界记录，以及六条 prior 原始输出。
- 全部 168 冻结源码与 86 绑定输入只做身份核对；其中执行、记录、验收、重放、参考与采样实现按本次调用路径阅读。
- 旧任务仅比较冻结 task/initial 身份；旧 P1/P2a/domain 物理结果不重新审查。
- 固定 checkpoint、codec 与 checkpoint 验证所需 training.json；只读神经前向重建六份既有输出。
- 完整输入路径、SHA-256、字节数和时间见 `inputs.json`。

明确禁止新增物理步、几何查询、优化器、训练、扩库、TEST、resume；不得改运行数据或源码。只向本目录写审查记录。审查不会把建议补实验自动执行，也不把未通过的两个样本删除。
