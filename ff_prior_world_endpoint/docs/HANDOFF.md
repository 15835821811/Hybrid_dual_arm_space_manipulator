# 后台运行和收尾交接

本轮程序与预注册协议提交7bac71d5；上一轮审计报告收尾提交b5921b54。运行world_endpoint_smoke_001，后台worker191828，启动记录evidence/launch.json。App线程heartbeat automationId=automation，每30分钟仅先读status/PID；未完成不重复分析，不重新启动实验。完成后按本文件和EXPERIMENT_PLAN.md收尾，交付报告后停用本接续。

## 当前承诺

保持原1024槽扩库未启动，不运行pilot。当前worker没有TEST路径。条件授权的最终TEST必须先全部预注册就绪门通过并经独立审查，再评估TRAIN覆盖与扩库必要性，最后冻结未触碰TEST协议和最终模型；不能用TEST调参重测。所有旧数据、失败、分支、用户修改保持。

## 完成时检查

1. 检查状态、worker退出、summary、data_summary、data_verification、training.json、independent_verification及ledger；实际完成和PASS不能混淆。失败时保持原输出，诊断具体软件或学习原因，不降低门。
2. 16固定TRAIN世界目标最多64修正、16预选执行重放；新训练全部旧13774及新增合格行，6000更新/384000消费，120次当前预测全102真实终点梯度，25440梯度传播。32TRAIN独立噪声终点探针，属于TRAIN诊断。
3. 原DEV4+已用8开发任务、3采样种子、4方法各5候选共720；144预选真实67力矩及独立重放、96固定raw/预处理诊断。八任务已用于设计，明确REUSED_DEV。当前冻结旧模型是13774模型。
4. 按experiment-result-to-claim技能委派fresh reviewer，fork none，原始证据直接核验；保存prompt/inputs/response/verdict/handoff。无paper claim audit只限制论文写作确认，不阻止事实审计。不要复用旧评审代替新结果。
5. 退出后实际执行resume，核对status/summary/ledger/launcher SHA不变且NO_WORK_ALREADY_COMPLETE。不能运行未知中断的物理任务来冒充完成。
6. 写中文docs/ANALYSIS.md，更新DIFFUSION_GOAL_CURRENT.md、ff_prior_feasible/docs/CORE_GOAL.md及版本化paper tracker；必要时明确神经收益和最终TEST仍BLOCKED。
7. 只提交本轮文件。保留.gitattributes、.gitignore、v6_lite/hierarchical_qp.py、v6_lite/run_v6_lite.py用户修改；不stage既有未跟踪raw_goal_diagnosis.json、root findings/MANIFEST全文件或其他无关内容。root notes可只追加自己的审查记录，不混入提交。
8. 最终答复含报告链接、实测摘要、门槛、终端命令及提交；停用automation接续。长期目标未完成不能调用complete。

## 说明

102维中央差分给PyTorch提供当前预测处的非线性耦合终点梯度；这是模型训练监督，既非教师局部线性替代，也非硬件实验或力矩认证。实际执行是MuJoCo真实67通道力矩和完整初态重放，单独证据。新LOSS、额外训练和新样本同时变化，一个训练种子，不能认定单因素因果或广泛优越。

命令：`.\ff_prior_world_endpoint\job.ps1 status|report|resume`。后台完成自动生成artifacts/v7_1/world_endpoint_smoke_001/REPORT.md，人工分析与独立审查收尾写docs/ANALYSIS.md。运行中不要更改冻结Python源码。
