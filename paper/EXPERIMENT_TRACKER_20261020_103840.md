# 端点分组Jacobian与解析平滑链式梯度：执行状态

|阶段|门槛/工作|状态|
|---|---|---|
|M0|继承原24对/32事件目标/1096输入绑定，新源码|FROZEN ecaf2519|
|软件|解析权重、102×4组Jacobian、原门不变、事务与随机序列|PASS 17/17|
|M1|48/48数值检查；至少16/24及各档4/8；恢复|RUNNING|
|M2|单一B、6000普通更新、60联合世界事件/120梯度|DEPENDENCY_M1|
|M3无引导|完整16步A/B×48VAL×2|DEPENDENCY_M2|
|M3DEV|新增B180候选/36执行；复用旧540/108|DEPENDENCY_M2|
|收尾|只读原始审计、新独立review、一次完成resume、中文报告|PENDING|
|扩库/TEST|禁止|NOT_RUN|

新包ff_prior_endpoint_jacobian，run endpoint_jacobian_bounded_001。完整协议见该包docs/EXPERIMENT_PLAN.md；旧smooth结果22/24、46/48保留，未作原地通过升级。不因效果差追加实验。物理门、候选预算、旧模型/数据/结果与用户修改保持。

启动回执ff_prior_endpoint_jacobian/evidence/launch_receipt.json；实际状态见run status.json。后台已启动，未预判M1或下游结果。
