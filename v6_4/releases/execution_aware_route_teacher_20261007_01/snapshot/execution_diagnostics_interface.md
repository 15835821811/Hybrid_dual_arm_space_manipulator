# B3.1 T1 执行版本绑定与最小旁路日志

生产入口 `execute_residual_attempt(..., execution_diagnostics=True, diagnostic_obstacle_name=task.scenario['workspace_obstacles'][1]['name'])`。日志默认关闭，旧 v1 默认数值路径不变；新 v2 必须打开，缺失消费身份会拒绝验收。球名接受 Task 中的短 name 或编译完整 geom 名。

`run_synchronous_scenario` 透传开关至原同一次 `HierarchicalVelocityQP.solve`。原未约束解仅赋名并拷贝，仍使用原 velocity box clip；未增加 solver、约束、几何查询、Jacobian 查询、physics 或动作分支。连续体实际速度取原已计算任务点 Jacobian 乘当前物理 qvel。开关打开后的日志开销包含在现有调度墙钟时间；不改 research_simulation 门禁。

所有 `task_*` 字段采用原 `task_time`（20 ms / 50 Hz）。失败前缀仅消费 `[:len(torque)//10]`，禁止给未消费尾行质量信用。

- `task_qp_raw_unconstrained_velocity`、`task_qp_box_nominal_velocity`、`task_qp_selected_velocity`：`(N,17)`；各另有 `_continuum` `(N,10)`、`_rigid` `(N,7)`。原 I_route 仍是 `RMS(norm(selected-box_nominal))`，不得改用 raw 或 proposal 差。
- `task_qp_continuum_target_position_m`、`actual_position_m`、`position_error_m`、`reference_velocity_m_s`、`desired_velocity_m_s`、`actual_velocity_m_s`、`selected_task_velocity_m_s`：均 `(N,3)`（后三项分别是原裁剪任务命令、当前物理任务点速度、QP reaction-map 命令速度）。最后一项不得命名为 actual。
- `task_qp_obstacle_rows_json`：`(N,)` Unicode JSON，不用 pickle。每条来源是原 `mujoco:<pair_class>:<geom_a>:<geom_b>`；`active=true` 表示已有行进入原 QP。保存原 candidate residual / lookahead residual；selected residual 由同一矩阵乘已选速度得到，绑定阈值沿用原 `2e-5 m/s`。未进入行仅记录 `active=false` 与来源，不新增距离查询或虚构残差。可用相关行 active/binding 的逐 tick union 乘 `.020 s` 统计时长，不累加每条行导致重复计时。
- `task_qp_obstacle_name`、`task_qp_obstacle_active_row_count`：`(N,)`。
- `task_consumed_reference_version`、`task_consumed_reference_definition_sha256`、`task_consumed_reference_plan_sha256`：`(N,)`；`task_consumed_reference_z_m`：`(N,6,2)`。来源是 runner 实际 sample 的同一 provider/plan，并在独立 reference binding 中与预声明 plan 精确核对。版本、定义、plan 或 z 不符均硬失败。

旧 v1 没有新增字段时，身份日志注明 unavailable，但仍按旧逐输入 reference binding 验证；不得从旧标量反推 10D/7D 或球行贡献。新增失败日志使用 JSON null 记录无选中动作，不将候选冒充执行。

`ExecutionCostLedger` 新增按 phase 的 `qp_solve` started/returned/raised 与 `qp_solve_calls`，透明转发原参数、返回和异常。原实际、私有 preview、saved-torque replay 与 native geometry 账目定义不变。原 unconstrained nominal 计算已经存在，不是新增反事实 QP。

必要检查：9 个独立单元测试通过（其中 1 项针对短球名修正复核一次）。真实 optimized ADMM + 原 action validator 在同一固定矩阵输入下，开关前后选中速度、candidate、clearance/lookahead、状态、迭代、objective、原干预范数均逐元素/值完全相同。该矩阵 fixture 明示为数值检查，不冒充真实场景或闭环验证；物理与 geometry 函数被禁止。另用真实 v1/v2 provider 读取原 Task，验证版本/plan/z 消费绑定，不创建 MjData。

本子任务检查总成本：4 次原 constrained QP 调用、4 次原已有 unconstrained nominal solve；其中开关数值比较 2 次、日志 roundtrip 检查 2 次（含一次修正复核）。新 actual 槽、physics、private preview、torque replay、native geometry、额外 Jacobian 查询、DDIM、训练、optimizer 均 0。已有成本 wrapper 测试调用的是 Python sentinel，不计入真实物理账。未执行正式 rollout、未提交；待 root 冻结后运行。
