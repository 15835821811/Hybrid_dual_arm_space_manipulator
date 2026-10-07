# V6.4-B.2 任务锚定路线残差先导试验

本目标补充已完成的 B.1。旧 M0/M1 权重、试验结论和失败证据保留。新表示只改变连续体末端在允许中间区间的位置路线，不改变刚性臂运动目标反馈、两臂姿态、构型偏好或冻结的执行和安全控制层。

先验证真正非零残差能否从原初态完成 27 秒任务，再用成功 TRAIN/VAL 残差进行一次小型 Diffusion 训练。研究交付、表示能力、学习收益分别判断。若固定候选无成功非零示范，保留失败并停止训练；缺少成功 TRAIN 或 VAL 标签同样不训练，不生成空 checkpoint。

协议为 12 个新任务（TRAIN 6 / VAL 2 / TEST 4），24 个预声明非零教师候选。最多新增实际运行为零接口 1 + 教师 24 + TEST 三组 K1 12 = 37。E2 固定 16 个参考候选，K1 永远为 slot 0，K4 不做实际闭环；非法输出占原分母，不裁剪、投影、修复或零参考 fallback。

```powershell
python -B -X utf8 -m v6_4.task_anchored_residual freeze --output <output>
python -B -X utf8 -m v6_4.task_anchored_residual freeze-source --output <output> --checks <unit_checks.json>
python -B -X utf8 -m v6_4.task_anchored_residual run --plan <output>/plan.json --output <output> --device cuda
```

冻结前 output 必须包含已核验的 bootstrap_identity.json 和 frozen_execution_config.json。`--stop-after zero|teacher|training` 可在阶段边界暂停；继续已有完整阶段须显式 `--reuse-completed`，核验配置、源码、计划和证据哈希。未完成的 slot 不自动重跑，终态试验不追加预算。

20ms 是冻结的仿真规划周期，也对应 50Hz 部署时的交付目标。本研究保留 20ms / 2ms / 27s 时序和全部安全门禁，但墙钟超时不否定研究功能结果。继承的 C.1 状态采集至发布计时以及算法计时全部保留；真实计算期间状态继续演化时的命令有效性仍需单独验证，部署为 NOT_MET。不得把仿真暂停、有限计时分位数或同模型预演一致性称为真实备份安全或硬实时证明。

独立评价沿用保存力矩的新物理重放、区间重算、robot-target 原生 500Hz 查询和整机 50Hz 配置细分 4。实际 QP 消费的十项参考逐规划时刻绑定；原步后刚性位姿日志保留其 MuJoCo 缓存诊断语义。没有显式 q_ref 的新路线表示，其旧 joint-codec 和名义 q_ref 几何检查标为 N/A，不使用静态 home 偏好冒充关节轨迹。
