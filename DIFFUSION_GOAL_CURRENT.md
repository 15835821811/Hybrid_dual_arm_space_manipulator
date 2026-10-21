# 当前执行目标：完整19×17输出与资格保持prepare的有界学习闭环

保留长期目标静态避障抓捕→动态航点。上一轮rep_fidelity已2b8e47b1完成；其323身份执行成功不是Diffusion学习收益。原目标归档ff_prior_full_control/evidence/previous_goal.md。

冻结协议ff_prior_full_control/docs/EXPERIMENT_PLAN.md；唯一run artifacts/v7_1/full_control_bounded_001。

1. 先用独立TRAIN已有6条复杂合格轨迹验证完整codec实际float32训练目标重建，3条预选绕障轨迹做新原生67力矩执行与独立重放。6/6模型＋3/3执行以及表示误差门通过再训练；不使用07 DEV教师。
2. 仅一个323维MLP从固定seed初始化，14222 TRAIN/48 VAL，6000普通更新，1200批完整16步采样教师约束；VAL不拟合、不进梯度、不选模型。教师损失不冒充真实非线性世界终点监督。
3. 完成96条完整无引导采样、原DEV180新候选/36执行单元、原三/五障碍10新候选/2单元；同原候选预算、late4和物理门。准确复用旧aligned/强样条/条件均值，旧结果不记为新执行；无科学优势也继续预定评估。
4. 新独立review、中文报告、紧凑原始证据、版本tracker与任务提交；完成resume只核一次，后台接续收尾后暂停。依赖门失败则下游BLOCKED，保留负结果，不调参追加。

仅不穿透、5mm/1度、2mm几何、17→67/完整初态/自由漂浮/全机几何/运动/力矩/重放门不变。软件表示误差1e-6rad用于float32重建，不替代物理门；纯float64往返≤1e-12。
不新增数据、不混DEV入TRAIN、不扩库、不pilot、不TEST；保留旧模型结果与用户修改。2数学库线程/6h/20GiB输出/30000耦合。输出/codec/初始化/prepare/损失组合比较，不宣称纯维数因果。

当前软件7测试通过，固定输入15411文件；后台已启动；重建门控制训练及后续评估。终端 .\ff_prior_full_control\job.ps1 status|report|resume。后续建议不自动执行。

源码f788a689；worker71700/launcher92180已确认存活心跳正常；接续automation ACTIVE，完成后独立审查报告。
