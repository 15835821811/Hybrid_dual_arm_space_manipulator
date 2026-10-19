# 评审范围内的 Research Findings

实验：paired_objectives_smoke_001。判定：整体partial；C1 yes（所测数值与配对实现）；C2 no（TRAIN未过，DEV/优势未检验）。置信度high。来源和训练审计PASS；paper claim audit unavailable，下游写作provisional。

真正获得的证据：8/8参数方向诊断有效；四组同初始化、同数据/序列、6000更新；C/D共240个当前预测世界梯度。独立160终点和预测复核通过。D对warm逐例损失比中位0.914089，对B0.853648，未过0.8门；双端点位姿通过warm5/32、D4/32。

失败定位与约束：局部项保留有益但不足；world事件加权梯度主导且全触发裁剪，为尺度诊断而非已证因果。不得把平方和下降、单步去噪、TRAIN探针、历史执行结果改写为新权重DDIM/DEV/物理成功。不得用绝对中位数之比替换逐例比中位数，不放松阈值。

路由：本轮收尾，保留DEV BLOCKED，不启动新实验。后续仅建议经单独冻结的world尺度单因素对照；多种子、未使用任务、DDIM、执行/重放和强基线优势仍缺。

按任务限制，仅在评审目录记录；根findings/tracker/目标文档的整合由主执行者处理。本地不存在PAPER_PLAN.md可更新。
