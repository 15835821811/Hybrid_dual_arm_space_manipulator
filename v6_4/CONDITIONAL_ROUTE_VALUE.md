# V6.4-B.3：场景依赖的路线价值

B.3从已发布B.2提交`28ef7889be16b4a504d9449f52c50e3a99e82a31`派生，分支为`v6.4-b3-conditional-route-value`。B.2原算法producer为`7d0a3fd4bd17f41b99b388c30c5ac4897ac4d31d`，原封存报告SHA256为`2664eec9104eff627dfccce3b04201db64134634d5fbaed5f595a713111539f8`。旧结果、权重和失败保持独立。

本轮只检验换边障碍是否带来有价值的路线选择。机器人、17维QP、67路执行、Cartesian provider、PCC、安全距离、拒绝守卫和20mm残差上限均沿用B.2；墙钟20ms不作研究门禁，部署仍为NOT_MET。

## 冻结输入与先行试验

使用7个新seed母场景，pilot1对、TRAIN3对、VAL1对、TEST2对，每对只改变原25mm工作空间球的位置。固定第三个允许残差区间；球心位于该区间基础参考中点的第一世界横向基两侧，偏移43/44/45mm，由母场景序号确定。各侧分别编译并保存scene model hash；不镜像重用实际轨迹或标签。

pilot共六个槽位：每侧z0、第一横向基+12mm与-12mm。固定完整区间的闭区间50Hz采样窗口，主要指标为原QP日志`task_avoidance_intervention`的RMS。该源字段保存17维名义速度与实际选中速度差的范数；名义值先按原速度边界裁剪，原始两个向量未保存。本轮复用这一已有诊断，不增加控制日志或改变控制行为。

完整Task及原独立安全门禁通过后，才比较完整质量。各侧较优成功非零方向须满足预声明A/B判据，两侧较优方向相反才继续；失败前缀单独保留。pilot无法区分路线价值就停止后续teacher、训练和新TEST，不扩大幅值或追加位置搜索。

旧B.2 update250权重另做32次固定噪声、仅连续体障碍中心条件交换的诊断，不执行物理、不重选权重。原公开TEST只称诊断数据。任何条件响应都不直接称为正确适应。

## 条件成立后的固定上限

正式TRAIN/VAL为6/2任务，每任务z0、z+、z-、z_perp四个teacher，共32槽位。以任务内完整成功参考的I_route精英集合构建标签，保留合法零残差，不平均不同模式。仅一次同规模两层128 MLP训练，最多4000更新、batch32，使用TRAIN拟合scaler并按任务均衡抽样。每250更新在固定VAL噪声下依次按合法率、最近精英解码路线距离、去噪损失、较早更新选权重。

四个新TEST比较Z0/R0/U0/D_true/D_swap，共20 actual槽位。正确与错配条件共32次DDIM；K1固定slot0，K4不执行actual，不重试、替换候选或fallback。总新actual槽位最多58；actual、私有预演、保存力矩重放和几何查询分别计费。

独占本地输出为`v6_4/output/conditional_route_value_20261007_01/`。唯一`plan.json`在所有actual和DDIM前冻结，`source_identity.json`在本地producer提交与必要检查后冻结。入口为`python -m v6_4.conditional_route_value`；具体阶段必须遵守已保存的pilot决策，不能仅凭入口运行成功宣称研究完成。最终仅生成两张固定配对路线图和方法表，独立审阅后发布此分支。
