# V6.4-C.1 portable evidence

[完整封存原报告](snapshot/REPORT.md) · [机器结论](snapshot/summary.json) · [可读可视化](../../visualization/continuous_route_optimizer_20261007_01/index.html)

## 本轮实际结论

|方法|完整Task与原独立门禁/4|完整且净空≥30mm/4|
|---|---:|---:|
|Z0|3|0|
|G0|3|1|
|OI|4|0|
|OC|3|3|

48共享候选槽中实际评价31个非历史连续点；16逻辑方法槽对应12个唯一actual。所有选择在任何actual开始前封存。

A输出优先采用strict+0.001工程并列规则，其中3/4来自初值。strict小幅改善不冒充已下发收益。相对两基线新增完整执行的任务：c1_mother_01_minus。

OC在以下任务预算内未达到偏好并返回NO_PLAN：c1_mother_01_minus。其他OC的净空、路径与干预取舍见原报告的同任务完整配对表；不宣称Pareto支配。

[两偏好的初值最优、全池strict、实际输出、新点收益与预算位置](../../visualization/continuous_route_optimizer_20261007_01/search_contribution.csv)。A收益单位rad/s，B收益单位m；无合格初值时收益为null，能力改变另列。

训练、神经采样和学习更新均为0；本轮收益来自非学习模型搜索。总体泛化、Diffusion优势、连续时间及硬件安全未建立，部署NOT_MET。

本导出保留所有复制原件的原始字节；完整状态/力矩/大日志留在本地原始归档，缺失项逐文件记录SHA/大小。Git克隆的字节核验不等于物理重放。

失败行的完整质量为null；已消费前缀质量单独保存在[failed_prefix_quality.json](../../visualization/continuous_route_optimizer_20261007_01/failed_prefix_quality.json)，不参与完整质量均值。原报告表中full_success=false行若显示数字，其范围仅为保存的失败前缀。

![路线](../../visualization/continuous_route_optimizer_20261007_01/routes.png)

![质量与成本](../../visualization/continuous_route_optimizer_20261007_01/quality_cost.png)

轨迹图的失败方法仅绘制到拒绝时刻；完整质量柱图排除失败前缀。原runner自动生成的PCC监视图保留在本地归档，本次发布只增加两张核心图，无视频或PDF。

命令/环境/producer、并行调度与0物理步的包装脚本导入修复见snapshot/RUN_NOTES.md和相应收据。优化算法与逐任务预算未改动。
