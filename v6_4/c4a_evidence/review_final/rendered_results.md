# C4-A 两个新 DEV Task 的受控验证结果

## 范围与证据状态

2026-10-10；状态 COMPLETED / 实际物理执行已完成。固定两个独立母场景的 DEV：`c4a_dev0_plus` (seed 2028300010) 与 `c4a_dev1_minus` (seed 2028404739)。P0=Rule8，P1=Diffusion8-Replacement，P2=Diffusion8-Rule-Preserving Portfolio。六个搜索流全部封存后才执行最终 Actual；原27秒任务和五门禁未改变。

完整机器表：`v6_4/c4a_evidence/results_01/{summary.json,dev_candidates.csv,dev_streams.csv,dev_endpoints.csv}`。原始日志与轨迹：`E:/v64c4a/v6_4/c4a_evidence/dev_01/`。紧凑发布包：`v6_4/releases/c4a_architecture_audit_20261010_01/`，保留全部原始文件 SHA 和未复制的大轨迹文件路径。

只有两个母场景；这些是逐任务先导结果，不报告显著性或总体泛化。相同 Actual 的严格 alias 不增加独立样本。

## 先评价任务能力与安全

|策略|完整 Actual + 五门禁|A|B|B端点实际B30|NO_PLAN|NOT_RUN|TOOL_ERROR|
|---|---|---|---|---|---|---|---|
|P0|4/4|2/2|2/2|2/2|0|0|0|
|P1|4/4|2/2|2/2|2/2|0|0|0|
|P2|4/4|2/2|2/2|2/2|0|0|0|

分母包括失败和 NO_PLAN。下表是逐逻辑端点结果，五门禁保留原字段；是否完整达成任务另列，B30不是五门禁的替代品。

|Task|策略/偏好|状态|完整+五门禁|五门禁|独立执行|alias|
|---|---|---|---|---|---|---|
|c4a_dev0_plus|P0/A|FULL_TASK_AND_FIVE_GATES_PASSED|True|{"task_requirements": true, "execution_contract": true, "independent_interval": true, "native_geometry": true, "reference_binding": true}|True|—|
|c4a_dev0_plus|P0/B|FULL_TASK_AND_FIVE_GATES_PASSED|True|{"task_requirements": true, "execution_contract": true, "independent_interval": true, "native_geometry": true, "reference_binding": true}|True|—|
|c4a_dev0_plus|P1/A|FULL_TASK_AND_FIVE_GATES_PASSED|True|{"task_requirements": true, "execution_contract": true, "independent_interval": true, "native_geometry": true, "reference_binding": true}|False|E:\v64c4a\v6_4\c4a_evidence\dev_01\validation\actual\c4a_dev0_plus\P0_A\slot.json|
|c4a_dev0_plus|P1/B|FULL_TASK_AND_FIVE_GATES_PASSED|True|{"task_requirements": true, "execution_contract": true, "independent_interval": true, "native_geometry": true, "reference_binding": true}|True|—|
|c4a_dev0_plus|P2/A|FULL_TASK_AND_FIVE_GATES_PASSED|True|{"task_requirements": true, "execution_contract": true, "independent_interval": true, "native_geometry": true, "reference_binding": true}|False|E:\v64c4a\v6_4\c4a_evidence\dev_01\validation\actual\c4a_dev0_plus\P0_A\slot.json|
|c4a_dev0_plus|P2/B|FULL_TASK_AND_FIVE_GATES_PASSED|True|{"task_requirements": true, "execution_contract": true, "independent_interval": true, "native_geometry": true, "reference_binding": true}|False|E:\v64c4a\v6_4\c4a_evidence\dev_01\validation\actual\c4a_dev0_plus\P0_B\slot.json|
|c4a_dev1_minus|P0/A|FULL_TASK_AND_FIVE_GATES_PASSED|True|{"task_requirements": true, "execution_contract": true, "independent_interval": true, "native_geometry": true, "reference_binding": true}|False|E:\v64c4a\v6_4\c4a_evidence\dev_01\validation\actual\c4a_dev1_minus\P2_A\slot.json|
|c4a_dev1_minus|P0/B|FULL_TASK_AND_FIVE_GATES_PASSED|True|{"task_requirements": true, "execution_contract": true, "independent_interval": true, "native_geometry": true, "reference_binding": true}|True|—|
|c4a_dev1_minus|P1/A|FULL_TASK_AND_FIVE_GATES_PASSED|True|{"task_requirements": true, "execution_contract": true, "independent_interval": true, "native_geometry": true, "reference_binding": true}|True|—|
|c4a_dev1_minus|P1/B|FULL_TASK_AND_FIVE_GATES_PASSED|True|{"task_requirements": true, "execution_contract": true, "independent_interval": true, "native_geometry": true, "reference_binding": true}|True|—|
|c4a_dev1_minus|P2/A|FULL_TASK_AND_FIVE_GATES_PASSED|True|{"task_requirements": true, "execution_contract": true, "independent_interval": true, "native_geometry": true, "reference_binding": true}|True|—|
|c4a_dev1_minus|P2/B|FULL_TASK_AND_FIVE_GATES_PASSED|True|{"task_requirements": true, "execution_contract": true, "independent_interval": true, "native_geometry": true, "reference_binding": true}|False|E:\v64c4a\v6_4\c4a_evidence\dev_01\validation\actual\c4a_dev1_minus\P1_B\slot.json|

## 再评价轨迹质量

新DEV没有R12参考：**near_R12=N/A_NOT_RUN_BUDGET**。没有借用其他Task的R12，也没有追加物理预算。辅助 near_P0 在结果产生前声明：A要求I≤P0+0.001 rad/s且L≤P0+0.005m；B要求L≤P0+0.005m且d≥0.030m。near_P0不冒充near_R12。

|策略|near_P0 A|near_P0 B|near_R12|
|---|---|---|---|
|P0|2/2|2/2|N/A_NOT_RUN_BUDGET|
|P1|1/2|1/2|N/A_NOT_RUN_BUDGET|
|P2|2/2|2/2|N/A_NOT_RUN_BUDGET|

|Task|策略/偏好|I_support rad/s|L_full m|d_support mm|基座平移峰 mm|基座旋转峰 rad|near_P0|
|---|---|---|---|---|---|---|---|
|c4a_dev0_plus|P0/A|0.112979168|1.145688839|29.197464|7.884227|0.057031729|True|
|c4a_dev0_plus|P0/B|0.113438229|1.147820556|40.956225|7.883816|0.057031772|True|
|c4a_dev0_plus|P1/A|0.112979168|1.145688839|29.197464|7.884227|0.057031729|True|
|c4a_dev0_plus|P1/B|0.113026759|1.153316351|41.206673|7.882963|0.057024204|False|
|c4a_dev0_plus|P2/A|0.112979168|1.145688839|29.197464|7.884227|0.057031729|True|
|c4a_dev0_plus|P2/B|0.113438229|1.147820556|40.956225|7.883816|0.057031772|True|
|c4a_dev1_minus|P0/A|0.113784669|1.087105444|26.294661|7.211130|0.056153213|True|
|c4a_dev1_minus|P0/B|0.113710993|1.103856117|34.174421|7.210480|0.056154297|True|
|c4a_dev1_minus|P1/A|0.112329927|1.094409899|26.363787|7.210186|0.056146396|False|
|c4a_dev1_minus|P1/B|0.113110690|1.100746135|31.853370|7.210278|0.056150871|True|
|c4a_dev1_minus|P2/A|0.113784669|1.087105444|26.294661|7.211130|0.056153213|True|
|c4a_dev1_minus|P2/B|0.113110690|1.100746135|31.853370|7.210278|0.056150871|True|

## 候选覆盖、拒绝和来源

|Task|策略|合法/生成raw|完整名义合格/槽|B30候选|首次完整槽|首次B30槽|缓存|
|---|---|---|---|---|---|---|---|
|c4a_dev0_plus|P0|N/A(无模型)|8/8|5|1|2|0|
|c4a_dev0_plus|P1|2/2|8/8|4|1|3|0|
|c4a_dev0_plus|P2|2/2|8/8|5|1|2|0|
|c4a_dev1_minus|P0|N/A(无模型)|8/8|3|1|4|0|
|c4a_dev1_minus|P1|1/2|7/8|3|1|4|0|
|c4a_dev1_minus|P2|1/2|7/8|3|1|4|0|

首次命中位置使用1基消耗槽号；未命中应保留RIGHT_CENSORED，不用0或删除任务。生成提案与物理预测分开计数。下表列出全部Diffusion raw，包括原样拒绝。P1/P2为相同噪声配对重复，不能当成8个独立噪声。

|Task|策略/槽|族|raw合法|拒绝原因|名义完整|B30|d_support mm|
|---|---|---|---|---|---|---|---|
|c4a_dev0_plus|P1/2|v1|True|—|True|False|28.226747|
|c4a_dev0_plus|P1/4|v2|True|—|True|True|41.769023|
|c4a_dev0_plus|P2/5|v1|True|—|True|False|28.226747|
|c4a_dev0_plus|P2/6|v2|True|—|True|True|41.769023|
|c4a_dev1_minus|P1/2|v1|False|raw seed exceeds the original 20mm interval disk|False|False|N/A|
|c4a_dev1_minus|P1/4|v2|True|—|True|True|31.853370|
|c4a_dev1_minus|P2/5|v1|False|raw seed exceeds the original 20mm interval disk|False|False|N/A|
|c4a_dev1_minus|P2/6|v2|True|—|True|True|31.853370|

|Task|策略/偏好|最终候选|来源分类|原始来源|完整谱系|
|---|---|---|---|---|---|
|c4a_dev0_plus|P0/A|C00|rule_seed_or_descendant|zero|[{"initial_position": 0, "source": "zero"}]|
|c4a_dev0_plus|P0/B|C01|rule_seed_or_descendant|geometry|[{"initial_position": 1, "source": "geometry"}]|
|c4a_dev0_plus|P1/A|C00|rule_seed_or_descendant|zero|[{"initial_position": 0, "source": "zero"}]|
|c4a_dev0_plus|P1/B|C02|rule_seed_or_descendant|geometry|[{"initial_position": 2, "source": "geometry"}]|
|c4a_dev0_plus|P2/A|C00|rule_seed_or_descendant|zero|[{"initial_position": 0, "source": "zero"}]|
|c4a_dev0_plus|P2/B|C01|rule_seed_or_descendant|geometry|[{"initial_position": 1, "source": "geometry"}]|
|c4a_dev1_minus|P0/A|C01|rule_seed_or_descendant|geometry|[{"initial_position": 1, "source": "geometry"}]|
|c4a_dev1_minus|P0/B|C03|rule_seed_or_descendant|geometry|[{"initial_position": 3, "source": "geometry"}]|
|c4a_dev1_minus|P1/A|C04|rule_seed_or_descendant|geometry|[{"initial_position": 2, "source": "geometry"}, {"parent_candidate_id": "C02", "source": "adaptive_poll", "coordinate": 0, "sign": 1}]|
|c4a_dev1_minus|P1/B|C03|direct_learning_seed|diffusion|[{"initial_position": 3, "source": "diffusion"}]|
|c4a_dev1_minus|P2/A|C01|rule_seed_or_descendant|geometry|[{"initial_position": 1, "source": "geometry"}]|
|c4a_dev1_minus|P2/B|C05|direct_learning_seed|diffusion|[{"initial_position": 5, "source": "diffusion"}]|

## 最后评价规划成本

每行成本只计一次共享A/B搜索。cold service含初值准备及搜索服务，outer process还含进程启动/导入/验证；不得混用口径。计时为此CPU单线程单次串行运行，微小差异不构成加速证据。

|Task|策略|槽|真实预测|cold service s|cold outer s|主预测积分|private preview|几何查询|QP|独立重放|
|---|---|---|---|---|---|---|---|---|---|---|
|c4a_dev0_plus|P0|8|8|656.399|657.993|108000|108000|132672567|10800|0|
|c4a_dev0_plus|P1|8|8|656.397|658.055|108000|108000|132672758|10800|0|
|c4a_dev0_plus|P2|8|8|649.132|650.790|108000|108000|132672788|10800|0|
|c4a_dev1_minus|P0|8|8|663.722|665.364|108000|108000|132823651|10800|0|
|c4a_dev1_minus|P1|8|7|568.767|570.352|94500|94500|116220797|9450|0|
|c4a_dev1_minus|P2|8|7|576.892|578.495|94500|94500|116220727|9450|0|

|策略|总cold service s|总cold outer s|主预测积分|private preview|几何查询|
|---|---|---|---|---|---|
|P0|1320.121|1323.357|216000|216000|265496218|
|P1|1225.164|1228.407|202500|202500|248893555|
|P2|1226.024|1229.285|202500|202500|248893515|

模型加载、条件编码和两次DDIM生成的独立计时保留在 `dev_streams.csv` 的 `initializer_setup` 与原 `planning_cost.json`。raw资格检验、提案构造和缓存查找包含在optimizer/cold墙钟中，但没有独立计时字段，记为INCLUDED_NOT_SEPARATELY_TIMED，不能填成零。每候选物理成本保留在 `dev_candidates.csv`。非法raw少做一次物理预测造成的时长下降属于工作量减少，不是已证明的有效搜索加速。

## 预算实账与独立证据

```json
{
  "limits": {
    "candidate_slots": 48,
    "actual_logical_slots": 12,
    "ddim_samples": 8
  },
  "reservations": {
    "candidate_slots": 48,
    "actual_logical_slots": 12,
    "ddim_samples": 8
  },
  "consumed_candidate_slots": 48,
  "prediction_calls": 46,
  "logical_actual_present": 12,
  "unique_actual_runs": 7,
  "actual_aliases": 5,
  "no_plan": 0,
  "main_actual_steps": 94500,
  "new_training_updates": 0,
  "new_C3_TEST_runs": 0,
  "static_geometry_queries": 5978
}
```

全部Actual计费合计（alias没有新物理证据）：

```json
{
  "actual_physics_steps": 94500,
  "private_preview_physics_steps": 94500,
  "native_geometry_query_calls": 235992780,
  "independent_saved_torque_replay_steps": 94500,
  "qp_solve_calls": 9450
}
```

离线账另列：阶段A为128个诊断输出+16次复现=144次冻结D采样和16次S forward；数值单测另有2次冻结D确定性采样。DEV为8次DDIM（4组唯一Task/条件噪声在P1/P2重复），每次20网络forward。所有阶段训练更新为0；没有新C.3 TEST物理运行。

## 验证和复现边界

137个不同测试节点全部通过；计入复用测试共176次通过执行。新增核心数值15项、Portfolio9项、DEV合同4项。最早一次旧测试收集因namespace相对导入失败，没有执行测试；改为 `--import-mode=importlib` 后72项通过。原失败回执与修正命令完整保留，详见 `review_01/test_coverage.json` 和 `docs/audit_receipts/c4a/`。

最终 `final_integrity.json` 验证原C.3冻结字节、控制/执行配置、DEV输入与源码身份、候选封印、独立初态、P0/P2四规则一致、P1/P2 raw一致及全部Actual封印。该核验没有再次运行物理或模型。P0/P1/R12默认调度和输出的回归为固定mock环境下对旧实现的字节比较；真实DEV性能仅来自本次实际日志。

物理producer为 `dd834783847bfbcc4f8ec624b7bce0d22777b6de`；最终交付commit仅追加报告/证据/报告脚本，不改变已封存物理源码。确切命令、UTC时间、退出码、stdout/stderr及SHA见回执。新视频未生成。解释和下一阶段门禁见 `V6_4_C4A_DECISION.md`。
