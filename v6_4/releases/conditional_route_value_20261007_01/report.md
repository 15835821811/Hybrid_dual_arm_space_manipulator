# Portable evidence view — V6.4-B.3

The original sealed [REPORT.md](snapshot/REPORT.md) and all copied files retain
their original bytes. The report below changes only local link destinations.
Absolute paths in the original JSON are provenance strings. Available files
resolve through [portable_paths.json](portable_paths.json); omitted or unbound
local references are explicitly unavailable from a Git clone.

This is a curated release, not a complete physics/replay archive. The complete
source inventory, omitted file SHA256/size ledger, exact frozen source copies,
and source/producer relationships are preserved in [release_manifest.json](release_manifest.json).
Verification checks copied bytes and ledger fidelity; it does not re-execute
physics or certify the unavailable replay files.

```sh
python -m v6_4.export_conditional_release --verify --output v6_4/releases/conditional_route_value_20261007_01
```

Export adds zero physics steps, optimizer updates, and samples. It does not
change any research conclusion or deployment status.

---

# V6.4-B.3 固定配对路线价值研究

终态：`ROUTE_VALUE_NOT_IDENTIFIABLE_WITHIN_CURRENT_REPRESENTATION`。按预声明停止规则结束P1，P2/P3/P4均未运行。有限负结果不否定Diffusion。

实际执行producer：`f5f1687f5ba0b119c91bbc385dfdc841b56d091e`；从已发布B.2 `28ef7889be16b4a504d9449f52c50e3a99e82a31` 派生。旧B.2算法producer为 `7d0a3fd4bd17f41b99b388c30c5ac4897ac4d31d`，旧结果与失败保持。

| 槽 | 任务 | 固定候选 | 状态 | 保存时长/s | 完整合格 | I_route/(rad/s) | 相关窗口净空/mm |
|---|---|---|---|---:|---|---:|---:|
| PILOT_00 | b3_mother_00_c_plus | z0 | TASK_COMPLETED | 27.000 | True | 0.108601632 | 25.8842929 |
| PILOT_01 | b3_mother_00_c_plus | z+ | TASK_COMPLETED | 27.000 | True | 0.111963142 | 25.7778153 |
| PILOT_02 | b3_mother_00_c_plus | z- | TASK_COMPLETED | 27.000 | True | 0.105570475 | 25.8612794 |
| PILOT_03 | b3_mother_00_c_minus | z0 | TASK_COMPLETED | 27.000 | True | 0.112069755 | 25.8777431 |
| PILOT_04 | b3_mother_00_c_minus | z+ | TASK_COMPLETED | 27.000 | True | 0.10779655 | 25.9001972 |
| PILOT_05 | b3_mother_00_c_minus | z- | TASK_COMPLETED | 27.000 | True | 0.114338061 | 25.7664573 |

破折号表示没有可用于完整质量比较的值，失败前缀单独保存在paired_metrics.json，未填入完整任务均值。

P0固定旧update250权重：32 DDIM，16组同噪声对照有16组观察到条件响应，raw幅值合法27/32；未裁剪或替换。公开旧TEST仅用于诊断，无新增物理或训练。

## 五个问题的直接回答

1. **零残差已经足够时，残差是否有可测收益？** b3_mother_00_c_plus较好非零z-相对零残差I_route变化-2.791%；b3_mother_00_c_minus较好非零z+相对零残差I_route变化-3.813%。完整成功记录中未达到冻结的双侧A/B路线价值门槛。安全通过和非零运动变化不能代替可辨识的路线收益。

2. **障碍换边后，正确路线是否随之改变？** b3_mother_00_c_plus: z-；b3_mother_00_c_minus: z+。完整合格非零候选的数值排名随障碍换边反转，但未建立满足冻结门槛的双侧相反有利方向；数值排名与达到可辨识价值门槛分别报告。

3. **真实条件是否比错条件/无条件经验抽样更好？** 未评价新模型的五组TEST。旧权重P0仅观察到条件响应，缺少可信偏好标签，不能据此称正确适应。

4. **与简单检索相比是否值得增加Diffusion？** 本轮没有建立新增Diffusion的收益证据；维持B.2的非学习检索基线，不能将未运行对照写成检索胜出。

5. **负结果下一步指向任务区分度、数据还是方法？** 当前已建立非学习可执行性见证；瓶颈是路线质量差额不足。优先评估任务区分度以及20mm残差经过安全执行层后的有效作用，再考虑高质量数据与学习方法。本轮停止，不追加障碍搜索、幅值、种子或网络。

## 成本与边界

```json
{
  "actual_physics_steps": 81000,
  "private_preview_physics_steps": 81000,
  "independent_saved_torque_replay_steps": 81000,
  "native_geometry_query_calls": 201187130,
  "preview_calls": 8100,
  "additional_route_quality_geometry_queries": 5022372,
  "input_precheck_geometry_queries": 41846,
  "preflight_geometry_binding_check_queries": 186,
  "native_geometry_query_calls_by_phase": {
    "actual": 98469703,
    "independent_torque_replay": 102717427,
    "private_preview": 0
  },
  "actual_slots_consumed": 6,
  "actual_runner_entries": 6,
  "slots_with_physics": 6,
  "old_checkpoint_ddim_calls": 32,
  "teacher_actual_slots": 0,
  "new_model_ddim_calls": 0,
  "new_training_runs": 0,
  "optimizer_updates": 0,
  "new_TEST_actual_slots": 0,
  "count_scope": "actual, private previews and same-torque independent replays are distinct; geometry queries are not independent experiments"
}
```

## 固定窗口中点的保存轨迹诊断

以下是距预声明窗口中点最近的原生保存样本，描述实际端点相对基础参考的第一横向偏置；不用于选择或修改pilot门槛，也不构成独立因果证明。

- PILOT_00 / z0，t=10.220s：-22.091mm。
- PILOT_01 / z+，t=10.220s：-15.899mm。
- PILOT_02 / z-，t=10.220s：-23.437mm。
- PILOT_03 / z0，t=10.220s：+22.096mm。
- PILOT_04 / z+，t=10.220s：+23.434mm。
- PILOT_05 / z-，t=10.220s：+15.911mm。

I_route由原日志中17维名义速度与选中速度差的范数恢复；名义值经过原速度边界clipping，两个源向量未保存。固定T_route取完整预声明区间的闭区间50Hz样本。完整Task与原五项独立安全门禁通过后才作完整质量比较；相关连续体-球净空不被整机最小值替代。

20ms规划、2ms物理、27s任务及原QP/67路力矩/私有预演/安全标准保持。20ms墙钟不作研究门禁，部署NOT_MET；无硬实时、连续时间或模型失配保证。正式TRAIN/VAL/TEST已冻结但未执行，不能将其计作独立测试成功。
