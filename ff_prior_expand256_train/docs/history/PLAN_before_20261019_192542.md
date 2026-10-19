# 已授权256槽扩库执行协议

2026-10-19；依据已交付next_256_coverage_proposal_v1及用户本轮明确运行授权。独立分支v7.1-lightweight-free-floating-prior；独立输出artifacts/v7_1/qualified_expand256_001，源程序ff_prior_expand256。旧M0—M4交付和所有原始实验不改写。

1. 使用已交付13966模型和原codec，late4采样器，零训练更新。冻结model.pt/codec/training元数据、导入源码闭包、父数据和任务身份后才启动。
2. 从已有13966 TRAIN元数据按稀疏初态和世界目标覆盖选择64个不同目标/家族/初态，有障碍和无障碍各32；排除之前32个目标、已知开发/验证家族及任务。未读取heldout轨迹数组或结果；读取已有frozen_tasks元数据只为排除泄漏。
3. 64×4=256槽，每槽最多2次，最多512尝试。所有512个样本ID和种子已写入target_selection.json；成功即停止该槽，失败不重标世界目标，不借DEV/VAL/TEST填槽。目标/完整状态合同ID及父来源已冻结。
4. 32个执行目标预先按hash选定，障碍/无障碍各16。每单元按冻结槽序选第一条合格轨迹，执行失败无替补。保留完整初态、17→67、自由漂浮、全机几何、原生67力矩执行及独立重放。
5. 物理门完全沿用：历史仅不穿透及数值容差、双臂世界目标5mm/1°、2mm几何步长与其他原运动/耦合/力矩/重放门。16次网络调用，后四步修正27/27/27/29查询，候选15秒/113耦合查询/15872积分上限。查询和耗时记录但无需优于基线。
6. 去重拒绝已有或本批相同reference hash；按真实轨迹RMS报告差异，四槽不声称四模式。主先验库无需同目标五条，strict子集不在本轮运行。
7. 未执行只标MODEL_VALID；真实执行并重放通过才逐条标EXECUTION_VERIFIED。已执行失败/未知的对应样本排除训练清单；其他未执行合格样本保留原等级。合并TRAIN索引，VAL48保持；不更新旧模型、不把新数据说成已训练。
8. 下一阶段门：至少128合格、至少32目标、至少26/32执行重放、实际尝试资格率≥50%、全部256槽/32单元记录完整且无未决错误，原始审计PASS。未知中断消耗保留。预算结束即停止，部分合格如实保存，不循环填满。
9. worker CPU2线程、4小时、20GiB。隐藏后台、PID/心跳、status/report/resume。完成resume核验一次，不重复工作。冻结源码不能原地修改后冒用原身份；有明确bug才另版本修复，不能为效果重跑。
10. 完成后原始审计、中文报告、数据/模型使用manifest、失败账本、一次新独立experiment-result-to-claim审查，判断下一阶段。软件、模型数据资格、真实执行、Diffusion收益分开。仅补库不能宣称Diffusion优势改善。

本轮禁止最终TEST、pilot、study、strict_multimodal、旧1024及递归生产扩库，不上传大数据、不停止旧作业、不回退或强推。原应用delivery_v1目标已完成；本轮是用户新增授权的有界扩库，终端和接续以本协议为准。

进入下一阶段只做判断及建议，不自动训练新模型或启动另一批扩库/TEST。准确冻结清单见FROZEN_TARGETS.md及evidence/target_selection.json。

```powershell
.\ff_prior_expand256\job.ps1 start
.\ff_prior_expand256\job.ps1 status
.\ff_prior_expand256\job.ps1 report
.\ff_prior_expand256\job.ps1 resume
```
