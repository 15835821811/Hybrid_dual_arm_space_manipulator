# 扩库诊断工具

这组工具读取真实保存文件，不修改生成器、训练协议或旧实验目录，也不运行新的物理仿真。

- `analyze.ps1 -Output ff_prior_pilot_tools/evidence/snapshot_003`：检查当前已提交数组、训练视图、完整初态、冻结目标及准入证明，统计覆盖和拒绝原因。运行时是非原子快照，不能当成整轮验收。snapshot_001/002已保存，后续使用新的编号保留旧结果。
- `.\.venv\Scripts\python.exe -m ff_prior_pilot_tools.noise_diagnostic --stage ten_k --output ff_prior_pilot_tools/evidence/ten_k_noise_reproduction_001`：待10k正式训练完成后，对相同VAL噪声诊断比较网络、零v预测和条件均值。只有3次批量网络前向，无训练更新或物理调用。存在的输出会在推理前拒绝覆盖。

诊断报告和JSON保存在指定目录。`one_k_noise_reproduction_001`已经复现1000条模型：t500和t999网络轨迹重建略好于零v/均值，t0比零v略差。该诊断不代表完整采样后的世界目标成功率或物理优势，不改变已冻结训练或扩库门槛。

`origin_*`表示没有额外添加的障碍，并不移除默认目标卫星等全机环境几何。保存MODEL_VALID、单样本执行验证和整个任务的最终通过必须分别解释。
