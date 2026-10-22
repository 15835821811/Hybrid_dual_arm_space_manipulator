# static_fullchain_001 部署失败与独立修复交接

本轮没有完成静态规划验收，不能声称产生 MODEL_VALID 或 EXECUTION_VERIFIED 路径，也不能判断 Diffusion 优劣。独立修复 `static_fullchain_002` 已实际后台启动；其新协议位于 `ff_prior_static_fullchain_v2/docs/EXPERIMENT_PLAN.md`，源码提交 `8d3d562a`，结果尚待完成。

## 实际发生的工作

首轮源码提交 `25b98dd6`，冻结原三/五障碍、3种子、两方法各5提案，共60提案/12执行单元。11项软件检查通过后启动worker。worker在 **34.844秒** 后停止，第一结构方法单元尚未完成；失败位于首轮323修正保存 `trial_01_pairs.json` 的原子临时文件。

实际保存了第一单元5个结构提案的生成与初筛，进入候选02修正；尚无完整单元结果。失败路径共 **260字符**，目标JSON为251字符；同目录的trial NPZ为244字符，其临时名为253字符且已成功保存。父目录存在，Windows `LongPathsEnabled=0`。这些证据高度支持Windows路径长度导致的文件系统异常；未额外重现OS故障，不将该因果写成绝对证明。

原运行保留 **58个文件、2,616,481字节**，清单及SHA256在 `evidence/failed_run_manifest.json`。没有完整 validation、execution selection/report、unit result/calls 或 budget.json。ledger保留inflight，**精确耦合/积分成本未知**；没有将缺失记作零，也没有试图恢复或重放此单元。

## 证据等级与独立审查

|项目|结论|
|---|---|
|软件|11项检查通过，但漏掉了实际最长输出路径验证；部署不完整|
|模型资格|未完成完整认证；不能报合格率0/60或通过率|
|真实力矩执行和重放|本轮没有进入执行阶段；12冻结单元的结果未完成|
|C1静态工程可行|本轮不支持；不能从I/O中断推断任务不可行|
|C2 Diffusion相对优势|本轮不支持；没有完成方法间对照|

新独立 `experiment-result-to-claim` 审查：`paper/review-traces/experiment-result-to-claim/20261022_static_fs_failure_run01`。审查确认172个源码文件的运行归档/当前字节与identity匹配，7个绑定输入匹配；C1/C2均为no，confidence=high。科学主张无可用性能证据，负判定是“本轮未支持”，不是“方法已被证伪”。论文主张审计文件不可用，论文写作结论按技能标为provisional；这不妨碍部署修复。v2不在该失败审查的认证范围。

源码身份应写为 **提交＋精确执行归档**。工作区原有 `v6_lite/hierarchical_qp.py` 用户修改已按实际字节冻结并归档，未擅自提交或回退。因此不能将运行称为完全干净的25b98dd6 checkout。其余历史结果不重跑、不重新审查。

本轮不适用“完成后resume幂等性”测试：状态FAILED且存在未知inflight，已按原规则禁止resume。`evidence/resume_not_performed.json`记录0次调用，原文件hash保持。

## 独立修复与预算

v2改为短目录u00—u11，完整场景/种子/方法仍在selection中。没有更换原目标、改变采样/修正算法、放宽门或按性能筛选新任务。新预检增加最长实际原子临时路径写读，最大 **221字符**；12项软件检查通过，仅证明软件检查。

未知工作按旧单元完整预留保守计入：13000耦合、32000000积分、40秒、128MiB；这些是预算扣减，**不是测得的调用量**。新run上限147000耦合、352000000积分、21560秒、20GiB减128MiB，仍在原P1合计预算内。两方法均等扣减每单元上限，旧结果不作新结果复用。新run新身份，不在旧未知运行上resume。

当前只保留一个重worker和原唯一15分钟接续。终端：

```powershell
.\ff_prior_fullchain\job.ps1 status
.\ff_prior_fullchain\job.ps1 next
.\ff_prior_fullchain\job.ps1 report
```

v2结束后才分析60提案、认证、执行与重放以及配对误差，做新的独立结果审查。此失败审查不能替代v2审查。P1工程阶段后继续已授权P2动态表示与连续交接，不等待网络相对优势；不扩库、不放宽物理门、不启动TEST。
