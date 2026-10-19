# 256槽有界TRAIN扩库：完成分析

2026-10-19；独立运行qualified_expand256_001，源码3b51595c。实际耗时1814.891秒，约30.25分钟。256槽、256次实际尝试，256/256合格（100%），无重试、失败或未知尝试；64固定TRAIN目标各4条，有障碍/无障碍各32目标。原始审计PASS。

32个预选目标的真实原生67力矩执行及独立重放全部通过，核对336000个执行和重放步。新增224条MODEL_VALID＋32条EXECUTION_VERIFIED，抽查不会升级剩余样本。合并TRAIN13966→14222，VAL48保持；合并TRAIN分级13951 MODEL_VALID、271 EXECUTION_VERIFIED。每槽首次成功，全部256槽/32执行单元完整，达到既定>=128合格、>=32目标、>=26/32执行、>=50%实际尝试合格率及审计要求。

目标与初态来自既有TRAIN，64个目标/初态均不同于上一批32，增加的是同目标轨迹密度及批间覆盖。全库初态仍1578种、精确任务锚13495，不声称新增64个全库独特目标。384个同目标轨迹对RMS中位0.02467rad、最小0.01264rad、最大0.07448rad；拒绝相同reference hash，四条不声称四种模式，不等于strict_multimodal合同。

仅不穿透及原物理容差、5mm/1°、全机几何2mm步长、完整初态、17→67、自由漂浮、真实力矩和独立重放都保持。旧模型和旧结果保留。此扩库worker零训练；用户后续授权的独立训练worker已完成14222 TRAIN、6000更新与原冻结DEV对照。四方法均180/180、36/36，但新模型位置指标有升有降，Diffusion稳定优越性未成立；不继续扩库或TEST。

一次完成resume证据已由等待器产生：1651文件及7核心hash不变、NO_WORK_ALREADY_COMPLETE。本轮没有重复该核验。软件7/7检查与物理数据结果分开。新联合独立审查C1 yes/high、C2 no/high，无限定工程交付阻断；训练细节见[完整闭环报告](../../ff_prior_expand256_train/docs/ANALYSIS.md)。

证据保存于原独立run，紧凑副本及交付manifest在ff_prior_expand256_train/evidence。数据索引artifacts/v7_1/qualified_expand256_001/training_index.json，旧TRAIN/VAL/数据不覆盖。最终TEST、pilot、study、旧1024及strict_multimodal均未启动。

```powershell
.\ff_prior_expand256\job.ps1 status
.\ff_prior_expand256\job.ps1 report
```
