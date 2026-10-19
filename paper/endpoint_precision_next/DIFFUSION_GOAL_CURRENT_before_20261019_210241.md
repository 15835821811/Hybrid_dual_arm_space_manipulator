# 当前目标进程：本轮扩库—训练—冻结开发验证已完成并交付

2026-10-19。用户授权的256槽有界扩库、一次合并TRAIN训练与原冻结开发任务对照已实际完成；没有继续扩库或接触最终TEST。原长期研究中的稳定Diffusion优越性仍NOT_ESTABLISHED，不据此标记整个研究目标完成。

- 扩库qualified_expand256_001：256/256尝试合格、64既有TRAIN目标、32/32真实力矩执行及独立重放；分级新增224 MODEL_VALID＋32 EXECUTION_VERIFIED；原始审计PASS。TRAIN14222/VAL48，全库独特初态1578、任务锚13495保持。
- 训练expand256_train_dev_001：准确13966权重/codec温启动，14222 TRAIN全部参与，6000真实更新、384000抽样，VAL梯度0；权重重载和原始审计PASS。沿用v＋教师局部G损失，不是新非线性世界终点监督。
- 原DEV4＋已用开发8、3种子、4方法，720/720候选和144/144执行重放均通过；各方法180/180、36/36。全部冻结工程门PASS，不要求查询/耗时优势，也不追加局部验证。
- 新旧模型成功率相同；新模型候选位置中位1.0328→1.0866mm，实际执行位置均值0.7103→0.7378mm；姿态部分指标改善，不能宣称全面精度增益。保留旧13966为默认建议，新14222为工程合格候选；worker自动选新仅代表工程资格，不是优越性判定。

结束报告ff_prior_expand256_train/docs/ANALYSIS.md和ff_prior_expand256/docs/ANALYSIS.md，逐对数字/身份/数据等级见evidence/final_statistics.json、paired_endpoint_results.csv及delivery_manifest.json。新的独立联合审查已完成：C1 yes/high、C2 no/high，无工程交付阻断；本轮收尾后接续暂停，不继续实验。扩库及训练完成resume分别只核一次且文件/corehash不变。

下一阶段为本轮数据与模型版本交付及研究主张收窄，不自动追加实验。保留历史仅不穿透、5mm/1°、2mm几何及全部运动/力矩/耦合门，完整初态、17→67、自由漂浮、全机几何与独立重放。旧模型/数据/结果/用户修改/main及旧分支作业保持。禁止再扩库、旧1024、pilot、study、strict_multimodal和最终TEST；不因效果不足重训或改门。

终端 .\ff_prior_expand256_train\job.ps1 status|report|resume。源代码3b51595c/2278fe9f，冻结运行及中途快照保留；原应用delivery_v1已完成卡片不重开，本轮授权以此文件和交付证据为准。
