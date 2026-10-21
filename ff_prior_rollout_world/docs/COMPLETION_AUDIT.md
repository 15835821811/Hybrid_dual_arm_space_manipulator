# 有界目标逐项完成核对

此处判断是否完成用户授权的实验与交付，不把“科学收益成立”改成工程通过。

|原要求|实际证据|判定|
|---|---|---|
|323维旧功能迁移与受控细节保留|冻结protocol/model/codec及executed_source.zip；M0复用迁移，初始完整采样逐元素一致；新独立审查source/refinements|实现完成；训练后全分布旧功能未保证|
|完整采样世界监督|TRAIN_world及candidate_world的96份12×323 J、扰动位姿、独立半步/方向检查；完整16步VJP、首末更新权重/optimizer|实测完成；内层局部代理，非逐步真实非线性传播|
|原合格保护、实际步长验收|6轮全部保存before/proposed/retained、trial、真实全机验收；M1接受3轮，formal1轮；其它拒绝恢复|固定锚点接受结果保护成立|
|TRAIN门通过才训练|learning_gate PASS、M1 15/16≥12、Q改善、原索引12合格；ledger阶段序列|门通过后执行|
|一个有界正式候选|ordinary初始hash等于原迁移而非M1final；14222样本计数均>0；6000optimizer steps、300摘要、1200full16批；full权重重载|完成，无重复训练|
|无引导诊断|新96条各16步、physics_guidance_calls=0、raw=prepare；旧96哈希复用；同task/seed配对|完成；新4/96低于旧24/96|
|原冻结DEV同预算|原12×3×5新180；旧540仅复用；每条113/15872/15秒内；task/seed身份及选择记录|完成；新180/180、36/36，但收益未建立|
|静态应用验证|原三/五障碍各5新候选；三障碍原生67力矩和独立重放；五障碍几何失败不执行|完成既定对照；五障碍目标未解决|
|分级资格与真实执行|37个新report/native/replay、ctrl一致、全初态/物理门；405500物理步|37实际EXECUTION_VERIFIED，不升级未执行样本|
|DEV不入TRAIN、不扩库、不TEST|training_input/codec fit_ids、forbidden_TRAIN_task_hashes不相交；TRAIN_added=0；协议/ledger无TEST阶段|保持|
|旧证据/用户修改保留|输入身份绑定；完成resume的3351文件hash一致；4个用户修改hash一致|保持；只读复用，不重跑旧任务|
|后台与终端可查、完成不重复|job.ps1 status/report；完成resume仅一次返回NO_WORK_ALREADY_COMPLETE|已验证|
|全新独立审查|paper/review-traces/experiment-result-to-claim/20261021_rollout_world_run01；CLAIM_VERDICT.md|须以最终review文件为准，不复用旧审查|
|报告/目标/tracker/提交|ANALYSIS.md、CLAIM_NOTES.md、DIFFUSION_GOAL_CURRENT.md、版本tracker、本轮收尾提交|交付文件齐全后结束本有界目标|

不满足的长期目标：稳定无引导世界目标命中、稳定Diffusion相对收益、固定五障碍绕行抓捕、最终TEST及动态航点。它们如实保留未完成；本轮没有授权继续追加实验。
