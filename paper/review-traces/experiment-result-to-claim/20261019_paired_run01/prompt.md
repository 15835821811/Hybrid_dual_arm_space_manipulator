# RESULT-TO-CLAIM 独立评审任务

任务来源：主执行代理 `/root` 对新的独立 paper architect/reviewer 的委派。评审前未提供其他代理的结果表或先前判词。技能：`C:/Users/admin/.codex/skills/experiment-result-to-claim/SKILL.md`，以及其 review-tracing 与 assurance-contract 引用。

初始委派原文：

> 你是新的独立 paper architect/reviewer，按 C:/Users/admin/.codex/skills/experiment-result-to-claim/SKILL.md 执行 RESULT-TO-CLAIM 评审。工作区 E:/桌面/mujoco_irregular_waypoints_diffusion_adapter_bundle_20260902-main/Hybrid_dual_arm_space_manipulator。先读 ff_prior_paired_objectives/docs/EXPERIMENT_PLAN.md 和 DIFFUSION_GOAL_CURRENT.md。实验 artifacts/v7_1/paired_objectives_smoke_001 四组6000训练、各32 TRAIN探针和 training_verification.json 已完成；主worker目前仅进行 raw_results_audit（来源哈希）；TRAIN_selection.json已BLOCKED，无DEV。请直接读原始 single_step_summary.json、probes/{warm,A_v,B_local,C_world,D_both}.json、learning/*/training.json、paired_random_sequence.json、world_gradients/*、ledger.json及对应源码。独立复算32例终点损失/位置/姿态/准入门、核对同起始权重/数据/随机序列/6000更新/240梯度的证据和边界，适当抽查保存权重对应预测及梯度。不要重新运行训练、采样、物理传播、执行、已有全来源大规模审计，不修改冻结源码或实验原始文件。可以只用已保存终点和权重验证；明确自己审查范围，不冒称未检查内容。评估 C1（可用真实梯度及配对实现）和C2（改进终点、DEV资格及Diffusion优势）分别是否支持，区别一训练种子TRAIN探针与DDIM/DEV/物理执行。检查关于局部项删除、真实项尺度、裁剪、高噪声退化的因果表述是否过度。核对论文claim audit存在则遵技能，不存在标写作provisional而非数据阻塞。输出到 paper/review-traces/experiment-result-to-claim/20261019_paired_run01/，包含 prompt.md, inputs.json, response.md, verdict.json, handoff.md 及你的小型只读验证脚本和原始验证结果。仅在该评审目录写文件。不要git commit，不再委派。主worker结果若尚未terminal，先完成独立数据分析并告知我pending source audit；待终态后可补最终判定。现有结果不提供给你以减少锚定，请自行读取。

评审实施与输入：

- 先读取冻结协议和当前目标，再读训练、单步、真实终点、模型、随机序列及审计源码。没有再次委派。
- 完整输入路径、SHA256、大小和修改时间保存在 `inputs.json`。核心输入是本实验的 5 份探针、4 份训练记录、6000 步序列、8 个单步诊断、240 份世界梯度 JSON、联合优化器终态和原始账本。
- 独立计算 160 个保存终点的损失/位姿/准入；重建 160 个固定单步去噪网络预测；检查全部 8 单步诊断的保存终点与102维差分；抽查 C/D 的事件 0、31、64、119 共8个世界梯度检查点及网络参数 VJP。
- 核对序列内部一致性和实际抽取次数；只重建5个已记录更新的固定噪声哈希，不重复完整 CoverageSampler 审计。
- 原始训练证明与全部来源审计由主worker执行；评审途中收到已终态消息后读取 `independent_verification.json`、`status.json`，结果 PASS / FINISHED_WITH_BLOCKED_ITEMS。来源审计 pending 标记已清除。
- 独立数据复算完成后，主执行者提示补查 D 的位置/姿态与双门转移；评审重新从原始终点独立计算并确认，结果写入 `D_pose_tradeoff`。
- 主执行者完成 resume 后，读取 `ff_prior_paired_objectives/evidence/completed_001/resume_noop.json` 并核对9项当前文件哈希。评审者没有再次执行 resume。

本评审不进行新训练、不调用 optimizer.step、不生成 DDIM 候选、不做物理传播、碰撞检查、原生执行或重放。固定噪声重构仅供检查已保存网络预测。仅在本评审目录写入文件；项目根 findings/tracker 的整合交由主执行者完成。

评审问题：分别判定 C1、C2 的支持度；给出数据支持/不支持的范围、缺失证据、建议改写、下一有界实验建议与置信度。不得用单一训练种子、开发用途 TRAIN 探针或成功的数值检查替代 Diffusion 优势与真实执行证据。
