# V6.4-A result-to-claim verdict — 2026-10-02

Intended claim: learned trajectory proposals complete independent Tasks through the original safety control chain and improve on fixed references.

Verdict: **no**. Confidence: high for the measured negative conclusion. Paper claim audit: unavailable; review is provisional.

The completed raw comparison has fixed 3/3 TEST successes; Teacher and Diffusion each have 0/24 raw candidates accepted, K1/K8 0/3 total completion, and no actual execution. Conditional completion is N/A. A repaired VAL candidate ran 11,560 force steps and stopped at 23.12 s; it is neither a raw nor a TEST success. The original Task/safety tolerances and all failures remain unchanged.

The results support real data, training, weights, generation and safety integration. They do not support learned planning effectiveness, superiority, successful repair, broad generalization or deployment.

Missing minimum evidence: independent-Task raw reference actual actions and failures. Routing action: a separate, predeclared negative closed-loop diagnostic after full unchanged safety checks, preserving primary rejection and Task failure. The 72-slot engineering expansion is prepared but NOT_RUN. More static optimization and wall timing are deprioritized.

Review prompt, raw response, hashes and routing: [review trace](paper/review-traces/experiment-result-to-claim/20261002_run01/handoff.md). This note predates the diagnostic run; subsequent results must be appended without rewriting this verdict's inputs.

Update at 2026-10-02 18:36 +08:00: the three TEST candidate0 negative-diagnostic preflights all failed complete native geometry. A finite follow-up checked the other 21 saved candidates: all were also unsafe. Thus 24/24 original v5 TEST candidates lack safety admission and no diagnostic physical execution occurred. The aggregate recovery used already saved rows after a schema-key error, with source/input/asset verification; original error and cost remain. See [recovered geometry aggregate](v6_4/output/raw_k8_geometry_probe_01/report_recovered.json).

The clean_x0 mechanism completed its predeclared 20,000 steps and six TRAIN Tasks × eight proposals. Raw acceptance was 0/48, with 255/384 individual requirements met; VAL/TEST/physics were not run under the fixed stop rule. This is an additional negative model result, not independent Task execution. See [frozen TRAIN diagnostic](v6_4/output/training/three_family_clean_x0_01/diagnostic/report.json). A single already saved v5 EE VAL candidate0 safety preflight is separately authorized, with no resampling, repair or physics yet; any safety failure terminates that follow-up.

The next structural hypothesis is a generation-time Task-conditioned finite Teacher prior plus learned raw residual. The current network predicts absolute control points; its epsilon_residual name refers only to noise coordinates. Prior-assisted generation needs its own representation, checkpoint, provenance, complete prior cost and prior_only comparison. It cannot change old raw-network acceptance or establish a network-only benefit. It remains a proposal, not a completed experiment.

The unique saved v5 EE VAL candidate0 preflight has now finished: safety 0/1, actual steps 0, complete native minimum −10.800966 mm. All 3,954,377 queries and original evidence are retained; fresh prediction matches the old NPZ byte hash. This follow-up is stopped. [Report](v6_4/output/raw_val_negative_preflight_01/report.json), SHA `01c42bc17c919a4486dea474a1731d1b3ff9f1cd25242bec978d2b840b46717b`. Structural B implementation and focused tests are authorized; centers, training and physics have not started. [Predeclared protocol](v6_4/docs/PRIOR_RESIDUAL_EXPERIMENT.md).

Subsequent terminal update: B constructed all six TRAIN centers and retained all 13 labels; 31/48 prior candidates and four selected centers passed full qualification, with two geometry-failing centers kept only as conditions. Its unique 20,000-step residual Transformer checkpoint completed, but all 48 generated TRAIN proposals failed Task precision: 294/384 requirements met, 45 continuum terminal, 41 rigid terminal and four waypoint failures. Acceleration passed 48/48; geometry was not run after Task precheck failures. The fixed stop rule left VAL/TEST/physics at zero. Source/input and 1,802 historical-file checks passed. [Terminal summary](v6_4/output/prior_residual_result_01/REPORT.md) and [qualification](v6_4/output/training/three_family_teacher_prior_01/raw_qualification/report.json) (`4a1060342b8c81fd7f1be2eb49324864f59ebd7618f55ecf243433e3747fde20`). No independent learned actual evidence has been added.

A new bounded mechanism C is in implementation: conditional empirical residual-mixture score diffusion, with real conditional-weight fitting and analytic DDIM denoising, without output snapping or repair. It is a separate prototype-supported method, not a successful Transformer, new topology or network-only result. [Predeclared protocol](v6_4/docs/EMPIRICAL_SCORE_EXPERIMENT.md). No C fit, sampling or physics has occurred at this update.


追加范围更正与C终态（2026-10-02T20:49:01+08:00，保留以上全部历史判断及review trace原件）：C已完成13 TRAIN标签/6条件的有限经验条件score拟合，TRAIN原full gate∧原ddq为48/48，VAL为0/24；VAL24/24原ddq通过、独立任务点/终端均失败，几何NOT_RUN。按固定停止规则C TEST/物理0，coordinator为PREPARED_NOT_ELIGIBLE。[保存式终态记录](v6_4/output/empirical_score_result_01/REPORT.md)；[机器报告与来源SHA](v6_4/output/empirical_score_result_01/report.json)。C的泛化/独立TEST能力或网络收益判断仍为no，confidence=high；工程结果pending，paper claim audit仍unavailable，不能确认论文优势或goalcomplete。

此次按原目标附件字面澄清最低证据边界：第58行要求真实训练/权重/生成/力矩闭环，第323行明确允许各方法同预算的工程增强组，第400行不要求第一版所有任务优于传统规划，第407—409行要求独立任务实际执行并保存原始提案、选中参考、命令、力矩与失败，且分开直接提案、后处理、多候选、fallback贡献。附件SHA `4021dea68d42b77387e1dc0ec7d7fdd4045c6ed079fae4c1003bba999226610a`。因此，历史notes中的“最低必须independent-Task raw actual”是根代理额外收紧，本条追加更正为“独立TEST来源关联的真实执行/失败，以及公平三方法归因报告”；旧raw主矩阵、旧负诊断、prompt/output/trace原封不动。该更正不是删除raw指标，也不豁免原Task/full安全门槛。

下一路由为supplement而非宣称已有收益：旧V5/Teacher原24slots及fixed_joint_prior每Task8个相同种子，组成统一冻结repair-v2预算的3TEST×3方法×8=72slots工程组。当前仅PREPARING_NOT_RUN，不能写工程完成。原固定Cartesian3/3仅历史背景；fixed_joint_prior不能冒称原baseline。完整Teacher替换属于fallback/接管，不得给模型proposal信用；修复若执行仍独立保留原raw失败与修改量、成本、真实命令/力矩/失败。原QP、Task、安全距离/PCC域/半径、导数、执行器限制与容差均不放宽；不以TEST未来actual作输入，不按实际结果换选、不重试失败来填配额。C本身停止，不追加fit、TEST或物理。


追加最终有限负结果与两项自产门槛更正（2026-10-02T21:59:01+08:00；此前全部字节与review trace原封不动）：原附件L323允许共同预算工程增强，L400允许首版失败/不优于传统方法，L407—409要求独立任务实际执行与来源归因；未要求accepted raw，也未要求最终V5正式TEST actual。此前raw-min及TEST/V5-only是代理额外加严，本条同时纠正；原raw0负结果不删除。旧a88 EE VAL与其3TRAIN groups独立性已有[范围审查](v6_4/output/minimum_delivery_scope_audit_01/report.json)（SHA c6e86e3414cc3c258188a4524a94a0a458b2252fcf0e548839ace60445183d46），旧11560步/23.120s非raw/nonTEST/nonV5/非完整fresh。

当前公平工程02已终态72slots/18cells。fixed_joint_prior及Teacher资格各0/24；V5 Diffusion repair16/24原gate∧原ddq合格，固定K1/K8选择后4独特TEST actual共33540保存力矩步，均下一servo前iteration_limit拒绝，六方法K分组Task均0/3。新增真实V5 TEST修复执行是独立工程负证据，非raw/NN成功，无fallback、无按actual换选/重试。原fixed Cartesian3/3主表、Teacher/Diffusion raw K1/K8 0/3与所有失败保持。

独立保存力矩fresh重放33540步与controller33540步分账；不能合称新增实际67080。原评价把未消费NaN末拒绝行当命令缺失而ERROR/false，原件保留。只读已存fresh恢复4/4 physical-prefix evidence：已消费合同/独立区间/robot-target通过；wholebody NOT_AVAILABLE，full27/fulltrial/Task成功false。恢复0新controller/force replay/native geometry；original row evaluation=null/evidence false不回填。全部path/base/17D向量L2 shape/force/延迟、原失败和成本见[自动报告](v6_4/output/delivery_summary_final_01/REPORT.md)及[机器来源manifest](v6_4/output/delivery_summary_final_01/manifest.json)。

原目标有限负结果最低研究证据与报告已具备；性能/学习效果claim_supported=no，confidence=high；paper claim audit仍unavailable/provisional。直接raw学习actual与Task成功尚未建立，不把repair收益归NN，不声称优势、广泛泛化、硬件、鲁棒或连续安全。A/B/C均停止不追加：A/B TRAIN0/48后VAL/TEST0，C TRAIN48/48但VAL0/24后TEST0，三机制物理0。sim20ms/2ms与原全部安全不变；wall20ms非研究门禁，旧wall NOT_MET/hardware未建立。goal最后状态由根代理决定，本次仅结果归档与文档闭合。


追加最终有限负结果与两项自产门槛更正（2026-10-02T22:00:49+08:00；此前全部字节与review trace原封不动）：原附件L323允许共同预算工程增强，L400允许首版失败/不优于传统方法，L407—409要求独立任务实际执行与来源归因；未要求accepted raw，也未要求最终V5正式TEST actual。此前raw-min及TEST/V5-only是代理额外加严，本条同时纠正；原raw0负结果不删除。旧a88 EE VAL与其3TRAIN groups独立性已有[范围审查](v6_4/output/minimum_delivery_scope_audit_01/report.json)（SHA c6e86e3414cc3c258188a4524a94a0a458b2252fcf0e548839ace60445183d46），旧11560步/23.120s非raw/nonTEST/nonV5/非完整fresh。

当前公平工程02已终态72slots/18cells。fixed_joint_prior及Teacher资格各0/24；V5 Diffusion repair16/24原gate∧原ddq合格，固定K1/K8选择后4独特TEST actual共33540保存力矩步，均下一servo前iteration_limit拒绝，六方法K分组Task均0/3。新增真实V5 TEST修复执行是独立工程负证据，非raw/NN成功，无fallback、无按actual换选/重试。原fixed Cartesian3/3主表、Teacher/Diffusion raw K1/K8 0/3与所有失败保持。

独立保存力矩fresh重放33540步与controller33540步分账；不能合称新增实际67080。原评价把未消费NaN末拒绝行当命令缺失而ERROR/false，原件保留。只读已存fresh恢复4/4 physical-prefix evidence：已消费合同/独立区间/robot-target通过；wholebody NOT_AVAILABLE，full27/fulltrial/Task成功false。恢复0新controller/force replay/native geometry；original row evaluation=null/evidence false不回填。全部path/base/17D向量L2 shape/force/延迟、原失败和成本见[自动报告](v6_4/output/delivery_summary_final_01/REPORT.md)及[机器来源manifest](v6_4/output/delivery_summary_final_01/manifest.json)。

原目标有限负结果最低研究证据与报告已具备；性能/学习效果claim_supported=no，confidence=high；paper claim audit仍unavailable/provisional。直接raw学习actual与Task成功尚未建立，不把repair收益归NN，不声称优势、广泛泛化、硬件、鲁棒或连续安全。A/B/C均停止不追加：A/B TRAIN0/48后VAL/TEST0，C TRAIN48/48但VAL0/24后TEST0，三机制物理0。sim20ms/2ms与原全部安全不变；wall20ms非研究门禁，旧wall NOT_MET/hardware未建立。goal最后状态由根代理决定，本次仅结果归档与文档闭合。

运行后仅报告路径修复已终态：[receipt.json](E:/桌面/mujoco_irregular_waypoints_diffusion_adapter_bundle_20260902-main/Hybrid_dual_arm_space_manipulator/v6_4/output/post_run_partial_reporting_fix_01/receipt.json)，SHA `f8036faa92e324e0217b0553a5cc9e339d3c7183fca0807b2407d0783f22eb2e`。5项focused测试通过；4实际prefix纯执行合同与冻结recovery逐字典一致；25选定历史文件bytes/SHA不变；0新增mj_step/controller/force replay/geometry。future evaluate_planning raw SHA `29127f4ddc692a783661145e1e9aca6e28ecb83843c4af6ab6a87ee3c8e165ad`，future run_planning raw SHA `929b8de2e52aaa1081db848f343bc83a5f7c7ee0b2fc7b2b60ae69b13af635fd`。这是单独后续源码身份，原run当时的source unchanged成立；不回填旧null/ERROR/sourceSHA、不得假装历史已用新版本跑过，也未补全wholebody或Task成功。

上一215901文档的修复pending为当时收到receipt前的快照，保留原件；本次新timestamp追加实际终态，不覆盖该历史。


## 20261016_181913 qualified_modes_smoke_001

Independent experiment-result-to-claim verdict: partial; confidence medium; no paper claim audit available. 48/48 local fixed-world TRAIN additions,4/4 preselected real executions; TRAIN256 with30families. Matched local goal loss reduces queries22.89% and candidate time21.12% versus v-only, but learned mean remains cheaper. All5methods50/50candidate,10/10execution/replay. Full reproducible superiority and production expansion remain incomplete. Route: TRAIN residual-amplitude mechanism diagnosis, then independently frozen multiseed confirmation. Report: ff_prior_modes/docs/ANALYSIS.md; reviewer trace: paper/review-traces/experiment-result-to-claim/20261016_modes_run01/. Do not reinterpret local enrichment as full expansion or posthoc quality diagnostics as gate success.

## 20261017_220443 world_endpoint_smoke_001 closeout

Claim verdict partial: C1 stable implementation/data/native flow; C2 and TRAIN endpoint improvement no. 64 qualified corrections; 13838 TRAIN/6000 updates/120 true gradients. TRAIN paired ratio4.608628, reusedDEV8 new/old1.239667, raw0/12. New model experimental only; no1024 expansion orTEST. Review trace20261017_world_endpoint_run01; reportff_prior_world_endpoint/docs/ANALYSIS.md. Missing causal evidence: old local G removed, sparse world coverage, all world mixed gradients clipped, no separate-gradient logs, one training seed. Route to TRAIN-only optimization diagnosis, not blind expansion. Root append-only notes not staged wholesale.


## 2026-10-19 qualified_expand256_001 + expand256_train_dev_001 closeout

Fresh independent review: C1 yes/high (bounded engineering closure), C2 no/high (no stable Diffusion superiority). Expansion 256/256 actual attempts, 32/32 native executions/replays; 224 MODEL_VALID + 32 EXECUTION_VERIFIED; existing 64 TRAIN goals densified, no new global goal/origin coverage. TRAIN14222 all consumed in6000 updates/384000 draws; all four methods180/180 candidates and36/36 executions on originalDEV4+used8,3 sampling seeds. New/old executed mean position0.737807/0.710278mm (13 better,23 worse), mean orientation0.074039/0.075929deg; mixed tradeoff, no overall improvement claim. Data augmentation and extra updates confounded. Confirm bounded engineering delivery, narrow scientific claim, retain old default recommendation and new qualified candidate, no further experiments/expansion/TEST. Fresh trace paper/review-traces/experiment-result-to-claim/20261019_expand256_train_run01; full report ff_prior_expand256_train/docs/ANALYSIS.md. Paper claim audit unavailable; downstream writing provisional only. No repeated physics/training/audit, one no-work resume per run verified.

2026-10-20 capture_feasibility_001：C1=no、C2=no，总体no；confidence high（证据边界）。12槽未取得严格见证，表示模块BLOCKED；粗网格漏碰及局部求解问题有保存证据，不能证明不可达或102维瓶颈。路由修订求解器与细几何反馈，后续另冻有界协议，不自动追加；review见paper/review-traces/experiment-result-to-claim/20261020_capture_feasibility_run01。

2026-10-21 capture_refine_001收尾：398.641s，原00/07最终路径0/2 MODEL_VALID，0真实力矩执行/重放；07同16384网格穿透5.143923→2.154987mm，仍碰撞。六次SLSQP均160迭代上限，零步和返回可行性保护有效，4更新接受；不能称收敛、精度提升、不可达或Diffusion优势。独立review总体no，C1 no/C2 BLOCKED，confidence high限负结论；133源码/192绑定/1938差分/27试探/10网格/1991耦合/1822572积分独立代数复核。M0非位姿数组和部分距离/回调原数组未保存，科学证据限制保留。完成resume一次后121文件hash不变，stdout编码捕获失败且未重调。路由revise/stop；下一步可行步恢复和自适应信赖域需新授权，当前不追加。未训练、扩库、TEST，物理门和旧结果不变。报告ff_prior_capture_refine/docs/ANALYSIS.md；review 20261021_capture_refine_run01；paper audit不可用，下游论文provisional。

2026-10-21 capture_trust_001收尾：1161.968s；原固定00/07两起点1/2 MODEL_VALID，07原生67力矩13750步+独立重放13750步通过。模型正间隙0.036019mm，执行报告含中点0.026969mm；单任务微小间隙不代表鲁棒性。00仍穿透33.958mm。纯codec/prepare/8步102修复全不合格（未执行）；修复终点1.798mm但穿透10.283mm，表示诊断完成。原audit FAIL及FINISHED_WITH_BLOCKED_ITEMS保留，新tools/audit_saved_v2仅恢复实际乘法算序、不改容差/物理门，8568列/31接受/5验证/1执行PASS。全新review 20261021_capture_trust_run01裁决partial/medium，C1单任务见证yes/C2保真no，paper audit unavailable/provisional。第二重放轨迹/中点序列/全初态向量等缺口明确。一次resume实际NO_WORK_ALREADY_COMPLETE，350文件清单/hash不变。不训练扩库TEST，不宣称Diffusion收益。下一阶段仅建议另冻表示保真验证，当前不追加。报告ff_prior_capture_trust/docs/ANALYSIS.md。

## 2026-10-21 rep_fidelity_001

预定紧凑表示C1=no；prepare资格保持C2=partial，rank19同输入配对可限定yes；overall partial/high（单例），新独立review已完成，paper claim audit unavailable/provisional。238维世界终点通过但穿透17.904mm，只有323原样身份路径通过新执行重放。支持本案例完整控制保留与平滑破坏诊断，不支持学习、泛化或Diffusion优势。工程交付完成，不自动补实验。下一步仅建议TRAIN-only完整控制学习闭环。证据边界见ff_prior_rep_fidelity/docs/ANALYSIS.md及新review trace。


## 2026-10-21 full_control_bounded_001 完整323控制闭环

意图：独立TRAIN资格保真后，检验完整控制学习与冻结DEV相对收益。新的独立experiment-result-to-claim：C1 yes（仅6例重建/3例新执行）、C2 no、总体partial；confidence high（限定证据），paper claim audit unavailable，论文写作结论provisional。6/6重建及3/3执行支持完整表示/prepare在该范围可用；新无引导0/96 vs旧24/96，DEV15/180及13/36 vs旧180/180及36/36，不支持学习或Diffusion优势。DEV165拒绝仅密度不足；三/五障碍实际穿透仍存在。6000真实训练和四保存更新原始重建通过，不能把失败归为未训练。输出/codec/初始化/损失/prepare混合变化，缺少因果隔离、硬件和未触碰TEST证据。下一步pivot到函数保持迁移和全维时间规律，先TRAIN可学性；仅建议，用户本轮禁止追加实验。保留旧基线，不扩库、不放宽门、不TEST。详见ff_prior_full_control/docs/ANALYSIS.md与paper/review-traces/experiment-result-to-claim/20261021_full_control_run01/。


# control_transfer_bounded_001：结果与主张

本轮源码df13c4e1，后台135.03秒；保留完整323控制、旧低102功能、221受控细节及资格保持prepare。
独立review：paper/review-traces/experiment-result-to-claim/20261021_control_transfer_run01/。
结论partial：16/16迁移等价、6/6教师重建、3/3本轮原生执行与独立重放；256次临时TRAIN更新将全门合格从1/16升至6/16，Q中位下降79.63%，全部16条通过模型物理门。
差距：未达到12/16，原唯一合格样本退化；正式6000步候选、VAL与DEV均BLOCKED且未运行。C2相对收益未评价，不可称新候选无效。临时权重不进入候选，诊断6条未执行，不能升级执行等级。
退化样本原教师/目标/初态与缓存身份一致，控制RMSE下降但世界误差增大。当前证据支持控制MSE与逐条世界终点验收不一致，未证明微重力建模错误。
下一建议：在完整16步采样输出上加入真实世界终点方向监督/约束与原合格样本保资格约束；先冻结导数和预算进行有界TRAIN诊断。此处只建议，不自动追加试验。
保留旧结果、用户修改，DEV未入TRAIN；不扩库、不放宽物理门、不TEST。论文级结论provisional，参见独立verdict的paper claim audit状态。长期目标未完成。


## 2026-10-21 rollout_world_bounded_001 新独立结果到主张

完整323采样世界监督+真实资格保护：固定TRAIN M1 1→15/16，formal0→14/16；C1限定8目标2种子及接受的细化点=yes。VAL24→4/96，原DEV180/180+36/36仅工程持平，五障碍0/5；C2稳定相对收益=no。整体partial、confidence high，论文状态provisional（PAPER_CLAIM_AUDIT缺失），新reviewer /root/rollout_world_fresh_review，非旧review复用。
本轮停止，保留旧默认模型；下一轮需新的事前协议，考虑功能保持、独立TRAIN检查池与障碍条件，不自动补实验、扩库或TEST。证据ff_prior_rollout_world/docs/ANALYSIS.md、CLAIM_VERDICT.md及paper/review-traces/experiment-result-to-claim/20261021_rollout_world_run01。旧应用仅2条成功执行复用，新37条实际执行。长期科学目标未完成。
