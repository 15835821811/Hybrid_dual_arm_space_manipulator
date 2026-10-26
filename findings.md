# Research findings — V6.4-A

The completed raw TEST comparison does not support the planned superiority claim: fixed succeeds on 3/3 Tasks; Teacher and Diffusion accept 0/24 candidates each and perform no actual steps. This is a quality failure, not successful safe planning merely because the controller refuses motion. These three fixed successes leave no room for a higher success-rate claim on the same sample.

The v5 model also fails raw gates on all 48 TRAIN and 24 VAL samples. TRAIN terminal errors exceed original tolerances; training loss cannot substitute for task accuracy. The isolated CUDA runtime did not change the old MuJoCo/numerical environment. The observed normalized condition values do not support a zero-variance normalization explosion hypothesis.

Static repair is insufficient to establish execution: a full-gate-passing VAL reference reached a state where original ramp-domain and acceleration bounds directly conflict. This reached-state infeasibility is not whole-Task infeasibility. Weak posture preference allows actual shape to differ from the shape used for nominal free-base prediction; raising solver iterations does not remove the conflicting bounds.

Next action is bounded: complete a separately labeled raw negative closed-loop diagnostic, only after full unchanged safety screening. Do not restore raw acceptance or alter the primary matrix. No new training, resampling, repair, fallback or TEST future-state input is authorized by this diagnostic. The engineering expansion remains prepared and unrun. A later attempt to obtain planning success should model the original controller during planning and stop at a fixed TRAIN/VAL budget.

Sources and provisional reviewer verdict: [EXPERIMENT_RESULT_TO_CLAIM.md](EXPERIMENT_RESULT_TO_CLAIM.md).

Terminal follow-up, 2026-10-02 18:36 +08:00: complete native checks reject all 24 original v5 TEST references; the raw negative-diagnostic route therefore produced zero actual steps and is closed. A single saved v5 EE VAL candidate0 independent safety preflight is being prepared separately; no resampling or physical execution has occurred.

One clean_x0 model was trained from scratch for 20,000 steps using the same 13 successful TRAIN labels. All 48 TRAIN proposals failed original Task precision and stopped before geometry; 255/384 individual requirements passed. The declared zero-acceptance stop rule left VAL/TEST/physics unrun. The closest candidates still lacked full acceptance, so more training steps were not appended.

The 32-control-point representation exactly reconstructs the 13 actual-success reference labels. Condition position quantization was only about 1 nm; all frozen task points, times, orientations and terminal flags enter the condition. The current model has no Task-dependent trajectory center: its normalizer supplies one global TRAIN-derived 17D mean/scale. A finite Teacher-prior-assisted residual mechanism is a materially different hypothesis. Its common-path and precision contribution must be separated from learned residuals, while the original Task/safety gate remains unchanged. It has not yet been implemented or trained.

The one saved EE VAL preflight also failed the unchanged native geometry threshold (−10.800966 mm), closing this route with zero physical steps. Structural B is now in implementation/testing; no new centers, model weights, generated candidates or actual execution yet. The finite [protocol](v6_4/docs/PRIOR_RESIDUAL_EXPERIMENT.md) fixes the prior source, optimizer budgets, independent identities, attribution and stopping conditions.

Later terminal update: structural B also stopped at TRAIN raw0/48 after its unique20k fit. Prior-only selected centers qualified4/6, but network outputs met294/384 Task requirements and all48 acceleration guards; failures were continuum terminal45, rigid terminal41 and waypoints4. Geometry was not run after these Task failures. VAL/TEST/physics remain0; prior-only quality cannot be credited to the network. [Saved-only summary](v6_4/output/prior_residual_result_01/REPORT.md).

The next bounded hypothesis is [C, conditional empirical score diffusion](v6_4/docs/EMPIRICAL_SCORE_EXPERIMENT.md): preserve finite TRAIN residual modes through an explicit conditional distribution and analytic reverse diffusion. It fits group probabilities from task features, with uniform within-group route weights; it does not look up the inference Task id, snap outputs to a prototype, or claim new topology. C is only in implementation at this point. Independent learned reference actual execution remains the minimum evidence gap.


追加C终态与最小范围修正（2026-10-02T20:49:01+08:00）：C经验条件score机制完成真实有限拟合，固定TRAIN48/48原gate∧ddq合格，但固定VAL0/24（独立任务点/终端失败；ddq24/24通过，geometry NOT_RUN），因此按原停止规则无C TEST/物理。支持TRAIN拟合与资格，不支持泛化、独立TEST闭环、Transformer成功、新拓扑或网络独立收益。[终态报告/原件快照与SHA](v6_4/output/empirical_score_result_01/REPORT.md)。

原附件第323行允许统一预算工程增强组，第400行允许首版产生失败/劣于传统方法的可判断结果，第407—409行要求独立任务真实执行、原始/选中参考、命令/力矩/失败及直接模型/后处理/多候选/fallback归因。此前accepted-raw TEST actual作为最低缺项是代理额外加严；本次追加修正最低剩余项为独立TEST来源关联的真实执行/失败和公平三方法报告，不删除原raw负结果、不修改原Task/安全、不改旧review trace。真实工程执行可以补该证据边界，但不能冒充raw成功、网络收益或当前goalcomplete。

当前唯一下一步为原V5/Teacher/fixed_joint_prior三方法共72slots的冻结repair-v2工程对照，仍PREPARING_NOT_RUN；不继续Cfit/TEST/physical，不追加模型与样本。每Task8个相同fixed_joint_prior仍按实际处理成本记录，不是8个不同路线，且不是原Cartesian固定3/3。完全替换原模型提案为Teacher则按fallback/接管报告，不给模型proposal credit。结果尚不存在，维持pending，所有失败与旧raw0主矩阵保留。


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

## 2026-10-17 qualified_origin_coverage_001

Independent result-to-claim: partial/high; C1 original coverage gate not passed (503/512 origins,119/128 scenes,39 retained ERROR), C2 not evaluated. 535 qualified new TRAIN,12679 total,43 to546 unique TRAIN origins;32/32 native torque/replay and raw audit PASS. Retain samples without unblocking old training. Freeze bounded supplement for9 missing obstacle origins, preserve failures/physics, qualify a separate merged parent, then actual training and originalDEV/NEW_VAL comparison. No finalTEST/100k; paper claim audit unavailable/provisional; full goal incomplete. Report ff_prior_origin_coverage/docs/ANALYSIS.md; reviewer paper/review-traces/experiment-result-to-claim/20261017_origin_run01/REVIEW.md.

## 2026-10-17 补充库独立结果审查

C1 yes/high：新9执行重放通过，新合并12688 TRAIN/555初态，512/128门和原始审计PASS。C2未评估，完整目标false；旧39 ERROR/ready=false保持。后续覆盖平衡组按producer_run_id取544条。trace: paper/review-traces/experiment-result-to-claim/20261017_fill_run01/REVIEW.md。

## 2026-10-17 新覆盖库公开条件只读诊断

独立初态43→555；NEW_VAL8初态逐维经验越界由7/8降至1/8。最近48维公开条件距离部分下降、两任务不变；冻结codec标准化特征最大值15.1835→33.3167。只读哈希核验，0网络/物理/训练调用，不能作失败因果结论或神经收益证据；不改变当前冻结训练。报告ff_prior_origin_train_tools/evidence/condition_coverage_001/ANALYSIS.md。

## 2026-10-17 新覆盖真实训练闭环独立评审

qualified_origin_train_001 FINISHED，6920.985秒，12688 TRAIN实际384000消费，原DEV4+NEW_VAL8三种子四方法各180/180合格/36/36执行重放，144原始轨迹1512000步，原始审计PASS。claim_supported partial/high：C1 yes，C2对样条/更新均值no。新/旧NEW_VAL配对任务残差中位比.7957486、geo95CI[.5688619,.9318133]；对强样条3.0127842、均值.9877841。raw新12物理门通过但0指定目标；原DEV posepass新6/60、旧15/60。完整目标未完成，时间/查询不作扩库门；另冻有界1024独立初态扩库与新验证，原NEW_VAL后续调参须作DEV。trace paper/review-traces/experiment-result-to-claim/20261017_origin_train_run01/REVIEW.md。

## 2026-10-17 qualified_origin_batch_001 结果到主张

独立评审yes/high仅支持有界合格数据扩充及逐条分级：新增1086 TRAIN，合并13774，1578不同初态，56/56真实MuJoCo执行重放、原始审计PASS。原C2负结果保持，本批新训练/神经收益未评估，完整目标进行中。接续另冻结全体训练、13495锚点容量、组抽样和新验证同预算smoke，无效率相对门、最终TEST或自动100k。原计划人口笔误另附勘误，冻结原件保留。原始评审及VERDICT在paper/review-traces/experiment-result-to-claim/20261017_origin_batch_run01。


## 2026-10-17 qualified_origin_learn_002
独立结果审查partial/high：C1通过，强样条/均值C2未通过；实际6000更新/13774全行、各180/180和36/36。固定raw0/12指定目标，缩窄主张为共同物理引导下完整机制稳定。下一转真实非线性世界终点训练监督＋少量固定TRAIN修正；1024新前向批次保留未启动。不预设优势，详见独立review trace。

## 20261017_220443 world_endpoint_smoke_001 closeout

Claim verdict partial: C1 stable implementation/data/native flow; C2 and TRAIN endpoint improvement no. 64 qualified corrections; 13838 TRAIN/6000 updates/120 true gradients. TRAIN paired ratio4.608628, reusedDEV8 new/old1.239667, raw0/12. New model experimental only; no1024 expansion orTEST. Review trace20261017_world_endpoint_run01; reportff_prior_world_endpoint/docs/ANALYSIS.md. Missing causal evidence: old local G removed, sparse world coverage, all world mixed gradients clipped, no separate-gradient logs, one training seed. Route to TRAIN-only optimization diagnosis, not blind expansion. Root append-only notes not staged wholesale.

## 2026-10-19 paired_objectives_smoke_001

四组同数据/初始化/序列训练完成。D TRAIN损失中位比0.914089，门要求0.8；位姿4/32对旧5/32，DEV未运行。局部项抑制退化；世界项与局部组合有小幅收益，优化尺度/频率原因待验证。独立评审：{'audit_skill': 'experiment-result-to-claim', 'experiment_id': 'paired_objectives_smoke_001', 'review_role': 'independent paper architect/reviewer', 'review_trace': '20261019_paired_run01', 'generated_at': '2026-10-19T03:13:45.003873+00:00', 'claim_supported': 'partial', 'confidence': 'high', 'claims': {'C1': {'claim_supported': 'yes', 'confidence': 'high', 'scope': 'Only tested numerical world-model finite-difference gradients, network VJP/small-step descent and paired implementation. Eight TRAIN diagnostic cases and one paired training seed.', 'what_results_support': ['8/8 parameter-direction checks and descent gates passed; all three tested negative-gradient steps decrease each case.', 'Same initial checkpoint, frozen codec, data manifest and shared random tensors; 6000 updates and 384000 draws per arm; 240 current-prediction world-gradient events.'], 'what_results_dont_support': ['Globally valid analytical gradients or derivatives of native physical execution.', 'Guaranteed descent of mixed AdamW updates, stable learned endpoint generation, generalization or multiple-seed robustness.']}, 'C2': {'claim_supported': 'no', 'confidence': 'high', 'scope': 'Pre-registered TRAIN eligibility failed; downstream DEV/DDIM/native execution and superiority were not run in this round.', 'best_arm': 'D_both', 'median_paired_ratio_vs_warm': 0.9140885911213292, 'median_paired_ratio_vs_B': 0.8536475351189914, 'TRAIN_gate_threshold_vs_warm': 0.8, 'TRAIN_gate_threshold_vs_B': 1.0, 'selected_arm': None, 'DEV_status': 'BLOCKED_BEFORE_RUN', 'what_results_support': ['D improves paired scalar loss relative to B on 25/32 development probes and relative to warm on 17/32.', 'Retaining local loss improves this fixed paired run, while world-loss effects depend on local-loss presence.'], 'what_results_dont_support': ['Meeting the 20% TRAIN improvement gate.', 'Increasing simultaneous 5mm/1degree endpoint qualification: warm5/32 versus D4/32.', 'DEV failure or success, full DDIM improvement, execution qualification, unseen generalization, or Diffusion superiority over strong spline and conditional mean.']}}, 'what_results_support': 'Numerical world-model endpoint gradients are usable at measured points; paired objective comparisons show a local-loss benefit in this one-seed fixed TRAIN setting. D mitigates degradation but remains ineligible.', 'what_results_dont_support': 'Stable target learning, qualification of new models for DEV/execution, Diffusion advantages or unique causal explanations involving deleted local loss, clipping or high noise.', 'missing_evidence': ['A world-loss arm passing the frozen TRAIN gate.', 'An isolated scale/clipping/noise intervention to establish the proposed failure mechanism.', 'Complete DDIM and frozen same-budget DEV comparisons, followed by native execution and replay for eligible models.', 'Strong spline and conditional-mean comparisons, multiple training seeds, and unused development tasks.', 'Paper-to-evidence claim audit when downstream writing is prepared.'], 'suggested_claim_revision': '在单个固定配对TRAIN实验中，当前预测处的完整102维数值世界终点梯度通过参数链式传递及小步下降诊断。保留局部项明显缓解本轮终点退化，但最佳组合未达到预注册TRAIN门，端点同时达标率未提高，尚无本轮DEV、完整DDIM和真实执行优势证据。', 'next_experiments_needed': ['Suggestion only, not authorization: freeze a bounded single-factor world-gradient scale/normalization comparison while retaining local loss and holding the other choices fixed.', 'If noise attribution is pursued later, separately freeze a variant/noise-decoupled design.', 'Only after the existing TRAIN gate passes, perform the frozen downstream comparisons and then seek multiple-seed and unused-task evidence.'], 'routing_action': 'supplement_after_separate_protocol; close current round; keep DEV BLOCKED; do not automatically launch new experiments', 'causal_boundaries': {'local_deletion': 'A controlled contribution is supported within this paired run, not a unique explanation for the previous run.', 'world_scale': 'Weighted world gradients dominate on recorded world events; the finding is a diagnostic, not a proven cause of endpoint degradation.', 'clipping': '240/240 world events clip, but no unclipped/normalized causal control exists; clipping coefficient does not equal AdamW parameter step scaling.', 'noise': 'Both measured noise bands degrade for A/C; training noise time is bound to correction variant and event counts differ64/56. No high-noise-only causal claim.', 'probe_definition': 'One denoising prediction from a noisy TRAIN teacher trajectory plus saved coupled endpoint propagation; not 16-step DDIM or DEV.', 'pose_objective': 'Squared aggregate loss improvement does not imply every position and orientation inequality improves or passes.'}, 'validation': {'reviewer_read_only_checks': 'PASS', 'checks_passed': 7411, 'reviewer_scope': {'endpoint_cases': 160, 'fixed_one_step_neural_predictions': 160, 'single_step_cases': 8, 'single_step_full102_arithmetic': 8, 'world_event_metadata': 240, 'world_full102_and_weight_VJP_spot_checks': 8, 'paired_sequence_internal_checks': 6000, 'optimizer_final_states': 4, 'all_source_row_audit_repeated': False, 'fixed_random_tensor_spot_checks': 5, 'resume_execution_repeated_by_reviewer': False, 'all240_current_prediction_audit_repeated': False, 'training_replayed': False, 'DDIM_or_execution_run': False}, 'worker_training_verification': 'PASS', 'worker_source_audit': 'PASS', 'worker_terminal_status': 'FINISHED_WITH_BLOCKED_ITEMS', 'executor_resume': 'NO_WORK_ALREADY_COMPLETE', 'reviewer_additional_physics_calls': 0}, 'paper_claim_audit': 'unavailable', 'writing_verdict': 'provisional — no paper claim audit available', 'data_blocked_by_missing_paper_audit': False, 'pending_external_review': False, 'pending_source_audit': False, 'files': {'response': 'response.md', 'verification_script': 'verify_saved_evidence.py', 'verification_results': 'verification_results.json', 'inputs': 'inputs.json', 'handoff': 'handoff.md', 'findings': 'findings.md'}}。下一路由：保留局部G，TRAIN优化尺度诊断，不启动扩库或TEST。

## 2026-10-19 delivery_v1 工程交付

新独立审查C1 yes/high、C2 no/high，工程PASS，无阻塞。M1四方法各80/80和16/16；M2新增128/128、16/16执行分级，TRAIN13966/VAL48；M3全部TRAIN参与6000更新，四方法各180/180和36/36，原始审计PASS。新旧Diffusion同后四步引导；完整成功仍依赖物理修正，优势NOT_ESTABLISHED。新13966模型按工程资格选用，不因声称胜出。确认有界C1，收窄C2，无新增补充实验；下一批256仅协议，TEST未冻结且不运行。旧D=.914089及DEV BLOCKED保留。审查trace20261019_delivery_run01；论文主张审计缺失，下游写作provisional，不阻塞工程交付。


## 2026-10-19 qualified_expand256_001 + expand256_train_dev_001 closeout

Fresh independent review: C1 yes/high (bounded engineering closure), C2 no/high (no stable Diffusion superiority). Expansion 256/256 actual attempts, 32/32 native executions/replays; 224 MODEL_VALID + 32 EXECUTION_VERIFIED; existing 64 TRAIN goals densified, no new global goal/origin coverage. TRAIN14222 all consumed in6000 updates/384000 draws; all four methods180/180 candidates and36/36 executions on originalDEV4+used8,3 sampling seeds. New/old executed mean position0.737807/0.710278mm (13 better,23 worse), mean orientation0.074039/0.075929deg; mixed tradeoff, no overall improvement claim. Data augmentation and extra updates confounded. Confirm bounded engineering delivery, narrow scientific claim, retain old default recommendation and new qualified candidate, no further experiments/expansion/TEST. Fresh trace paper/review-traces/experiment-result-to-claim/20261019_expand256_train_run01; full report ff_prior_expand256_train/docs/ANALYSIS.md. Paper claim audit unavailable; downstream writing provisional only. No repeated physics/training/audit, one no-work resume per run verified.


## 2026-10-20 endpoint_precision_bounded_001

M1 completed: 10/24 accepted; numerical gradient 21/24; 5248 coupled calls. M2/M3 BLOCKED_M1, no new candidate or execution. C1/C2 not supported, not disproven. Main issues: max branch switching (3 cases), event/control update conflict (11 cases), finite AdamW step overshoot. Preserve gates and all results; no added experiment. Full report: ff_prior_endpoint_precision/docs/ANALYSIS.md. Fresh independent trace: paper/review-traces/experiment-result-to-claim/20261020_endpoint_precision_run01/. Route: pivot proposal only, no automatic execution.

Fresh independent reviewer verdict: C1=no, C2=no, confidence high; downstream paper claims provisional (no paper claim audit). No observed parameter increment bug. Missing: base-only true endpoint counterfactual and real nonempty AdamW-state restoration. No further experiment authorized in this round.


## 2026-10-20 smooth_endpoint_bounded_001 收尾

C1=no、C2=no，confidence high；有限M1描述支持。22/24临时接受，46/48数值检查；M2/M3未运行。base同尺度21/22可接受，不能隔离世界项收益。新独立review见paper/review-traces/experiment-result-to-claim/20261020_smooth_endpoint_run01/。路由：记录负结果并暂停；未来仅考虑新冻结版本的残差Jacobian链式数值验证，不自动追加、不放宽物理门。无论文主张审计，论文层面provisional。


## 2026-10-20 endpoint_jacobian_bounded_001 独立结果到主张

意图：C1稳定世界终点精度改善、C2Diffusion优于强基线。全新独立审查C1=partial、C2=no，confidence=high；总体claim partial，仅有限工程与局部作用得到支持。M1 24/24接受48/48梯度；真实6000更新、120/120训练梯度、31/60世界事件接受，18个同缩步base不能接受而combined接受。A/B无引导均8/96，均值微降但候选/部分姿态退化；完整四方法180/180与36/36，无成功率优势。数值问题不再阻碍工程闭环。稀疏监督/采样分布对齐是待验证假说，不声称唯一原因；保留AdamW历史矩的延续影响。路由narrow claim + bounded stop，不自动实验/扩库/TEST。未决科学证据：实际意义、姿态非劣与独立训练复现；不得看到结果后只挑均值/P95放行。新review见paper/review-traces/experiment-result-to-claim/20261020_jacobian_run01/。paper claim audit缺失，下游论文provisional。


# 本轮结果与主张

backbone_pair_smoke_002完成，冻结源码4ccba221，耗时2741.157秒。
两个主干各6000正式更新、14222TRAIN全部参与、300随机序列摘要相同。
无引导各0/96，Transformer位置均值19.0167→21.0883mm、尾部恶化；C1=no。
原冻结DEV两组各180/180与36/36；Transformer相对新MLP执行位置均值
0.7872→0.6593mm，下降16.24%；成功率持平，条件均值平均精度更优；C2=partial。
新增360候选/72执行；旧简单基线360/72仅复用。工程PASS。
单训练种子、共同物理引导、已用开发任务，不证明普遍优势。
完整replay第二轨迹未保存；审查核残差/源码/native_trace，未追加物理仿真。
一次完成resume返回NO_WORK_ALREADY_COMPLETE，956文件清单/hash不变。
保留旧默认MLP与Transformer实验候选；无扩库或TEST。长期科学目标未完成。
新review见paper/review-traces/experiment-result-to-claim/20261020_backbone_run01。
paper claim audit不可用，下游论文写作为provisional。


## sample_alignment_001：完整采样对齐与静态应用（2026-10-20）

两组同14222权重及普通序列各1200更新；aligned增加完整16步教师与60真实世界端点事件。
无引导9/96→24/96，位置均值16.635→13.769mm，P95/最坏改善；C1仅有界比较yes。
原DEV两模型各180/180和36/36，执行位置均值0.7125→0.7177mm；C2=no，总体partial。
三障碍两Diffusion5/5及选定执行通过，五障碍全方法0/5；aligned穿透39.46–52.37mm，
20条局部求解记录16条goal-only回退。路径不可达尚未证实，覆盖缺口只是待检验假设。
410新候选374 MODEL_VALID、373同时命中目标；75次新执行重放、7无候选单元；
旧360/72仅复用，不升级未执行样本。保持所有原物理门，无扩库/TEST。
新独立审查核173源码/29027输入、64保存AdamW、60世界VJP、原始候选执行；
精确BLAS环境2解决1.49e-12重建微差，无容差更改。重放第二完整轨迹未保存。
一次完成resume，1403核心文件清单/hash不变。建议下一项是固定五障碍可行路径与
目标保持碰撞引导诊断，不盲目扩库/换主干/反复加端点训练。当前TEST门未满足。
独立review: paper/review-traces/experiment-result-to-claim/20261020_sample_alignment_run01。
paper claim audit不可用，论文写作provisional；本轮完成不等于长期目标完成。

2026-10-20：冻结五障碍可行路径诊断12起点，全323维教师与102维表示归因，原目标/物理门不变；不训练扩库TEST。软件7/7及初始化核对通过，物理结果待运行。

2026-10-20 capture_feasibility_001收尾：12固定五障碍起点0全门合格，7终点通过，0实际执行；259.343s/8646耦合。粗128网格+0.203mm路径被加密验收检出与卫星-4.8404mm穿透；2048几何步长14.7503mm。6次已知零步可行却返回线性位姿不满足；低维系数尺度混杂；不能归因102维或证明不可达。新独立review C1=no/C2=no，总体no，confidence high仅限证据边界；无paper claim audit，下游写作provisional。路由revise/supplement待另行冻结授权，不本轮追加实验，保持原物理门/旧结果/无训练扩库TEST。

2026-10-21：按用户授权冻结00/07旧终态一个修正版，修复可行初始化、实际控制信赖域、精细碰撞反馈；两起点，不训练扩库TEST。

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


## 2026-10-22 preserve323_bounded_001：独立结果裁决

partial/high；C1实现保持与回退支持，但无有效适配；C2因M1阻断未测试。32梯度、64临时更新，三个比例Q均值仅降0.4270/0.0912/0.0277%，均8/32合格且0新增，最终权重原样恢复。独立检查PASS不能冒充新候选泛化；复杂层无非空原合格保护。归一化细节J相对低102敏感度很弱，codec相关，不作普适不可控证明。停止局部扫参，依新用户授权转静态避障—原生执行—动态追踪整链；不要求科学优势先成立。原门不变，不扩库/TEST。详见ff_prior_preserve323/docs/ANALYSIS.md及本轮新review-traces。


### 2026-10-22 static_fullchain_001 部署中断
C1/C2本轮不支持，性能和物理资格未完成。260字符临时路径、父目录存在、253字符NPZ临时路径已写且LongPathsEnabled=0，高度支持Windows长路径故障。禁止同身份resume未知工作；58文件保留。独立v2仅改短ID并在总预算内保守扣失败预留，最长221字符实写检查通过。不得把11/12软件测试当成物理结果。独立审查见paper/review-traces/experiment-result-to-claim/20261022_static_fs_failure_run01；路线为修复部署后继续工程，非据中间性能重训或调门。


## 2026-10-22 static_fullchain_002 Result-to-Claim
C1 partial/high：三障碍公开输入到原生67执行/重放6/6完成，五障碍0/6单元且未执行，双场景门false。C2 partial/medium仅三障碍3配对种子执行位置均值2.577016→0.078901mm；合格率与执行率优势no。60提案、24选中、23验证、1预算未决、36仅初筛、12合格、6真执行；失败分母保留。原始范围核对4211+39通过；不能把226次SLSQP非success的可行返回写成内层收敛。五障碍11条终点失败（8还碰撞），3条重复结构路径非穿透但rigid终点差33.938mm。缺口是公开初始化/联合局部修正未解决全机绕行与终点精度，不证明微重力错误或任务无解。review trace：paper/review-traces/experiment-result-to-claim/20261022_static_fullchain_v2_run01。路由：缩窄主张、保留旧失败与预算，继续独立P2a受约束动态参考；不再无限局部扫参、不扩库、不TEST。PAPER_CLAIM_AUDIT unavailable，写作provisional。

2026-10-22 P2a新独立审查：4参考合格、3执行合格，实际t00加速度1.001152542倍门；完整重放差0。C1 partial/high，C2未评估，P2b blocked。唯一余量修复固定rho=.9695632054405793，需新物理证据；旧结果不追认。

2026-10-22 dynamic_margin_001：新独立review局部P2a yes/PASS、C1整体partial/high、C2 not_evaluated。1新执行+3精确复用，累计真实4/5；完整重放差0，加速度限值比.970613。路由原P2b有界开发交接，物理门不变、不再扫描rho，五障碍和学习仍缺。

### dynamic_handoff_001：partial/high
P2b局部门PASS，6/8独特执行、10/12逻辑；C2不支持，终点先验漂移尾部恶化。缺时序/障碍条件、P1五障碍及P4连续组合。转向固定既有数据资格和时序学习，不追加终点种子试错；review 20261022_dynamic_handoff_run01。


### 2026-10-22 temporal_data_001 — independent result-to-claim

Overall partial/high; local P3a qualification yes/PASS/high; full C1 partial; C2 no/not evaluated. Fixed 8/8 TRAIN and 4/4 VAL qualify: 4 exact historical reuses plus 8 new native/replay executions. The remaining 28 sources are not upgraded. Fresh saved-data/URDF algebra audit: 1913 checks, 0 failures; 170 current/ZIP sources, 214 input bindings, 146 preserved core files. No new reviewer physics, geometry queries, optimizer/training updates, or resume calls. Proceed to the already planned single temporal Diffusion candidate and deterministic sequence baseline within the original P3 budget; this result proves label qualification, not learning or scientific advantage. Paper claim audit unavailable; writing provisional. Full trace: paper/review-traces/experiment-result-to-claim/20261022_temporal_data_run01/.

# Compact research finding

2026-10-22, static_goal_seed_001: independent result-to-claim verdict **no**, confidence **high**, provisional because PAPER_CLAIM_AUDIT unavailable. New public nonlinear full-path goal seeding reaches Q≤0.5 for30/30 saved seeds; all seeds still penetrate. Six preselected repaired paths pass original target/convergence/refinement gates but fail geometry, with clearance −16.700 to −35.421 mm, all worst pairs v7_capture_3/fore_arm_link.24 unselected paths remain screen-only.0 MODEL_VALID,0 native executions,0 verified replays. Positive Diffusion contribution is not established.

Worker remains FAILED due postprocessing JSON-list/ndarray TypeError after six DONE units; all saved accounting reconstructs exactly (29202 coupled,18298744 intervals,0 native). All176 frozen source hashes and86 inputs match. Git head alone is insufficient for reproduction because frozen archive contains a user modification in hierarchical_qp.py; use full archive+manifest. No frozen files were changed.

Route: proceed only to already-authorized independent P4 integration/development, carrying five-obstacle gap; overall engineering goal incomplete. No further local goal/collision retuning, expansion, new training or TEST. Missing C1 requires an actually qualified path then complete native67+INTEGRATION replay evidence; no automatic new experiment is authorized by this gap list. Executor may append this compact finding to root findings.md without altering its existing content.


# Research Findings

- fullchain_acceptance_001：overall partial，高置信；完整C1 no、局部组件覆盖支持；C2稳定正向收益no，高置信。完整工程未通过。
- 2119保存证据检查0不一致。6新三障碍全部通过；836维保存独立重放差0；82500原生+82500重放=165000。
- 12动态逻辑来自8独特历史执行，4共享。全30机会中18合格证据、12阻塞；每配置9/15；覆盖3/5场景。
- 五障碍合格公开路径缺失，6连续组合任务尚未冻结。既有PhaseSession软件检查不提供物理连续组合证明。
- 125源码/589输入匹配。Git producer包含1项非换行旧用户依赖差异；复现用archive+manifest。
- 当前固定run收尾，不自动新增物理/训练/TEST；不缩小长期任务范围。缺PAPER_CLAIM_AUDIT，论文结论provisional。


P4最终裁决澄清：overall partial/high；完整C1 no/high，component coverage partial；C2 no/high。见20261022_fullchain_acceptance_run01/verdict.json最终版。

## global_bypass_001：新独立result-to-claim

C1原五障碍完整工程、C2 Diffusion相对收益均no/high；限定M0与保存状态延续支持。30候选0路线，0新执行；213连接边拒绝204碰撞、9采样步长失败。局部连接消耗全部剩余时间且无未完成进展复用、关节距离与世界目标度量不一致是待验证机制假说，不能单因归因、证明任务无解或追认323/控制失败。

路由pivot：本轮关闭；若新授权再冻短时可继续世界目标引导机制。禁止追加旧搜索/预算/训练/扩库/TEST，原物理门不变。完整trace paper/review-traces/experiment-result-to-claim/20261022_global_bypass_run01；论文使用provisional（无PAPER_CLAIM_AUDIT），独立review已完成。


## 2026-10-22 diffusion_demo_001
单次三障碍静态C1 yes/high（新独立review）；2/5合格、1/1实际执行和完整836重放通过。C2未评判，不把共同修正贡献归网络。原始源包括已绑定既有用户修改，不能称纯净Git树。后续从真末态连续接原七航点，五障碍与移动卫星缺口保留；无优势追加。


## 2026-10-22 dual_obstacle_chain_001 — confirmed bounded engineering

Fresh independent result-to-claim: yes/high, 47 checks plus neural-proposal/budget reconstruction, no blocker. Nine-obstacle static-to-seven-waypoint chain executed/replayed, 54500 new native steps total, full836 replay/handoff error0. Predefined C2 detour plus feedback; 1350 collision-filter deltas0, not network autonomous obstacle avoidance. No advantage comparison. Resume stdout capture PARTIAL, no repeated call. Long-term five-obstacle/moving-satellite gaps remain; route to already-authorized D3 implementation and freezing. Report ff_prior_dual_obstacle_chain/docs/ANALYSIS.md.


### 2026-10-22 world_steer_001 — experiment-result-to-claim
Intended claim: original five-obstacle public-input full path, original target in 27 s, native67 execution and 836 replay.
Verdict: no/high for complete C1; yes for limited Caux short model-state continuation. 15 complete paths absent, 0 native/replay; 2276 local nodes and M0 fixed comparison are not complete qualification.
Fresh review: 533440 final saved-evidence checks, 0 production inconsistency; provisional paper writing without whole-paper claim audit.
Bottleneck evidence: 12436 penetrating trial edges, 6177 foreground forearm/fourth obstacle; 2372 repeated failed world-parent attempts and 224 duplicate full states. Do not repeat goal-only fixed directions, add candidates blindly, or call unrun conversion/execution a failure.
Routing: first read-only technical decision; at most one remaining-budget collision-informed short-edge/tangent and exact-state cache repair after new implementation and freeze. No training, expansion, advantage benchmark or TEST. Completed resume called once only, 40730 files unchanged.


## Research Findings — contact_steer_001 / 2026-10-22

Intended claim: original five-obstacle complete27s MODEL_VALID path, native67 execution and full836 replay. Fresh experiment-result-to-claim reviewer: C1=no/high; Caux=partial/medium; paper writing provisional (no paper claim audit). Bounded run finished256.500s:0/3 complete,0 execution/replay,3 dependent opportunities BLOCKED;249 new sample-screened model nodes with bestQ reductions61.37%–76.42% within3 selected historical prefixes.6359 coupled calls; no new neural calls/training/data/TEST.

What failed: all bestnodes still exceed world target5mm/1deg; full323/validation/execution were not reached. Three-prefix local progress does not establish component causality or neural advantage. Active-set truncation omitted4900 identities/distances (counts only); no continuous-time collision proof. Independent13919 checks resolve2 BLAS-thread differences;28 supplement checks pass; once-only completedresume2682corefiles/180418308bytes unchanged. No new physics or old review reuse.

Routing=pivot_architecture_and_narrow_claim. Current one-repair scope exhausted; do not tune tree depth/lr/weights/seeds or repeat physics. Read-only architecture decision proposes public unqualified initialguess→full27s constrained restoration→unchanged fullcertification→originalinitialstate nativeexecution/replay; untested and no new experiment launched. Longterm full engineering incomplete. Versioned evidence:ff_prior_contact_steer/docs/ANALYSIS.md;paper/EXPERIMENT_TRACKER_20261022_contact_steer.md;paper/review-traces/experiment-result-to-claim/20261022_contact_steer/verdict.json. No paper prose upgraded to a success claim.


## Research Findings — horizon_restore_001 / 2026-10-23

C1=yes/high and Caux=yes/high from fresh independent reviewer; paper writing provisional without paper claim audit. One original-five APP_DEV candidate MODEL_VALID/executed/replayed. 13751×836 full-state replay exact zero; 2681 coupled,1987216 reserved model intervals,27500 native including replay,468.375s. Sole completed resume PASS,127 core files/458379716bytes unchanged. Source identity is Git base bfc4ea9 plus142 actual hashes/zip and233 inputs; the prior hierarchical_qp working-tree difference is pre-frozen and not invoked.2249 checks=2248 pass+1 explained HEAD difference,0 unresolved.

Bounded static system claim only, not bare network/advantage/generalization/dynamic/physical satellite drift. Missing physical intermediate arrays/source-only boundaries are retained in the fresh review. Routing=confirm static; latest user selects rigid2/continuum5 scene with original7 waypoints and body-shape coverage. New complete-scene run scope pending concrete layout/cost review; no hidden obstacle swap in actual old terminal state. Old static run permanently closed; no re-audit/resume/physics.

## 2026-10-23 balanced_tracking_001 独立收尾

完整组合C1=no/high，19.66秒已执行前缀Cprefix=yes/high，计划全程臂身证据Caux=partial/high；18737保存证据核对一致。停止为theta6加速度下界高于实测内缩域速度阻尼上界，非SLSQP失败或碰撞停止。W6/W7与最终保持未执行，不能用终态距离冒充到达误差。旧run已永久封存，完成resume恰一次PASS。用户新明确要求修正，路由至独立balanced_brake_001，提前离散制动并保持全部原门；不改变旧负面结论。写作provisional，无paper claim audit。详细trace：paper/review-traces/experiment-result-to-claim/20261023_balanced_dynamic。

## 2026-10-23 balanced_brake_001 修复通过并收尾

新独立experiment-result-to-claim审查C1=yes/Caux=yes，单场景保存证据high，94104/94104检查PASS，0未决。原七点7/7与最后1.5秒保持完成，13500执行+13500完整独立重放，13501×836差0。提前离散制动取交原约束，1350实际命令均满足原界与新界，未放宽门。17生产门保存数组复算，geometry_step/no_hidden_reset两门保留源码/报告/重放的明确证据边界；非连续时间证明。写作provisional，无paper claim audit。实际完整媒体54.5秒已交付；浏览器origin授权阻断自动打开，本地端帧/视频/数据核验通过。原失败run不追认，全部旧run封存。路由：当前修复范围完成，长期物理一致实体移动起源仍未完成，无自动追加物理。


## 2026-10-23 moving_satellite_001 — fresh result-to-claim review

`moving_satellite_001`: C1 complete all-gate continuation **no/high**; Caux disclosed physical satellite origin plus recorded tracking/replay/body evidence **yes/high**, explicitly conditional on failed robot momentum; overall partial. The robot momentum gate fails in both groups from new initial state and in all 14,501 saved states. Peak/threshold ratios are 1.06698 and 1.33400. Initial inherited momentum dominates new drift; this continuation's lower exchange magnitude makes the unchanged gate's numerical threshold smaller than the sealed predecessor's.

One 29-second APP_DEV execution and complete saved 836-state replay agree exactly; satellite final Z velocity is 0.002 m/s and displacement is 0.056002 m under the disclosed 2-second COM force followed by 27 seconds engine-off. All other producer gates reproduce as passing. No unresolved arithmetic discrepancy; no physical retry or gate change. Narrow the claim and close the bounded trial as failed C1. No paper claim audit: provisional writing only. See response.md and audit_results.json for exact limits.

Full trace: paper/review-traces/experiment-result-to-claim/20261023_satellite_origin.

## 2026-10-26 guided_diffusion_chain_001 — 独立结果裁决

C1（0–83.5秒冻结Diffusion全机引导连续链及原门动量修复）no/high；Caux仅“新冻结推理与引导准备实际发生”yes/high。13次前向、12个完整保存DDIM步骤、323列Jacobian和四组方向探针；R1球—back_arm_link网格5个时刻的12项导数不一致触发冻结停止。0接受引导/0完成采样/0完整认证/0原生/0重放。完整表面机制尚未证明，不称最近特征切换已确诊。1094保存算术/身份检查一致；独立评审另行复核保存数据，未新增物理。一次完成resume PASS、34文件7058866bytes不变。第13步完整数组、投影求解报告、稀疏稠密历史和最终内存权重hash审计的异常落盘缺口明确保留。原权重文件不变。

路由：收尾本唯一零重试单元，下一步只形成有界距离导数/异常归档修复方案；不自动新候选、改门或追加物理。RK4修复仍未实际验证；长期目标未完成。无paper claim audit，写作provisional。报告ff_prior_guided_chain/docs/ANALYSIS.md；trace paper/review-traces/experiment-result-to-claim/20261026_guided_chain。
