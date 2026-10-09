# C.3 required-test coverage audit — static source review

Observed 2026-10-08 17:39 +08:00 in `E:/v64c3`; frozen producer `9f39b42775283432eb933f63a9047c488ba22070`. The full attachment objective was read. Its SHA-256 is `bcd7a46bd90aa28050737d2fe2ef18dcc0f3b71548ae8867774ac23e35c1fdd9`; the reviewed [completion evidence matrix](C3_COMPLETION_EVIDENCE_MATRIX.md) SHA-256 is `466d22c79cbad09182b5e8e9eb749674aad62b15c0e20f9e5970b10785da7f7b`.

**12 of 18 obligations have relevant assertion coverage when current C.3, inherited and helper tests are combined; six have a specific untested clause. None is wholly missing.** This is a source-coverage verdict, not a pass receipt, real-study acceptance, or a finding that the frozen experiment violated its protocol. No test, core import, physics, training, DDIM, render, export or large NPZ read was performed. Only this report and its [matching JSON](C3_REQUIRED_TEST_COVERAGE_AUDIT.json) were added for the initial audit.

The reported earlier 66 integration and 34 helper passes have no retained identity-bound command receipt observed here. Static function counts are consistent with 66 = 45 in the four C.3 modules + 10 initializer tests + 11 inherited diffusion tests; this does not establish their actual prior argv. Helpers contain 12 reporting + 7 media + 15 exporter functions. The completion matrix's `O/P` rows identify anchors, not full assertion coverage.

## Exact test reference key

All names below are pytest node suffixes. The JSON gives the full node ID and source line for every reference.

| Key | File | Class |
|---|---|---|
| TT | [test_search_effect_teacher.py](../v6_4/tests/test_search_effect_teacher.py) | `SearchEffectTeacherTests` |
| TM | [test_search_aware_models.py](../v6_4/tests/test_search_aware_models.py) | `SearchAwareModelTests` |
| TV | [test_closed_loop_warmstart_validation.py](../v6_4/tests/test_closed_loop_warmstart_validation.py) | `ClosedLoopValidationTests` |
| TO | [test_search_aware_orchestrator.py](../v6_4/tests/test_search_aware_orchestrator.py) | module functions |
| TI | [test_route_initializers.py](../v6_4/tests/test_route_initializers.py) | `InitializerTests` |
| TC | [test_continuous_route_optimizer.py](../v6_4/tests/test_continuous_route_optimizer.py) | `OptimizerTests` |
| TP | [test_preference_warmstart_pipeline.py](../v6_4/tests/test_preference_warmstart_pipeline.py) | `PipelineTests` |
| TD | [test_preference_teacher_dataset.py](../v6_4/test_preference_teacher_dataset.py) | `PreferenceDatasetTests` |
| TB | [test_residual_binding.py](../v6_4/tests/test_residual_binding.py) | `ResidualBindingTests` |
| TR | [test_residual_binding.py](../v6_4/tests/test_residual_binding.py) | `ResidualReuseTests` |
| TE | [test_evaluate_planning.py](../v6_4/tests/test_evaluate_planning.py) | `PlanningEvaluationTests` |
| TG | [test_a1_geometry_evidence.py](../v6_4/tests/test_a1_geometry_evidence.py) | `A1GeometryEvidenceTests` |
| TX | [test_execution_aware_reference.py](../v6_4/tests/test_execution_aware_reference.py) | `ExecutionAwareReferenceTests` |
| TL | [test_preference_diffusion_warmstart.py](../v6_4/tests/test_preference_diffusion_warmstart.py) | `PreferenceTrainingTests` |
| TH | [test_report_search_aware_warmstart.py](../v6_4/visualization/test_report_search_aware_warmstart.py) | `ReportAuditTests` |
| TS | [test_residual_execution.py](../v6_4/tests/test_residual_execution.py) | `ResidualExecutionTests` |

`COVERED_COMPOSED` means the current wrapper and inherited original checker are tested separately. `PARTIAL` identifies the exact uncovered clause; it does not assert a defect.

## 13.01 — 物理候选去重、A/B视图与actual alias分离

**COVERED.** Same physical fact collapses while retaining preference IDs and actual bindings; execution/report tests separately count logical, unique and alias actuals.

Exact assertions:

- [TT::test_physical_dedup_merges_preference_views_and_actual_alias](../v6_4/tests/test_search_effect_teacher.py#L72)
- [TV::test_alias_same_task_config_plan_history_only_and_resume](../v6_4/tests/test_closed_loop_warmstart_validation.py#L116)
- [TH::test_report_keeps_denominators_aliases_cost_scope_and_failures](../v6_4/visualization/test_report_search_aware_warmstart.py#L175)

Limit: The dedup fixture is a positive merge case; it does not alone establish all real physical keys.

Real evidence still needed: Reconcile imported physical keys, preference-label views and logical/unique/alias actuals against original seals.

## 13.02 — 新外部分组不改历史TaskSpec，镜像母场景不跨组

**PARTIAL.** C3 validates exact split totals and mother-source leakage; inherited external-split/scaler tests retain embedded TaskSpec split=test and compare historical source bytes.

Exact assertions:

- [TO::test_exact_split_and_mother_isolation](../v6_4/tests/test_search_aware_orchestrator.py#L21)
- [TP::test_external_learning_split_drives_task_selection](../v6_4/tests/test_preference_warmstart_pipeline.py#L29)
- [TD::test_external_SHA_split_and_train_only_discrete_scaler](../v6_4/test_preference_teacher_dataset.py#L79)
- [TL::test_prepare_never_trains_and_preserves_external_split](../v6_4/tests/test_preference_diffusion_warmstart.py#L171)

Limit: No inspected test calls C3 prepare and verifies that its copied historical TaskSpec bytes and original files remain unchanged. The inherited byte assertion exercises the C2 scaler/split path, not C3 preparation.

Real evidence still needed: Compare each frozen historical task's bytes/SHA with C2 originals; reconcile 6/2/4 tasks and 3/1/2 mothers. Proposed check: G01.

## 13.03 — 留一母场景检索确实排除同母场景

**COVERED.** Every frozen transfer proposal excludes its whole query mother from eligible mothers and selected retrieval source; base retrieval test preserves TRAIN/same-mask/stable ties/zero labels.

Exact assertions:

- [TT::test_pairs_prefreeze_and_transfer_excludes_whole_mother](../v6_4/tests/test_search_effect_teacher.py#L144)
- [TI::test_retrieval_uses_train_same_mask_and_stable_ties_zero_allowed](../v6_4/tests/test_route_initializers.py#L195)

Limit: These are fixed-pair and synthetic-library checks, not proof that every later real pool is correctly bound.

Real evidence still needed: Bind every real transfer source to its frozen query mother and TRAIN library.

## 13.04 — 教师搜索效果与候选本身质量分层，rule-only/NO_PLAN不被伪造为神经成功

**COVERED.** Tests separate route/effect source types, forbid common-rule/cache-alias or raw-rejected credit, retain original initializer z and partner, and keep NO_PLAN/right-censor states. The descendant test's B=None and exactly one A label also checks no B effect credit.

Exact assertions:

- [TT::test_rule_only_and_common_cache_alias_are_not_effect_labels](../v6_4/tests/test_search_effect_teacher.py#L90)
- [TT::test_descendant_label_is_original_seed_not_final_z](../v6_4/tests/test_search_effect_teacher.py#L97)
- [TT::test_raw_rejection_cannot_receive_descendant_supervision](../v6_4/tests/test_search_effect_teacher.py#L109)
- [TT::test_no_plan_preserves_right_censor_and_no_B_label](../v6_4/tests/test_search_effect_teacher.py#L115)
- [TT::test_supervision_dedup_retains_route_and_pair_sources](../v6_4/tests/test_search_effect_teacher.py#L130)

Limit: The dedicated NO_PLAN test asserts censor fields, not an explicit empty initializer_effect_labels list; adding that assertion is a useful small strengthening, not a separate unsatisfied software clause.

Real evidence still needed: Bind every final positive effect label to its original legal slot, pair/partner and qualifying seed/lineage; do not infer causality.

## 13.05 — B标签不把仅有A成功或不足30mm的结果视作B已满足

**COVERED.** A can qualify at 29.999mm while B cannot; B=None yields no B effect label; exact30mm/missing/failed-prefix and actual B30 priority are covered.

Exact assertions:

- [TT::test_B_never_uses_A_success_or_below_30mm](../v6_4/tests/test_search_effect_teacher.py#L81)
- [TT::test_descendant_label_is_original_seed_not_final_z](../v6_4/tests/test_search_effect_teacher.py#L97)
- [TC::test_exact_B_threshold_missing_clearance_and_failed_prefix_excluded](../v6_4/tests/test_continuous_route_optimizer.py#L138)
- [TV::test_B30_and_near_quality_precede_raw_and_cost](../v6_4/tests/test_closed_loop_warmstart_validation.py#L221)

Limit: Mock clearances do not establish actual clearance.

Real evidence still needed: Inspect B endpoint qualification, original five gates, measured d_support and B label provenance.

## 13.06 — 两个模型相同条件/schema、TRAIN-only scaler、相同监督范围

**COVERED.** Shared tensor/mask/architecture scope ignores poisoned VAL data; source-balanced deterministic TRAIN indices, 4000x32 paired draws, TRAIN scaler fitting and unequal exposure refusal are tested.

Exact assertions:

- [TM::test_same_conditions_scalers_mask_and_honest_parameter_counts](../v6_4/tests/test_search_aware_models.py#L30)
- [TM::test_source_balancing_and_paired_draws_are_train_only](../v6_4/tests/test_search_aware_models.py#L66)
- [TM::test_prepare_has_no_optimizer_no_sampling_and_seals_paired_references](../v6_4/tests/test_search_aware_models.py#L103)
- [TM::test_pair_wrapper_requires_identical_exposure_reports](../v6_4/tests/test_search_aware_models.py#L169)
- [TT::test_mock_teacher_dataset_and_resume_do_not_need_VAL_labels](../v6_4/tests/test_search_effect_teacher.py#L182)
- [TD::test_external_SHA_split_and_train_only_discrete_scaler](../v6_4/test_preference_teacher_dataset.py#L79)

Limit: Tests prepare/mocks do not prove real training executed or identical actual checkpoint inputs.

Real evidence still needed: Verify dataset/scaler/schema and paired-draw hashes in both real training reports; verify 4000 updates and 128000 exposures/model.

## 13.07 — S source通过原raw检查，不能绕过inactive/范数/锚点/速度

**COVERED_COMPOSED.** Regression source is sent through raw_seed_plan: inactive/nonfinite/disk failures precede precheck and generic failed analytic precheck rejects. Inherited original-reference tests independently exercise speed rejection and protected anchor/cutoff/norm behavior.

Exact assertions:

- [TM::test_regression_same_raw_qualifier_rejects_all_illegalities](../v6_4/tests/test_search_aware_models.py#L155)
- [TM::test_regression_source_lineage_uses_original_shared_search](../v6_4/tests/test_search_aware_models.py#L178)
- [TI::test_analytic_reference_rejection_never_reaches_evaluator](../v6_4/tests/test_route_initializers.py#L168)
- [TX::test_analytic_derivatives_and_unchanged_speed_gate_with_conservative_bounds](../v6_4/tests/test_execution_aware_reference.py#L111)
- [TX::test_protection_and_unchanged_two_dimensional_norm_limit](../v6_4/tests/test_execution_aware_reference.py#L139)

Limit: C3 regression test mocks precheck failed rather than producing a real anchor/speed violation. Coverage is compositional: same S wrapper plus independently tested original checker. It should not be described as a direct regression-speed integration test.

Real evidence still needed: Check real S source tags, original .24m/s precheck settings and raw/reference diagnostics. Static source shows .24 is passed unchanged.

## 13.08 — 12维容器与最多4维搜索mask一致

**COVERED.** Legal predecessor/key mapping has at most two intervals, four coordinates in a 12-D container; inactive exact zero, per-interval disk norms and wrong-size raw rejection are checked.

Exact assertions:

- [TC::test_active_interval_nearest_legal_predecessor_and_disabled_key](../v6_4/tests/test_continuous_route_optimizer.py#L48)
- [TC::test_map_four_coordinates_inactive_exact_zero_separate_disks](../v6_4/tests/test_continuous_route_optimizer.py#L57)
- [TM::test_same_conditions_scalers_mask_and_honest_parameter_counts](../v6_4/tests/test_search_aware_models.py#L30)
- [TI::test_illegal_raw_seeds_consume_slots_without_nominal_calls_or_repair](../v6_4/tests/test_route_initializers.py#L109)

Limit: The concrete fixture has two intervals; inherited active-interval test also covers one/zero applicable intervals.

Real evidence still needed: Check all real frozen/generated masks and raw arrays.

## 13.09 — D采样次数固定，非法raw不修复、不补抽

**PARTIAL.** One C3 sample invokes one mocked DDIM, preserves invalid raw and never repairs/resamples; invalid slots consume budget; C2 wrapper explicitly has two task-bound noise draws; C3 global sample caps are asserted.

Exact assertions:

- [TM::test_sampler_preserves_invalid_raw_never_resamples_or_projects](../v6_4/tests/test_search_aware_models.py#L123)
- [TO::test_new_stage_budgets_and_fixed_ddim](../v6_4/tests/test_search_aware_orchestrator.py#L42)
- [TI::test_illegal_raw_seeds_consume_slots_without_nominal_calls_or_repair](../v6_4/tests/test_route_initializers.py#L109)
- [TI::test_legal_zero_duplicates_cache_without_regeneration](../v6_4/tests/test_route_initializers.py#L145)
- [TL::test_noise_exactly_two_fixed_task_bound_draws](../v6_4/tests/test_preference_diffusion_warmstart.py#L155)

Limit: No inspected test exercises C3 SearchAwareSampler.initializer_proposals plus C3 persisted _initializers/resume to assert exactly slot1/3 once and zero extra generation on resume or raw rejection.

Real evidence still needed: Verify formal D generation ledger (VAL8, TEST8, total<=32) and two prefrozen proposals/stream, including unsupported/illegal cases. Proposed check: G02.

## 13.10 — 两个checkpoint白名单与闭环VAL排序固定

**COVERED.** 250/4000 whitelist, complete-phase gate and capability/B30/near/raw/first-hit/work/earlier-update order have direct positive/negative tests.

Exact assertions:

- [TM::test_fixed_seeds_and_only_two_predeclared_checkpoints](../v6_4/tests/test_search_aware_models.py#L89)
- [TV::test_whitelist_and_complete_phase_before_any_actual](../v6_4/tests/test_closed_loop_warmstart_validation.py#L103)
- [TV::test_B30_and_near_quality_precede_raw_and_cost](../v6_4/tests/test_closed_loop_warmstart_validation.py#L221)
- [TV::test_quality_capability_precedes_cheaper_failure](../v6_4/tests/test_closed_loop_warmstart_validation.py#L194)
- [TV::test_raw_illegal_precedes_first_hit_and_cost](../v6_4/tests/test_closed_loop_warmstart_validation.py#L213)
- [TV::test_first_hit_and_workload_precede_update_tie](../v6_4/tests/test_closed_loop_warmstart_validation.py#L204)
- [TV::test_score_ties_select_earlier_update_no_test_reads](../v6_4/tests/test_closed_loop_warmstart_validation.py#L182)

Limit: Static test constants do not establish which real weights were selected.

Real evidence still needed: Recompute real D/S choices from sealed VAL only, with exact reference sets and cost/first-hit ordering.

## 13.11 — VAL选择不读取TEST文件或字段

**PARTIAL.** Scorer refuses phase=TEST; TEST search requires model freeze. Existing tie test leaves malformed root/test.json and asserts test_read=False.

Exact assertions:

- [TV::test_score_ties_select_earlier_update_no_test_reads](../v6_4/tests/test_closed_loop_warmstart_validation.py#L182)
- [TV::test_test_phase_cannot_select_checkpoint](../v6_4/tests/test_closed_loop_warmstart_validation.py#L263)
- [TO::test_test_search_requires_freeze_before_initialization](../v6_4/tests/test_search_aware_orchestrator.py#L57)

Limit: Malformed root/test.json is outside actual test_search/frozen_test locations. No inspected test enforces file-read scope or poisons TEST-derived fields; a literal test_read=False is not read instrumentation.

Real evidence still needed: Inspect real stage chronology/freeze, scorer inputs and exact source/config identity. Runtime chronology is distinct from the missing adversarial software test. Proposed check: G03.

## 13.12 — R8封存后不受R9—R12影响

**COVERED.** Adversarial late candidates change R12 winner while R8 prefix bytes/selection remain fixed before ninth call; prefix tamper and retrospective cost reconstruction are rejected.

Exact assertions:

- [TI::test_r8_is_r12_prefix_and_sealed_before_ninth_evaluation](../v6_4/tests/test_route_initializers.py#L81)
- [TI::test_resume_detects_sealed_prefix_tampering](../v6_4/tests/test_route_initializers.py#L234)
- [TH::test_prefix_digest_and_request_time_are_not_reconstructed_from_r12](../v6_4/visualization/test_report_search_aware_warmstart.py#L317)

Limit: A mock's sealed prefix is not the real R8 receipt.

Real evidence still needed: Reconcile real R8 seals against first-eight registry bytes and sequence; retain actual/predicted and R8/R12 timing scope.

## 13.13 — 状态/控制器缓存隔离、同一候选内部热启动保留

**PARTIAL.** Distinct candidate provider/config instances and separate stream directories/cache files are tested; exact stream-local cache works.

Exact assertions:

- [TC::test_candidate_calls_create_distinct_provider_and_config_objects](../v6_4/tests/test_continuous_route_optimizer.py#L245)
- [TO::test_stream_directory_isolation](../v6_4/tests/test_search_aware_orchestrator.py#L49)
- [TP::test_streams_share_configuration_bytes_but_no_cache](../v6_4/tests/test_preference_warmstart_pipeline.py#L37)
- [TC::test_exact_cache_hits_no_new_calls_and_proposal_cap](../v6_4/tests/test_continuous_route_optimizer.py#L153)

Limit: The candidate test aborts at runner entry, so it does not create/observe MjData/controller/QP history. No inspected test verifies retained previous_velocity/constraint-dual between two solves within one candidate.

Real evidence still needed: Source path shows fresh runner/MjData/QP and retained in-run velocity/dual history, but final source/config and runtime isolation identities still need verification. Proposed check: G04.

## 13.14 — 工具/管线错误、真实无方案、预演拒绝、actual失败分别记录

**PARTIAL.** Tests distinguish tool error, NO_PLAN, reference precheck, actual failure, positive-step nominal refusal and late quality-tool failure without retry.

Exact assertions:

- [TV::test_tool_error_is_saved_separately_and_never_retried](../v6_4/tests/test_closed_loop_warmstart_validation.py#L143)
- [TV::test_late_quality_tool_failure_retains_consumed_actual_count](../v6_4/tests/test_closed_loop_warmstart_validation.py#L152)
- [TV::test_refusal_precheck_and_actual_failure_are_distinct](../v6_4/tests/test_closed_loop_warmstart_validation.py#L162)
- [TV::test_no_plan_keeps_zero_steps_and_no_runner_call](../v6_4/tests/test_closed_loop_warmstart_validation.py#L136)
- [TC::test_tool_error_is_retained_distinct_from_research_negative](../v6_4/tests/test_continuous_route_optimizer.py#L173)
- [TH::test_stream_nominal_failures_are_counted_and_unbound_registry_is_rejected](../v6_4/visualization/test_report_search_aware_warmstart.py#L298)

Limit: No inspected focused test constructs a private-preview rejection receipt and checks its original failure reason/stage is retained separately from NO_PLAN and actual failure. Generic EXECUTION_REFUSED does not prove preview provenance survived.

Real evidence still needed: Read original structured preview/action-guard failure receipts and reconcile terminal categories, consumed work and no-retry status. Proposed check: G05.

## 13.15 — 消费非有限命令必须失败，未消费拒绝行不污染已有前缀

**COVERED.** Consumed nonfinite/short/wrong-shape commands fail; unconsumed refused NaN tail is excluded from command and QP-input binding; consumed-prefix view preserves source bytes and adds no physics.

Exact assertions:

- [TV::test_original_consumed_command_check_rejects_nan_but_ignores_refused_row](../v6_4/tests/test_closed_loop_warmstart_validation.py#L268)
- [TE::test_consumed_command_nan_short_and_wrong_shape_still_fail](../v6_4/tests/test_evaluate_planning.py#L52)
- [TE::test_unconsumed_refusal_command_tail_does_not_invalidate_consumed_ramp](../v6_4/tests/test_evaluate_planning.py#L41)
- [TB::test_nonfinite_consumed_or_diagnostic_logs_are_rejected](../v6_4/tests/test_residual_binding.py#L136)
- [TB::test_rejected_unconsumed_tail_is_excluded_from_QP_binding](../v6_4/tests/test_residual_binding.py#L177)
- [TS::test_missing_last_boundary_uses_saved_actual_state_and_preserves_source](../v6_4/tests/test_residual_execution.py#L12)

Limit: Small saved-array tests exercise the checker, not final actual gate receipts.

Real evidence still needed: Inspect original independent reference/execution bindings and consumed length for each unique actual.

## 13.16 — 当前安全几何范围不升级，缺测NOT_RUN/null

**COVERED_COMPOSED.** Original geometry tests retain pair policy/config, count queries without adding native queries and explicitly keep continuous_time_certified=False and limited whole-body scope. Reporter checks null geometry gate/missing quality, NOT_RUN/censor semantics and withholding benefit.

Exact assertions:

- [TG::test_geometry_survives_later_consumed_command_binding_failure](../v6_4/tests/test_a1_geometry_evidence.py#L33)
- [TG::test_whole_body_negative_queries_count_once_without_more_native_queries](../v6_4/tests/test_a1_geometry_evidence.py#L59)
- [TH::test_missing_reference_is_na_and_failure_cannot_be_near](../v6_4/visualization/test_report_search_aware_warmstart.py#L159)
- [TH::test_full_method_or_r12_missing_quality_is_na_and_withholds_benefit](../v6_4/visualization/test_report_search_aware_warmstart.py#L235)
- [TH::test_first_miss_right_censored_never_zero_or_slot9](../v6_4/visualization/test_report_search_aware_warmstart.py#L168)
- [TV::test_missing_reference_quality_is_na_and_shared_reference_set](../v6_4/tests/test_closed_loop_warmstart_validation.py#L237)

Limit: C3 missing-quality tests alone do not test geometry scope. Scope coverage comes from inherited geometry contracts plus frozen-source audit; no mock certifies physical geometry or continuous time.

Real evidence still needed: Verify unchanged source/config/pairs, real native2ms related-pair saved-state evidence, five-gate nulls and exact scope language.

## 13.17 — 别名条件、预算恢复与完整日志终态

**PARTIAL.** Same-task/config/plan/history aliases and cross-task nonalias, byte seals, budget idempotence, complete-stage reuse, early stop versus tool-error prefixes and missing receipt incompleteness are directly checked.

Exact assertions:

- [TV::test_alias_same_task_config_plan_history_only_and_resume](../v6_4/tests/test_closed_loop_warmstart_validation.py#L116)
- [TV::test_checkpoint_and_registry_bytes_are_bound](../v6_4/tests/test_closed_loop_warmstart_validation.py#L111)
- [TO::test_budget_resume_does_not_grant_more_work](../v6_4/tests/test_search_aware_orchestrator.py#L29)
- [TC::test_resume_verifies_source_config_and_does_not_evaluate_again](../v6_4/tests/test_continuous_route_optimizer.py#L164)
- [TI::test_normal_early_stop_and_tool_errors_have_distinct_prefix_receipts](../v6_4/tests/test_route_initializers.py#L214)
- [TR::test_valid_reuse_does_not_execute_and_wrong_slot_or_plan_hash_refuses](../v6_4/tests/test_residual_binding.py#L212)
- [TR::test_changed_or_unprotected_execution_config_refuses_before_actual](../v6_4/tests/test_residual_binding.py#L222)
- [TH::test_missing_seal_or_formal_process_receipt_keeps_report_incomplete](../v6_4/visualization/test_report_search_aware_warmstart.py#L278)

Limit: No inspected test exercises C3 unfinished initializer/candidate/actual paths or selection-without-original-timer/failed-outer-worker resume with strict zero rerun and unchanged original journal bytes/costs. Reporter rejection of a missing receipt is not executor no-retry coverage.

Real evidence still needed: Reconcile all actual/stream terminal records, original started/end times, outer receipts, logs and reservation-versus-consumption counts. Proposed check: G06.

## 13.18 — 成本包含非法槽与NO_PLAN而不奖励提前失败

**COVERED.** Illegal slots consume evaluation budget despite zero rollout work; NO_PLAN remains a zero-step logical slot; cheap failed/no-quality methods cannot win and extra raw rejection confounds a cost benefit claim.

Exact assertions:

- [TI::test_illegal_raw_seeds_consume_slots_without_nominal_calls_or_repair](../v6_4/tests/test_route_initializers.py#L109)
- [TV::test_no_plan_keeps_zero_steps_and_no_runner_call](../v6_4/tests/test_closed_loop_warmstart_validation.py#L136)
- [TV::test_quality_capability_precedes_cheaper_failure](../v6_4/tests/test_closed_loop_warmstart_validation.py#L194)
- [TV::test_missing_formal_workload_cannot_win_by_omission](../v6_4/tests/test_closed_loop_warmstart_validation.py#L255)
- [TH::test_report_keeps_denominators_aliases_cost_scope_and_failures](../v6_4/visualization/test_report_search_aware_warmstart.py#L175)
- [TH::test_more_illegal_raw_cannot_win_on_saved_physics](../v6_4/visualization/test_report_search_aware_warmstart.py#L206)
- [TH::test_full_comparator_missing_quality_cannot_hide_capability_loss_or_win](../v6_4/visualization/test_report_search_aware_warmstart.py#L217)

Limit: Missing-formal-workload test tampers after sealing: it directly proves seal rejection, not the pre-seal missing-cost validator branch. No mock validates measured wall-clock coverage.

Real evidence still needed: Verify original request/outer timers include setup, illegal/NO_PLAN requests and sealing; A/B time once; offline, preview, replay, geometry and aliases separately.

## Six focused additions authored after the static audit; none executed

After the initial audit, six checks were authorized and authored in [test_search_aware_additional_contracts.py](../v6_4/tests/test_search_aware_additional_contracts.py). The file has not been imported, compiled or executed; there is no pass claim. Its SHA-256 is `7469570742e50dce0ae0d5631264008ef1bece1275bd7c3ae664842330e868c3`. Static JSON inspection confirms it is absent from the frozen producer's existing 430-file source identity, so it is additional verification and does not relabel or alter the early experiment producer. Existing source/tests remain unchanged. The 12-covered/six-partial verdict describes the original inspected suite before these additions.

The new autouse fixture blocks MuJoCo model/data construction, integration/forward/distance calls, real `sample_ddim`, `AdamW.step` and subprocess dispatch. Temporary NPZ data is only a 10x67 torque array. The QP test exercises real velocity/dual history plumbing through two mocked solves and actual reset; it does **not** prove complete runner construction or per-candidate MjData/controller/reference-integral history. That last part of 13.13 still needs unchanged-source/runtime evidence. The following specifications describe the intended checks; their execution remains pending.

### G01 / 13.02 — `test_c3_prepare_preserves_historical_TaskSpec_bytes_and_external_roles`

C2 scaler immutability does not execute C3's file-copy/preparation path.

- Use temporary copies of six existing small historical TaskSpec JSONs; record source and copied bytes before/after C3 prepare.
- Mock git/source identity and new-scene/task/geometry seams; block mj_step, mj_forward, mj_geomDistance, model construction, training, DDIM and subprocess dispatch.
- Assert external roles are TRAIN without rewriting embedded historical split or source/copy hashes; mirrored source IDs remain in one group; retain a mocked failed precheck without reseeding.

### G02 / 13.09 — `test_c3_two_proposals_raw_rejection_and_retained_initializers_never_regenerate`

Existing tests exercise a single C3 sample and a C2 two-output wrapper, but not C3 paired wrapper plus persistence/resume.

- Mock SearchAwareSampler.sample and exercise real initializer_proposals: exactly A/v1 slot1 and B/v2 slot3; same frozen noise across checkpoint updates; no extra calls for invalid or unsupported raw.
- Exercise _initializers with temporary dataset/checkpoint bytes and mocked sampler/loaders; assert reservation is two samples, persisted illegal raw is unchanged and second call loads retained proposals with zero sampler calls.
- Block the real sample_ddim/AdamW.step/physics/runner entry points throughout.

### G03 / 13.11 — `test_val_scoring_forbids_real_TEST_paths_and_ignores_TEST_fields`

A malformed root/test.json and reported test_read=False do not observe read scope or field influence.

- Build the existing small VAL fixture and actual frozen_test/test_search trees with malformed or poison values.
- Guard every scorer read/hash path against those TEST roots while allowing its explicit sealed VAL/checkpoint/config inputs; assert score and selected updates remain identical.
- Poison extraneous TEST-derived fields in the synthetic VAL input schema, reseal through the fixture helper, and confirm they cannot affect ordering; keep phase=TEST refusal.

### G04 / 13.13 — `test_same_candidate_retains_qp_history_and_next_candidate_starts_cold`

The existing evaluator test stops before runner/controller construction, so it cannot detect per-tick hot-state clearing or cross-candidate history leakage.

- Reuse a tiny fixed-QP fixture or fully mocked solver/validator and clearance matrices; no MuJoCo state, integration, native geometry or actual optimization workload.
- Perform two mocked solves on the same QP instance; assert second initial velocity and initial_dual equal the first accepted selected command and returned dual keyed by original constraints.
- Create a new candidate/controller fixture and assert velocity/dual/provider-integral history starts cold; do not reset between the first candidate's two calls.

### G05 / 13.14 — `test_preview_refusal_keeps_original_stage_reason_and_is_not_NO_PLAN_or_actual_failure`

Generic EXECUTION_REFUSED/report status tests do not prove the source preview refusal receipt survives the candidate pipeline.

- Provide a tiny saved trace plus structured original preview/action-guard refusal JSON to the nominal evaluator with runner, fresh_quality, trace checks and all physics/geometry fully mocked.
- Assert nominal refusal retains failure stage/reason and work already consumed; no actual started; absence of a structured receipt is TOOL_ERROR rather than a fabricated scientific failure.
- Compare those records with existing NO_PLAN zero-step and actual-failure fixtures without introducing a new scientific success category.

### G06 / 13.17 — `test_incomplete_consumed_units_and_missing_original_timers_never_retry`

Budget reservations, completed reuse and reporter missing-receipt detection do not exercise incomplete executor recovery.

- Construct temporary started-only initializer/candidate/actual directories; monkeypatch sampler/evaluator/executor/subprocess.run to fail if invoked; assert the documented technical-incomplete errors retain original bytes.
- Construct selection.json without planning_cost.json and planning_cost.json without outer_process.json; assert no replacement timing or work is manufactured.
- Construct a nonzero outer_process exit receipt and assert resume refuses without subprocess dispatch; successful retained receipts preserve original logical budget, journal/status and costs.

## A final focused suite is safe only outside formal timing

The updated explicit selection below contains **162 test functions by static source count** (122 existing current/inherited + six new unexecuted checks + 34 helpers). It uses mock evaluators, pure reference math or tiny synthetic saved evidence; no real training, DDIM, physical rollout, rendering, video encoding or scientific release export is requested. Torch import, CPU tensor/reference calculations and file hashing still impose load, so **do not execute it while formal sequential VAL/TEST request timing is running**. This is a proposed command, not an executed or collected test count.

Exclude `v6_4/test_preference_teacher_dataset.py` from this receipt run: its class setup calls historical `import_c1`/`PortableEvidence`, which hashes bound original evidence including archived actual traces. Its relevant inherited assertion was read statically; rerunning that setup would violate the lightweight/no-large-archive scope.

Suggested environment: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`, `OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, `NUMBA_NUM_THREADS=1`, `CUDA_VISIBLE_DEVICES=""`. Use the known working CPU interpreter. Exact argv (cwd `E:/v64c3`):

```powershell
& 'E:\v64c2\.venv-c2\Scripts\python.exe' -B -X utf8 -m pytest --import-mode=importlib -p no:cacheprovider -q `
  v6_4/tests/test_search_effect_teacher.py `
  v6_4/tests/test_search_aware_models.py `
  v6_4/tests/test_closed_loop_warmstart_validation.py `
  v6_4/tests/test_search_aware_orchestrator.py `
  v6_4/tests/test_route_initializers.py `
  v6_4/tests/test_preference_diffusion_warmstart.py `
  v6_4/tests/test_continuous_route_optimizer.py `
  v6_4/tests/test_preference_warmstart_pipeline.py `
  v6_4/tests/test_evaluate_planning.py `
  v6_4/tests/test_residual_binding.py `
  v6_4/tests/test_a1_geometry_evidence.py `
  v6_4/tests/test_execution_aware_reference.py `
  v6_4/tests/test_residual_execution.py `
  v6_4/visualization/test_report_search_aware_warmstart.py `
  v6_4/visualization/test_search_aware_warmstart_media.py `
  v6_4/visualization/test_export_search_aware_release.py `
  v6_4/tests/test_search_aware_additional_contracts.py
```

Save the actual command/environment, interpreter identity, UTC start/end, exit code, collected/pass/skip/fail counts and full output in new Markdown/JSON receipts. Bind current HEAD, test-source hashes in the JSON inventory and final implementation/helper identities. Do not alter historical hashes/tolerances or claim this rerun fills the six missing assertions without implementing and checking them. No command above was executed by this audit.

## Coverage versus required runtime proof

A passing mock suite would establish software behaviors only. Final acceptance still needs original terminal teacher streams, real one-time D/S training and checkpoint bytes, complete sealed VAL/checkpoint choice, freeze-before-TEST chronology, all TEST selections/actual logical slots, original five gates, alias/source identity, generation/candidate/physics budgets, request/outer timers and failure journals. Publication and saved-state media require their own final verification. The detailed per-obligation runtime column above deliberately remains separate from software coverage.

