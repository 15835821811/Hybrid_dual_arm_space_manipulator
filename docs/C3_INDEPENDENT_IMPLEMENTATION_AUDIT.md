# C.3 independent implementation audit

Audit date: 2026-10-08 (Asia/Shanghai). Workspace: `E:/v64c3`.

**Verdict: no confirmed implementation issue invalidates the frozen current study.** This is a read-only source/protocol audit while the first formal teacher stage is running. It does not certify completed training, VAL, TEST, an observed learning benefit, or the eventual result tables. No experiment, physics, DDIM sampling, training, or test command was run for this review. Only these new documentation files were written.

## Identity and scope

- Objective: the attached `goal-objective.md`, including its fixed C.3 protocol and 328 candidate / 60 actual logical-slot limits.
- Base: `1758e13b01735b80c5a512b81cbc1d5e47a07ec9`.
- Frozen algorithm producer and reviewed HEAD: `9f39b42775283432eb933f63a9047c488ba22070`.
- Run: `v6_4/output/search_aware_warmstart_20261008_01`.
- Reviewed all four new core modules, their relevant inherited call paths, frozen plan/splits/source identity/teacher pairs/history, and the mock test source. Tests were inspected, not rerun.
- All **430 frozen source files** and **37 protected artifacts** matched their recorded SHA-256 values at review. Both frozen execution/run configs match the published C.2 config bytes. The earlier identity revisions are explicitly retained as pre-execution revisions; they are not reported as the actual experiment producer.

## Findings

### A1 — P3: conservative seed association edge; untriggered in this run

At `v6_4/search_effect_teacher.py:382-384`, the seed-ID fallback accepts equality between a missing proposal `plan_sha256` and a rejected row's null `plan_sha256` before testing the real content-key match. A legal seed following a raw-rejected/unsupported seed can therefore bind to that rejected row. This can suppress otherwise valid **direct-seed** effect evidence, or record the wrong seed candidate ID when real selected-descendant evidence independently qualifies the label.

The positive-label checks at lines 380, 387-394 still require raw legality, a qualified preference result, and direct qualification or genuine selected lineage. This edge cannot turn an illegal raw proposal or an unqualified B endpoint into positive supervision. The supervised z remains the original pair input (line 408).

**Current-study applicability: none observed.** All **12 frozen pairs / 24 raw inputs are legal**, so none of their initial slots produces the null-plan rejected row needed to trigger this ambiguity. The two local B rule-construction fallbacks (`c1_mother_01_minus`, `c2_new_train_minus`) are legal and explicitly tagged; the `rule_construction` gate prevents them earning new effect labels. A future implementation should guard non-null hash/key equality and add an unsupported-slot case. Do not silently modify the already-frozen producer to repair this untriggered edge.

## Protocol checks

| Area | Evidence and assessment |
|---|---|
| Original backend | `git diff` against the base shows no changes in `v6_lite`, `model_test`, route protocol/evaluator, residual execution, conditional execution, or original diffusion modules. The only inherited-file changes are regression allowlisting in `route_initializers.py:60-61` and source attribution in `continuous_route_optimizer.py:79-81,254`. No numeric/search ordering/safety threshold change was found. |
| TRAIN boundary | `search_effect_teacher.py:155-161,179-189` binds exactly the six original C.2 TRAIN tasks and their registry facts. Imported history contains **72 unique physical facts**: 10 validated execution, 40 predicted complete, 22 failed/incomplete; **24 old VAL/TEST facts are excluded**. There are 52 historical preference-label views, not 52 independent trajectories. |
| Historical availability | Import records 845 portable and 637 local-archive SHA-verified inventory entries, with no missing nominal evidence. This audit independently hashed the current frozen source/protocol artifacts; it did not reread every large historical raw trace. The implementation rejects unbound paths and SHA disagreement (`BoundArchive.path`, lines 57-88). |
| New split | Frozen external manifest has 6/2/4 tasks and 3/1/2 mother groups, distinct new seeds 2027985823, 2028090552, 2028195281, and no outcome redraw. Historical TaskSpec split fields remain unchanged, including four original tasks whose embedded split is `test`; the authoritative external role is TRAIN. |
| Teacher experiment | All 12 pairs freeze before any new search, and transfer excludes the entire query mother (`search_effect_teacher.py:308-330`). Each pair uses one shared eight-slot A/B search, no added actual. Hierarchical comparison preserves matched achieved preference sets and ties (lines 423-440). |
| Label validity | Qualified labels require full 13500-step prediction, online/task guards, complete saved horizon, and B measured clearance >=30 mm (lines 335-349). Effect labels use the evaluated initial z, retain partner/pair/lineage, and are marked nominal, not actual (lines 394-416). Common-rule-only/cache aliases and explicit rule-construction fallbacks do not automatically become effect labels. |
| D/S fairness | One sealed shared dataset/scalers and exactly the same 4000x32 reference draw indices feed both models (`simple_warmstart_regression.py:169-230,316-365,383-400`). Both have two 128-unit SiLU layers; D reuses the original cosine100/internal-v/DDIM20 implementation. S receives the same task condition/mask and direct per-reference masked z-MSE. No reference averaging or D-specific better-label pool is introduced. |
| Raw handling | D/S each produce exactly the two declared inputs, and inverse TRAIN scaling does not clamp/project/resample (`simple_warmstart_regression.py:455-501`). Regression goes through the same raw source/preference/family/inactive-coordinate/disk/reference-precheck logic. Rejections consume search slots in the inherited optimizer. |
| Closed-loop VAL | Only D250/D4000/S250/S4000 are eligible. All ten task/endpoint selections and their registries/proposals/cost/checkpoint bytes seal before any actual (`closed_loop_warmstart_validation.py:89-157`). Scoring reads only that VAL phase and prioritizes full actual/five gates, B30, near quality, raw legality, bounded first hit, measured workload, then update (lines 388-465). Missing reference is N/A; misses retain censoring and explicit 9 encoding. |
| TEST freeze | Model selection, both shared data/scalers, retrieval pool, seeds and protocol seal before TEST starts (`search_aware_warmstart_experiment.py:463-509`). TEST R8 is read from the inherited sealed eight-slot prefix; N/S/D really stop at eight. All TEST selections seal before final actual. |
| State isolation and actual | Every candidate uses the unchanged fresh runner/model/MjData/controller/provider/integrator/QP history (`route_candidate_evaluator.py:203-278`). Final actual calls original feedback execution and original five independent gates. Alias identity requires same task/model/plan/config/source and cold initial history (`closed_loop_warmstart_validation.py:170-187,237-342`); aliases add no unique execution. |
| Budget/recovery/cost | Stage reservations sum to 96+88+144=328 candidates and 20+40=60 actual logical slots; DDIM caps are 8 VAL + 8 TEST + up to 8 smoke + 8 diagnostics. Serial scheduling satisfies the reservation mechanism's intended use; it is not a concurrency-safe general scheduler. Incomplete/tool-failed consumed work is retained and not silently rerun. Setup/cold inference/search and separate outer-process times are recorded; workload is separately counted. |

## Result audit still required

At this review, formal teacher execution is active and there is no completed C.3 training/VAL/TEST evidence to audit. The final result review must verify all stage terminal records, actual/alias counts, selected checkpoint identities, source lineage, failure/NO_PLAN denominators, qualified R12 reference sets, planning-cost accounting, and learning-benefit decisions directly against the sealed outputs. In particular, 8/12 quota reduction alone is not acceleration, and selected lineage is not a counterfactual proof of causal necessity.

The declared limits remain: three TRAIN mothers, one VAL mother, two TEST mothers, one training seed per model; deployment NOT_MET, continuous-time and hardware safety NOT_ESTABLISHED. Negative or inconclusive results remain valid outcomes of a completed finite study.
