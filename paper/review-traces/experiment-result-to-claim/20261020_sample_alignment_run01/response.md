# Fresh independent result-to-claim review

Run: `artifacts/v7_1/sample_alignment_001`. Producer source: `b242dbebbfc9e71880ec7c70f2a078ba45778ef6`. Date: 2026-10-20. Role: paper architect/reviewer under experiment-result-to-claim.

**Overall: partial. C1: yes within the explicitly bounded single-training-seed comparison. C2: no as a claim that this training change improves complete obstacle-avoidance planning. Bounded engineering completion passes; two-scene application readiness and TEST readiness fail.** Confidence is high in the saved-evidence accounting and observed comparisons, medium in attributing a repeatable benefit beyond this run. No independent training replicates establish general superiority.

This review read the protocol/addendum, implementation, frozen identities, training records, numerical evidence, original candidate/execution records, and explicitly reused control records. Compact collection files were navigation aids. It ran saved-array/neural/autograd/algebra/hash checks only: no optimizer step, coupled propagation, torque simulation, physical replay, training, experiment, TEST, or resume was initiated. Only this response and its parsed verdict are written by the reviewer.

## Engineering evidence and reproducibility

- Direct SHA256 checks matched all **173** current source files and the corresponding `executed_source.zip` entries, and all **29,027** bound inputs in `run.json.identity.binding.files`. Frozen DEV entries match the reused `expand256_train_dev_001/frozen_tasks.json` exactly. The bound control candidate/execution records were used for structured/learned_mean comparisons; their 360 candidates/72 execution units are reused evidence, not new work.
- `learning/{plain,aligned}/initial.pt`, `training.json`, `progress.pt`, `model.pt`, and `training_input.json` support identical warm-start weights from the exact 14,222-row model: initial weight hash `29e769465d0ecc25ea811cbc694d8e2df30cc7377103c41b3840059212506ac9`. Both optimizers terminate at step 1200. Each group draws 76,800 TRAIN rows, uses all 14,222, and has 2–28 draws per row. I independently reconstructed all 1200 ordinary batch/t/noise draws and their 60 segment digests; both models' recorded segments and row counts match. Saved update indices are TRAIN; implementation uses no VAL gradients or VAL checkpoint selection.
- Source `ff_prior_sample_alignment/learning.py` and `objectives.py` differentiates through all 16 DDIM steps with deployment float types and epsilon recomputation. Aligned adds 1200 batches of eight full-rollout teacher samples and 60 current-model world events. `prepared_fixed` reconstructs smoothing, clipping, projection, and the selected shrink branch; clipping has its actual derivative, and the discrete branch choice is held fixed. Thus this is an active-branch gradient, not a globally differentiable projection claim. Teacher and world contributions are separately capped at half the ordinary gradient norm before overall clipping and one AdamW update.
- The saved-only audit was actually rerun after inspecting its implementation. With the original worker runtime (`OMP_NUM_THREADS=MKL_NUM_THREADS=OPENBLAS_NUM_THREADS=2` set before numerical imports, `torch.set_num_threads(2)`), it returns **PASS: 64 saved update reconstructions, 60 world VJP reconstructions, 360 new DEV candidates, 72 DEV execution units, 50 application candidates, 10 application units, 75 verified native executions, 838,500 recorded native-plus-replay steps**. `reconstruct_adamw` uses explicit moment, bias-correction, decay, and clipping algebra; it does not call an optimizer.
- The reviewer additionally rebuilt both diagnostic VJPs, and independently recomputed 256/512 convergence differences for all 62 diagnostic/training gradient records from saved central/fine poses. Across these records, maxima are approximately **0.00002143 mm** position and **0.0000007321 degrees** rotation, half-step relative error **0.005260**, and directional relative error **0.002405**. The audit rebuilds all 102-dimensional gradients, four-group Jacobians and analytic aggregation weights from saved perturbation poses, then neural VJPs and saved updates. Numerical correctness does not imply task success.
- A reproducibility qualification must be retained: the first audit attempt with BLAS environment variables unset failed the existing `1e-12` prepared-normalized-output equality check at aligned update 0020. A direct diagnostic measured **1.4903633882568101e-12** normalized error and **1.6653345369377348e-15 rad** prepared-control error. Matching the worker's pre-import environment gives exactly **0.0** at that event and the complete audit passes. No source or tolerance was changed. This is a resolved runtime-configuration sensitivity, not an unresolved training failure; future audit commands should reproduce the launch environment.
- `ledger.json` has 806 completed units, no inflight unit and no unresolved-call accounting. Actual coupled calls total **45,270**, below reserved upper bound **60,050** and cap **75,000**. Completed elapsed time is **3482.656 s**, below six hours. New DEV maxima are plain 83 queries/4.803 s/11,136 intervals and aligned 107 queries/6.036 s/14,208 intervals; application candidate maxima are 109 queries/7.214 s. These remain within 113 queries, 15 s and the applicable 15,872-interval cap. No candidate is omitted from denominators.
- `selection.json` is checked against the eligible minimum `(score, candidate_id)` and the candidate-pool digest before using execution results. Native traces bind controls, task, model, initial state and reference; raw `qpos` width is 81 and `ctrl` width is 67. The audit recomputes hold errors from saved end-effector arrays. The physical contract retains nonpenetration, 5 mm/1 degree, 2 mm geometry resolution, full free-floating state and the 17→67 mapping.
- Replay evidence has a defined limit. `ff_prior/physical_audit.py::replay` resets a separate state and advances stored controls without copying actual states each step. Reports record all replay discrepancies at zero and successful replay gates. However, only the native trajectory and replay summary are stored, not a complete second replay trajectory. This review verifies source, saved native arrays and report consistency; it does not independently re-create a second physical trajectory. Neither hardware validation nor continuous-time collision certification is established.
- Existing `resume_command.json` records one completed `NO_WORK_ALREADY_COMPLETE` return. The before/after core manifests contain identical 1403 entries. This review inspected that evidence but did not invoke resume again.

## C1: bounded unguided improvement

The source for these independently recomputed numbers is every record in `unguided/{0,1}/{plain,aligned}/*.json`, using the maximum end-effector error across both arms after preparation. All 192 records exist; no endpoint is missing. Q is `max(position_mm/5, rotation_deg)`.

| Metric | Plain | Aligned |
|---|---:|---:|
| All-gate eligible / 96 | 9 | 24 |
| Seed 0 eligible / 48 | 3 | 13 |
| Seed 1 eligible / 48 | 6 | 11 |
| Position mean / median, mm | 16.634619 / 9.086610 | 13.768546 / 6.131922 |
| Position P95 / maximum, mm | 52.675607 / 85.079154 | 46.225167 / 74.330652 |
| Rotation mean / P95 / maximum, degrees | 2.544930 / 7.430000 / 8.913067 | 2.303731 / 6.864510 / 7.901113 |
| Q mean / P95 / maximum | 3.644355 / 10.535121 / 17.015831 | 3.099025 / 9.245033 / 14.866130 |

Aligned improves position in 84/96 pairs and worsens 12; rotation improves in 79 and worsens 17. There are 15 gained eligible outcomes and no lost eligible outcomes. Per seed, position improves in 43/48 and 41/48 pairs; mean paired deltas are −3.183739 and −2.548406 mm. Both seed groups improve position P95 and maximum as well.

These observations support **C1 = yes only for the frozen comparison**: this combined training package improves the observed unguided hit rate and empirical tails relative to the matched plain continuation. They do not establish generally controlled tails, stable autonomous planning, or sampling success near saturation: aligned still fails 72/96 all-gate outcomes and its worst position error is 74.33 mm. The 96 outcomes comprise two sampling seeds on 48 existing VAL tasks, not 96 independent tasks or training replicates. No inference of statistical independence or population-level significance is made. Teacher-rollout and world-gradient additions change together; neither component's isolated causal contribution is identified.

Suggested claim: “For one matched training seed and the frozen 48-task, two-sampling-seed validation set, combined full-DDIM teacher and world-endpoint alignment raised unguided eligibility from 9/96 to 24/96 and reduced observed position/rotation tails.”

## C2: complete planning benefit is not established

Raw sources are `candidates/{seed}/{sample}/{method}/*.json`, matching `executions/.../{selection,outcome,report}.json`, and the identity-bound structured/learned_mean directories under the reused control run.

| Method | Eligible / 180 | Execution verified / 36 | Final position mean, mm | Final position P95, mm |
|---|---:|---:|---:|---:|
| Plain, new | 180 | 36 | 0.712523 | 1.072020 |
| Aligned, new | 180 | 36 | 0.717673 | 1.237723 |
| Structured, reused | 180 | 36 | 1.028364 | 3.541589 |
| Conditional mean, reused | 180 | 36 | 0.366634 | 1.414932 |

All methods saturate success counts. Aligned's candidate position mean is slightly lower than plain (1.133788 versus 1.144183 mm), yet only 86/180 candidate pairs improve and 94 worsen. Selected execution position improves in 19/36 pairs and worsens in 17, with a **+0.005150 mm** mean delta and a worse P95/maximum. Execution mean deltas by seed are approximately **+0.017823, −0.008775, +0.006402 mm**. This mixed pattern is not consistent superiority. Conditional mean has a lower execution mean; aligned is worse in 28/36 paired positions against it. All these are already used DEV tasks, mostly without external obstacles, and they cannot replace application evaluation.

The independent application has exactly 50 new candidates and 10 execution units, with only **three actual torque executions**. Seven units return `NO_CANDIDATE`; those are explicit failures, not seven absent data points.

Across all 410 new DEV/application candidates, **374 are MODEL_VALID but only 373 are eligible**. The extra MODEL_VALID record is `application/capture_3_obstacles/aligned_unguided/candidates/01.json`, which fails the frozen-goal pose requirement. Model validation must not be conflated with complete goal-qualified eligibility.

| Application scene | Plain | Aligned | Structured | Conditional mean | Aligned unguided |
|---|---:|---:|---:|---:|---:|
| Three obstacles: eligible / 5 | 5 | 5 | 3 | 0 | 0 |
| Five obstacles: eligible / 5 | 0 | 0 | 0 | 0 | 0 |

In the three-obstacle scene, selected plain/aligned/structured executions pass with final position errors 0.419846/0.222277/0.402613 mm. This establishes a bounded simulated static noncontact pregrasp witness for those three selections. One selected aligned trajectory with smaller error does not establish a training-method advantage.

Failure classification is important:

- In five obstacles, all 20 guided candidates hit the pose thresholds but fail geometry. Aligned's five clearances are approximately **−41.011, −46.368, −39.457, −52.373, −49.687 mm**; the worst obstacle/body pairs are `v7_capture_3` with `fore_arm_link` (candidates 0/1) or `wrist_link` (candidates 2/3/4). Raw sampling solver records show **16/20** aligned goal-only fallbacks. This identifies collision-handling/guidance failure in the observed outputs, not a failure of endpoint precision and not evidence that microgravity or model mismatch caused the result.
- Three-obstacle conditional mean hits pose and has positive sampled clearance **6.907–8.465 mm**, but all five fail `geometry_refinement`: maximum geometry steps **2.115–2.227 mm** exceed the unchanged 2 mm budget. Calling these five observed penetrations would be incorrect. Five-obstacle conditional mean fails both geometry and refinement.
- Three-obstacle structured has two geometry failures; aligned unguided fails pose in all five and geometry in four. Five-obstacle unguided fails pose and geometry in all five. The same-task ablation supports dependence on the full guidance pipeline for this application, not separate causal attribution to collision, actuator, or goal subcomponents.
- Both scene initial-state audits pass, with positive initial clearance and preserved state/mapping. This is not a feasible-path proof. Three obstacles has actual successful path/execution witnesses; five obstacles has no successful witness and no infeasibility proof. Five failures per method in this bounded search do not establish an impossible task.

Thus **C2 = no** for the stated improvement claim. A narrower supported statement is that the new training package preserves complete original-DEV engineering success and is usable in one fixed three-obstacle static pregrasp scene, while the fixed five-obstacle scene remains unsolved under this budget.

## Routing and TEST decision

**TEST_READY = false; application gate blocked.** The new DEV engineering gates pass, and this fresh review supports completion of the bounded experiment. However, aligned must reach at least 4/5 candidates and selected execution/replay success in **both** application scenes. Five obstacles is 0/5 with `NO_CANDIDATE`, so the condition fails regardless of any relative advantage. Do not initiate TEST, expand data, relax physical limits, move obstacles, or reclassify these failures.

The immediate bounded action is to close this run with the narrowed C1 claim and explicit negative C2/application result. Use the saved five-obstacle failed pairs, guidance constraints and goal-only fallback records to formulate a separate, frozen feasibility/guidance diagnostic proposal. Distinguish existence of a legal trajectory from the local planner's ability to discover it before proposing more training or data. No new simulation is requested as part of this review. A later authorized diagnostic should preserve original outcomes and identify whether representation, collision constraints or search initialization prevents a feasible detour; it should not use TEST as development data.

Missing evidence for broader claims includes independent training seeds, genuinely held-out tasks/scenes, separate teacher-only/world-only controls if component causality is claimed, and a legal five-obstacle execution witness under the frozen contract. Dynamic irregular waypoints, moving-target capture, contact/grasp closure, hardware performance, a full upstream sensitivity grid, and full component ablations were not evaluated. The reference adaptation is fixed three/five-obstacle comparison plus one all-guidance-off ablation; the reference's hard joint endpoint boundary is not equivalent to this free-floating world-pose endpoint task.

`paper/PAPER_CLAIM_AUDIT.json` is absent. Downstream writing remains **provisional — no paper claim audit available**; this does not block the bounded engineering completion finding. No pending external reviewer flag is needed for this delegated fresh reviewer, but manuscript-level claim/number/citation audits remain separate.
