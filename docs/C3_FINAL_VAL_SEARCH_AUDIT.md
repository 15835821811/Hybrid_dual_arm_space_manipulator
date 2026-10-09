# C.3 final VAL search metadata audit

Verdict: **NO_DISCREPANCY_IN_REVIEWED_VAL_SEARCH_EVIDENCE**.

Audited at 2026-10-08T11:35:19.732645+00:00 to 2026-10-08T11:35:20.295771+00:00 (UTC): 1604 checks, 0 discrepancies. Scope is the completed search stage of `search_aware_warmstart_20261008_01`. Actual execution, independent gates and checkpoint scoring remain pending; this document supplies no checkpoint or whole-VAL completion verdict.

All 10 requests match the predeclared two new paired VAL Tasks and R12/D250/S250/D4000/S4000 schedule. The first Task ran R12 → D250 → S250 → D4000 → S4000; the second ran D250 → S250 → D4000 → S4000 → R12. Every fresh worker has the exact VAL/Task/stream argv, exit 0, one completion event and no overlap with the preceding request.

The registries contain 88 consumed slots, 89 parameter proposals, 1 exact within-stream cache hit and 88 nominal rollouts. The immutable reservations agree: 24 rule slots and 64 model slots, totaling 88; four D requests reserve and record exactly eight DDIM outputs. Four S requests record eight direct outputs. No new experiment computation was run for this audit.

| Task | Endpoint | Slots / proposals / cache | Nominal statuses | Inner cold seconds | Outer seconds |
|---|---|---:|---|---:|---:|
| c3_val_plus | R12 | 12 / 12 / 0 | PREDICTION_ADMISSIBLE=12 | 967.166 | 967.580 |
| c3_val_plus | D250 | 8 / 8 / 0 | PREDICTION_ADMISSIBLE=8 | 659.747 | 660.324 |
| c3_val_plus | S250 | 8 / 8 / 0 | PREDICTION_ADMISSIBLE=8 | 656.843 | 657.416 |
| c3_val_plus | D4000 | 8 / 8 / 0 | PREDICTION_ADMISSIBLE=8 | 654.147 | 654.745 |
| c3_val_plus | S4000 | 8 / 8 / 0 | PREDICTION_ADMISSIBLE=8 | 657.362 | 657.952 |
| c3_val_minus | D250 | 8 / 9 / 1 | EXECUTION_REFUSED=3; PREDICTION_ADMISSIBLE=5 | 509.202 | 509.787 |
| c3_val_minus | S250 | 8 / 8 / 0 | EXECUTION_REFUSED=6; PREDICTION_TASK_UNMET=2 | 341.525 | 342.107 |
| c3_val_minus | D4000 | 8 / 8 / 0 | EXECUTION_REFUSED=3; PREDICTION_ADMISSIBLE=5 | 499.323 | 499.904 |
| c3_val_minus | S4000 | 8 / 8 / 0 | EXECUTION_REFUSED=2; PREDICTION_ADMISSIBLE=6 | 549.697 | 550.281 |
| c3_val_minus | R12 | 12 / 12 / 0 | EXECUTION_REFUSED=10; PREDICTION_TASK_UNMET=1; PREDICTION_ADMISSIBLE=1 | 451.755 | 452.191 |

Both checkpoint updates use the same per-Task A/v1 and B/v2 noise identifiers, full-noise SHA and masked-noise SHA, with fixed VAL seed 64325. The four distinct per-Task/preference noise identities were compared as saved metadata; this audit did not derive noise or run DDIM. Checkpoint paths and SHA strings agree across the original phase seal, initializer metadata and training reports. Current weight bytes were deliberately not reopened.

All 16 D/S outputs are finite 12-dimensional raw proposals at declared initial positions 1 and 3. Their saved source is diffusion or regression respectively, with unchanged raw numbers entering the common raw legality diagnostics, inactive coordinates zero, original masks, and per-interval norms at most 20 mm. All 16 are legal; raw_repaired and resampled are false. All D/S proposals declare identical TRAIN configuration, condition/residual scalers and condition-schema identities. Common initial positions 0 and 2 are identical across methods within each Task. Position means initializer proposal position: exact cache aliases may shift consumed candidate row IDs.

The 50 current JSON members of the original phase seal match their saved SHA values. All 10 selection copies and sources, registries, proposals and planning-cost files are bound before the phase seal at 2026-10-08T11:25:40.566187+00:00. Every outer completion precedes that seal. The original first actual marker is created afterward (2026-10-08T11:25:40.908920+00:00); the first actual budget reservation also follows it. The marker has no embedded UTC, so its creation time is supporting filesystem metadata, combined with the source guard that refuses sealing after any actual directory and the caller that seals all entries before execution.

All 22 original 4/8/12 prefix files pass self seals, consumed-registry and original-proposal prefix digests, exact candidate IDs and costs, and no-later-slot/no-actual-read flags. The contemporaneous root observation at 2026-10-08T09:58:11.0595547Z saw R8 with eight terminal slots, no completed R12 request and no actual slots. Its R8 SHA remains `287461bda72fa8ab104d444f9ad0ed1ca8e8530fcd288a5cb6f1922bda16de41` after R12 ended at 2026-10-08T10:02:35.705261+00:00. Source control flow writes the prefix before the next candidate is constructed/evaluated.

Nominal status totals: {"RAW_INITIALIZER_REJECTED": 0, "REFERENCE_PRECHECK_REJECTED": 0, "TASK_INITIAL_OR_ANCHOR_PRECHECK_REJECTED": 0, "EXECUTION_REFUSED": 24, "PREDICTION_TASK_UNMET": 3, "TOOL_ERROR": 0, "PREDICTION_ADMISSIBLE": 61}. Selected prediction status totals: {"BUDGET_EXHAUSTED_WITH_INCUMBENT": 14, "PREFERENCE_UNMET_WITHIN_BUDGET": 4, "NO_ADMISSIBLE_PLAN_WITHIN_BUDGET": 2}. Four B endpoints have PREFERENCE_UNMET_WITHIN_BUDGET and no selected plan; this differs from the two endpoints with no admissible plan. Raw rejection is zero; nominal execution refusal, task-unmet, preference-unmet, NO_PLAN and TOOL_ERROR remain distinct. A nominal admissible prediction is not actual acceptance, and a NO_PLAN result does not establish collision or task infeasibility. Short failed prefixes cannot establish a cost benefit.

Recorded work totals are 997,920 main prediction steps, 998,160 private preview steps, and 1,070,364,078 native geometry queries. Inner cold planning sums to 5946.768 seconds; outer request wall time sums to 5952.286 seconds. The inner timer includes initializer setup and search/selection work; the outer receipt separately includes fresh-interpreter startup/import. These are measured cold requests. Warm latency remains null. A/B share each search cost once; no cost or quality ranking is adjudicated here.

The audit uses Python standard-library metadata reads and eight frozen source-file SHA checks, including the original inverse_raw implementation. It imports no experiment core, opens no standalone result.json, weight or NPZ, and runs no tests, inference, physics, training, rendering, encoding or export. Original run evidence and frozen core remain untouched. Embedded preview record arrays are not analyzed or emitted. The matching JSON retains input SHA/size inventory, UTC, check counts, command argv/exit, source identity, per-stream failures/costs, and retained audit-script SHA.

The first metadata audit attempt returned exit 1 because its own assertion did not allow the original B-specific PREFERENCE_UNMET_WITHIN_BUDGET status for absent selected plans. That assertion was corrected; the attempt is retained as audit tooling history in JSON. It is outside the experiment and changes no experiment terminal or cost record.

Pending:

- Finish all 20 logical actual slots and verify strict aliases and all five independent gates.
- Evaluate actual A/B quality, NO_PLAN denominators and the original lexicographic checkpoint scorer.
- Freeze the selected models, shared TRAIN/N data and protocol before independent TEST.
- Keep DATA_LIMITED, default C.1 and deployment NOT_MET; this search audit establishes no learning benefit.
