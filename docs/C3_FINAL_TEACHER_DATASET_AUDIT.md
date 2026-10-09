# C.3 finalized teacher, TRAIN dataset and paired-training audit

Audit UTC: 2026-10-08T10:00:40.215424+00:00. Frozen producer: `9f39b42775283432eb933f63a9047c488ba22070`.

**NO_DISCREPANCY_IN_REVIEWED_EVIDENCE**. 1182 read-only evidence checks; 0 discrepancies.

This finalizes the teacher, TRAIN dataset and training audit only. Earlier 10-stream interim documents are preserved. Formal VAL is running; VAL checkpoint choice, TEST and overall completion are outside this verdict.

All 12 teacher groups completed 96 shared candidate slots and 0 teacher actual slots. All 24 original raw seeds were legal, unrepaired and unresampled. Pair identities, original seed targets, fixed partners, source references, seed/cache/lineage association, B30, near bands, first-hit censoring and the predeclared hierarchy were independently recomputed. Local teacher ranking, inherited parent lineage and the complete transfer pool identities also matched.

The 24 nominal preference endpoints contain 19 qualified endpoints and 14 near historical references. Statuses: `{"INITIALIZER_EFFECT_EVIDENCED": 12, "NO_QUALIFIED_INITIALIZER_EFFECT_LABEL": 5, "RULE_ONLY": 7}`. RULE_ONLY, NO_PLAN, failed/incomplete predictions and 2 explicit local B rule fallbacks receive no positive effect credit.

The historical pool contains 72 unique physical TRAIN candidates from 3 mothers and 6 Tasks; 24 old C.2 VAL/TEST facts remain excluded. Transfer sources exclude the query mother and retain the original historical pool, family/mask and retrieval declarations.

The D/S/N pool contains 55 unique Task/family/preference/z labels: 52 with route provenance, 8 with effect provenance and 5 with both. The 10 selected-pair effect rows deduplicate into those effect-bearing labels; all source references remain. Zero labels: 3. No parameter averaging or extra physical sample credit was found.

Both 896-dimensional model condition normalizers and the residual normalizer match the dataset exactly. Condition statistics were independently refitted from 17 unique TRAIN declarations; residual statistics came from 48 unique Task/family/z parameters and 96 active intervals. Model schema and every supported-condition sample/mother ID group bind to the same TRAIN pool.

| Model | Updates | Paired exposures | Parameters | Seed | Training seconds |
|---|---:|---:|---:|---:|---:|
| D | 4000 | 128000 | 152332 | 64321 | 9.794225 |
| S | 4000 | 128000 | 134412 | 64331 | 8.694570 |

Each model records one fresh 4000-update AdamW run, batch 32, lr 1e-4, weight_decay 0.01 and clip 1. The shared 4000 × 32 array matches both exposure maps and hashes. For all four update 250/4000 checkpoints, metadata, normalizers, schema, paired-prefix exposures, parameter counts, finite tensor bytes, file hashes and state hashes were independently verified without importing torch or repository core. The 250/4000 states differ for each model; training performs no model selection, DDIM or physics.

## Scope limits

- No core imports, tests, physics, DDIM, training, rendering or export.
- No large physical NPZ/archive reads or hashes; historical saved seals are metadata evidence, not renewed byte-level physical verification.
- Teacher effects are nominal finite shared-search evidence, not final actual validation or independent causal necessity.
- Small checkpoint tensor bytes and paired draw arrays were read without torch/core imports; optimizer execution was not replayed.
- One seed per model and 3 TRAIN mothers remain DATA_LIMITED; VAL/TEST completion and learning benefit are pending.

The companion JSON records UTC, argv, exit, audit script/source identity, input SHA inventory, per-stream findings, pair hierarchy, checkpoint identities and phase receipts. The exact successful metadata script is retained at `docs/audit_scripts/c3_teacher_dataset_metadata_audit.py`, alongside its original temporary argv and SHA identity. Audit tooling notes separately preserve an overlength dispatch rejection and a corrected receipt-path lookup error; neither changed an experiment record. Default C.1, DATA_LIMITED and deployment=NOT_MET remain in force.
