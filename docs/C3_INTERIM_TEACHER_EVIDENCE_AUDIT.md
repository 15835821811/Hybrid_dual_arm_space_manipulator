# C.3 interim teacher evidence audit

2026-10-08. Frozen producer: `9f39b42775283432eb933f63a9047c488ba22070`. Run: `v6_4/output/search_aware_warmstart_20261008_01`.

**No material discrepancy found in the two reviewed completed streams.** This audit covers only `teacher_search/c1_mother_00_plus/T_local` and `T_transfer`. It is not the final teacher/dataset/result audit. Existing scientific evidence was read only; the audit added zero physics, preview integrations, training updates, DDIM samples, or tests. Only these documentation files were created.

| Check | T_local | T_transfer |
|---|---|---|
| Consumed evaluation slots | 8 | 8 |
| Parameter constructions / exact cache aliases | 9 / 1 | 8 / 0 |
| Slot 1 A association | Exact zero common-rule alias C00 | Direct seed C01 |
| A final selection | Common-rule C00 | Common-rule C00 |
| A new effect label | No: RULE_ONLY | Yes: directly qualified C01; not selected |
| Slot 3 B association | Direct seed C02, selected | Direct seed C03, not B30-qualified |
| B final selection | C02, d_support=0.03346263570324469 m | NO_PLAN / preference unmet |
| B new effect label | Yes, original C02 seed z | No |
| Full nominal rollouts | 8 x 13500 | 8 x 13500 |
| Main prediction / separate preview steps | 108000 / 108000 | 108000 / 108000 |
| QP solve calls | 10800 | 10800 |
| Native geometry calls | 132907023 | 132907082 |
| Actual / independent torque-replay steps | 0 / 0 | 0 / 0 |
| Shared A/B planning seconds | 643.8832423998974 | 642.5100692999549 |

Verified the frozen raw contents and legality, registry/input/prefix canonical digests, teacher-effect source hashes, candidate result/plan/started/reference-precheck JSON seals, plan identity, parent and selected lineage, qualification flags, and original-seed label z. The optimizer and physical evaluator intentionally use different content-key schemas; each was independently recomputed in its own scope and matched.

For each candidate, the saved prediction count equals its prediction phase returned integrations. Its 1350 preview receipts each record ten returned steps and eleven coverage states, totaling 13500 separate preview steps. Geometry and QP totals equal their recorded phase counts. No teacher actual execution is represented as having occurred. Large raw trace arrays were not reread or replayed.

T_local has two qualified endpoints and T_transfer one, so the declared hierarchy selects T_local for this Task. T_transfer's directly qualified A label is permitted by the protocol even though the common-rule candidate wins final A; it must not be described as improving or causing that outcome. The unified dataset was not yet built at review. The general unsupported-input null-hash edge is untriggered because all four reviewed raw inputs are legal.

The companion JSON records all input paths and SHA-256 values used in this audit.
