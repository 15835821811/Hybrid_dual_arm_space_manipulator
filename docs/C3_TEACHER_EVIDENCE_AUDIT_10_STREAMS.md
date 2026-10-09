# C.3 teacher evidence audit: 10 completed streams

2026-10-08. Frozen producer: `9f39b42775283432eb933f63a9047c488ba22070`. Run: `E:\v64c3\v6_4\output\search_aware_warmstart_20261008_01`.

**NO_DISCREPANCY_IN_REVIEWED_EVIDENCE**. This is an interim evidence audit of the first ten completed teacher searches (80 consumed slots), captured while the eleventh stream was running. It does not certify final teacher selection, dataset, training, VAL, TEST, or research completion. Only the two new audit documents were written; no physics, tests, DDIM, training, export, or rendering was run.

| Task | Pair | Slots / proposals / aliases | A selected / label | B selected / label | Main / preview steps |
|---|---|---:|---|---|---:|
| c1_mother_00_minus | T_local | 8 / 8 / 0 | C01 / INITIALIZER_EFFECT_EVIDENCED | C03 / INITIALIZER_EFFECT_EVIDENCED | 108000 / 108000 |
| c1_mother_00_minus | T_transfer | 8 / 9 / 1 | C02 / INITIALIZER_EFFECT_EVIDENCED | C07 / RULE_ONLY | 108000 / 108000 |
| c1_mother_00_plus | T_local | 8 / 9 / 1 | C00 / RULE_ONLY | C02 / INITIALIZER_EFFECT_EVIDENCED | 108000 / 108000 |
| c1_mother_00_plus | T_transfer | 8 / 8 / 0 | C00 / INITIALIZER_EFFECT_EVIDENCED | None / NO_QUALIFIED_INITIALIZER_EFFECT_LABEL | 108000 / 108000 |
| c1_mother_01_minus | T_local | 8 / 9 / 1 | C01 / INITIALIZER_EFFECT_EVIDENCED | None / NO_QUALIFIED_INITIALIZER_EFFECT_LABEL | 75820 / 75860 |
| c1_mother_01_minus | T_transfer | 8 / 9 / 1 | C01 / INITIALIZER_EFFECT_EVIDENCED | None / NO_QUALIFIED_INITIALIZER_EFFECT_LABEL | 83920 / 83950 |
| c1_mother_01_plus | T_local | 8 / 10 / 2 | C00 / RULE_ONLY | C01 / RULE_ONLY | 108000 / 108000 |
| c1_mother_01_plus | T_transfer | 8 / 9 / 1 | C00 / RULE_ONLY | C02 / INITIALIZER_EFFECT_EVIDENCED | 108000 / 108000 |
| c2_new_train_plus | T_local | 8 / 10 / 2 | C00 / RULE_ONLY | C01 / RULE_ONLY | 108000 / 108000 |
| c2_new_train_plus | T_transfer | 8 / 8 / 0 | C05 / INITIALIZER_EFFECT_EVIDENCED | C01 / INITIALIZER_EFFECT_EVIDENCED | 108000 / 108000 |

The reviewed streams contain 17 qualified nominal preference endpoints, 12 near historical-reference endpoints, and 10 initializer-effect labels before formal pair selection and final deduplication. 7 RULE_ONLY endpoints and 3 NO_PLAN/preference-unmet endpoints received no positive effect label. No failed or incomplete prefix was credited.

All 12 frozen pairs / 24 raw inputs bind to original TRAIN tasks and are legal, unrepaired, and unresampled. Local source labels and transfer leave-one-mother sources match frozen z and identities. The two explicit local B rule-construction fallbacks remain excluded from positive effects. The known P3 null-plan seed-association edge in `C3_INDEPENDENT_IMPLEMENTATION_AUDIT.md` is untriggered: no illegal or unsupported frozen input creates the requisite null-plan rejected row.

Independently recomputed endpoint completion, B >=30 mm, historical references, A/B near bands, first-hit positions and right-censoring; backend A/B ranking; seed cache association and parent lineage; original-seed supervision z, fixed partner, pair and source identities; sealed prefix digests and candidate/result/plan/reference/started JSON metadata seals; consumed-slot, rollout, preview, geometry and QP accounting. A directly qualified initializer can earn an effect label even when common-rule A wins; it must not be described as causing an improvement. Shared A/B search is counted once.

Work totals: 1023740 main prediction steps, 1023810 separate preview steps, 1215308515 geometry calls and 102381 QP calls. Teacher actual and torque-replay work remain zero. These are existing recorded workloads; this audit added none.

Historical pair hierarchy was recomputed for the five Tasks with both streams complete and is in the companion JSON. No final results seal or dataset existed when this audit snapshot was captured; the 12/12 teacher-result and unified TRAIN dedup/shared D-S-N/scaler/leakage audit remains required. Large NPZ archives were neither read nor rehashed; this review binds their metadata inventory but does not renew their byte-level verification.

The companion JSON records 1536 evidence checks, 0 discrepancies, per-stream endpoints and all JSON paths and hashes read.
