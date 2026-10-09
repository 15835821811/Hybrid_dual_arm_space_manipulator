# C.3 final VAL selection audit

Verdict: **NO_DISCREPANCY_IN_REVIEWED_VAL_SELECTION_METADATA**.

Independent metadata audit at 2026-10-08T12:01:29.310282+00:00–2026-10-08T12:01:29.698794+00:00 UTC: **1975 checks, 0 discrepancies**. Only completed VAL and model-freeze identity/timing are adjudicated. No TEST score or whole-study completion is asserted.

The independently recomputed result is **D4000 and S4000**. D4000 wins at criterion 3 (near quality 3 versus 1, after both tie at three complete endpoints and one B30). S4000 wins at criterion 1 (three complete endpoints versus two). Cost is later in the fixed ordering and decides neither winner.

| Endpoint | Full 27s + five gates /4 | B30 /2 | Near /3 eligible | NO_PLAN /4 | Unique actual | Strict aliases |
|---|---:|---:|---:|---:|---:|---:|
| R12 | 3 | 1 | 3 | 1 | 3 | 0 |
| D250 | 3 | 1 | 1 | 1 | 3 | 0 |
| D4000 | 3 | 1 | 3 | 1 | 2 | 1 |
| S250 | 2 | 1 | 2 | 2 | 1 | 1 |
| S4000 | 3 | 1 | 3 | 1 | 2 | 1 |

All 20 logical slots are retained: **11 unique actual + 3 strict aliases + 6 NO_PLAN**; 14 logical endpoints pass full 27s and the original five gates. All 11 executed actuals pass. No actual precheck rejection, actual failure or tool error is present. The three aliases are plus-Task S250-A/D4000-A/S4000-A to its R12-A, with matching Task, plan, source/config and independently reconstructed cold-history digest, original slot SHA and zero new work. Aliases are not independent trials.

R12 actual references qualify for minus-A, plus-A and plus-B. Minus-B has NO_PLAN, so relative quality is N/A for every checkpoint and is excluded from both near-quality count and first-hit sum. D250 fails near quality on plus-A and plus-B; S250 has NO_PLAN on minus-A. Both 4000 checkpoints keep all three qualified references. Each model's minus-B remains NO_PLAN; a complete chosen model result for every preference is not established.

The full lexicographic minimum keys are preserved below in the order: negative full endpoints, negative B30, negative near endpoints, raw illegal, first-hit encoding sum, prediction steps, preview steps, native geometry queries, measured cold planning seconds, update.

```json
{
  "D250": [
    -3,
    -1,
    -1,
    0,
    14,
    191820,
    191850,
    216631555,
    1168.9494225000963,
    250
  ],
  "D4000": [
    -3,
    -1,
    -3,
    0,
    9,
    192140,
    192170,
    216647661,
    1153.4698987999,
    4000
  ],
  "S250": [
    -2,
    -1,
    -2,
    0,
    12,
    169440,
    169500,
    168229016,
    998.3686891999096,
    250
  ],
  "S4000": [
    -3,
    -1,
    -3,
    0,
    5,
    199910,
    199930,
    232790609,
    1207.059026999399,
    4000
  ]
}
```

| Checkpoint | First near minus-A | minus-B | plus-A | plus-B | Sum |
|---|---:|---|---:|---:|---:|
| D250 | 4 | N/A | 1 | >8 (right-censored; code 9) | 14 |
| D4000 | 4 | N/A | 1 | 4 | 9 |
| S250 | >8 (right-censored; code 9) | N/A | 1 | 2 | 12 |
| S4000 | 2 | N/A | 1 | 2 | 5 |

First hits use only each saved consumed eight-slot registry against the same actual-qualified R12 set. Code 9 means no hit within budget 8 and is never claimed as an observed ninth slot. All 16 original D/S raw proposals are legal, finite, mask-valid and within 20mm per interval; no raw repair/resampling or rejection. First-near results are prediction screening, not additional independently executed actuals.

The 88 consumed search slots contain 89 parameter proposals and one exact within-stream cache hit. Recorded main prediction steps total 997,920, preview steps 998,160, geometry queries 1,070,364,078. Cold inner planning sums to 5946.768s, outer request wall sum 5952.286s. A/B share each request's cost once. The 11 unique actual pipelines add 148,500 feedback steps, 148,500 preview steps and 148,500 independent replay steps. Actual/validation offline wall sums to 1616.309s; original whole-VAL phase wall is 7571.585s. These are separate scopes; partial predictions and NO_PLAN costs cannot establish benefit.

The original all-selections seal (2026-10-08T11:25:40.566187+00:00) binds all 10 requests and precedes every unique actual marker/start. Its 50 small JSON members still match; four sealed weight SHA identities agree with training, initializer, scoring and freeze records without reopening weights. All 20 local slot seals and 11 independent-evaluation seals have their current small JSON members checked. Original raw trace/replay/interval SHA strings agree across evidence layers, but raw archive bytes are deferred.

Root and closed_loop_val/model_selection.json are byte-identical (`fb10fd35a70a7cf03475949968d9384a18b5d0670220e6a8f8237cecd48ca1be`), and their four full score dictionaries exactly match this independent recomputation. Model freeze at 2026-10-08T11:52:40.802152+00:00 binds those chosen checkpoint identities and the root selection. Its 22 small current artifacts match their hashes. Original closed-loop exit 0 precedes freeze-models exit 0; the freeze completion at 2026-10-08T11:52:40.8497757Z precedes the first TEST command-log creation (2026-10-08T11:52:41.653579+00:00, filesystem metadata only). No TEST file content was read or hashed. The frozen source also verifies model freeze before starting TEST workers.

The original five-gate evidence is the saved independent evaluation/report.json, not the slot's five boolean declarations. The audit crosschecks eight task errors/windows/tolerances, execution subchecks, independent interval counts/mismatch 0, native geometry subfile and recorded zero violations, reference residuals/binding counts, full 13500 steps and 27s saved horizon. Whole-body geometry retains its original 50Hz boundaries plus four configuration subdivisions; robot-target geometry uses saved 500Hz states. This does not upgrade discrete evidence to continuous-time safety.

The auditor did not recompute the physical gates or quality from raw traces, reopen/hash NPZ/weights/large candidate facts/media, run tests or import experiment core. Only eight relevant sources are byte-verified against producer `9f39b42775283432eb933f63a9047c488ba22070` and the 430-entry source identity. Cold-history hashes bind frozen initial arrays and a reset policy supported by source control flow; live controller/QP dual/reference-integral histories and trace initial-state telemetry were not independently re-audited. Embedded preview-record arrays were not analyzed or emitted by the audit script.

Scope remains DATA_LIMITED: one VAL mother, two paired Tasks, one training seed. This completes the VAL selection metadata audit only. TEST, overall research execution, independent generalization, learning benefit, release and visualization refresh remain outside this verdict. Default C.1 and deployment NOT_MET remain; continuous-time and hardware safety are NOT_ESTABLISHED.

The matching JSON retains UTC, exact audit argv/exit/script SHA, original phase argv/exit/time, input SHA inventory, check counts, per-logical-endpoint quality/source identities, full keys, aliases, all differences and explicitly deferred artifact inventory.

Logical actual endpoints (quality units: I rad/s, L m, d mm):

| Task | Endpoint | Pref | Actual kind | Full27s/five | I | L | d | Near R12 |
|---|---|---|---|---|---:|---:|---:|---|
| c3_val_minus | R12 | A | unique | True | 0.110403 | 1.192945 | 25.584524 | True |
| c3_val_minus | R12 | B | NO_PLAN | False | N/A | N/A | N/A | N/A |
| c3_val_minus | D250 | A | unique | True | 0.110807 | 1.196765 | 25.587947 | True |
| c3_val_minus | D250 | B | NO_PLAN | False | N/A | N/A | N/A | N/A |
| c3_val_minus | D4000 | A | unique | True | 0.108265 | 1.193269 | 25.643549 | True |
| c3_val_minus | D4000 | B | NO_PLAN | False | N/A | N/A | N/A | N/A |
| c3_val_minus | S250 | A | NO_PLAN | False | N/A | N/A | N/A | False |
| c3_val_minus | S250 | B | NO_PLAN | False | N/A | N/A | N/A | N/A |
| c3_val_minus | S4000 | A | unique | True | 0.109828 | 1.196641 | 25.650069 | True |
| c3_val_minus | S4000 | B | NO_PLAN | False | N/A | N/A | N/A | N/A |
| c3_val_plus | R12 | A | unique | True | 0.091742 | 1.127014 | 28.361736 | True |
| c3_val_plus | R12 | B | unique | True | 0.093484 | 1.129219 | 40.099908 | True |
| c3_val_plus | D250 | A | unique | True | 0.090322 | 1.134047 | 26.139393 | False |
| c3_val_plus | D250 | B | unique | True | 0.094360 | 1.134782 | 40.373848 | False |
| c3_val_plus | D4000 | A | alias | True | 0.091742 | 1.127014 | 28.361736 | True |
| c3_val_plus | D4000 | B | unique | True | 0.091826 | 1.128778 | 30.941828 | True |
| c3_val_plus | S250 | A | alias | True | 0.091742 | 1.127014 | 28.361736 | True |
| c3_val_plus | S250 | B | unique | True | 0.092535 | 1.127361 | 32.524095 | True |
| c3_val_plus | S4000 | A | alias | True | 0.091742 | 1.127014 | 28.361736 | True |
| c3_val_plus | S4000 | B | unique | True | 0.092186 | 1.127208 | 31.483944 | True |

Audit tooling history is retained in JSON. Corrections affected only this metadata audit script, never original experiment evidence.
