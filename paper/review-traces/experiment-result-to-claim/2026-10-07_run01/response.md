- **claim_supported: partial**
- **confidence: high**
- **paper_claim_audit: unavailable — downstream paper wording remains provisional**
- **blocking_delivery_inconsistency: none found**

The finite research delivery is complete, and nonzero representation capability is established within this pilot’s frozen TaskSpec and declared safety scope. Diffusion learning advantage is **not established**. These judgments must remain separate.

### what_results_support

I derived the following from all 37 attempt results and their separately saved independent evaluation reports:

| Quantity | Independently verified result |
|---|---:|
| Actual runner entries / budget | 37/37 |
| Actual physical steps | 488,260 |
| Complete independently accepted tasks | 35/37 |
| Nonzero complete accepted attempts | 30 |
| Teacher successes | 23/24 |
| Successful TRAIN references / source tasks | 17 / 6 |
| Successful VAL references / source tasks | 6 / 2 |
| TEST E0 | 4/4 |
| TEST E1 | 4/4 |
| TEST E2, fixed K1 | 3/4 |
| K4 actual | NOT_RUN |

The 30 nonzero successes comprise 23 teachers, four TEST retrieval attempts, and three TEST Diffusion attempts. They are **attempts, not 30 independent tasks**.

All 37 saved prefixes have valid independent evidence and pass their executed-prefix reference-binding, execution-contract, interval-recomputation, and native-geometry checks. Full acceptance additionally requires completion and all frozen Task requirements; the two incomplete prefixes remain failures.

I independently reconstructed all **296 requirement evaluations** from existing fresh-replay arrays, including target-frame terminal transport and simultaneous position/orientation satisfaction. Results matched every saved requirement verdict and eligible-state count: 287 requirements passed; nine unexecuted future requirements failed across the two refused attempts. Terminal evaluation uses the actual final state at 27 s.

The first fixed nonzero teacher consumes approximately 10 mm of reference displacement and completes the Task. All four E2 K1 outputs consume more than 5 mm of reference displacement. Seven eligible saved paired-path comparisons reproduce exactly from the persisted replay arrays. Thus the residual changes both controller input and actual motion.

A real, freshly initialized residual Diffusion was trained:

- 150,028 denoiser parameters; two hidden layers of width 128.
- One fixed training run with 4,000 optimizer updates and 128,000 TRAIN-reference exposures.
- All six optimizer parameter-step counters equal 4,000.
- Fixed VAL selection chooses **update 250**, whose checkpoint has **8,000**, not 128,000, training exposures.
- VAL training exposures are zero; the 96 fixed validation draws reference VAL only.
- TRAIN and VAL remain 17 and six independent reference labels, respectively; repeated exposures do not enlarge that dataset.
- TEST sampling uses the selected checkpoint’s matching SHA.

The 16 frozen E2 raw arrays are finite, nonzero, correctly shaped, and numerically distinct. Four predeclared seed values are reused across the four task conditions; this is 16 task–seed slots, not 16 independent seed studies. **14/16** satisfy the fixed 20 mm coefficient bound and pass reference Task/velocity prechecks. The two rejected slot-1 outputs retain norms of 20.498345778 and 20.679655465 mm, their original raw coefficients, rejection records, and unevaluated legal-reference metrics. No clamp, projection, replacement, or zero fallback is evidenced.

All K1 bindings use slot 0. E1 sources only successful TRAIN labels, with the frozen condition-distance and tie-breaking rules. Candidate generation precedes TEST actual outcomes, and the implementation exposes declared Task/geometry inputs rather than future actual trajectories.

### what_results_dont_support

The results do not establish learning advantage. E2 achieves 3/4, while E0 and E1 each achieve 4/4. On its three completed paired tasks, E2 produces slightly shorter continuum paths than E1, but longer paths than E0 and greater intervention RMS than E1. Micrometre-scale clearance differences and changed paths are descriptive observations, not demonstrated quality gains. The failed E2 task must remain in the four-task denominator and outside complete-task quality comparisons.

The results also do not show that nonzero attempts pass the historical strict full-curve runtime acceptance. I independently inspected all completed runtime reports:

- All **30 successful nonzero attempts retain `runtime_reported_passed=false`**.
- Their sole failed historical check is `continuum_irregular_waypoint_path_rmse`.
- Five completed zero-residual attempts retain runtime pass.
- The two failed prefixes retain runtime status `null`.

This is consistent with the separately frozen supplementary `end_effector_detour` TaskSpec allowing intermediate-route changes. The revised report explicitly distinguishes that contract from the unchanged historical strict curve protocol. Consequently, “complete Task” is supportable only with this scope; it must not be rewritten as historical full-curve acceptance or compared directly with old joint-codec 0/24 results.

Reference anchor acceptance is principally supplied by the analytic representation. It does not demonstrate learned exact task reconstruction, joint-space feasibility, whole-body safety of an unexecuted reference, continuous-time safety, or hardware accuracy.

The accepted geometry scope is native robot–target queries at 500 Hz and whole-body checks at 50 Hz boundaries plus configuration-space subdivisions. It is not whole-body scanning at every 500 Hz state or a continuous-time certificate.

Deployment remains **NOT_MET**. The current Cartesian provider has not established valid asynchronous execution while real state continues evolving during calculation delay.

### missing_evidence

There is **no missing experimental evidence required to finish this authorized finite pilot**.

Evidence absent for stronger claims includes broad generalization, repeated-seed statistical advantage, a demonstrated beneficial learning contribution beyond retrieval, hardware/model-mismatch validity, continuous-time safety, and delayed-state deployment validity. These are limits on potential future claims, not completion conditions for this delivery.

Private preview costs are preserved as inherited execution activity but their exact step total is explicitly `NOT_SEPARATELY_AGGREGATED`. This disclosure supports completion without inventing a count; it does not support a fully decomposed preview-cost claim.

### suggested_claim_revision

> Under a frozen finite pilot with six TRAIN, two VAL, and four TEST tasks, a task-anchored Cartesian residual representation produced nonzero controller references and completed 30 independently accepted attempts within the declared discrete execution and geometry scope. A genuine residual Diffusion training run was completed, but its fixed K1 achieved 3/4 TEST Task successes versus 4/4 for both zero residual and TRAIN-only retrieval. The representation is effective in this pilot; learning benefit is not established. Historical strict full-curve runtime checks remain unchanged and fail for the completed nonzero routes. Deployment and continuous-time certification remain unmet.

The revised `REPORT.md` and `delivery_analysis.json` express this distinction appropriately. The error-table qualification is important: waypoint and orientation errors are measured at each requirement’s selected simultaneous satisfaction time, not maxima over the entire window or trajectory; terminal errors remain at 27 s.

### next_experiments_needed

**None within the current task. Stop experimental work.** Preserve the nonlearning baselines, checkpoint, rejected outputs, and failed prefixes. Do not increase network size, candidates, seeds, teacher search, or actual attempts to obtain a positive learning result. Any future investigation of stronger claims requires a separate user request and budget.

### Failure preservation and delivery checks

Both failures remain visible and excluded appropriately:

- `attempts/teacher_17/attempt_result.json`: original domain refusal at **14.900 s / 7,450 steps**; excluded from labels.
- `attempts/TEST_01_E2/attempt_result.json`: original domain refusal at **16.620 s / 8,310 steps**; retained as TEST failure.

Both report `RAMP_MICROSTATE_OUTSIDE_DECLARED_DOMAIN; no further servo step executed`. Passing their consumed-prefix checks does not establish completion or a real-world backup-control guarantee.

I independently recomputed dispatch statistics from all 37 saved timing JSONL files: **48,828 cycles**, **638 exceeding 20 ms**, and a longest within-run sequence of **24** in `TEST_02_E2`. Per-run p95 spans 17.102075–19.089850 ms; the maximum per-run p99 is 24.680413 ms and maximum individual latency is 61.305300 ms. The report correctly describes per-run aggregation rather than a pooled percentile and preserves startup and tail observations.

The 11 complete TEST error/quality table rows match saved evaluations, including displayed rounding. Failed quality comparisons remain blank. Independent cost aggregation matches 488,260 replay steps, 48,863 interval boundaries, 90,901 rows, zero interval mismatches, 36,622,275 robot–target queries, and 571,763,107 whole-body queries.

### Reviewed paths and actions

Primary output directory:

`E:\v64b2work_20261007_01\v6_4\output\task_anchored_residual_20261007_01`

Reviewed:

- `plan.json`, `source_identity.json`, `summary.json`, `table_A.json`, `table_B.json`, `budget_ledger.json`.
- `teacher_results.json`, `teacher_manifest.json`, `dataset/manifest.json`, frozen sample records and source bindings.
- `training/model/training_report.json`, both checkpoint dictionaries, training curves, validation draws, and training configuration.
- `test_candidate_manifest.json`, all 16 raw/precheck records, `raw_reference_diagnostics.json`, and `paired_path_comparison.json`.
- All 37 `attempts/*/attempt_result.json`, separately saved evaluation reports and manifests, native/interval evidence identities, replay arrays, historical runtime reports, and dispatch timing JSONL files.
- All `audits/*.json`, the revised `REPORT.md`, and `delivery_analysis.json`.
- Relevant implementation in `residual_execution.py`, `proposal_gate.py`, `task_protocol.py`, `residual_protocol.py`, `task_anchored_reference.py`, `residual_diffusion.py`, and `task_anchored_residual.py`.

Independent integrity checks found no mismatch in **321 frozen source files, 70 protected inputs, 72 dataset source bindings, 68 candidate artifact bindings, all 37 trace hashes, all 37 evaluation-report hashes, or 261 evaluation-manifest entries**.

This review was read-only. It performed **zero physics steps, optimizer updates, inference calls, samples, installations, or artifact edits**.
