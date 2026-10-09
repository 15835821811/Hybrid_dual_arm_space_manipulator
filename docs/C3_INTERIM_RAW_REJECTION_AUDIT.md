# C.3 interim raw rejection audit

Verdict: **CONSISTENT_WITH_ORIGINAL_RAW_REJECTION_PROTOCOL**.

Observed 2026-10-08T12:26:14.101637+00:00 to 2026-10-08T12:26:14.113682+00:00 UTC: 68 checks, 0 discrepancies. This audits only two existing raw-rejection events in live TEST `c3_test0_plus/S`: C01 (slot 1, A/v1) and the subsequently observed C03 (slot 3, B/v2). It supplies no final S8, TEST or research-completion verdict.

Both saved proposals declare `source=regression`, model S, frozen update 4000, and exactly one deterministic regression forward per condition, zero DDIM units and zero generation physics. Initializer metadata is self-sealed, generated once and records no candidate-quality read. Its checkpoint path/SHA agree with the original frozen model, VAL selection and S training report. Checkpoint bytes were not opened or rehashed.

| Event | Preference/family | Active interval 1 norm | Active interval 2 norm | Excess over 20 mm | Consumed slot ordinal |
|---|---|---:|---:|---:|---:|
| C01 | A/v1 | 3.743843002 mm | 29.169357360 mm | 9.169357360 mm | 2 |
| C03 | B/v2 | 4.117369309 mm | 29.127887207 mm | 9.127887207 mm | 4 |

The independent scalar calculation uses the original two-dimensional norm of each of the six residual interval pairs. Both outputs have exactly 12 finite coordinates. Mask `[false,true,true,false,false,false]` leaves four active coordinates; all inactive values, including signed zero, equal zero. Only interval index 2 (the third pair, using zero-based source indices) exceeds 0.020 m. The original `raw_seed_diagnostics` preserve the exact initializer raw numbers, report `raw_legal=false`, `raw_repaired=false`, `resampled=false`, and reject with `raw seed exceeds the original 20mm interval disk`. No reference-precheck result is invented after this earlier norm failure.

Both registry rows independently record `INITIALIZER_RAW_REJECTED`, `prediction_rollout_started=false`, `prediction_steps=0`, null x/plan/plan-SHA/metrics, and `costs={prediction_physics_steps:0}`. Their compact original terminal events show consumed counts 2 and 4, so the rejected proposals occupy their slots. Neither row participates in the exact plan cache. Neither C01 nor C03 has a prediction directory at observation time.

The frozen source confirms the same path: `route_initializers.raw_seed_plan` checks the original 20 mm disk before plan construction and returns null plan on rejection; the optimizer retains a consumed rejection row and returns before cache lookup or nominal evaluator invocation. Prediction directories are created inside that evaluator. The S sampler makes one direct forward per declared condition, applies only the original inverse TRAIN scaler, and has no refusal-driven resampling loop. Retained initializers return their original proposals; an unfinished initializer cannot be silently regenerated.

The matching JSON retains exact raw values and norms, both original selected rejection rows, compact terminal events, source anchors, UTC, check counts, argv/exit and input identities. Each live registry/log input is bound only by its selected line SHA (including its original newline), byte offset/length and snapshot file size/mtime. Other candidate rows and their preview arrays are neither parsed nor emitted. No whole live-file SHA is presented as a final seal.

The audit reads small metadata and six frozen source files with Python standard-library code. It runs no core imports, tests, model inference, physics, training, DDIM, rendering, encoding or export; it opens no weight, NPZ or large archive. Frozen source and run evidence remain unchanged.

Pending:

- Final S8 selection, measured planning cost and outer process completion are outside this event audit.
- Complete the remaining TEST searches, whole-phase selection seal, actual execution, strict aliases and independent gates.
- Keep failure accounting separate from quality or benefit: fewer integrations caused by illegal raw proposals cannot establish learning value.
- TEST and research remain incomplete for this interim audit; default C.1 and deployment NOT_MET are preserved.
