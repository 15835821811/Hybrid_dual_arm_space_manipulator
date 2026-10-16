# Pilot training stage: independent prelaunch review

Decision: **no concrete blocking defect found in the reviewed implementation**. The stage can be queued under the existing conditional authorization, with its implemented fail-closed parent checks. This is a software/protocol review, not a result-to-claim pass for the 10,000-row experiment. The live data job was not polled, altered, stopped, or restarted; no physics or training was run by this reviewer.

Reviewed experiment.py, learning.py, sensitivity.py, audit_results.py, queue.py, job.py, docs/EXPERIMENT_PLAN.md, tests/test_training.py and the shared array loader, codec, and storage contracts. The saved software proof is PASS, all its recorded source hashes match current files, and its JUnit hash matches four passing tests. Those tests use actual saved six-TRAIN/two-VAL arrays, real coupled sensitivities, optimizer updates and checkpoint reuse, plus live-parent rejection, empty-output rejection and missing-sensitivity blocking. They do not establish full-scale completion or exercise the complete final audit/queue lifecycle.

## Population and metadata consistency

- require_data requires exactly 10,000 TRAIN and 48 VAL rows, rejects known execution failures, checks family separation, requires FINISHED plus actual owner exit, both data-stage pass flags and independent audit PASS. It cannot accept merely a completed-looking marker on a still-live worker.
- Loader iterates the actual 10,048 index entries, checks record and array hashes, reads Z arrays, constructs public initialization and complete-state identities, and saves the exact index in training_input_audit.json. The final auditor's 10,048 expectation agrees with that loader output.
- Codec fitting masks TRAIN only. Its fit_ids, fit_split and fit_hash schema matches the final auditor's reconstruction. Training metadata data_hash uses the same z/bases/conditions/IDs/splits/state-IDs schema that the auditor reconstructs from loaded arrays and source records. The shared runtime compatibility check binds the loaded initial state to each task.
- Sensitivity capacity is 11,000, above the maximum 10,000 unique TRAIN task hashes. Each exact TRAIN task receives its own saved 34-direction local surrogate; each accepted anchor requires 39 coupled calls, half-step checks and the mixed-direction check. No nearest-VAL substitute is present. Matrices are stacked in actual TRAIN-row order, and the auditor recomputes the same order and digest.
- Matrix shape is checked against (12, loaded.dim), not a legacy fixed latent dimension. The shared codec derives rank from the actual training population. Training and inference load each checkpoint's own bound codec; frozen_diffusion uses the verified 1,000-row checkpoint.
- Fixed 12,000 updates and batch 64 imply 768,000 draws. Per-row counts, changed weights, checkpoint identity, TRAIN-only fit and zero VAL updates are independently checked. No legacy 1,000-row count, 3,600-update count, or 1,100-anchor cap was found in the new stage's acceptance path.

## Conditional queue and preserved gates

Queue creation requires the specified current parent to be live/RUNNING and a current software proof. It freezes parent PID, run.json hash, source hashes and output path. The worker waits for actual process exit, rejects ownership change, invokes require_data, checks source/run identity, and only then calls launch. A failed parent, insufficient population or failed data audit cannot take the normal launch path. Queue wait is bounded independently from the training budget. The training worker repeats the dependency and source-bound software checks.

The four original frozen DEV tasks, three seeds, five candidates per task/method, common public initialization, shared guidance implementation, candidate resource ceilings, minimum-score selection and one selected execution without replacement remain in place. Success requires all 10,000 TRAIN rows actually consumed, saved sensitivity evidence, new-model stability (at least 18/20 eligible per seed and all tasks reached), 12/12 selected execution/replay, independent audit and no unresolved/error entries. Relative efficiency is not a gate; final TEST and automatic 100k production are disabled. The full goal remains incomplete even if this bounded stage passes.

Failure to obtain sensitivities returns no matrices and blocks new-model training; structured and frozen-model methods remain independent paths. The new-model and mean candidates are explicitly BLOCKED if no trained bundle exists, so the code does not substitute an untrained model. Final audit and summary cannot interpret that path as successful closure.

## Limits and operational interpretation

The queue summary means dispatch only. launch() returns after worker creation and can include an already failed worker status; therefore a queue FINISHED record must never be presented as successful training initialization or training completion. This is already consistent with the documented distinction and is not a scientific launch blocker. Consumers should inspect the actual training job's status/report.

The four tests provide credible component evidence but not proof that 10,000 sensitivities finish within the active budget. Runtime budget or numerical failures should remain recorded failures/BLOCKED results. The independent final auditor is intended to discover any full-scale mismatch before closure is marked passed. No result from the active 10k data job was used or inferred in this review.

Routing: allow the existing conditionally gated queue. Preserve source identity after queuing; source changes should intentionally fail the queued identity check rather than silently change the experiment. After actual execution, evaluate ingestion, candidate feasibility, selected physical execution and comparative benefit separately. No new scientific condition or approval request is introduced by this review.
