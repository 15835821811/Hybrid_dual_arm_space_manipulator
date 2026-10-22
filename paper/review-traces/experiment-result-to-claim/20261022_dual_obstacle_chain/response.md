# Independent reviewer response

**claim_supported: yes; confidence: high for this single bounded engineering execution.** The downstream writing verdict is **provisional — no paper claim audit available**. No blocking defect was found in the saved evidence for the intended physical claim. This is not approval of a broader network, generalization, superiority, moving-satellite, continuous-time or hardware claim.

## What the evidence establishes

The current run completed the frozen nine-obstacle static-to-dynamic engineering chain. The static system used the existing trained diffusion prior, public initialization and geometric bow, full323 correction, and physical execution. Of five proposals, candidates 4 and 2 obtained MODEL_VALID; candidate 4 was selected by the frozen minimum qualified score. Exactly one static candidate was executed, with no replacement. The subsequent dynamic segment used the predefined C2 path and existing feedback controller with whole-machine constraints.

`verification.py` passed 47 saved-evidence checks. Every one of the 208 source entries matches `executed_source.zip` and the source files read for this review. All 93 bound inputs match. Producer head is `46b552fded439dce19a85b02430159e18508ee89`. The new compiled scene identity is `827730a8f4214cbcbda8f8ef323aadc9a0d8d588c1ef65e2fbe2f9c007f52398`, different from the original bound three-obstacle identity `dcd1054bb7c527edab312c48edb962708375536d4ad67537f0d01261b0d61633`. Task, model, initial-state and pair-policy hashes agree across the new records.

`verify_proposals_budget.py` reproduced all five 16-call neural raw proposals bit for bit without physics or geometry calls. The public bow reconstruction agrees to at most 1.008e-13 rad; it is not labeled bit exact. The script disabled simulator stepping, forward kinematics and distance entry points. The network is explicitly not obstacle-conditioned; the detail head is the existing zero-detail configuration, with no new training.

The static native trace has 13,751 states and 13,750 controls; its replay has the same complete 836-dimensional states. The dynamic native/replay traces have 13,501 states and 13,500 controls each. Every saved replay field agrees exactly, including all 836 integration components. The static last state, saved dynamic initial state and dynamic first state are exactly equal in full integration state, qpos, qvel, absolute time, end-effector poses and base pose. The saved last static reference and reference velocity are inherited exactly. The two saved boundary triplets are exact slices of the native sequence.

The original static 15 reported gates and dynamic 19 reported gates are all true. The review independently recomputed saved-array pose, timing, joint, motion, raw-torque, tracking, manifold and replay checks, and checked saved geometry minima and their correspondence to actual stored states. Static hold position maxima are 0.054606 / 0.004979 mm. Dynamic position maxima are 0.180514 / 0.004931 mm; rotation maxima are 0.006549 / 0.000244 degrees. The static minimum saved full-machine clearance is 1.543031 mm and dynamic minimum is 8.481855 mm. The raw torques equal controls without saturation; maximum torque ratios are 0.004925 static and 0.011083 dynamic. Dynamic observed velocity and acceleration ratios are 0.251800 and 0.370053 or less. Both stages' full INTEGRATION external-force tails are zero. The actual clock runs 0 → 27.5 → 54.5 seconds with 2 ms increments.

Geometry values here remain the run's already-saved observations: 27,501 static geometry samples and 27,001 dynamic samples. No geometric or physical re-certification was performed. The saved maximum geometry-step values are below the original 2 mm gate. Static momentum peak was checked against its saved threshold; dynamic momentum peak and threshold were both recomputed from saved momentum and exchange arrays. This does not become a continuous-time collision certificate.

## Fresh execution despite the identical static native hash

The new static native hash is `03bff077eb560ad30450184ad83db8039c1bbe2be3da6c062e0de8af74f4d3d5`. Its equality to D1 was supplied in the task; D1 was not re-audited. Hash equality alone proves identical bytes, not reuse or freshness.

The supporting chain is stronger than the summary: the frozen new-run source creates the nine-obstacle model before static planning; `run_unit` calls `audit_execution` on that model; `track` directly calls `mujoco.mj_step` for each control; its independent replay resets once and then applies the saved controls through its own step loop. The saved new-run calls ledger records one torque rollout and 27,500 physics steps. A new native execution event precedes the handoff by approximately the reported static audit runtime. The new model/task/policy identities and archived saved initial/native/replay/geometry files are mutually consistent. The source has no branch that substitutes D1's native trace. Dynamic handoff reads this run's static file.

Identical static physical arrays are consistent with unchanged dynamics, deterministic controls, and added world spheres that do not contact or become the minimum pair along that static trajectory. Scene identity is stored separately from the state-array NPZ, so a different model identity need not change the NPZ hash. These observations support a new invocation in the new scene; they do not claim independent signed process attestation.

## Six added spheres and the seven original points

The new scene preserves the three original objects verbatim and appends six radius-12-mm spheres exactly at the six adjacent waypoint midpoints. The archive's model compiler adds each as `v7_continuum_segment_i`, `contype=1`, `conaffinity=1`, opaque red geometry. The original collision backend includes all contact-enabled geoms and excludes only welded/direct mechanical neighbours, not robot-to-world spheres. The recorded initial transfer contains nine added world geoms and the new policy identity; all later gates use that identity. The saved render source renders this same model with all geom groups enabled, and the HTML payload contains these same nine objects. Its complete media manifest hashes match.

Algebraically, each original minimum-jerk straight segment passes through the corresponding sphere center at its temporal midpoint: the signed point-to-sphere distance is -12 mm. The predefined C2 offset is exactly -60 mm in world x at each midpoint and vanishes with its first two derivatives at both endpoints. Saved desired positions match that curve at every dynamic time. Saved actual midpoint samples lie about 60.164–60.177 mm from the corresponding centers. These coordinate calculations explain why the new spheres are relevant to the original path; they are not fresh whole-body clearance measurements.

The seven full-precision original points, rotations, six durations, transition duration 4.5 s and path duration 21 s match the bound original task exactly. The same points and timing are retained through the C2 detour. The final hold continues for 1.5 s. Each reference boundary is checked algebraically; actual arrival is reported at the nearest stored sample, not at an unobserved continuous instant.

| Point | Scheduled local s | Nearest sample local s | Time offset ms | Actual point error mm |
|---|---:|---:|---:|---:|
| 1 | 4.500000000 | 4.500000000 | approximately 0 | 0.010661 |
| 2 | 7.979548146 | 7.980000000 | +0.451854 | 0.051203 |
| 3 | 11.592160704 | 11.592000000 | -0.160704 | 0.051902 |
| 4 | 15.049767079 | 15.050000000 | +0.232921 | 0.050101 |
| 5 | 18.541613940 | 18.542000000 | +0.386060 | 0.051679 |
| 6 | 22.013217514 | 22.014000000 | +0.782486 | 0.052912 |
| 7 | 25.500000000 | 25.500000000 | approximately 0 | 0.050134 |

Absolute time is local time plus the actual 27.499999999995637 s handoff time. Full coordinates, actual absolute times, signs of time offsets and rotations are in `compact_evidence.json`; all seven nearest samples satisfy the original tolerances. The exact first/last time offsets are only floating accumulation errors, -2.49e-9 and +2.32e-8 ms.

## Budget and attribution limits

Budget reconstruction from saved grid histories, all 323 perturbations in each repair iteration, trial records and validation grids gives: screening 10 calls / 3,840 intervals; candidate 4 1,959 / 1,158,760; candidate 2 1,963 / 1,141,524. Each repair has six completed iterations and stays below its independent 3,450-call, 8,300,000-interval, 16-iteration limits. Total 3,932 calls / 2,304,124 intervals matches the saved budget and actual counters exactly. New physical execution plus replay is 27,500 + 27,000 = 54,500 steps; saved elapsed time is 421.516 s and reviewed output size is 449,906,442 bytes. Pending markers are clear, status FINISHED and no replacement/candidate expansion/training/comparison is indicated. Memory enforcement is present in the frozen source; a time-resolved peak-memory log was not retained, so this review does not invent a measured peak-memory value.

The prior D2 debit (27,000 steps, zero coupled calls, 56.985 s) is inherited from the bounded request/protocol rather than independently re-audited. Arithmetic totals including that declared debit are 81,500 native/replay steps and approximately 478.501 s; this is not a new prior-run certification or refund of prior work.

The dynamic `control_filter.json` contains 1,350 records, every desired velocity equal to the filtered velocity and every delta exactly zero. Thus the existing constrained feedback controller was used, but the constraint filter did not actually modify this run's requested velocities. None of the six new spheres became the global minimum pair. The successful geometry-designed path is meaningful, but evidence does not demonstrate a triggered online collision correction, learned dynamic detour, or necessity/benefit of the filter in this run.

The frozen protocol dictionary inherits some stale static labels (`tasks=['capture_3_obstacles']`, `full_static_dynamic_chain_claim=false`). These do not change the actual scene, source branches or saved results; the version, new selection, contract and frozen narrative identify the intended nine-obstacle engineering case. They should not be quoted as an accurate full-scope metadata summary. Frozen files were not edited.

## Non-blocking resume capture defect and routing

The executor's `completed_resume.json` is explicitly `PARTIAL_STDOUT_CAPTURE_ERROR`. It records exactly one call, no retained stdout, no persisted before-manifest, and a short-circuit inference that the command returned zero and the two inventories compared equal. The reviewer cannot independently reconstruct the missing stdout or before snapshot, and does not mark this a full resume PASS. A separate reviewer readback confirms that all 222 run files in the review inventory remain byte-identical at closeout. This is a within-review integrity check, not a substitute before/after resume certificate.

This capture defect affects closeout trace completeness, not the completed native trajectory, saved controls, 836-state handoff, physical gates or independent replay evidence. Do not repeat resume merely to repair its transcript. No further physical experiment is needed for this bounded claim. Confirm the local engineering result and report these limits; broader paper claims require their own evidence and a paper-to-evidence audit. `paper/PAPER_CLAIM_AUDIT.json` is unavailable, so downstream writing remains provisional.

Suggested claim: “In this single frozen nine-obstacle development scene, the existing diffusion-based static planning-and-correction system produced and executed a qualified static path. Its actual complete terminal state was handed to the existing feedback controller, which tracked a predefined C2 detour through the original seven waypoints at the original schedule. Both stages passed the original sampled physical gates and complete independent state replay. The dynamic constraint filter was present but inactive on this run.”
