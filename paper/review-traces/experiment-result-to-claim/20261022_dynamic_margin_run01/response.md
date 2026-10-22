# Independent result-to-claim judgment

**Local P2a gate: PASS. Overall claim_supported: partial. Confidence: high for this bounded evidence judgment. Paper writing verdict: provisional — no paper claim audit available.**

The new composite protocol supports four qualified fixed TRAIN units: one new t00 execution and three exact historical reuses. It does not provide four new executions. `dynamic_reference_001` remains 3/4, including its failed t00. Across the two runs there are five real execution attempts, four passed and one failed; control replays are verification, not additional independent task attempts. This adaptive TRAIN repair is not an estimate of unseen success probability.

## Identity and reference evidence

The producer head is `d5ba210681721de3e1f33ace46f9cbec3af05565`. All 168 source-file hashes match both current bytes and the exact 168-member executed-source archive; all 151 bound input files match. The frozen selection is exactly the original four TRAIN rows with seeds 31000000, 31104729, 31209458, 31314187. All 166 old source hashes remain equal, and all 64 old run files remain hash-bound. Hash manifests and the executed archive establish exact source identity; the commit label alone is not treated as a claim of a clean working tree.

All four references retain 277 knots, 273×17 cubic coefficients, 0.1 s spacing, exact [0,27] domain, and original source arrays. Direct polynomial extrema over all 270 positive knot spans reproduce every reported bound; C² join discrepancies are at most 1.94e-13 in the three checked orders. Initial reference position and velocity errors are zero. Free coefficient position bounds and derivative coefficient hull constraints pass without changing physical limits.

The fixed internal acceleration fraction is 0.9695632054405793, exactly `1 - 2 × 0.015218397279710373` from the frozen prior TRAIN residual. Only coefficient column 4 changes in t00. Its one constrained fit records SLSQP success after 91 iterations; the remaining 16 coordinates use feasible unconstrained optima. All unconstrained stationarity residuals are below 3.5e-18 and recomputed objectives match logs within 6.7e-16. No optimizer was invoked for this audit. The reference's maximum acceleration ratio is 0.9695632054405792, and sample fit maximum/RMS errors are 0.0005940066482 / 0.00002642458552 rad. These support feasible saved references, not a general optimality or response-bound theorem.

|Unit|Reference acceleration ratio|Evidence provenance|New native steps|
|---|---:|---|---:|
|t00|0.969563205441|New native execution plus independent saved-control replay|13500|
|t01|0.828022344168|Exact historical reuse|0|
|t02|0.808882111221|Exact historical reuse|0|
|t03|0.825559016552|Exact historical reuse|0|

For t01–t03, nodes and coefficients are element-for-element identical to old references; complete identity JSON/digests, all initial-array keys and values, limits, source identities and old execution status/gates match. Old native traces, which contain the controls/raw torques, match their recorded SHA256; the independent replay artifacts also remain within the immutable 64-file binding. New result files contain precisely the old results plus `new_execution=false`, `actual_new_steps=0`, and their `reused_from` provenance. No new native/replay files exist for those cases. All three reuse qualifications pass. Prior physical qualification is reused, not re-performed.

## New t00 physical evidence

The complete 836-value INTEGRATION initial state has the correct state ID and is exactly the first native and replay state, also equal to the old full initial state. Original source qpos/qvel and the moving target's complete initial twist match. The native trace contains 13501 states and 13500 controls; replay contains the same counts. Time runs from 0 to 26.999999999995914 s at 0.002 s spacing, with maximum spacing residual 1.11e-15 s.

All 16 original physical gates pass when evaluated against saved raw evidence and frozen code. Maximum actual planner/native acceleration ratios are **0.970608708096 / 0.970613193346**; there are zero violations. Planner/native velocity ratios are 0.370532909725 / 0.370532959689. The original planner limits remain 0.8/1.2 rad/s and 2.5/4 rad/s², with their unchanged 67-dimensional mapping. Original position limits, the ±1 continuum domain, reference limits and finite-state gates pass.

Maximum world-position errors are **0.231939 / 0.675595 mm** for the continuum/rigid tips; maximum orientation errors are **0.0122738 / 0.000934718°**. Both are below unchanged 5 mm/1° gates throughout. Relative hold at ≥25.5 s also passes. Desired continuum and moving-rigid world trajectories reconstructed from frozen path, target state and approach metadata match saved desired poses within 5.56e-16. An independent URDF forward-transform calculation from qpos matches both saved end-effector transforms within 5.78e-15, substantiating the world-frame errors without invoking physics.

All 27001 geometry scan states are accounted for: initial, 13500 native endpoints and 13500 quaternion-geodesic joint-space midpoints. Every endpoint is exact; midpoint maximum error is 1.23e-15. Recorded minimum-clearance reductions exactly reproduce the native clearance history; minimum clearance is **4.8496722846 mm** under the unchanged nonpenetration gate. Frozen collision code covers all relevant body/obstacle/target pairs with explicit topological exclusions, no intentional grasp exemption, and conservative sphere-bound pruning. The saved pair witnesses include both wrist/end-link and end-link/target minima. No collision engine was rerun; this accepts the raw verifier output under the inspected frozen verifier, not a second independent distance algorithm.

The saved geometry-step metric is 0.216860 mm. Independently computing URDF transforms at every native and midpoint state and bounding vertex displacement with original mesh radii over all 154 URDF visual/collision geometries gives a conservative maximum consecutive-sample bound of **0.239990 mm**, below 2 mm. Scene obstacles are static. Neither calculation is a continuous-time swept collision certificate.

Raw 67-dimensional torque maximum ratio is 0.041224792142, with raw torque exactly equal to applied saved controls; no torque clipping is needed. Tracking maximum is 0.000687730282 rad against 0.02, manifold deviation 7.025e-7 rad against 0.02, and both momentum components are below the original exchange-scaled thresholds. Applied generalized forces and Cartesian wrenches are all zero in every saved full integration state.

The independently generated replay matches **qpos, qvel, controls and all 836 integration components with maximum difference zero**. Inspection of the frozen replay path confirms it restores the same full initial state into separate simulator data and advances saved controls, without copying actual intermediate states. This review itself performed no simulation. The 4.5 and 25.5 s transition archives exactly match the three-state slices from both native and replay histories; both are ordinary adjacent native steps with no reset branch. Reference position/velocity/acceleration commands also exactly match the saved spline evaluations at every command time.

## Budget and scope

Observed monitor elapsed time is 76.985 s; the experiment's own timer is 75.813 s. These timers start at different points and are both below 600 s. New computation is exactly 13500 native + 13500 replay = 27000 steps. Run output is 146201093 bytes, below 1 GiB; the final budget snapshot is 276 bytes smaller because final metadata is written afterward. Launch/source inspection enforces at most two numerical compute threads; this is a configured cap, not a sampled OS CPU-utilization measurement. The four fits and events show native execution only for case 0. There is no training, library addition, DEV or TEST execution in this run.

P2 accounting remains 108000 old P2a steps + 27000 repair steps + 297000 remaining P2b new steps = 432000. Twelve original logical P2b task/seed/method units remain required; only eleven new full execution/replay units fit in the balance. At least one strict identity reuse must therefore be independently qualified. This P2a trace does not establish that future P2b reuse. Any missing/unqualified unit stays missing/BLOCKED; targets, seeds, split and denominators may not change to manufacture eligibility.

## Claim boundaries and routing

What results support: this one frozen margin repair removes the observed t00 acceleration violation while preserving original gates, and the other three unchanged references retain their exact historical execution qualifications. It completes the local P2a prerequisite for subsequent bounded work.

What results do not support: completion of the broader C1 task-input-to-path system, P2b matched DEV execution, held-out/generalization robustness, a theorem that this empirical rho prevents all future violations, zero-failure history, expanded TRAIN eligibility, continuous-time collision safety, hardware or contact capture. These references are fitted to known frozen TRAIN trajectories. P1 was not re-reviewed and its existing conclusions are neither changed nor inferred from these results.

C2 Diffusion contribution is **not evaluated**. No new network training or matched Diffusion/structure ablation occurs, and no causal performance, speed or sample-efficiency benefit follows from this repair. C² spline continuity is unrelated to C2 as a paper claim.

Suggested claim: “On four preselected moving TRAIN trajectories, a single frozen internal acceleration-margin repair yields one newly verified 27 s execution plus three exactly equivalent historical qualifications under the original physical gates. Across the original and repair runs, four of five real execution attempts pass. This establishes the bounded P2a prerequisite; downstream dynamic task comparison and Diffusion contribution remain open.”

Route: confirm local P2a; supplement broader C1 with the original bounded P2b/P3/P4 evidence. Before new P2b physics, freeze its accurate unit identities, reuse eligibility and 297000-step accounting. Do not repeat this same rho repair or silently add a twelfth new execution. A deterministic baseline reused across seed-labelled logical slots is shared evidence, not independent repeated trials; it requires its own protocol/identity review and is not endorsed by this trace. C2 remains a later matched comparison. No supplementary experiment was started by this reviewer.

`paper/PAPER_CLAIM_AUDIT.json` is unavailable. This does not negate the bounded evidence check, but every downstream paper-writing verdict remains **provisional — no paper claim audit available**. This is a completed independent reviewer judgment, not `[pending external review]`.
