# Independent reviewer response — moving_satellite_001

**C1: no, high confidence. Caux: yes within the qualified descriptive scope below, high confidence. Overall: partial; paper writing remains provisional because no paper claim audit is available.** The 29-second continuation executed and its saved complete-state replay agrees exactly, but the unchanged original robot momentum gate fails. `FINISHED_WITH_BLOCKED_ITEMS`, `EXECUTION_FAILED`, and rejected reference grade are correct and must be retained. A finished execution is not an all-gate qualified engineering success.

## Scope and identity

This fresh review independently reconciled this run only. Its actual source identity is Git base `9e4fb4fb85ea45d3b307a18885786c1f5a3f6b53` plus 185 current source hashes and identical archived source bytes, with 274 bound-input hashes. All identities match. The newly bound 836-value terminal is identical to native state zero and the run's initial snapshot. Its absolute start is 54.50000000002064 s and end 83.4999999999868 s. The predecessor closeout is provenance; its raw trajectory was not reanalyzed.

`audit_saved.py` was executed once and completed in 11.812 seconds. It compiled the same frozen model and independently reconstructed static kinematics and mass matrices at all 14,501 saved native states, with dynamic stepping, forward dynamics and position integration disabled. This is observation of saved evidence, not another physical experiment or replay. The arithmetic audit reports 24 named checks passing, zero producer gate disagreements, and one failed physical gate. The check count represents grouped checks over arrays, not 25 scalar observations.

## Why momentum fails

The original test is unchanged: for each of the two 3-vector groups, compare the maximum norm of the robot generalized free-base momentum with `1e-5 + 0.002 × maximum exchange norm`, where the maxima are over this run. It is not a test of change from this run's initial momentum, and satellite momentum is not subtracted.

| Quantity | Linear group | Angular group |
|---|---:|---:|
| New initial norm | 4.314516949040302e-5 | 5.966906504994827e-5 |
| New final norm | 4.315719970851054e-5 | 5.964605292548784e-5 |
| New maximum norm | 4.315719970851054e-5 | 5.966906504994827e-5 |
| New maximum exchange norm | 0.015223997428951386 | 0.017364663805364486 |
| Frozen-formula threshold for this run | 4.044799485790277e-5 | 4.472932761072897e-5 |
| Maximum / threshold | 1.0669799544 | 1.3340031750 |
| Maximum norm of change from initial | 5.461214251894060e-8 | 1.103372542893922e-7 |
| Saved states exceeding threshold | 14,501 / 14,501 | 14,501 / 14,501 |

Both groups fail **already at index zero**, before any new force interval. All subsequent states remain over the current threshold. The angular maximum occurs at index zero; the linear maximum occurs at the final state. Reconstructing `M(q)qvel` and the exchange term from the frozen model and all saved `qpos/qvel` gives exactly zero disagreement with the saved momentum arrays.

For context only, the predecessor's sealed producer report records thresholds `[9.129619343742031e-4, 9.646531758726157e-4]` and maximum momenta `[1.549670465776948e-4, 2.077821018230808e-4]`, with `EXECUTION_VERIFIED`. These are retained scalar metadata, not a new independent verdict on that run. The current thresholds are about 22.57 and 21.57 times smaller because the current continuation has much smaller peak exchange; no tolerance coefficient was changed. The inherited initial momentum was acceptable within the predecessor's broader exchange window but is above the threshold recomputed for the low-motion continuation. This explains how exact-state handoff and previous acceptance can coexist with failure from the first new state.

The evidence therefore supports an **inherited nonzero momentum plus a smaller exchange-scaled numerical allowance**, rather than a newly created large momentum impulse. The new vector drift is small but nonzero. This does not excuse the failure, justify subtracting the initial value, change the threshold, or demonstrate its ultimate historical cause. No old trajectory was opened to diagnose where the inherited residual arose. The rotational generalized momentum is expressed in free-joint coordinates, so its vector change is not itself world angular-momentum drift. A static coordinate conversion gives a maximum change of world-origin robot angular momentum of 1.993583975895941e-8; this is a diagnostic, not a replacement gate.

## Supported physical-origin and saved tracking evidence

There are exactly 14,500 executed native intervals and 14,500 producer replay intervals. Comparing every one of 14,501 × 836 = **12,122,836** saved integration-state values gives maximum absolute difference zero; `qpos`, `qvel`, controls and full external-input arrays agree. Time spacing and transition snapshots at shutdown (56.5 s) and final hold start (82 s) agree with the same saved native states. Exact replay is evidence of the recorded execution and frozen implementation, not an independent continuous-physics proof.

The force ledger contains exactly 1,000 nonzero input intervals, evaluated at the 2 ms interval midpoints. Its world +Z force matches `0.02(1-cos(pi*t))` exactly, with sampled peak 0.03999990130403717 N. All other body forces, COM torques and generalized forces are zero. Inputs at index 1,000 onward are exactly zero. Every force point equals the pre-interval satellite COM. The 20 kg satellite starts with zero linear velocity and ends at +0.0020000000000000013 m/s along Z, with displacement +0.05600200000003598 m. The 27-second engine-off interval has zero saved velocity change. No velocity injection is needed to explain these saved values.

Independently summing force, force moments about the world origin, and force dotted with midpoint velocity reproduces the saved impulse, angular impulse and work arrays exactly. Final impulse is `[0,0,0.03999999999999997]` N s; angular impulse about the world origin is `[0,-0.048,0]` N m s; work is 4.0000000000000044e-5 J. Zero torque at the COM is compatible with a nonzero world-origin force moment. Maximum linear balance residual is 5.551115123125783e-17, angular residual 4.163336342344337e-17 and energy residual 6.098637220230962e-20 J, all below their predeclared 1e-9 bounds. The position discrepancy from the continuous analytic solution is 2.000000103669075e-6 m, within the frozen 3e-6 m allowance; velocity error is 5.23599989591967e-10 m/s, within 1e-8 m/s. The 2 micrometre displacement excess is compatible with the disclosed discrete integration/quadrature convention. It is not post-run threshold tuning.

At every saved native state, continuum/rigid position maxima are **0.000415311879 / 0.006725508266 mm**, and rotation maxima are **0.000006189733 / 0.000001312073 degrees**. The last 1.5 seconds contains 751 saved states and satisfies both pose gates. The continuum desired pose remains the predecessor's desired W7 pose, while the rigid desired position follows the satellite-relative offset. This continuation does not repeat or newly certify the original seven waypoint arrivals.

All saved full joint/reference limits, actual and command velocity/acceleration limits, 67 raw torque limits, 0.02 rad tracking/manifold bounds, finite-value gates, and time/handoff gates reproduce as passing. Detailed normalized limits are in `audit_results.json`. Command/reference integration identities and raw/control equality were checked. Static pose reconstruction agrees exactly with recorded EE poses and body positions. `no_hidden_reset` is supported jointly by source inspection, the one-restore metadata, exact initial handoff and full saved replay; saved arrays alone cannot prove the absence of all possible unrecorded software actions.

## Geometry and body evidence

The 29,001 saved whole-machine geometry observations consist of the initial native state and one native plus one interpolation observation for every subsequent interval. Their saved minima reproduce native clearance exactly; the global minimum is **0.002301157545053001 m**, R1 against rigid `back_arm_link` (`collision_0068`) at the final native state. Static kinematics independently reconstruct the maximum geometry interpolation displacement as **2.4184733816302994e-6 m**, under 0.002 m, with 14,500 interpolation states, matching the report. No distance engine was rerun: exact closest-distance values remain producer observations whose indexing, minima, connector arithmetic and source policy were checked. Finite sampling is not a continuous-time nonpenetration certificate.

All 3,406 pair identities across 1,450 control ticks were retained: 4,938,700 pair-time entries comprise 4,465,998 screened lower bounds, 275,444 queries outside activation, and 197,258 selected Jacobian rows. Sphere-bound reconstruction differs by at most 2.220446049250313e-16 m; every omitted lower bound exceeds 0.09 m (minimum 0.0900048131621769 m). The 108,282 queried values at the 0.09 m cap are capped observations, not exact distances. All selected-row identities, Jacobians, right-hand sides and command/event slack arithmetic agree. Minimum filtered and unfiltered slack are both 0.0018274656133463038; no row binds within 1e-6, no geometry projection or SLSQP solve occurs, and no unfiltered row violates the saved constraints. Thus proximity is documented, but the data does **not** show active collision correction or a causal obstacle-avoidance benefit in this continuation.

Body diagnostics cover all 1,451 recorded control states, seven obstacles, and 71 named arm bodies: 721,147 body-distance values and a consistent reported 761,775 underlying geometry queries. Connector lengths reproduce distances within 2.4424906541753444e-15 m, and the saved curvature equals its stated turning-angle/chord-length formula. Terminal commands are explicitly NaN because no following command exists.

| Obstacle | Minimum body distance (mm) | Body | Native index |
|---|---:|---|---:|
| R1 | 2.301157545 | back_arm_link | 14,500 |
| R2 | 36.033935763 | fore_arm_link | 1,620 |
| C1 | 31.853625516 | link_30 | 14,500 |
| C2 | 222.488353999 | link_30 | 0 |
| C3 | 173.009006278 | link_30 | 0 |
| C4 | 258.667500938 | link_19 | 0 |
| C5 | 88.427856748 | link_25 | 14,500 |

The 10 mm display band is diagnostic, not the zero-penetration acceptance threshold. The shape curvature is a geometric diagnostic, not material strain or a stress limit.

## Missing evidence, limits and route

There are no unresolved arithmetic/identity discrepancies among reviewed saved artifacts. The physical blocker remains the unchanged robot momentum gate. The earlier pure software preflight did not predict this all-trajectory momentum failure; its documented qualification was limited to input identity and analytic satellite origin, not a whole-robot `MODEL_VALID` trajectory. No report or rendering should turn its PASS into full pre-execution physical qualification.

Caux supports this exact wording: **In one APP_DEV continuation of the bound 2R5C scene, disclosed satellite-COM force created nonzero satellite motion followed by 27 seconds of recorded inertial coast and accurate dual-arm tracking; force/impulse/energy accounting and the complete saved replay agree, while the original robot momentum acceptance gate fails from the inherited initial state.** It does not support complete all-gate success, long-term goal completion, hardware propulsion, fuel/exhaust modeling, independent learned dynamic control, Diffusion advantage, obstacle-correction causality, distributional generalization or TEST performance. No new neural sample, training, library expansion or comparison was performed.

Route: **narrow the claim and close this bounded trial as failed C1 with supported Caux.** No additional experiment, ablation, retry, threshold change, or physics is authorized by this review. Any future repair requires a separately frozen design addressing the original momentum requirement before execution; do not repair this run by subtracting its inherited momentum or recategorizing it as passing. Parent owns the once-only completed-resume receipt, final media and closeout. Downstream paper writing remains **provisional — no paper claim audit available**.
