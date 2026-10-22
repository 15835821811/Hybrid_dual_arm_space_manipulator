# Independent result-to-claim review

- claim_supported: **yes**, strictly for C1's single static three-obstacle APPLICATION_DEV episode.
- confidence: **high** for the recorded episode under the frozen development acceptance gates.
- C2: **NOT_EVALUATED**.
- pending_external_review: **false**; this is the fresh delegated reviewer judgment.
- paper_claim_audit: **unavailable**. Any downstream paper-writing verdict is **provisional — no paper claim audit available**. Engineering continuation is not blocked.
- overall_long_term_project_complete: **false**.

## What the saved evidence supports

The system actually generated five new proposals using the bound aligned model and fixed seeds 20264611–20264615. Each provenance record reports 16 network calls, and the archived implementation loads the bound model and enforces exactly 16 calls. All five saved 19×17 raw/proposal/prepared arrays match their provenance hashes. The network is the trained 102-dimensional low component with a zero 221-dimensional detail head, embedded in the complete 323 representation; its 65-dimensional condition does not encode obstacles. The generic public bow construction, common full-323 correction and native controller are part of the successful system.

All five prepared proposals fail the screening pose gate and have negative saved minimum clearance. Candidate 04's prepared minimum is −4.249609433656 mm. This is direct evidence that the delivered feasibility should not be attributed to a bare obstacle-aware network. Preparation used scale 1.0 for all five proposals, so this run did not shrink or clip these proposed paths. Candidate 04 and candidate 02 were the two lowest frozen screening ranks, in that order. Each used six correction iterations, within the cap of sixteen. Both reached MODEL_VALID on the unchanged goals and full geometry gates at the 2048-interval validation grid. Candidate 04's validation score 0.00014680070357439502 is lower than candidate 02's 0.010099966751717666 and therefore correctly won the predeclared execution selection.

The honest denominator is **2 qualified / 5 proposed**, **2 / 2 selected repairs qualified**, and **1 execution / 1 planned execution verified**. Candidates 00, 01 and 03 remain SCREENED, in the denominator, and unexecuted. They are not silently relabelled physical failures or successes. There is one ledger attempt, one completed unit, one torque rollout, and no unresolved/inflight work. The recorded 3932 coupled calls and 27500 native-plus-replay steps agree across unit calls, ledger and budget. No replacement candidate, training, dataset expansion, or comparison run was introduced.

The actual execution uses 13750 torque steps at 0.002 s: 27 s motion plus 0.5 s hold, retaining 13751 actual states. Native qpos is 81-dimensional, qvel 79-dimensional, controls 67-dimensional, and INTEGRATION 836-dimensional. The complete initial-state array is exactly the first native and replay INTEGRATION row and hashes to the frozen task's scene initial-state identity. The final selected correction array, spline hash and all archived commanded reference position/velocity/acceleration arrays agree. The control loop in the hash-checked source initializes once before the loop, then writes torque controls and calls mj_step; it does not overwrite qpos/qvel or reset actual state. The saved INTEGRATION slices match time/qpos/qvel/warmstart/ctrl, and all external-force/wrench entries remain zero.

Independent replay starts with a separate reset to the same complete initial state, then applies the saved controls through its own 13750 mj_step calls. The replay loop never copies the recorded qpos/qvel into itself. All 13751×836 full INTEGRATION values agree exactly, as do qpos, qvel, act, time, qacc_warmstart and ctrl. This is a full saved-trace comparison, not a sampled replay check. The reviewer did not rerun physics.

## Recomputed or source-traced main metrics

| Metric | Result | Evidence |
|---|---:|---|
| Whole-body minimum sampled clearance | 0.0015430314478252878 m | Minimum of all 27501 saved native/midpoint scans |
| Geometry sample spacing bound | 0.00011378588585581875 m < 0.002 m | Frozen validator calculation and execution report; not independently re-simulated |
| Final dual-arm position errors | 0.000054484857803028664 / 0.000004930952741399045 m | Recomputed from native ee and frozen goals |
| Hold maximum dual-arm position errors | 0.000054605534054727555 / 0.000004979057139475872 m | All 251 saved hold states, limit 0.005 m |
| Hold maximum dual-arm rotation errors | 0.00011430379364026927 / 0.0000043262582578657355 rad | Recomputed rotations, limit π/180 rad |
| Maximum raw torque / limit | 0.004924068370732259 | All 13750×67 raw torques, 4/80 N·m frozen limits |
| Saturation fraction | 0 | Raw torques equal applied ctrl exactly |
| Maximum velocity / limit | 0.16534963549704565 | Saved native qvel and reconstructed documented 17→67 map |
| Maximum acceleration / limit | 0.05492205529319116 | Differences of decoded native qvel at 0.002 s |
| Maximum command tracking error | 0.000014913734695642944 rad | Saved actual positions minus all 13750 command references |
| Maximum 17→67 manifold residual | 0.000000006254498041008416 rad | Recomputed from native positions and documented map |
| Maximum free-base translation | 0.00578522811677713 m | Native base state |
| Linear/angular generalized momentum peaks | 0.00009197033978254917 / 0.00007716950102547787 | Saved native momentum; below frozen thresholds 0.0010448548789729294 / 0.0013475720008369026 |
| Complete replay maximum error | 0 | Every saved full-state component at every native time |

The execution report's tracking result differs from the command-aligned recomputation only at floating-point roundoff (~4×10⁻¹⁵ rad), due to evaluating the spline at saved accumulated time versus k×dt. There is no material metric mismatch. Geometry scans retain every native state and every interval midpoint; their arrays and JSON records agree, and their endpoint minima exactly match the native clearance timeseries.

## Source and media provenance

All **176 archived source hashes** and **87 bound input hashes** match the frozen run identity; software tests bind the same sources, and ledger identity matches the run identity. There is one precise source-provenance qualification: **producer commit 8456203e1c0f957f02b6932bc54480897a418146 is not by itself the entire pristine source snapshot**. Of the 176 files, 112 match Git bytes exactly and 63 differ only in Windows line endings; `v6_lite/hierarchical_qp.py` contains the pre-existing optional posture-reference patch. Its actual bytes are included and hash-bound in `executed_source.zip`. This class is imported via `v6_lite/__init__.py`; the reviewed episode uses `floating_ff_torque_pd_v1`, not that QP. Reproduction should use **producer commit plus frozen source_hashes/executed_source.zip**. This is a nonblocking documentation qualification, not unexplained source drift.

The independent evidence script records 69 checks: all 68 required evidence checks pass; the additional pristine-Git-content check is false exactly for the disclosed file. `compact_evidence.json` retains that false observation instead of erasing it.

Media source hash equals this run's native_trace.npz. All **24 manifest file hashes and sizes** match. Seven figure sets have PNG and PDF; both views have start/middle/end PNGs. Both MP4s decode as **960×720, 400 frames, 15 fps**, with 26.666667 s playback of 27.5 s recorded motion. The small playback-time rescaling is explicitly recorded, not an extra physics trajectory. The CSV contains **13751 rows**, and every saved time, position/rotation error, clearance and hold flag matches recomputation. The HTML's **400 selected times**, dual-arm positions, goals, errors, clearance and all three obstacle definitions match saved data exactly. Renderer source uses separate MjData and mj_forward on existing qpos/qvel, without mj_step. The HTML clearly identifies its skeleton representation; actual robot and obstacle geometry appear in the MuJoCo video. The reviewer visually inspected the overview endpoint image and the summary figure and found no contradictory or missing result depiction.

## What these results do not support

- Bare-network autonomous avoidance: all five prepared screens show penetration and failed world-pose gates before the common correction.
- A relative Diffusion advantage, causal component attribution, statistical success rate, new-scene generalization or superiority in query/runtime cost. C2 was intentionally not evaluated. The reused public scene/seeds make this an engineering regeneration, not a new independent generalization sample.
- Completed dynamic continuous handoff, seven irregular waypoints with a moving satellite, or the five-obstacle long-term objective. No such execution appears in this run.
- Continuous-time collision certification, hardware operation, contact capture, or production-calibrated guarantees. The saved report explicitly sets continuous_time_certified=false and hardware_validated=false and retains the development acceptance profile.

## Missing evidence and routing

There is no missing run evidence that prevents confirming the narrowly worded static C1. Keep the source provenance qualification in the closeout. A full paper claim audit is absent, so downstream paper claims stay provisional.

Route **confirm this static episode and deliver its media**, then **continue the authorized engineering pipeline** from the verified actual static terminal full state and clock. Freeze the bridge/reference, scene, satellite motion and seven-waypoint dynamic task; run that next bounded phase when its dependencies are satisfied. Do not substitute the dynamic module's default initial state for the true predecessor terminal state. Five-obstacle completion remains a distinct later acceptance target. Do not schedule an advantage benchmark or additional static replication merely to unlock engineering progress. Any future broad claim needs its own actual evidence and fresh scoped review.

Suggested claim: “在固定三障碍静态开发任务中，包含旧模型16步Diffusion提案、公共初始化、全323维共同修正及原生67通道力矩控制的系统，重新生成并完成了一条27秒运动加0.5秒保持轨迹，通过原开发验收门和完整836维独立重放，并交付保存状态来源可追踪的可视化。”
