# C.3 interim nominal-refusal evidence audit — 2026-10-08

Verdict: **NO_MATERIAL_DISCREPANCY_IN_SAVED_REFUSAL_EVIDENCE**. Scope is only `c1_mother_01_minus / T_local / C00`, frozen producer `9f39b42775283432eb933f63a9047c488ba22070`. This is a private nominal prediction, **not formal actual execution or independent five-gate validation**. The unfinished stream's registry/selection and all-results verdict remain outside scope.

The sealed common-rule zero plan passed reference precheck, then stopped with `EXECUTION_REFUSED`, `UncertifiedExecutionError`, and `RAMP_MICROSTATE_OUTSIDE_DECLARED_DOMAIN` at **11.080 s**. The rejected ten-step preview's last microstate at 11.100 s has fourth shape coordinate **−1.0000003087215876 rad**, about **3.0872×10⁻⁷ rad** below the original −1 rad bound. The final saved nominal state has coordinate −0.9999588378517785 rad. The receipt records `next_servo_step_executed=false`, `simulation_stop_is_safe_backup=false`, and `continuation_guaranteed=false`; no tool error is recorded. This supports a scientific guard refusal, with no retry or protocol change.

The saved prefix contains **5,540 consumed 2 ms nominal steps**, **5,541 native states**, and **554 consumed ten-step command blocks** through the command beginning at 11.060 s. Every saved servo command equals its consumed block; consumed references bind the same zero plan/definition and frozen initial state. The **555th input/planner invocation at 11.080 s is unconsumed**. Its target matches the refusal diagnostic, and it never enters consumed-command or quality-vector histories. Quality uses 554 command rows and only the saved 11.080 s prefix. `prediction_task_passed`, `online_guards_passed`, and `prediction_admissible` are false; partial metrics cannot qualify as full A or B30 supervision.

| Recorded work | Count |
|---|---:|
| Main nominal physics | 5,540 |
| Private preview physics | 5,550 |
| Preview calls / QP solves | 555 / 555 |
| Native geometry queries | 422,978 |
| Formal actual physics / independent replay | 0 / 0 |

All phase counters reconcile. The extra ten preview steps and one QP solve belong to the refused command and remain counted. Geometry is 251,486 prediction-phase queries plus 171,492 saved-state quality queries. All 14 candidate manifest entries match their SHA-256 seals; canonical plan/evaluator identities, frozen config/source bindings, and trace/failure/quality hash links agree.

An informational logging nuance is preserved: rejected timeline 554 carries previous accepted certificate 553 as context. It is explicitly `accepted=false`, has no servo dispatch checks or torque publication, and is excluded from consumed histories. The metric `command_certificates_checked=555` counts timing rows containing certificates, not consumed commands; there are 554 consumed certificates. This does not invalidate this refusal or authorize the rejected command.

The companion JSON records 55 successful consistency checks and 30 input paths/hashes. The audit only read saved files/arrays and wrote these two audit documents: **zero new physics, geometry queries, training, DDIM, tests, replay, or reruns; no source edits**. It makes no collision, safe-backup, continuous-time, hardware, or completed-study claim.
