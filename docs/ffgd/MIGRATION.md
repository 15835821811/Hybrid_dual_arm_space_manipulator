# Explicit migration boundary

Base: b0e34e75e94ec53f1cd0a9cde246d4c727a596ce.
Original dirty checkout is untouched; initial status and tracked diff are in
artifacts/ffgd/orchestration. No later branch code has been migrated.

- Runtime imports: model_test.robot_model_spec_v5 and its angle_convention only.
- Robot assets: pinned dual_arm_space_robot_2026 URDF and complete meshes.
- Native numerical profile: ff_prior/robot_adapter.py (implicitfast, 0 gravity,
  2 ms, named mechanical connection excludes, fixed frames, 67 torque motors).
- Planner work domain: continuum ±1 rad from continuum_model_spec; rigid ±pi.
  Native mechanical ranges remain separately available and unchanged.
- New implementation: immutable snapshot/candidate, boundary codec, tree FK,
  mass, recursive inverse dynamics. No old planning/search/bypass imports.
- Reference-only torque tracking is implemented in `ffgd/execution.py`: full
  floating inverse-dynamics feedforward plus the inherited fixed joint PD gains,
  raw/applied torque ledgers, native geometry/limit/warning aborts and independent
  replay. No task-space controller generates a replacement reference.
- Calibrated execution substeps are explicit through
  `execute --run <planning_directory> --config configs/ffgd/execution_substeps_v1.json`.
  This configuration uses 1 ms physics with the original 2 ms control period.
  Full initial state bytes remain unchanged; compiled-model identity checks prove
  only timestep differs. Saved provenance lets `replay --run <execution_directory>`
  reconstruct the same model. Safety is checked at every physics substep.
  The 5 s selected-arm counterexample showed first-order momentum convergence
  under fixed recorded inputs; both arm and free closed-loop regressions passed.
  Original 2 ms angular-momentum failure remains failed, and longer-duration
  calibration and M5–M8 acceptance remain pending.

No external reference source code has been copied. Full-model numerical
calibration is new evidence; historical failures are not reclassified.
