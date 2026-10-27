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
- Planned later migration: frozen reference-only torque tracking and necessary
  safety protections; not yet integrated, not execution evidence.

No external reference source code has been copied. Full-model numerical
calibration is new evidence; historical failures are not reclassified.
