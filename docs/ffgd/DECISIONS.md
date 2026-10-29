# Mechanism decisions and retained failures

- 2026-10-27: preserve exact pinned robot, native free-base state and all 67
  torque motors. New modules depend on the parameter authority only.
- Boundary nullspace uses deterministic QR/elimination in flattened coordinates.
  No low-rank latent truncation. The 24-point zero-derivative template has 323
  free scalars; masks and nonzero derivatives determine other ranks.
- Full body Jacobians and recursive Newton-Euler bias produce differentiable
  mass and inverse dynamics, including armature and damping.
- World physical momentum about the fixed origin is mapped to MuJoCo base
  tangent before each midpoint solve; target satellite is excluded.
- Enclosing mesh boxes covered all vertices but produced 12 false-unsafe home
  pairs (worst -86.7 mm versus positive native clearance). No pairs removed.
  SAT boxes remain a coverage diagnostic. The initially tested native witness VJP
  failed two of 64 whole-path checks; CCD tolerance and support-search root-cause
  experiments did not repair it. Production now uses differentiable supports of
  every native convex hull vertex, with 48 separating-axis candidates. Native
  distance is retained for independent validation. No full-path finite differences
  are used as production gradients.
- Root-cause experiment: mesh/box witness normals had exactly opposite finite
  difference signs, including outside contact. MuJoCo 3.3.2 nativeCCD returns
  type-sorted witnesses before the legacy unflip branch. Explicit ordering and
  restoring pair order fixed all 3 x 8 x 3 geometric derivative probes.
  Source: https://raw.githubusercontent.com/google-deepmind/mujoco/3.3.2/src/engine/engine_support.c
- Compiled sameframe shortcuts must be applied to inertial/geometry frames;
  using tiny source offsets ignored by native kinematics introduced 5e-8 m
  disagreement. Matched compiled semantics; physical parameters unchanged.
  Source: https://raw.githubusercontent.com/google-deepmind/mujoco/3.3.2/src/engine/engine_core_smooth.c
- The distance query cutoff is explicitly capped. Native max-float output is
  not a measured clearance. Feature switches remain piecewise differentiable;
  a degenerate zero-distance witness returns an explicit numerical failure.
- Numerical profile v1 was written before execution calibration. Absolute
  momentum budgets do not depend on the run's action amplitude. Long-horizon
  and final continuous-chain suitability is NOT_RUN.
- Initial writeable-array binding and NumPy-bool JSON errors remain failed
  work units; later successful runs do not rewrite them.

- The 64-case convex-support gradient run retained one large-step discrepancy.
  Frozen case 51/direction 3 switches support vertices 226/234 at epsilon 1e-4;
  smaller feature-stable steps converge to AD. This is a documented nonsmooth
  case, not a relabeled PASS of the failed run. A subsequent 323-variable suite
  passed all active energy directions. All 64 geometry-pressure cases had no
  observed false-safe pair; this is empirical, not a global guarantee.
- Initial data pilot is restricted to 4/5-second zero-momentum starts within
  numerical-profile scope; training data are FF_MODEL_PRIOR and every accepted
  path passes the independent native refined validator. All rejected controls
  and identities are retained. Higher-dimensional and nonzero-start training
  coverage is a later milestone, not inferred from a small pilot.

- 2026-10-29: Preserved final approach0ebf8d rejected with known unsafe geometry/derivatives. Adopt optional bounded quadratic second trial (7f62c6 native decrease) and optional collision time peak (beca9b contact-domain contrast;9d51c0 scalar/AD subchecks), inside existing denoising budget. Legacy defaults remain; no acceptance/execution change.9d51c0 composite remains FAILED at0.418mm versus0.4mm on unsafe reference. New moving/final configs require actual sampling and execution regression before any full architecture or engineering claim.

- 2026-10-29: After final224733 rejection, task-only instantaneous curvature probe865ce0 is rejected; combined task/support-gap SPD metric f34524 gives native435.216→222.684 without a candidate. Admit optional instantaneous_multispace ONLY for new-noise regression, refreshing inside each guided step from its current full-state prediction. Exact whole-path gradient and readonly gates unchanged; no diagnostic path input. Added local FK/momentum/geometry FD and network-consumption tests pass. Full engineering remains incomplete.
