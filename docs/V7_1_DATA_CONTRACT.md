# V7.1 data and validation contract

## Authoritative representation

The single decoder is `ff_prior/spline.py`: cubic clamped B-spline, 24 controls,
degree 3, 28 knots. `Z[19,17]` contains 18 internal controls plus one shared
terminal vector. Decode reconstructs three equal initial and three equal
terminal controls. Position, velocity and acceleration come from the same
SciPy BSpline and its analytic derivatives, scaled by T and T². No fourth
fixed endpoint control creates an extra dwell. The 128-point training view is
derived data, never the geometry resolution or authoritative trajectory.

Five bounded proposal families are provenance labels, not demonstrated mode
classes. No RRT, high-dimensional SQP or actual rollout enters generation.
Durations are selected from 10/15/20/25/29.5 s by derivative-control bounds;
the explicit hold is 0.5 s. Every attempt, rejection and interruption consumes
its proposal budget. There is no success quota or unbounded refill.

## Conditions, targets and identity

Public conditions contain only fixed initial q17 features, two world poses,
duration and complete pose masks. The complete initial-state identity is also
bound through task/checkpoint metadata. The network uses world positions and
the first two rotation-matrix columns (6D), so quaternion sign equivalence is
preserved. Extra future/witness/terminal-base fields are rejected.

Private terminal seeds are saved only in provenance. A synthetic reference
first passes coupled propagation; its predicted dual terminal poses then become
`goal_source=coupled_forward_generated`. This is label self-consistency, not
success at a previously specified goal. Physical audit and benchmark tasks are
hashed and frozen before use. Planning, tracking and replay never relabel a
failed target. Online structured initialization uses only the public pose
error and initial generalized Jacobian; the complete coupled path is used for
guidance thereafter.

Each record identifies sample/root/family/parent, seed and proposal family,
producer commit plus exact source hashes, robot/scene/mapping/pair/config hashes,
full initial-state identity, reference hash, and final split. JSON records are
drafts until split freezing, then immutable. An atomic sorted JSONL index is
rebuildable. Arrays are per-sample lossless NPZ shards. Failure ledger and
native execution arrays are separate from reference arrays.

## Levels and profiles

MODEL_VALID requires finite data, identity, declared continuum/native ranges,
boundary derivatives, motion bounds, time/hold, coupled numerical convergence,
full-body geometry, separate geometry refinement and valid forward labels.
128 vs 256 intervals (at most 512) must meet 1 mm terminal/body-extent and
0.2 degree terminal rotation convergence. Geometry also checks intermediate
states and an explicit 2 mm maximum sampled finite-volume movement budget;
this remains a dense-discrete audit, not continuous-time certification.

Reference torque feasibility is explicitly NOT_EVALUATED and is not a required
gate of the development model-prior profile. A release requiring it must not
treat that missing check as passing. MODEL_VALID never implies execution.

The default main `data_role` is FF_MODEL_PRIOR. Rejected proposals retain that
intended role with REJECTED validation and no training eligibility. Executed
DEV evidence is separately EXECUTION_AUDIT, training_eligible=false, with
execution_status PASS/FAIL and its own validation level. Only the individually
executed, independently replayed and fully accepted sample can carry
EXECUTION_VERIFIED evidence. Unaudited records keep MODEL_VALID.

The user's current nonpenetration authorization sets both planning and native
geometry thresholds to 0 m. Native verification uses the actual 67 joint
coordinates, never a B+ projection for geometry. B+ is diagnostic only. The
tracker uses per-physics-step analytic references, free-base feedforward and
67-channel PD, unclipped and clipped torques, ctrl and mj_step. It does not
replan a task. Trace state[k+1] follows ctrl[k] applied to state[k]. Replay
starts with a fresh data object and the same complete initial snapshot, then
applies stored ctrl without copying per-frame physical state.

## Splits and training

Lineages are grouped before training. Same goal neighborhoods (both positions
within 1 cm and both rotations within 1 degree), near-duplicate path descriptors
and nonempty scene families merge transitively. Empty scene IDs do not merge
all roots. A group touching independent DEV is quarantined to DEV before
training. Other groups use stable 80/10/10 hash allocation. Revisions and
retimings inherit lineage; final TEST is not used for development.

Separate trajectory and condition scalers fit TRAIN only; they save schema,
robot and initial-state hashes. The original smoke_002 Transformer predicted
epsilon. The 2026-10-15 repair predicts velocity v on the same 19 free tokens
with hidden 128, four layers and four heads. VAL v MSE selects the new checkpoint;
it is not directly comparable with historical epsilon MSE. Old checkpoints
retain their explicit or legacy prediction interpretation.
Weights, optimizer, RNG and update count are saved for resume; schema/robot/scaler
mismatches fail closed. Synthetic test-fixture training is not robot evidence.

The repaired sampler keeps DDIM state and affine transforms in float64 while
network inference remains float32. Guidance and clean projection rederive
epsilon at the same noisy state. The final guided physical array is returned
without a float32 round trip; physical-domain tolerances are unchanged.
A temporal SVD projection fits only TRAIN control deviations from the public
initial q17, retaining 0.9999999 energy. It acts on clean estimates inside DDIM,
is bound to TRAIN sample/array/robot/initial hashes, and uses no private endpoint
or DEV witness. It is a proposal regularizer, not a model or execution gate.
All projected outputs must still pass the complete frozen physical contract.

DDIM makes exactly the requested network calls. In the last two reverse
iterations, SUBSPACE_FD_DEV_ONLY guidance evaluates eight centered directions
(16 full coupled path costs), plus base/trial/step-halving evaluations. Costs
include dual world-pose errors, all-path geometry and joint/motion bounds.
Terminal shared variables remain in the directions. It is neither a complete
exact gradient nor a production backend.

## Protocols and budgets

Main priors have sample-level admission with no five-candidate minimum. The
explicit strict subset retains same-task hash, five distinct MODEL_VALID paths,
independent roots and complete frozen-pool processing. Copies, retimings and
same-root revisions do not increase its quota. The gated study interface takes
a frozen TEST archive and strict-pool manifests; no study is run by smoke.

Random and risk audits use separate independent DEV families, one sample per
family, frozen before execution, with no failure replacement. Benchmark tasks
come from other reserved DEV families and are frozen before any tested method.
All three methods get the same public task, five candidates and the same verifier.
One candidate is selected by model gates and pose error before execution.
Failures, errors and missing candidates remain in the predefined denominator.

The run lock is exclusive. Identity conflicts fail. Resume preserves completed
proposal IDs, cumulative attempts/wall/CPU/call counts and interrupted work.
There is no automatic pilot, study, strict subset, final TEST or production.
Historical data import defaults off. Model-prior and executed-expert costs
are reported separately; no equal-quality speedup is inferred.
