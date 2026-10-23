# APP_DEV: physical satellite origin and continuous tracking, one unit

Latest explicit authorization is evidence/authorization.json. G1 is resolved.
Run moving_satellite_001 is the only new physical unit. Old runs are sealed.

## Identity and actual continuation

Read the last native state of balanced_brake_001 once as the new input, verify
its hash against the sealed execution receipt, and bind all 836 integration
values, absolute clock (54.5 s), qpos/qvel, controller reference and prior command.
Do not use that package's terminal.npz (which describes its 27.5 s input).
Same model and 2R5C scene; original seven points, rotations and arrival times stay
in predecessor evidence. The continuum holds the predecessor's desired W7 pose,
not its achieved pose. The rigid arm maintains the saved satellite-relative pose.

## Disclosed origin

From local 0 to 2 s, apply at the satellite COM, in world +Z,
F(t) = 0.02 (1 - cos(pi*t)) N. Evaluate at each interval midpoint. Peak 0.04 N,
1000 intervals, discrete impulse 0.04 N s. COM torque zero. All other xfrc and all
qfrc inputs zero. At input index 1000 the force is exactly zero and stays zero
for another 13500 steps (27 s). No qpos/qvel/time writes in the actual loop.
Initialize once by exact full-state restore; independent replay has its own
identical initial restore and consumes saved controls AND full external inputs.

20 kg constant-mass external-force model: analytic speed increment 0.002 m/s,
burn displacement 0.002 m, coast displacement 0.054 m. These are design values,
not measured results. Fuel, exhaust, mass loss and hardware are not modeled.
The legacy all-body no_external_wrench assertion is explicitly split into an
unchanged zero-force robot gate and the authorized satellite burn/coast ledger.
The original robot generalized momentum gate is unchanged, not canceled against
satellite momentum. Linear/angular impulse and work-energy are separate gates.

## Qualification, checks and thresholds

Pure preflight disables mj_step and checks exact handoff, reference continuity,
same model/COM/mass, zero inherited target velocity, force algebra and fixed
sphere separation from an analytic satellite swept AABB. This is NOT a full
robot MODEL_VALID trajectory certificate. The actual feedback continuation is
qualified only by all recorded physical gates and full independent replay.
No new model rollout or neural samples; no training, expansion, TEST, comparisons.

Maintain 5 mm / 1 degree at every saved actual state, nonpenetration, whole-body
geometry interpolation displacement <=2 mm, full joint/reference ranges,
actual and command derivatives, all 67 raw torque limits, 0.02 rad tracking and
shape-manifold limits, original free-base momentum tolerance, absolute time and
full-state replay <=1e-10. The final 1.5 s must hold both goals relative to their
defined frames. Full 29 s origin+coast required, no unexecuted suffix acceptance.

New predeclared absolute tolerances: linear/angular impulse balance 1e-9,
work-energy balance 1e-9 J, analytic position 3e-6 m (covers semiimplicit step
quadrature), analytic velocity 1e-8 m/s, coast velocity change 1e-10 m/s,
input force agreement 1e-14 N. Actual final speed increment >1.9 mm/s and
displacement >55 mm. No tolerance changes after physics. World momentum uses
COM linear velocity and rotated inertia; angular momentum is about world origin.
Force snapshots stored with an interval apply to that interval; integration
snapshots retain the preceding input, including the last burn input at shutdown.

## Cost and stopping

One execution, zero retry. 1450 control ticks, 10 native steps each =14500;
independent replay at most14500; total29000 <63340 remaining. Planned coupled
calls0, cap2000 <12873 remaining. Controller at most1450 SLSQP solves, at most200
iterations per solve, no new random seed or candidate. Geometry scans are
observations, separately counted, not mislabeled native/model propagation.

Unit cap6000 s: implementation/freeze1200 plus worker/review/media4800. Worker
physics hard bound3600 s; read-only review/media reserve1200 s. Unit output1GiB;
reserve192MiB for new readonly delivery/review, binding/source bookkeeping.
Before each control block a conservative uncompressed trace envelope is checked:
64MiB +26000*completed_native +18*pairs*recorded_ticks +240*active_rows + exact compact JSON bytes
+1000*saved_geometry_scans.
Stop before new physics when it exceeds800MiB or active_rows exceeds600000.
Worst topology could exhaust evidence capacity: this is an explicit resource
failure, never permission to omit observations or continue outside budget.
Native count reserved BEFORE stepping; an ambiguous in-flight step prohibits
resume/replay. All new package/run/media/review bytes count against the unit.
Any refusal saves the known prefix; no restart, local retuning or second trial.

## Evidence and delivery

Record every native state and all836 values,67controls/raw torques,command
derivatives,actual EE/Goal SE3,robot base/momentum,force application point and
full force vectors,linear/angular impulses and work, satellite COM/R/v/omega/P/L/E.
Save all geometry pair IDs/bounds/selection reasons, active rows and commands,
and per-obstacle/body distances with exact native indices at every control tick.
Do not claim far lower bounds are exact distances or discrete checks prove
continuous-time collision freedom. Save native scan minima/interpolated states.

After the run: one complete saved-evidence reconciliation, fresh independent
experiment-result-to-claim review of THIS RUN only, one completed-resume identity
check with premanifest persisted first and raw stdout/stderr bytes, then closeout.
No paper claim audit -> provisional writing. No ablation or extra physics.
Actual videos and interactive3D display dual EE/Goal RGB XYZ, accumulated paths,
W1-W7 labels, body distance highlights/connectors; deliver errors,17motion,
67torques,geometry/base/momentum,force/impulse and full836replay curves and CSVs.
No synthetic success animation or physics performed just for media.
