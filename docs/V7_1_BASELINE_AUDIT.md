# V7.1 baseline and reuse audit

Audit date: 2026-10-15, Asia/Shanghai. The task package is a specification,
not evidence that its proposed software or experiments had already run.

## Workspace and preservation

The supplied outer directory is not a Git repository. Its nested repository is
`Hybrid_dual_arm_space_manipulator/`, remote
`https://github.com/15835821811/Hybrid_dual_arm_space_manipulator.git`.
The effective starting HEAD is
`070e7008923bf2642dcecf9806745564388669b9`, on
`v6.4-a-diffusion-planning`. The new branch
`v7.1-lightweight-free-floating-prior` was created at that HEAD without checkout
of an old commit. `main`, old branches and old worktrees were not reset or cleaned.

Pre-existing tracked modifications are `.gitattributes`, `.gitignore`,
`v6_lite/hierarchical_qp.py`, and `v6_lite/run_v6_lite.py`. Large untracked
research directories, data, reports and paper files also exist. These are not
included in the V7.1 commit. The initial status and patch are retained locally
under `artifacts/v7_1/baseline/`; the patch is preservation evidence, not a patch
to apply. No previous data was renamed or counted as V7.1 output.

Windows process inspection at entry found no running `python.exe`,
`pythonw.exe`, MuJoCo or WSL process. No old job was stopped. Existing job status
files were treated as historical records, not proof of a live process.

## Existing expert implementation

The current nested checkout has no `expert_data/`. Git worktree inspection
located it at `E:/v70expert`, HEAD
`7e17f44e165ba7fc3758039d05b15ff80374af11`, branch
`v7.0-multimodal-expert-dataset`. That worktree contains its own uncommitted
changes, including GPU prediction and pipeline work. It was inspected read-only.
Missing old worktree directories were not pruned. Historical commit `e2941462`
was used only as provenance, never as a checkout target.

V7.1 uses the current tracked `model_test/robot_model_spec_v5.py`,
`model_test/angle_convention.py` and `v6_lite/continuum_model_spec.py` directly.
Only the small physical adapter, topology distance policy, complete initial-state
snapshot, feedforward tracker and replay/native-validator cores were adapted
from `E:/v70expert/expert_data/`. Their source hashes are recorded in
`artifacts/v7_1/baseline/reuse_hashes.json`. The new code does not import that
external worktree at runtime and does not copy its RRT/SQP/population framework.
Shared old source functions were not edited. In particular, the dirty
`hierarchical_qp.py` and `run_v6_lite.py` are not imported by the V7.1 pipeline.

## Robot and numerical contract

- Planner coordinates: ten continuum bending components followed by seven rigid
  joint angles, in radians. The continuum components are not polar curvature
  magnitude/plane-angle coordinates.
- `RobotModelSpecV5.planner_to_low_level` is the authoritative 67-by-17 B;
  `low_level_to_planner` is its left inverse. The MuJoCo continuum ordering is
  intentionally distinguished from the old MATLAB ordering.
- The continuum domain is intersected with the current audited shape domain
  `[-1,1]` rad per component. Lengths, mounting transform and the continuum
  `[0.0475,0,0]` m tip offset come from the current shape contract/URDF.
- Runtime scene dimensions are measured as nq=81, nv=79, nu=67. Code resolves
  base, arm, actuator, tip and other free-body indices by names; total scene
  dimensions are not used as hardcoded state slices.
- The base is a free joint, gravity is zero, dt is 0.002 s. Direct torque motors,
  armature, damping and gains match the audited V6/V7 profile: continuum
  kp/kd=100/5, rigid kp/kd=200/20, torque limits 4/80 Nm.
- Fixed bodies are retained for naming. Only explicitly connected rigid
  assemblies/direct articulated neighbors and adjacent two-axis continuum
  modules are excluded by topology. Those adjacent CAD bearing overlaps were
  already present in the old physical profile. No grasp contact allowlist or
  target-satellite exemption is inherited. All exclusions and all 2,860 checked
  pairs are materialized in each run's `pair_policy.json`.
- Reference propagation uses fresh scratch data, current mass matrices, named
  robot-only Mbb/Mba blocks and `solve`, with midpoint `mj_integratePos`.
  MuJoCo free-joint tangents are world translation followed by body-frame angular
  velocity. No Euler-angle integration or explicit matrix inverse is used.
- Full `mjSTATE_INTEGRATION` is frozen, including other scene bodies, time,
  actuator state, control, applied forces and solver warmstart. Nonzero initial
  velocities/momentum or external applied wrench are rejected for this backend.

## Acceptance provenance

The direct user instruction in this chat authorizes the historical
**nonpenetration** override: planning and actual minimum clearance are both
0 m. See `V7_1_EXECUTION_TARGET.md` and the versioned YAML. The initial minimum
8.5406459 mm at `arm_2_wrist_link` / `end_link` is retained as measured evidence;
that pair remains checked. No physical gate was silently weakened.

The 5 mm / 1 degree terminal tolerance and 0.5 s hold / 30 s deadline follow the
V7 task contract. Tracking/manifold 0.02 rad and replay 1e-10 come from the
existing V7 acceptance configuration. Momentum uses separate linear/angular
groups, 1e-5 absolute floor plus 0.002 times internal exchange. Its historical
DEV basis is `E:/v70expert/expert_data/configs/momentum_dev_v2.json` and the
actual dt-halving evidence in `expert_data/output/momentum_dev_001/summary.json`.
These thresholds are frozen before this smoke and labeled inherited DEV;
historical calibration is not counted as a new V7.1 experiment. Production
calibration remains a separate gate.

## Environment

The default PATH Python is 3.14 and is not used. The repository `.venv` provides
Python 3.12.14, NumPy 1.26.4, SciPy 1.11.2, MuJoCo 3.3.2 and PyTorch 2.6.0+cpu.
Pytest 8.4.2 and PyYAML 6.0.3 were installed into this existing local environment;
no global Python environment was changed. The CUDA environment failed loading
`torch._C` because of a DLL dependency. All smoke learning is explicitly CPU.
Exact dependencies are saved in `ff_prior/requirements.lock`.

MuJoCo 3.3.2 on this Windows host failed to open the Chinese absolute pathname.
The adapter therefore passes the repository-relative URDF pathname while being
run from the repository root. It does not hardcode another machine's Python.

## Scope and evidence boundaries

Final measured evidence is in `artifacts/v7_1/smoke_002/`; the first run and its
executed-source archive are retained separately. Tests in
`ff_prior/tests/` distinguish synthetic software fixtures from real native
physics. Each run records source hashes as well as producer commit, compiled
model identity, mapping, initial state, pair policy and configuration. Only
small code/evidence is committed; full arrays, checkpoints and native traces
remain local. No remote push or large production run is part of this task.
