# FFGD handoff

Goal ACTIVE. M0-M3 passed their scoped implementation exits. M4 RUNNING.
Overall architecture/engineering/learning-benefit gates remain NOT_RUN.
Workspace E:/ffgd-core, branch codex/ffgd-core-rebuild, pinned base
b0e34e75e94ec53f1cd0a9cde246d4c727a596ce. Original dirty checkout is preserved;
its tracked diff was independently compared and unchanged. No push.

CPU: .venv/Scripts/python.exe; GPU: .venv-gpu/Scripts/python.exe.
Python 3.10.19, MuJoCo 3.3.2, torch 2.6.0+cu124, RTX 4080.
Read state.json, live PID/create-time/command and latest checkpoint before work.
Source/config snapshots and all failures are retained per run.

M2: 256 dynamics comparisons; 64 geometry-pressure cases; full-energy and
323-variable gradients; 9 native executions/replays (21000 steps).
One nonsmooth finite-difference feature switch is separately diagnosed; its
failed run is unchanged. Physical calibration was on small 2-second references.
Latest full software regression: 16 passed, 20261027T042907_software_94339f.

M3 data: artifacts/ffgd/runs/20261027T034523_prior_data_d6601f:
2000 valid / 2711 attempted, TRAIN 1604 / VAL 396, family-disjoint, rank 323.
586 acceleration, 91 collision and 34 incomplete-refinement rejections remain.
Data hash: 065ec2aed993a47403e2947a52e3f81f12d3a4055976582889b2993fcc6f8217.
Training: 20261027T043312_training_711731, 3000 updates, VAL v-MSE 0.346681.
Best checkpoint: artifacts/ffgd/runs/20261027T043312_training_711731/best.pt
SHA256 b80ce0d14e2c95bde6191f7f0e0a0ecd8b65c8c19f5883b0b889dcb8eccfbf0f.
Learning checks: 20261027T043411_learning_checks_85441c, five full-noise
60-step samples; reload/repeat errors zero, boundary residual <=1.3e-12.
Endpoint position errors 0.073-0.474 m: these are NOT valid executable plans.
Prior covers zero velocity/momentum starts only. No learning benefit established.
The serial M3 queue m3_queue_99464 finished; do not assume its old PID is alive.

Current M4 command:
.venv-gpu/Scripts/python.exe -m ffgd.cli plan --config configs/ffgd/plan_profile_v1.json --task configs/ffgd/dev_v1/free_translation.json
One fixed seed, 60 network calls, last 18 steps guided, one update per step,
max two line trials. All steps audit physical energies, no postprocessing.
Inspect actual native validation, denoising trace and timing; diagnose failures,
then proceed to the eight frozen DEV tasks. Do not add arbitrary seeds.

Independent validate/replay, selected execute and partial architecture-audit
CLI are implemented; they still need real learned-plan evidence. Architecture
contract/order/execution-policy gates remain NOT_RUN until reviewed/tested.
compiled_physics.py is an UNTESTED optional performance experiment, DISABLED
in production. benchmark/report/resume still explicitly fail as unimplemented.
No final-scene or unseen-task experiment has run. No unattended agent promised.
