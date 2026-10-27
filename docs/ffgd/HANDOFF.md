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
Latest full software regression: 17 passed, 20261027T045009_software_ba87c1.

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
.venv-gpu/Scripts/python.exe -m ffgd.cli plan --config configs/ffgd/plan_profile_v3.json --task configs/ffgd/dev_v1/free_translation.json
V1 run 20261027T043505_planning_7df1ee was REJECTED on task only:
continuum/rigid position errors .29171/.03628 m; all other native gates passed.
Its 60 calls, 18 guide updates, input consumption and untouched candidate were recorded.
Root probes 20261027T044229_guidance_diagnostic_28d753 and
20261027T044506_guidance_diagnostic_dcf601 showed the .1 normalized step cap
was weak; larger raw steps create acceleration spikes. The DDIM clean-to-state
gain is .057-.857 before the last step; network energy gains offset the guides.
V2 uses the SAME seed 13001 and frozen task/weights, 60 network calls, 18 guided
steps, max 3 updates each, max 2 line trials. Full-rank acceleration metric weight
10, energy-scaled Polyak step, normalized cap 1, physical control cap .1 rad.
All updates are in denoising; no postprocessing. No candidate has passed yet.
Inspect actual native validation, denoising trace and timing; diagnose failures,
then proceed to the eight frozen DEV tasks. Do not add arbitrary seeds.

Independent validate/replay, selected execute and partial architecture-audit
CLI are implemented; they still need real learned-plan evidence. Architecture
contract/order/execution-policy gates remain NOT_RUN until reviewed/tested.
TorchScript physics passed 32 native value cases and a full 323-variable,
all-energy gradient case in 20261027T044747_compiled_calibration_3c3723.
It is ENABLED only in V2 config; default backend stays eager. Compilation time
is recorded. V2 live status/PID must be read from state, not inferred here. benchmark/report/resume still explicitly fail as unimplemented.
No final-scene or unseen-task experiment has run. No unattended agent promised.

V2 completed REJECTED, task-only: .0330/.0277 m and .1946/.1294 rad.
All other native candidate gates passed. No learned plan has passed yet.
Task-Jacobian root experiment 20261027T050930_guidance_diagnostic_79094a:
12 singular values range .3104 to 58.0644. Acceleration weight1000 plus
Jtask.T@Jtask reduces task energy126.8 to .293 in a bounded one-step ray,
max acceleration ratio .8373. This diagnostic returned/executed no plan.
Native residual/Jacobian calibration 20261027T051107_guidance_diagnostic_3f0055
passed 8 directions x3 epsilon, native value error1.26e-13.
Compiled residual-Jacobian run 20261027T050212_guidance_diagnostic_fbff1a
exceeded180s and failed; verified-owned PID36004 was terminated at~422 CPU s.
Eager completed derivatives26.8s. Task metric now requires eager backend and
checks budget between every residual row; no compiled Jacobian retry assumed.
V3 same seed13001/task/checkpoint: 60 network/18 guided, at most3 updates and
2 line trials each; full-rank accel1000+task-J metric refreshed each guided
noise level, reused only within its3 updates; Polyak2, normcap2, physicalcap.16.
No postprocessing or final-step extra work. Before launch verify software tests.

SO(3) near-pi derivative cancellation fixed using symmetric-axis extraction;
20261027T051120_software_0ceae3 remains FAILED; regression
20261027T051241_software_f98e5c passed all19 software tests.
V3 is authorized to launch; read live state for actual launch/completion.
