# FFGD handoff

Goal ACTIVE. M0-M3 passed scoped exits. M4 RUNNING. M5-M8 stage gates NOT_RUN.
Overall architecture/engineering/learning-benefit remain NOT_RUN; one vertical
DEV success does not complete any later milestone.

Workspace E:/ffgd-core, branch codex/ffgd-core-rebuild, pinned base
b0e34e75e94ec53f1cd0a9cde246d4c727a596ce. Original dirty checkout is preserved;
tracked diff independently compared unchanged. No push/cloud/deletion.
CPU .venv/Scripts/python.exe; GPU .venv-gpu/Scripts/python.exe.
Python3.10.19/MuJoCo3.3.2/torch2.6.0+cu124/RTX4080. One heavy worker.
Read MASTER/state, actual PID/create-time/command and checkpoint each session.
`python -m ffgd.cli resume` now verifies identities and recovers stale bookkeeping;
it never blindly retries a failure or starts a background agent. Foreign host or
AccessDenied is unverifiable, not assumed dead. Source/config snapshots per run.

M2 evidence: 256 dynamics comparisons,64 geometry-pressure cases, full323-variable
gradients,9 native executions/replays(21000steps). A nonsmooth support switch was
separately diagnosed; old failed runs unchanged. Native witness-gradient approach
was retired in favor of differentiable full convex-support geometry. All native
geometry remains independent. Numerical profile v1 is frozen; absolute momentum
budgets .001 each, physical geometry0, goals5mm/1deg, replay1e-10.

M3 data20261027T034523_prior_data_d6601f:2000valid/2711attempts,TRAIN1604/VAL396,
family-disjoint/full rank323.586acceleration,91collision,34incomplete-refinement
rejections retained. Hash065ec2aed993a47403e2947a52e3f81f12d3a4055976582889b2993fcc6f8217.
Training20261027T043312_training_711731:3000updates, bestVALvMSE .346681.
Best checkpoint artifacts/ffgd/runs/20261027T043312_training_711731/best.pt
SHA256 b80ce0d14e2c95bde6191f7f0e0a0ecd8b65c8c19f5883b0b889dcb8eccfbf0f.
Learning checks20261027T043411_learning_checks_85441c:5new-noise60-step samples,
finite/full323/boundary-valid/reload-repeat identical. Not all model-valid.
Prior covers zero velocity/momentum starts. No learning benefit established.
Old M3 queue FINISHED; no live queue or unseen agent is assumed.

M4 first fixed task free_translation, seed13001, same checkpoint:
V1 20261027T043505_planning_7df1ee REJECTED task only(.2917/.0363m).
V2 20261027T045128_planning_5746f7 REJECTED task only(.0330/.0277m,
.1946/.1294rad). All other native candidate gates passed. Failures remain.
Root experiments showed weak capped raw gradients and task Jacobian singular
values .3104..58.0644. Full-rank acceleration1000 + Jtask.T@Jtask metric,
Polyak2, normcap2, physicalcap.16 are used INSIDE last18/60 denoising steps,
max3updates and2line trials each. No last-step extra optimization or repair.
Jacobian includes whole floating-path propagation, never terminal-only IK.
Native 8directions x3eps Jacobian calibration20261027T051107_guidance_diagnostic_3f0055
passed(valueerror1.26e-13). Near-pi SO3 derivative cancellation fixed; old failed
test20261027T051120_software_0ceae3 retained, subsequent tests passed.
Compiled residual-Jacobian20261027T050212_guidance_diagnostic_fbff1a exceeded
180s; verified-owned PID terminated after~422CPU s. Task metric REQUIRES eager.

V3 20261027T051345_planning_cb9d2e PASSED all native model gates,1177.7s:
position .00123395/.00009262m;rotation .00190452/.00011926rad;
clearance .00846159m, nominal torque ratio .079523,1045validation gridpoints.
Candidate048e47afeea36931ab734ddc16052813b16b2b0c54883c95d3baf9bef3f9ac67.
Native execution20261027T053343_selected_execution_312958 PASSED2000steps:
actual positions .00122838/.00011053m, no clipping/abort/Cartesian assist,
P/L residual .00030336/.00034808, maxmanifold residual8.08e-7.
Independent replay in execution AND separate CLI20261027T053550_independent_replay_5a0256
passed with state difference0 and single initial restoration. This measured
nonzero manifold residual still needs a supported M6 continuation policy.

Batch-four VJP calibration20261027T053430_guidance_diagnostic_1068c3 PASSED:
derivatives16.8s vs26.8s scalar; originalclean/gradient identical, one-step
proposal difference4.28e-14, native Jacobian check passed. Evidence summary
orchestration/batched_jacobian_equivalence.json. Main config now
configs/ffgd/plan_profile_v3_batched.json, same mathematical budget/seed.

CURRENT: frozen arm_body_box planning20261027T053742_planning_228010,
PID90188/create1793079460.9813397; VERIFY live identity, not this stale text.
Command .venv-gpu/Scripts/python.exe -m ffgd.cli plan --config configs/ffgd/plan_profile_v3_batched.json --task configs/ffgd/dev_v1/arm_body_box.json
Its previous setup20261027T053603_planning_f09cb1 failed before sampling because
strict input audit misclassified NPZ w+b creation. Fixed only O_TRUNC files
inside fresh output directory; historical input reads still fail. Failure and
requested1/sampled0 denominator retained. Three guard regression tests passed.

Strict runtime sentinel now records file reads, actual function code hashes,
module identities and denied legacy calls; permits checkpoint+identity sidecar.
IO preflight20261027T052001_software_7ff773 passed, earlier sidecar allowlist
failure20261027T051927_software_e514d0 remains. V3 free run predates file audit,
so cannot retroactively claim it contains that gate. New runs include query
counters for energy/mass/pair/backward work. Full ARCH audit still NOT_RUN.
Architecture audit implements static interfaces/rank, semantic filename/order
physics+network comparison, counterfactual terms and execution safety separation.
Latest full software regression20261027T052121_software_b600a8:25passed;
after that recovery2tests and native budget-abort/replay test passed. Execution
now saves atomic partial inputs/states every500steps and final complete ledger
on budget abort; interrupted partial replay never implies completed execution.

NEXT: inspect current obstacle candidate, classify actual failure if any; if
valid execute/replay it. Continue all8frozen DEV then40, complete real ARCH01-10
(including filename/order and active obstacle counterfactuals). Re-run free task
under the strict audited version for complete architecture/repro evidence.
Do not change gates, add blind seeds or relabel old failed versions. M5 needs
20 frozen tasks, M6 hold/moving target/full nonzero state, M7 final combination,
M8 matched-budget research/clean reproduction. benchmark/report unimplemented;
resume implemented as verified recovery. No final scene has been run.

IMPORTANT: current one-seed V1/V2/V3/arm runs are diagnostic versions only.
MASTER6.1 requires at least5distinct noise candidates per evaluated task; full
M4 cannot pass on these single-seed successes. Freeze the5-seed group after
root corrections, retain all5 outcomes, then execute task-level selections.
Current arm run's untrained scene token was identified as a train/inference
schema mismatch: data had zero external-box tokens. Prepared correction uses
initial-state/timed-pose tokens only, with ALL scene geometry still in every
guide and native validator. MASTER6.1 explicitly permits this split.
Current process retains its archived old schema; do not relabel it.
condition_diagnostic_v1.json is prepared (NOT_RUN), compares same cold seed
with/without untrained scene token against identical physical task/scene.
New scene contract regression passed2 tests. Latest full software run
20261027T054101_software_ee192b passed29 tests before this correction.

UPDATE: old-schema arm run20261027T053742_planning_228010 completed
INCOMPLETE_VALIDATION, velocity/acceleration false, geometry step .00470m at
2048intervals though integration error1.1e-6m. All60network/54guide gradients
were recorded. No execution. Condition diagnostic20261027T055742_condition_diagnostic_e0c337
completed: TRAIN external-scene tokens0; same noise, true-token naked energy
13185 and accelratio1.293 vs corrected state/pose-only energy2960 and
accelratio.581. Corrected prior collision energy.0286 with all2938pairs, so
this is an active obstacle pressure task. No gate changed.
Next/current command uses plan_profile_v4.json and arm_body_box.json, same
seed13001/budget/guide weights. Check state for actual launch/PID.

V4 actually started20261027T055956_planning_f9149f PID204764/create1793080794.5176945.
Verify status. First guided steps reduce task energy strongly but collision
penalty remains active. If final native geometry rejects, a bounded diagnostic
CLI `weight-diagnostic --config configs/ffgd/weight_diagnostic_v4.json` is
prepared (NOT_RUN), compares one step at the immutable final guided vector
under4constraint weights; never returns/executes a repaired candidate.
If V4 passes all native gates, skip that unnecessary diagnostic and execute it.

UPDATE: V4 20261027T055956_planning_f9149f completed REJECTED on geometry
only: minimum native clearance -0.006507m. All world task/velocity/acceleration/
actuator/momentum/boundary gates passed. No execution. Weight diagnostic
20261027T061922_weight_diagnostic_b9f955 failed on numpy.bool_ JSON serialization
after computing derivatives; cast fixed, same diagnostic rerunning (read state).
Never use its one-step rays as repaired plans; only calibrate the next cold run.

Weight diagnostic20261027T062126_weight_diagnostic_9b75c6 PASSED native scalar
relativeerror1.23e-14. At the fixed failed guided point, weight1 rejected both
line trials. Weight10 accepted: task .02192->.01755 and collision .002286->
.0000505; acceleration hullratio .648. Weights100/1000 were less effective
in this one-step comparison. Next V5 changes ONLY collision guide weight1->10,
keeping joint/actuator1 (their gradients were zero in the probe), same cold
seed13001/task/weights/checkpoint/60steps18guides3updates. Gates unchanged.
Never execute the diagnostic ray. Run plan_profile_v5.json on arm_body_box.

V5 actually running20261027T062443_planning_00217f PID197992; verify live
state. At step47 collision penalty.001856, task.1433, not final validation.
Prepared guard-diagnostic config guard_diagnostic_v1.json (NOT_RUN) compares
profile vs call-only trace auditing on identical physical energies/gradients.
Default is STILL profile. Seven software guard tests passed, including
already-loaded forbidden calls in both backends and whole-experiment abort
after an architecture violation instead of treating it as a failed seed.
After V5: if valid execute/replay; then consider guard performance diagnostic
before the minimum-five-candidate evaluations. Do not run competing heavy jobs.

Full software regression20261027T063930_software_dd8049 passed34tests. Guard performance diagnostic now includes exact batched task Jacobians as well as full energies/gradients, four reversed-order passes,300s bounded budget; still NOT_RUN and default backend remains profile.

V5 20261027T062443_planning_00217f completed REJECTED geometry only(-.002715845m), all other gates passed; no execution. Guard diagnostic20261027T064409_guard_diagnostic_f27777 passed with exact energy/gradient/J agreement. Profile23.87/23.97s vs trace23.05/23.23s, only about3% overall gain; keep default profile for now. Current/next collision-diagnostic_v5 reads immutable candidate and compares coarse54 and dense1046 native minima and same-pose proxies. Check state for actual PID/result.

Collision diagnostic20261027T064608_collision_diagnostic_96a479 reproduced V5 dense min exactly. Coarse54point min-.001017 at2.34375s; dense1046 min-.002716 at2.28027s, pair97/154. Same-pose proxy conservative (-.003862 at dense worst), so temporal undersampling plus incomplete descent, not hidden geometry. Step diagnostic20261027T064811_guidance_diagnostic_ce44cb reconstructed last-step cached task-J metric: coefficient6.2197 rejected(total.07568);1 accepted(total.01658 vs .02075, task and collision decrease). Six bounded rays were saved; old manifest hardcoded five-ray label is a metadata defect, not deleted; CLI now derives actual ray count. V6 changes only max_step_coefficient=1, same seed13001/task/grid/weights/60x18x3 budget. Seven diffusion software tests passed including nonzero energy floor. Run V6, then read-only native validation; no diagnostic ray execution.

V6 started20261027T065037_planning_cac48b, actualPID184416/create1793083836.2305388; verify live state. Prepared exact base-mass-row optimization, default OFF. One native-model matrix/VJP equivalence test passed. Full whole-path gradient/Jacobian/timing diagnostic configs/ffgd/mass_rows_diagnostic_v5.json is NOT_RUN; run only after current planning finishes. No main backend accepts this mode yet.

Full software run20261027T065527_software_557c0b passed36tests. Read-only report CLI implemented and tested:20261027T065931_evidence_report_f8c350 inventories all new FFGD units and all requested planning candidates, including setup failure and unstarted seeds, grouped by source/checkpoint/budget without pooling version successes. It is not final engineering or learning-benefit acceptance. Native validator now records worst pair/time and candidate timing is split into per-candidate/cumulative fields for future runs.

V6 20261027T065037_planning_cac48b ended geometry-only REJECTED(-.000919343m), all other gates passed; no execution. Last-step all3updates accepted. Exact mass-row diagnostic20261027T070948_mass_rows_diagnostic_91900c PASSED: values/grad/J difference about2e-17,24.3s full vs22.4s rows. Backend eager_mass_rows now available, preserving full67 actuator matrix. V6 geometry diagnostic20261027T071258_collision_diagnostic_ad50a3 reproduced failure: guide native minimum+.000664m, dense-.000919m at2.265625s between2.1875/2.34375s. Proxy remains conservative. Grid64 diagnostic20261027T071406_guidance_diagnostic_cdc6ad PASSED native task-J8directions3eps(value2.82e-13, rel~1e-8); same bounded one-step decreases all active terms. V7 uses uniform64 guide intervals, same60steps18guides3updates/weights/cap1/coldseed13001; exact mass-row optimization only changes contraction order. Wallbudget2700s covers more grid work, not more optimization. Run V7 then native validation; independent execution/replay only if valid. No candidate from diagnostics executed. All one-seed versions remain diagnostics, not five-candidate protocol.

用户2026-10-27追加的七条架构准入规则已合并到现有MASTER1.1和AGENTS，state记录新hash，事件保留旧hash。继续M4，不新增审计工程框架。当前失败根因：V6在相邻引导时刻之间穿透；通用修复：统一64段引导网格和已校准的等价质量矩阵行计算；架构是否改变：否，仍为60步扩散/18步内引导/最多3次更新，无后置恢复；回归证据：原生Jacobian与质量矩阵/梯度对照已通过，V7实际链路、原反例结果和相关任务族回归尚未完成，不能认定修复通过；下一动作：核对V7 PID160920/create1793085358.2235348及结果，完成实际架构审计和同版本相关任务族复验，合格候选才执行/重放。

37项完整软件回归20261027T072348_software_f69f61通过，不能代替真实架构与任务族回归。仅在现有guidance-diagnostic中准备隔离的编译Jacobian根因试验(compiled_jacobian_diagnostic_v2.json，NOT_RUN)：此前编译后J停滞，本次只检查关闭JIT执行器优化是否消除停滞，仍保留完整物理图和原生8方向3epsilon对照；主采样器继续禁止compiled+task_metric。不得与V7重型worker并行，失败不启用该实验后端。

失败根因/准入缺口：旧入口的运行哨兵未覆盖先于plan执行的任务/机器人加载。通用修复：将同一个哨兵前移到算法导入和公开输入加载，白名单仅增加显式配置、任务、profile及声明初态snapshot；不新增规划包装/补救链。架构是否改变：否，只扩大已有检查覆盖。回归：真实加载20261027T073525_software_3fb866通过；9项哨兵回归通过，包含加载阶段禁止调用直接中止。新增测试曾因错误假设报告排序出现1失败，已在events保留并修正测试定位，不涉及物理结果。ARCH01以后检查实际load_problem/Robot调用记录；V7运行早于此改动，不追认为覆盖加载阶段。下一动作：完成V7反例结果，再用更新入口做同版本任务族及架构回归，禁止提前升级M4。

失败根因：V7 20261027T071559_planning_e540d2仍几何拒绝(-.000530125m)，最坏2.265625s已在64段引导网格，其他门全过，无执行。停止仅加密网格/微调的方向，检查2938几何对平均导致局部违规稀释，下一步在通用五项标量代价内验证平均+峰值聚合，不能添加场景补丁或末步优化。架构是否改变：职责不变。原反例和相关任务族回归仍待通过。入口冷启动首次GPU检查20261027T074349_software_cbd388失败(误拒Python标准库zip探测)，仅放行当前解释器精确归档路径后20261027T074702_software_c7a2cc通过；11项哨兵回归通过，并保证吞掉异常也会在退出时拒绝整次规划/清除选择授权。旧失败保留，原生硬门未改。先运行已准备的有界compiled-J诊断以减少梯度成本，然后进行新聚合的数值校准。

根因：V7局部碰撞被2938对平均稀释。通用修复：可选pair_mean_plus_max，保持五项标量代价及去噪内预算；新代价单位权重1。架构职责不变。回归：20261027T075112_full_space_peak_gradients_d9e987完整323维/三例/八方向/三epsilon原生对照通过；V7固定点单步诊断20261027T075301_weight_diagnostic_d832a5碰撞.1251降至.00961，非可执行计划；41项软件20261027T075310_software_d6ef67通过。编译Jacobian关闭JIT优化诊断20261027T074913_guidance_diagnostic_84b6d9通过，主后端仍eager_mass_rows。下一动作V8相同冷seed复验原反例，再相关任务族和实际ARCH，未跑门NOT_RUN。

V8原反例已启动20261027T075858_planning_e7658f，PID165984/create1793087938.3622732，核实真实身份后等待；未完成勿判通过。等待期间在ffgd.baselines实现C/D独立比较入口与严格能量/梯度/Jacobian上限，主入口哨兵禁止其调用，未修改主采样。4项预算/边界/隔离/报告软件测试通过；原生比较集成NOT_RUN，已准备comparison_c_smoke.json和comparison_d_smoke.json，仅集成小预算不算科研比较。编译等价配置compiled_equivalence_v8.json已准备NOT_RUN；一个重型worker，待V8后再运行。

完整软件20261027T080753_software_8133f4通过44项。之后修复主方法与比较方法验收耗时越过墙钟上限后仍可能选择的计时缺口；保留物理validation但拒绝超预算选择。新增软件时间注入反例及基线测试4项通过，不算物理成功。V8仍使用启动快照，不能追认包含此修改。

输入合同检查：任务/场景未知语义原已明确拒绝，无需重写。修复两项实际缺口：Robot.restore拒绝非完整mjSTATE_INTEGRATION快照，防止缺失速度/ctrl被默认清零；Pose/Box旋转正交检查明确rtol=0，避免默认1e-5相对松弛。8项适配/输入/原生中断控制重放测试通过，8个未改DEV配置仍合法；并非放宽验收或改变状态。完整回归继续。
