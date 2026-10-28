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

完整软件20261027T082101_software_768926通过47项（先于批量质量矩阵实验）。新的实验性Propagator.bodyframe_mass_rows利用浮动子树质量矩阵刚体变换等变性，在单位基座姿态批量计算全部机器人质量块；世界固定原点动量和基座速度显式变换，完整67路力矩仍原图。默认关闭，主API不接受此后端。3状态非零H/基座姿态原生速度及q/base梯度对照通过，质量行共2tests通过；首次int32 index_copy错误已events保留。下一步仅在当前V8结束后运行mass-rows-diagnostic --config configs/ffgd/bodyframe_equivalence_v8.json及test --suite full-space-bodyframe，未执行NOT_RUN。不得跳过完整能量/J/native FD校准启用。

V8 20261027T075858_planning_e7658f已完成PASSED：相同冷seed13001/训练权重/冻结障碍任务/60x18x3预算，原生最小间隙+.003695668m，全部模型门通过，运行1833.8s。原失败V4-V7保留且不改判。严格入口哨兵无禁止调用。执行已发起（查state真实PID），随后单独重放。原反例通过不代表相关任务族、ARCH总门或五候选协议通过。下一阶段优先验证bodyframe批量后端以减少完整任务族成本；不启用未验证后端。

V8原生执行20261027T083007_selected_execution_77b070 PASSED：2500步/5s，实际间隙+.003718868m，raw力矩比.09722，0裁剪/安全中止/状态注入，P/L残差.00019071/.00018755，实际流形偏差1.08e-6（M6仍待处理，不修改实际状态）。独立CLI重放20261027T083042_independent_replay_07157e PASSED，最大状态差0。M4仍RUNNING，相关任务族/五候选/完整ARCH未过。当前已发起bodyframe_equivalence_v8整路径校准，查state真实PID；主后端未启用。

批量后端校准20261027T083121_mass_rows_diagnostic_d6884f通过：相同完整323维能量/gradient/J误差<=6.4e-16，反向顺序计时42.5/42.9s对11.6/10.6s。20261027T083343_full_space_bodyframe_gradients_56504b三例全五项活跃/8方向/3epsilon原生FD通过。主API现在显式支持batched_bodyframe；不会自动fallback。冻结plan_profile_v9_eval5.json：五seed13001..13005，所有8个DEV原hash不变，每候选60/18/3/2预算不变，总墙钟13500。先free_translation相关族，再arm_body及其余固定任务，所有候选保留；完整ARCH和M4仍未通过。

五候选首次启动20261027T083629_planning_47c9e8加载阶段FAILED，0候选生成/请求5分母保留。根因Windows没有os.O_ACCMODE，冷导入写pyc触发raw-fd审计分支。改为O_WRONLY|O_RDWR位掩码，仍拒绝未声明读取；12项哨兵回归通过。新增测试首次缺少ArchitectureViolation导入导致1失败/11通过，已修复并保留events。下一动作按原v9_eval5/free_translation配置重新启动，非换seed。

当前五候选真实worker：20261027T083755_planning_bb2f7d，PID125388（启动时间和命令以state为准），v9_eval5/free_translation，正在运行，不能启动第二重型worker。49项完整软件20261027T083824_software_28a076通过；证据索引20261027T083957_evidence_report_a3ac5b一致。下次继续核实当前run，再只执行其最终预选的合格候选/重放，继续同版本arm_body和其他冻结DEV、实际ARCH；不得将单候选诊断当五候选任务完成。C/D集成smoke与compiled_equivalence_v8仍NOT_RUN，批量bodyframe已通过并优先采用。

五候选free组已完成3/5：前两条PLAN_VALID，第3条(seed13003)仅acceleration拒绝，hull与guide网格峰值均1.57960176(关节3、t=.19047619s)，非时间漏采；final joint代价仅.0005274，均值稀释是待相关任务族共同诊断的机制。未执行/修复该候选，继续原5seed，后续无需补seed。
执行验收补齐全保存状态(含末态)的q17/q67界、既有17轴速度/加速度及abs(B)映射67轴包络；加速度只在scratch上用已记录输入原生复算，不积分、不修改actual。末态安全检查前移；原控制器及保护保留。两个输入/中断重放回归通过。障碍执行只读补查20261027T091036_execution_state_audit_6084f1通过2501状态，速度比.47874/加速度比.77929，原结果文件不改。跟踪/流形额外硬阈值尚未冻结，M5仍NOT_RUN。现有validate --run <execution> --unit actual-states可复查；属于限60s只读检查，不启第二重型worker。

旧free执行全状态只读补查20261027T091405_execution_state_audit_459d52通过，0新增积分；完整软件20261027T091414_software_c14190通过50项。现执行验收含保存末态和实际导数界，原profile未改；跟踪/流形专门硬阈值NOT_FROZEN，M5仍NOT_RUN。

第4候选(seed13004)同样仅acceleration拒绝，峰值1.071429倍/关节12，joint平均代价3.78e-6。当前五候选不改实现/seed。已在现有MultiSpaceEnergy准备默认关闭joint_peak_penalty：原时间平均加整路径所有关节/导数归一化违规最大平方，保持五项标量与去噪预算，不修改硬门。API/CLI/独立NumPy对照/架构反事实及基线显式记录该标志。NOT_RUN待当前worker结束后执行test --suite full-space-joint-peak，再weight-diagnostic --config configs/ffgd/joint_peak_diagnostic_c2.json及c3.json；诊断仅有限单步，不返回/执行修复轨迹。先完成当前组预选执行及实际ARCH，再据诊断决定新冷采样回归。

V9五候选组20261027T083755_planning_bb2f7d完成2/5有效，后三条均仅加速度拒绝，旧失败不改判。按预选index0原生执行20261027T093309_selected_execution_d939b2通过2000步/4s，间隙.008461588m、raw力矩比.07951、0裁剪/安全中止/状态注入；独立重放20261027T093337_independent_replay_d7e6e3状态差0。实际ARCH01-10审计20261027T093404_architecture_audit_b07e65通过，范围V8障碍+V9free实际调用链及同权重机制干预；不覆盖尚未启用joint_peak，不代表M4/工程/学习收益完成。当前启动full-space-joint-peak原生梯度对照，随后固定失败点c2/c3有限单步诊断，再决定通用修复冷采样回归；不补seed、不执行诊断轨迹。

full-space-joint-peak原生梯度20261027T093550_full_space_joint_peak_gradients_91850f三例323维/8方向3epsilon通过。另修通用候选排序缺口：原先硬编码双分量门且计入disabled分量；现主方法/隔离基线共用已有validation中的只读score，遵守TaskSpec掩码及声明容差。6项相关软件测试通过，V8与V9全部3条有效候选分数逐值不变，旧文件不改；后续新采样实际ARCH仍须复验。

两个固定失败点诊断20261027T093715_weight_diagnostic_33184c和20261027T093748_weight_diagnostic_28e084通过原生值对照；同预算单步加速度峰值1.580→1.423、1.071→1.057，仍不算合格计划。准备下一冷5seed回归。为降低任务族计算开销，实验性完整时刻批处理只把已传播各时刻的原全机kinematics/mass/inverse操作向量化；67力矩及非零base速度/姿态/VJP单测通过，主API尚未接受新后端。当前full_path_equivalence_v10整路径能量/梯度/J反序计时诊断，先核实state实际进程；未经原生完整梯度校准不启用。

完整时刻批处理20261027T094028_mass_rows_diagnostic_203611整路径五项能量/gradient/18行J一致(相对误差<=3.1e-17)，10.7/9.2s降至5.8/5.6s；20261027T094125_full_space_batched_path_gradients_b7c1ae三例完整323维全部五项活跃原生FD通过。显式batched_full_path后端现可使用，完整67力矩/所有机体保留。冻结v10_eval5仅启用已校准joint_peak及等价批处理，原5seed/60x18x3x2/64网格/权重/硬门不改，下一步free原反例冷组及arm_body相关族和其余DEV；实际ARCH待新链复验，不能只凭这些数值校准认定核心回归完成。

当前V10free五候选worker 20261027T094234_planning_05f840，PID203936/create1793094154.2365916，checkpoint仍best.pt b80ce0d1，冻结8DEV hash全部一致；核实进程后继续，不启动第二重型worker。53项软件20261027T094145_software_36c13a通过，启动快照在主API显式启用完整批处理之前，数值后端另有原生对照。只读v9完整去噪峰值轨迹记录artifacts/ffgd/orchestration/acceleration_root_cause_v9.json：c2全活跃段违规、末步修正仅1.58036→1.57960；c3/c4后期网络干净估计逐步超界而均值引导改善很小，支持通用局部风险稀释诊断。V10结果未出，不预判修复成功。

现有执行状态检查补充q67位置/速度流形残差及位置/速度参考跟踪峰值，均从保存真实状态只读计算，不改controller/验收阈值、不写实际状态。2项相关软件回归通过；V9free只读补查20261027T094852_execution_state_audit_ee82aa通过2001状态，流形q8.08e-7rad/v1.65e-5rad/s，参考跟踪q9.04e-5rad/v8.69e-4rad/s，专门硬门仍NOT_FROZEN，不能据这些观测直接放宽初态域。

V10free第3候选seed13003仍加速度拒绝，1.57960→1.05207倍，其他全部模型门通过；54次固定去噪内更新未追加。局部风险稀释改善但未完全消除，旧/新失败均保留；完成原5seed后继续同版本arm_body和其他8DEV，不为5/5候选成功盲增seed。最后2候选仍运行。隔离C/D入口修复相对snapshot路径的重定位，保存submitted_problem并引用本run完整initial_state；主算法未变，C/D集成NOT_RUN，smoke配置已显式使用同个校准批处理/峰值项。

V10free五候选全部完成20261027T094234_planning_05f840：2/5有效，3拒绝(加速度1.0520746/1.0191281/1.0136852倍)，总采样1315.1s，未提高通过率、未隐藏失败。预选index0与V9同hash，新的原生执行20261027T100509_selected_execution_1565d6通过2000步/4s/0安全干预，独立重放20261027T100542_independent_replay_624a29误差0。当前architecture_audit_v10实际链检查和最终源码53项软件回归运行；完成后立即v10_eval5/arm_body_box固定五候选相关族，再其余DEV，M4仍RUNNING。

V10free实际ARCH01-10审计20261027T100610_architecture_audit_140e42通过，joint_peak打开的物理反事实及真实状态消费已覆盖；第二任务仍引用V8诊断，不当作V10相关族完成。现已启动同配置arm_body_box五seed。对照V10free启动source.zip，17个主链模块字节完全相同；仅后续执行只读诊断/隔离baseline源有变化，全项目source_hash不同，旧run完整来源保留。完整M4和学习收益仍未完成。

最终同源码53项软件20261027T100657_software_79699c通过(76s)，包括原生批处理VJP和新增执行速度残差检查。当前arm_body_box五候选worker：20261027T100838_planning_731808，PID131860/create1793095717.9756744，session65801；先核实真实PID身份，完成原5seed，预选合格才执行/独立重放。源码/核心/配置不变，尚未完成不能标相关族通过。之后剩余DEV顺序irregular_times、rigid_body_box、continuum_near_limit、rigid_near_limit、short_duration_torque、short_irregular_torque。

只读学习先验根因补查artifacts/ffgd/orchestration/prior_derivative_coverage.json：TRAIN1604条加速度峰值中位.691/99%.993，全部<=1；先前保存的5条裸先验中2条越界(1.4388、1.0021)。TRAIN标准化自由变量相关矩阵满秩323但条件数2.62e5，表明高频/跨控制点相关性很强；这是诊断，不证明任何重训方案有效，不改当前权重/表示或启动扩库。继续当前相关族，若任务族结果指向先验覆盖/相关性问题，再做受控表示/训练对照，不能仅靠不断加大引导迭代。

等待arm五候选时，仅扩展现有mass-rows-diagnostic准备CPU吞吐对照full_path_threads_diagnostic.json，固定同完整批处理/能量/梯度/J/输入，在1/2/4/4/2/1线程反序比较；最大预留4/32CPU、单重型worker仍保持。NOT_RUN，主plan仍1线程且当前worker不变；需arm结束后才能执行，不因理论并行就启用。资源ledger现在按显式budget记录预留CPU(默认仍1)，无门槛/求解预算改动。

隔离C比较身份修复：旧未运行实现虽不调用网络，却借用TRAIN scaler做坐标预条件；现在C直接用物理正交自由坐标，既不加载checkpoint也不借TRAIN统计，D保持同A权重/归一化。4项baseline软件测试通过，包含不存在checkpoint与非法神经设备条件下仍不触碰模型的入口身份测试(优化/验收mock仅用于软件身份，非机器人证据)。C/D真实集成仍NOT_RUN，当前主arm worker不受影响；此改动是为确保后续学习归因可信，不预设哪方法胜出。

V10arm五候选20261027T100838_planning_731808全部模型有效，预选index3/seed13004/hash4e790ffc保持不变。执行20261027T103912_selected_execution_ab3916仅角动量残差.00120103994超过冻结.001失败，其余门通过；独立重放20261027T104027_independent_replay_fe08c7状态误差0。根因实验20261027T104825_recorded_momentum_sweep_899ed2固定原控制/外力序列，2/1/.5ms残差.001201/.000601/.000300，均重放0且原步长逐状态精确复现。证据支持积分误差，未改候选/门/时限/初态。通用execution子步已实现，500Hz控制保持、每物理子步安全检查、完整初态字节不变、仅模型timestep变化经编译hash验证；软件与原失败/相关free闭环回归待完成，不能先判通过。下一命令见state，期间DEV40固定草稿预检在240s预算恢复，未规划新任务。

原预选arm闭环子步回归20261027T105105_selected_execution_a019bf通过5000物理步/2500控制更新，角动量.000602296、线动量.000442036、间隙.008466339、raw力矩比.098189，全部冻结门通过，0安全干预/状态注入。单独重放20261027T105211_independent_replay_5ad625误差0。4项子步/完整初态/实际状态相关软件通过。DEV40固定输入预检40/40通过，manifest已冻结，20执行子集预选；仍是DEV局部扰动而非独立TEST，全部新增规划/执行NOT_RUN。当前free相关族执行及完整软件运行，核实state进程，完成后做same-V10两任务实际ARCH，再其余DEV。

通用子步完整回归通过：free执行20261027T105237_selected_execution_3b6ac1为4000物理步/2000控制更新，P.000152715/L.000175313，全部门通过，独立重放20261027T105339_independent_replay_c21bf9误差0。完整56软件20261027T105247_software_b23aee通过；same-V10两组实际ARCH01-10审计20261027T105407_architecture_audit_bb6872通过，同时保留原失败arm执行行。M4仍RUNNING。下一动作既定有界CPU线程等价/吞吐诊断，随后irregular_times五候选，统一execute显式execution_substeps_v1；未完成项NOT_RUN。

CPU对照20261027T105532_mass_rows_diagnostic_5a3a20通过：1/2/4/4/2/1线程整路径5项/gradient/J差<=5.6e-16，耗时7.195/4.219/3.546/3.525/4.217/5.634s。V11仅显式budget.cpu_threads=4(32核预留4)，A/C/D同字段，原数学预算/候选/门不改；17项预算/隔离相关软件通过。现irregular_times五候选20261027T105736_planning_549bb5运行，session97156；核实真实PID后继续。冻结配置新增限定Git属性防止clean checkout行尾转换改变hash，40任务及profile逐一checkout过滤字节hash验证通过；完整M8复现仍NOT_RUN。

实际验收缺口修复：原ff_prior/contracts.py和physical_audit.py已有q17参考跟踪误差.02rad、q67位置流形残差.02rad硬门，新核心此前只测量未接成硬门。现原值/原定义迁移到execution_tracking_profile_v1，所有新execute合并到保存profile并检查全部N+1状态，不改原数值profile/任务/旧结果；不能把这两个执行容差当M6预测流形域。4项相关软件通过(含q17映射缩小不得漏检、正交流形误差不得平均隐藏)。已保存arm5001状态只读复验20261027T110627_execution_state_audit_5482b3通过，free4001状态复验运行；0积分/未改旧报告。当前irregular_times 3/5完成且3有效，继续固定组，之后新执行需包含新增原有硬门。

当前HEAD773791c1，完整58项软件20261027T110750_software_65c1d3通过。新增原跟踪/流形门只读arm/free复验20261027T110627_execution_state_audit_5482b3、20261027T110655_execution_state_audit_6cb3e9通过，未运行新原生门的任务仍不得追认。irregular_times运行20261027T105736_planning_549bb5/PID53708/create1793098655.944116/session97156，目前4/5完成且4有效，第5条继续；完整组结束后用execute --config configs/ffgd/execution_substeps_v1.json做预选候选新闭环，再replay及实际ARCH，之后C/D隔离smoke和rigid_body_box。已提交变化不影响worker归档源码，下一次从MASTER/state及真实PID核对接续。

V11 irregular_times五候选20261027T105736_planning_549bb5完成5/5模型有效，969.184s总采样，source14fedd6d；预选index0/seed13001/hashf7b1d03e保持不变。新原生20261027T111424_selected_execution_b21476通过5000步/5s，全部16门含继承tracking/manifold，P.000432234/L.000360157，最大任务位置.000435233m，间隙.008460808，raw力矩比.112764，0裁剪/安全/状态注入；独立重放20261027T111523_independent_replay_5b3709状态差0。当前architecture_audit_v11_irregular运行，完成后C/D各1seed小预算集成，再rigid_body_box固定5seed；这些smoke不是正式对照。M4仍RUNNING，其余族/DEV40未完成，M5–M8 NOT_RUN。

V11实际ARCH01-10 20261027T111602_architecture_audit_0c4a23通过：free/arm/irregular共15真实候选，2/3/4位姿数量及原失败执行都保留。隔离C/D真实集成各1任务1候选/8引导段/3梯度：C20261027T111809_baseline_comparison_0eff1a、D20261027T111903_baseline_comparison_01586a均模型有效，各6energy/3gradient/3J；C无checkpoint/TRAINscaler，D仅3次网络(小预算smoke，不是60步正式比较)。C执行20261027T112016_selected_execution_b16c34全门通过4000步，单独重放20261027T112105_independent_replay_60b0fb误差0；D执行20261027T112151_selected_execution_353319全门通过4000步，其单独replay正在运行。两者provenance比较方法明确且comparison_only=true，主链未调用，不据此称学习收益。之后立即rigid_body_box V11五seed，state给出命令。

D单独重放20261027T112329_independent_replay_8a294a通过状态差0，C/D集成闭环全部结束，仍仅小预算smoke不是正式学习比较。当前主任务rigid_body_box V11固定五候选已实际启动20261027T112424_planning_e1f024，PID27488/create1793100264.5707712/session28069/source0ee7c4c7；首候选模型通过，剩余4个继续，核实进程与checkpoint后接续，不启动重复worker。之后预选原生execute/replay，下一冻结任务continuum_near_limit；普通失败按原机制诊断，未跑门NOT_RUN。

rigid_body_box V11五候选20261027T112424_planning_e1f024完成4/5有效，810.025s采样；c2/seed13003仅加速度1.008839534倍拒绝，全机其他门通过，失败保留未执行/后修。预选index0/hashb6d47549执行20261027T113942_selected_execution_027c77通过5000步/5s全部16门，P5.52e-5/L5.79e-5，raw力矩比.06002，间隙.00846690，最大目标位置.00012246m，0干预/状态注入。单独replay运行后继续continuum_near_limit固定5seed，source0ee7c4c7不变；M4仍RUNNING。

continuum_near_limit V11正在20261027T114113_planning_dbc8a6/PID143156/create1793101273.276425/session28602，4/5完成且0有效(速度/加速度超限并2048细化未完成)，最后固定seed继续；不补seed/不执行失败候选。根因只读证据orchestration/continuum_initial_state_root_v11.json：真实q0[0]=.85时原TRAIN均值解码初始速度4.449/加速度5.980倍，c0后段NN重复抬升引导后违反，最终1.197速度/1.609加速度。候选新通用表示仅换仿射原点initial_position，不改N/323维/边界/物理集合、不输入目标或隐藏路径；12相关软件已过。BoundaryCodec可选origin，默认pivot_zero不变；主A和D明确拒绝未迁移新origin权重。已准备test --suite full-space-origin(3例含近限位、8方向3eps五项原生对照)和data --config configs/ffgd/reencode_initial_origin_v1.json(原2000轨迹和split逐数组保留，父2711尝试/全部拒绝引用保留，0新增轨迹)；均NOT_RUN，重训NOT_RUN。先核实当前worker结束再跑原生梯度，之后数据重编码/重训/显式codec身份传播与反例及相关族回归，不直接给末步加优化。

初态坐标根因修复：原continuum五候选已全部失败并保留。新initial_position仅改变仿射原点、N及323维空间不变；原生全梯度20261027T115503_full_space_origin_gradients_45aaa2通过3例(含两近限位)。20261027T115635_prior_coordinate_reencoding_18ffb0完成原2000轨迹等价重编码，物理数组/split/attempt逐数组不变，2711父尝试及拒绝仍引用保留，最大解码差7.11e-15，0新物理轨迹。训练20261027T115809_training_8a676d已结束3000更新，VALvMSE.39146957；best SHA1d086f60da982bc0a51489a7ff2611eb8347c30919c1773d039b179ef5daea56。20261027T120402_learning_checks_78a82a通过5样本60步、边界、重载复现和条件/权重干预，仅依赖检查不代表可行性/学习收益。所有消费者显式传播origin并拒绝checkpoint错配，12相关测试通过；全软件运行中。V12保持V11五seed/60步/18x3x2引导和全部冻结门，只换表示及相同物理数据重训权重；接着原反例和相关任务族、实际ARCH回归，均尚未通过。M4仍RUNNING，M5–M8 NOT_RUN。

新origin全软件20261027T120415_software_e10c24通过62项。V12 continuum_near_limit固定五候选正在20261027T120606_planning_16cb1e/PID136352/create1793102766.4782913/session68838，真实身份已核对；不重复启动。HEAD984916ee，source归档以manifest为准。此组原反例/相关族实际回归尚未通过。

V12 continuum近限位20261027T120606_planning_16cb1e完成2/5模型有效，773.47s；c0仅q17超.028704rad，c1/c2仅加速度失败，其余门均过。原V11五失败不改判。预选index3，正在execute/session48807/substeps2，未执行其他候选。完整表示修复已有原反例改善，相关族/实际ARCH待跑。诊断实验tolerance_squared_hinge仅可由数值测试/weight-diagnostic指定，主api默认center_squared且尚未接此配置；首个容差内梯度/原生标量测试通过，其余数值对照NOT_RUN，不能宣称该实验已改善主方法。新增position_hull_violation诊断字段不改规划。

V12近限位预选index3/hash554ad004执行20261027T121913_selected_execution_48654d通过5000步/5s全部16门，P.000257685/L.000345093、raw力矩比.0975315、全机间隙.00846353、最大任务位置6.41634e-5m，0裁剪/中止/状态注入。独立重放20261027T122025_independent_replay_0dff50通过状态差0。五类任务分别已有不同版本成功，不能拼为单版本全通过。新origin相关族和实际ARCH仍待跑。当前full-space-origin --task-objective tolerance_squared_hinge实验数值对照/session67687，全软件/session30885；主V12未切该实验。完成实验对照后继续V12 irregular_times相关回归，不加seed、不选失败候选。

实验损失容差内不再追精确中心：20261027T122105_full_space_origin_gradients_b7b403原生323维/3例/8方向3eps全部通过；全软件20261027T121955_software_b62fa7 63项通过。固定c0末态单步敏感性center20261027T122155_weight_diagnostic_488a7e与tolerance20261027T122220_weight_diagnostic_fbbed5原生标量一致；只读诊断，没有把单步提案用于主结果或执行。当前主V12仍center_squared、原预算、原权重，继续相关族回归；不因诊断就假定新损失有收益。

当前V12 irregular_times正在20261027T122247_planning_e794e8/PID107104/create1793103766.9665723/session90827，sourcee19d6f95e1dfbfbb33b02978c6819c7879872cb3e9798f618595f7abbdbc7d79，HEADc7309251。新损失隔离实验否定结果：同一失败c0/同一步长及metric，weight1时center后q17越界.0286498与tolerance后.0286392几乎相同；不能称解决根因，未合入主采样调用。完整几何/动力学/最终门未变。先完成当前五候选及相关族，不重复启动worker。

V12 irregular_times 20261027T122247_planning_e794e8完成5/5，910.458s采样；预选index3/hash6a648af0，20261027T123810_selected_execution_6deed0全16门通过5000步，P.000168766/L.000221556/raw.0418349/间隙.00849262/最大任务位置.000841783m/0干预。独立replay20261027T123935_independent_replay_5ac07e差0。新origin实际ARCH01-10 20261027T124032_architecture_audit_9c6108通过，10候选含3失败、2/4pose、两原生执行；每个run的checkpoint/origin及learning checks身份一致核对已加强。原工作区HEAD/分支和保存diff内容复核未变(保存patch换行格式与git原始字节不同，记录里已区分)。下一步arm_body_box相关障碍族，然后rigid_near_limit和两短时长任务，最终DEV40统一版本；不先重复每个已知无障碍族两遍。M4 RUNNING，M5–M8 NOT_RUN，学习收益NOT_RUN。

当前arm_body_box V12固定五候选已实际启动20261027T124216_planning_8c7545/PID96828/create1793104936.6649225/session97926/source142ebe7d6fe0c3c5f1a84f5a189fbba0d647cc89ec9ce0543a55ad3b1eeb0ba5。进程真实身份匹配。继续核实此worker后接续；不要重复启动或把本次状态答复当作总目标结束。

V12 arm_body_box 20261027T124216_planning_8c7545完成2/5，939.097s；前三候选仅加速度1.0002262886/1.0152537541/1.0142841516倍失败，其他门均过。V10旧5/5与新2/5分开报告，旧成功/新失败不覆盖。预选index3/hashb933938f，正在execute/session31918。已有joint_gate_domain_root.json追加末步证据：c1后两更新使已在容差内的目标代价下降但加速度违反上升。已准备诊断center/tolerance两目标×scalar cap1/None，固定失败c1，一步、原norm/control caps和两次line trials，未改变主链；仅诊断NOT_RUN。完成execute/replay及这两个诊断后继续V12 rigid_near_limit和两短时长固定任务，先收集完整8族缺口，不用新seed掩盖。

V12 arm选中index3执行20261027T130015_selected_execution_ce28dc通过5000步全部16门，P.000159971/L.000215718/raw.0588928/间隙.00846873/最大任务位置.000935563m/0干预；独立replay20261027T130249_independent_replay_776ee9差0。根因2x2x两状态诊断4f9c77/c46076/0d09de/59cabf均原生标量一致：c1末步clean原本joint=0，center单步引入acc1.00053564；tolerance保持.97186796(cap1)或.29606718(自适应系数、实际最大控制变化.0596rad)。最终失败状态center两系数均仍>1.0146；tolerance+cap1到1.00346565，tolerance+自适应到.99541867且task/collision/joint/actuator代价0。仅诊断、无执行/替换候选。证据改变下一动作：V13接入通用tolerance_squared_hinge和原Polyak自适应系数(max_step_coefficient=None)，保留maxnorm2/maxcontrol.16/18x3/2trials/60NN/同权重同种子；所有消费端传播objective身份，数值代价仍是已通过b7b403校准的同一实现。当前全软件20261027T131027_software_8f1eec/session45482，之后原arm五seed和near/irregular相关回归、执行/重放/实际ARCH，尚NOT_RUN；先不切换到剩余三族也不称已修复整组。

V13全软件20261027T131027_software_8f1eec通过63项。当前原arm五seed复验已启动20261027T131344_planning_78937d/PID195856/create1793106823.9273715/session87704/source14fc3607fceca765f71cda3d8198b69c0986a775996b3a0d580550e6299c4aea/HEAD1db9505b；真实身份匹配。当前任务仍M4，V13完整采样/相关族/原生执行/实际ARCH尚未通过，等待当前组而非启动重复worker。

V13 arm固定5候选20261027T131344_planning_78937d完成4/5，c2加速度/姿态失败保留。预选index3/hashb8463c2d，20261027T132934_selected_execution_f49355完成5000步全部16门，独立replay20261027T133109_independent_replay_00ad17状态差0。只读预测测量100572/1286c0已分别验证V13/V12arm5001状态，原候选/执行/验收文件hash不变；V13双末端全程位置差最大.000151778/.000047845m，非M6跟踪界。新增artifact_checks.compare_tracking_prediction与CLI只读入口，不改主链。继续V13 continuum_near_limit固定5seed，然后irregular/实际ARCH；剩余3族/DEV40待跑。M4 RUNNING，M5–M8 NOT_RUN。

V13 continuum_near_limit20261027T133752_planning_7a6ebb完成4/5，751.117s；c0 q17仍失败(连续曲线max1.0044784、控制点1.0088674)，其余4过。预选index1/hash0fd7f3a7，原生20261027T135039_selected_execution_487e77完成5000步但task5.00097191mm及实际acc1.000704/1.001167失败；其余门通过，0安全干预。独立replay20261027T135132_independent_replay_dfb9a6差0，不改失败。只读预测/导数测量20261027T135325_tracking_prediction_diagnostic_18eb38：失败末端任务时刻预测差.135461mm，全程17维acc预测差最大.01004756倍限值，峰值参考.992089/实际1.000704。根因是guide在硬门内代价已为0而缺少执行裕量。新通用planning_margins仅作用去噪内：task_tolerance_fraction=.8、derivative_limit_fraction=.95，不改TaskSpec/限位/profile/controller；所有消费端/基线/审计显式传播身份。323维3例8方向3eps原生梯度20261027T135702_full_space_origin_gradients_5d22d7通过；5相关软件通过。失败候选固定末态单步baseline5a91c3无可接受更新，reserve0d5add接受且task.070585到.000012535、acc hull.97882248；诊断提案未执行、未替代主结果。当前全软件/session73344，之后plan_profile_v14_eval5.json重跑同一近限位五seed，执行/重放、arm/irregular相关族、实际ARCH。V13 irregular未运行，不能追认；M4 RUNNING、M5–M8 NOT_RUN。

V14全软件20261027T135917_software_ea6c71通过65项。HEAD4fc4f17c，当前近限位固定5seed已真实启动20261027T140125_planning_b0cc7b/PID201700/create1793109685.6680377/session69417/source8ff4525ccbf12e520231259bf9cb7e5c302edbde727168e57a1859f7eb99ae50；进程身份已匹配。先核实此worker再接续，勿重复启动。V14原反例完整采样/执行/相关族/实际ARCH均未完成，继续保留NOT_RUN，主目标ACTIVE。

V14近限位20261027T140125_planning_b0cc7b完成4/5，755.514s；c0仍q17失败，其余4通过。冻结评分仍选index1/seed13002/hashc068f948。原生20261027T141416_selected_execution_46f897完成5000步全部16门：实际位置.004000295/.003987085m，acc17/67=.954417087/.954879256，P7.78e-5/L5.03e-5，raw力矩比.074735，最小间隙.008503823，0裁剪/中止/状态写入。独立重放20261027T141509_independent_replay_f09881状态差0。原V13失败不改判；这一原反例修复有实际证据，但相关arm/irregular和V14实际ARCH仍待完成。下一任务V14 arm_body_box固定5seed，之后剩余3族与DEV40，M4仍RUNNING。

当前V14 arm_body_box五候选已启动20261027T141558_planning_36d59f/PID202804/create1793110558.5550506/session1989/source8ff4525ccbf12e520231259bf9cb7e5c302edbde727168e57a1859f7eb99ae50；已核实真实进程，HEAD4fc4f17c。先完成此worker再接续，不重复启动。未提交仅本轮HANDOFF追加，代码已提交。

V14 arm_body_box20261027T141558_planning_36d59f完成5/5，935.309s；每条60NN/54gradient，冻结评分选index4/seed13005/hash47aa7ede。原生20261027T143149_selected_execution_64e7a9完成5000步全部16门：最大实际任务位置.004022176m，acc17/67=.5219565/.5236049，P5.956e-5/L4.734e-5，raw力矩比.0635134，最小间隙.008462923，0干预/状态写入。独立重放20261027T143242_independent_replay_81e07b差0。近限位原反例与arm相关族现均有V14真实执行，但irregular和V14实际ARCH仍待完成；不追认旧版本。下一条命令plan_profile_v14_eval5.json + dev_v1/irregular_times.json，同5seed。

当前V14 irregular_times固定五候选已启动20261027T143328_planning_228638/PID90856/create1793111608.483466/session70365/source8ff4525ccbf12e520231259bf9cb7e5c302edbde727168e57a1859f7eb99ae50；真实身份已核实，HEAD4fc4f17c。不要重复启动；此组完成后预选execute/replay，随后一次V14三族实际ARCH。M4 RUNNING，M5–M8 NOT_RUN，学习收益NOT_RUN。

V14 irregular_times20261027T143328_planning_228638完成5/5，910.638s；预选index3/seed13004/hash2337fff3。原生20261027T144854_selected_execution_212958完成5000步全部16门，4个时序位姿位置最大.003978197m，raw力矩比.0404119、P.000165449/L.000222096、最小间隙.008489507，0干预/状态写入。独立重放20261027T145012_independent_replay_485f8a状态差0。三组源码8ff4525c、checkpoint/配置/预算/裕量/profile均逐项一致，V14实际架构20261027T145038_architecture_audit_de6117全ARCH01-10通过：15真实候选含1拒绝、2/3/4pose、两场景、三原生执行、目标/几何反事实有效。这里只是V14三族修复回归；M4仍RUNNING，不能当完整8/40或学习收益。下一步首次rigid_near_limit固定5，再两个短时长任务；uniform DEV40中free/rigid-box也必须用当前版本，旧成功不可合并。

当前首次rigid_near_limit V14固定5候选已启动20261027T145309_planning_a62c3f/PID170176/create1793112789.5797317/session11796/source8ff4525ccbf12e520231259bf9cb7e5c302edbde727168e57a1859f7eb99ae50；真实身份已匹配。HEAD4359ffe7，仅HANDOFF新追加未提交。先核对此worker再继续，不能重复启动或假定首个刚性近限位任务已通过。


V14 rigid_near_limit固定五候选20261027T145309_planning_a62c3f全部失败，速度/加速度超限及2048细化几何步长不达标；0选择/0执行，未追加seed。只读TRAIN覆盖rigid_initial_state_root_v14.json显示初态q10=2.9rad远超TRAIN约±.1；同噪声四因子诊断20261027T150639_condition_diagnostic_bfccd4，原条件未引导加速度2.0141倍，初态token置TRAIN均值后.5010倍，单独目标token置均值仍1.4341倍。此为条件敏感性证据，不证明完整根因，消融token绝不进入主链，诊断样本未选择/执行。新增通用数据initial_policy=factorized_limits：所有17坐标逐轴实际限位分8层、每组初态独立、失败不重采，保持原目标/几何/验收器/数据前向生成；主采样不变。旧home_box逐随机数结果保持，新origin可直接生成。6相关软件测试通过。下一命令data_initial_domain_pilot_v1.json，256有效目标/最多544尝试/2400秒，只是覆盖根因pilot，先检查接受覆盖及拒绝分母再决定扩展重训，不宣称修复成功。M4 RUNNING，M5–M8及学习收益NOT_RUN。

当前通用初态覆盖pilot已实际启动20261027T150930_prior_data_370e36/PID171320/create1793113769.1505113/session15607/source014e104691c2b9071617380fb6397449a646e849035c9f6c6a34ce80d9cfe804，HEADc1f941ec。进程身份已核实；先接续该单worker，完成后查dataset_manifest初态TRAIN/VAL范围及所有拒绝。此批尚未训练，不能称刚性近限位已修复，不启动重复worker。


初态覆盖pilot20261027T150930_prior_data_370e36完成396尝试/256有效(201TRAIN55VAL)，拒绝101acc/22collision/14q17/3refinement全部保留，耗时530s。TRAIN q10范围[-2.889865,2.813815]，2.9任务仍略外推，不能称已完全覆盖。合并20261027T151826_prior_union_ebd703逐数组核实原2000+新256物理轨迹与split不变，3107总尝试及两父拒绝均保留，group仅命名空间隔离；3合并测试+2域采样测试通过。新训练20261027T151842_training_36420a保持seed913/原网络/3000更新，best VALvMSE.41123392(VAL组成改变不能直接与旧值比较)，best SHA f73277cbd62b197b1ef20f824ffa664422138e321d6f9eb99c0c6151c97a329f，data SHA eeffbf6fc624c563cf9bc320c47152db07f805a9244aa57a35c3bde077da8b31。learning_checks20261027T151955_learning_checks_a95d91通过5样本；同噪声条件诊断20261027T152007_condition_diagnostic_43fa42在真实原条件下未引导加速度从旧2.014109降至.578461倍，task代价仍5382.918不合格，只有根因改善证据，不是规划通过。V15配置只换checkpoint，guide/seed/验收全保持V14。当前全软件/session79820；结束后V15原刚性近限位固定5，预选执行/重放、相关族和实际ARCH均NOT_RUN。架构职责未变、无消融token进入主链、无后置修复。M4仍RUNNING。

全软件20261027T152015_software_958b56通过70项。V15原rigid_near_limit固定5已实际启动20261027T152229_planning_257148/PID59796/create1793114549.8296883/session97499/source2d400892d3c31bd3162c0e8e6c9f6bb575a95548c8c6ce7921d097bfce60d046，HEAD32e54e5b；真实进程及best权重SHA已核对，配置逐项比较仅checkpoint不同。先接续此worker，完整采样/执行/相关族/实际ARCH尚未通过，不能重复启动。

V15 rigid_near_limit首个固定候选seed13001完成PLAN_VALID，12个模型门全true，位置2.452/2.044mm、姿态.014879/.009670rad、名义力矩比.19447549、全机间隙.008463629m、2048细化步长.001960414m，输入只读核对通过。原V14同seed失败保持失败；这里只是第一条候选，不提前选择执行。PID59796/create1793114549.8296883仍真实运行，session97499，剩余4候选继续。完整组、原生执行、相关族和新权重实际ARCH未完成，M4仍RUNNING。

V15 rigid固定5完成3/5，817.539s采样；c3仅刚性末端5.1248997mm失败，c4加速度/2048几何步长.00238677m失败，均保留未修复。冻结评分选index1/seed13002/hash9597f3a1，原生20261027T153625_selected_execution_1044aa完成5000步但P=.00123354846/L=.00100864780超冻结.001，其余14门全通过，maxacc17/67=.91300360/.91341984，位置3.818805/2.442742mm，raw.14698050、clearance.00846653、0干预/状态写入。独立重放20261027T153729_independent_replay_a76701状态差0，不改失败。发现recorded_input_sweep原先取planningdt(.002)而非已细化executiondt(.001)，已按provenance重建模型并校验dt/hash，防止拉长输入时间轴；3原生回归测试通过47.07s。当前固定输入收敛实验20261027T153948_recorded_momentum_sweep_9c96ac/PID60556/create1793115586.8955789/session41618已实际启动，先核实接续该worker；1/.5/.25ms均只重放记录输入，不重算控制/不规划/不投影。若收敛证据支持，再同一预选候选做明确新版更细步长原生执行及独立重放，冻结门不改、旧失败保留。M4 RUNNING，相关族/新权重实际ARCH与M5–M8仍未完成。

固定输入收敛20261027T153948_recorded_momentum_sweep_9c96ac通过：1/.5/.25ms P=.001233548/.000616743/.000308364，L=.001008648/.000504355/.000252185，原记录重现差0、每组重放差0、仅dt模型差且初态字节不变，支持一阶积分误差。新增通用execution_substeps_v2.json=4(物理.5ms/控制2ms)，后续新回归和比较统一采用此配置，不按场景决定步长。原预选index1/hash9597f3a1原生复验20261027T154120_selected_execution_2152c2通过10000步全部16门：P=.0006178701/L=.0005052294、raw.14711986、clearance.008466530、实际位置3.793267/2.405780mm、acc17/67=.91359094/.91394229、0干预/状态写入。独立replay20261027T154256_independent_replay_88e91f状态差0。原1ms失败保留；不是重规划/重选/改门。下一V15 arm_body_box固定5相关回归，然后新权重实际ARCH、两短任务和统一DEV40。诊断模块修正导致源码hash由2d400892变0468f9ee，主采样及权重/配置未改；不能将此称所有版本同源码全通过。M4 RUNNING，M5–M8 NOT_RUN，学习收益NOT_RUN。

当前V15 arm_body_box固定五候选已真实启动20261027T154403_planning_9f40ca/PID135264/create1793115843.5275843/session92201/source0468f9ee4a22c6908b6f48fb4ffa8979b2778b15525e90e8a3230d8e9a86666c，HEAD2a521163；已核实真实身份。先接续该worker，勿重复启动；完成后冻结预选执行统一execution_substeps_v2.json再独立replay。相关族与实际ARCH尚未通过。

V15 arm首候选seed13001 PLAN_VALID，固定组继续，PID135264/create1793115843.5275843已再次核实。审计补强execution_plan_binding：执行provenance必须指向本次审计planning_runs；预选index/记录identity/实际candidate.npz/provenance/执行summary哈希一致，execution.npz保存control_points与duration必须精确匹配，避免把旧权重执行证据配给新权重。3身份回归通过；另对真实刚性1ms失败和.5ms成功两执行只读核对，均绑定同一9597f3a1候选。此修改仅审计，不改变主采样、模型、控制或门；实际ARCH待整个相关族完成后运行，未提前标通过。state.architecture_evidence已纠正为最新历史V14审计de6117并显式列出V15 pending，避免旧V12指针或历史通过被误认为当前权重完成。继续同一session92201，不启动重复worker。

V15 arm_body_box20261027T154403_planning_9f40ca完成5/5，939.791s采样，冻结评分选index3/seed13004/hashf47c469e。统一.5ms原生20261027T160001_selected_execution_77fa6e通过10000步全部16门：实际位置最大3.964997mm、acc17/67=.45240679/.45264608、P8.2355e-5/L.0001105926、raw.05678370/clearance.008467814，0干预/状态写入。独立replay20261027T160144_independent_replay_ae83ef状态差0。V15实际ARCH20261027T160231_architecture_audit_fe270c全01-10通过，刚性+障碍10候选含2拒绝、2/3pose、2scene、3执行含保留1ms失败，绑定预选候选检查通过。两规划checkpoint/budget/weights/margins/profile逐项一致；源码差异仅后续诊断/审计修正，分别保留sourcehash。新权重原反例和相关族有真实证据，尚非完整8/DEV40或学习优势。下一首次short_duration_torque(.8s)固定5，再short_irregular_torque(1.2s)，不修改时限/目标，执行统一substeps_v2；ordinary失败按根因处理。M4 RUNNING、M5–M8 NOT_RUN。

当前首次0.8s short_duration_torque V15固定5已实际启动20261027T160408_planning_0f6da0/PID122756/create1793117048.0936613/session10113/source29f6dc7bc563ee85ce0de54027de61ab67e5668f2f60d782064f0f6846d71e2b，HEAD94aa34c9；真实身份已匹配。先核对此worker再接续，不重复启动，不假定短时任务已通过。

V15 .8s short_duration_torque20261027T160408_planning_0f6da0固定五候选全部REJECTED、0执行；首条acc7.18417倍/名义torque1.258477倍/刚性姿态.029rad失败，所有分母保留。short_duration_root_v15.json只读证据：TRAIN4s639/5s1166，同一控制标签acc随T^-2放大，数值恒等误差6.33e-15。通用可逆duration_squared_v1令物理free=T_seconds^2*学习free，N/323维、实际T、边界、物理和门不变；model_io.bind_free_coordinates拒绝跨T重复绑定/未知schema，旧权重分支逐值保持；主A、隔离D、学习检查、诊断、语义及实际ARCH已显式传播单位，C仍非学习物理坐标。重编码20261027T161856_prior_duration_reencoding_7ba525保留2256物理轨迹/初态/token/split/3107父尝试，0新物理轨迹，decode误差8.88e-16，data SHA e6fe96af4d0f5d62eec1fbcf3cbeafa1e731390f7053449ee99555538b928afc。训练20261027T162010_training_fedd0a完成3000同seed更新，bestSHA a1a2216e8b74de667811c12cb867c56352ea8c6b7fc595bcf128219fbf915c20，VALvMSE.43142621(标签单位变化不用于跨版本优势结论)。中断后确认训练正常结束、无残留训练worker；20261028T011712_learning_checks_7058b2通过5样本，20261028T011729_software_d84167通过77项/146s(软件资源预算180s，不是任何验收门调整)。20261028T012029_full_space_origin_gradients_d849ec通过新坐标链323维/3例/8方向3eps五项+总梯度原生FD，包含近限位初态，实际检查时长.6/.65/.7s；不是规划可行性通过。V16配置只换新权重，同60步/18x3引导/五seed；下一原.8s固定5，再相关族与实际ARCH。1.2s任务和DEV40仍待跑；M4 RUNNING，M5–M8/学习收益NOT_RUN。

当前V16原.8s固定5已实际启动20261028T012307_planning_26fc92/PID183048/create1793150587.6488469/session44819/sourcee8ba4cc9754b0b768145f4beb9eecf596cdcc5b6759e1e3a1c550738c1e5236f，HEAD2f44acdb；真实身份已匹配。先接续此worker，不重复启动；坐标修复的新采样、执行、相关族和实际ARCH仍未完成。


V16完成结果补记（架构/进度答复时核对）：20261028T012307_planning_26fc92已于北京时间2026-10-28 09:36:15自然结束；五候选2/5 PLAN_VALID，c2(seed13003)加速度失败，c3/c4(seed13004/13005)加速度与目标失败。冻结评分选index1/seed13002/hash65f7e02f465ea3ba67933dd5c0345a0fb6cc56cba4fe2e46a315d6e798b677a3；不补seed、不修复失败轨迹。核查无活跃ffgd.cli worker，active_unit为空。此前启动记录及state.next_action已滞后，本次仅补准确结果/下一动作，不改验收门、不追认阶段。下一动作按substeps_v2执行此预选候选、独立重放，再相关族及V16实际ARCH；三个失败保留并做根因定位。M4 RUNNING，M5–M8 NOT_RUN，V16实际ARCH和学习收益NOT_RUN。已交付exports/ffgd_review_20261028T013929Z网页审阅包及完整状态包；它们是09:39历史快照，不随本次交接追加重写。

用户2026-10-28最高优先级已合并MASTER/AGENTS并同步master hash：工程全流程优先，优势/DEV40/全部候选成功/四方法比较不再阻塞。当前无ffgd worker；HEAD2f44acdb，V16 selected_index=1/hash65f7e02f，执行substeps_v2随后独立重放，不重选。已有数据/权重复用，M4与后续工程工作仍未完成，研究NOT_RUN。

工程优先执行：V16 .8s预选index1未更换，20261028T025519_selected_execution_697534完成1600原生步全部16门，双末端3.391411/3.976540mm，实际acc17/67=.97076175/.97217874，P4.5773e-5/L5.4469e-5，0裁剪/安全中止/状态覆盖。20261028T025546_independent_replay_31793d独立重放差0。相关irregular任务使用事先记录plan_engineering_v16_one（单seed13001，其他引导/权重不变，非五候选统计），20261028T025641_planning_d57eaf正在执行。新增continue-state只导出完整实际snapshot，不宣称已完成预测接续；实际残差仍需完整状态预测，不得放宽1e-10或投影。

V16不规则时序单候选20261028T025641_planning_d57eaf通过，预选index0/hash8e4dd4c2；原生20261028T030006_selected_execution_198261完成10000步全部16门，最大位置4.015816mm，P8.98494e-5/L.0001078412，0干预/状态写入；独立20261028T030153_independent_replay_71ba38差0。实际架构20261028T030239_architecture_audit_bccf4c全ARCH01-10过，6候选含3拒绝、两种pose数量、两执行，非优势研究。continue-state接口已实装，20261028T030357_continuation_state_f2e9af实际末态逐字节交接，保留非零基座/关节速度、ctrl/积分全状态与时间；仅交接通过，不是后续规划通过。q67位置残差5.956838e-7、速度1.329662e-5，目前Propagator只支持1e-10约化流形，下一最小核心工作为完整状态残差动态预测/原生对照，然后实际接续；不投影/重置/放宽域。M5/M6改为RUNNING（非完成），M7/M8仍NOT_RUN；总工程PARTIAL，学习优势NOT_RUN。无需等待DEV40/对比。

M6完整状态预测核心增量：physics.tracking_command/tracking_acceleration/full_state_acceleration在实际nq81/nv79上实现现有FF+PD与全67路饱和力矩、质量/偏置/阻尼/惯量动力学，禁止投影；FullStatePredictor显式中点传播全部关节及两自由体，控制2ms保持/物理.5ms，初始qpos/qvel逐值不变。尚为opt-in数值实现，未替代主Propagator，未宣称去噪内接通或M6完成。原生校准20261028T030922_full_state_dynamics_4866b7：实际末态/通用残差扰动/移动目标3状态，力矩差<=2.78e-16，加速度差<=4.06e-14，3方向×3eps梯度最大相对误差2.08e-9。短路径187ee8通过.02s前向与.004s梯度；完整.8s前向20261028T031249_full_state_dynamics_0f31ca通过，qpos最大绝对差1.069142e-6、速度6.845342e-6，整单元56.05s，梯度仍仅.004s两方向最大相对差2.214e-6。全时域梯度/引导/只读验收扩展NOT_RUN。失败保留：efa4d5索引Int/Long错误已修复；cc36fa诊断自行选用.04s边界模板数值一致性拒绝（非物理不可行），改用真实原任务.8s模板只观察前.02s；另一次可选时长参数改写造成导入NameError、无run目录，已修复。不改任何任务时限/硬门。下一优先有界内存整路径梯度与成本测量，再接通full-state物理代价和独立验证，不能只放宽原1e-10域；最终卫星目标接口/接续仍未完成。

M6离散伴随实现中：FullStateAdjoint保存数值x/v及保持力矩/raw序列，逆序重建单步局部AD，显式回传held-control依赖；不是连续伴随近似、前缀detach或有限差分。full-state短诊断20261028T031820_full_state_dynamics_c5f451通过（.004s与直接AD对照，.008s原生方向差分）；新增非线性保持输入软件回归1passed，覆盖初态和全部参考导数。当前.8s完整伴随20261028T031900_full_state_dynamics_d4d17c/session53976/PID184552、预算900s，已核实存活，等待原生方向对照；不可因输出静默重启。尚未改主链、全时域结果NOT_RUN直到run完成。

完整伴随根因修正完成：d4d17c初始绝对尺度检验PASS但.8s方向1 AD=-.000451 vs原生FD+.0000717符号不同，不能当完整认证。a22af6原生dt .5/.25/.125ms收敛对照FAILED保留：差每减半近似减半，定位中点预测与原生离散积分差，而非删状态或增加seed。新增可选implicit_damping步骤，实际全质量+隐式阻尼解，半隐式位置积分，完整状态和2ms控制保持不变；37bd07短对照通过。20261028T032706_full_state_dynamics_4e3d08整.8s/1600步全状态离散伴随通过：与直接展开短AD差0，两个原生方向绝对差1.71722e-9/1.11308e-8，按信号相对差1.59539e-6/.000155186（新门abs<1e-7或signal_relative<.001，旧绝对尺度结果不改写）；前向20.03s/反向68.33s，RSS3.3323GB，整单元99.62s。非线性保持输入软件回归1passed（初态和全部参考导数均对照）。原生目标仍当前实际末态/接触自由域，不称全任务全状态泛化。主A链尚未切换，下一直接实现full-state energy（真实q67/速度/反馈力矩、目标/几何）和独立原生只读验收，再从真实handoff新噪声生成/执行/重放；已有证据够行动，不重复无关研究。无后台worker，目标ACTIVE。

M6全状态接线：新增显式full_state backend，FullStatePropagation以2ms整状态/控制预测和离散伴随生成同一候选代价，保留初态qpos/v与67维残差；代价增加实际q67/v67/a67和反馈raw torque，原参考约束/名义执行器项保留。独立full_state_validation用原生.5/.25ms预测、全机扫描、全部actual速度/加速度、动量、未驱动方程和只读输入hash，不调用execute/track，也不冒充执行证据。执行前复验、独立validate和架构审计按manifest显式选择相同backend，无静默fallback。初次0a4fa2点全部约束代价零，仅平滑项有效，不外推为约束梯度通过；增强通用扰动f51229实际task/joint/actuator非零，全项数值相对差<=1.44e-13，总梯度原生FD相对差1.271e-9，合成参考只读验收PLAN_VALID（不是扩散结果）。碰撞该点为零，不能称新碰撞反例通过。4软件回归通过。下一配置plan_full_state_continuation_v1事先固定：V16权重、真实handoff状态、新seed23001、12NN/后6步各1gradient/2trial、task_metric关、16guide intervals、1CPU、3600s预算；主链无后处理，真实采样/执行/ARCH尚待跑。

第一条完整状态接续20261028T034311_planning_6488b4完成0/1，约291s；reference/actual加速度及末端姿态失败（tip0 .0178764rad），初态逐值、动力学、几何、力矩、动量均过。6次引导均接受并下降，最终task.029856/joint.014286仍非零，不是传播/梯度对照失败。预先增加分布预算v2：同seed23001重新从新噪声生成24NN/18x2guide，保持任务/权重/物理门，不能复用失败末态作初猜。旧失败保留；若仍停滞，定位预条件和梯度方向，不加seed。

当前接续v2 20261028T035040_planning_3260d8/session49265/PID174028/create1793159440.5354416（HEAD7b7500a8/source722cc3bf）运行中。工作线程启动源码保存在source.zip；准备中的新版代码尚未进入该已加载采样链。球体场景/解析support/AD-原生对照及盒体回归4passed31.97s。通用interval与C2参考边界实现中；初测9pass/1fail是旧IO fixture缺少新要求的真实参考身份，修正fixture并增加绝对时刻一致性验证后3项接续回归passed50.13s。尚未接通所有验证消费端，不能宣称temporal/C2全链通过。

接续v2 20261028T035040_planning_3260d8完成0/1，1391s；双末端任务已过（4.31842/3.06248mm，姿态.014439/.014494rad），剩reference/actual加速度失败。只读定位b99dc5确认q3参考1.017608倍、实际q67坐标23在.038s达到1.034730倍，guide预测1.029422倍；不是初态重置或传播大偏差。根因：actual_peak原先取67维均值再跨时间最大，稀释单坐标违规。改为actual q17/q67/v17/v67/a17/a67逐坐标最坏项；原全状态3个活跃约束原生FD e2d363通过，gradient相对1.13e-9。新配置v3保留原seed23001/24NN/18x2和原任务，joint引导权重3，尚未启动。C2真实交接cc474f逐字节保留原实际状态，另外传上一预选计划局部q/v/a，不传路径；静态时钟一致性也核对。Temporal首次5ef2a4数值计算后因numpy bool序列化失败，已修复并保留失败；正在复验。球体/区间/实际C2完整采样与实际ARCH仍NOT_RUN。

区间+C2新版数值0210d3通过（原生FD相对6.6137e-10；合成零自由参考只读PLAN_VALID，不是扩散结果）。保持/目标相对区间按实际每个原生状态验收，非网格时刻用公开记录的插值括号和四元数slerp，不移动目标时刻；终端自由速度模板尚未实现，不能宣称完整运动追踪通过。最终2R5C静态/七点模板已原值迁移，原7球体初态间隙.008540645871636473与公开记录一致，只是配置/初态检查，M7 NOT_RUN。

全软件20261028T042303_software_887474通过88项/199.90s（软件预算300s，非物理门）。HEADfadeda25后v3接续已启动20261028T042656_planning_87865e/session35541/PID28136/create1793161615.9421785/source492b6e7536abd84b543484a02933fcd896254e0d51df19928cb990a6cbd4a241。仍原真实初态/任务/seed23001，最坏actual坐标代价修正+joint权重3；等待候选，不重启。独立准备终端free模板357维（初始q/v/a固定，末端q/v/a51维满秩），目标相对追踪必须显式free，旧zero_derivatives权重不能加载此模板。已有2256物理路径计划从两个平坦父dataset精确重编码复用，另256上限544尝试的非零一致动量初态/自由末端pilot已预配置，尚未生成或训练。新边界/重编码/原合并/时序/单位软件16passed6.53s，不是物理训练或规划证据。
