# V7.1 实现与 smoke 验收

2026-10-15。分支 `v7.1-lightweight-free-floating-prior`，基点
`070e7008923bf2642dcecf9806745564388669b9`。最终运行 `smoke_002` 的源码 SHA256
为 `ba712f4d675766a27d8558a9abc6fe1262caa71b9c741f5f03b483b3d1895560`。
运行在提交之前，因此 producer_commit 如实保留基点，同时保存实际执行源码逐文件
哈希及 ZIP；提交的是同一源码，不把未提交代码冒称成基点原有实现。

用户明确授权的“仅不穿透”已写入 `V7_1_EXECUTION_TARGET.md` 和
`configs/ff_prior/v7_1.proposed.yaml`：参考与实际净空均要求 >=0 m。
其他门槛、自由漂浮基座和全机检查保留；不删除碰撞对来取得通过。
任务包原始 checklist 是待执行规格，原文件保留 NOT_RUN；测量结果独立保存为
`ff_prior/evidence/smoke_002/acceptance_results.json`，逐项给出证据和范围。

## 分别报告的结论

| 状态 | 实测结论 |
|---|---|
| OFFLINE_PIPELINE_STATUS | IMPLEMENTED / GENERATION_EVIDENCE_AVAILABLE |
| MODEL_PRIOR_STATUS | MODEL_VALID_SAMPLES_AVAILABLE；主库 200/200，DEV 199/200 |
| PHYSICAL_AUDIT_STATUS | 24/24 离线审计通过真实执行及独立重放 |
| TRAINING_STATUS | PASS；真实 CPU 训练 200 更新、权重变化、重载一致 |
| GUIDED_SAMPLING_STATUS | DEV_EVIDENCE_AVAILABLE；去噪内完整路径代价，SUBSPACE_FD_DEV_ONLY |
| BENCHMARK_STATUS | NOT_ESTABLISHED；没有 Diffusion 收益证据 |
| SCALEUP_ELIGIBLE | false；pilot 未运行，DEV 引导后端非生产级，物理容差仍为 inherited DEV |

M0–M5 的默认开发工作已完成。软件完成、先验模型有效、个别物理执行通过和
Diffusion 收益不是同一结论。不得把 MODEL_VALID 率称为预先指定目标规划成功率。

## 数据与真实物理

主库尝试 200，保留 200，拒绝 0；独立 DEV 尝试 200，保留 199，拒绝 1。
`dev_000083` 的最大几何步长 0.00203253586 m 超过冻结的 0.002 m 预算，
即使离散净空为正、端点数值已收敛，也保留 REJECTED，不补样、不放宽门槛。
冻结分组得到 TRAIN 150、VAL 30、封存 TEST 20、DEV 199。测试集未用于训练、
选 checkpoint 或开发评测。运行后的完整性审计重新验证全部 799 个发布文件、
399 条共享解码器视图/目标标签、谱系划分和 TRAIN-only scaler。

离线随机抽查 16/16，Wilson 95% 区间 [0.8064,1]；风险组 8/8，区间
[0.6756,1]。每组从不同 DEV 族预先选定，失败不替换。风险组不是总体无偏估计，
不能合并给出总体成功率；样本很小、构型幅度有限，不能外推整个工作空间。

另有最小对照中选定的 1 条实际执行成功，合计最终运行 **25 次执行 + 25 次独立重放**。
每条 5250 物理步（10 秒运动 +0.5 秒保持），0.002 秒步长，67 路力矩。
重放均创建新 data，仅施加保存的 ctrl；qpos/qvel/act/warmstart/time 最大误差均为 0。
原始先验记录仍保持 MODEL_VALID，审计证据单独保存，未批量升级。

全部 25 条的最低原生全机净空为 0.00846071426 m；最大跟踪误差
9.56544e-6 rad，最大流形残差 2.28985e-8 rad；最大基座平移 0.00321165 m，
证明这里不是固定基座执行。命令饱和率均为 0，动量分组诊断均满足冻结门槛。
最差保持位置误差 0.00484812 m、姿态误差 0.00685197 rad，包含对照所选候选，
分别满足 5 mm 和 1 degree；不得仅展示误差更小的离线抽查。

模型哈希：`76edfae2ce103d0191dbbf7911abfc5d3de0bedd06725ffb86361a596830ca19`。
完整初态 ID：`9eb21ed3fa3f7d5fd475de862d75adce52e582f5ad86911b09330c45e0c01c7b`。
原生 state/ctrl、冻结目标及全机检查细节留在本地 `audits/` 和 `benchmark/`。

## M5 最小闭合与负结果

真实 TRAIN 150 / VAL 30，Transformer hidden128/layers4/heads4，200 次更新。
epsilon train MSE 从 1.83544 到 0.28475，VAL MSE 从 1.66203 到 0.27201；
checkpoint 仅由 VAL 选取，权重哈希发生变化，重载最大差异 0。
损失下降证明训练链路工作，不证明可用规划器。

4 个另外预留 DEV 族生成的 MODEL_DERIVED 任务在方法运行前冻结。
三方法各 5 候选、同一公开条件/表示/物理门槛；按模型规则先选至多一条再执行。

| 方法 | 完成任务 | 无合格候选 | 所选候选实际成功/预定分母 |
|---|---:|---:|---:|
| 公开目标结构化提案 + 同一引导 | 4 | 3 | 1/4 |
| Diffusion + 去噪内引导 | 4 | 4 | 0/4 |
| Diffusion 无引导 | 4 | 4 | 0/4 |

两种 Diffusion 方法各 20 候选均未通过 domain/motion 模型门槛，无引导另有
5 条 native_range 不通过；没有合格候选时不启动物理执行，全部保留在失败分母。
结构化候选通过模型门槛但多数不能满足冻结目标。没有运行时异常被删去。
200 更新只完成开发闭合，不能支持 Diffusion 优于基线；结果为 NOT_ESTABLISHED。

有引导 Diffusion 的每候选 DDIM 恰好 16 次调用，在反向迭代 14/15 内更新。
每次 8 个冻结方向、16 次整路径中心差分，再加基础/试探/半步差分检查。
使用完整耦合基座路径、双末端世界误差、全机几何和运动代价，终端变量参与更新。
后端标识 SUBSPACE_FD_DEV_ONLY，不是完整精确梯度，不触发在线生产扩展。

## 覆盖与限制

399 条有效参考的最大单关节幅度范围 0.06148–0.30125 rad，中位 0.16918；
17维路径长度 0.11886–0.69779 rad，中位 0.33911；预测基座平移峰值
0.267–4.448 mm，中位 1.690 mm。双末端位置与姿态分布详见
`release_verification.json.coverage` 和 `visuals/goal_coverage.png`。
这些是相关生成族、固定初态附近的局部样本，不是均匀工作空间覆盖。

参考驱动力估计 NOT_EVALUATED；若发布 profile 将它设为必需，入口返回 BLOCKED。
任意初态/非零动量、动态障碍、连续时间认证、硬件均未验证。
pilot、study、20任务严格多模态物理实验、最终200任务 TEST 均 NOT_RUN：
默认仅 smoke，必须另行显式 profile、冻结任务池/预算和前置证据。
严格五候选合同、去重及完整池验证已实现并测试，不把软件测试算成严格专家实验。

## 测试与恢复证据

43 项测试通过，0 failures/errors/skipped，原始 JUnit：
`ff_prior/evidence/smoke_002/tests.xml`。覆盖映射/初态/坐标/样条、零动量限制、
自由基座反作用与网格收敛、路径依赖目标、未来条件注入拒绝、严格配额、
TRAIN-only scaler、源/发布文件篡改拒绝、真实子进程崩溃恢复及并发锁、
预算停止、DDIM 调用数、真实 ctrl 执行和独立重放。
自碰撞、臂间、安装座、中途障碍、离流形原生碰撞反例实际在 MuJoCo 几何上测试。
合成软件 fixture 标为 SOFTWARE_FIXTURE_ONLY，不充当新物理实验。

最终 smoke 完成后实跑 `resume`，前后 attempts/calls/completed 完全相同，
没有重置预算或重复生成/物理执行。发布审计可重跑，失败会直接报错。

## 可复现命令与证据位置

从真实仓库根目录执行；使用当前 `.venv` 或按 requirements.lock 建立兼容环境。
现有 `smoke_002` 要用 resume；新运行需新的 output 路径。

```powershell
.venv\Scripts\python.exe -m ff_prior.cli audit --config configs/ff_prior/v7_1.proposed.yaml
.venv\Scripts\python.exe -m pytest ff_prior/tests -q
.venv\Scripts\python.exe -m ff_prior.cli run --config configs/ff_prior/v7_1.proposed.yaml --profile smoke --output artifacts/v7_1/smoke_003 --dry-run
.venv\Scripts\python.exe -m ff_prior.cli run --config configs/ff_prior/v7_1.proposed.yaml --profile smoke --output artifacts/v7_1/smoke_003
.venv\Scripts\python.exe -m ff_prior.cli status --run-dir artifacts/v7_1/smoke_002
.venv\Scripts\python.exe -m ff_prior.cli resume --run-dir artifacts/v7_1/smoke_002
.venv\Scripts\python.exe -m ff_prior.cli report --run-dir artifacts/v7_1/smoke_002
.venv\Scripts\python.exe -m ff_prior.evidence.verify_release --run-dir artifacts/v7_1/smoke_002 --before-resume artifacts/v7_1/resume_002_before.json --tests artifacts/v7_1/tests_final.xml
.venv\Scripts\python.exe -m ff_prior.evidence.replay_sample
```

`smoke_003` 是供复现的新路径，本次没有运行它。证据目录包含一个保存的样条、
完整初态、完整 5250 条 ctrl、101 个原生检查点和可独立重放命令；
全量数据与 checkpoint 留在本地。参考/实际、基座、净空、末端误差及流形图位于
`ff_prior/evidence/smoke_002/visuals/`，代表视频来自原生 qpos 渲染，
仅用于展示，物理验收依靠原始执行与独立重放。

附加可携带重放工具的初版因遗漏记录前 forward 时序而比较失败；修正并缓存
解压后的 ctrl 后，实跑 5250 步、101 检查点最大差异为 0（6.577 秒）。失败及
修正前后的附加资源保留在 portable_replay_history.json，不改写原 25 次全轨迹结果。

首次 smoke 与最终回归各自的源代码和成本均保留。两轮 800 次提案只得到
399 个不同有效参考；累计重复和实际调用详见 `V7_1_OFFLINE_COST_REPORT.md`。
不提交或推送约 2.8 GB 运行数据，不改变 main、旧分支、旧数据或用户未提交修改。
