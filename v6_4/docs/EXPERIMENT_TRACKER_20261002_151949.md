# V6.4-A 实验进度

**状态快照：2026-10-02 15:19:49 +08:00。** 本版依据 terminal 文件更新；旧 timestamp 与全部失败原件保留。部署/硬件状态不随研究成功改变。范围见 [补充说明](V6_4_SCOPE_SUPPLEMENT.md)。

| Run ID | Milestone | Purpose | System / Variant | Split | Metrics | Priority | Status | Notes |
|--------|-----------|---------|------------------|-------|---------|----------|--------|-------|
| P01 | M0 | 冻结独立任务 | 三族 12 groups | 6 TRAIN / 3 VAL / 3 TEST | task SHA/group/初态/中段 FK | MUST | FROZEN | `output/tasks/protocol_b_01.json` 在新物理前冻结；SHA `dfddf2dca0c2f21e3153cf9e127eaf889e97e072129d324d1a36591f3384bdba` |
| R01 | M0 | 原行为保持 | planner disabled Protocol A | 历史五场 | 25/25、11/11、6755/12350 零失配、5×90 array_equal | MUST | COMPLETE | `output/regression_a_01/`；只属于旧协议回归，不计新学习测试 |
| R02 | M0 | 固定参考对照 | Protocol B 三族 fixed | TEST 3 groups | 3/3；40500 步；原任务/执行/区间/几何 | MUST | COMPLETE | `output/comparison_fixed_01/summary.json`；source 补核50应用+76assets；TEST trace 不进训练/Teacher prior |
| R03 | M1 | 启动拟合 prior | 六个 fixed TRAIN bootstrap | TRAIN 6 groups | 各27s实际/fresh独评 | MUST | COMPLETE_WITH_PRESERVED_RETRY | `output/teacher_bootstrap/`；首个 EE 原ModuleNotFound失败保留，`evaluation_retry_01/report.json` 补评通过且无新物理；其余result通过。不是Teacher标签 |
| R04 | M1 | 第一轮多任务数据 | 原 Teacher pilot | TRAIN/VAL 9 groups | 24次新raw拒绝+2条EE旧成功复用；26entries | MUST | COMPLETE_FAILED_TARGET | `output/cohorts/teacher_01/summary.json`；任务全部attempt，原覆盖目标未满足，不用新成功改写 |
| R05 | M1 | 三族第一份真实数据 | 各 train000 route0/1 | TRAIN 3 groups | 6 actual完整成功样本，三族各2标签 | MUST | COMPLETE_LIMITED | `output/datasets/three_family_01/manifest.json`严格注册；不同路线/拓扑仍待定量，六条≠六个独立任务 |
| R06 | M1 | 更多合格 Teacher | 第二 TRAIN groups/VAL；intrinsic native几何≤8轮×4 line trials | TRAIN/VAL | fixed prior隔离；完整gate+actual/fresh；全部开销 | MUST | IN_PROGRESS | 当前机制/验证进行中；不预报新样本成功，不使用TEST实际轨迹 |
| R07 | M2 | 初次训练链路 | `chain_smoke_01`单样本线性旧日程 | TRAIN 1 group | 500 optimizer steps、checkpoint/loss/source | MUST | TRAINING_COMPLETE_ONLY | 真权重存在；仅链路检查，非模型/闭环/泛化成功；旧checkpoint保留 |
| R08 | M2 | cosine修正诊断 | `chain_smoke_cosine_02`同组2路线 | TRAIN 1 group | 5000 optimizer steps，K1 raw0/K8 raw0，physics0 | MUST | COMPLETE_RAW_FAILED | `output/generated/chain_smoke_cosine_02/study_report.json`；八候选全保留，无repair/fallback |
| R09 | M2 | 数值放大归因 | TRAIN-only epsilon诊断 | TRAIN 1 group/2 routes | 原K8重现；0更新/0物理；新模型未测 | MUST | COMPLETE_DIAGNOSTIC | `output/diagnostics/cosine_epsilon_chain_smoke_02_01/report.json`；代数不等于已训练成功 |
| R10 | M2 | 真正可执行学习模型 | 固定residual epsilon新配置/新训练 | TRAIN / VAL | 新权重/normalizer/SHA；raw候选与实际闭环 | MUST | IN_PROGRESS_IMPLEMENTATION | 新参数化仍epsilon MSE，需新训练；不改日程身份/重解释旧权重，不裁剪输出；正式训练与成功尚未建立 |
| R11 | M3 | 独立主比较 | fixed / Teacher / Diffusion K1 | TEST 3 groups | 共9主组单元，raw/conditional/total与失败 | MUST | PARTIAL_FIXED_ONLY | fixed3/3完成；Teacher/Diffusion正式TEST暂留。主组无repair/fallback，不挑任务 |
| R12 | M4 | 候选搜索归因 | 同方法K8 | TEST 3 groups | 全8候选/固定预算/全部成本 | MUST | NOT_RUN_FORMAL | TRAIN过拟合K8不计正式TEST。网络/传统/repair/fallback分别归因 |
| R13 | M4 | 工程增强归因 | repair / fallback | 独立变体 | 来源/成本/分母 | NICE | NOT_RUN | 不加入直接网络成功分子，不用于填补主组未完成 |
| R14 | M4 | 全尝试结果报告 | task/安全/执行/性能/来源/局限 | splits分别 | 完整matrix + manifest + 定量路线差异 | MUST | PENDING | 首要缺口是学习raw参考真实执行与独立比较，不是RT优化；三TEST不能作大样本泛化结论 |
| H01 | 历史独立维度 | 部署/硬件边界 | C.1.1 native/HIGH / hardware | 历史 | wall NOT_MET；hardware NOT_ESTABLISHED | 非本轮门禁 | PRESERVED | 原默认墙钟全五场/三轮/26-11待办保留，管理员支线暂缓 |

## 本次核对的原件 SHA-256

| 原件 | SHA-256 |
|---|---|
| `output/regression_a_01/validation.json` | `115a76a0a0d59e4ff5aaebbf0bd2d56e5495b9ae18469ef22dd265a503772138` |
| `output/regression_a_01/execution_validation.json` | `51a4c2390d78b92084d7a3711539638b0d79008bf1d65e99d0f65940b795c46f` |
| `output/regression_a_01/interval_recompute/online_recompute_summary.json` | `b8ab0f6ad40acbcf339eec8f8c7d73ea0f57c69a7eb90f2bc394e970c26dd7c7` |
| `output/regression_a_01/non_timing_parity.json` | `f11794f9c9fbaf19e9448b3b983ed6ea10e450186b40a24a2c6d6793006a56ae` |
| `output/comparison_fixed_01/manifest.json` | `61223d56c0f35cd06e460d680d1e77140e88ed13c39c8166b0333383871d5293` |
| `output/datasets/three_family_01/manifest.json` | `b0831e526a32e95d37825372788ef55436d6173b24b8469345ef5c84de4d5dfb` |
| `output/cohorts/teacher_01/summary.json` | `cfaeaff52202fc6de197d4b77a9137dec9a5fffe84d7728244c42ffbd9da5413` |
| `output/generated/chain_smoke_cosine_02/study_report.json` | `b1259cd80c810525eff6b7bacf219801f47518e199bcebc383fa581c5a5d4055` |

源码身份：新增task/gate/eval在实际固定对照中 raw SHA 分别为 `7746550a5cc80b59c5c6f083fe4416d5943581100bc62189a408763363a17150` / `8ce72ecd57524b08cc888f1c999fa1e9d55f6a47a1930a0cb7c8d6b2c579ae15` / `76def5635be32540b5ad0c1d3dd6e6c9a844ba151be79c85ff32ca46c19efdef`；基点070不被写作新增源码commit。各新候选/训练仍需绑定自己运行时的源码与模型清单。
