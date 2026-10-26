# V6.4-A 实验进度

**状态快照：2026-10-02 17:09:31 +08:00。** 原始运行与此前timestamp保留。本版根据终态原件更新；`complete/evidence_valid`仅分别说明流程结束/证据合格，不表示模型或全配额成功。当前base070，sim20 ms/2 ms及全部原安全保持；wall20 ms非门禁，部署`NOT_MET`、硬件未建立。范围见 [补充](V6_4_SCOPE_SUPPLEMENT.md)。

| Run ID | Milestone | 试次/范围 | 真实结果 | 状态与归因 |
|---|---|---|---|---|
| P01 | M0 | `tasks/protocol_b_01.json`；12groups=6TRAIN/3VAL/3TEST | 新物理前冻结三族任务/初态/实际中段布局 | FROZEN；不是旧五场或多视角独立测试 |
| R01 | M0 | `regression_a_01`原协议A五场 | 25/25、11/11、6755/12350零失配、5×90 array_equal | COMPLETE；旧9cd的200/200+16/16不转移 |
| R02 | M3 | `comparison_fixed_01`三个TEST | 3/3实际完整PASS，40500采集步 | COMPLETE；raw=N/A，TEST actual不进入训练/prior |
| R03 | M1 | 六fixed TRAIN bootstrap | 完整27s/fresh评价；首EE缺evaluator失败保留后0新物理补评 | COMPLETE_WITH_RETRY；仅TRAIN prior，不是Teacher标签 |
| R04 | M1 | `cohorts/teacher_01`pilot | 24新raw拒绝+2旧成功复用，26entries/9groups | COMPLETE_FAILED_TARGET；后续成功不改旧拒绝 |
| R05 | M1 | `datasets/three_family_01`；三TRAIN groups×2routes | 六条完整actual成功；`route_diagnostics_01`数值差异已保存 | COMPLETE_LIMITED；实际刚性路径RMS约7mm，未证明不同拓扑 |
| R06 | M1 | `cohorts/teacher_02`原路径失败 | 3完整TRAIN+3失败部分VAL，1部分步数未知；131旧文件保留 | INTERRUPTED_FAILED；原先误称6完整已在correction明示更正 |
| R16 | M1 | `E:/v64_cohort_20261002_03`短路径恢复 | 13严格成功TRAIN/6groups；VAL0/TEST0，9groups均attempt | TERMINAL_TARGET_NOT_MET；复用6、0新采集补评3完整、另4新增完整；3同proposal VAL失败重试独立计成本 |
| R07 | M2 | `chain_smoke_01`单样本旧线性日程 | 500optimizer steps/真实checkpoint | TRAINING_COMPLETE_ONLY；不算raw/闭环/泛化成功 |
| R08 | M2 | `chain_smoke_cosine_02`同TRAIN2routes | 5000steps；K1/K8 raw0，八候选范围拒绝，0物理 | COMPLETE_RAW_FAILED；无repair/fallback |
| R09 | M2 | `cosine_epsilon_chain_smoke_02_01`归因 | 旧八候选重现，0更新/0物理 | COMPLETE_DIAGNOSTIC；稳定代数不代表训练成功 |
| R10 | M2 | `three_family_residual_01`纯EPS新训练及`three_family_residual_fit_01` | 20000steps/6TRAIN标签；TRAIN0/24、VAL0/24，TRAIN点4/192，0物理 | COMPLETE_RAW_FAILED；真实新权重，仍无学习actual |
| R17 | M2 | `three_family_epsilon_ema_sanity_02`训练/诊断 | 5000steps/192noise每步/960000抽样，固定末EMA；TRAIN0/24、VAL0/24，点125/192，0物理 | COMPLETE_RAW_FAILED；4→125点只拟合改善，不按VAL/TEST挑权重 |
| R11 | M3 | 原三方法K1主比较 | fixed3/3、Teacher0/3；Diffusionactual尚未执行 | PARTIAL_UNFINISHED；共9主单元固定分母，无repair/fallback |
| R12 | M4 | 同方法K8正式来源比较 | Teacher0/3已保留；Diffusion PENDING | PARTIAL_UNFINISHED；TRAIN K8不计TEST，不把prefix再计候选 |
| R15 | M3/M4 | `comparison_teacher_01`正式三个TEST；各8候选 | 24raw拒绝；K1=0/3、K8=0/3、0物理 | TERMINAL_REJECTED；K1为candidate0前缀，六failure cells完整 |
| R13 | M4 | `engineering_repair_driver_03`/`er_val01`，单EE VAL `task_projection_v1` | 八点通过；wrist_target min−17.633168899mm，fullgate拒绝、0物理 | STATIC_REPAIR_REJECTED_NO_ACTUAL；非raw能力；driver01/02异常原件也保留 |
| R18 | M2 | v5 `training/three_family_v_mse_03`及diagnostics同名 | 13TRAIN、fromscratch/seed68/5000steps/v_mse/416noise；唯一末EMA已预声明 | PENDING；本快照未报新训练/模型成功 |
| R19 | M4 | v2 `task_native_projection_v2`/`er_val02` | 同旧模型EE VAL seed64/cand0；fullgate PASS才另实际 | PENDING；工程独立，不裁剪/不terminalsettle，不充当主组 |
| R20 | M3/M4 | 正式Diffusion三TEST×K1/K8；短输出`v64diff01` | 预声明v5唯一末EMA，无repair/fallback；source/task/checkpoint冻结 | PENDING；缺候选/拒绝/未实际均保留在分母 |
| R21 | M4 | fallback | 未执行 | NOT_RUN；不能用隐式固定参考替换raw失败 |
| R14 | M5 | 全尝试三方法matrix/真实学习actual/有限总结 | fixed已完成，Teacher负结果完成；Diffusionactual缺口仍在 | PENDING；checks/manifest条数不当新trial pass |
| H01 | 独立历史 | C.1.1默认native/HIGH与hardware | 原默认五场未完整；wall NOT_MET，hardware NOT_ESTABLISHED | PRESERVED；不是当前门禁，不重复RT/管理员支线 |

## 路径恢复更正与成本

原 `receipt.json` 将六份新trace称完整是错误，`correction.json`明示3份完整TRAIN、3次失败部分VAL；旧文件不改。3份完整40500步复制后只fresh评价/注册，不新增控制器采集、不重复计成本。原阶段已知61220步+1次未知采集，恢复新采集82970步（包括同proposal重试28970）；两阶段已知下界144190步，连历史六示范81000步则下界225190步，**精确总数未知**。两个可读拒绝被不必要重试的事实保留，后续不再追加；一个缺partial写入的proposal仅作路径恢复重试。独立保存力矩重放、TRAIN fixed bootstrap和旧五场prior另计，不用这个下界冒充全部计算成本。

扩展数据13=EE train000/001各2/3、MID各2/2、MULTI各2/2；为6独立TRAIN groups，VAL=0、TEST=0，全success targets为false。原13标签不会把VAL部分拒绝注册为成功，也不读取TEST未来actual。

## 本版原件 raw SHA-256

路径以`v6_4/`为基准；外部短路径以完整绝对路径标记。每次运行源码/输入细项仍见自己的manifest/source snapshot，base070不是新增源码已提交的断言。

| 原件 | SHA-256 |
|---|---|
| `output/tasks/protocol_b_01.json` | `dfddf2dca0c2f21e3153cf9e127eaf889e97e072129d324d1a36591f3384bdba` |
| `output/regression_a_01/artifact_manifest.json` | `c685bba6eb5b05ca8a9b30a5c4c11e6af9d33e6535d28e1027f98a65d0e7fbd8` |
| `output/comparison_fixed_01/summary.json` | `daa2406006b499b2fab805486666cc0340ae660dcd9ae2722b3626413677e03c` |
| `output/comparison_fixed_01/manifest.json` | `61223d56c0f35cd06e460d680d1e77140e88ed13c39c8166b0333383871d5293` |
| `output/datasets/three_family_01/manifest.json` | `b0831e526a32e95d37825372788ef55436d6173b24b8469345ef5c84de4d5dfb` |
| `output/cohorts/teacher_01/summary.json` | `cfaeaff52202fc6de197d4b77a9137dec9a5fffe84d7728244c42ffbd9da5413` |
| `E:/v64_cohort_20261002_03/summary.json` | `7748e4505988676e60159ddcbd15f6c3161e697cb86cf68eb011f162a97ed2d7` |
| `E:/v64_cohort_20261002_03/dataset/manifest.json` | `f65ae1056026edbf582d6c28907b32f487cd71883ba9dd674bc77005db24660d` |
| `E:/v64_cohort_20261002_03/artifact_manifest.json` | `e8f51cc88a749415851e562df0411df0997eeee00a292196fafd281329e9bc72` |
| `output/cohort_recovery_receipt_03/receipt.json` | `28464e4c6d6f2384ea01d1d60285a93795dc178fcd2367340e903537b78d3133` |
| `output/cohort_recovery_receipt_03/correction.json` | `add632cb48255023c4ed9b300c3f0d58f266b914cef79263d6c564dd43d56661` |
| `output/teacher_cohort_resume_audit_03/report.json` | `60440dc6d4a524c631c80844ea7f7d1de3524afdbe796bf248997f1fb394d03d` |
| `output/training/three_family_residual_01/training_report.json` | `f12bea9bafb74d19cc3e3d66235ecd261f1dd103dd02575cf35ddb36b0ef81f5` |
| `output/diagnostics/three_family_residual_fit_01/report.json` | `b3adad7bda9d95c5bd13ccac62b310142921e6d032fdaf8a4de762502e102903` |
| `output/training/three_family_epsilon_ema_sanity_02/training_report.json` | `9512a0c9023592d18934b82aab5284a62b661cb96919448c8c7782975d0d7c49` |
| `output/diagnostics/three_family_epsilon_ema_sanity_02/report.json` | `303aa8ed837d2730b797d9241aafa4ba189f8297f9ba9f4fdc43f929a56d3696` |
| `output/comparison_teacher_01/report.json` | `eb49ab6d93da63d1c3b30f3a2f1b34b56b583449f6cf8ddf026abe4d53f6e061` |
| `output/comparison_teacher_readonly_summary_01/report.json` | `9e8a9f69862ae429d88b9de06d3da8841e938605996c198f04973f06fd329274` |
| `output/engineering_repair_driver_03/report.json` | `1715850676e3119606c81f7ceef2fa9f14594d83253b3ff1eab2e3a38bf69bc9` |
| `output/er_val01/candidate_000/repaired_full_gate.json` | `46ac3b2456034fd82a97015e86f7b17bf47bfb690304c889e8d9084d5182e2b9` |

源码身份：实际fixed对照的 task/gate/eval raw SHA 为 `7746550a5cc80b59c5c6f083fe4416d5943581100bc62189a408763363a17150` / `8ce72ecd57524b08cc888f1c999fa1e9d55f6a47a1930a0cb7c8d6b2c579ae15` / `76def5635be32540b5ad0c1d3dd6e6c9a844ba151be79c85ff32ca46c19efdef`。学习参数化/损失/权重变化是新模型身份，按每次source snapshot冻结，不能重解释旧checkpoint或把并行ML改动说成旧控制器变化。
