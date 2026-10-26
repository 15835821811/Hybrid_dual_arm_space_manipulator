# V6.4-A 实验进度

**状态快照：2026-10-02T21:59:01+08:00。** 原始运行与此前timestamp保留。本版根据终态原件更新；`complete/evidence_valid`仅分别说明流程结束/证据合格，不表示模型或全配额成功。当前base070，sim20 ms/2 ms及全部原安全保持；wall20 ms非门禁，部署`NOT_MET`、硬件未建立。范围见 [补充](V6_4_SCOPE_SUPPLEMENT.md)。

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
| R11 | M3 | 原三方法K1主比较 | fixed3/3、Teacher0/3、Diffusion0/3，后两者raw全部拒绝、actual0 | TERMINAL_NEGATIVE_MATRIX；共9主单元固定分母，无repair/fallback |
| R12 | M4 | 同方法K8正式来源比较 | Teacher/Diffusion各0/3、各24原始拒绝、actual0 | TERMINAL_NEGATIVE_MATRIX；TRAIN K8不计TEST，不把prefix再计候选 |
| R15 | M3/M4 | `comparison_teacher_01`正式三个TEST；各8候选 | 24raw拒绝；K1=0/3、K8=0/3、0物理 | TERMINAL_REJECTED；K1为candidate0前缀，六failure cells完整 |
| R13 | M4 | `engineering_repair_driver_03`/`er_val01`，单EE VAL `task_projection_v1` | 八点通过；wrist_target min−17.633168899mm，fullgate拒绝、0物理 | STATIC_REPAIR_REJECTED_NO_ACTUAL；非raw能力；driver01/02异常原件也保留 |
| R18 | M2 | v5 `training/three_family_v_mse_03`及diagnostics同名 | 13TRAIN/fromscratch/seed68/5000steps/v_mse/416noise；TRAIN0/48、VAL0/24 | TERMINAL_RAW_REJECTED；不是模型成功 |
| R19 | M4 | v2 `task_native_projection_v2`/`er_val02` | 旧模型EE VAL seed64/cand0；静态gate PASS，actual11560步/23.120s局部QP拒绝 | TERMINAL_PARTIAL_FAILED；工程独立，首中断成本UNKNOWN |
| R20 | M3/M4 | 正式Diffusion三TEST×K1/K8；短输出`v64diff01` | 预声明v5唯一末EMA，24raw拒绝，K1/K8各0/3、0物理 | TERMINAL_REJECTED；全部候选和分母保留，无repair/fallback |
| R21 | M4 | fallback | 未执行 | NOT_RUN；不能用隐式固定参考替换raw失败 |
| R14 | M5 | 原主matrix/独立工程actual/有限fresh-prefix/自动报告 | 原raw负结果与18工程cells完整；4独特33540实际步全部失败 | LIMITED_NEGATIVE_DELIVERY_EVIDENCE_READY；rawcredit0，非性能成功 |
| R22 | M4 | unique v5 EE VAL raw candidate0独立preflight | 原native min−10.800966mm，安全0/1，0物理 | TERMINAL_REJECTED；不查c1..7/其他VAL |
| R23 | M2 | B finite Teacher-prior residual Transformer | TRAINraw0/48，VAL/TEST/物理0 | STOPPED；prior与network分列，无新TEST先验 |
| R24 | M2 | C empirical conditional score | TRAIN48/48 gate∧ddq，VAL0/24，TEST/物理0 | STOPPED；非Transformer/新拓扑，无泛化成功 |
| R25 | M4 | E:/v64eng02公平工程3TEST×3方法×8 | 72slots/18cells，Diffeligible16/24，其他0/24；六K分组Task均0/3 | TERMINAL_COMPLETE_NEGATIVE；repair/raw/选择/成本独立 |
| R26 | M4 | V5 TEST修复后4独特actual | 12240/13310/4040/3950步，共33540；全iteration_limit拒绝 | PARTIAL_TASK_FAILED；原row evaluation=null/false保留 |
| R27 | M5 | E:/v64eng02eval01保存力矩prefix评价 | fresh force replay33540步，4case evaluator ERROR/false | ERROR_PRESERVED；未消费NaN拒绝行误入finite检查 |
| R28 | M5 | E:/v64eng02evalrec01已存prefix有限恢复 | 4/4 physical-prefix valid、合同/区间/robot-target通过；wholebody NOT_AVAILABLE | SAVED_ONLY_RECOVERY；0新controller/replay/geometry；非全27s/Task成功 |
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

## 最新终态与停止条件

v5 `three_family_v_mse_03` 已完成真实5000步训练，13条TRAIN、每步416噪声、固定最后EMA；TRAIN raw0/48、VAL raw0/24，全部原始拒绝保留。正式三TEST的Diffusion K1/K8已完成：24原始候选均拒绝、0实际物理；三方法主比较终态为fixed3/3、Teacher0/3、Diffusion0/3。Teacher/Diffusion接受后条件完成率均N/A，不能把0接受写成0/0成功率；fallback=0。

为了判断是否存在可安全执行的原始负结果，另对既有TEST候选做完整原生几何核对：candidate0正式preflight三任务均不安全；其余21候选也均不安全，故24/24拒绝，未启动物理。三个完整candidate0审计为1351状态×2927pairs各3,954,377查询；额外21候选83,041,917查询。更早candidate0几何探针11,863,131查询另计，不重复称新候选。K8汇总脚本因两种report schema键名不同而失败，原件保留；独立汇总恢复复用了已存结果并核对输入/source/assets，没有重新计算几何或执行物理。负signed distance仅作unsafe indicator。

工程v2 `er_val02` 的静态完整原gate通过、最小间隙6.618282983 mm；实际独立VAL在23.120 s、11,560力矩步后因原局部QP不可行拒绝，未完成27 s。拒绝状态的原domain下界+0.0207782642 rad/s与原acceleration上界−0.0361568378 rad/s互相冲突，原容差仍无法消除差距；它只证明该到达状态的局部QP不可行。先前中断试次未保存trace的物理步数仍UNKNOWN，重试成本独立保留。此实际证据属于task_native_projection_v2工程修复，不能重标raw。

机制A `three_family_clean_x0_01` 为新v6 checkpoint身份，20,000步×416=8,320,000噪声样本，从头训练，耗时222.326 s，固定唯一最后EMA。已有TRAIN六任务×8候选全部原raw拒绝，单项要求255/384通过；48/48失败包含Task项，原gate前置失败时几何未运行，不能称候选安全。遵循预声明停止条件，VAL/TEST/physics全部未启动，不追加步数。35 unique focused tests通过，scope修正后同一测试定向复测通过；不把复测计为第36个独立测试。七源和1,432份旧文件保持不变。

## 最低交付边界更正与本轮终态

两项自产加严均已纠正：原附件第323行允许各方法同预算工程增强组，第400行允许首版得到失败或劣于传统方法的可判断结果，第407—409行要求独立任务的实际执行、参考/命令/力矩/失败及来源归因；它没有要求最低交付必须有 accepted raw，也没有限定实际证据必须来自最终V5的正式TEST。旧timestamp、旧review prompt/output与原始run保持字节，不回写历史。当前raw0仍是性能负结果，不能删除或改称工程成功。

[有限范围审查](../output/minimum_delivery_scope_audit_01/report.json)核对旧a88 checkpoint训练仅来自3个TRAIN groups（0VAL/0TEST优化），其EE VAL group独立。因此旧repair-v2的11560步/23.120s真实失败prefix可作为有限独立任务证据；它非raw、非TEST、非V5、不是完整fresh验收。当前又增加了真实V5正式TEST修复组执行，不必用旧VAL替代这一更强来源。

统一工程02已终态：3TEST×3方法×8=72实际处理slots，K1/K8共18计分cells，三个方法各K的Task成功均0/3。fixed_joint_prior/Teacher修复后资格各0/24；Diffusion资格16/24。K1是candidate0，K8按原静态score/index选择，不按actual结果换选、不重试失败。固定joint seed每Task八次相同，全部成本保留，不称八路线，且不是原Cartesian fixed3/3基线。rawcredit=0、fallback=0。

Diffusion有4个独特工程actual prefix：t0 K8为12240步/24.480s，t1 K8为13310步/26.620s，t2 K1为4040步/8.080s，t2 K8为3950步/7.900s；均在下一servo之前iteration_limit拒绝，全部Task失败。控制器保存力矩采集共33540步；独立保存力矩fresh重放另33540步，不能相加称67080新增实际试验步。私有十步preview总步数未单独报告，保留UNKNOWN。

原工程row的evaluation=null/actual_evidence_valid=false保留。独立prefix评价因把最后一条未消费NaN拒绝行纳入命令完整性而ERROR/false，原错误保留。之后只读已存fresh状态恢复4/4 physical-prefix证据：每份恰一条未消费拒绝行独立标明，已消费命令执行合同通过、独立区间零失配；robot-target原重放最小距离分别6.979929849/17.336940858/6.669432463/5.785685757mm。全身原生结果因原局部返回未序列化为NOT_AVAILABLE；不能把robot-target通过写成whole-body通过。恢复0新控制器/0新force重放/0新geometry查询；完整27s/full-trial/Task成功均false。

A、B、C机制均按原有限停止条件结束：A clean_x0 TRAINraw0/48，VAL/TEST/物理0；B Teacher-prior residual Transformer TRAINraw0/48，VAL/TEST/物理0；C经验条件score模型TRAIN48/48原gate∧ddq通过，但VAL0/24，TEST/物理0。C不是成功Transformer或新拓扑。三个机制不再追加预算，TEST coordinator均未执行；没有用失败center执行或未来TEST actual作输入。

[自动机器报告](../output/delivery_summary_final_01/report.json)、[可读报告](../output/delivery_summary_final_01/REPORT.md)与[输入/产物manifest](../output/delivery_summary_final_01/manifest.json)已生成，有限负结果最低研究交付证据具备。该结论不声称raw学习成功、完整工程Task成功、网络独立贡献、优于固定基线、广泛泛化、拓扑差异或鲁棒/连续安全。原fixed Cartesian 3/3、Teacher/Diffusion raw K1/K8 0/3主表不变；学习/工程性能结论为负。goal状态最终由根代理处理，不靠扩大试验填自产门槛。

用户补充仍追加原V6.4、base070不变；sim20ms/2ms与全部原Task/QP/PCC半径/工作域/5mm/导数/力矩限制不变。wall20ms是诊断性能，旧wall NOT_MET、hardware NOT_ESTABLISHED保留，不重复RT/管理员支线。

运行后报告路径修复另在准备，尚不写其测试通过或SHA；这不影响本轮已冻结运行来源，不能回填旧row/evaluator原件。


## 新增终态原件锚点

| 原件 | SHA-256 |
|---|---|
| [三方法主比较负结果](../output/comparison_main_01/report.json) | `680013a27847370c1addb738c55bafd05081e3f4868578a43caf9c82685b7a51` |
| [v5训练终态](../output/training/three_family_v_mse_03/training_report.json) | `044296590f741ab2f447253428d0e5055ef5c8b1479e55e929d8ba935dc6188d` |
| [v5原始采样诊断](../output/diagnostics/three_family_v_mse_03/report.json) | `1b0e10ce6d918f067ce6eea78b2d11ce6548e959cf862cc006487f8ef075238d` |
| [工程v2失败独立核对](../output/engineering_repair_v2_failure_audit_01/report.json) | `6c228e08b2b6d6557981c8a1b770afdd0db787a69875ef0d86fb945232ddcd7b` |
| [独立TEST candidate0安全拒绝](../output/raw_closed_loop_diagnostic_driver_01/preflight/report.json) | `b04f98480b187598143a5b3a92e92c6baee0cde40e241967950aace3cb412b95` |
| [全24 TEST原生几何拒绝及汇总恢复](../output/raw_k8_geometry_probe_01/report_recovered.json) | `333425d2dc5b1a21e49883bf8dd5574b25867fe0636cd5ce64687e3a4555be9b` |
| [clean_x0训练终态](../output/training/three_family_clean_x0_01/model/training_report.json) | `29ba6034b49b07dd9e74720c9aa4e59f00503763d40c273a4e771b5ff653609a` |
| [clean_x0 TRAIN48拒绝终态](../output/training/three_family_clean_x0_01/diagnostic/report.json) | `6c62251940e226a1ff1e6d5f311e6d3bebf511aab1dfa3052e0ac559c83fea42` |
| [已存TRAIN48精度与多样性](../output/training/three_family_clean_x0_01/saved_failure_summary.json) | `3d689d4119f9a7b39a0f25b78a9f45742a6094ab48d4b6964ee5ce8010e678e8` |

## 本轮新增原件SHA

| 原件 | SHA-256 |
|---|---|
| [自动V6.4-A机器报告](../output/delivery_summary_final_01/report.json) | `fd0057b1e81b49672cffcc7c0220d011b0c140a34e2c0cba51d705f7476ec579` |
| [原工程02报告](E:/v64eng02/report.json) | `cbb625532cac0342d79a3515e1a42ae54ab1b56c20e2dac3503f21affd3020b8` |
| [原prefix ERROR报告](E:/v64eng02eval01/report.json) | `9403711c69334255ee100e5c6228bf982a430a9bbb5eb86a5359df859dbfaf4f` |
| [已存prefix有限恢复](E:/v64eng02evalrec01/report.json) | `aeaa2914fd81c80cfc56f90f8dee4dcc3c5a75760300cd9dbee4b7fb1c310c58` |
| [最低范围独立核对](../output/minimum_delivery_scope_audit_01/report.json) | `c6e86e3414cc3c258188a4524a94a0a458b2252fcf0e548839ace60445183d46` |
| [B资格终态](../output/training/three_family_teacher_prior_01/raw_qualification/report.json) | `4a1060342b8c81fd7f1be2eb49324864f59ebd7618f55ecf243433e3747fde20` |
| [C终态保存汇总](../output/empirical_score_result_01/report.json) | `4959643d63c37cf402945ff8c7b4b7cbaf26a3e4f8a876a753977f5c4888b47b` |
