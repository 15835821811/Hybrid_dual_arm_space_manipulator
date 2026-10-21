# 独立 paper architect/reviewer：capture_refine_001

裁决：**claim_supported = no**。C1（原五障碍下合格绕行路径及原生67力矩执行/重放见证）未成立：MODEL_VALID 为 **0/2**，实际执行为 **0**。C2（有合格来源后的102维 codec / prepare / 同尺度修复资格）为 **BLOCKED_NO_MODEL_VALID_WITNESS**，不是表示失败。对于这两个结论，confidence = high；缺少 `paper/PAPER_CLAIM_AUDIT.json`，论文写作裁决标记 **provisional — no paper claim audit available**。独立 reviewer 已执行，不标记 pending external review。

本审查只读冻结源码、selection/binding、运行 JSON 与 NPZ；`compact.json`、`raw_manifest.json`只用于定位和哈希索引。未运行项目审计器、MuJoCo、FK、distance、积分、优化、仿真、旧审计或 resume。独立脚本 `independent_algebra.py` 仅使用保存数组、NumPy 与 Rotation 的矩阵代数，结果见 `independent_algebra.json`。本审查仅写当前 trace 目录。

## 证据身份与运行范围

133 个执行源码文件当前哈希及 executed_source.zip 内容一致；192 个绑定输入/旧源码、121 个运行清单文件均匹配，无失配。producer_head 为 `345e1e5da1c2d5e64b6783026e4f3d8983f06d45`；启动记录提交 `a8771c9f` 不应被写成新的算法版本。task_hash 为 `e66384242aaacf174ab1d7e642f0ba84b55ea580ab1b62951988409c75115496`。本轮 selection.entry 与上一轮 entry 完全相等，含五个原球障碍、原双臂世界目标和原初态身份；物理验收合同相等。两条 seed 的 Z 和 reference_hash 均绑定上一轮最终 validation；实际开始 Z 与冻结值逐元素完全相同，clip 未改变任一元素，未退回初始弓或重抽起点。

`propagation.py` 是显式非均匀 phase 上的同一零动量自由漂浮中点方程，保持17→原生关节映射和完整初态 reset；没有用关节路径插值伪造基座。`experiment.py:103` 的 M0 实际运行旧/新512均匀节点传播。保存 NPZ 的 qpos、EE 差均严格0；geom、geom_rotation、velocity 的0差只有运行 JSON，三组完整数组没有保存，不能声称本 reviewer 从原始数组重新核算了这三项。

## 求解器与反馈确实做了什么

读取 `solver.py:93` 的约束与 `solver.py:149` 的优化循环，并独立重算六组保存差分：每组323个非零、域内、绝对值≤1e-4 rad 的扰动，共1938列；由 perturb_ee 与 perturb_distances 重算 jg/jd，与保存值逐元素相同。这里只证明保存差分的代数一致，不证明导数的步长收敛或优化器收敛。

每次目标约束为四组归一化误差的球约束，上限 max(1,当前组范数)，所以 u=0 可行。full323 基底确为标准控制点坐标基底；变量乘0.15后受真实控制点的域与±0.15 rad限制。独立重算 initial/solved/returned 的 pose_min、control_min、最大控制步长、目标值和初值归一化，全部一致。

六次 SLSQP **全部 success=false，message="Iteration limit reached"，nit=160**：

| 路径/轮次 | solved pose_min | returned | 可行 callback / 161 | 接受尺度 |
|---|---:|---|---:|---:|
| 00 / 00 | -0.0008901191624 | ZERO，pose_min=0.9759734302 | 0 | 无 |
| 07 / 00 | 0.9999999946 | BEST_FEASIBLE_INCUMBENT | 159 | 0.125 |
| 07 / 01 | 0.9999944700 | BEST_FEASIBLE_INCUMBENT | 158 | 0.125 |
| 07 / 02 | 0.9999996890 | BEST_FEASIBLE_INCUMBENT | 159 | 0.125 |
| 07 / 03 | 0.9999964441 | BEST_FEASIBLE_INCUMBENT | 158 | 0.25 |
| 07 / 04 | 0.9999999916 | BEST_FEASIBLE_INCUMBENT | 155 | 无 |

161条含160次迭代 callback 和求解返回后的额外 callback；不能写成161次优化迭代。00 的 solved 不可行，返回原零点，最大控制步长0；其余五次 solved 与返回可行向量相同，最大控制步长0.15 rad。每条 callback 的可行标志与所存残差吻合，返回 objective 等于初始零点及可行 callback 中最小值。callback 的完整 u 没有逐条保存，因而无法从向量重算中间各条残差；solved_u 与 returned_u 的残差则已独立重算。保存可行 incumbent 是保护机制成功，**不是 SLSQP 收敛成功**。

六次反馈均用512基准节点并入当前精细 active phase，得到609、742、745、745、746、744个区间，即新增97、230、233、233、234、232个节点。每组240个 active 三元组唯一；全部相位精确属于 dense_phase，也被精确保留于非均匀 phase，无最近点替换。由保存的 current_ee/dense_ee 独立核算最大位置差3.587582e-6 m、姿态差1.876637e-6 rad，均远低于1mm/0.2°门。运行记录的最大固定对距离差为2.369368e-6 m；**未保存 dense 每个 active pair 的距离及 sparse qpos，无法在禁止物理重算的条件下完整独立复算该距离最大差、top-240排序或其物理距离来源**。这里应写“源码实现且运行记录通过”，不能写“全部稀疏/精细距离已原始数组独立证明”。接受试探的 phase/qpos/minima/EE 与下一轮 dense 保存数组完全相等，说明精细反馈进入了下一轮线性化，并非仅末尾验收。

独立核查27个保存试探的 Z 更新、minima最小值、碰撞罚、四组物理目标误差与接受布尔值，全部吻合。四个接受步实际最大控制变化分别0.01875、0.01875、0.01875、0.0375 rad；所有接受前已合格的目标分项均保持≤1。07 第一、第三、第四、第五轮全尺度试探位置组超门，分别 Q=1.05453、1.08227、1.05028、1.40817，均被拒绝；其他拒绝试探也没有降低规定的碰撞罚。由于这些全尺度试探同时恶化碰撞，本轮不能单独因果证明逐分项保护对结果的贡献。

## 最终路径资格

| 路径 | 初始/最终精细网格 | 初始最小距离 | 最终最小距离 | 最终最大位置误差 | 最终几何步长 | 资格 |
|---|---|---:|---:|---:|---:|---|
| 00 | 1024 / 1024 | -41.011134 mm | -41.011134 mm | 0.775122 mm | 1.656234 mm | REJECTED |
| 07 | 16384 / 16384 | -5.143923 mm | -2.154987 mm | 0.463022 mm | 1.356261 mm | REJECTED |

00 控制点完全不变，零步的六个尺度并未产生新路径。07 同一16384精细网格下接受四步后，最小距离改善2.988936 mm；碰撞罚0.80257047降为0.17289688。四步后最大归一化目标误差从0.00154566变为0.09260445，仍在5mm/1°内；这是允许门内精度交换，不能写成所有目标误差同时改善。最终双臂位置误差0.463022/0.144156 mm，旋转误差0.000263962/0.000294100 rad。两条最终最深穿透均为 v7_capture_3 与 collision_0069（fore_arm_link）。接受过程最深对还涉及 wrist_link 与 target_satellite/end_link，不能概括为只修了一个障碍的单一碰撞。

独立读取00的256/512/1024及07的256/512/1024/2048/4096/8192/16384全部10个网格的 qpos/base/ee/body/geom/geom_rotation/velocity/phase，均有限。重算几何步长、对齐几何收敛、终点位置/旋转差，与记录一致。07的512至16384几何步长依次43.387791、21.698572、10.849899、5.425020、2.712520、1.356261 mm；直到16384才通过2mm门。其各级已扫描网格均记录负距离；最后收敛的 aligned extent=1.461398e-8 m。00在1024通过步长及收敛门。原零穿透门保持，`validation.py:36` 不抹除较粗网格已经检测的穿透。

验证 grid NPZ 保存了完整传播数组，但不含逐状态碰撞距离；末次网格可与 final_path 的保存 minima对应，中间网格的距离证据仍是 JSON 扫描摘要，不能在不调用物理距离的前提下重算。256级只做传播收敛基准，没有独立碰撞扫描摘要；更细512网格包含它的相位。没有任何离散网格被升级成数学连续无碰撞证书。最终失败由约2.155mm和41.011mm的保存穿透直接解释，数值收敛不能代替零穿透门。

`teacher_execution_selection.json` ids=[]，ledger 中 physics_steps 与 torque_rollout 均0，且无执行单元；0执行表示没有合格候选，不能称执行成功率0%或执行失败。`experiment.py:155` 的102分支未进入，本轮没有 codec往返/prepare/102修复的实测资格或执行结论；其控制域映射约束存在于源码，并不等于102分支已经运行通过。

## 预算与完成性质

从两条 initial grid_history、每轮 feedback与323次扰动、27个试探grid_history、10个最终验证网格和M0两次传播，独立精确重建 **1991次耦合 / 1,822,572积分区间**，与budget及ledger相符，低于12000/2400万；wall=398.641s。两条均以 NO_ACCEPTED_STEP 停止，未触达外层12次上限，不能把停止原因归为总计算预算耗尽。6个内部 SLSQP 则都触达各自160迭代限。

当前原始run体积557,509,601 bytes。源码12GiB约束是 **output_bytes_cap（输出磁盘体积）**；2CPU通过OMP/MKL/OPENBLAS线程数2设置，未见CPU亲和性硬隔离或进程峰值内存记录。因此能核对时间、计数与输出体积，不能声称验证了12GiB RAM峰值或硬CPU配额。所有已完成单元账本无inflight；训练、扩库、TEST不在本轮执行分支中。

主执行器另保存 `evidence/completed/completed_resume.json`：记录完成后resume仅一次、121个文件前后哈希一致、身份及四个用户文件未变，但Windows编码导致stdout捕获失败，resume_response=null。本 reviewer没有再次resume；目前支持“原run文件与计数未变”，不能引用缺失的 `NO_WORK_ALREADY_COMPLETE` 实际回执，亦不声称独立观察了调用次数。该限制应原样进入handoff。

## 主张修订与后续路由

建议论文/分析表述：“对两个预先冻结的失败路径实施一次有界组合修正。在保留物理目标资格的条件下，07在同一16384采样网格的碰撞罚与最深穿透有所改善；00未变化。两条均未达到零穿透MODEL_VALID，故没有执行见证，102维表示诊断未启动。”

不支持：合格绕行、执行或重放成功、全局不可达、102维表示不足、Diffusion新收益、一般化性能提升、任一单项改动的因果贡献、SLSQP收敛。一个APPLICATION_DEV任务的两条固定路径也不是独立任务成功率基准。组合改动同时改变初始化、尺度约束和碰撞反馈，新增精细网格还增加计算量，不能做单项归因。

缺失证据首先是严格MODEL_VALID路径，其后才是冻结选路的原生力矩执行与独立重放、codec/prepare/102分支同门验证；当前不进入论文确认或消融。当前授权运行已结束，不新增实验、不扩预算、不追加seed、不重跑旧结果。

若用户另行授权下一轮，先单独冻结新的协议：针对局部模型到全路径试探的巨大偏差，记录active集合更替、每个active pair稀疏/精细原始距离、callback向量及KKT/驻点诊断，并检验实际信赖域与active覆盖的局部可靠性；不能仅凭本轮效果假定“多迭代必然成功”。未来M0保存全部被比较数组；最终各网格保存逐状态距离/对标识。先获得C1合格路径及执行证据后再进入C2；因果拆分实验仅在有正向核心见证和独立授权后规划。当前 routing_action = **revise_claim_and_stop_current_run; future protocol change requires new authorization**。
