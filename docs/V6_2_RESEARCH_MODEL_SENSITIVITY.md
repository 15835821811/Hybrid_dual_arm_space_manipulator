# V6.2 有限模型惯量敏感性研究

预声明的五场景 × 三种惯量倍率研究已完成：每组 27 s、2 ms 步长，共 15 次、202,500 个物理步。倍率 1 的五组与原名义轨迹在九类保存量上严格复现，残差均为零。主惯量 ±5% 会造成毫米级末端执行偏差；降低 5% 时，最小观察间隙由名义约 15.00 mm 降至 9.03 mm。全部声明采样状态仍未发现低于 5 mm 或原生负距离，但这些结果不能构成未知模型误差下的闭环鲁棒保证。

这是原目标的补充研究。本轮保持模拟规划 20 ms、物理步 2 ms、安全半径、距离、容差、QP 和名义执行合同；纯研究不以墙钟 20 ms 为前置门槛。名义研究验收仍为 PASSED；原 C.1.1 墙钟部署仍为 NOT_MET，管理员支线为 DEFERRED/BLOCKED，硬实时与硬件安全未建立。2 倍目标速度实验的失败没有改成通过。

## 冻结协议与因素

[执行前计划](../v6_lite/output/runs/research_inertia_shadow_01/plan.json) 绑定生成源码 `19ad85c7db81a2e254fd176077b6aae39333a439`、239 个 Python 源文件和 76 个模型资产。机器人子树由命名自由关节 `world_joint` 所属 `base_of_satelltte` 及其全部后代确定，共 73 个 body，包括自由基座和两条臂。仅将其三个主惯量分量同乘 0.95、1.00 或 1.05；world、目标和工作空间不变。body mass、COM、armature、damping、geom、actuator 与数值选项逐项保持原值。

输入来自 [原名义五场景](../v6_lite/output/runs/research_acceptance_01/report.json)，源码 `9cd1831e7c77a340464a669eac3b5aa7cd9c3c3c`，manifest SHA-256 `379b9977a441c540a20cfc50a9e90614a1e0985fa2a5a73bbb7c9690f823eec0`。每个 2 ms 步原样输入保存的 67 路力矩，不重新计算反馈、QP 或参考。完整五组名义复现先执行，随后依次执行两个扰动组；未筛选场景或改变任务时长。

工具修改了 body inertia 后调用 `mj_setConst`，重置并恢复原始 qpos/qvel/time，再积分。扰动后的模型身份不同，原执行证书不适用于该模型。本实验为不提交到运行时执行器的、未经执行证书认证的开环数值 shadow，不绕过名义模型/载荷/状态/顺序/预演门禁取得通过。固定反射 armature 可能支配较小的连杆物理惯量，因此这里的 ±5% 结果不能外推到质量、COM、执行器输入或更广的模型误差。

[物理运行前测试资格](../v6_lite/output/runs/research_inertia_shadow_01/test_qualification.json) 在正式首步前保存；[当前/历史报告](../v6_lite/output/runs/research_inertia_shadow_tests_01/report.json) 为 220/220 加历史冻结 16/16。结果属于实际冻结的 19ad85c 源码，原名义 200/200 与压力实验 213/213 保留各自来源。10 步 [smoke](../v6_lite/output/runs/research_inertia_shadow_smoke_01/replay_report.json) 仍为 SMOKE_ONLY、不完整且不构成正式证据。

## 完整结果与暂停续作

原 [物理重放报告](../v6_lite/output/runs/research_inertia_shadow_01/replay_report.json) 完整 15 组、202,500 步，但阶段状态为 REPLAY_ONLY_GEOMETRY_NOT_RUN，不能把阶段报告改成最终通过。其 64 项 replay manifest SHA-256 为 `a8543ecf3a62c7adef87e7a698a0536860ab4a8a1800ed95905c870eef256a0a`。倍率 1 的五组在 planner q/dq、base qpos、task qpos、rigid tip/rotation、continuum tip/body origin/rotation 九类量上与原保存数组一致。

原几何观察在目标暂停时中断，进程 terminal exit 1。其 139 个文件、14 组完整观察、最后一组的部分 target 观察和 [暂停回执](../v6_lite/output/runs/research_inertia_shadow_01/goal_pause_receipt.json) 全部保留，原目录没有最终 report/manifest。不能声称原中断流程成功。

恢复后在独占新目录核对并逐字节继承 14 组完整结果，仅补 `scene_04_alpha_1.05` 的全范围观察，重新计算物理步为 0，terminal exit 0。[续作计划](../v6_lite/output/runs/research_inertia_shadow_resumed_01/resumption_plan.json)、原目录完整 inventory 和 [终态回执](../v6_lite/output/runs/research_inertia_shadow_resume_tool_01/terminal_execution.json) 保存了该事实；原目录所有 139 项 SHA/字节数仍相同。

[合并终态报告](../v6_lite/output/runs/research_inertia_shadow_resumed_01/report.json) 为 DIAGNOSTIC_COMPLETE、complete/evidence_valid 均为真。其 [最终 manifest](../v6_lite/output/runs/research_inertia_shadow_resumed_01/manifest.json) 覆盖 146 项文件，SHA-256 为 `ee98249e6ba936278e005943dc942724e57fb5ef2f59d2fe226b0e101d89158e`。这表示预声明有限诊断完成，不是新增扰动闭环 25/11/区间验收。

## 几何观察范围

每组的全部 13,501 个真实 2 ms 状态（含初始）核对原策略的 75 对机器人/目标几何；每组另对原 1,351 个 20 ms 状态及四等分配置插值，共 5,401 个配置，核对完整 2,927 对整机几何。75 对包含连续体、刚性臂及基座的 target 类，保留原两个命名终端抓取豁免；没有选择性增删 pair。配置插值不等于额外的实际物理步。

15 组共 15,188,625 次 target 查询与 237,130,905 次全整机查询，总计 252,319,530 次。这里不把原中断/续作的重复观察另计成新的独立证据。

所有查询使用当前 qpos 的独立 MjData 加 `mj_forward`，不直接把原 mj_step 后的运动学缓存当成当前积分状态。原生 `mj_geomDistance` 的 distmax 为 target 0.25 m、整机 2.5 m；返回 distmax 的值标记为截断下界，不当作精确距离。15 组全局最小值均未截断。10 组配对 target 最小距离差只使用双方都未截断的状态，分别排除 319–370 个状态；精确差的范围为 −5.967862 至 +5.203110 mm，不能将截断值直接相减构造精确响应。

这些是声明模型与 pair 策略下的有限离散观察，不覆盖每个 2 ms 状态的全部整机 pair，也不是连续时间 CCD 或安全证明。接触响应未启用；原生负距离计数是几何诊断，不是实体接触/冲击实验。

没有保存全部 pair 的逐查询距离；总查询量由冻结循环、完成计数和来源记录支持。独审没有全量重建类别最小值及截断查询计数，不能把有限样本核对说成全部原生查询的独立复算。

## 执行、跟踪和间隙响应

下表每行汇总对应五场景：间隙列取最小值，误差列取最差单场景值，单位 mm。跟踪误差采用 fresh 当前状态；连续体参考为原时间对齐的路径参考，刚性臂抓取参考由同一当前目标姿态重建。连续体只取原路径窗口，刚性稳态取最后 1.5 s，初始状态不纳入误差统计。

fresh 当前状态的误差统计与原 mj_step 后缓存日志的误差统计不同；原名义验收仍按原声明数组判定，不用本表改写其数值或结论。

| 惯量倍率 | target 最小间隙 | 全整机采样最小间隙 | 连续体路径 RMSE | 连续体路径最大误差 | 刚性臂最终误差 | 刚性臂稳态 RMSE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1.00 | 14.996316 | 14.996316 | 0.108120 | 0.507292 | 0.002765 | 0.004372 |
| 0.95 | 9.032702 | 9.032702 | 2.525157 | 3.335631 | 6.816825 | 6.687787 |
| 1.05 | 16.541059 | 16.541134 | 2.353197 | 3.088831 | 6.279526 | 6.164008 |

全部 15 组在所述采样中低于 5 mm、原生负距离的状态与查询计数均为 0。降低惯量减少了观察间隙；增加惯量在这组输入下增加间隙，但两侧的跟踪误差都明显变大。刚性最终误差与连续体路径 RMSE 已高于原名义 0.10 mm / 0.18 mm 的数值门槛；这里仅作相同窗口的诊断比较，没有对扰动 shadow 重贴原名义功能验收标签。

[保存结果汇总](../v6_lite/output/runs/research_inertia_shadow_analysis_01/summary.json) 和终态的十组配对结果还显示：67 路低层关节最大偏移 0.010502 rad；自由基座平移最大偏移 64.447 μm、姿态 0.165034°；刚性末端最大偏移 6.817033 mm、连续体末端 3.543718 mm；对应末端姿态最大偏移 0.383998° / 0.175012°。这些是相对同场景名义轨迹的偏移，与目标跟踪误差不是同一量。

## 独立核对与原始失败

[独立审计 02](../v6_lite/output/runs/research_inertia_shadow_independent_audit_02/replay/report.json) 为 2,395/2,395，核对全 15 组保存记录、全部模型参数/输入因素、来源和时序；另外独立执行五场各前 20 步，共 100 个名义物理步，残差均为零。没有独立重放全部 202,500 步。

[几何独审 03](../v6_lite/output/runs/research_inertia_shadow_independent_audit_03/geometry/report.json) 为 1,898/1,898，独立重建所有 15 组的保存最小值及固定代表状态，实际重新查询 254,045 次，包含降低惯量后的不利间隙状态；另外重算保存 fresh frame 的原路径/稳态/末态位置统计，并核对参考时序。没有增加物理步。[保存数组完整性独审](../v6_lite/output/runs/research_inertia_shadow_evidence_audit_02/report.json) 02 为 768/768，核对全部 146 项 SHA、15 个唯一运行、45 个观察 NPZ、十组配对统计、实际时间窗口及原 139 项中断文件。path 窗口 4.5–25.5 s 各有 10,500 个 post-step 样本，稳态窗口各有 750 个；按保存的实际浮点时钟取样，不换成理想网格增加边界样本。十组共 131,555 个双方未截断状态、3,455 个排除状态，合计 135,010 个。有限独立几何样本不等于重新计算全部 252,319,530 次查询。

保存证据审计 [01](../v6_lite/output/runs/research_inertia_shadow_evidence_audit_01/report.json) 保留 terminal exit 1、763/768：审计器把声明的末态三维向量范数换成批量 axis 范数的末元素，在五个运行上产生 1 ULP 的严格相等误判。独占 02 只修正计算形状，按保存的末态三维向量重算，未放宽容差、未改 producer 或重跑物理/几何；原失败及 [修订回执](../v6_lite/output/runs/research_inertia_shadow_evidence_audit_02/previous_audit_receipt.json) 保留。

首个 [审计 01 失败](../v6_lite/output/runs/research_inertia_shadow_independent_audit_01/replay/report.json) 因 MuJoCo 解析包含中文的绝对 URDF 路径失败，terminal exit 1，实际物理步为 0，不能用于物理一致性结论。修正成仓库 cwd 下相对 URDF 路径的第二次实际执行曾从 01 路径发起，随后将第二版本证据移入独占 02 并恢复首版本脚本/计划/来源/清单；[迁移回执](../v6_lite/output/runs/research_inertia_shadow_independent_audit_02/migration_receipt.json) 如实记录 100 步只执行一次。01 有额外 revision 副本，不能声称整个 01 目录从未改变；首版本原件及失败回执的原始字节已核对保留。

[磁盘与 Git 归档核对](../v6_lite/output/runs/research_inertia_shadow_archive_check_01/verification.json) 保存完整/中断主试验、续作、测试、smoke、失败和正式独审、原草稿与结题核对的原始字节，分别验证文件 SHA、字节数及 Git 暂存区原始字节。原机器状态 57 个根字段值原样保留，新结果仅追加，历史 NOT_EXECUTED 计划另由新增范围字段解释为执行前快照。源码与资产实验期间未改变；归档后文档提交不改变冻结实验的来源。

## 本补充目标的完成范围

当前补充完成了非实时研究入口与名义验收、保守性/形状归因、有限复杂任务尝试及模型导致的执行敏感性研究。2 倍速度失败给出了原控制器的任务能力边界；惯量实验表明同一力矩下精度和间隙会改变。失败是已完成研究的结果，不以放宽 R、距离、容差、QP 可行性或执行一致性换取通过。

本轮没有独立施加力矩/增益/通信延迟扰动，没有重新求解扰动反馈闭环，没有校准连续体包络误差界、推导鲁棒 CBF 条件或验证安全备份。原目标中的墙钟完整五场景、正式三轮及新原生 26/11/区间验收仍未完成，继续作为独立部署目标。下一次若要声称更强的模型鲁棒性、复杂任务成功或真实执行能力，应另行确定具体范围和协议；这些扩展不是本有限补充结题的前置条件。
