# 有限结构实验 B：Teacher 先验辅助残差 Diffusion

状态：2026-10-02，代码与验证阶段已授权；尚未生成中心、训练或实际执行。它是原 V6.4-A 主目标的后续有限机制，不改旧 absolute-network 负结果。墙钟 20 ms 只作研究性能指标，仿真 20 ms/2 ms、原 Task/安全/执行合同不变。

此前 clean_x0 固定20k训练的六TRAIN×K8全部原始拒绝，主要阻断是任务精度。32CP可精确表达全部13成功标签，任务条件包含必要点、姿态和时刻。机制B检验一个不同的结构假设：生成前有限传统规划承担共用路径与Task关系，网络学习条件化路径残差。不得把先验收益归给网络。

## 固定结构与输入

对每个Task使用同一个有限、显式的新Teacher先验算法。输入只有该Task初态、声明的目标运动、障碍、Task点与冻结历史SC04通用bootstrap；TRAIN/VAL/TEST均采用这一来源规则，不用当前VAL/TEST未来actual。原13条成功TRAIN标签、六TRAIN groups全保留。

中心P由有限候选确定后，构造每条标签的自由控制点残差R=C_free−P_free。新条件为原517维Task条件和510维先验自由CP：原517正常化仅拟合13 TRAIN标签；中心CP直接以physical radians按row-major展开，不作510逐特征跨Task近零std正常化。残差使用仅13 TRAIN标签、全30rows共享17维mean/scale（floor1e−6）；禁止VAL/TEST拟合或调整正常化。零正常化输出的raw残差包含TRAIN mean，不能误称P本身。

网络沿用30×17自由CP、4层128 hidden/4heads、cosine100/DDIM20和clean_x0/v_mse数学。新的representation/condition/checkpoint独立命名为teacher_prior_residual版本，旧v1–v6权重及身份保持。生成后只计算P_free+denormalize(R_generated)，然后固定C0/C1消元解码并走原完整gate；无裁剪、投影、补IK、repair或fallback。

实现核查发现原proposal_gate的名义检查没有参考ddq项，因此本机制另保存独立原加速度guard（2.5/4、dense1351），用于中心full资格、生成candidate执行资格及prior_only。原gate源码/报告字段不改；其passed不能单独当作参考加速度已核。实际交接必须原gate与独立ddq guard同时通过，不能削弱原限制或用solver success替代核对。

## 先验算法与预算

每Task固定seed64、8 starts。每start先最多8轮耦合预测/anchor IK/有限rate整形；每anchor最多35 nfev。几何阶段仍最多8轮×4 line trials×8 witnesses、stride10；如几何产生accepted update，原流程再最多8轮coupled anchor/rate refine。因此**每start anchor/rate总上界为16轮**，不能把两段报告成8轮。所有私有预测、IK、SLSQP、距离查询和失败计入先验成本，source冻结后不得追加预算或retry。

在真实中心生成前预声明early stop：每轮更新后fresh预测满足全部原Task要求、CP/PCC域、内部0.95速度目标与原加速度时，可结束该段anchor/rate迭代；几何阶段及最终原full gate仍必须执行。geometry更新后的post段重新核对。8/16是最大上界，无须已满足条件后继续耗满；真实停止轮数与成本记录，不据后续VAL/TEST结果改变规则。

Rate整形是新Teacher intrinsic操作，发生在网络生成前：只对offending坐标求30D最近自由CP，保留所属臂required q anchors、完整固定C0/C1贡献、原CP/PCC域、dense1351 D/D2；内部速度目标0.95×原限，原加速度2.5/4不变。SLSQP maxiter35、ftol1e−10、解析Jacobian、无retry；结果独立重算约束，失败解不得裁剪或宣称成功。随后耦合预测重新检查基座holonomy，最终原full gate保持。

中心选择只用于表示：有原fullpass时按原static_score再index；否则在finite且C0/C1 exact候选中按原Task同刻位置/姿态最大容差比，再index。失败中心的原Task/安全拒绝保持，中心没有自动执行资格；只有所有8starts都无法构造有限CP时才停止整个机制，不能丢弃困难Task或成功标签。

## 训练、评价与停止

在所有六TRAIN中心/13残差、源输入身份冻结后，从头训练一次：seed70、lr1e−4、20,000 optimizer steps×416独立噪声样本；固定预声明最后warmup EMA，不按VAL/TEST选checkpoint。

原六TRAIN Tasks×8 raw candidates（seed64..71、candidate0共享K1）均执行原full gate，保存P、生成raw残差、重建CP、预测、gate和全部失败。另报prior_only（R_physical=0，直接使用P；不是网络正常化空间的零输出）用同一P、同一gate以及完全相同的先验成本。中心构造成本共享时必须明示，不能重复称网络八次独立Teacher搜索。

TRAIN raw0/48则结束该有限机制，不进入VAL/TEST/physics、不追加步数。有TRAIN入场时先选原Task顺序中第一组有资格者，按原static_score/index选一条，交根代理核对后只一次原27s TRAIN闭环，明确overfit smoke。真实原QP/安全拒绝保存为失败，不因此重复原候选、增加posture权重或扩预算。

固定VAL三任务×8只要求TRAIN raw≥1；该次TRAIN实际成功、失败或拒绝均不改变固定VAL一轮的资格。VAL raw0/24则终止，不做Test选择。全安全/fullTask合格VAL候选先交根代理source-ready核对，再独占原27s执行，不能写实际qpos/qvel或生成力矩。正式TEST前冻结全部选择/模型/协议，原三个TEST全部保留分母和失败。任何新增Test actual只评价，不进入先验、训练或规则选择。

## 可支持的结论边界

有实际入场也只支持教师先验辅助模型参考改变原QP/PCC/67力矩执行链的有限证据。实际partial失败、prior_only成功、repair或fallback均独立归因；旧network-only raw0不回填。若输出与中心等价，不能宣称网络产生额外路线贡献；若有数值多样性，也不能自动称不同绕障拓扑或更高完成率。原最小目标仍需独立学习参考实际闭环，未获得前不标complete。
