# B.3.1 实验计划

问题：20 mm任务锚定残差经过既有安全执行层后，提前建立且保持横向偏置是否比对称单峰更有效？固定有限非学习教师的同预算质量潜力有多大？

本轮从bdd6df6独立派生，不重开B.3，不训练、不采样模型、不搜索seed/位置。当前输出 execution_aware_route_teacher_20261007_01。

| 主张 | 最小证据 | 对照 |
|---|---|---|
| v2平台参考改变真实可执行路线并减少原I_route | 同任务同z的v1/v2完整Task+全部原安全证据；全窗口实际响应与其他代价同时记录 | z0、v1/v2 ±12mm；v2 ±20mm |
| 固定有限条件选择相对简单规则/最佳常量有质量潜力 | 四任务七模式全候选质量矩阵与V_cond，共同完整任务子集，失败不填成本 | 最佳正/负、零、远离球侧v2±12mm规则、最佳固定七模式 |

T0读取旧6槽原SHA本地NPZ，分开reference-base、actual-base、非零actual-同任务zeroactual与中点/全窗口/全任务。无历史物理重放，不从标量逆推分量或单障碍贡献。

T1保留v1全部默认字节/数值路径，新增r=.25/f=.75的独立v2版本和hash，解析p/v/a、固定世界基、保护窗与23.98s后零残差。原参考.24m/s gate保持，旧无Cartesian加速度cap不虚构新cap；下游原限制保持。必要测试仅零/连接/导数/保护/范数/消费绑定；最小日志记录原同一次QP已有量，开关不得改变动作。

T2同一母场景，d43/55mm × 两侧；每任务z0、v1±12、v2±12、v2±20。43mm原六槽保持Task/plan exactbytes并复用，55mm仅球中心与必要身份metadata变化。28槽，预期6复用+22新actual；预检/执行拒绝占槽，无重试替换。每个新actual都按原27s或门禁拒绝结束，完整验收仍原5项。全部是开发教师数据，未运行B3 TEST不加入。

T3主要J=原17D裁剪名义→选中速度干预RMS。先成功安全资格，再I_route升序。tie-band事先固定绝对0.001rad/s，按组最小值形成band（不链式扩张），band内按全程连续体路径、系数范数、候选ID；最低J和tie选择分开。不是统计可信区间。展示全部质量向量及冲突；新10D/7D/球活动仅有真实新源向量时计算，旧六槽标未测。旧10%/0.001门槛逐任务/配对另报，旧负结果不改。

V_cond仅在可比完整矩阵直接计算，失败先报固定分母成功率和共同完整子集；不为失败填虚构J。几何规则固定v2±12mm选择远离球侧，不事后调幅值。跨任务最佳常量包含零和所有七模式，不能只用最差固定方向。

| 顺序 | 工作 | 预算/门槛 |
|---|---|---|
| T0/T1 | 来源诊断与必要参考/日志不变测试 | 0actual、0DDIM、0optimizer |
| 冻结 | 4Task/28候选/0.001tieband/source/config | 新actual前commit+hash+独立只读审查 |
| T2/T3 | 固定全部槽与质量矩阵 | 6来源复用+22新槽，不超过28新actual；安全失败不补 |
| 交付 | 两张参考/实际图、全候选表、报告、独立审阅、seal、单独GitHub分支与入口刷新 | 不批量视频/PDF；不训练 |

actual、preview、独立保存力矩replay、原生geometry与路线质量geometry分别计账。墙钟只记录，20ms规划/2ms物理/27s及QP17D/力矩67保持；deployment NOT_MET。若没有改善结束参考/权限假设，不扩大网络、障碍搜索或时限。


## 终态交付指针（2026-10-07）

上文固定计划保持原文。终态证据见 [portable 报告](../v6_4/releases/execution_aware_route_teacher_20261007_01/report.md)、[全部 28 槽](../v6_4/visualization/execution_aware_route_teacher_20261007_01/all_candidates.csv)、[两张 PNG](../v6_4/visualization/execution_aware_route_teacher_20261007_01/index.html) 与 [run03 审查](review-traces/experiment-result-to-claim/2026-10-07_run03/response.md)。完整安全通过 28/28，质量可比 28/28。

有限teacher较最佳常量低3.27%、较冻结几何规则低1.03%；同幅值v2没有一致优势，Diffusion收益未建立。 下一步：结束本轮，保留有限质量向量和非学习基线，暂不训练新网络。

旧 B.3 计划与进度保存在 `EXPERIMENT_PLAN_B3_archive_20261007_190539.md` / `EXPERIMENT_TRACKER_B3_archive_20261007_190539.md`；旧停止结论不改写。
