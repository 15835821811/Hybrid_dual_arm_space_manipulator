# V6.4-B.3.1 后处理与发布源码审查

结论：**PASS，blocking_issues=[]**。本次审查结果归约、两张PNG生成、dashboard/publication audit/seal与轻量export的最终repo源码；完整结果claim review仍pending。actual producer为`d4464c8ae2aa7913a730ebbd775917e7a3b1af71`，334项冻结源码SHA全部匹配。没有执行physics、真实geometry、QP/model或真实reduce/plot/audit/seal/export，没有重复已通过的单元测试。

T3只排序原完整Task与安全门禁通过、质量可用的候选。严格最低I与tie-selected分开；0.001rad/s并列带锚定组内最低值、不链式扩大，组内路径、系数范数、mode排序符合冻结解释。旧pilot函数直接复用，zero OR opposite、两侧优选方向反转及原43mm/v1负结论保持；质量缺测不冒充物理失败。失败保留在28槽/4任务成功率分母，前缀不混入完整路径均值。

V_cond仅对七mode共同完成的矩形集合计算；全4×7缺失则full值unavailable，另报共同完整任务子集。零残差、七种常量、固定几何规则、teacher严格最低及tie-selected都有同一共同子集均值和直接比较，避免跨分母优越性结论。新17D向量复现原scalar，10D/7D拆分核对；旧向量与球行缺测不反推。球行持续时间按tick/source union计，不叠加重复行。

完整路线响应区分reference-base、actual-base、actual-reference及同任务actual-zero，路线窗口与27s分别保存；actual精确同钟、不插值。图表明确固定世界横向投影/lateral RMS，不冒充控制器传递增益或单球因果作用。局部净空最小时刻从保存series取；路线特定pair witness未保存时明确缺测，不替换为全局witness、不新增几何。

最初发现的具体缺口已修复并复核：

- retained quality失败只有route_quality_failure.json，原reducer写死成功文件会KeyError；现按stage选择且必须有terminal明确绑定。
- safe actual但quality不可用时fresh+manifest原可自证；现由原bound evaluation/report的fresh SHA或已有独立quality来源认证，并交叉核对producer manifest。plot仅对已识别0步未执行evaluation合法缺manifest，其他缺证仍失败。
- 共同子集缺几何/teacher可比均值；现补相同task_ids上的全部策略均值及直接差值。
- publication原可能接受陈旧review/展示；现校验review schema/run/producer、verdict绑定inputs SHA、八类关键输入覆盖、snapshot精确字节及当前非report输入哈希。report仅允许delivery-complete/review-status两个metadata变化，科学payload须一致。actual producer/reducer SHA、copied_inputs源和展示目的双向SHA/bytes、dashboard当前report/matrix投影、CSV逐字段均核对。

最后成本汇总增补也已审：execution与route-quality phase counts分开并保留started/returned/raised，初态+execution+route-quality总native查询不重复计数；297000新actual步上界及actual full-safe/quality-comparable总数分开核对。HTML明确质量stage状态。

local seal保持独占和完整payload/external身份核对。export要求clean本地seal、included/omitted完整互斥分区、逐项SHA/大小、冻结execution源码强制复制及外部遗漏账；portable verify不打开历史绝对路径，明确clone无法验证省略physical replay原字节。未发现额外exporter blocking。seal前只延期三个明确未来release control链接，export后严格核对全部链接。

已读取owner保存的必要focused收据：旧7项纯归约规则和旧plot检查没有重复，旧plot hash不冒充修改后完整覆盖。reducer新增4项纯文件绑定检查；最终cost-only补充有AST隔离收据；plot新增5类绑定检查，均PASS且0真实科学成本。详细输入/收据SHA见JSON。

限制：actual尚未全部终态，未提前判断赢家、收益、V_cond或v2闭环成功。最终28槽完整结果仍须正式独立claim review、PNG视觉核对及GitHub remote/发布seal验证。四任务同母场景、有限离线oracle不能支持Diffusion优势、独立泛化、硬实时或连续时间安全；部署NOT_MET。最终repo四模块为本审查身份；旧staging reducer d54版本被repo043成本汇总版取代。

- `evaluate_execution_aware_teacher.py`：`043b9134e7c6fb3e08dc7ad68314f89b10b870e81fc0920f02c7b3d8e1fb05cd`
- `plot_execution_aware_teacher.py`：`a46c2e9bdd26197ab225caf3a9fdecdbb88b3254e6792d4bb607254c094e4a8e`
- `deliver_execution_aware_teacher.py`：`f898526e6ad06c68a045630ff7e14ed90e8a90893a3397bbd693348615497e59`
- `export_execution_aware_release.py`：`0db8f1a1cd13a1cf5d84bc0c6092e3d2d67ced85a963049dd6c5e4fa0398e3a9`
