# V6.4-B.3.1 独立运行前审查

结论：**PASS，blocking_issues=[]**。可以按既定流程提交并冻结本次 actual producer，再执行原固定槽。该结论只表示运行前协议、来源和实现准备通过，不表示新路线收益、v2 闭环成功或 Diffusion 优势已经成立。

本审查读取用户执行目标、当前冻结 plan/task_manifest/config、T0 诊断及其来源清单、五处 inherited diff、执行 wrapper、既有参考与日志收据、canonical protocol_clarifications，并对清单所指原件做纯字节/SHA核对。没有调用物理、真实几何、模型采样或优化器；只按 root 新授权完成一项隔离 mock 异常检查。T3 reducer、制图和发布不在本次运行前审查范围。

## 实现与原控制决策

旧 producer 的330个源码输入均可认证；规范化换行后仅五个声明模块改变，其余325个一致。task_anchored_reference 新增独立 v2 schema/version/hash 和提前平台型解析 p/v/a，原 v1 默认数值及序列化路径保留，原合法区间、固定世界横向基和保护窗口不变。原0.24m/s参考检查没有放宽；28条已有检查全部通过，最大保守速度上界0.12664496902083272m/s。已有六项必要参考测试为一次完成，覆盖零残差直通、v1一致性、C²连接、解析导数、保护窗口/范数和版本绑定。

hierarchical_qp 在原 unconstrained solve、裁剪、动作选择之后复制诊断，Jacobian 速度乘积复用已经计算的矩阵，没有第二次求解或几何查询。run_v6_lite 增加 consumed version/z/definition/plan 字段和日志，residual_execution 独立验收 v2 缺失或错配身份会失败。conditional_execution 只透明转发并计账。冻结QP配置与旧原件逐字节相同。已有固定矩阵真实优化器日志开/关精确数值对照及其当前源码SHA一致；这是必要局部对照，不是实际场景验证或对所有输入的普遍证明。

## 固定任务、候选和来源

64个 task/candidate/precheck 冻结文件全部匹配清单。顺序为43mm正侧、43mm负侧、55mm正侧、55mm负侧，每任务固定 z0、v1±12、v2±12、v2±20mm。新55mm Task 除声明的任务/场景ID、layout元数据及该球心外，其余字段与对应43mm原件完全一致；四个已保存初态/锚点几何检查通过，新增查询5978次已入账。这不证明完整轨迹可行。四任务来自一个母场景，是开发参数研究，不是独立泛化TEST。

复用映射为 PILOT_00/01/02→EA_00/01/02，PILOT_03/04/05→EA_07/08/09。原task_id及Task/plan精确字节保持；attempt、trace、evaluation、quality及其绑定SHA均核对通过。旧314个manifest payload全匹配固定manifest，复用不是新增actual。新旧向量缺失仍为NOT_MEASURED，不能由旧标量I重建10D/7D或单球作用。

预算为28槽、6复用、22新增；每条新actual最多13500个2ms步，合计上限297000新actual步，拒绝会更少。训练、模型采样、seed搜索为0。private preview、独立保存力矩replay、native geometry及额外route-quality geometry分账，原门禁不变。拒绝占槽、不补候选；终态槽核对绑定后保留，未完成槽需人工检查，不自动重试。

## 已修复的阻断与纯mock核对

最初发现 NEW 槽直接调用 build_route_quality，后处理异常会中断后续固定槽且无法保留部分geometry成本。现改为 _quality_with_retained_failure：独立ledger记录route_quality，失败保存quality_label_eligible=false、full/prefix metrics=null、异常及已发生/raised成本；原actual Task verdict保留，随后继续后续冻结槽。前后source_guard及原attempt/task/plan/trace/evaluation绑定异常仍硬停止，不能伪装成候选失败。

单项mock检查一次完成：fake质量构建器调用3个mock native query sentinel后抛错，记录started=3/returned=3/raised=0；质量不可用、原TASK_COMPLETED状态及证据不变、构建调用一次、无重试。真实actual、preview、replay、geometry、QP、模型分配、DDIM及训练成本均为0。合成fixture的13500步只是输入字段，不是测量结果。检查收据的runner整文件SHA与最终审查源码一致，没有重跑该检查。

canonical clarification已消除两个解释歧义：candidate_ID固定为mode，slot_id只是执行/来源alias；旧A/B门槛保持相对zero **OR** opposite，同时发布两种比较；更严AND只能另报。旧B.3六槽负结论及停止结论不可变。质量后处理不可用不能作为物理失败赚取A门槛。

## 边界与后续封存

本审查没有验证尚未运行的新actual，因此不能提前断言v2被实际消费、完整Task/安全通过、降低I_route或工程收益。T3比较分母、完整成功候选集、常量/几何规则/V_cond、成本冲突和最终图表还须结果阶段独立核对。当前source_identity及actual目录尚未生成是运行前状态；root须按已审代码commit/freeze/source_guard后开始实际执行，最终发布和delivery seal另验。墙钟仍只记录性能，部署NOT_MET，未认证硬实时或连续时间安全。

关键SHA及实际读取输入列表见同目录preflight_review.json；下面列出主要绑定：

- `plan.json`：`92b62e02f3310169cdb6d336f09d3a834cd3c70d9eec8c4e8695accc86198ea0`
- `task_manifest.json`：`2f2647173e9a0eb91a6501a659ff7d8cb4bac1b50d98f4c98c1e4c896469a154`
- `frozen_execution_config.json`：`25a8f4aa060fcc3bc1f8aa6e5c791541d1f30a92b32565b8868f24e7da9f291e`
- `old_B3_manifest.json`：`57ba4bbf31ca3ff106d93d1ade0d1700bb4ab50840075847d5299878cf429033`
- `protocol_clarifications.json`：`b17c73d3556a67df0f7f3ef1fbefcb63d0d8b63e475313e52bcf9d3b35a833cb`
- `final_execution_aware_route_teacher.py`：`b68f388703e43a00f1260c7dbab057a5b1686ef87b40672c280f8ff2e549aed5`
- `quality_failure_helper_text`：`ab06905266dfbf177b9bf9bdcc03fdc7370a0c8aa92fbfee8c0e1d1e427d2764`
- `reference_tests.json`：`99c81929c8e3df4c8d939f795a80371528c750581e8efe44c3b25e3721f11936`
- `execution_diagnostics_checks.json`：`917700a7f32793acf0d1de8ac603dcf7b3026b6d5bd1f5f062855b3fac4cbdfb`
- `preflight_quality_failure_check.json`：`710c48496087da5cdcb43b29542bb45677e8f2c014d1322df60910d227165490`
