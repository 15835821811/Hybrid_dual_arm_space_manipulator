# 独立结果到主张审查

审查对象仅为 `artifacts/v7_1/static_fullchain_001`。独立 paper architect/reviewer 结论：**claim_supported = no，confidence = high**。此处的 no 表示该次运行不提供 C1/C2 的充分证据，不表示科学假设被证伪，也不表示原场景无解。`paper/PAPER_CLAIM_AUDIT.json` 不存在，任何下游论文写作结论标为 **provisional — no paper claim audit available**。本审查已独立委派完成，`pending_external_review = false`。

## 发生了什么

`status.json` 为 FAILED，监控耗时 34.84399999957532 秒；`stage=complete` 是监控器结束标志，不能读成实验完成。`failure.json` 与 stderr 的调用链一致：首个 `capture_3_obstacles_20264611_structured_geometry` 单元，候选 02 的 `repair/iterations/00`，在 `solver.py` 保存 `trial_01.npz` 后、写 `trial_01_pairs.json.<PID>.tmp` 时抛出 FileNotFoundError。这属于工程部署/证据落盘故障；修正搜索已经进行部分模型传播，科学结论所需的原物理认证与完整实际执行结果尚未产生。

源码明确先保存 trial NPZ，再保存 pairs JSON，再决定是否接受该步，并且在 optimize 返回后才调用 validate。现存 NPZ 是一个 19×17 控制提案及 513 个预测状态，不能证明该修正被接受；`linearization.npz`、迭代 `record.json`、修正 `result.json`/最终路径均未落盘。不能从存活 trial 推断收敛、迭代收益或完整修正结果。

## 路径故障证据

异常报告中解码后的临时文件绝对路径有 **260 个字符**；目标 `trial_01_pairs.json` 为 251 字符；同目录已保存 `trial_01.npz` 为 244 字符，其 PID=4880 临时名为 253 字符。失败父目录存在，目标 JSON 和失败 tmp 均不存在。系统注册表 `LongPathsEnabled=0`。写入函数先创建父目录，再在文件名后添加 `.4880.tmp`，失败点是打开 tmp，和已保存较短 NPZ 形成直接对照。

这组证据高度支持 Windows 长路径限制导致的落盘失败。审查没有在旧目录做写入/失败重现，没有取得 OS 专用长度错误码，故不能写成已通过受控重现排除了全部其它文件系统原因。路径字符数是本次直接计算结果；长路径根因是证据充分的工程诊断。

## 实际证据覆盖

- 冻结分母为 12 单元、60 提案、30 提案/方法；只进入第一结构方法单元。
- 该单元 5 个候选有 controls/provenance、raw 与 prepared 初筛文件，10 个 screen 均明确标记 `SCREEN_ONLY_NOT_MODEL_VALID`。5 个 provenance 的 network_calls 均为 0，符合本单元是结构方法；没有 Diffusion 单元对照结果。
- 修正选择表为候选 02、04；只有 02 有 initial 和第一个 trial。落盘 outcome 仍为 SCREENED/eligible=false/selected_for_repair=false，这表示最后已保存初筛状态，不能解释为从未开始修正。
- 没有任何完整 `validation.json`、执行选择、执行 report、native67 控制轨迹/独立重放证据、单元 `result.json`、单元 `calls.json` 或 `budget.json`。首单元未完成，已完成单元数为 0。
- 因此本轮可报告“0 条已取得完整认证证据、0 个已完成执行单元”；不能将其转述为“物理成功率 0%”或“60 条均经完整物理验证失败”。未开始、未完成、未认证必须分开保留。

## 身份与保存完整性

独立枚举得到 58 文件，共 **2,616,481 字节**，逐文件 SHA-256 与 `failed_run_manifest.json` 完全匹配。ZIP 完整性检查通过；执行归档的 172 个成员、当前 172 个冻结源码文件均逐字节匹配 run.json 的 source_hashes，无缺失、额外归档成员或哈希不符。7 个 bound input 哈希均保持；selection、binding、software-test source map 和 ledger identity digest 都一致。

运行记录 producer_head 为 `25b98dd6e9f5da5c24ca9eee2e3194f3532d992c`，但不能声称仅凭该 commit 即可还原执行字节：56 个 Git blob 差异是 CRLF/LF；另一个 `v6_lite/hierarchical_qp.py` 相对 commit 有已存在的 14 行增加、1 行删除（可选 posture reference 分支）。该用户工作区修改本身已纳入 run identity 和 executed_source.zip，当前文件仍精确匹配归档；这不是本次运行后漂移。正确身份是 **producer commit + 实际归档哈希/源码清单**。ZIP SHA-256 为 `be865312c532d3e9889676a339653a017b64371f6d2625165ce4dd3af9c506b6`；identity digest 为 `546a4ed3fde812adc64c6ba69235d85a1de83626516debd323681dee91f4a661`。

## 未知成本与禁止重放

ledger 保留第一单元 inflight，completed/calls/wall_s/cpu_s 均为空。源码只在单元正常返回或已知预算异常处理后写调用差值、finish 和 budget checkpoint；本次 FileNotFoundError 越过这些写点。空 calls 表不能表示没有发生物理计算。现有保存信息不能恢复精确完整调用账本；审查不根据源码循环数或一个 trial 捏造实测成本。

旧失败首单元按 **13,000 耦合、32,000,000 积分区间、40 秒、128 MiB** 作为保守预算扣除是合理的恢复政策：前两者取冻结单元上限，40 秒高于已保存监控耗时，128 MiB 高于实际保留的 2,616,481 字节。这些数值必须标为 charged/reserved upper accounting，不能称为 measured/actual 调用、CPU 时间或内存峰值。128 MiB 属于输出空间扣额，不是内存测量。

若沿用原全局资源包，新轮可用余额应为 147,000 耦合、352,000,000 积分区间、21,560 秒和 21,340,618,752 输出字节。新轮应冻结扣减并执行总额约束，同时保持方法之间同等候选/单元上限，保留因全局耗尽而缺失的分母；不能把跨版本恢复导致的不平衡或未完成样本解释成 Diffusion 因果收益。本审查只评价这一政策，未认证 v2 具体实现或结果。

`assert_resume_safe` 对 inflight 有显式拒绝，`resume_not_performed.json` 与失败状态一致。本次不应做 completed-resume 测试；它仅适用于完整结束的运行。保留旧输出、另起修复包/输出、扣减保守未知成本、禁止旧 inflight resume 是适当处理。

## 主张与路由

**C1 = no。** 当前证据只支持公开输入生成、初筛和部分修正过程已运行，不能证明三/五障碍接近通过原门，更不能证明 native67 执行及独立重放成功。需要新的完整认证/执行/重放产物，两个场景各至少一条合格执行以及完整分母、身份、资源证据，才能按原协议审判 C1。

**C2 = no。** 本轮无 Diffusion 方法结果、无配对完整结果、无精确完整实际调用账本，不能判断收益、劣势、等效或资源优势。

建议报告语句：“静态全链首轮在首单元首轮修正落盘阶段发生文件系统异常，尚未产出完整物理资格或执行重放结果；未完成工作保留为未知成本并按冻结单元上限保守扣减。该运行不提供 Diffusion 性能比较结论。”

路由为 **pivot / engineering_deployment_repair**：保留失败 run，使用独立版本和短路径部署修复，在新 run 中获取原计划要求的证据。这里的 pivot 是工程修复分流，不是科学方法被判无效。无需复跑旧物理、无需旧 run resume、无需对同一冻结失败证据反复审计，也不触发消融或确认论文写作。新运行完成后对新增完整证据进行一次正常 result-to-claim gate；若又因工程故障停止，按实际故障收尾，不预设无限重试。
