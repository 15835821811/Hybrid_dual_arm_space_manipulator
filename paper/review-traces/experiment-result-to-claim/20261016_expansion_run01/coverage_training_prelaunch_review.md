# ff_prior_coverage_train 启动前独立只读审查

结论：未发现warmstart、人口规模、缓存身份、元数据审计或条件队列的确定执行阻断；发现一处应在冻结排队前修正的明确资源合同不一致。该评审不是正式接续训练结果判定。未运行物理、训练或测试，未轮询或干预仍运行的数据父作业。

## 需要修正

sensitivity.py prepare()的存储检测仍为40*1024**3，而本包PROTOCOL和EXPERIMENT_PLAN均承诺20GiB。该分支在缓存导入和新敏感度计算阶段执行，应与20GiB合同统一。虽未证明实际输出会达到阈值，代码不能把已承诺20GiB实现成40GiB。只需修改新包并刷新软件证明，不应改动冻结父包或增加科学推进条件。

## 已核查的一致路径

1. require_data要求父status=FINISHED且实际进程退出、两个数据门槛标记、raw audit PASS、12096–12144 TRAIN与48 VAL、无已知执行失败及家族分区一致。队列固定父PID/run哈希/源码身份，父PID变化、未完成、失败或门槛不足都不能经正常路径启动训练。队列完成只表示派发，不能表示训练通过。
2. inputs绑定父数据全部保存文件、已完成10k训练来源和冻结DEV；正式run和终态审计复核来源/源码及源码快照。正确旧模型路径为qualified_pilot_train_002。新作业5400秒从0计，resume仅继承本阶段elapsed；旧已结束阶段成本通过历史记录保留，未错误扣入新阶段活跃预算。
3. fit_codec加载原10k模型对应的真实codec，验证其原fit_ids仍是继承TRAIN，并用实际数组、公共base、conditions及ID重算fit_hash。新数据不重新拟合表示。train实际加载10k权重，核对同codec，初始化新optimizer；最终audit比对初始weight_hash与原模型及warm_start_checkpoint_sha256。不是仅在metadata声明warmstart。
4. 全体新旧TRAIN由实际索引加载，6000×64=384000消费；最终audit要求消费ID集合等于实际TRAIN集合，每行次数>0，总draw正确，参数改变，检查点重载一致。VAL梯度为0，固定最终checkpoint。数据哈希与loader输出的schema对应；人数范围和总数组数采用动态len(index)，未残留固定10048总量断言。
5. 缓存要求同codec、task_hash、anchor ID、Z摘要、原NPZ哈希、39查询、校验通过与正确矩阵维度，复制后再次逐字节比较；不匹配会失败关闭而不隐式变成近邻缓存。新增任务才调用原数值build；13000容量高于本次最大任务数。按实际TRAIN任务顺序构造矩阵并审计矩阵hash。
6. cached_sensitivity.coupled为历史39次/锚点，cached_sensitivity.new_coupled为0；新sensitivity.coupled单列。继承原9912锚点时历史为386568次，不应称本阶段新增计算。最终audit比对每份复用JSON/NPZ原字节及proof哈希，合并已确认矩阵成本与分账一致。旧001失败及最多39次额外未知仍由inherited_failure_accounting单列，不会因当前unresolved为空自动消失。
7. 原DEV240候选、最多48选中native执行、共同修正、最低误差选择、无替补、18/20每种子与12/12新模型执行等门槛保留。固定10k条件均值的定义在文档明确，不错误称为新数据重新拟合均值。新版网络仍不编码障碍，文档没有把warmstart训练等同新场景泛化证明。

## 软件证据及范围

JUnit当前记录4项通过（约5.014秒）；测试代码确实使用保存的实际TRAIN/VAL、精确旧缓存、四步真实optimizer更新和拒绝不匹配codec/Z、活父拒绝、依赖阻断。该测试不能代表全规模新敏感度和最终DEV审计已通过。读取时evidence/software_tests.json尚不存在，因而尚不能核对最终源码绑定的PASS证明；queue.start和run本身会拒绝缺失/过期证明。

资源阈值修正、最终源码证明写入并核验后，可按原条件授权排队，无需额外用户确认或效率门槛。全规模完成后仍应分别审查实际消费、表示误差、采样、物理闭环及相对收益；本审查不提前判定任何正式结果通过。

## 两项关闭复核（2026-10-17）

已只读确认sensitivity.prepare存储阈值为20*1024**3，与合同一致。software_tests.json现为PASS，139个源文件哈希全部匹配当前文件，JUnit哈希匹配；最终JUnit记录4项通过、0失败/错误/跳过、耗时4.997秒。上述资源合同不一致与最终源码证明缺失两项均关闭，当前没有未解决的已识别启动阻断。可依既有授权启用等待父数据成功且实际退出后的条件队列；这不预判父数据或后续正式训练结果。本复核没有运行任何新测试、物理或训练，也未干预活父作业。
