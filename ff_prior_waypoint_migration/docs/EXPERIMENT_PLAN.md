# 单案例旧数据迁移与静止数据补查协议

2026-10-20，用户授权后新增独立运行waypoint_migration_001。旧waypoint_direct_audit_001不修改、不重跑。

选择冲突TRAIN案例seed20261012：local10mm_v1/teacher_a01/pilot_00与ui_pilot/data/episodes/train_20261012。对两份result/trace冻结哈希、比较标签字段。按当前保存初始化代码构造旧TrackingEnv（仅初始化/forward，不推进物理），比对已保存qpos/qvel/17维q/dq，记录完整integration state为“重建候选”，不能冒充历史原始快照。查找不含TEST路径的历史source_snapshot，计算原四文件code_hash，检查是否能绑定原记录。比较旧/当前合同的17→67映射、工作域、接触、碰撞策略及力矩限值。对相同关节状态只作模型/初始化核对，不重跑旧整段控制器。

静止补查：用真实旧模型target_world_joint的地址读取上一轮可用TRAIN/VAL的初始六维速度，验证线/角速度同时为零；另对tracking_output与artifacts/v7_1的manifest/split/dataset_report元数据作有界目录清单，排除TEST、diagnostic及静态终点库，记录来源搜索范围。未分割历史baseline不改成TRAIN；DEV及衍生不可入训练。

预算2CPU/1小时/2GiB新增输出；本次来源诊断0训练、0轨迹生成、0物理步，不扩库。若有足够通过迁移的静止/移动TRAIN/VAL，才继续原有最多4TRAIN表示执行门并实现依赖训练；不能单凭旧标签或重建快照放行。若原案例有未解决违规或覆盖缺口，则下游BLOCKED，生成报告和证据，停止，不追加试验、不改门、不TEST。
