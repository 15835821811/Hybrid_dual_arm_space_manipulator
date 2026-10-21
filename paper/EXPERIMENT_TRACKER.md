# 功能保持迁移与受控时间细节tracker

|run|阶段|状态|门和证据|
|---|---|---|---|
|control_transfer_bounded_001|M0|软件/输入预检完成后启动|8软件测试；15412输入；旧权重精确迁移；独立TRAIN|
|control_transfer_bounded_001|M1|待后台|6重建/3执行；8目标×2种子完整采样等价；256临时更新可学性与物理门|
|control_transfer_bounded_001|M2|依赖M1|唯一候选；14222 TRAIN、6000更新、1200完整16步教师批次|
|control_transfer_bounded_001|M3|依赖M2|新96完整无引导，旧96准确复用|
|control_transfer_bounded_001|M4|依赖M2|原DEV180新候选/36单元＋应用10/2；旧对照仅复用|
|control_transfer_bounded_001|M5|待收尾|原始审计、新独立review、中文报告、一次完成resume、目标/提交|

任何M1门失败则M2—M4 BLOCKED，仍完整负结果收尾；不追加试验、不扩库、不放宽物理门、不TEST。临时诊断权重不保留给候选，正式重新从相同迁移初态开始。收益以实测为准。
