# 完整控制有界闭环tracker

|run|阶段|工作|状态|
|---|---|---|---|
|full_control_bounded_001|M0|7软件测试；14222 TRAIN/48 VAL；固定完整323 codec；输入绑定|软件与初态预检PASS；165源码|
|full_control_bounded_001|M1|6固定TRAIN复杂轨迹的实际float32目标重建＋3预选力矩重放|RUNNING；源码f788a689|
|full_control_bounded_001|M2|唯一新MLP 6000更新＋1200完整16步教师批次|依赖M1|
|full_control_bounded_001|M3|VAL48×2完整无引导＋旧aligned准确复用|依赖训练完成|
|full_control_bounded_001|M4|原DEV12×3×5＋原应用3/5障碍×5；旧对照复用|依赖训练完成，不依赖科学优势|
|full_control_bounded_001|M5|原始审计、新独立review、中文报告/目标/提交、一次完成resume|PENDING|

无新增TRAIN，无DEV拟合，物理门不变，不扩库、不TEST。数据及模型初始化/输出/prepare/损失组合变化，不孤立归因维数或架构。方法失败仍保留完整分母。
