# 当前执行目标：固定五障碍可行路径与引导诊断

2026-10-20。用户授权先验证原固定五障碍绕行路径，再定位轨迹表示和局部引导。
原静态预抓捕目标、连续体世界目标、完整初态、五障碍及旧结果保持；原目标全文归档于
ff_prior_capture_feasibility/evidence/previous_goal.md。静态闭环优先，动态航点仍后续独立处理。

有效协议：ff_prior_capture_feasibility/docs/EXPERIMENT_PLAN.md。
运行：artifacts/v7_1/capture_feasibility_001；终端 .\ff_prior_capture_feasibility\job.ps1 status|report|resume。

|阶段|状态与完成门|
|---|---|
|M0实现/冻结|PASS：7软件测试；129源码；12准确起点和原任务hash已冻结；原机器人/场景初态核对通过|
|M1可行路径搜索|待后台运行：12起点各最多10更新；全323维；独立严格验证，最多2预选执行重放|
|M2表示与引导|待M1：合格见证纯codec/部署prepare/102维修复；局部24/102/323方向和旧late4另作诊断|
|M3审计交付|待完成：新独立review、一次完成resume核验、中文报告与本任务提交|
|训练/扩库/最终TEST|本轮禁止，不自动追加|

2CPU/4小时/8GiB/45000耦合，最多5实际原生67力矩执行与独立重放。
保持仅不穿透、5mm/1度、2mm几何步长、全机几何、自由漂浮、17→67、完整初态及全部原运动/力矩门。
未找到见证不宣称不可达；局部方向秩不证明表示不可能；本轮不是同预算Diffusion优势实验。
MODEL_VALID与EXECUTION_VERIFIED逐条分开，不抽查升级全库，不失败重标目标。
本轮结束门为已冻结有限实验完成或明确阻断、原始证据审计与独立结论、报告和提交。
长期可靠规划和Diffusion稳定优势不因此宣称完成。保留全部旧模型/数据/用户修改/作业。
