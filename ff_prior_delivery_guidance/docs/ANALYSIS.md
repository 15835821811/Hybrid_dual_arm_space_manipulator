# M1 实施与运行状态

已实际实现后四步 clean 引导、低维表示反馈、固定预算四方法比较、原生力矩与独立重放接口、冻结来源、原始审计和后台恢复程序。9/9 软件检查通过；尚无新物理通过结论。

运行中以 `artifacts/v7_1/delivery_guidance_smoke_001/status.json`、PID、心跳和同目录 REPORT.md 为准。终端：`./ff_prior_delivery_guidance/job.ps1 status` / `report` / `resume`。完成后从原始结果补齐本报告，不复用旧实验冒称本轮结果。

原始实验输出不修改旧模型、旧结果或旧冻结任务。查询/时间优势不阻塞工程交付。M2 是新 128 槽有界补库；旧 1024、pilot、TEST 继续关闭。

运行已启动，源码提交 a5e894df。已核对活跃 PID 与心跳；后台接续已绑定本次独立运行，正常运行期间不重复分析。
