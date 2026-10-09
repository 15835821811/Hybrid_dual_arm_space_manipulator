# C.3 最终实际运行二进制与初态核验

2026-10-08 编制时仅完成 AST 检查。2026-10-09 正式流水线结束后已实际执行：**退出码 0，16,342 项检查通过**，覆盖 60 个逻辑槽和 39 次 unique actual 的保存初态。完整输入身份、时序和逐项结果见 [执行回执](C3_FINAL_ACTUAL_BINARY_AUDIT.json)。以下保留原执行方法与范围限制。

脚本：`docs/audit_scripts/c3_final_actual_binary_audit.py`。只依赖标准库和 NumPy，不导入实验模块。它补充最终元数据审计中明确暂缓的实际证据字节核验，并收束状态隔离审计的 R02/R04；R01/R05 由最终元数据审计覆盖，R03 由正式计时后的小数组 mock 测试覆盖。

必须等顺序正式流水线 `report` 正常退出、所有正式 Python 工作进程结束后运行。不要与正式规划或实际执行并行。输出在 `docs` 中，不能写进冻结 RUN；已有输出拒绝覆盖。命令为：

```powershell
& 'E:\v64c2\.venv-c2\Scripts\python.exe' -B -X utf8 'E:\v64c3\docs\audit_scripts\c3_final_actual_binary_audit.py' --repo 'E:\v64c3' --run 'E:\v64c3\v6_4\output\search_aware_warmstart_20261008_01' --output 'E:\v64c3\docs\C3_FINAL_ACTUAL_BINARY_AUDIT.json'
```

核验范围：

- 原 `source_identity.json` 的全部 430 项源码与模型资源字节、37 项受保护配置/任务文件。
- VAL 20 与 TEST 40 个逻辑槽的原 manifest 每个成员，包括 NPZ、重放、原独立验收、区间证据、计时及原 actual trace；另核对 evaluation manifest。
- 每次 unique attempt 的运行前后完整源码映射、QP 配置与原 producer 身份；原 metadata 的有效 run/QP 配置、模型、控制器、研究仿真模式及运行时规范化源码身份。
- 原 NPZ 中 initial_qpos/initial_qvel 与冻结 Task 逐值完全相等；保存步数与槽记录一致。只读已有数组，不新增物理。
- 冷启动声明哈希、严格 alias 原槽字节与 Task/plan/alias 身份。没有 trace 的零步结果和 NO_PLAN 保留为不适用，不伪造初态证据。

输出保留检查数、所有已读取输入的 SHA/尺寸/mtime、每槽结论、起止时间、实际 argv、脚本 SHA 和错误。成功退出码 0；失败退出码 1 且保留具体失败位置。若核验脚本自身有错误，保留失败回执，用新输出文件记录修正后的执行；不得修补原证据或重新运行科学样本。

该核验不重新签发安全结论，不证明没有记录的控制器对象/对偶历史实时遥测，也不代替所有教师与预测档案的完整 inventory、报告数值审计、媒体视觉检查或 portable release 迁移核验。最终 export 仍需核验其声明的完整来源清单。
