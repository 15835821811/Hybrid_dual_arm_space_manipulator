# 动态工作域修复输出

- `docs/EXPERIMENT_PLAN.md`：原任务/预算与域约束修复。
- `docs/LAUNCHER_REPAIR.md`：首次中断、独立启动修复和精确复用说明。
- `docs/ANALYSIS.md`：中文完整实测分析。
- `docs/INDEPENDENT_REVIEW.md`：新独立只读审查。
- `docs/findings.md`：工程结论与科学主张分离。
- `evidence/`：紧凑原始摘要、配对统计、冻结协议、复用来源、旧结果保留、52文件resume不重复核验、浏览器检查、输出hash清单。
- `artifacts/v7_1/waypoint_domain_002/index.html`（仓库根相对）：四组原模型视频、完整曲线/CSV、修复前后配对图。
- 各案例`native_trace.npz`：本地保留完整500Hz状态与力矩，不推送大数据。
- 线程可视化目录`waypoint-domain.html`：独立交互回放，旧可视化不覆盖。

终端入口：`.\ff_prior_waypoint_domain\job.ps1 status|report|resume`。已完成resume仅返回完成消息。

最终入口worker_v2/experiment_v2；worker/experiment是保留的中断版本，不再次启动。最终依赖的adapter与首次版本相同，修复启动器导入顺序后精确复用已完成基线及冻结先验，没有扩大物理执行预算。
