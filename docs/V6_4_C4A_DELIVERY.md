# V6.4-C4-A 交付索引与复现说明

## 身份

- 工作区：`E:/v64c4a`。
- 基准分支：`v6.4-c3-search-aware-closed-loop-val`。
- 基准提交：`18f6329e3960c76dce8b787749b269695888f9c9`；开始前本地与远端引用一致。
- 开发分支：`v6.4-c4a-diffusion-architecture-audit`。
- 物理实验 producer：`dd834783847bfbcc4f8ec624b7bce0d22777b6de`。DEV在该提交后封存，执行期间未改实验源码。
- 最终报告/证据提交在该producer之上；以 `git -C E:/v64c4a log -2 --oneline` 查询提交身份。报告自身不写包含它的commit哈希，避免自引用。
- 原有脏工作区、main及全部历史研究分支保留。没有push、merge、reset、clean、rebase或历史改写。

## 必需报告

|内容|准确绝对路径|
|---|---|
|数学、架构、数据审计|`E:/v64c4a/docs/V6_4_C4A_ARCHITECTURE_AUDIT.md`|
|C.3全量失败法证|`E:/v64c4a/docs/V6_4_C4A_C3_FORENSICS.md`|
|C.3全部343次提案诊断矩阵|`E:/v64c4a/docs/V6_4_C4A_SEED_COMPARISON.csv`|
|DEV预声明协议|`E:/v64c4a/docs/V6_4_C4A_EXPERIMENT_PROTOCOL.md`|
|DEV全部结果与预算|`E:/v64c4a/docs/V6_4_C4A_RESULTS.md`|
|决策与下一阶段门禁|`E:/v64c4a/docs/V6_4_C4A_DECISION.md`|
|交付文件SHA清单|`E:/v64c4a/docs/V6_4_C4A_DELIVERY_MANIFEST.json`|

## 实现与测试

|用途|准确绝对路径|
|---|---|
|冻结C.3模型的明确迁移入口，不训练|`E:/v64c4a/v6_4/c4a_frozen.py`|
|阶段A/B离线审计|`E:/v64c4a/v6_4/c4a_audit.py`|
|P0/P1/P2显式策略适配层|`E:/v64c4a/v6_4/c4a_portfolio.py`|
|DEV准备、硬预算、串行搜索与Actual入口|`E:/v64c4a/v6_4/c4a_experiment.py`|
|唯一修改的原算法文件；可选seed_schedule，默认路径保留|`E:/v64c4a/v6_4/continuous_route_optimizer.py`|
|15项新增数值测试|`E:/v64c4a/v6_4/tests/test_c4a_numerics.py`|
|9项新增调度/旧输出回归测试|`E:/v64c4a/v6_4/tests/test_c4a_portfolio.py`|
|4项新增DEV预算、封存、别名合同测试|`E:/v64c4a/v6_4/tests/test_c4a_experiment.py`|
|137个不同通过测试节点及原JUnit来源|`E:/v64c4a/v6_4/c4a_evidence/review_01/test_coverage.json`|

Python运行时：`E:/v64c2/.venv-c2/Scripts/python.exe`。Python 3.12.14、NumPy 1.26.4、PyTorch 2.6.0+cpu、MuJoCo 3.3.2、pytest 9.1.1。DEV串行单线程CPU，无GPU。已有环境复用，没有更换动力学依赖版本。

数值和mock测试可从上述工作区运行，完整原始命令以回执的argv为准。测试冻结D确定性时会执行模型forward，但不执行新的DEV物理实验。新代码选择P2必须通过 `RulePreservingPortfolio` / `optimize_strategy('P2', ...)` 显式启用；旧调用默认不启用。

## 数据与证据

- 阶段A/B全部机器审计：`E:/v64c4a/v6_4/c4a_evidence/audit_01/`。含条件字段表、标签覆盖、raw诊断、冻结输入SHA、38流成本和40个C.3 Actual端点记录。
- 四个C.3 B案例完整链：`E:/v64c4a/v6_4/c4a_evidence/audit_01/B_failure_evidence_chains.json`；易读摘要：`E:/v64c4a/v6_4/c4a_evidence/review_01/C3_B_mechanism.json`。
- DEV完整本地原始目录：`E:/v64c4a/v6_4/c4a_evidence/dev_01/`。该大目录由.gitignore排除，但保留全部文件，不删除轨迹或错误记录。
- DEV运行配置/任务/初态/噪声/母场景：该目录中的 `plan.json`、`generation_contract.json`、`source_identity.json`、`learning_split_manifest.json`、`frozen_*config.json` 和 `frozen_tasks/`。
- 预算预留账：`E:/v64c4a/v6_4/c4a_evidence/dev_01/budget_ledger/`；实际消耗汇总：`E:/v64c4a/v6_4/c4a_evidence/results_01/summary.json`。
- 全六流候选、选择与Actual机器表：`E:/v64c4a/v6_4/c4a_evidence/results_01/`。
- 最终冻结字节和执行合同核验：`E:/v64c4a/v6_4/c4a_evidence/final_integrity.json`。
- 紧凑发布包：`E:/v64c4a/v6_4/releases/c4a_architecture_audit_20261010_01/`。包含未改写的配置、任务、噪声、命令、预算、注册表、选择、五门禁和结果；`manifest.json`绑定复制文件字节，`raw_inventory_sha256.json`列出**所有**本地原始文件的SHA、大小与是否复制。
- 大轨迹CSV、NPZ及图片保留在原始目录，紧凑包不是完整离线轨迹包。绝对producer路径未改写；跨机器复核需一并搬运清单所指的原始文件，并明确重定位，不能把缺失轨迹冒充已验证。
- 新视频：未生成。

## 命令、退出码与不重跑原则

所有正式测试、离线审计、DEV prepare/run、最终汇总与导出命令的argv、cwd、UTC起止、PID、退出码、stdout/stderr及SHA保留在：

`E:/v64c4a/docs/audit_receipts/c4a/`

六次搜索和两次Actual子命令的精确回执也保留在：

`E:/v64c4a/v6_4/c4a_evidence/dev_01/command_logs/`

第一次旧测试收集失败保留在 `baseline_regression_01`，后续 `baseline_regression_02` 仅改用pytest importlib导入方式，72项通过；不是物理重试。`partial_results_01` 是实验中间的未完成快照，不能替代 `results_01` 最终表。

本轮物理命令是既往执行记录：

```text
E:\v64c2\.venv-c2\Scripts\python.exe -B -X utf8 -m v6_4.c4a_experiment run --run v6_4/c4a_evidence/dev_01
```

本轮预算不授权再次运行物理。入口拒绝已有execution_started或重复预算预留；没有自动物理恢复/重试。检查已有证据应使用以下只读报告脚本，并为新报告指定新的输出目录：

- `E:/v64c4a/docs/audit_scripts/c4a_report.py`：重算表格和能力/质量/预算。
- `E:/v64c4a/docs/audit_scripts/c4a_integrity.py`：核验冻结输入、独立初态和封印，不积分、不采样。
- `E:/v64c4a/docs/audit_scripts/c4a_export.py`：复制证据、计算完整原始文件清单。
- `E:/v64c4a/docs/audit_scripts/c4a_write_results.py`：从最终机器表生成结果文档。
- `E:/v64c4a/docs/audit_scripts/c4a_review_evidence.py`：C.3机制摘要和测试节点计数。
- `E:/v64c4a/docs/audit_scripts/c4a_command.py`：精确命令回执。
- `E:/v64c4a/docs/audit_scripts/c4a_monitor.py`：只读进度。

最终交付SHA清单覆盖上述新报告、源码、测试、脚本、回执和紧凑证据。原始DEV树由发布包raw inventory完整索引，C.3原始冻结产物由阶段A输入SHA和最终完整性回执索引，避免重复复制历史数据。
