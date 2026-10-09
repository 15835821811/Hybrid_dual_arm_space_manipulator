# C.3 final mock wrapper 静态独立审查

2026-10-08；对象：[c3_final_mock_suite.ps1](audit_scripts/c3_final_mock_suite.ps1)。本次仅只读文本、PowerShell AST、文件清单与 SHA；**未执行或 dot-source wrapper，未启动 Python、pytest 或核心模块，未运行测试。** 正式 TEST 仍在进行。

审查版本 SHA-256：`fd5b53f990073b9cf70d2e932dc9de673f04584ae667119f99de8d50b45b7bf2`。当前 PowerShell 7.6.5 的 `Parser.ParseFile` 返回 0 个语法错误；这不等于在声明的最低 PowerShell 5.1 上实跑通过。

**结论：未发现需修改的具体可执行性或留档缺陷。** 以下为静态依据及边界，行号对应上述 SHA。

| 核查项 | 静态结论 |
|---|---|
| Windows 参数引用，189–203 | 有 `ArgumentList` 时逐项传递；旧 .NET 走 CRT quoting，双引号前的反斜杠与结尾反斜杠分别正确加倍。`UseShellExecute=false`，没有经 cmd/PowerShell 再解释 probe 字符串或含空格的 JUnit 路径。 |
| stdout/stderr/exit，193–235 | 两条 BaseStream 并发持久复制，先等待进程再等待复制完成；保存 PID、native exit、捕获是否完成、起止与 elapsed。invocation 在 launch 前即挂到主 receipt；异常保留 capture/launch error，finally 结束计时并释放资源。 |
| 原 phase 身份，137–164、254–260 | validate/report 必须是原 exit-zero、非 retry receipt，完整 argv 逐项匹配。与 `RUN/command_logs/continue_pipeline.ps1` 的实际 argv 构造及已有 closed-loop-val receipt 格式一致；另检查 UTC 区间和 report 在 validate 后启动。未来 validate/report receipt 尚待正式阶段实际产生。 |
| venv/runtime，272–302 | 记录指定 venv python.exe 的 SHA/PE 版本及存在的 pyvenv.cfg 内容/SHA；probe 记录 sys.executable、prefix/base_prefix、版本与位数，sys.executable 按完整路径与指定 venv 解释器比较。现有 pyvenv.cfg 可读；本次没有调用解释器，不能给 runtime 通过结论。 |
| 前后身份，268–280、307–319 | 测试文本、显式 bound sources、HEAD 与 wrapper 做前后比较；formal receipt 和 python.exe SHA 后验核对。stdout/stderr 成功收集后登记 SHA。其声明明确不覆盖全部传递依赖或安装包；pyvenv.cfg 仅记录前置 SHA，未声称它有前后比对。 |
| JUnit，321–340 | 读取原 pytest XML 的 `/testsuites/testsuite` 计数，分别保留 reported/failure/error/skipped/passed；未把静态函数数当通过数。pytest native exit 与 wrapper audit/count exit 分开；exit-zero 的零 case 或可选 expected count 不符会失败。 |
| 失败留档，341–376 | 已创建独立目录后，catch/finally 仍尝试保存 RECEIPT.md 与 receipt.json，记录 wrapper error/native pytest 信息；日志不覆盖既有文件。更早的 preflight 拒绝尚未创建目录，只写 stderr；JSON 最终写盘本身失败也以非零退出，不能保证磁盘故障下仍有完整 receipt。 |
| 仅指定模块，28–46、281–290 | AST 清单恰好 17 项、17 个唯一文件，全部存在；pytest argv 直接列这 17 项，无目录扫描。清空 ADDOPTS/PLUGINS 并禁用第三方 plugin autoload/cacheprovider；未找到当前仓库 conftest 或 pytest 配置另加收集项。相邻测试文件只作为显式 fixture/source 引用，不增加本次指定模块清单。 |
| 正式作业隔离，112–135、254–255、294–317 | 新鲜 CIM 检查在 probe/pytest 前及结束时执行；查询失败不会伪装为空。它明确只证明 launch/end 观察，不声称连续监测。当前禁止执行要求保持有效。 |

本次没有改 wrapper、冻结核心、测试清单或运行目录；仅新增本审查文档。正式阶段结束后的单次 wrapper 执行、实际解释器版本、JUnit 原输出及完整 receipt 仍待根任务安排，不能用本静态审查代替。
