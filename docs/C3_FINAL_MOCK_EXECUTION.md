# C.3 最终软件测试执行记录

2026-10-09，在正式 report 退出后运行当前入口。最终 [attempt 06 receipt](audit_receipts/c3_final_mock_suite_06/receipt.json) 为 `PYTEST_EXIT_ZERO`：162 tests 和 50 subtests 通过；JUnit 212 reported cases，0 failures、0 errors、0 skipped。13 条警告来自 Matplotlib/PyParsing 弃用接口。测试前后输入源码 SHA 未变，未重跑正式搜索、训练、DDIM 或 actual。

保留全部诊断过程：01 在运行前发现两个既有 localhost HTTP server；02 发现 PowerShell JSON 日期转换丢失原时区；03 修正异步输出捕获返回值；04 实际运行后发现历史 residual-binding fixture 缺少当前 consumed-reference 身份字段；05 因新 fixture 文件尚未成功生成而在 pytest 前退出。已有 receipt/log 均保留，前置检查未建目录的尝试不冒称已有回执。

当前 [test_residual_binding_current.py](../v6_4/tests/test_residual_binding_current.py) 保留原十项测试断言，补齐 plan definition、plan SHA 和逐周期 consumed-reference 身份 fixture。冻结历史 `test_residual_binding.py` 未修改，也未修改生产容差或原始实验记录。当前入口与历史入口明确分离。原 attempt 04 的失败仍可复核。

两个获显式 PID 白名单的 Python 进程只运行绑定 `127.0.0.1` 的标准库 HTTP server，核对命令、创建时间和累计 CPU 后保留；未终止用户的既有浏览服务。此软件测试不替代原五门禁、科学结果审计或媒体视觉验收。
