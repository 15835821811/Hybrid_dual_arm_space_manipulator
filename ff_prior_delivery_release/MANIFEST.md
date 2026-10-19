# delivery_v1 本轮交付清单

完整报告：docs/ANALYSIS.md。独立审查：paper/review-traces/experiment-result-to-claim/20261019_delivery_run01。

代码：build.py仅封装完成实验，不启动训练/执行；cli.py和job.ps1只读查询、报告和文件完整性。正式M1/M2/M3源码分别a5e894df/4aeced8a/09ba8bae，冻结源码未原地修改。

版本化证据位于evidence/：dataset_manifest.json绑定完整13966 TRAIN/48 VAL索引及128增量；model_manifest.json绑定选定新模型、codec、配置和原始源码归档；comparison_metrics.json、failure_ledger.json、next_256_protocol.json、TEST_readiness.json；completion.json绑定交付产品和新独立审查。

原始完整索引、轨迹数组、模型和原生执行trace保留在artifacts/v7_1/，不复制/上传大规模数据。packet.json保存审查前封装快照；completion.json与最终TEST_readiness.json补充最新状态，不改写旧packet或实验结果。

工程PASS，C1 yes，C2 no/NOT_ESTABLISHED；TEST NOT_READY且未运行，下一批256只有覆盖协议。完成resume已核验一次，禁止为了状态查询重复训练、执行或旧全量审计。根findings.md只追加本轮结论，未整体暂存既有未跟踪文件。
