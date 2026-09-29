# V6.2-B.2 开关式区间 PCC 在线证据

本报告由已保存的运行、重放和故障注入文件自动生成。`legacy_pcc` 与 `bounded_interval_pcc` 都从同一五场景配置运行；后者关闭旧 PCC 查询，保留原 MuJoCo 与实际链胶囊约束、17 维加权速度 QP、67 路力矩执行和 0.45/0.55 十步斜坡。

## 功能与原门槛

| 项目 | 旧 PCC 新运行 | 区间 PCC 新运行 |
| --- | ---: | ---: |
| 五场景完整 trace | 5/5 | 5/5 |
| 原 26 项真实 MuJoCo 力矩重放 | 24/26 | 24/26 |
| 原 11 项执行合同 | 11/11 | 11/11 |
| 完整控制或旧 QP p95 最大值 | 23.281 ms | 22.987 ms |
| 整机最小 signed clearance | 14.996 mm | 14.996 mm |

![B.2 在线证据性能图](../v6_lite/output/v6_2_b2/online_evidence_report_20260930/b2-online-evidence.png)

新区间模式的原 26 项失败项是 `reported_summary_passed, measured_rate_deadlines`。五场景完整调用 p95 最大值 22.987 ms，仍高于原 20 ms 门槛；这一阶段依据用户授权仅作为功能闭环接入，原性能门槛保持未通过。三轮预声明的 6 s 计时 p95 为 24.407 / 24.508 / 25.020 ms。p95 通过也不等于硬实时保证。

## 独立与故障证据

新区间行从保存的真实力矩轨迹独立重算：6755 个规划边界、12350 条区间行、0 个不一致。此重算不调用在线 QP 求解或约束装配，但共享声明的几何与动力学模型。

故障注入 4/4：查询未知、QP 故障、斜坡包络失效、命令过期均在下一力矩步前拒绝。

A.1 历史产物仅是开发基线；本报告的 26 项检查从本次保存的 67 路力矩重新运行 MuJoCo。失败轨迹和性能反例保留在各自独立目录。数值区间实现没有形式化浮点认证，代理包络与真实机器人全域安全不能等同，本报告不宣称连续时间安全证明。

## 输入 SHA-256

- `old_metrics`: `331a8fb4bce64c2fa196bd1dfe4bd4e0cb22091428f43f093f7244133dd4f634`
- `old_native_validation`: `40e6bdf8263925b4a244d02d7e70a73d3df8c23b8bd59e96556414d4c31e352d`
- `old_execution_validation`: `212b7df6ad3132c32eb63effb706b9488b0a961248e30774b31446826c268b08`
- `new_metrics`: `b5332723270a062bba9f99dcd0bf1eae63655208f2cd028d7a08a9956b16e7b1`
- `new_native_validation`: `30f4127cfb790b1d06c76296bb94bcf33d36b757276f6ad1a0415462016f95b5`
- `new_execution_validation`: `51a4c2390d78b92084d7a3711539638b0d79008bf1d65e99d0f65940b795c46f`
- `unoptimized_metrics`: `c2e244778d3c73bac44f3a2b5b59136ea580f095c4b86d8e71171ab890f69a8c`
- `interval_recompute`: `b9fb7a764567c4d93ad4675d99ba3030f1ede201a96321207d0550b485404979`
- `failure_injection`: `9a6c32a21e7f283f4d7df1248990d0a78e3a09c506dc6536fd430235193dd37c`
- `repeated_timing`: `f7b23f5061ee7c1986d5f829c08c5936ffddc14ee28684a676f7dcd8761b1484`
- `online_visual`: `9dc90e95c25fdf3010ca65461e4cb4e7be08ebf63b5ec73216b1e8c2ddd7560b`
- `aabb_negative`: `6e95e3696ddad2bc64063130483ce58038d1808b758e79643ff67a2c1b257c57`
- `qp_breakdown`: `6731ae2d445fdec029be9ee800d77aab8e0bb970fb4c50c9d202424f78fbb65c`
