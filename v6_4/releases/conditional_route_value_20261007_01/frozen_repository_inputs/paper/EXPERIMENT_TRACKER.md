# B.3实验进度

| 阶段 | 任务 | 上限 | 状态 | 证据位置 |
|---|---|---:|---|---|
| 输入 | 7母场景/14Task、split与固定候选 | 0 actual | DONE | route_pair_inputs.json |
| P0 | 旧权重条件响应 | 32 DDIM | DONE：16/16配对响应 | old_checkpoint_probe/report.json |
| P1 | 配对路线区分度 | 6 actual槽 | DONE：6/6安全完整，价值门槛未通过 | pilot/decision.json |
| P2 | 质量teacher与标签 | 32 actual槽 | NOT_RUN_PILOT_STOP | 不追加teacher |
| P3 | 单次训练与VAL选择 | 4000更新 | NOT_RUN_PILOT_STOP | 新模型未训练 |
| P4 | 五组固定TEST | 20 actual槽 | NOT_RUN_PILOT_STOP | 未评价，不填0/4 |
| 审阅 | no / high，无blocking | 0新actual | DONE | review-traces/experiment-result-to-claim/2026-10-07_run02 |
| 本地交付 | 两图/PDF/方法表/报告/来源与哈希验证 | 0新actual | FINALIZING | 最终seal与verification.json |
| 发布 | 独立分支上传与当前入口刷新 | 0新actual | FINALIZING | 发布提交和远端SHA另行核验 |

P1有限负结果按冻结规则结束。B.2已发布28ef7889保持；B.3研究执行结束与GitHub发布完成分开。发布后的最终核验保存到docs/V6_4_B3_PUBLICATION_CHECK.json。
