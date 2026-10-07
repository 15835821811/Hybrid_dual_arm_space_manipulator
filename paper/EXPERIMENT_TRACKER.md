# B.3实验进度

| 阶段 | 任务 | 上限 | 状态 | 证据位置 |
|---|---|---:|---|---|
| 输入 | 7母场景/14Task、split与固定候选 | 0 actual | DONE | 独占run的route_pair_inputs.json |
| P0 | 旧权重条件响应 | 32 DDIM | TODO | old_checkpoint_probe/report.json |
| P1 | 配对路线区分度 | 6 actual槽 | TODO | pilot/decision.json |
| P2 | 质量teacher与标签 | 32 actual槽 | CONDITIONAL | 仅pilot通过后启动 |
| P3 | 单次训练与VAL选择 | 4000更新 | CONDITIONAL | 仅数据资格通过后启动 |
| P4 | 五组固定TEST | 20 actual槽 | CONDITIONAL | 仅模型准备完成后启动 |
| 审阅交付 | 两图/方法表/报告/独立审阅/上传分支 | 0新actual | TODO | 最终报告与发布清单 |

TODO/CONDITIONAL不代表已有结果。完整数值与失败以独占run结果为准。
