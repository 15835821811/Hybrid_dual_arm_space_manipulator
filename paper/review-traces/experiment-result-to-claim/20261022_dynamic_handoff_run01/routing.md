# 路由

- `claim_supported: partial`；局部工程门 `PASS`；整体 C1 `NOT_ESTABLISHED`；C2正向贡献 `NOT_SUPPORTED`。
- `pending_external_review: false`；当前即为独立 reviewer 审查。`paper_claim_audit: unavailable`，论文路由 `provisional`。
- 保留 run 原终态与 u09/u11 的失败，不修改/重标原始结果，不追加本轮执行，不将共享基线计为独立重复。
- 执行方可引用审查完成 P2b 报告与tracker。执行方已完成唯一一次完成resume/hash核对，收据 `ff_prior_dynamic_handoff/evidence/completed_resume.json` 记录135个核心文件清单/hash不变、exit0、`NO_WORK_ALREADY_COMPLETE`。本review只读该收据，不再resume。
- P2b局部门已满足，可推进独立授权的P3资格检查；达到既有资格门以后才运行其训练。不得从本轮DEV、旧07或未经认证raw Z构造TRAIN。
- 不路由为完整工程交付或已确认论文写作；P1五障碍缺口、P3及P4组合验收仍未完成。后续依原阶段预算和一次候选约束，不用partial触发无边界补样本。
- 针对滤波原因只记录观测关联。未来如要因果主张，再设计事前冻结对照；当前没有执行此建议。

本次仅写指定trace，未按技能默认修改根findings/EXPERIMENT_RESULT_TO_CLAIM；这服从任务明确“仅写该目录”的范围。执行方可把verdict与上述限定合入项目状态。
