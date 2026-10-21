# 本轮主张与交付边界

运行：rollout_world_bounded_001；源码744c4f2d；2026-10-21。

- C1固定TRAIN完整采样可学性：新真实世界残差监督使临时诊断1/16→15/16，正式候选细化0/16→14/16；原索引12在接受的世界细化后合格。只支持固定8目标、2种子的有限范围，保护不覆盖普通6000步和全分布。
- C2无引导与完整方法收益：VAL24/96→4/96，未建立泛化收益。DEV两Diffusion均180/180与36/36；候选位置P95/最坏增加，执行位置略优，不能宣称稳定相对优势。五障碍仍0/5，静态抓捕闭环未完成。
- 工程交付：M0—M4完成，原始只读审计PASS；本轮37实际执行重放，旧重建/执行仅复用，软件测试不作物理证明。完成resume一次、3351文件不变，旧数据与用户修改保留。
- 混杂：表示、prepare、普通训练、世界细化均变化；未单独评价正式ordinary在VAL上的结果，不能孤立归因最后世界监督。旧VAL指标含旧prepare，新raw与prepare不变。
- 元数据注意：candidate_world/gate.json中的retained_for_candidate=false是共享细化函数字段；正式候选是否保存应核learning/full/model.pt的实际权重与final_hash，不能依赖该标志。
- 新独立审查结论以CLAIM_VERDICT.md及paper/review-traces/experiment-result-to-claim/20261021_rollout_world_run01为准。审查已完成：C1限定范围yes，C2 no，总体partial；无PAPER_CLAIM_AUDIT，论文使用为provisional；pending_external_review=false。
- 路由：保留旧候选，本轮停止；下一轮建议现有TRAIN上的功能保持/分层保护与独立TRAIN检查池，障碍条件和整轨迹避障另列应用瓶颈。没有授权追加试验、扩库或TEST。

软件完成、模型层资格、原生力矩执行与独立重放、科学收益是四个不同结论。长期完整科研目标未完成。
