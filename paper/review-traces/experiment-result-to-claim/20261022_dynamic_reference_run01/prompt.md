# Bounded independent result-to-claim review

Review role: fresh independent paper architect/reviewer under `C:/Users/admin/.codex/skills/experiment-result-to-claim/SKILL.md`. Parent delegated this review; this is not an executor self-endorsement. No additional agent was spawned.

## Exact initial task

按C:/Users/admin/.codex/skills/experiment-result-to-claim/SKILL.md担任全新独立结果审查者。Repo E:\桌面\mujoco_irregular_waypoints_diffusion_adapter_bundle_20260902-main\Hybrid_dual_arm_space_manipulator。新完成run artifacts/v7_1/dynamic_reference_001，源码ff_prior_dynamic_reference，commit3d759e8f，冻结协议docs/EXPERIMENT_PLAN.md。仅此轮原4TRAIN受约束C2样条/真实原生67执行/独立重放，禁止重审旧P1/旧物理、禁止重仿真/训练/优化器求解/任何resume。原始结果：4/4参考解析门过；4个新13500步执行/13500步重放完成，3/4 EXECUTION_VERIFIED，t00仅derivative_limits失败，其余世界5mm/1度/碰撞/力矩等通过，完整重放积分差0。没有训练新增/DEV/TEST。请独立核166源zip/86input、4冻结任务和完整初态/时长、已保存系数与原数组的导数映射/逐段极值及约束、fit与实际执行误差/实际速度加速度、力矩/几何扫描覆盖、native/replay/连续边界4.5/25.5、预算reserved vs trace实际、不能仅信summary。P2b原冻结门是4/4，当前必须BLOCKED不能改3/4放行。目标C1受约束参考修复+所有4实际执行闭合，C2不涉及新Diffusion收益。主侧将计算compact saved statistics和一次completed resume/hash。请分析t00实际导数失败机制、支持的一次有证据机制修复（路线允许同瓶颈最多一次、保留原门/27秒/原目标、不扫参），或建议独立P3资格先行。输出trace paper/review-traces/experiment-result-to-claim/20261022_dynamic_reference_run01，包括prompt/raw/verdict/routing/inputs和范围局限；检查PAPER_CLAIM_AUDIT，不扩宽审计到历史。只编辑此trace，根findings/tracker交由主agent。

## Follow-up proposed route received during review

采纳一次经验参考加速度余量机制，不改原门。拟rho=1-2*max全4原生/17维实际-参考归一化残差=.9695632054405793，速度内缩不变；新独立身份仅对4固定来源重建，若t01-03系数/原controller/初态/任务/真实control身份精确相同则明确历史复用3，不重复物理，仅失败t00预选一次新执行重放（失败不补）。原001仍3/4、P2b当前BLOCKED。为总预算不自动增加：此27000额外steps从未启动P2b的324000预留扣除，P2总432000不增，P2b最多297000新steps，12逻辑单元至少一条准确历史复用否则缺失单位BLOCKED。时间与输出仍共享P2a原2h/8GiB余额。请只评估原001和这个机制依据，不审计未运行修复，不需重跑已有任何物理。主侧收尾统计/resume进行中。

## Intended claims and input set

- C1 local subclaim: the original four frozen TRAIN references can be represented under original position/velocity/acceleration constraints and all four can close actual native execution and independent replay under the unchanged gates.
- C2: no new Diffusion benefit is evaluated by this run.
- `ff_prior_dynamic_reference/docs/EXPERIMENT_PLAN.md`, its source package and exact 166-member executed-source ZIP, producer `3d759e8f53e2b2ef4f4880ff2084c8a1d6d6dfeb`.
- Run identity, exact 86 input bindings, four fixed source `result.json`/`trace.npz`, selection and source freeze. These are provenance inputs only; old physical outcomes are not re-reviewed.
- Each `t00`–`t03`: saved reference coefficients/knots/source arrays, limits, task/scene/path identity, full initial state, native trace, independent replay trace, geometry scans/minimum witnesses, transitions, representation and execution reports.
- Current P2/P3 route and cumulative-budget constraints from `paper/EXPERIMENT_PLAN.md`; no old P1 result review.
- `paper/PAPER_CLAIM_AUDIT.json` existence check; absent at review time.

`inputs.json` records exact paths, SHA-256 values, bytes and modification timestamps. The reviewer independently computed `saved_array_audit.json` using only `read_only_audit.py`, NumPy and SciPy interpolation/rotation algebra. No project module was imported, no optimizer was invoked, no model was created, and no physics, training, resume, DEV or TEST was run.
