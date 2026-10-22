# Exact bounded reviewer request

作为全新独立 experiment-result-to-claim reviewer，审查刚完成的单次三障碍Diffusion工程运行。工作区 E:\桌面\mujoco_irregular_waypoints_diffusion_adapter_bundle_20260902-main\Hybrid_dual_arm_space_manipulator。先读 C:\Users\admin\.codex\skills\experiment-result-to-claim\SKILL.md 及 shared-references/review-tracing.md。仅本轮 artifacts/v7_1/diffusion_demo_001，协议 ff_prior_diffusion_demo/docs/EXPERIMENT_PLAN.md，冻结源码8456203e。用户目标：完整采样/避障/真实原生67力矩执行/独立全836INTEGRATION重放与全部可视化；不做优势对照。summary称5候选2合格，1次执行重放通过，原始审计PASS，媒体COMPLETE。核对原始arrays/report/ledger/source绑定和模型调用、失败分母、实际状态无reset、完整重放、媒体来源。不要仅信summary。只读保存证据/代数可，不运行新物理/optimizer/旧run审计/resume。不需重复历史审查或比较优势。核对C1仅这个静态场景可否yes；长期全工程未完成（动态实际连续交接与五障碍未完成），C2本轮NOT_EVALUATED。保存本次新review-traces完整prompt/raw/parsed/routing及紧凑核对证据到 paper/review-traces/experiment-result-to-claim/20261022_diffusion_demo/；可写此目录，其他不修改。反馈精确主要指标/发现与源路径。不要阻碍工程推进等待科学优势。

# Intended claim and bounded inputs

C1 from `ff_prior_diffusion_demo/docs/EXPERIMENT_PLAN.md`: public task, complete initial state, original three obstacles and world goals → actual model load → 16-step Diffusion sampling → complete 323 representation → prepare → whole-body geometry/world-goal correction → model qualification → native 67-channel torque execution → complete independent INTEGRATION replay → visual evidence from saved actual states. Claim only this static development task.

C2: not evaluated in this run. No advantage comparison is required. The 65-dimensional network condition has no obstacle condition. Dynamic continuous handoff and the five-obstacle objective remain outside the completed scope.

Read and check:

- Current run `artifacts/v7_1/diffusion_demo_001/`: frozen run identity, source archive, physical contract, scene initial audit and limits, model provenance, calls, ledger/budget, selection/results, every candidate's controls/provenance/screens and both selected repairs/validations, native and replay arrays, complete initial state, geometry scans, execution report, media manifest/CSV/HTML/videos/images.
- `ff_prior_diffusion_demo/`: protocol, inputs, experiment, audit_results, visuals, job and frozen evidence.
- Hash-bound dependency source, especially `ff_prior_static_fullchain_v2/proposals.py`, `validation.py`, `ff_prior_control_transfer/model.py`, `codec.py`, `ff_prior_static_goal_seed/tracker.py`, `physical_audit.py`, `ff_prior/initial_state.py`, `spline.py`, `geometry.py`, `robot_adapter.py`, `model_test/robot_model_spec_v5.py`, `angle_convention.py`.
- Existing model, codec, original selection and physical assets are accessed only to verify the current run's binding hashes, not to re-audit historical results.
- `paper/PAPER_CLAIM_AUDIT.json` existence check: absent; downstream paper-writing judgment must remain provisional. This does not block the engineering episode verdict or subsequent engineering work.

The exact artifact paths, SHA256 values, sizes and timestamps are recorded in `inputs.json`. This reviewer is delegated independently from the executor and does not delegate a further reviewer. No prior reviewer verdict content is used as proof of this run.
