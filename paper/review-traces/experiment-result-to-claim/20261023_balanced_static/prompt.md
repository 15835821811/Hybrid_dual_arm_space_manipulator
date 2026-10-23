# Exact bounded reviewer task

Fresh paper architect/reviewer for the new static run `artifacts/v7_1/balanced_restore_001`, applying `C:/Users/admin/.codex/skills/experiment-result-to-claim/SKILL.md` and its review-tracing contract. Production source is `5a21e08868cbff6a90d93e8225a43b77e8302bcd`, qualified by actual `executed_source.zip` and source hashes, not Git HEAD alone.

Bounded input list: this new run's complete saved files; `ff_prior_balanced_restore/docs/EXPERIMENT_PLAN.md` and `evidence/*`; `ff_prior_balanced_tracking/docs/EXPERIMENT_PLAN.md`; the frozen shared `ff_prior_balanced_evidence` implementation. Historical reports/bound inputs establish exact source identity only; do not reanalyse old raw experiments. Supplementary permitted inputs supplied during review: `ff_prior_balanced_delivery/static_body_completion/*`, `ff_prior_balanced_delivery/model_metadata.json`, their read-only producer scripts, and `ff_prior_balanced_restore/evidence/completed_001/*`.

Independently audit all saved data rather than sampling: source archive/current source/input hashes; all model qualification grids and restoration records; all 13,751 × 836 replay elements; saved motion, torque, pose, clearance and momentum reductions; all near/selected/omitted records and all 323 derivative columns; complete real initial and terminal state. State the boundary of independently reconstructing unsaved quantities. No new simulation integration, model propagation, geometry experiment, optimization, neural reconstruction, training, comparison or TEST. Only root may invoke the one completed resume; reviewer must never invoke it.

Claim schema:

- C1 is the frozen plan's allowed static predecessor subclaim: the one 2+5 scene, original full initial state, 27.5 s model qualification/execution/complete independent replay, suitable as the real predecessor of a later dynamic phase. It is not full-chain success.
- Cfull is the complete frozen static + original seven-waypoint continuous dynamic plan. Dynamic has not run, so its evidence remains incomplete.
- Caux covers body-distance/shape records and the observed scope of constraints; it must not imply causal advantage. Do not finalize Caux before the complete read-only diagnostic receipt arrives.

For every claim give `claim_supported: yes|partial|no`, `what_results_support`, `what_results_dont_support`, `missing_evidence`, `suggested_claim_revision`, `next_experiments_needed`, and `confidence: high|medium|low`. No automatic expansion into ablation, comparison, training or TEST. Any downstream writing remains provisional without a current paper claim audit.

Treat the executor's asserted 673 coupled calls, two recovery rounds, 13,750 execution + 13,750 replay steps, 15 true execution gates and 0.262127 mm C4/link_14 minimum as hypotheses to check. Preserve the original diagnostic PARTIAL result (9,912/13,751 after its 600 s deadline); the permitted saved-state completion adds only the missing 3,839 states in a separate delivery directory. Half-step/directional probes have no frozen pass threshold, so they cannot certify all gradients. Report SLSQP termination accurately. FINISHED is a worker state, not a scientific pass.

Only write in `paper/review-traces/experiment-result-to-claim/20261023_balanced_static/`. Save prompt, raw review response, verdict, inputs, handoff, reproducible algebra-only scripts and numerical outputs.
