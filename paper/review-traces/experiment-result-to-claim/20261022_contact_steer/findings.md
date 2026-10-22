# Research Findings — contact_steer_001

- C1 (`original five obstacles / full initial state / original dual world goals / 27 s / full323 MODEL_VALID / native67 / independent836 replay`): **no**, high confidence within this bounded experiment; 0/3 complete-qualified candidates and three blocked execution opportunities.
- Caux (`contact-distance feedback and complete-state dedup can safely continue and improve residual`): **partial**. The data support sampled model continuation and descriptive within-prefix Q reduction, not continuous safety or component-level causality. Three chosen prefixes share one task.
- Reconstructed 249 new nodes, 741 distinct actual actions, 140 complete 37-query linearizations, 1,179 edge-grid propagations, 6,359 coupled calls, 1,383,008 reserved model intervals, and zero native/replay steps.
- Best Q remains 4.180432 / 4.761763 / 2.897692. Rigid-arm best position errors are 20.902160 / 23.808816 / 14.488462 mm against 5 mm. u01 also has 3.826964 deg rigid orientation error.
- Full state, prefix, source/archive/input, motion algebra, FD algebra, SLSQP returns, accepted-edge gates, and exact-key arithmetic agree. Initial two 1e-10-rad nominal mismatches reproduce exactly at producer's two BLAS threads; raw initial FAIL preserved with targeted reconciliation.
- Limitation: all 140 active sets truncate at 96; 4,900 omitted phase/pair records have counts but not saved identities/distances. No new geometry was used to fill the gap.
- u00/u02 hit 50-parent cap while 11/83 feasible unexpanded parents remain. u01 exhausts its available parents by 40 expansions and only reaches 17 s; not all 27 s consumed. Exact dedup does not ensure useful branch diversity.
- Thirteen success=false SLSQP returns satisfy the frozen linear rule and produce independently screened retained nodes. No saved returned delta exceeds its bounds. `sqp=0` is an uninstrumented generic counter, not evidence of zero optimization (420 saved SLSQP returns).
- Full323 fitting/repair/certification and native execution/replay never ran; do not call them tested failures. The dynamic handoff has no qualified actual terminal state.
- Missing evidence for C1: complete original-task certified path; original-initial-state native67 execution; independent entire836 replay and required terminal hold. Missing evidence for causal Caux: controlled component comparisons and independent broader tasks; not authorized or required for closing this bounded engineering result.
- Routing: **architecture decision / narrow claim**, not another local trial. Assess a future full-horizon constrained optimization design that can accept unqualified public initial guesses while keeping final gates, but do not claim it works yet. Unused budget does not authorize another local attempt. Do not resume/replay old work or add seeds/training/TEST to rescue this result.
- Paper claim audit: unavailable. Downstream writing remains provisional.

This trace-local findings record is the authorized reviewer output. The executor owns integration into project status and the root findings record; no frozen source or run was modified by the reviewer.
