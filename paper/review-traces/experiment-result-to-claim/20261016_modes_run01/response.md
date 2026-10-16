# Independent result-to-claim review

**Verdict: partial. Full objective: not achieved.**

C1 passes its narrow local enrichment gate. C2 is supported as a useful controlled effect in this run. Reproducible Diffusion superiority over both strongest baselines, followed by actual production expansion, remains unsupported and must remain the objective.

## Evidence and practical significance

All seven input hashes match the frozen manifest. I read the plan, summary, independent audit, lineage audit, training evidence, historical analysis, core goal, objective/sensitivity/report/audit source, and recomputed aggregate costs from the 250 raw candidate JSON records. I did not run training, physics, or the full audit program.

| Method | Eligible | Executed + replay | Coupled queries | Candidate seconds |
|---|---:|---:|---:|---:|
| Strong spline | 50/50 | 10/10 | 1977 | 116.8793 |
| Frozen latent Diffusion | 50/50 | 10/10 | 2198 | 128.9474 |
| Augmented mean spline | 50/50 | 10/10 | 1710 | 103.7063 |
| Augmented v-only Diffusion | 50/50 | 10/10 | 2250 | 133.1591 |
| Augmented goal-loss Diffusion | 50/50 | 10/10 | 1735 | 105.0333 |

The paired goal-loss ablation improves total queries by 22.89%, candidate time by 21.12%, and repair-only queries by 24.03% (2139 to 1625). Query totals improve on every one of the ten tasks, so the aggregate gain is not a single-task artifact. Validation-query totals are essentially equal (111 versus 110), locating the gain primarily in repair. Median pre-repair position error falls from 68.74 to 34.16 mm. This is useful evidence for the regularizer within the tested setting.

Relative to strong spline, the goal-loss model saves 12.24% queries and 10.13% candidate time, just over the time threshold in one run. Relative to the stronger learned mean, it costs 1.46% more queries and 1.28% more time. Goal-loss has fewer queries than mean on three tasks, more on two, and equal on five; these local cost tradeoffs are worth retaining, but do not establish aggregate superiority or extra coverage. Success is saturated across methods. Equal observed success counts are not a statistical population noninferiority guarantee.

The 6552 offline sensitivity queries and about 104.7 seconds are a real extra investment. Relative to matched v-only, an illustrative query-only amortization would require about 636 candidates at the observed 10.3-query saving per candidate, ignoring all other differing offline costs. There is no demonstrated break-even versus the mean, since online aggregate cost is already higher. Report online, offline and execution/replay costs separately; do not describe the candidate-time gain as total system speedup.

## C1: local dataset quality

48/48 fixed-target additions passed model validation and the distinct-control-point criterion; the four preselected actual executions and independent replays passed. TRAIN grew from 208 to 256, with 30 families unchanged; the eight targets belong to five existing families. Added-obstacle rows grew from 3 to 21. This meets the planned local quality gate and fills local label scarcity.

Only four new rows are EXECUTION_VERIFIED; 44 remain MODEL_VALID. Neither control-point RMS separation nor six trajectories per existing goal demonstrates different obstacle routes, homotopy classes, useful multimodality, new-family coverage or production expansion. The current count of 256 is not the required eventual larger dataset.

## C2 and causal scope

New v-only and goal-loss training share data, codec, initialization, update count, sampled RNG proofs and row exposure, with fixed last checkpoints and no VAL gradients/checkpoint selection. Source inspection shows the loss difference is the local coupled-pose penalty. This supports the within-run C2 interpretation. Five inference candidates per task do not replace independent training seeds.

**Correction to the supplied prompt:** the old latent training record and both new records use seed 20261081 and the identical initial-weight hash `b3caf91eeb0b25197a5c10f9aaee413a24fa46f3066846680b921affa39e86ab`. The different-initialization-seed statement is false. The old-to-new-v-only comparison still does not isolate pure data effects: the data-fitted temporal codec/center/scaling, sample membership, shuffled batches and per-row exposure change together. Treat it as a changed training pipeline comparison unless separately factored. Historical mean time is not a concurrent comparison.

All learned methods have 0/50 pre-repair pose passes, versus 5/50 for strong spline. Their successful final outputs depend on the common nonlinear repair and adaptive validation. The goal-loss penalty is linearized pose change of denoised error under 34 selected directions mapped into 119 coefficients; it is not an exact full-dimensional physical objective. All rows with the same task share an anchor derivative. The stored omitted-temporal-direction check averages 0.1562% relative error and reaches 0.6991% across 168 anchors, but tests only a small random displacement per anchor. It does not certify accuracy at real denoising amplitudes or all new label locations.

## Audit limitations

Additional posthoc saved-trajectory diagnostics suggest a useful multiobjective question for the next preregistered round: goal-loss selected trajectories have median joint path 0.560775 rad, torque-squared integral 0.011307 Nm²s, and reference acceleration-squared integral 0.002747 rad²/s³, versus mean 0.572263/0.017156/0.003041. Strong spline remains better on those three medians (0.447450/0.007706/0.001951). Goal-loss median hold-position error is 0.445 mm, versus mean 0.619 mm and spline 2.579 mm. These are descriptive metrics over ten trajectories selected by the existing goal-error score, not independent optimized quality comparisons. Torque-squared integral is an effort proxy, not energy. The mixed tradeoffs motivate a frozen multiobjective evaluation but cannot replace the failed preregistered gate or establish superiority over both baselines. This supplemental artifact was supplied after the original seven-file manifest and is not claimed to be one of its hashed inputs.

The independent audit is supportive evidence; this bounded reviewer recomputed raw candidate cost totals and inspected source, but did not independently rerun every acceptance assertion or physical trace. Raw MODEL_VALID evidence explicitly says reference torque feasibility is NOT_EVALUATED and continuous-time geometry is not certified. Preserve these limitations alongside selected actual 67-channel execution/replay evidence. A model-valid label must not be described as a universally torque-executed or continuously collision-certified trajectory.

The shared audit code checks paired RNG hashes at 100-update intervals rather than storing every batch. Together with deterministic source and matched metadata this supports the experimental match, but I did not replay training. Original DEV/TEST were not executed. Repeated use of the same exploratory VAL tasks remains development feedback even without VAL gradients.

## Next research step and routing

The immediate question is whether goal loss creates useful target-preserving stochastic variation or mainly suppresses the harmful residual toward a good mean. Run a bounded TRAIN-only residual-amplitude mechanism study: the identical learned mean with zero, attenuated and full learned residual, under the same nonlinear repair, gates and charged physics budget. Measure raw target error, repair cost and valid, target-preserving diversity rather than only coordinate spread. Freeze any choice on TRAIN.

If the mean remains dominant, investigate target-preserving residuals or correction during denoising using the surrogate, with nonlinear rechecks and all extra calls charged. Test obstacle conditioning as a separate factor if needed; the present network does not encode obstacles, and the current evidence does not prove conditioning is the unique missing ingredient.

Then freeze a fresh bounded confirmation with independent development families, multiple training seeds, at least three predefined sampling-seed groups, interleaved repeated timing and paired task/family-level uncertainty. Keep the strong spline and the same-data learned mean intact, retain v-only/goal-loss ablations and the old frozen model reference, preserve all reverse wins/failures, and predeclare meaningful effect sizes and noninferiority margins. Do not retroactively relax the present gate.

After reproducible advantage is established, execute the separately authorized staged approximately 1k → 10k → 100k qualified-data route with actual arrays, provenance/failure ledgers, recoverability, stratified execution/replay, retraining and frozen reevaluation. Neither a plan nor the present 48-row enrichment substitutes for that delivery.

Suggested current claim: “On ten reused exploratory VAL tasks, a TRAIN-only local coupled-pose regularizer reduced repair queries and online candidate time relative to otherwise matched diffusion training, with unchanged observed execution success. Small fixed-world TRAIN enrichment passed its local quality gate. Learned mean remains the lowest aggregate-cost baseline; reproducible Diffusion superiority and production-scale expansion remain open.”

**Structured outcome:** claim_supported = partial; confidence = medium (high for observed arithmetic and current gate failure, limited for broader inference); route = supplement and mechanism-focused method refinement. paper_claim_audit = unavailable. **Provisional — no paper claim audit available.** The independent reviewer role was used; this is not a pending-review executor self-assessment.

