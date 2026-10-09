# C.3 final TEST metadata audit: prepared usage

The script is prepared for review and has **not been executed**. Current TEST work must finish before running it. Preparation used only source text and small existing metadata; the only executable validation was an AST syntax/import inspection of the script text, without importing or executing it.

After `test-search`, `execute-test` and the original `validate` command have completed, run from `E:\v64c3`:

```powershell
& 'E:\v64c2\.venv-c2\Scripts\python.exe' -B -X utf8 `
  'E:\v64c3\docs\audit_scripts\c3_final_test_metadata_audit.py' `
  --run 'E:\v64c3\v6_4\output\search_aware_warmstart_20261008_01' `
  --output-prefix 'E:\v64c3\docs\C3_FINAL_TEST_METADATA_AUDIT'
```

The command writes only `docs/C3_FINAL_TEST_METADATA_AUDIT.md` and `.json`. It never changes formal run evidence, selections, weights or core source. Rerunning replaces only those audit documents. To compare a completed report's numbers after report generation, add:

```powershell
--report-summary 'E:\v64c3\v6_4\output\search_aware_warmstart_20261008_01\summary.json'
```

`summary.json` is at the run root. Report completion and benefit declarations are never used as evidence. The optional comparison checks independently reconstructed endpoint counts, NO_PLAN/refusal/failure/alias counts, near-R12 counts, stream/work totals and measured cold/outer service times.

Exit codes:

- `0`: `METADATA_COMPLETE_WITH_DEFERRED_BINARY_VERIFICATION`. Required terminal metadata is complete and consistent; binary/source verification listed below is still separate work.
- `2`: `PARTIAL`. Required terminal files or seals are absent. Missing evidence never passes, and no final numbers are inferred from a live JSONL file.
- `1`: `FAIL`. A contradiction, unsupported schema, SHA mismatch, tool failure or resource-count inconsistency was found. Inspect the JSON discrepancies; do not repair experiment evidence or rerun physics to obtain a pass.

The initial preflight requires all 16 completed TEST requests, all 36 prefixes, all 40 TEST and 20 VAL logical slot/seal pairs, whole-phase selection seals, successful stage receipts, `actual_complete.json` and original `validation/protocol_validation.json`. Until these exist, the script returns PARTIAL before opening candidate registries. It uses final `candidate_registry.json`, never a live registry's current hash as a final seal.

The audit independently checks:

- 12 teacher, 10 VAL and 16 TEST requests; 96/88/144 consumed rows; rotated sequential method order, exact worker identities and measured outer timing.
- Model/data freeze before TEST, selected D/S checkpoint metadata, shared TRAIN-only N sources, fixed eight formal D samples and eight S forwards, common seeds, raw legality and illegal proposals consuming slots without repair or resampling.
- R4/R8/R12 and N/S/D4/8 self seals, registry/proposal prefixes and next-candidate timing; all 20 TEST A/B selections sealed before actual execution.
- Strict same-Task/plan/config/source/cold-history aliases and their copied fields, zero new alias cost, NO_PLAN and failures, original full 27 seconds and five independent gates, quality, B30, prediction/actual differences and near-R12 N/A handling.
- The 40-slot disjoint partition: unique actual success, unique actual failure, aliases, NO_PLAN and pre-actual refusal/failure. Unique failures are part of unique attempts and are not counted twice. Failed or short prefixes earn no success or quality credit.
- Immutable 328-candidate/60-logical-actual reservations, main-step maxima and separate preview/replay/native-geometry/QP work; original validation and final receipt counters are recomputed from saved inputs.

R8 has an original optimizer-prefix timer and no separately measured cold-request total. The R12 request cost is counted once, and A/B never double the search cost. Illegal raw outputs and short failed rollouts can reduce physical work; the audit does not turn that fact into a learning-benefit verdict.

Scope limits are explicit in the output. The script reads small JSON, command logs and 11 relevant frozen source files with the Python standard library. It does not import core modules, NumPy, torch or MuJoCo; run tests, inference, DDIM, training, physics, encoding or media; or open/hash weight, trace/NPZ, large interval archives or the complete 430-source inventory. Current small JSON seal members are checked; large members are recorded by original SHA strings, existence and size as **deferred current-byte verification**. Their recorded hashes do not prove their current bytes. A separate final source/archive verification remains required. Cold-history equality is reconstructed from declared initial-state/config identities and frozen control flow; raw state/trace revalidation is outside this script.

The script is tied to this original checkout/run schema and is not a portable-release verifier. Its output supplies no overall research-completion, checkpoint-reselection, statistical noninferiority, deployment or benefit decision.
