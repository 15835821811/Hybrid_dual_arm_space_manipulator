# V7.1 free-floating lightweight prior

Run these commands from the repository root. The audited interpreter on this
workspace is `.venv/Scripts/python.exe`; a fresh compatible environment can be
created from `ff_prior/requirements.lock`. No global environment is required.

```powershell
.venv\Scripts\python.exe -m ff_prior.cli audit --config configs/ff_prior/v7_1.proposed.yaml
.venv\Scripts\python.exe -m pytest ff_prior/tests -q
.venv\Scripts\python.exe -m ff_prior.cli run --config configs/ff_prior/v7_1.proposed.yaml --profile smoke --output artifacts/v7_1/smoke_003 --dry-run
.venv\Scripts\python.exe -m ff_prior.cli run --config configs/ff_prior/v7_1.proposed.yaml --profile smoke --output artifacts/v7_1/smoke_003
.venv\Scripts\python.exe -m ff_prior.cli status --run-dir artifacts/v7_1/smoke_002
.venv\Scripts\python.exe -m ff_prior.cli resume --run-dir artifacts/v7_1/smoke_002
.venv\Scripts\python.exe -m ff_prior.cli report --run-dir artifacts/v7_1/smoke_002
.venv\Scripts\python.exe -m ff_prior.visualize_evidence --run-dir artifacts/v7_1/smoke_002 --video
```

`audit` is read-only; `run` executes real generation, selected physical audits,
training and matched DEV planning. Existing output identities require `resume`.
`smoke_002` is the completed release run; `smoke_003` above is an unused example
output for a new reproduction, not a third run already performed.
`--stages generate` or `--stages audit` can stop at an explicit engineering
checkpoint; the default is the full smoke. `--budget-json` overrides named
budgets and creates a distinct run identity. The same output cannot be silently
reused for changed source/config/model/budget identities.

After an abnormal process exit, explicitly run
`python -m ff_prior.recovery --run-dir <run>` before resume if a lock remains.
Recovery checks the OS process handle and refuses a live or ambiguous owner.
It preserves state files and charges the interrupted proposal. Calls lost during
an abrupt crash are marked unknown, not assumed zero; such a run is not a
complete resource-certification run. Ordinary smoke_002 finished without a crash.

The main and DEV generation budgets are 200 each and 1,800 seconds each. The
selected physical audit is 16 random + 8 risk samples from different DEV
families. Learning uses at most 200 real optimizer updates, and the matched
comparison uses four other DEV task families, three methods, five candidates
and at most 12 selected executions. Replay is additional physical work and is
counted. No candidate is executed merely to select a successful one afterward.

Pilot needs explicit `--profile pilot --prior-evidence <frozen-gate.json>`;
study additionally requires a frozen 200-task TEST / 20 strict-pool archive,
checkpoint and resource-budget JSON. These are not launched by smoke. A study
invocation without required evidence returns BLOCKED. SUBSPACE_FD_DEV_ONLY
never makes online production eligible.

The main API functions are `pipeline.generate_prior`,
`coupled_predictor.predict_coupled`, `model_gates.validate_model`,
`physical_audit.audit_execution`, `dataset.export_dataset`, and `benchmark.plan`.
They share the same reference decoder, physical contract and frozen task schema.
Unknown mandatory gates fail closed. Native traces use actual low-level state;
the public planner never sees the private terminal seed or witness trajectory.

See `docs/V7_1_BASELINE_AUDIT.md`, `V7_1_DATA_CONTRACT.md` and the measured smoke
report for the evidence and scope. The clearance requirement is the user's
explicit nonpenetration override (>=0 m), with the full topology pair policy.

Runtime outputs/checkpoints/native traces remain local. The Git evidence bundle
contains only compact reports and representative samples; it does not include
large historical or new datasets. No remote push is automatic.
