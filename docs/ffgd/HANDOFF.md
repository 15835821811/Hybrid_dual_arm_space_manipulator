# FFGD handoff

Goal ACTIVE; M0/M1 passed, M2 RUNNING. Overall architecture/engineering/learning NOT_RUN.
Workspace E:/ffgd-core, branch codex/ffgd-core-rebuild, base b0e34e75e94ec53f1cd0a9cde246d4c727a596ce.
Original dirty checkout untouched; tracked diff and status preserved under orchestration.
Interpreter: E:/ffgd-core/.venv/Scripts/python.exe (Python 3.10.19, MuJoCo 3.3.2, torch 2.6 CPU).
No new dataset or learned checkpoint. No background job promised. Check live PID before any resume.

Evidence and gates: artifacts/ffgd/orchestration/state.json; append-only work_units.jsonl.
32 and 256 native FK/mass/inverse comparisons passed. 6 software tests passed.
3 initial states x 8 path directions x 3 finite difference steps passed, including nonzero robot momentum.
Whole geometry, full energy gradients, native torque execution/replay and later M3-M8 are unfinished.

Next runnable command (software regression):
`E:/ffgd-core/.venv/Scripts/python.exe -m ffgd.cli test --suite architecture`
Run from E:/ffgd-core. Next implementation: finish M2 geometry/actuation and full energy calibration.
`plan`, `train`, etc. currently fail explicitly; do not use them as claimed working commands.
Original snapshot restore writable-array error and result JSON serialization failure retained in old run directories.
