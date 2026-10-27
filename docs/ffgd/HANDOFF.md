# FFGD handoff

Goal ACTIVE. M0/M1/M2 passed their implementation/calibration exits; M3 RUNNING.
Overall architecture/engineering/learning NOT_RUN. No learned checkpoint yet.
Workspace E:/ffgd-core, branch codex/ffgd-core-rebuild, pinned base b0e34e75e94ec53f1cd0a9cde246d4c727a596ce.
Original dirty checkout untouched. No push. All failed experiments retained.
CPU interpreter .venv/Scripts/python.exe; GPU .venv-gpu/Scripts/python.exe,
Python 3.10.19, MuJoCo 3.3.2, torch 2.6.0; CUDA 12.4 RTX 4080 verified.

Authoritative evidence: artifacts/ffgd/orchestration/state.json and work_units.jsonl.
M2: 256 native dynamics comparisons, 64 geometry-pressure cases, full energy
and 323-variable gradients, 9 native executions/replays (21000 steps).
One finite-difference feature switch classified separately; failed run unchanged.
Execution calibration covers small 2-second trajectories, not diffusion or final tasks.
Latest software regression: 10 passed in 20261027T033456_software_ec6602.

Next: profile independent model-valid data generation, generate 2000 valid pilot,
train full-variable conditional model and test actual full-noise sampling.
Check actual PID/create time in state and heavy_worker.lock before resuming.
Data attempts persist controls and results individually; --run resumes same config.
Training and public plan are being implemented; unimplemented commands fail nonzero.
No promise of autonomous work after the current agent/session stops.
