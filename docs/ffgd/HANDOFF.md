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
Latest software regression: 14 passed in 20261027T040240_software_4da4d9.
Includes exact optimizer/RNG interruption recovery in a two-update CPU software fixture.
This fixture is not a learned pilot checkpoint.

Active dataset run: artifacts/ffgd/runs/20261027T034523_prior_data_d6601f.
PID 48956, create_time 1793072721.628873; verify these against live state.
Real serial queue: artifacts/ffgd/orchestration/m3_queue_99464/queue.json,
PID 99464. It waits for the exact dataset worker, then runs frozen debug32,
full pilot training and 5 full-noise learned checks. It stops on first failure.
It is a bounded experiment queue, not a self-programming/background agent.
Check queue.json and actual child identity before starting another heavy unit.
Next agent action: inspect M3 data/checkpoint/checks when queue completes,
repair ordinary failures or update evidence and proceed to M4 frozen DEV tasks.
Check actual PID/create time in state and heavy_worker.lock before resuming.
Data attempts persist controls and results individually; --run resumes same config.
Training, plan, selected execute and learning-checks CLI are implemented.
Real training/learned plan has not run yet at this handoff revision.
Eight configs/ffgd/dev_v1 tasks are frozen before training/planning outcomes.
New plan/runtime sentinel/selected execution interfaces still require M3/M4 runs.
Remaining CLI commands explicitly fail rather than claim completion.
No promise of autonomous work after the current agent/session stops.
