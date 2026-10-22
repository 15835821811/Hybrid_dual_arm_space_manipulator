from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
PACKAGE = Path(__file__).resolve().parent
OUT = ROOT/'artifacts/v7_1/dynamic_reference_001'
SOURCE = ROOT/'artifacts/v7_1/waypoint_moving_gate_001/freeze.json'
P = dict(stage='P2a_constrained_reference', duration=27., knot_spacing=.1, degree=3,
         derivative_fit_weight=.0004, ridge=1e-10, optimizer_maxiter=200,
         optimizer_ftol=1e-13, coefficient_margin=1e-9,
         derivative_fraction=1-1e-8, initial_tolerance=1e-10,
         executions=4, steps_each=13500, dt=.002, wall_s=7200,
         max_bytes=8*1024**3, threads=2, max_native_replay_steps=108000,
         training=False, expansion=False, TEST=False)
