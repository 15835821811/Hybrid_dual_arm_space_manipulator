"""Bounded numerical inspection of the pinned upstream implementation.

This is not a reproduction of the paper's dataset or a local robot experiment.
No upstream CLI, frozen benchmark, checkpoint, or training job is launched.
"""
import argparse
from collections import Counter
from dataclasses import replace
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
from scipy.interpolate import BSpline
import torch

COMMIT = '900d781ba63f0123dff95e49a76d51cb49a74803'
ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / 'artifacts/reference_audits/htdcr_900d781_20261019/upstream'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_files(root):
    names = subprocess.check_output(['git', 'ls-files'], cwd=root, text=True).splitlines()
    return {name: sha(root / name) for name in names
            if not name.startswith('data/') and Path(name).suffix in ('.py', '.yaml', '.md')}


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference', type=Path, default=REFERENCE)
    p.add_argument('--output', type=Path, default=Path(__file__).parent / 'evidence/reference_probe_001.json')
    args = p.parse_args()
    ref = args.reference.resolve()
    actual = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ref, text=True).strip()
    if actual != COMMIT:
        raise ValueError('upstream revision changed')
    if args.output.exists():
        raise FileExistsError('preserve previous probe output; choose a new output file')
    before = source_files(ref)
    sys.path.insert(0, str(ref.parent / 'deps'))
    sys.path.insert(0, str(ref / 'src/htdcr_planning'))
    genmod = module(ref / 'scripts/training/generate_training_data.py', 'upstream_generator')
    refine = module(ref / 'scripts/training/refine_training_data.py', 'upstream_refiner')
    from robot_model import RobotModel
    from constraints.feasibility import FeasibilityConfig, FeasibilityEvaluator
    from constraints.actuator import load_actuator_limits
    torch.set_num_threads(2)
    np.random.seed(20261019)
    torch.manual_seed(20261019)
    robot = RobotModel(section=4, segment=4, device='cpu')
    gen = genmod.MultiModalTrajectoryGenerator(robot, 20, 5, 20)
    started = time.perf_counter()
    pairs = [gen.generate_valid_multimodal_trajectory() for _ in range(8)]
    controls = np.asarray([x['control_points'] for x in pairs])
    flat = controls.reshape(-1, 20, 8)
    fallback = gen.generate_valid_multimodal_trajectory(max_attempts=0)
    fallback_unique = len({np.asarray(x).tobytes() for x in fallback['control_points']})
    try:
        refine.make_open_space_config()
        default_config = {'status': 'AVAILABLE'}
    except FileNotFoundError as exc:
        default_config = {'status': 'MISSING', 'error': str(exc)}
    # Explicit supplied repository config; source bytes and physical limits unchanged.
    cfg = replace(FeasibilityConfig.load(ref / 'configs/robot/strict_verifier.yaml'),
                  obstacle_environment_path=None)
    limits = load_actuator_limits(ref / 'configs/robot/actuator_limits.yaml')
    evaluator = FeasibilityEvaluator(robot, None, cfg, limits)
    refiner = refine.BSplineActuatorRefiner(robot, refine.RefinementConfig(), gen.knots, torch.device('cpu'))
    refined, optimization = refiner.refine_batch(flat)
    rows = []
    u20, u100, u1001 = (np.linspace(0, 1, n) for n in (20, 100, 1001))
    for i, (old, new) in enumerate(zip(flat, refined)):
        pair, mode = divmod(i, 5)
        original_curve = BSpline(gen.knots, old, 3)
        new_curve = BSpline(gen.knots, new, 3)
        goal = pairs[pair]['end_config']
        reports = [evaluator.evaluate(curve(u100), u100 * 10, goal)
                   for curve in (original_curve, new_curve)]
        lo, hi = robot.joint_limits_np
        dense = original_curve(u1001)
        rows.append({
            'pair': pair, 'mode': mode,
            'stored_vs_reconstructed_max_rad': float(np.max(np.abs(
                pairs[pair]['trajectories'][mode] - original_curve(u20)))),
            'dense_joint_violation_rad': float(max(np.max(lo-dense), np.max(dense-hi), 0)),
            'before': reports[0], 'after': reports[1],
            'boundary_four_changed_max_rad': float(max(np.max(abs(new[:4]-old[:4])), np.max(abs(new[-4:]-old[-4:])))),
            'new_endpoint_joint_velocity_max_rad_s': float(np.max(abs(new_curve.derivative()(np.array([0., 1.])) / 10))),
        })
    summary = {
        'scope': '8 newly generated upstream pairs / 40 numerical curves; not paper dataset statistics',
        'upstream_commit': actual, 'seed': 20261019, 'source_sha256': before,
        'upstream_SOURCE_unchanged': source_files(ref) == before,
        'upstream_TEST_read': False, 'local_TEST_read': False, 'local_training_or_execution': False,
        'paper_training_size_verified': False,
        'default_refinement_config': default_config,
        'explicit_evaluation_config': 'configs/robot/strict_verifier.yaml; obstacle-free diagnostic',
        'fitted_control_shape': list(controls.shape), 'duration_s': 10, 'evaluation_points': 100,
        'mode0_reconstruction_max_rad': max(r['stored_vs_reconstructed_max_rad'] for r in rows if r['mode'] == 0),
        'nonlinear_reconstruction_max_rad': max(r['stored_vs_reconstructed_max_rad'] for r in rows if r['mode'] != 0),
        'fallback_max_attempts_zero_unique_modes': fallback_unique,
        'before_feasible': sum(r['before']['feasible'] for r in rows),
        'after_feasible': sum(r['after']['feasible'] for r in rows),
        'before_failure_counts': dict(Counter(f for r in rows for f in r['before']['failure_reasons'])),
        'after_failure_counts': dict(Counter(f for r in rows for f in r['after']['failure_reasons'])),
        'dense_joint_violation_count': sum(r['dense_joint_violation_rad'] > 0 for r in rows),
        'dense_joint_violation_max_rad': max(r['dense_joint_violation_rad'] for r in rows),
        'refinement': optimization,
        'refinement_boundary_four_change_max_rad': max(r['boundary_four_changed_max_rad'] for r in rows),
        'configured_60_steps_at_1000_timesteps_actual_range_length': len(range(0, 1000, 1000//60)),
        'elapsed_s': time.perf_counter()-started, 'rows': rows,
    }
    if not summary['upstream_SOURCE_unchanged']:
        raise ValueError('upstream source changed during audit')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False)+'\n', encoding='utf-8')
    print(json.dumps({k:v for k,v in summary.items() if k not in ('source_sha256', 'rows')}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
