"""Measure distinct Teacher routes from synchronized actual replay evidence."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def compare(first, second):
    roots = [Path(first), Path(second)]
    reports = [json.loads((p / 'evaluation/report.json').read_text(encoding='utf-8')) for p in roots]
    if not all(r['complete'] and r['evidence_valid'] and r['task_success'] for r in reports):
        raise ValueError('only complete independently validated successful routes may be compared')
    if reports[0]['task_sha256'] != reports[1]['task_sha256']:
        raise ValueError('route difference requires the identical immutable task')
    paths = [p / 'evaluation/fresh_replay.npz' for p in roots]
    actual = [np.load(p, allow_pickle=False) for p in paths]
    try:
        if not np.array_equal(actual[0]['time'], actual[1]['time']):
            raise ValueError('actual route timestamps differ')
        proposals = [json.loads((p / 'raw_proposal.json').read_text(encoding='utf-8')) for p in roots]
        free = [np.asarray(p['controls_free']) for p in proposals]
        metrics = {}
        for key in ('q', 'rigid_position', 'continuum_position'):
            difference = actual[1][key] - actual[0][key]
            metrics[key] = {'per_coordinate_rms': float(np.sqrt(np.mean(difference**2))),
                'vector_rms': float(np.sqrt(np.mean(np.sum(difference**2, axis=1)))),
                'maximum_vector_distance': float(np.max(np.linalg.norm(difference, axis=1)))}
        return {'schema': 'v6_4_actual_teacher_route_difference_v1',
            'task_sha256': reports[0]['task_sha256'], 'task_id': reports[0]['task_id'],
            'trial_paths': [p.resolve().as_posix() for p in roots],
            'fresh_replay_sha256': [hashlib.sha256(p.read_bytes()).hexdigest() for p in paths],
            'control_points_exactly_equal': bool(np.array_equal(*free)),
            'free_control_per_coordinate_rms_rad': float(np.sqrt(np.mean((free[1]-free[0])**2))),
            'actual_route_metrics': metrics, 'actual_state_count': len(actual[0]['time']),
            'different_homotopy_classes_established': False,
            'scope': 'numeric differences in two real routes; no claim of distinct obstacle-side route classes'}
    finally:
        for data in actual:
            data.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--first', type=Path, required=True)
    parser.add_argument('--second', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = compare(args.first, args.second)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    print(json.dumps({'task_id': report['task_id'], 'control_equal': report['control_points_exactly_equal'],
                      'actual': report['actual_route_metrics']}), flush=True)
