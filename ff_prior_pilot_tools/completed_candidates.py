"""Read-only attribution summary; refuses incomplete runs and existing outputs."""
import argparse
import json
import math
from pathlib import Path
from statistics import median

from ff_prior.storage import file_digest, read_json


def analyze(source, output):
    source, output = Path(source).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    if read_json(source / 'status.json')['status'] != 'FINISHED':
        raise ValueError('requires completed run')
    audit = read_json(source / 'independent_verification.json')
    if audit['status'] != 'PASS':
        raise ValueError('requires completed run and independent raw evidence audit')
    methods = ('structured', 'frozen_diffusion', 'learned_mean', 'expanded_diffusion')
    hashes, results = {}, {}

    def read(path):
        hashes[str(path.relative_to(source))] = file_digest(path)
        return read_json(path)

    for method in methods:
        initial, final, accepted, counts = [], [], [], []
        statuses = {}
        for seed in range(3):
            eligible = 0
            for task in range(4):
                sid = f'DEV_{task:02d}'
                for candidate in range(5):
                    r = read(source / 'candidates' / str(seed) / sid / method / f'{candidate:02d}.json')
                    if r['status'] != 'DONE':
                        raise ValueError('non-DONE record: cannot silently omit failed candidates')
                    guide = r['sampling']['guidance'][-1]
                    history = guide['history']
                    def pose_norm(key):
                        error = guide[key]
                        if len(error) != 12 or not all(math.isfinite(v) for v in error):
                            raise ValueError('invalid stored pose error')
                        return max(math.sqrt(sum(v*v for v in error[j:j+3])) for j in range(0, 12, 3))
                    initial.append(pose_norm('initial_goal_error'))
                    final.append(pose_norm('final_goal_error'))
                    accepted.append(sum(bool(h['accepted']) for h in history))
                    eligible += int(r['eligible'])
                outcome = read(source / 'executions' / str(seed) / sid / method / 'outcome.json')['status']
                statuses[outcome] = statuses.get(outcome, 0) + 1
            counts.append(eligible)
        results[method] = {
            'candidates': len(initial), 'eligible_per_seed': counts,
            'execution_outcomes': statuses,
            'pre_guidance_pose_within_threshold': sum(v <= 1 for v in initial),
            'pre_guidance_max_pose_norm_min_median_max': [min(initial), median(initial), max(initial)],
            'post_guidance_max_pose_norm_min_median_max': [min(final), median(final), max(final)],
            'accepted_corrections_min_median_max': [min(accepted), median(accepted), max(accepted)],
        }
    result = {'source': str(source), 'methods': results, 'file_hashes': hashes,
              'independent_audit_sha256': file_digest(source / 'independent_verification.json'),
              'scope': 'pose threshold before/after common guidance; pre-guidance pose pass is not full MODEL_VALID',
              'new_physics_calls': 0, 'new_network_calls': 0}
    output.mkdir(parents=True)
    (output / 'candidate_attribution.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    analyze(args.source, args.output)
