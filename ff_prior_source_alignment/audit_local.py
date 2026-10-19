"""Read the current TRAIN/VAL index and saved probes; never open TEST records."""
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
import time
from .acceptance import compare

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).parent / 'evidence/local_alignment_002.json'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def record_problem(entry, row):
    if entry['split'] not in ('TRAIN', 'VAL'):
        return 'forbidden partition'
    if row.get('execution_status') == 'FAIL':
        return 'execution failed'
    if entry['validation_level'] not in ('MODEL_VALID', 'EXECUTION_VERIFIED'):
        return 'unsupported validation level'
    if entry['split'] == 'TRAIN' and row.get('training_eligible') is not True:
        return 'TRAIN not eligible'
    # Held-out VAL records intentionally have training_eligible=False.
    return None


def main():
    if OUT.exists():
        print('NO_WORK_ALREADY_COMPLETE: ' + str(OUT))
        return
    start = time.perf_counter()
    index_path = ROOT / 'artifacts/v7_1/world_endpoint_smoke_001/training_index.json'
    index = read(index_path)['records']
    if any(r['split'] not in ('TRAIN', 'VAL') for r in index):
        raise ValueError('index contains forbidden partition; no records opened')
    task_counts = defaultdict(Counter)
    families = defaultdict(set)
    levels, sources, initial_states = Counter(), Counter(), set()
    problems = []
    for entry in index:
        path = ROOT / entry['record']
        row = read(path)
        if sha(path) != entry['record_sha256']:
            problems.append([entry['sample_id'], 'record changed'])
        problem = record_problem(entry, row)
        if problem:
            problems.append([entry['sample_id'], problem])
        # Arrays remain bound to the previously audited immutable source index;
        # do not imply that this read-only inventory re-runs physical validation.
        split = entry['split']
        task_counts[split][row['task']['task_hash']] += 1
        families[split].add(entry['family_id'])
        levels[(split, entry['validation_level'])] += 1
        sources[row.get('goal_source', 'legacy_unspecified')] += 1
        initial_states.add(row['task']['contract']['initial_state_id'])
    overlap = sorted(families['TRAIN'] & families['VAL'])
    if overlap:
        problems.append(['TRAIN_VAL', 'family overlap'])
    saved = ROOT / 'ff_prior_paired_objectives/evidence/completed_001/probes'
    probes = {name:read(saved/f'{name}.json') for name in ('warm', 'A_v', 'B_local', 'C_world', 'D_both')}
    if any(p.get('TRAIN_only') is not True for p in probes.values()):
        raise ValueError('not TRAIN-only probes')
    report = {
        'status': 'PASS' if not problems else 'BLOCKED',
        'scope': 'read-only source inventory and prospective gate calculation; no new physics',
        'index_sha256': sha(index_path), 'index_path': index_path.relative_to(ROOT).as_posix(),
        'problems': problems, 'TEST_read': False, 'arrays_revalidated': False,
        'source_record_hashes_checked': len(index), 'complete_initial_states': len(initial_states),
        'data': {split:{'rows':sum(counts.values()), 'exact_tasks':len(counts),
                       'single_trajectory_tasks':sum(n==1 for n in counts.values()),
                       'tasks_with_at_least_2':sum(n>=2 for n in counts.values()),
                       'tasks_with_at_least_5':sum(n>=5 for n in counts.values())}
                 for split,counts in task_counts.items()},
        'validation_levels': {f'{s}:{l}':n for (s,l),n in levels.items()},
        'goal_sources':dict(sources), 'TRAIN_VAL_family_overlap':overlap,
        'probe_sha256':{name:sha(saved/f'{name}.json') for name in probes},
        'prospective_metrics':{name:compare(probes['warm']['cases'], probes[name]['cases'], probes['B_local']['cases'])
                               for name in ('A_v', 'B_local', 'C_world', 'D_both')},
        'historical_gate_rewritten':False, 'elapsed_s':time.perf_counter()-start,
        'audit_revision': 'v2: VAL training_eligible=False is correct; v1 false alarms preserved',
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
