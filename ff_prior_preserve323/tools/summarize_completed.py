"""Read saved results only; never calls training, optimizer, physics, or resume."""
from pathlib import Path
import json
import math
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'artifacts/v7_1/preserve323_bounded_001'
DEST = ROOT / 'ff_prior_preserve323/evidence/completed'


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def distribution(values):
    a = np.asarray(values, dtype=float)
    finite = a[np.isfinite(a)]
    assert len(finite) == len(a), 'keep missing denominator; unexpected nonfinite value'
    return dict(mean=float(np.mean(a)), median=float(np.median(a)),
                p95=float(np.percentile(a, 95)), maximum=float(np.max(a)), count=len(a))


def metrics(rows):
    position = [max(r['validation']['frozen_goal_position_error_m']) * 1000 for r in rows]
    rotation = [max(r['validation']['frozen_goal_rotation_error_rad']) * 180 / math.pi for r in rows]
    q = [max(p / 5, a) for p, a in zip(position, rotation)]
    return dict(total=len(rows), eligible=sum(r['eligible'] for r in rows),
                model_valid=sum(r['validation']['validation_level'] == 'MODEL_VALID' for r in rows),
                raw_equal_prepare=sum(r['raw_equal_prepared'] for r in rows),
                position_mm=distribution(position), rotation_deg=distribution(rotation), Q=distribution(q))


def grouped(rows):
    return dict(all=metrics(rows), strata={key: metrics([r for r in rows if r['stratum'] == key])
                                          for key in sorted({r['stratum'] for r in rows})},
                seed_slots={str(k): metrics(rows[k::2]) for k in range(2)})


def main():
    gate = read(OUT / 'TRAIN_gate.json')
    fit = read(OUT / 'TRAIN_baseline/fit/records.json')
    check = read(OUT / 'TRAIN_baseline/check/records.json')
    check_final = read(OUT / 'TRAIN_check_final/records.json')
    baseline_q = metrics(fit)['Q']['mean']
    trials = []
    for i, trial in enumerate(gate['fit']['rounds'][0]['trials']):
        rows = read(OUT / f'TRAIN_fit/rounds/00/trial_{i}/records.json')
        result = grouped(rows)
        result.update(scale=trial['scale'], accepted=trial['accepted'], checks=trial['checks'],
                      mean_Q_improvement_percent=100 * (1 - result['all']['Q']['mean'] / baseline_q),
                      old_qualified_retained=sum(a['eligible'] and b['eligible'] for a, b in zip(fit, rows)),
                      lost_qualified=sum(a['eligible'] and not b['eligible'] for a, b in zip(fit, rows)),
                      new_qualified=sum(not a['eligible'] and b['eligible'] for a, b in zip(fit, rows)))
        trials.append(result)
    ledger = read(OUT / 'ledger.json')
    summary = dict(run=OUT.name, status=read(OUT / 'status.json'),
                   fit_baseline=grouped(fit), fit_retained=grouped(gate['fit']['final_records']),
                   check_baseline=grouped(check), check_retained=grouped(check_final),
                   retained_weights_hash_equal=gate['fit']['initial_hash'] == gate['fit']['final_hash'],
                   trials=trials, numerical_gradients=gate['fit']['gradient_count'],
                   numerical_gradient_pass=sum(read(p)['status'] == 'PASS' for p in (OUT / 'TRAIN_fit/rounds/00/jacobians').glob('*.json')),
                   temporary_optimizer_updates=gate['fit']['actual_proposed_updates'],
                   accepted_rounds=sum(x['accepted'] for x in gate['fit']['rounds']),
                   formal_training_updates=0, new_VAL_samples=0, new_DEV_candidates=0,
                   new_torque_executions=0, new_replays=0,
                   reused_teacher_reconstruction=6, reused_teacher_execution_evidence=3,
                   new_TRAIN_data=0, downstream=read(OUT / 'summary.json')['downstream'],
                   actual_coupled_calls=sum(v for k, v in ledger['calls'].items() if k.endswith('.coupled')),
                   physics_steps=sum(v for k, v in ledger['calls'].items() if k.endswith('.physics_steps')),
                   reserved_coupled=read(OUT / 'budget.json'), inflight=ledger['inflight'],
                   output_bytes=sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file()),
                   raw_audit=read(OUT / 'independent_verification.json'),
                   statistical_gate_not_physical_relaxation=True,
                   note='check PASS after exact rollback does not demonstrate a trained candidate generalizes')
    assert not (OUT / 'learning').exists() and not (OUT / 'unguided').exists() and not (OUT / 'comparison').exists()
    DEST.mkdir(parents=True, exist_ok=True)
    (DEST / 'statistics.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({key: summary[key] for key in ['actual_coupled_calls', 'physics_steps', 'temporary_optimizer_updates', 'accepted_rounds', 'output_bytes']}))
    for role in ['fit_baseline', 'check_baseline']:
        print(role, json.dumps(summary[role]['all']))
    for trial in trials:
        print('trial', json.dumps({k: trial[k] for k in ['scale', 'mean_Q_improvement_percent', 'old_qualified_retained', 'new_qualified', 'all']}))


if __name__ == '__main__':
    main()
