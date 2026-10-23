"""Read receipts and file sizes only. Never import or execute a physics module."""
from pathlib import Path
import hashlib
import json
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'ff_prior_moving_origin/evidence'
GIB = 1024 ** 3

def read(name):
    return json.loads((ROOT / name).read_text(encoding='utf-8-sig'))

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    receipt_path = 'ff_prior_fullchain/evidence/balanced_brake_closeout.json'
    used = read(receipt_path)['D3_cumulative']
    names = ['world_steer', 'contact_steer', 'horizon_restore', 'balanced_restore',
             'balanced_tracking', 'balanced_brake']
    roots = ['artifacts/v7_1/' + n + '_001' for n in names]
    roots += ['ff_prior_' + n for n in names]
    roots += ['ff_prior_balanced_scene', 'ff_prior_balanced_evidence', 'ff_prior_balanced_delivery']
    roots += ['paper/review-traces/experiment-result-to-claim/' + n for n in [
        '20261022_world_steer', '20261022_contact_steer', '20261023_horizon_restore',
        '20261023_balanced_static', '20261023_balanced_dynamic', '20261023_balanced_brake']]
    rows = []
    seen = set()
    for root in roots:
        files = list((ROOT / root).rglob('*'))
        sizes = []
        for p in files:
            if p.is_file():
                resolved = p.resolve()
                assert resolved.is_relative_to(ROOT.resolve()), 'OUTSIDE_WORKSPACE'
                assert resolved not in seen, 'DUPLICATE_INVENTORY_PATH'
                seen.add(resolved)
                sizes.append(p.stat().st_size)
        rows.append(dict(path=root, files=len(sizes), bytes=sum(sizes)))
    core = sum(r['bytes'] for r in rows[:6])
    assert core == used['core_bytes']
    inventory = sum(r['bytes'] for r in rows)
    caps = read('ff_prior_balanced_brake/evidence/budget_derivation.json')['original_D3_caps']
    cycle = read('ff_prior_balanced_brake/evidence/budget_derivation.json')['original_fullcycle_caps']
    # Preserve, rather than silently release, the previous 15 GiB reserve.
    # File inventory counts actual bytes, reserves count allocations, not usage.
    old_storage_reserve = 15 * GIB
    plan_storage_reserve = 16 * 1024 ** 2
    # Previously separate 83.399... s rounded upward; keep the whole 600 s
    # final-media reservation rather than subtract only the measured 52.735 s.
    old_readonly_reserve = 84 + 600
    plan_wall_reserve = 1800
    used_reserved = dict(coupled=used['coupled'], native_including_replay=used['native_including_replay'],
        wall_s=used['worker_wall_s'] + old_readonly_reserve + plan_wall_reserve,
        bytes=inventory + old_storage_reserve + plan_storage_reserve)
    remaining = {k: caps[k] - used_reserved[k] for k in caps}
    # Retain whole original D1 and D2 ceilings; no old raw accounting replay.
    d1d2 = dict(coupled=20000, native_including_replay=147500, wall_s=21600, bytes=20*GIB)
    full_remaining = {k: cycle[k] - d1d2[k] - used_reserved[k] for k in cycle}
    remaining = {k: min(v, full_remaining[k]) for k, v in remaining.items()}
    proposal = dict(coupled=2000, native_including_replay=29000, wall_s=6000, bytes=GIB)
    assert all(0 < proposal[k] <= remaining[k] for k in proposal)
    payload = dict(schema='moving_origin_readonly_budget_v1', timestamp_utc=datetime.now(timezone.utc).isoformat(),
        status='PROPOSAL_NOT_EXECUTABLE_ORIGIN_GATE_CONFLICT',
        receipt=receipt_path, receipt_sha256=hashlib.sha256((ROOT/receipt_path).read_bytes()).hexdigest(),
        exact_D3_worker_usage=used, D3_caps=caps, original_cycle_caps=cycle,
        file_size_inventory=rows, inventory_bytes=inventory,
        inventory_scope='Logical file lengths in listed disjoint D3 roots; no raw result contents or hashes read. Not total disk allocation.',
        carried_legacy_storage_reserve_bytes=old_storage_reserve,
        plan_storage_reserve_bytes=plan_storage_reserve,
        old_readonly_reserved_s=old_readonly_reserve, plan_reserved_s=plan_wall_reserve,
        time_scope='Recorded worker wall plus explicit compute/plan reserves, not calendar elapsed time. Unmetered historical human/agent review time is not falsely reported as measured compute.',
        conservative_D1_D2_reservation=d1d2, effective_remaining_after_reserves=remaining,
        conditional_one_unit_envelope=proposal,
        after_conditional_unit={k:remaining[k]-proposal[k] for k in proposal},
        native_derivation=dict(dt_s=.002, origin_s=2, tracking_s=27,
            execution_steps=14500, independent_replay_steps=14500, retries=0),
        proposal_note='Resource envelope only, not a runtime forecast, software implementation freeze, or permission to change physical gates.',
        new_native_steps=0, new_coupled_calls=0, new_geometry_queries=0,
        no_old_resume=True, no_old_scientific_reaudit=True)
    (OUT/'remaining_budget.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:payload[k] for k in ['inventory_bytes','effective_remaining_after_reserves',
        'conditional_one_unit_envelope','after_conditional_unit']},ensure_ascii=False,indent=2))

if __name__ == '__main__':
    main()
