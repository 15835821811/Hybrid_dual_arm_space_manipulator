"""Read-only summary of frozen accepted traces; no new simulation or binding claim."""
import hashlib
import json
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[4]
OUTPUT = Path(__file__).resolve().parent
SOURCE = ROOT / 'v6_lite/output/runs/research_acceptance_01/simulation'
def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()
def save(name, data):
    (OUTPUT / name).write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
manifest_path = SOURCE / 'artifact_manifest.json'
manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
items = [manifest['metrics'], *manifest['traces']]
for item in items:
    assert digest(Path(item['path'])) == item['sha256'], item['path']
rows = []
for item in manifest['traces']:
    with np.load(item['path'], allow_pickle=False) as trace:
        lower = trace['task_interval_proxy_lower_m']
        i = int(np.nanargmin(lower))
        capsule = trace['task_capsule_clearance']
        slack = trace['task_interval_realized_next_start_minimum_slack_m_s']
        rows.append({'scene': Path(item['path']).stem, 'trace_sha256': item['sha256'],
                     'planning_tick_count': len(trace['task_time']),
                     'interval_selected_rows_total': int(trace['task_interval_selected_rows'].sum()),
                     'minimum_proxy_lower_m': float(lower[i]), 'minimum_proxy_lower_tick': i,
                     'minimum_proxy_lower_simulation_time_s': float(trace['task_time'][i]),
                     'minimum_recorded_capsule_clearance_m': float(np.nanmin(capsule)),
                     'minimum_realized_next_start_slack_m_s': float(np.nanmin(slack))})
report = {'schema': 'frozen_research_trace_summary_v1', 'complete': True,
          'source_run_commit': '9cd1831e7c77a340464a669eac3b5aa7cd9c3c3c',
          'source_manifest_sha256': digest(manifest_path), 'sources': items,
          'all_source_hashes_verified': True, 'scenes': rows,
          'all_five_scenes_used': len(rows) == 5,
          'planning_tick_count': sum(x['planning_tick_count'] for x in rows),
          'interval_selected_rows_total': sum(x['interval_selected_rows_total'] for x in rows),
          'claim_scope': 'Read-only old traces; no new simulation, no inferred binding counter. '
                         'Small proxy lower bound may reflect conservative early stopping; '
                         'legacy PCC binding counters do not measure bounded interval rows.',
          'next_factor': '2x target linear velocity for all original five seeds and full 27s, '
                         'with angular velocity, RNG draws, safety and simulation periods unchanged',
          'next_factor_execution_status': 'NOT_EXECUTED'}
save('report.json', report)
files = [Path(__file__), OUTPUT / 'report.json']
save('artifact_manifest.json', {'artifacts': [{'path': p.name, 'sha256': digest(p),
                                              'bytes': p.stat().st_size} for p in files]})
print(json.dumps({'scenes': len(rows), 'ticks': report['planning_tick_count'],
                  'rows': report['interval_selected_rows_total'], 'source_hashes_verified': True}))
