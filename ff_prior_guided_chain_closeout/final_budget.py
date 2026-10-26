"""Conservative end-of-unit accounting, including implementation and closeout."""
from pathlib import Path
import json,time,datetime
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'ff_prior_guided_chain_closeout/evidence'
auth=json.loads((ROOT/'ff_prior_guided_chain/evidence/authorization.json').read_text(encoding='utf-8'))
base=json.loads((ROOT/'ff_prior_guided_chain/evidence/budget.json').read_text(encoding='utf-8'))
worker=json.loads((ROOT/'artifacts/v7_1/guided_diffusion_chain_001/budget.json').read_text(encoding='utf-8'))
folders=['ff_prior_guided_chain','ff_prior_guided_chain_closeout','artifacts/v7_1/guided_diffusion_chain_001','paper/review-traces/experiment-result-to-claim/20261026_guided_chain']
sizes={f:sum(p.stat().st_size for p in (ROOT/f).rglob('*') if p.is_file()) for f in folders}
stamp=time.time();elapsed=stamp-auth['started_at']
# Includes remaining commits/receipt, modified existing goal/control/findings text,
# bytecode, git-object/metadata overhead. This is a reservation, not measured use.
buffer=dict(wall_s=300,bytes=64*1024**2)
charged=dict(coupled=worker['reserved']['coupled'],native=worker['reserved']['native'],wall_s=elapsed+buffer['wall_s'],bytes=sum(sizes.values())+buffer['bytes'])
extension=dict(coupled=30000,native=50000,wall_s=39600,bytes=8053063680)
available={k:base['available_after_reservation'][k]+extension[k] for k in extension}
remaining={k:available[k]-charged[k] for k in charged}
cap=base['proposed_total_unit_caps']
assert all(charged[k]<=cap[k] and remaining[k]>=0 for k in charged)
v=dict(status='CLOSED_WITH_FINALIZATION_RESERVE',snapshot_utc=datetime.datetime.fromtimestamp(stamp,datetime.timezone.utc).isoformat(),snapshot_epoch=stamp,
 measured_authorization_to_snapshot_s=elapsed,worker_only_s=88.093,worker_authorization_elapsed_s=worker['elapsed_s'],
 measured_owned_folders_bytes=sizes,measured_owned_bytes=sum(sizes.values()),finalization_reserve=buffer,
 charged_conservative=charged,available_before_this_unit=available,remaining_after_conservative_charge=remaining,
 remaining_outside_original_full_unit_reservation=base['after_extension_and_full_reservation'],
 counters=worker['reserved'],model_intervals_are_native=False,old_reservations_reclaimed=False,
 resources_are_not_new_run_authorization=True,production_unit_retries_allowed=0,
 note='Measured snapshot plus explicit finalization reserve. Review/checker made no new neural/coupled/native/geometry calls. No old raw reread. Do not subtract worker time again; it is already within authorization elapsed.')
(OUT/'final_budget.json').write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(dict(charged=charged,remaining=remaining,measured_owned_bytes=sum(sizes.values()))))
