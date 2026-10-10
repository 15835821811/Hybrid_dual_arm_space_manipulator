"""Read-only DEV result accounting, retaining failures and unrun denominators."""
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from v6_4.route_optimizer_protocol import read,write,sha,digest
from v6_4.closed_loop_warmstart_validation import actual_near_quality,verify_actual_slot,verify_phase_selections


def csv_write(path,rows):
    fields=list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open('x',newline='',encoding='utf8') as f:
        w=csv.DictWriter(f,fields);w.writeheader()
        for row in rows:w.writerow({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(dict,list)) else v for k,v in row.items()})


def hit(rows,predicate):
    for i,r in enumerate(rows,1):
        if predicate(r):return dict(status='HIT',slot=i)
    return dict(status='RIGHT_CENSORED' if len(rows)==8 else 'TECHNICAL_INCOMPLETE',slot=None,observed_slots=len(rows))


def metrics_sum(rows):
    keys=('prediction_physics_steps','private_preview_physics_steps','native_geometry_query_calls','independent_saved_torque_replay_steps','qp_solve_calls','actual_physics_steps')
    return {k:sum((r.get('costs') or {}).get(k,0) for r in rows) for k in keys}


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False);run=a.run.resolve();plan=read(run/'plan.json')
    complete=(run/'execution_complete.json').exists()
    if (run/'validation/sealed_selections/all_selections.json').exists():verify_phase_selections(run/'validation')
    streams=[];candidates=[];endpoints=[];actuals={};source_paths=set()
    for frozen in plan['tasks']:
        tid=frozen['task_id']
        for method in ('P0','P1','P2'):
            stream=run/'search'/tid/method;root=stream/'planning'/tid
            reg=root/'candidate_registry.json';partial=root/'candidate_registry.jsonl'
            rows=read(reg) if reg.exists() else [json.loads(line) for line in partial.read_text().splitlines()] if partial.exists() else []
            selection=read(root/'selection.json') if (root/'selection.json').exists() else None
            proposals=read(root/'proposals.json') if (root/'proposals.json').exists() else []
            cost=read(stream/'planning_cost.json') if (stream/'planning_cost.json').exists() else {}
            receipt=run/'command_logs'/('search_'+tid+'_'+method)/'receipt.json'
            receipt=read(receipt) if receipt.exists() else {}
            raw=[r for r in rows if r.get('source')=='diffusion']
            eligible=lambda r:bool(r.get('prediction_admissible'))
            b30=lambda r:eligible(r) and (r.get('prediction_metrics') or {}).get('d_support',-1)>=.03
            sr=dict(task_id=tid,strategy=method,status='COMPLETED' if cost.get('complete') else 'NOT_RUN' if not rows else 'INCOMPLETE',
                candidate_slots=len(rows),proposal_attempts=len(proposals),prediction_calls=sum(r.get('prediction_rollout_started',False) for r in rows),
                raw_generated=len(raw),raw_legal=sum(r.get('raw_seed_diagnostics',{}).get('raw_legal',False) for r in raw),
                raw_rejections=dict(Counter(r.get('raw_seed_diagnostics',{}).get('rejection_reason') for r in raw if not r.get('raw_seed_diagnostics',{}).get('raw_legal'))),
                full_nominal_qualified=sum(eligible(r) for r in rows),B30_candidates=sum(b30(r) for r in rows),
                first_complete=hit(rows,eligible),first_B30=hit(rows,b30),near_R12='N/A_NOT_RUN_BUDGET',
                cache_hits=selection['budget']['cache_hits'] if selection else None,
                cold_service_s=cost.get('end_to_end_cold_service_s'),cold_outer_process_s=receipt.get('elapsed_s'),
                optimizer_s=selection.get('elapsed_wall_s') if selection else None,initializer_setup=cost.get('setup'),
                cost_counting='once per shared A/B pool',**metrics_sum(rows))
            streams.append(sr)
            for i,r in enumerate(rows):
                m=r.get('prediction_metrics') or {};diag=r.get('raw_seed_diagnostics') or {}
                candidates.append(dict(task_id=tid,strategy=method,slot_1based=i+1,candidate_id=r['candidate_id'],
                    source=r['source'],origin_source=r.get('origin_source'),parent_candidate_id=r.get('parent_candidate_id'),
                    family=r['family'],raw_z_m=r.get('raw_z_m', (r.get('plan') or {}).get('z_m')),
                    raw_legal=diag.get('raw_legal'),rejection=diag.get('rejection_reason'),status=r['status'],
                    prediction_admissible=r.get('prediction_admissible'),B30=b30(r),I_support=m.get('I_support'),L_full=m.get('L_full'),d_support=m.get('d_support'),
                    prediction_steps=r.get('prediction_steps'),evidence_alias=r.get('evidence_alias_of_candidate_id'),
                    plan_sha256=r.get('plan_sha256'),**metrics_sum([r])))
            if selection:
                assert digest(rows)==selection['registry_content_sha256']
                source_paths.update([reg,root/'selection.json',root/'proposals.json',stream/'planning_cost.json'])
            for pref in ('A','B'):
                directory=run/'validation/actual'/tid/(method+'_'+pref);path=directory/'slot.json'
                slot=verify_actual_slot(directory) if path.exists() else None
                if slot:actuals[tid,method,pref]=slot;source_paths.add(path)
                chosen=selection['preferences'][pref] if selection else {}
                q=(slot or {}).get('quality') or {}
                endpoints.append(dict(task_id=tid,strategy=method,preference=pref,
                    status=slot['diagnostic_category'] if slot else 'NOT_RUN',
                    nominal_preference_met=chosen.get('preference_met'),
                    full_task_and_five_gates=bool(slot and slot['full_task_success']),
                    five_gates=(slot or {}).get('five_gates'),actual_steps=(slot or {}).get('actual_steps'),
                    actual_B30=(slot or {}).get('clearance_30mm_met'),
                    selected_candidate_id=chosen.get('source_candidate_id'),source_attribution=chosen.get('source_attribution'),
                    selected_origin=chosen.get('selected_origin_source'),selected_lineage=chosen.get('selected_lineage'),
                    unique_run=bool(slot and slot['unique_run']),alias_of_slot=(slot or {}).get('alias_of_slot'),
                    plan_sha256=(slot or {}).get('plan_sha256'),near_R12='N/A_NOT_RUN_BUDGET',
                    actual_I_support=q.get('I_support'),actual_L_full=q.get('L_full'),actual_d_support=q.get('d_support'),
                    base_translation_peak_m=q.get('base_translation_peak_m'),base_rotation_peak_rad=q.get('base_rotation_peak_rad'),
                    actual_service_s=(slot or {}).get('elapsed_wall_s'),
                    prediction_actual_difference=(slot or {}).get('prediction_actual_difference'),
                    **metrics_sum([slot] if slot else [])))
    for r in endpoints:
        key=r['task_id'],r['strategy'],r['preference'];base=actuals.get((r['task_id'],'P0',r['preference']))
        r['near_P0']=actual_near_quality(actuals[key],base,r['preference']) if key in actuals and base else None
    aggregate={}
    for method in ('P0','P1','P2'):
        ss=[s for s in streams if s['strategy']==method];ee=[e for e in endpoints if e['strategy']==method]
        aggregate[method]=dict(full_actual_passed=sum(e['full_task_and_five_gates'] for e in ee),full_actual_denominator=4,
            A_passed=sum(e['full_task_and_five_gates'] for e in ee if e['preference']=='A'),
            B_passed=sum(e['full_task_and_five_gates'] for e in ee if e['preference']=='B'),preference_denominator=2,
            B_actual_B30=sum(e['full_task_and_five_gates'] and e['actual_B30'] is True for e in ee if e['preference']=='B'),
            NO_PLAN=sum(e['status']=='NO_PLAN' for e in ee),NOT_RUN=sum(e['status']=='NOT_RUN' for e in ee),
            TOOL_ERROR=sum(e['status']=='TOOL_ERROR' for e in ee),
            raw_legal=sum(s['raw_legal'] for s in ss),raw_generated=sum(s['raw_generated'] for s in ss),
            full_nominal_qualified=sum(s['full_nominal_qualified'] for s in ss),B30_candidates=sum(s['B30_candidates'] for s in ss),
            candidate_slots=sum(s['candidate_slots'] for s in ss),prediction_calls=sum(s['prediction_calls'] for s in ss),
            cold_service_s=sum(s['cold_service_s'] for s in ss) if all(s['cold_service_s'] is not None for s in ss) else None,
            cold_outer_process_s=sum(s['cold_outer_process_s'] for s in ss) if all(s['cold_outer_process_s'] is not None for s in ss) else None,
            near_P0_A=sum(e['near_P0'] is True for e in ee if e['preference']=='A'),
            near_P0_B=sum(e['near_P0'] is True for e in ee if e['preference']=='B'),
            near_R12='N/A_NOT_RUN_BUDGET',
            selected_source_counts=dict(Counter(e['source_attribution'] for e in ee)),
            prediction_physics_steps=sum(s['prediction_physics_steps'] for s in ss),
            private_preview_physics_steps=sum(s['private_preview_physics_steps'] for s in ss),
            native_geometry_query_calls=sum(s['native_geometry_query_calls'] for s in ss))
    reservations={k:sum(read(p)['count'] for p in (run/'budget_ledger'/k).glob('*.json')) for k in plan['limits']}
    budget=dict(limits=plan['limits'],reservations=reservations,
        consumed_candidate_slots=sum(s['candidate_slots'] for s in streams),
        prediction_calls=sum(s['prediction_calls'] for s in streams),logical_actual_present=len(actuals),
        unique_actual_runs=sum(s['unique_run'] for s in actuals.values()),
        actual_aliases=sum(bool(s.get('alias_of_slot')) for s in actuals.values()),
        no_plan=sum(s['diagnostic_category']=='NO_PLAN' for s in actuals.values()),
        main_actual_steps=sum((s.get('costs') or {}).get('actual_physics_steps',0) for s in actuals.values()),
        new_training_updates=0,new_C3_TEST_runs=0,
        static_geometry_queries=read(run/'prepare_receipt.json')['geometry_queries'])
    assert budget['consumed_candidate_slots']<=48 and budget['logical_actual_present']<=12
    write(a.output/'summary.json',dict(status='COMPLETED' if complete else 'INCOMPLETE',strategies=aggregate,budget=budget))
    csv_write(a.output/'dev_candidates.csv',candidates);csv_write(a.output/'dev_streams.csv',streams);csv_write(a.output/'dev_endpoints.csv',endpoints)
    write(a.output/'input_sha256.json',{str(p.relative_to(run)):sha(p) for p in sorted(source_paths)})
    print(json.dumps(dict(status='COMPLETED' if complete else 'INCOMPLETE',strategies=aggregate,budget=budget),ensure_ascii=False))


if __name__=='__main__':main()
