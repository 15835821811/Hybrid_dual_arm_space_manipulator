"""Read saved results only; no additional simulation or acceptance decisions."""
import json
from pathlib import Path
import numpy as np
from ff_prior.coupled_predictor import pose_errors

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / 'artifacts/v7_1/static_fullchain_002'
OUT = ROOT / 'ff_prior_static_fullchain_v2/evidence'

def read(p):
    return json.loads(p.read_text(encoding='utf-8'))

def stats(a):
    a = np.asarray(a, dtype=float)
    return dict(n=len(a), mean=float(a.mean()), median=float(np.median(a)),
                p95=float(np.quantile(a, .95)), worst=float(a.max())) if len(a) else dict(n=0)

def main():
    sel = read(RUN/'run.json')['identity']['selection']
    goals = {e['sample_id']: e['task']['condition']['goals'] for e in sel['entries']}
    rows, executions = [], []
    for unit in sel['units']:
        u = RUN/'units'/unit['id']
        for i in range(5):
            p = u/'candidates'/f'{i:02}'
            o = read(p/'outcome.json')
            row = {**unit, **o}
            row.pop('proposal_seeds')
            row['unit_id'] = unit['id']
            for stage in ['raw', 'prepared']:
                s = read(p/stage/'screen.json')
                row[stage] = dict(position_mm=max(s['position_error_m'])*1000,
                    rotation_deg=float(np.rad2deg(max(s['rotation_error_rad']))),
                    pose_pass=s['pose_pass_at_screen_grid'],
                    clearance_mm=s['metrics']['minimum_clearance_m']*1000 if 'metrics' in s else None)
            rp = p/'repair/result.json'
            row['repair'] = {k:v for k,v in read(rp).items() if k!='Z'} if rp.exists() else None
            vp = p/'validated/validation.json'
            if vp.exists():
                v = read(vp); g = v['validation']
                row['validated'] = dict(position_mm=max(g['frozen_goal_position_error_m'])*1000,
                    rotation_deg=float(np.rad2deg(max(g['frozen_goal_rotation_error_rad']))),
                    pose_pass=v['frozen_goal_pass'],
                    clearance_mm=g['geometry']['minimum_clearance_m']*1000,
                    failed_model_gates=[k for k,b in g['gates'].items() if not b])
            rows.append(row)
        ep = u/'execution/report.json'
        if ep.exists():
            e = read(ep)
            with np.load(u/'execution/native_trace.npz') as a:
                hold = np.flatnonzero(a['time']>=27-1e-9)
                poses = [pose_errors(a['ee'][j], goals[unit['task_id']]) for j in hold]
                pos = np.array([np.max(x[0])*1000 for x in poses])
                rot = np.array([np.rad2deg(np.max(x[1])) for x in poses])
                assert np.all(np.diff(a['time'])>0)
            executions.append({**unit, 'hold_position_mm':stats(pos), 'hold_rotation_deg':stats(rot),
                'clearance_mm':e['minimum_clearance_m']*1000, 'gates':e['gates'],
                'replay':e['replay'], 'native_states':e['native_states'],
                'candidate_id':read(u/'execution_selection.json')['candidate_id']})
    groups = {}
    for scene in goals:
        for method in ['structured_geometry','diffusion_geometry']:
            rr=[r for r in rows if r['task_id']==scene and r['method']==method]
            ee=[r for r in executions if r['task_id']==scene and r['method']==method]
            z=dict(proposals=len(rr), selected=sum(r['selected_for_repair'] for r in rr),
                validated_count=sum('validated' in r for r in rr), qualified=sum(r['eligible'] for r in rr),
                budget_exhausted=sum(r['status']=='BUDGET_EXHAUSTED' for r in rr),
                actual_executions=len(ee), frozen_execution_denominator=3)
            for stage in ['raw','prepared','validated']:
                ss=[r[stage] for r in rr if stage in r]
                z[stage]=dict(available=len(ss), missing=len(rr)-len(ss),
                    pose_pass=sum(s['pose_pass'] for s in ss), position_mm=stats([s['position_mm'] for s in ss]),
                    rotation_deg=stats([s['rotation_deg'] for s in ss]))
            z['execution_hold_worst_position_mm']=stats([x['hold_position_mm']['worst'] for x in ee])
            z['execution_hold_worst_rotation_deg']=stats([x['hold_rotation_deg']['worst'] for x in ee])
            groups[scene+'/'+method]=z
    # Seed-level comparisons use the selected execution, not whichever candidate looks best afterwards.
    pairs=[]
    for scene in goals:
        for seed in sorted({u['seed'] for u in sel['units']}):
            pair={'task_id':scene,'seed':seed}
            for method in ['structured_geometry','diffusion_geometry']:
                rr=[r for r in rows if r['task_id']==scene and r['seed']==seed and r['method']==method]
                ee=[r for r in executions if r['task_id']==scene and r['seed']==seed and r['method']==method]
                pair[method]=dict(qualified=sum(r['eligible'] for r in rr), proposals=5,
                    executed=len(ee), hold_position_mm=ee[0]['hold_position_mm']['worst'] if ee else None,
                    hold_rotation_deg=ee[0]['hold_rotation_deg']['worst'] if ee else None)
            pairs.append(pair)
    OUT.mkdir(exist_ok=True)
    result=dict(run=str(RUN.relative_to(ROOT)), kind='saved_result_statistics_no_new_physics',
                candidates=rows, executions=executions, paired_units=pairs, groups=groups,
                raw_screen_is_not_model_validation=True,
                unselected_not_measured_failures=36, candidate_denominator=60,
                actual_validated=23, repair_budget_unresolved=1)
    (OUT/'completed_statistics.json').write_text(json.dumps(result,indent=2,ensure_ascii=True,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps(groups,indent=2))

if __name__=='__main__':main()
