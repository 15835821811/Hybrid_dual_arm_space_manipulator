"""Read saved results only; no new sampling, optimizer updates or physics."""
import json,hashlib,shutil
from pathlib import Path
from collections import Counter
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
RUN=ROOT/'artifacts/v7_1/endpoint_jacobian_bounded_001'
OUT=Path(__file__).resolve().parent
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def dist(values):
    a=np.array(values,dtype=float)
    return dict(n=len(a),mean=float(a.mean()),median=float(np.median(a)),p95=float(np.percentile(a,95)),max=float(a.max()),min=float(a.min())) if len(a) else dict(n=0)
def pose(r,execution=False,event=False):
    if event:p=r['position_m'];a=r['rotation_rad']
    else:
        r=r if execution else r.get('validation',r)
        p=r.get('final_position_error_m' if execution else 'frozen_goal_position_error_m')
        a=r.get('final_rotation_error_rad' if execution else 'frozen_goal_rotation_error_rad')
        if p is None or a is None:return None
    p=float(max(p)*1000);a=float(np.rad2deg(max(a)))
    return dict(position_mm=p,rotation_deg=a,Q=max(p/5,a))
def aggregate(rs,execution=False):
    values=[pose(r,execution) for r in rs];valid=[v for v in values if v is not None]
    return dict(denominator=len(rs),missing=len(rs)-len(valid),pose_pass=sum(v['Q']<=1 for v in valid),
                qualified=sum(bool(r.get('eligible',r.get('level')=='EXECUTION_VERIFIED')) for r in rs),
                **{k:dist([v[k] for v in valid]) for k in ['position_mm','rotation_deg','Q']})

stats=read(RUN/'paired_statistics.json');answer={'scope':'saved artifacts only; per trajectory max of both arms'}
answer['methods']={'B':stats['B'],**stats['reused']}
answer['unguided']={}
for label in ['A','B']:
    answer['unguided'][label]=aggregate([read(p) for p in sorted(RUN.glob('unguided/*/'+label+'/*.json'))])
answer['unguided_by_seed']={k:v for k,v in stats['unguided'].items() if not k.endswith('paired')}
answer['paired']={};answer['per_task']={}
for kind in ['candidates','executions']:
    allrows=[]
    for seed in range(3):
        p=stats[f'paired_{kind}_seed{seed}'];rows=p['rows'];allrows+=rows
        answer['paired'][f'{kind}_seed{seed}']={k:v for k,v in p.items() if k!='rows'}
        for metric in ['position_delta_mm','rotation_delta_deg']:
            answer['paired'][f'{kind}_seed{seed}'][metric]=dist([v[metric] for v in rows if v[metric] is not None])
    answer['per_task'][kind]={}
    for task in sorted({v['key'].split('/')[1] for v in allrows}):
        subset=[v for v in allrows if v['key'].split('/')[1]==task]
        answer['per_task'][kind][task]={'pairs':len(subset),**{metric:dist([v[metric] for v in subset if v[metric] is not None]) for metric in ['position_delta_mm','rotation_delta_deg']}}
    answer['paired'][kind+'_all']={'pairs':len(allrows),**{metric:dist([v[metric] for v in allrows if v[metric] is not None]) for metric in ['position_delta_mm','rotation_delta_deg']}}

answer['gradient_checks']={};answer['updates']={}
for name,folder in [('M1','diagnostic'),('M2','world_events')]:
    gs=[read(p) for p in sorted(RUN.glob(folder+'/*/gradient_*.json'))]
    answer['gradient_checks'][name]={'count':len(gs),'passed':sum(v['status']=='PASS' for v in gs),
        'direction_relative':dist([v['direction_relative_error'] for v in gs]),
        'half_relative':dist([h['relative_error'] for v in gs for h in v['half_checks']]),
        'per_noise':{str(t):{'count':sum(v['t']==t for v in gs),'passed':sum(v['t']==t and v['status']=='PASS' for v in gs)} for t in [333,667,999]}}
    records=[read(p) for p in sorted(RUN.glob(folder+'/*/result.json'))]
    accepted=[v for v in records if v['accepted']]
    base_accept=0;by_role={x:[] for x in ['event','control']};detail=[]
    for index,v in enumerate(records):
        item={'index':index,'accepted':v['accepted'],'fraction':v['fraction'],'cosine':v['cosine'],'trials':len(v['trials']),'gradient_status':v['gradient_status']}
        if v['accepted']:
            a=next(s for s in v['trials'] if s['fraction']==v['fraction']);b=next(s for s in v['base_trials'] if s['fraction']==v['fraction'])
            base_accept+=bool(b['admissible']);item['base_admissible']=b['admissible']
            for j,role in enumerate(['event','control']):
                by_role[role].append({'before':pose(v['before'][j],event=True),'combined':pose(a['after'][j],event=True),'base':pose(b['after'][j],event=True)})
        detail.append(item)
    effects={}
    for role,rows in by_role.items():
        effects[role]={stage:{k:dist([v[stage][k] for v in rows]) for k in ['position_mm','rotation_deg','Q']} for stage in ['before','base','combined']}
        effects[role]['combined_better_than_base']={k:sum(v['combined'][k]<v['base'][k] for v in rows) for k in ['position_mm','rotation_deg','Q']}
    answer['updates'][name]={'count':len(records),'accepted':len(accepted),'selected_fraction_counts':dict(Counter(str(v['fraction']) for v in accepted)),
        'base_admissible_at_combined_selected_fraction':base_accept,'actual_coupled':sum(v['actual_coupled_calls'] for v in records),
        'cosine':dist([v['cosine'] for v in records]),'effects_conditional_on_accepted':effects,'rows':detail}
    selection=read(OUT/'selection.json')['cases' if name=='M1' else 'world_sequence']
    answer['updates'][name]['accepted_per_noise']={str(t):{'count':sum(s['t']==t for s in selection),
        'accepted':sum(v['accepted'] for v,s in zip(records,selection) if s['t']==t)} for t in [333,667,999]}
answer['prior_failures']={}
for case in ['08','12']:
    g=read(RUN/'diagnostic'/case/'gradient_1.json')
    answer['prior_failures'][case]={k:g[k] for k in ['sample_id','t','status','direction_relative_error','direction_slope','direction_expected']}
t=read(RUN/'learning/training.json');old=read(ROOT/'artifacts/v7_1/expand256_train_dev_001/learning/goal_loss/training.json')
counts=t['sampling']['train_draw_counts']
answer['training']={k:t[k] for k in ['updates','actual_TRAIN_draws','TRAIN_rows_seen','VAL_updates','world_events','world_gradients','world_accepted','initial_weight_hash','trained_weight_hash','checkpoint_sha256','wall_s']}
answer['training'].update(segments=len(t['segments']),all_segments_match_A=t['segments']==old['segments'],row_counts_match_A=counts==old['train_draw_counts'],row_draw_min=min(counts.values()),row_draw_max=max(counts.values()),positive_draw_rows=sum(n>0 for n in counts.values()))
ledger=read(RUN/'ledger.json')
answer['ledger']={k:v for k,v in ledger.items() if k not in ['completed','identity']}
answer['ledger']['completed_units']=len(ledger.get('completed',{}))
answer['total_output_bytes']=sum(p.stat().st_size for p in RUN.rglob('*') if p.is_file())
for name in ['status','summary','independent_verification','query_budget','training_input','physical_contract']:
    shutil.copyfile(RUN/(name+'.json'),OUT/('run_'+name+'.json'))
(OUT/'final_statistics.json').write_text(json.dumps(answer,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({'training':answer['training'],'unguided':answer['unguided'],'paired':answer['paired'],
 'events':{k:{a:v[a] for a in ['count','accepted','base_admissible_at_combined_selected_fraction','actual_coupled','selected_fraction_counts','cosine']} for k,v in answer['updates'].items()},
 'gradient_checks':answer['gradient_checks'],'ledger':answer['ledger'],'bytes':answer['total_output_bytes']},ensure_ascii=True))
