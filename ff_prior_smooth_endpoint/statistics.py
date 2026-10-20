"""Read-only paired summaries; missing endpoint values retain failure denominators."""
import numpy as np
from ff_prior.storage import read_json
from ff_prior_pilot_train.storage import atomic_json
from .protocol import CONTROL

def distribution(values):
    a=np.asarray(values,dtype=float)
    if not len(a):return {'n':0}
    return {'n':len(a),'mean':float(a.mean()),'median':float(np.median(a)),'p95':float(np.percentile(a,95)),'max':float(a.max())}

def pose(record,execution=False):
    v=record if execution else record.get('validation',record)
    pk='final_position_error_m' if execution else 'frozen_goal_position_error_m'
    rk='final_rotation_error_rad' if execution else 'frozen_goal_rotation_error_rad'
    if pk not in v or rk not in v:return None
    p=max(v[pk])*1000;r=float(np.rad2deg(max(v[rk])))
    return {'position_mm':p,'rotation_deg':r,'Q':max(p/5.,r)}

def aggregate(records,execution=False):
    pairs=[pose(v,execution) for v in records];valid=[v for v in pairs if v is not None]
    return {'denominator':len(records),'missing_endpoint':len(records)-len(valid),
       'position_mm':distribution([v['position_mm'] for v in valid]),'rotation_deg':distribution([v['rotation_deg'] for v in valid]),
       'Q':distribution([v['Q'] for v in valid]),'pose_pass':sum(v['Q']<=1 for v in valid),
       'qualified':sum(v.get('eligible',v.get('level')=='EXECUTION_VERIFIED') for v in records)}

def paired(a,b):
    rows=[]
    for key in sorted(set(a)|set(b)):
        x,y=a.get(key),b.get(key)
        rows.append({'key':key,'A':x,'B':y,'position_delta_mm':y['position_mm']-x['position_mm'] if x and y else None,
                     'rotation_delta_deg':y['rotation_deg']-x['rotation_deg'] if x and y else None})
    valid=[r for r in rows if r['position_delta_mm'] is not None]
    return {'denominator':len(rows),'complete_pairs':len(valid),'position_better':sum(r['position_delta_mm']<0 for r in valid),
      'position_worse':sum(r['position_delta_mm']>0 for r in valid),'rotation_better':sum(r['rotation_delta_deg']<0 for r in valid),'rows':rows}

def summarize(out):
    answer={'scope':'original DEV4 plus already-used development8; no TEST; per-row max of both arms','reused':{},'B':{}}
    for label,method in [('structured','structured'),('mean','learned_mean'),('A','expanded_diffusion')]:
        cs=[read_json(p) for p in (CONTROL/'candidates').glob('*/*/'+method+'/*.json')]
        es=[read_json(p) for p in (CONTROL/'executions').glob('*/*/'+method+'/report.json')]
        os=[read_json(p) for p in (CONTROL/'executions').glob('*/*/'+method+'/outcome.json')]
        answer['reused'][label]={'candidate':aggregate(cs),'execution':aggregate(es,True),'execution_attempts':len(os),
           'execution_pass':sum(v['status']=='EXECUTION_VERIFIED' for v in os),'new_runs':0}
    cs=[read_json(p) for p in out.glob('candidates/*/*/*.json')];es=[read_json(p) for p in out.glob('executions/*/*/report.json')]
    os=[read_json(p) for p in out.glob('executions/*/*/outcome.json')]
    answer['B']={'candidate':aggregate(cs),'execution':aggregate(es,True),'execution_attempts':len(os),'execution_pass':sum(v['status']=='EXECUTION_VERIFIED' for v in os)}
    for sj in range(3):
        for kind,ex in [('candidates',False),('executions',True)]:
            pattern=f'{sj}/*/expanded_diffusion/'+('report.json' if ex else '*.json')
            a={('/'.join(p.relative_to(CONTROL/kind).parts[:2])+('/'+p.stem if not ex else '')):pose(read_json(p),ex) for p in (CONTROL/kind).glob(pattern)}
            pattern=f'{sj}/*/'+('report.json' if ex else '*.json')
            b={('/'.join(p.relative_to(out/kind).parts[:2])+('/'+p.stem if not ex else '')):pose(read_json(p),ex) for p in (out/kind).glob(pattern)}
            answer[f'paired_{kind}_seed{sj}']=paired(a,b)
    answer['unguided']={}
    for sj in range(2):
        sets={}
        for method in ('A','B'):
            ps=list(out.glob(f'unguided/{sj}/{method}/*.json'));rs=[read_json(p) for p in ps]
            answer['unguided'][f'{sj}_{method}']=aggregate(rs);sets[method]={p.stem:pose(r) for p,r in zip(ps,rs)}
        answer['unguided'][f'{sj}_paired']=paired(sets['A'],sets['B'])
    atomic_json(out/'paired_statistics.json',answer);return answer
