"""Read saved results only; no optimization, physics, or run mutation."""
import json
from collections import Counter
from pathlib import Path
import numpy as np
from ff_prior_rollout_world.protocol import ROOT, OUT, SOURCE, PACKAGE

def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))

def clean(v):
    if isinstance(v,dict):return {k:clean(x) for k,x in v.items()}
    if isinstance(v,(list,tuple)):return [clean(x) for x in v]
    if isinstance(v,np.generic):v=v.item()
    if isinstance(v,float) and not np.isfinite(v):return None
    return v

def dist(x):
    x=np.asarray(x,dtype=float)
    return dict(n=len(x),mean=float(x.mean()),median=float(np.median(x)),p95=float(np.percentile(x,95)),worst=float(x.max())) if len(x) else None

def endpoints(r):
    v=r.get('validation',{})
    if 'frozen_goal_position_error_m' not in v:return None
    return max(v['frozen_goal_position_error_m'])*1000,np.rad2deg(max(v['frozen_goal_rotation_error_rad']))

def aggregate(rs):
    ep=[endpoints(r) for r in rs];valid=[x for x in ep if x is not None]
    fail=Counter(k for r in rs for k,v in r.get('validation',{}).get('gates',{}).items() if not v)
    return dict(total=len(rs),eligible=sum(bool(r.get('eligible')) for r in rs),
        model_valid=sum(r.get('validation',{}).get('validation_level')=='MODEL_VALID' for r in rs),
        missing=len(rs)-len(valid),position_mm=dist([x[0] for x in valid]),rotation_deg=dist([x[1] for x in valid]),
        Q=dist([max(x[0]/5,x[1]) for x in valid]),failed_gates=dict(fail),
        prepare_comparison_available=sum('raw_Z' in r and 'prepared_Z' in r or 'raw_equal_prepared' in r for r in rs),
        prepare_unchanged=sum(np.array_equal(r['raw_Z'],r['prepared_Z']) if 'raw_Z' in r and 'prepared_Z' in r else r.get('raw_equal_prepared',False) for r in rs))

def paired(a,b):
    assert len(a)==len(b)
    pos=[];rot=[];wins=Counter()
    for x,y in zip(a,b):
        assert x['task_hash']==y['task_hash']
        if 'seed' in x and 'seed' in y:assert x['seed']==y['seed']
        wins[str(bool(x.get('eligible')))+'->'+str(bool(y.get('eligible')))]+=1
        px,py=endpoints(x),endpoints(y)
        if px and py:pos.append(py[0]-px[0]);rot.append(py[1]-px[1])
    return dict(n=len(a),transitions=dict(wins),position_delta_mm=dist(pos),rotation_delta_deg=dist(rot),
        position_improved=sum(x<0 for x in pos),rotation_improved=sum(x<0 for x in rot))

def execution(d):
    outcome=read(d/'outcome.json');p=d/'report.json'
    r=read(p) if p.exists() else {}
    return dict(status=outcome['status'],position_mm=max(r.get('final_position_error_m',[float('nan')]))*1000,
        rotation_deg=float(np.rad2deg(max(r.get('final_rotation_error_rad',[float('nan')])))),
        hold_position_mm=max(r.get('hold_max_position_error_m',[float('nan')]))*1000,
        hold_rotation_deg=float(np.rad2deg(max(r.get('hold_max_rotation_error_rad',[float('nan')])))),
        replay=r.get('replay',{}),all_gates=all(r.get('gates',{}).values()) if r else False)

def aggregate_ex(es):
    return dict(units=len(es),success=sum(x['status']=='EXECUTION_VERIFIED' for x in es),
        missing=sum(not np.isfinite(x['position_mm']) for x in es),
        **{k:dist([x[k] for x in es if np.isfinite(x[k])]) for k in ['position_mm','rotation_deg','hold_position_mm','hold_rotation_deg']})

def main():
    result={'TRAIN':{},'VAL':{},'DEV':{},'APPLICATION_DEV':{}}
    for name in ['TRAIN_world','candidate_world']:
        g=read(OUT/name/'gate.json')
        result['TRAIN'][name]={k:g[k] for k in ['status','checks','gradient_count','actual_proposed_updates']}
        result['TRAIN'][name].update(before=aggregate(g['initial_records']),after=aggregate(g['final_records']),
            original12_before=endpoints(g['initial_records'][12]),original12_after=endpoints(g['final_records'][12]),
            rounds=[dict(round=x['round'],accepted=x['accepted'],gradient_pass=x['gradient_pass'],
                trials=[dict(scale=t['scale'],accepted=t['accepted'],failed=[k for k,v in t['checks'].items() if not v]) for t in x.get('trials',[])]) for x in g['rounds']])
    allold=[];allnew=[]
    for sj in range(2):
        a=[read(SOURCE/'unguided'/str(sj)/'aligned'/f'{j:02d}.json') for j in range(48)]
        b=[read(OUT/'unguided'/str(sj)/f'{j:02d}.json') for j in range(48)]
        result['VAL'][str(sj)]=dict(old=aggregate(a),new=aggregate(b),paired=paired(a,b))
        allold+=a;allnew+=b
    result['VAL']['combined']=dict(old=aggregate(allold),new=aggregate(allnew),paired=paired(allold,allnew))
    selection=read(OUT/'frozen_tasks.json')
    for scope in ['DEV','APPLICATION_DEV']:
        newrows=[];newex=[];units=[]
        for sj in ([0,1,2] if scope=='DEV' else [0]):
            for e in (selection['entries'] if scope=='DEV' else selection['application']):
                sid=e['sample_id'];d=OUT/'comparison'/scope/str(sj)/sid
                b=[read(d/'candidates'/f'{k:02d}.json') for k in range(5)];ex=execution(d/'execution')
                reuse=next(x for x in selection['reuse'] if x['scope']==scope and x.get('seed_group',0)==sj and x['sample_id']==sid and x['method']=='old_aligned')
                a=[read(ROOT/p) for p in reuse['candidates']]
                units.append(dict(task=sid,seed_group=sj,old=aggregate(a),new=aggregate(b),paired=paired(a,b),
                    old_execution=execution(ROOT/reuse['execution']),new_execution=ex))
                newrows+=b;newex.append(ex)
        result[scope]['units']=units
        result[scope]['world323']=dict(candidates=aggregate(newrows),execution=aggregate_ex(newex))
        for method in ['old_aligned','structured','learned_mean']:
            reuse=[e for e in selection['reuse'] if e['scope']==scope and e['method']==method]
            result[scope][method]=dict(candidates=aggregate([read(ROOT/p) for e in reuse for p in e['candidates']]),
                execution=aggregate_ex([execution(ROOT/e['execution']) for e in reuse]))
    train=read(OUT/'learning/full/training.json')
    result['training']={k:train[k] for k in ['updates','TRAIN_rows_seen','actual_TRAIN_draws','full16_teacher_batches','world_proposed_updates','VAL_gradient_updates','VAL_checkpoint_selection','candidate_count','checkpoint_sha256']}
    result['audit']=read(OUT/'independent_verification.json');result['budget']=read(OUT/'budget.json')
    result['duration_s']=read(OUT/'status.json')['elapsed_s']
    result=clean(result)
    p=PACKAGE/'evidence/completed/statistics.json';p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=True)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k not in ['DEV','APPLICATION_DEV']},ensure_ascii=True,indent=2))

if __name__=='__main__':main()
