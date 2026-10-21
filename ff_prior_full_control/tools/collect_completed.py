"""Read-only statistics from saved outcomes; no model/physics evaluation."""
import json
from collections import Counter
from pathlib import Path
import numpy as np
from ff_prior.storage import read_json, file_digest
from ff_prior_pilot_train.storage import atomic_json
from ff_prior_full_control.protocol import ROOT, OUT, PACKAGE, SOURCE


def dist(values):
    a=np.asarray([v for v in values if v is not None],dtype=float)
    return dict(n=len(a),mean=float(a.mean()),median=float(np.median(a)),p95=float(np.percentile(a,95)),worst=float(a.max()),minimum=float(a.min())) if len(a) else dict(n=0)


def candidate(path,scope,method,sj,sid,k):
    r=read_json(path);v=r.get('validation',{});g=v.get('gates',{});geo=v.get('geometry',{})
    z=np.asarray(r.get('Z',r.get('prepared_Z',[])),dtype=float)
    p=v.get('frozen_goal_position_error_m');a=v.get('frozen_goal_rotation_error_rad')
    row=dict(source=path.relative_to(ROOT).as_posix(),sha256=file_digest(path),scope=scope,method=method,seed_group=sj,sample_id=sid,candidate_id=k,
        task_hash=r.get('task_hash'),seed=r.get('seed'),eligible=bool(r.get('eligible')),goal_pass=bool(r.get('frozen_goal_pass')),
        validation_level=v.get('validation_level'),status=r.get('status','DONE'),failed_gates=[key for key in v.get('required',g) if not g.get(key)],
        position_mm=None if p is None else max(p)*1000,rotation_deg=None if a is None else float(np.rad2deg(max(a))),
        geometry_step_mm=geo.get('maximum_geometry_step_m',0)*1000,clearance_mm=geo.get('minimum_clearance_m',0)*1000,
        control_second_difference_rms_rad=float(np.sqrt(np.mean(np.diff(z,n=2,axis=0)**2))) if z.size else None,
        queries=r.get('model_calls',{}).get('coupled'),integration_intervals=r.get('integration_intervals'),wall_s=r.get('wall_s'))
    if 'raw_Z' in r and 'prepared_Z' in r:
        row['raw_prepare_max_rad']=float(np.max(abs(np.asarray(r['raw_Z'])-r['prepared_Z'])))
        row['preparation']=r.get('preparation')
    return row


def execution(directory,scope,method,sj,sid):
    outcome=read_json(directory/'outcome.json') if (directory/'outcome.json').exists() else {}
    r=read_json(directory/'report.json') if (directory/'report.json').exists() else None
    return dict(source=directory.relative_to(ROOT).as_posix(),scope=scope,method=method,seed_group=sj,sample_id=sid,
        outcome=outcome.get('status',r.get('level') if r else 'MISSING'),executed=r is not None,
        verified=bool(r and r.get('level')=='EXECUTION_VERIFIED' and r.get('replay',{}).get('passed')),
        final_position_mm=max(r['final_position_error_m'])*1000 if r else None,
        final_rotation_deg=float(np.rad2deg(max(r['final_rotation_error_rad']))) if r else None,
        hold_position_mm=max(r['hold_max_position_error_m'])*1000 if r else None,
        hold_rotation_deg=float(np.rad2deg(max(r['hold_max_rotation_error_rad']))) if r else None,
        clearance_mm=r['minimum_clearance_m']*1000 if r else None,
        failed_gates=[k for k,v in r['gates'].items() if not v] if r else [],
        native_steps=r['metadata']['physics_steps'] if r else 0,
        replay_steps=r['replay']['physics_steps'] if r else 0,
        replay_max_error=max(r['replay']['maximum_errors'].values()) if r else None)


def stats(rows,execs=()):
    return dict(total=len(rows),eligible=sum(r['eligible'] for r in rows),goal_pass=sum(r['goal_pass'] for r in rows),
        model_valid=sum(r['validation_level']=='MODEL_VALID' for r in rows),
        endpoint_missing=sum(r['position_mm'] is None for r in rows),
        failed_gates=dict(Counter(g for r in rows for g in r['failed_gates'])),
        failure_combinations=dict(Counter('|'.join(r['failed_gates']+([] if r['goal_pass'] else ['world_goal'])) for r in rows if not r['eligible'])),
        position_mm=dist([r['position_mm'] for r in rows]),rotation_deg=dist([r['rotation_deg'] for r in rows]),
        geometry_step_mm=dist([r['geometry_step_mm'] for r in rows]),clearance_mm=dist([r['clearance_mm'] for r in rows]),
        raw_prepare_max_rad=dist([r.get('raw_prepare_max_rad') for r in rows]),
        control_second_difference_rms_rad=dist([r['control_second_difference_rms_rad'] for r in rows]),
        execution=dict(slots=len(execs),attempted=sum(r['executed'] for r in execs),verified=sum(r['verified'] for r in execs),
            outcomes=dict(Counter(r['outcome'] for r in execs)),final_position_mm=dist([r['final_position_mm'] for r in execs]),
            final_rotation_deg=dist([r['final_rotation_deg'] for r in execs]),hold_position_mm=dist([r['hold_position_mm'] for r in execs]),
            hold_rotation_deg=dist([r['hold_rotation_deg'] for r in execs]),clearance_mm=dist([r['clearance_mm'] for r in execs])))


def paired(new,old,key):
    index={key(r):r for r in old};pairs=[(r,index[key(r)]) for r in new if key(r) in index]
    return dict(pairs=len(pairs),missing=len(new)-len(pairs),
        new_only_eligible=sum(a['eligible'] and not b['eligible'] for a,b in pairs),old_only_eligible=sum(b['eligible'] and not a['eligible'] for a,b in pairs),
        both_eligible=sum(a['eligible'] and b['eligible'] for a,b in pairs),
        position_new_minus_old_mm=dist([a['position_mm']-b['position_mm'] for a,b in pairs if a['position_mm'] is not None and b['position_mm'] is not None]),
        rotation_new_minus_old_deg=dist([a['rotation_deg']-b['rotation_deg'] for a,b in pairs if a['rotation_deg'] is not None and b['rotation_deg'] is not None]),
        position_improved=sum(a['position_mm']<b['position_mm'] for a,b in pairs if a['position_mm'] is not None and b['position_mm'] is not None))


def main():
    s=read_json(OUT/'frozen_tasks.json');rows=[];executions=[];reconstruction=[];val=[]
    for e in s['reuse']:
        rows.extend(candidate(ROOT/p,e['scope'],e['method'],e.get('seed_group',0),e['sample_id'],k) for k,p in enumerate(e['candidates']))
        executions.append(execution(ROOT/e['execution'],e['scope'],e['method'],e.get('seed_group',0),e['sample_id']))
    for scope,sj,e in [('DEV',sj,e) for sj in range(3) for e in s['entries']]+[('APPLICATION_DEV',0,e) for e in s['application']]:
        d=OUT/'comparison'/scope/str(sj)/e['sample_id']
        rows.extend(candidate(d/'candidates'/f'{k:02d}.json',scope,'full323',sj,e['sample_id'],k) for k in range(5))
        executions.append(execution(d/'execution',scope,'full323',sj,e['sample_id']))
    for sj in range(2):
        for j in range(48):
            for method,p in [('full323',OUT/'unguided'/str(sj)/f'{j:02d}.json'),('old_aligned',SOURCE/'unguided'/str(sj)/'aligned'/f'{j:02d}.json')]:
                r=read_json(p);val.append(candidate(p,'VAL',method,sj,r.get('sample_id',str(j)),j))
    for e in s['reconstruction']:
        d=OUT/'reconstruction'/e['sample_id'];r=candidate(d/'validation.json','TRAIN','reconstruction',0,e['sample_id'],0)
        r['preparation']=read_json(d/'preparation.json');reconstruction.append(r)
        if e['sample_id'] in s['reconstruction_execution_ids']:executions.append(execution(d/'execution','TRAIN','full323',0,e['sample_id']))
    groups={}
    for scope in ['DEV','DEV4','used8','capture_3_obstacles','capture_5_obstacles']:
        def match(r):
            if scope=='DEV':return r['scope']=='DEV'
            if scope in ('DEV4','used8'):return r['scope']=='DEV' and (r['sample_id'].startswith('DEV_')==(scope=='DEV4'))
            return r['sample_id']==scope
        groups[scope]={m:stats([r for r in rows if match(r) and r['method']==m],[r for r in executions if match(r) and r['method']==m]) for m in ('full323','old_aligned','structured','learned_mean')}
    valstats={m:stats([r for r in val if r['method']==m]) for m in ('full323','old_aligned')}
    val_by_seed={str(sj):{m:stats([r for r in val if r['seed_group']==sj and r['method']==m]) for m in valstats} for sj in range(2)}
    task_seed=[]
    for sj in range(3):
        for e in s['entries']:
            sid=e['sample_id'];task_seed.append(dict(seed_group=sj,sample_id=sid,methods={m:stats([r for r in rows if r['scope']=='DEV' and r['seed_group']==sj and r['sample_id']==sid and r['method']==m],[r for r in executions if r['scope']=='DEV' and r['seed_group']==sj and r['sample_id']==sid and r['method']==m]) for m in valstats}))
    key=lambda r:(r['seed_group'],r['task_hash'],r['candidate_id'])
    pair=paired([r for r in rows if r['scope']=='DEV' and r['method']=='full323'],[r for r in rows if r['scope']=='DEV' and r['method']=='old_aligned'],key)
    vp=paired([r for r in val if r['method']=='full323'],[r for r in val if r['method']=='old_aligned'],key)
    oldex={(r['seed_group'],r['sample_id']):r for r in executions if r['scope']=='DEV' and r['method']=='old_aligned'}
    ex_pairs=[(r,oldex[r['seed_group'],r['sample_id']]) for r in executions if r['scope']=='DEV' and r['method']=='full323' and r['executed']]
    ex_pair_stats=dict(common_executions=len(ex_pairs),excluded_new_no_candidate=36-len(ex_pairs),
        final_position_new_minus_old_mm=dist([a['final_position_mm']-b['final_position_mm'] for a,b in ex_pairs]),
        final_rotation_new_minus_old_deg=dist([a['final_rotation_deg']-b['final_rotation_deg'] for a,b in ex_pairs]),
        position_improved=sum(a['final_position_mm']<b['final_position_mm'] for a,b in ex_pairs),
        limitation='Conditional on both executed; not a full-denominator success comparison.')
    byseed={str(sj):{m:stats([r for r in rows if r['scope']=='DEV' and r['seed_group']==sj and r['method']==m],[r for r in executions if r['scope']=='DEV' and r['seed_group']==sj and r['method']==m]) for m in groups['DEV']} for sj in range(3)}
    led=read_json(OUT/'ledger.json');t=read_json(OUT/'learning/full/training.json')
    n=[r for r in executions if r['method']=='full323']; old=[r for r in executions if r['method']!='full323']
    result=dict(status=read_json(OUT/'status.json'),reconstruction=reconstruction,TRAIN_reconstruction=stats(reconstruction,[r for r in executions if r['scope']=='TRAIN']),groups=groups,VAL=valstats,VAL_by_seed=val_by_seed,DEV_paired=pair,VAL_paired=vp,DEV_task_seed=task_seed,DEV_by_seed=byseed,DEV_common_execution_pairs=ex_pair_stats,
        new_execution_accounting=dict(slots=len(n),attempted=sum(r['executed'] for r in n),verified=sum(r['verified'] for r in n),native_steps=sum(r['native_steps'] for r in n),replay_steps=sum(r['replay_steps'] for r in n)),
        historical_execution_reuse=dict(slots=len(old),attempted=sum(r['executed'] for r in old),verified=sum(r['verified'] for r in old)),
        training={k:t[k] for k in ['updates','full16_teacher_batches','TRAIN_rows_seen','actual_TRAIN_draws','initial_weight_hash','trained_weight_hash','checkpoint_sha256','VAL_gradient_updates','VAL_checkpoint_selection']},
        draw_count=dist(list(t['train_draw_counts'].values())),training_history=t['history'],ledger=dict(attempts=led['attempts'],calls=led['calls'],wall_s=led['wall_s'],inflight=led['inflight'],completed=len(led['completed'])),budget=read_json(OUT/'budget.json'),
        limitations=['Read-only saved-output statistics, no additional physics or training.','Failure rows retained; execution error distributions conditional on actual execution.','Finite control second differences are descriptive, not physical acceleration.','Old results are exact reuse, not new execution.'])
    dest=PACKAGE/'evidence/completed';atomic_json(dest/'statistics.json',result);atomic_json(dest/'compact_rows.json',dict(candidates=rows,unguided=val,executions=executions))
    print(json.dumps({k:result[k] for k in ['new_execution_accounting','historical_execution_reuse','VAL_paired','DEV_paired']},ensure_ascii=False))


if __name__=='__main__':main()
