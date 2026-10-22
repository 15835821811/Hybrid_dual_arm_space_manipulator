"""One new episode, saved-evidence accounting only; no comparison runs."""
import hashlib
import zipfile
import numpy as np
from ff_prior.storage import read_json,digest
from ff_prior_pilot_train.storage import atomic_json
from .protocol import P

def verify(out,ident):
    checks={}
    with zipfile.ZipFile(out/'executed_source.zip') as archive:
        checks['source_archive']=all(hashlib.sha256(archive.read(p)).hexdigest()==h
            for p,h in ident['source_hashes'].items())
    ledger=read_json(out/'ledger.json');budget=read_json(out/'budget.json')
    checks['no_unknown_work']=ledger['inflight'] is None and not ledger.get('unresolved_call_accounting')
    checks['one_unit']=len(ledger['completed'])==1
    checks['coupled_accounted']=sum(v for k,v in ledger['calls'].items() if k.endswith('.coupled'))==budget['actual_coupled']
    checks['native_accounted']=sum(v for k,v in ledger['calls'].items() if k.endswith('.physics_steps'))==budget['actual_native_and_replay_steps']
    checks['budget']=budget['actual_coupled']<=budget['reserved_coupled']<=P['coupled_cap'] and budget['integration_intervals']<=P['integration_cap'] and budget['actual_native_and_replay_steps']<=P['native_steps_cap']
    d=out/'units/u00';result=read_json(d/'result.json')
    checks['denominator']=len(result['candidates'])==5 and result['execution_denominator']==1
    for row in result['candidates']:
        pd=d/'candidates'/f'{row["id"]:02d}'
        if not (pd/'provenance.json').exists():continue
        prov=read_json(pd/'provenance.json');a=np.load(pd/'controls.npz')
        checks[f'proposal_{row["id"]}']=bool(prov['network_calls']==16 and not prov['private_solution_used']
            and digest(a['network_raw'])==prov['network_raw_hash'] and digest(a['proposal'])==prov['proposal_hash'])
    if (d/'execution/native_trace.npz').exists():
        n=np.load(d/'execution/native_trace.npz');rp=d/'execution/replay_trace.npz'
        checks['complete_native']=len(n['ctrl'])==13750
        checks['replay_exists']=rp.exists()
        if rp.exists():
            r=np.load(rp);checks['controls_exact']=np.array_equal(n['ctrl'],r['ctrl'])
            checks['full_state_replay']=bool(np.max(abs(n['integration']-r['integration']))<=1e-10)
        er=read_json(d/'execution/report.json')
        checks['execution_classification']=((er['level']=='EXECUTION_VERIFIED')==all(er['gates'].values()))
    return {'status':'PASS' if all(checks.values()) else 'FAIL','checks':checks,
        'scope':'one new episode raw accounting; not physical success by itself','new_physics':0}

def report(out,ident,checked):
    r=read_json(out/'units/u00/result.json')
    complete=checked['status']=='PASS' and r['execution_level']=='EXECUTION_VERIFIED'
    summary=dict(status='FINISHED' if complete else 'FINISHED_WITH_BLOCKED_ITEMS',
        engineering_static_episode_pass=complete,candidates=5,qualified=r['qualified'],
        execution_denominator=1,execution_level=r['execution_level'],raw_audit=checked['status'],
        new_training_updates=0,TRAIN_added=0,comparison_runs=0,TEST=False,
        dynamic_continuous_handoff='NEXT_STAGE_NOT_YET_EXECUTED',
        full_five_obstacle_goal_complete=False,scientific_advantage='NOT_EVALUATED_THIS_RUN')
    atomic_json(out/'summary.json',summary)
    (out/'REPORT.md').write_text('# 一次Diffusion系统完整静态运行\n\n'
        +f'静态完整链路通过：{complete}；合格候选{r["qualified"]}/5；执行等级{r["execution_level"]}。\n'
        +'无相对优势对照；动态连续交接属于下一阶段，不能把本轮静态完成称为全部长期目标完成。\n'
        +'可视化位于visuals/index.html；最终分析与独立review待收尾。\n',encoding='utf-8')
    return summary
