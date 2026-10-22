"""Read-only raw accounting; fresh independent claim review follows completion."""
import zipfile
import hashlib
import numpy as np
from ff_prior.storage import read_json, digest
from ff_prior_pilot_train.storage import atomic_json
from ff_prior.coupled_predictor import pose_errors
from ff_prior.spline import Reference
from ff_prior.contracts import MODEL_GATES, EXECUTION_GATES
from .protocol import ROOT, P


def verify(out, ident):
    checks={};notes=[]
    with zipfile.ZipFile(out/'executed_source.zip') as archive:
        checks['executed_source_exact']=all(hashlib.sha256(archive.read(p)).hexdigest()==h
            for p,h in ident['source_hashes'].items())
    checks['frozen_run_identity']=read_json(out/'run.json')['identity']==ident
    ledger=read_json(out/'ledger.json');budget=read_json(out/'budget.json')
    checks['no_unknown_work']=ledger['inflight'] is None and not ledger.get('unresolved_call_accounting')
    checks['all_12_units_recorded']=set(ledger['completed'])=={u['id'] for u in ident['selection']['units']}
    checks['coupled_counter']=sum(v for k,v in ledger['calls'].items() if k.endswith('.coupled'))==budget['actual_coupled']
    checks['native_counter']=sum(v for k,v in ledger['calls'].items() if k.endswith('.physics_steps'))==budget['actual_native_and_replay_steps']
    checks['global_limits']=(budget['actual_coupled']<=budget['reserved_coupled']<=P['coupled_cap']
        and budget['integration_intervals']<=P['integration_cap']
        and budget['actual_native_and_replay_steps']<=P['native_steps_cap'])
    checks['output_limit']=sum(p.stat().st_size for p in out.rglob('*') if p.is_file())<=P['output_bytes_cap']
    entries={e['sample_id']:e for e in ident['selection']['entries']}
    execution_steps=0
    for spec in ident['selection']['units']:
        key=spec['id'];d=out/'units'/key;result=read_json(d/'result.json');task=entries[spec['task_id']]['task']
        checks[key+'/denominator']=len(result['candidates'])==5 and result['execution_denominator']==1
        for row in result['candidates']:
            pd=d/'candidates'/f'{row["id"]:02d}'
            if (pd/'provenance.json').exists():
                prov=read_json(pd/'provenance.json');z=np.load(pd/'controls.npz')
                checks[key+f'/proposal{row["id"]}']=bool(
                    digest(z['network_raw'])==prov['network_raw_hash'] and digest(z['proposal'])==prov['proposal_hash']
                    and prov['private_solution_used'] is False and prov['network_calls']==(16 if spec['method']=='diffusion_geometry' else 0))
                if prov['preparation']['unchanged']:
                    checks[key+f'/prepare{row["id"]}']=np.array_equal(z['proposal'],z['prepared'])
            if row['eligible']:
                v=read_json(pd/'validated/validation.json');a=np.load(pd/'validated/prediction.npz')
                pos,rot=pose_errors(a['ee'][-1],task['condition']['goals'])
                ref=Reference(v['Z'],task['condition']['initial_features'],task['condition']['duration'])
                checks[key+f'/qualified{row["id"]}']=bool(v['eligible'] and v['task_hash']==task['task_hash']
                    and ref.reference_hash==v['reference_hash']
                    and all(v['validation']['gates'].get(k,False) for k in MODEL_GATES)
                    and max(pos)<=.005 and max(rot)<=np.deg2rad(1))
        if (d/'execution_selection.json').exists():
            selection=read_json(d/'execution_selection.json')
            eligible=sorted((r for r in result['candidates'] if r['eligible']),key=lambda r:(r['score'],r['id']))
            checks[key+'/selection']=(selection['candidate_id']==(eligible[0]['id'] if eligible else None)
                and selection['candidate_digest']==digest(result['candidates']) and selection['before_execution'])
        if (d/'execution/report.json').exists():
            report=read_json(d/'execution/report.json');native=np.load(d/'execution/native_trace.npz')
            execution_steps+=len(native['ctrl'])
            checks[key+'/execution_binding']=(report['task_hash']==task['task_hash']
                and digest(native['ctrl'])==report['ctrl_hash']
                and read_json(d/'execution/initial_state.json')['state_id']==report['initial_state_id'])
            if (d/'execution/replay_trace.npz').exists():
                replay=np.load(d/'execution/replay_trace.npz');execution_steps+=len(replay['ctrl'])
                checks[key+'/replay_controls']=np.array_equal(native['ctrl'],replay['ctrl'])
            if report['level']=='EXECUTION_VERIFIED':
                checks[key+'/verified_gates']=all(report['gates'].get(k,False) for k in EXECUTION_GATES) and report['replay']['passed']
                checks[key+'/native_time']=len(native['ctrl'])==round((task['condition']['duration']+.5)/.002)
        if result['status']!='DONE':notes.append({'unit':key,'status':result['status']})
    checks['native_trace_accounting']=execution_steps==budget['actual_native_and_replay_steps']
    return {'status':'PASS' if all(checks.values()) else 'FAIL','checks':checks,'incomplete_units':notes,
        'scope':'raw identity, denominators, saved pose/gates/control and call accounting; no new physics',
        'fresh_independent_claim_review':'PENDING_AFTER_COMPLETION'}


def report(out,ident,checked):
    methods={m:{'qualified':0,'candidate_denominator':30,'executed':0,'verified':0,'execution_denominator':6}
             for m in P['methods']}
    scenes={name:0 for name in P['tasks']};rows=[]
    for u in ident['selection']['units']:
        d=out/'units'/u['id'];r=read_json(d/'result.json');m=methods[u['method']]
        m['qualified']+=r['qualified'];passed=r['execution_level']=='EXECUTION_VERIFIED'
        m['verified']+=int(passed);m['executed']+=int((d/'execution/report.json').exists());scenes[u['task_id']]+=int(passed)
        rows.append({'unit':u,'qualified':r['qualified'],'execution_level':r['execution_level'],'status':r['status']})
    passed=checked['status']=='PASS' and all(v>=1 for v in scenes.values())
    summary={'status':'FINISHED' if passed else 'FINISHED_WITH_BLOCKED_ITEMS','methods':methods,
        'scene_verified':scenes,'engineering_static_gate':passed,'raw_audit':checked['status'],
        'units':rows,'diffusion_relative_advantage':'NOT_ESTABLISHED_BY_STATUS',
        'training_updates':0,'TRAIN_added':0,'TEST_run':False,'fresh_review':'PENDING'}
    atomic_json(out/'summary.json',summary)
    lines=['# P1 静态全链路原始结果','',
        '后台已结束；独立审查和最终分析待收尾。仅新实际执行与重放通过的条目可称 EXECUTION_VERIFIED。',
        '', '|方法|已认证候选/全部提案|执行重放通过/冻结单元|实际执行数|','|---|---|---|---|']
    for method,m in methods.items():lines.append(f'|{method}|{m["qualified"]}/30|{m["verified"]}/6|{m["executed"]}|')
    lines+=['',f'静态工程门：{passed}；原始审计：{checked["status"]}。',
        '每单元5提案，仅事前按公开初筛选2条做完整修正/认证；未选中不冒充物理失败或合格。',
        '65维旧网络没有障碍条件。几何绕行、完整323修正与执行控制贡献单列，网络优势待配对分析。',
        '没有新训练、扩库或TEST；动态跟踪与连续交接仍待后续阶段。']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return summary
