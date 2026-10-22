"""Single saved-evidence audit. No new physics and no historical experiment review."""
import hashlib
import zipfile
import numpy as np
from ff_prior.storage import read_json,digest
from ff_prior_pilot_train.storage import atomic_json
from ff_prior_fixed.correction import pose_vector
from ff_prior.contracts import MODEL_GATES,EXECUTION_GATES
from ff_prior.coupled_predictor import pose_errors
from .protocol import P,ROOT


def verify(out,ident):
    checks={}
    with zipfile.ZipFile(out/'executed_source.zip') as z:
        checks['source_archive']=all(hashlib.sha256(z.read(p)).hexdigest()==h for p,h in ident['source_hashes'].items())
    checks['frozen_identity']=read_json(out/'run.json')['identity']==ident
    b=read_json(out/'budget.json');l=read_json(out/'ledger.json')
    checks['no_unknown_work']=l['inflight'] is None and not l.get('unresolved_call_accounting')
    checks['units_complete']=set(l['completed'])=={u['id'] for u in ident['selection']['units']}
    checks['coupled_accounting']=sum(v for k,v in l['calls'].items() if k.endswith('.coupled'))==b['actual_coupled']==b['reserved_coupled']
    checks['budget']=(b['reserved_coupled']<=P['coupled_cap'] and b['integration_intervals']<=P['integration_cap']
        and b['actual_native_and_replay_steps']<=P['native_steps_cap'])
    checks['output_budget']=sum(p.stat().st_size for p in out.rglob('*') if p.is_file())<=P['output_bytes_cap']
    native_steps=0;entries={e['sample_id']:e for e in ident['selection']['entries']}
    for spec in ident['selection']['units']:
        key=spec['id'];d=out/'units'/key;r=read_json(d/'result.json');task=entries[spec['task_id']]['task']
        checks[key+'/denominators']=r['candidate_denominator']==r['execution_denominator']==1
        if (d/'prefix_identity.json').exists():
            proof=read_json(d/'prefix_identity.json')
            checks[key+'/prefix_identity']=proof['passed'] and all(proof['checks'].values())
        if (d/'tree/nodes.npz').exists():
            n=np.load(d/'tree/nodes.npz');tr=read_json(d/'tree/result.json');count=tr['historical_prefix_nodes']
            old=np.load(ROOT/spec['prefix_file'])
            checks[key+'/reuse_only_bound_prefix']=all(np.array_equal(n[k][:count],old[k])
                for k in ('q','state','qpos','qvel','elapsed','parent','duration','ee','edge_id'))
            for i in range(count,len(n['q'])):
                parent=int(n['parent'][i]);ed=d/'tree/edges'/f'{int(n["edge_id"][i]):04d}'
                a=np.load(ed/'trace.npz');er=read_json(ed/'report.json')
                checks[f'{key}/node{i}']=bool(0<=parent<i and er['accepted'] and er['numerical_ok']
                    and np.array_equal(a['parent_state'],n['state'][parent])
                    and np.array_equal(a['child_state'],n['state'][i])
                    and np.array_equal(a['qpos'][-1],n['qpos'][i]) and np.array_equal(a['qvel'][-1],n['qvel'][i])
                    and np.min(a['minimum'])>=0 and np.max(a['geometry_step'])<=.002
                    and abs(n['elapsed'][i]-n['elapsed'][parent]-n['duration'][i])<1e-10
                    and n['elapsed'][i]<=27.+1e-9)
            if tr['solution_node'] is not None:
                i=int(tr['solution_node']);pos,rot=pose_errors(n['ee'][i],task['condition']['goals'])
                checks[key+'/original_goal_route']=bool(max(pos)<=.005 and max(rot)<=np.deg2rad(1)
                    and abs(n['elapsed'][i]-27.)<1e-9)
            for p in sorted((d/'tree/linear').glob('*/jacobian.npz')):
                a=np.load(p);meta=read_json(p.with_suffix('.json'))
                e=pose_vector(a['nominal_ee'],task['condition']['goals'])
                jg=np.stack([(pose_vector(ee,task['condition']['goals'])-e)/h
                    for ee,h in zip(a['perturb_ee'],a['epsilon'])],axis=1)
                jd=np.stack([(v-a['distances'])/h for v,h in zip(a['perturb_distances'],a['epsilon'])],axis=1)
                checks[f'{key}/linear{p.parent.name}']=bool(np.array_equal(a['goal'],e)
                    and np.array_equal(a['jg'],jg) and np.array_equal(a['jd'],jd)
                    and a['perturb_qpos'].shape[0]==17 and meta['queries']==37
                    and len(meta['active'])<=P['contact_active_cap'] and all(0<k[0]<=256 for k in meta['active'])
                    and np.isfinite(jg).all() and np.isfinite(jd).all())
            acts=[]
            for attempt in read_json(d/'tree/attempts.json'):
                for trial in attempt['trials']:
                    if trial.get('edge_ids'):acts.append(trial['exact_action_key'])
                    if trial['accepted']:
                        checks[f'{key}/accept{attempt["iteration"]}_{trial["variant"]}']=bool(
                            trial['after_Q']<=P['contact_world_growth']*attempt['before_Q']
                            and n['elapsed'][trial['node']]<27.)
            checks[key+'/no_repeated_exact_actions']=len(acts)==len(set(acts))
        if r['eligible']:
            v=read_json(d/r['validation_path']/'validation.json');a=np.load(d/r['validation_path']/'prediction.npz')
            pos,rot=pose_errors(a['ee'][-1],task['condition']['goals'])
            checks[key+'/model_valid']=bool(v['eligible'] and v['task_hash']==task['task_hash']
                and all(v['validation']['gates'].get(k,False) for k in MODEL_GATES)
                and max(pos)<=.005 and max(rot)<=np.deg2rad(1))
        if (d/'execution/report.json').exists():
            report=read_json(d/'execution/report.json');n=np.load(d/'execution/native_trace.npz');native_steps+=len(n['ctrl'])
            checks[key+'/native_binding']=report['task_hash']==task['task_hash'] and digest(n['ctrl'])==report['ctrl_hash']
            if (d/'execution/replay_trace.npz').exists():
                rp=np.load(d/'execution/replay_trace.npz');native_steps+=len(rp['ctrl'])
                error=float(np.max(abs(n['integration']-rp['integration'])))
                checks[key+'/full_replay']=np.array_equal(n['ctrl'],rp['ctrl']) and error==report['replay']['maximum_errors']['integration']
            if report['level']=='EXECUTION_VERIFIED':
                checks[key+'/execution_gates']=all(report['gates'].get(k,False) for k in EXECUTION_GATES) and all(report['gates'].values()) and report['replay']['passed']
    checks['native_trace_accounting']=native_steps==b['actual_native_and_replay_steps']
    return dict(status='PASS' if all(checks.values()) else 'FAIL',checks={k:bool(v) for k,v in checks.items()},
        scope='saved new evidence and bound historical-prefix identity only; no physical regeneration',
        fresh_independent_review='PENDING_AFTER_COMPLETION')


def report(out,ident,checked):
    units=[read_json(out/'units'/u['id']/'result.json') for u in ident['selection']['units']]
    qualified=sum(r['qualified'] for r in units);verified=sum(r['execution_level']=='EXECUTION_VERIFIED' for r in units)
    executed=sum((out/'units'/u['id']/'execution/report.json').exists() for u in ident['selection']['units'])
    passed=verified>0 and checked['status']=='PASS'
    s=dict(status='FINISHED' if passed else 'FINISHED_WITH_BLOCKED_ITEMS',qualified=qualified,
        candidate_denominator=3,executed=executed,verified=verified,execution_denominator=3,
        raw_audit=checked['status'],engineering_static_gate=passed,units=units,
        TRAIN_added=0,training_updates=0,TEST_run=False,advantage_comparison=False,
        historical_prefixes_reused=3,new_network_samples=0,fresh_independent_review='PENDING')
    atomic_json(out/'summary.json',s)
    (out/'REPORT.md').write_text(f'# 碰撞反馈有界修复：后台终态\n\n完整合格 {qualified}/3；实际执行 {executed}/3；执行重放通过 {verified}/3。\n\n原始审计 {checked["status"]}；新独立审查与中文分析待收尾。\n\n旧前缀仅历史模型筛选，不是新执行。没有训练、扩库、TEST或优势比较。\n',encoding='utf-8')
    return s
