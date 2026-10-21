"""Saved-evidence consistency checks; no replay or physical recalculation."""
import hashlib,zipfile
import numpy as np
import torch
from ff_prior.storage import read_json,digest,file_digest
from ff_prior_pilot_train.storage import atomic_json
from ff_prior.spline import Reference
from ff_prior_delivery_guidance.audit_results import check_candidate_validation
from ff_prior_complex.tools.audit_completed import audit_physics
from .protocol import ROOT,SOURCE,PACKAGE,REUSE,P
from .inputs import identity
from . import learning

def check_result(r,task,z):
    if r.get('status')=='ERROR':assert not r['eligible'];return
    check_candidate_validation(dict(r,Z=z),task)

def verify(out,ident):
    errors=[];recon=0;execution=0;physics=0;count=0;unknown=0
    try:
        assert identity()==ident
        with zipfile.ZipFile(out/'executed_source.zip') as a:
            assert all(hashlib.sha256(a.read(p)).hexdigest()==h for p,h in ident['source_hashes'].items())
        selection=ident['selection'];assert selection==read_json(out/'frozen_tasks.json')
        assert read_json(out/'reconstruction_gate.json')['evidence_reused']
        assert read_json(REUSE/'reconstruction_gate.json')['status']=='PASS'
        assert not (out/'reconstruction').exists()
        for ex in out.rglob('execution/report.json'):
            task=read_json(ex.parent/'frozen_task.json');r=read_json(ex)
            report,n=audit_physics(ex.parent,task,r['metadata']['reference_hash']);execution+=1;physics+=n
            with np.load(ex.parent/'native_trace.npz') as a,np.load(ex.parent/'replay_trace.npz') as b:
                assert np.array_equal(a['ctrl'],b['ctrl'])
                for k in ('qpos','qvel','act','time','qacc_warmstart'):
                    assert float(np.max(abs(a[k]-b[k]),initial=0))==report['replay']['maximum_errors'][k]
            with np.load(ex.parent/'geometry_scans.npz') as a:assert float(min(a['minima']))==report['minimum_clearance_m']
        from .world_audit import check_refinement
        check_refinement(out/'TRAIN_world',ident)
        learning_gate=read_json(out/'learning_gate.json')
        dg=read_json(out/'TRAIN_world/gate.json')
        assert learning_gate['learnability']==dg
        assert learning_gate['status']==dg['status']
        if learning_gate['status']=='PASS':
            bundle=learning.load(out/'learning/full');t=read_json(out/'learning/full/training.json')
            state=torch.load(out/'learning/ordinary/progress.pt',map_location='cpu',weights_only=False)
            assert t['updates']==state['updates']==6000 and t['full16_teacher_batches']==state['rollouts']==1200
            assert len(t['train_draw_counts'])==14222 and min(t['train_draw_counts'].values())>0
            assert sum(t['train_draw_counts'].values())==384000 and len(t['segments'])==300
            assert {int(v['step']) for v in state['optimizer']['state'].values()}=={6000}
            assert not t['VAL_gradient_updates'] and not t['VAL_checkpoint_selection']
            assert t['initial_weight_hash']==learning_gate['learnability']['initial_hash']
            check_refinement(out/'candidate_world',ident)
            assert read_json(out/'candidate_world/gate.json')==t['world_refinement']
            assert bundle[1].rank==19 and np.allclose(bundle[1].basis.T@bundle[1].basis,np.eye(19),atol=1e-12)
            train=read_json(out/'training_input.json');assert set(bundle[1].state['fit_ids'])==set(train['TRAIN_ids'])
            assert not set(train['TRAIN_task_hashes'])&set(selection['forbidden_TRAIN_task_hashes'])
            for sj in range(2):
                for j in range(48):
                    r=read_json(out/'unguided'/str(sj)/f'{j:02d}.json')
                    old=read_json(SOURCE/'unguided'/str(sj)/'aligned'/f'{j:02d}.json')
                    assert r['task_hash']==old['task_hash'] and r['seed']==old['seed']
                    assert len(r['steps'])==16 and r['physics_guidance_calls']==0
            groups=[('DEV',sj,ti,e,seed+ti*101) for sj,seed in enumerate(P['sampling_seeds']) for ti,e in enumerate(selection['entries'])]
            groups += [('APPLICATION_DEV',0,ti,e,e['seed']) for ti,e in enumerate(selection['application'])]
            for scope,sj,ti,e,seed in groups:
                d=out/'comparison'/scope/str(sj)/e['sample_id'];records=[]
                for k in range(5):
                    r=read_json(d/'candidates'/f'{k:02d}.json');records.append(r);count+=1
                    assert r['task_hash']==e['task']['task_hash'] and r['seed']==seed+997*k
                    if r['status']=='ERROR':unknown+=1;assert not r['eligible'];continue
                    check_result(r,e['task'],r['Z'])
                    assert r['actual_network_calls']==16 and r['model_calls']['coupled']<=113 and r['integration_intervals']<=15872 and r['wall_s']<=15
                    assert r['sampling']['final_output_exact_guided_control']
                chosen=min((r for r in records if r['eligible']),key=lambda r:(r['score'],r['candidate_id']),default=None)
                chosen_id=None if chosen is None else chosen['candidate_id']
                assert read_json(d/'execution/selection.json')['selected_id']==chosen_id
                if chosen is None:assert read_json(d/'execution/outcome.json')['status']=='NO_CANDIDATE'
            assert count==190
        else:
            assert not (out/'learning/full/model.pt').exists() and not (out/'comparison').exists()
        ledger=read_json(out/'ledger.json');assert ledger['inflight'] is None and not ledger.get('unresolved_call_accounting')
        assert sum(v for k,v in ledger['calls'].items() if k.endswith('.physics_steps'))==physics
        assert read_json(out/'budget.json')['reserved_coupled_upper']<=P['coupled_cap']
    except Exception as e:
        import traceback
        errors.append({'reason':repr(e),'traceback':traceback.format_exc()})
    return {'status':'PASS' if not errors else 'FAIL','errors':errors,'reconstruction_cases':recon,'reused_reconstruction_cases':6,'reused_teacher_executions':3,
        'new_executions':execution,'native_plus_replay_steps':physics,'new_comparison_candidates':count,
        'candidate_errors':unknown,'audit_scope':'saved evidence only; new independent reviewer required'}

def distribution(records):
    good=sum(r.get('eligible',False) for r in records);p=[];a=[]
    for r in records:
        v=r.get('validation',{})
        if 'frozen_goal_position_error_m' in v:
            p.append(max(v['frozen_goal_position_error_m'])*1000);a.append(np.rad2deg(max(v['frozen_goal_rotation_error_rad'])))
    def metrics(x):return dict(mean=float(np.mean(x)),median=float(np.median(x)),p95=float(np.percentile(x,95)),worst=float(max(x))) if x else None
    return {'eligible':good,'total':len(records),'endpoint_missing':len(records)-len(p),'position_mm':metrics(p),'rotation_deg':metrics(a)}

def report(out,audit):
    gate=read_json(out/'learning_gate.json');selection=read_json(out/'frozen_tasks.json');comparison={}
    if gate['status']=='PASS':
        for name in ('world323','old_aligned','structured','learned_mean'):
            if name=='world323':paths=sorted((out/'comparison/DEV').glob('*/*/candidates/*.json'))
            else:paths=[ROOT/p for e in selection['reuse'] if e['scope']=='DEV' and e['method']==name for p in e['candidates']]
            comparison[name]=distribution([read_json(p) for p in paths])
            if name=='world323':
                ex=[read_json(p) for p in (out/'comparison/DEV').glob('*/*/execution/outcome.json')]
                success=sum(v.get('status')=='EXECUTION_VERIFIED' for v in ex)
                comparison[name].update(execution_success=success,execution_units=len(ex),
                    engineering_gate=comparison[name]['eligible']>=144 and success>=29)
        val={}
        for sj in range(2):
            val[str(sj)]={
                'world323':distribution([read_json(out/'unguided'/str(sj)/f'{j:02d}.json') for j in range(48)]),
                'old_aligned':distribution([read_json(SOURCE/'unguided'/str(sj)/'aligned'/f'{j:02d}.json') for j in range(48)])}
        atomic_json(out/'paired_statistics.json',{'DEV':comparison,'VAL_by_seed':val,'old_evidence_reused_not_new':True,'comparison_scope':'bounded full16 world refinement plus representation/prepare/training; not isolated world-supervision benefit'})
    summary={'status':'FINISHED' if audit['status']=='PASS' and gate['status']=='PASS' and not audit['candidate_errors'] else 'FINISHED_WITH_BLOCKED_ITEMS',
        'learning_gate':gate,'reconstruction_gate':read_json(out/'reconstruction_gate.json'),'audit':audit,'downstream':'COMPLETED' if gate['status']=='PASS' else 'BLOCKED',
        'DEV':comparison,'TRAIN_added':0,'TEST':False,'independent_review':'PENDING'}
    atomic_json(out/'summary.json',summary)
    d=gate['learnability']
    (out/'REPORT.md').write_text('# 完整采样世界监督有界后台结果\n\n'+summary['status']+'；TRAIN放行门'+gate['status']+'；原始审计'+audit['status']+'。\n\n'
        +f"TRAIN全门：{d['before']['eligible']}/16 → {d['after']['eligible']}/16；完整323 Jacobian {d['gradient_count']}份；实际提议更新{d['actual_proposed_updates']}。\n\n"
        +'正式训练/VAL/DEV：'+summary['downstream']+'。旧6重建和3执行仅复用，不是新执行。\n\n独立审查和中文分析待收尾，未完成不写通过。\n',encoding='utf-8')
    return summary
