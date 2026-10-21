"""Saved-evidence consistency checks; no replay or physical recalculation."""
import hashlib,zipfile
import numpy as np
import torch
from ff_prior.storage import read_json,digest,file_digest
from ff_prior_pilot_train.storage import atomic_json
from ff_prior.spline import Reference
from ff_prior_delivery_guidance.audit_results import check_candidate_validation
from ff_prior_complex.tools.audit_completed import audit_physics
from .protocol import ROOT,SOURCE,PACKAGE,P
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
        for r in selection['reconstruction']:
            d=out/'reconstruction'/r['sample_id'];v=read_json(d/'validation.json');check_candidate_validation(v,r['task'])
            with np.load(d/'representation.npz') as a:
                assert np.array_equal(a['model_normalized'],a['normalized'].astype(np.float32).astype(np.float64))
                assert np.array_equal(a['prepared'],v['Z'])
                prep=read_json(d/'preparation.json')
                assert float(np.max(abs(a['decoded']-a['teacher'])))==prep['decoded_error_rad']
                assert float(np.max(abs(a['prepared']-a['teacher'])))==prep['prepared_error_rad']
                assert digest({'values':a['initial_values'],'state_mask':prep['state_mask'],'model_hash':prep['model_hash']})==r['task']['contract']['initial_state_id']
            recon+=1
        for ex in out.rglob('execution/report.json'):
            task=read_json(ex.parent/'frozen_task.json');r=read_json(ex)
            report,n=audit_physics(ex.parent,task,r['metadata']['reference_hash']);execution+=1;physics+=n
            with np.load(ex.parent/'native_trace.npz') as a,np.load(ex.parent/'replay_trace.npz') as b:
                assert np.array_equal(a['ctrl'],b['ctrl'])
                for k in ('qpos','qvel','act','time','qacc_warmstart'):
                    assert float(np.max(abs(a[k]-b[k]),initial=0))==report['replay']['maximum_errors'][k]
            with np.load(ex.parent/'geometry_scans.npz') as a:assert float(min(a['minima']))==report['minimum_clearance_m']
        gate=read_json(out/'reconstruction_gate.json')
        expected_model=all(read_json(out/'reconstruction'/r['sample_id']/'validation.json')['eligible'] and
            read_json(out/'reconstruction'/r['sample_id']/'preparation.json')['prepared_error_rad']<=P['reconstruction_control_error_rad'] for r in selection['reconstruction'])
        expected_execution=all(read_json(out/'reconstruction'/sid/'execution/outcome.json')['status']=='EXECUTION_VERIFIED' for sid in selection['reconstruction_execution_ids'])
        assert gate['model_6_of_6']==expected_model and gate['execution_3_of_3']==expected_execution
        assert (gate['status']=='PASS')==(expected_model and expected_execution)
        learning_gate=read_json(out/'learning_gate.json')
        if gate['status']=='PASS':
            from .diagnostic import metrics
            dg=read_json(out/'TRAIN_diagnostic/gate.json')
            migration=read_json(out/'TRAIN_diagnostic/migration.json')
            errors_raw=[]
            for p in sorted((out/'TRAIN_diagnostic/migration').glob('*.npz')):
                with np.load(p) as a:errors_raw.append(float(abs(a['new']-a['old']).max()))
            assert len(errors_raw)==16 and errors_raw==migration['errors']
            assert max(errors_raw)==migration['max_control_error_rad']
            if dg.get('temporary_updates'):
                fixed=[r for r in selection['diagnostic'] for _ in P['diagnostic_seeds']]
                records={}
                for phase in ('before','after'):
                    records[phase]=[]
                    for j,row in enumerate(fixed):
                        v=read_json(out/'TRAIN_diagnostic'/phase/f'{j:02d}'/'validation.json')
                        check_candidate_validation(v,row['task']);records[phase].append(v)
                    assert metrics(records[phase])==dg[phase]
                saved=torch.load(out/'TRAIN_diagnostic/after.pt',map_location='cpu',weights_only=False)
                assert {int(v['step']) for v in saved['optimizer']['state'].values()}=={256}
                a,b=dg['before'],dg['after'];checks=dg['checks']
                assert checks['qualified']==(b['eligible']>=P['learnability_qualified_min'])
                assert checks['qualified_nonregression']==(b['eligible']>=a['eligible'])
                assert checks['old_qualified_preserved']==all(not x['eligible'] or y['eligible'] for x,y in zip(records['before'],records['after']))
                assert checks['loss_decreased']==(dg['after_loss']<=P['learnability_loss_ratio']*dg['before_loss'])
                assert checks['controlled_time_detail']==(b['roughness_rms_rad']<=P['learnability_roughness_ratio']*max(a['roughness_rms_rad'],dg['teacher_roughness_rms_rad']))
                assert (dg['status']=='PASS')==all(checks.values())
            assert learning_gate['learnability']==dg
            assert (learning_gate['status']=='PASS')==(dg['status']=='PASS')
        if learning_gate['status']=='PASS':
            bundle=learning.load(out/'learning/full');t=read_json(out/'learning/full/training.json')
            state=torch.load(out/'learning/full/progress.pt',map_location='cpu',weights_only=False)
            assert t['updates']==state['updates']==6000 and t['full16_teacher_batches']==state['rollouts']==1200
            assert len(t['train_draw_counts'])==14222 and min(t['train_draw_counts'].values())>0
            assert sum(t['train_draw_counts'].values())==384000 and len(t['segments'])==300
            assert {int(v['step']) for v in state['optimizer']['state'].values()}=={6000}
            assert not t['VAL_gradient_updates'] and not t['VAL_checkpoint_selection']
            assert t['initial_weight_hash']==learning_gate['learnability']['initial_hash']
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
    return {'status':'PASS' if not errors else 'FAIL','errors':errors,'reconstruction_cases':recon,
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
        for name in ('transfer323','old_aligned','structured','learned_mean'):
            if name=='transfer323':paths=sorted((out/'comparison/DEV').glob('*/*/candidates/*.json'))
            else:paths=[ROOT/p for e in selection['reuse'] if e['scope']=='DEV' and e['method']==name for p in e['candidates']]
            comparison[name]=distribution([read_json(p) for p in paths])
        val={}
        for sj in range(2):
            val[str(sj)]={
                'transfer323':distribution([read_json(out/'unguided'/str(sj)/f'{j:02d}.json') for j in range(48)]),
                'old_aligned':distribution([read_json(SOURCE/'unguided'/str(sj)/'aligned'/f'{j:02d}.json') for j in range(48)])}
        atomic_json(out/'paired_statistics.json',{'DEV':comparison,'VAL_by_seed':val,'old_evidence_reused_not_new':True,'comparison_scope':'combined representation/prepare/fresh-training intervention; not isolated dimensionality effect'})
    summary={'status':'FINISHED' if audit['status']=='PASS' and gate['status']=='PASS' and not audit['candidate_errors'] else 'FINISHED_WITH_BLOCKED_ITEMS',
        'learning_gate':gate,'reconstruction_gate':read_json(out/'reconstruction_gate.json'),'audit':audit,'downstream':'COMPLETED' if gate['status']=='PASS' else 'BLOCKED',
        'DEV':comparison,'TRAIN_added':0,'TEST':False,'independent_review':'PENDING'}
    atomic_json(out/'summary.json',summary)
    (out/'REPORT.md').write_text('# 完整控制有界后台结果\n\n'+summary['status']+'；重建门'+gate['status']+'；原始审计'+audit['status']+'。\n\n独立审查和中文分析待收尾，未完成不写通过。\n',encoding='utf-8')
    return summary
