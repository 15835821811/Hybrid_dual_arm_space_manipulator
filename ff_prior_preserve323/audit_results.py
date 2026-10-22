"""Read-only checks of actual saved evidence; independent review follows."""
import hashlib,zipfile
import numpy as np
import torch
from ff_prior.storage import read_json,digest
from ff_prior_pilot_train.storage import atomic_json
from ff_prior_delivery_guidance.audit_results import check_candidate_validation
from ff_prior_complex.tools.audit_completed import audit_physics
from ff_prior_control_transfer.diagnostic import metrics
from ff_prior_rollout_world.audit_results import distribution
from .protocol import ROOT,SOURCE,PACKAGE,REUSE,P
from .inputs import identity
from .world_training import fixed_pool,independent_gate,accept
from .world_endpoint import residual
from . import learning


def same_low(a,b):
    return all(torch.equal(v,b[k]) for k,v in a.items() if k.startswith('low.'))


def verify_fit(dest,ident,baseline):
    g=read_json(dest/'gate.json');fixed=fixed_pool(ident['selection'],'fit')
    for key in ['initial_records','final_records']:
        assert len(g[key])==32
        for r,(row,_) in zip(g[key],fixed):check_candidate_validation(r,row['task'])
    assert metrics(g['initial_records'])==g['before'] and metrics(g['final_records'])==g['after']
    current=g['initial_records'];count=0;updates=0
    initial=torch.load(dest/'initial.pt',map_location='cpu',weights_only=False)
    for number,info in enumerate(g['rounds']):
        d=dest/'rounds'/f'{number:02d}'
        for j,(row,seed) in enumerate(fixed):
            proof=read_json(d/'jacobians'/f'{j:02d}.json')
            with np.load(d/'jacobians'/f'{j:02d}.npz') as a:
                r=residual(a['central_ee'],row['task']['condition']['goals']);assert np.array_equal(r,a['residual'])
                perturb=np.stack([[residual(p,row['task']['condition']['goals']) for p in pair] for pair in a['perturbation_ee']])
                assert np.array_equal(perturb,a['perturbation_residuals'])
                assert np.array_equal((perturb[:,1]-perturb[:,0]).T/(2*P['world_difference']),a['J'])
                assert digest(a['x'])==proof['x_hash'] and proof['calls']['coupled']==654
            count+=1
        if info['status']=='GRADIENT_BLOCKED':assert info['gradient_pass']<32;break
        before=torch.load(d/'before.pt',map_location='cpu',weights_only=False)
        prop=torch.load(d/'proposed.pt',map_location='cpu',weights_only=False)
        retained=torch.load(d/'retained.pt',map_location='cpu',weights_only=False)
        assert same_low(initial['weights'],before['weights']) and same_low(initial['weights'],prop['weights'])
        assert {int(v['step']) for v in prop['optimizer']['state'].values()}=={64};updates+=64;chosen=None
        for k,trial in enumerate(info['trials']):
            rs=read_json(d/f'trial_{k}'/'records.json')
            for j,(r,(row,_)) in enumerate(zip(rs,fixed)):
                check_candidate_validation(r,row['task'])
                with np.load(d/f'trial_{k}'/f'{j:02d}'/'controls.npz') as a:assert r['raw_equal_prepared']==np.array_equal(a['raw'],a['prepared'])
            ok,checks=accept(current,rs,before['protected'],g['rough_limit'])
            checks.update(raw_unchanged_by_prepare=all(r['raw_equal_prepared'] for r in rs),frozen_low=True)
            assert all(checks.values())==trial['accepted'] and checks==trial['checks'] and metrics(rs)==trial['metrics']
            if trial['accepted']:chosen=trial['scale'];current=rs
        for k,v in before['weights'].items():
            expected=v+chosen*(prop['weights'][k]-v) if chosen is not None and k.startswith('detail.') else v
            assert torch.equal(expected,retained['weights'][k])
    assert current==g['final_records'] and count==g['gradient_count'] and updates==g['actual_proposed_updates']
    assert g['original_protected']==[bool(x['eligible']) for x in baseline]
    assert (g['status']=='PASS')==all(g['checks'].values())
    return dict(gradient_count=count,proposal_updates=updates,status=g['status'])


def verify(out,ident):
    errors=[];execution=0;physics=0;count=0;unknown=0;world={}
    try:
        assert identity()==ident
        with zipfile.ZipFile(out/'executed_source.zip') as z:assert all(hashlib.sha256(z.read(p)).hexdigest()==h for p,h in ident['source_hashes'].items())
        s=ident['selection'];train=read_json(out/'training_input.json')
        assert set(train['excluded_check_ids'])==set(s['adaptation_excluded_ids'])
        assert not set(train['TRAIN_task_hashes'])&(set(s['check_task_hashes'])|set(s['forbidden_TRAIN_task_hashes']))
        assert train['TRAIN']==s['adaptation_TRAIN_count'] and train['source_TRAIN']==14222
        baseline=read_json(out/'TRAIN_baseline/fit/records.json');check=read_json(out/'TRAIN_baseline/check/records.json')
        for role,rs in [('fit',baseline),('check',check)]:
            assert len(rs)==32
            for r,(row,_) in zip(rs,fixed_pool(s,role)):check_candidate_validation(r,row['task'])
        gate=read_json(out/'TRAIN_gate.json')
        if (out/'TRAIN_fit/gate.json').exists():
            world['M1']=verify_fit(out/'TRAIN_fit',ident,baseline)
            after=read_json(out/'TRAIN_check_final/records.json')
            for r,(row,_) in zip(after,fixed_pool(s,'check')):check_candidate_validation(r,row['task'])
            cg=independent_gate(check,after,[r['stratum'] for r in check]);assert cg==read_json(out/'TRAIN_check_final/gate.json')==gate['check']
            assert (gate['status']=='PASS')==(world['M1']['status']=='PASS' and cg['status']=='PASS')
        else:assert gate['status']=='BLOCKED' and min(gate['coverage'].values())<2
        assert read_json(out/'learning_gate.json')['status']==gate['status']
        if gate['status']=='PASS':
            model,codec=learning.load(out/'learning/full');t=read_json(out/'learning/full/training.json')
            state=torch.load(out/'learning/ordinary/progress.pt',map_location='cpu',weights_only=False)
            assert t['updates']==state['updates']==6000 and t['full16_teacher_batches']==state['rollouts']==1200
            assert len(t['train_draw_counts'])==train['TRAIN'] and min(t['train_draw_counts'].values())>0
            assert sum(t['train_draw_counts'].values())==384000 and len(t['segments'])==300
            assert not set(t['train_draw_counts'])&set(s['adaptation_excluded_ids'])
            assert {int(v['step']) for v in state['optimizer']['state'].values()}=={6000}
            assert model.low_hash()==t['frozen_low_hash']==t['metadata']['frozen_low_hash']
            assert not t['VAL_gradient_updates'] and not t['check_gradient_updates'] and not t['VAL_checkpoint_selection']
            world['formal']=verify_fit(out/'candidate_fit',ident,baseline)
            after=read_json(out/'candidate_check_final/records.json')
            assert independent_gate(check,after,[r['stratum'] for r in check])==t['independent_check']
            for sj in range(2):
                for j in range(48):
                    r=read_json(out/'unguided'/str(sj)/f'{j:02d}.json');old=read_json(SOURCE/'unguided'/str(sj)/'aligned'/f'{j:02d}.json')
                    assert r['task_hash']==old['task_hash'] and r['seed']==old['seed'] and len(r['steps'])==16 and r['physics_guidance_calls']==0
            groups=[('DEV',sj,e,seed+ti*101) for sj,seed in enumerate(P['sampling_seeds']) for ti,e in enumerate(s['entries'])]
            groups += [('APPLICATION_DEV',0,e,e['seed']) for e in s['application']]
            for scope,sj,e,seed in groups:
                d=out/'comparison'/scope/str(sj)/e['sample_id'];rs=[]
                for k in range(5):
                    r=read_json(d/'candidates'/f'{k:02d}.json');rs.append(r);count+=1
                    assert r['task_hash']==e['task']['task_hash'] and r['seed']==seed+997*k
                    if r['status']=='ERROR':unknown+=1;assert not r['eligible'];continue
                    check_candidate_validation(r,e['task']);assert r['model_calls']['coupled']<=113 and r['integration_intervals']<=15872 and r['wall_s']<=15
                chosen=min((r for r in rs if r['eligible']),key=lambda r:(r['score'],r['candidate_id']),default=None)
                assert read_json(d/'execution/selection.json')['selected_id']==(chosen['candidate_id'] if chosen else None)
                if not chosen:assert read_json(d/'execution/outcome.json')['status']=='NO_CANDIDATE'
            assert count==190
        else:assert not (out/'learning').exists() and not (out/'unguided').exists() and not (out/'comparison').exists()
        for ex in out.glob('comparison/*/*/*/execution/report.json'):
            task=read_json(ex.parent/'frozen_task.json');r=read_json(ex);report,n=audit_physics(ex.parent,task,r['metadata']['reference_hash']);execution+=1;physics+=n
            with np.load(ex.parent/'native_trace.npz') as a,np.load(ex.parent/'replay_trace.npz') as b:
                assert np.array_equal(a['ctrl'],b['ctrl'])
                for k in ('qpos','qvel','act','time','qacc_warmstart'):assert float(np.max(abs(a[k]-b[k]),initial=0))==report['replay']['maximum_errors'][k]
        ledger=read_json(out/'ledger.json');assert ledger['inflight'] is None and not ledger.get('unresolved_call_accounting')
        assert sum(v for k,v in ledger['calls'].items() if k.endswith('.physics_steps'))==physics
        assert read_json(out/'budget.json')['reserved_coupled_upper']<=P['coupled_cap']
    except Exception as e:
        import traceback
        errors.append(dict(reason=repr(e),traceback=traceback.format_exc()))
    return dict(status='PASS' if not errors else 'FAIL',errors=errors,world=world,new_executions=execution,native_plus_replay_steps=physics,
        new_comparison_candidates=count,candidate_errors=unknown,reused_reconstruction=6,reused_teacher_executions=3,new_TRAIN=0)


def report(out,audit):
    gate=read_json(out/'TRAIN_gate.json');statistics={}
    if gate['status']=='PASS':
        selection=read_json(out/'frozen_tasks.json');val={};dev={}
        for sj in range(2):
            val[str(sj)]={
                'new':distribution([read_json(out/'unguided'/str(sj)/f'{j:02d}.json') for j in range(48)]),
                'old_aligned':distribution([read_json(SOURCE/'unguided'/str(sj)/'aligned'/f'{j:02d}.json') for j in range(48)])}
        for method in ['preserve323','old_aligned','structured','learned_mean']:
            paths=list((out/'comparison/DEV').glob('*/*/candidates/*.json')) if method=='preserve323' else [ROOT/p for e in selection['reuse'] if e['scope']=='DEV' and e['method']==method for p in e['candidates']]
            dev[method]=distribution([read_json(p) for p in paths])
        statistics=dict(VAL=val,DEV=dev,reuse_is_not_new=True)
        atomic_json(out/'paired_statistics.json',statistics)
    status='FINISHED' if audit['status']=='PASS' and gate['status']=='PASS' and not audit['candidate_errors'] else 'FINISHED_WITH_BLOCKED_ITEMS'
    result=dict(status=status,TRAIN_gate=gate,raw_audit=audit,downstream='COMPLETED' if gate['status']=='PASS' else 'BLOCKED',independent_review='PENDING',TEST=False,new_data=0,statistics=statistics)
    atomic_json(out/'summary.json',result)
    (out/'REPORT.md').write_text('# 冻结主干与分层保护有界结果\n\n'+status+'；TRAIN门'+gate['status']+'；原始审计'+audit['status']+'。\n\n独立审查与中文分析待收尾；未完成不写通过。',encoding='utf-8')
    return result
