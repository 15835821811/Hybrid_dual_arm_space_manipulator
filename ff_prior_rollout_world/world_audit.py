"""Saved-array verification, without additional physical propagation."""
import numpy as np
import torch
from ff_prior.storage import read_json,digest
from ff_prior_delivery_guidance.audit_results import check_candidate_validation
from ff_prior_control_transfer.diagnostic import metrics
from .world_endpoint import residual,groups
from .world_training import accept
from .protocol import P


def check_refinement(dest,ident):
    g=read_json(dest/'gate.json');selection=ident['selection']
    tasks=[r['task'] for r in selection['diagnostic'] for _ in P['diagnostic_seeds']]
    assert len(g['initial_records'])==len(g['final_records'])==16
    for phase in ('initial_records','final_records'):
        for r,t in zip(g[phase],tasks):check_candidate_validation(r,t)
    assert metrics(g['initial_records'])==g['before'] and metrics(g['final_records'])==g['after']
    count=0;updates=0;current=g['initial_records']
    for number,round_info in enumerate(g['rounds']):
        directory=dest/'rounds'/f'{number:02d}'
        for j,t in enumerate(tasks):
            proof=read_json(directory/'jacobians'/f'{j:02d}.json')
            with np.load(directory/'jacobians'/f'{j:02d}.npz') as a:
                rr=residual(a['central_ee'],t['condition']['goals'])
                assert np.array_equal(rr,a['residual'])
                perturb=np.stack([[residual(ee,t['condition']['goals']) for ee in pair] for pair in a['perturbation_ee']])
                assert np.array_equal(perturb,a['perturbation_residuals'])
                J=(perturb[:,1]-perturb[:,0]).T/(2*P['world_difference'])
                assert np.array_equal(J,a['J']) and digest(a['x'])==proof['x_hash']
                expected=float(rr@J@np.asarray(proof['direction'])/2)
                assert np.isclose(expected,proof['direction_expected'],rtol=1e-12,atol=1e-12)
                assert proof['calls']['coupled']==P['world_queries_per_jacobian']
                assert proof['dimensions']==323 and proof['current_full16_output']
            count+=1
        if round_info['status']=='GRADIENT_BLOCKED':
            assert round_info['gradient_pass']<16
            break
        before=torch.load(directory/'before.pt',map_location='cpu',weights_only=False)
        prop=torch.load(directory/'proposed.pt',map_location='cpu',weights_only=False)
        retained=torch.load(directory/'retained.pt',map_location='cpu',weights_only=False)
        assert {int(v['step']) for v in prop['optimizer']['state'].values()}=={64}
        updates+=64;chosen=None
        for trial,rec in enumerate(round_info['trials']):
            records=[]
            for j,t in enumerate(tasks):
                d=directory/f'trial_{trial}'/f'{j:02d}'
                r=read_json(d/'validation.json');check_candidate_validation(r,t)
                with np.load(d/'controls.npz') as a:r=dict(r,raw_equal_prepared=bool(np.array_equal(a['raw'],a['prepared'])))
                records.append(r)
            ok,reasons=accept(current,records,before['protected'],g['rough_limit'])
            reasons['raw_unchanged_by_prepare']=all(r['raw_equal_prepared'] for r in records)
            ok=ok and reasons['raw_unchanged_by_prepare']
            assert ok==rec['accepted'] and reasons==rec['checks'] and metrics(records)==rec['metrics']
            if ok:chosen=rec['scale'];current=records
        for k,v in before['weights'].items():
            expect=v if chosen is None else (v+chosen*(prop['weights'][k]-v) if v.is_floating_point() else v)
            assert torch.equal(expect,retained['weights'][k])
    assert count==g['gradient_count'] and updates==g['actual_proposed_updates']
    assert current==g['final_records']
    assert g['checks']['qualified']==(g['after']['eligible']>=P['learnability_qualified_min'])
    assert g['checks']['old_qualified_preserved']==all(not keep or r['eligible'] for keep,r in zip(g['original_protected'],current))
    assert (g['status']=='PASS')==all(g['checks'].values())
