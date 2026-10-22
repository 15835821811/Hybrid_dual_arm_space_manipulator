"""Separate read-only audit v1: normalize JSON pose lists before algebra.

Original frozen verifier and run FAILED state are preserved. No simulation,
training, candidate generation, qualification change, or run-file writes.
"""
import zipfile
import hashlib
import numpy as np
from ff_prior.storage import read_json, digest
from ff_prior_pilot_train.storage import atomic_json
from ff_prior.coupled_predictor import pose_errors
from ff_prior.spline import Reference
from ff_prior.contracts import MODEL_GATES, EXECUTION_GATES
from ff_prior_static_goal_seed.protocol import ROOT, P
from ff_prior_fixed.correction import pose_vector
from ff_prior_static_goal_seed.goal_seed import directions,descent_step,accepts


def verify_seed(path,task,prepared):
    """Reconstruct saved finite differences and accepted updates; no physics."""
    a=np.load(path/'initial.npz');z=a['Z'];ee=a['ee'];e=pose_vector(ee,task['condition']['goals'])
    ok=np.array_equal(z,prepared) and np.allclose(e,a['goal'],rtol=0,atol=1e-10)
    count=0
    for p in sorted((path/'iterations').glob('*/step.npz')):
        s=np.load(p);r=read_json(p.parent/'record.json');count+=1
        columns=[]
        for h,pose in zip(s['epsilon'],s['perturb_ee']):
            de=pose_vector(pose,task['condition']['goals'])
            columns.append((de-e)/h if h else np.zeros(12))
        j=np.asarray(columns).T
        ok=ok and np.array_equal(s['before_Z'],z) and np.array_equal(s['ee'],ee)
        ok=ok and np.array_equal(s['directions'],directions()) and np.allclose(j,s['jacobian'],rtol=1e-10,atol=1e-8)
        ok=ok and np.allclose(descent_step(j,e),s['step'],rtol=1e-9,atol=1e-10)
        accepted=[]
        for trial in r['trials']:
            if 'ee' not in trial:
                ok=ok and not all(trial['quick'].values()) and not trial['accepted'];continue
            trial_e=pose_vector(np.asarray(trial['ee']),task['condition']['goals'])
            expected=all(trial['quick'].values()) and accepts(e,trial_e)
            ok=ok and expected==trial['accepted'] and np.allclose(trial_e,trial['goal'],rtol=0,atol=1e-10)
            if trial['accepted']:accepted.append(trial)
        ok=ok and len(accepted)==int(r['accepted'])
        if accepted:
            trial=accepted[0]
            z=z+trial['scale']*np.einsum('i,ijk->jk',s['step'],s['directions'])
            ee=np.asarray(trial['ee']);e=pose_vector(ee,task['condition']['goals'])
        ok=ok and np.array_equal(z,s['after_Z'])
    if (path/'result.npz').exists():
        final=np.load(path/'result.npz');r=read_json(path/'result.json')
        ok=ok and np.array_equal(z,final['Z']) and np.array_equal(ee,final['ee'])
        ok=ok and r['iterations']==count and r['collision_checked'] is False and r['goal_relabel'] is False
        ok=ok and r['status']=='SEED_ONLY_NOT_MODEL_VALID'
    return bool(ok)


def verify(out, ident):
    checks={};notes=[]
    with zipfile.ZipFile(out/'executed_source.zip') as archive:
        checks['executed_source_exact']=all(hashlib.sha256(archive.read(p)).hexdigest()==h
            for p,h in ident['source_hashes'].items())
    checks['frozen_run_identity']=read_json(out/'run.json')['identity']==ident
    ledger=read_json(out/'ledger.json');budget=read_json(out/'budget.json')
    checks['no_unknown_work']=ledger['inflight'] is None and not ledger.get('unresolved_call_accounting')
    checks['all_6_units_recorded']=set(ledger['completed'])=={u['id'] for u in ident['selection']['units']}
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
                checks[key+f'/proposal_seed{row["id"]}']=prov['sampling_seed']==spec['proposal_seeds'][row['id']]
                if (pd/'goal_seed/initial.npz').exists():
                    checks[key+f'/goal_seed{row["id"]}']=verify_seed(pd/'goal_seed',task,z['prepared'])
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
                full_error=float(np.max(abs(native['integration']-replay['integration'])))
                checks[key+'/full_state_replay_gate']=(report['replay']['maximum_errors']['integration']==full_error
                    and (report['level']!='EXECUTION_VERIFIED' or full_error<=1e-10))
            if report['level']=='EXECUTION_VERIFIED':
                checks[key+'/verified_gates']=all(report['gates'].get(k,False) for k in EXECUTION_GATES) and all(report['gates'].values()) and report['replay']['passed']
                checks[key+'/native_time']=len(native['ctrl'])==round((task['condition']['duration']+.5)/.002)
        if result['status']!='DONE':notes.append({'unit':key,'status':result['status']})
    checks['native_trace_accounting']=execution_steps==budget['actual_native_and_replay_steps']
    return {'status':'PASS' if all(checks.values()) else 'FAIL','checks':checks,'incomplete_units':notes,
        'scope':'raw identity, denominators, saved pose/gates/control and call accounting; no new physics',
        'fresh_independent_claim_review':'PENDING_AFTER_COMPLETION'}

