"""Verify admission, immutable targets, validation grades and raw native traces."""
import hashlib
import zipfile
import numpy as np
from ff_prior.storage import read_json, file_digest, digest
from ff_prior.contracts import assert_frozen
from ff_prior.initial_state import InitialState
from ff_prior.spline import from_arrays
from ff_prior_complex.tools.audit_completed import audit_physics
from ff_prior_delivery_guidance.audit_results import check_candidate_validation
from .inputs import identity_for
from .protocol import PROTOCOL, proposal_id, proposal_seed
from . import data


def verify(out,identity,ledger):
    errors=[];attempts=qualified=executions=physics_steps=0
    try:
        expected,(_,selection,index,inherited_hashes)=identity_for()
        assert expected==identity and read_json(out/'run.json')['identity']==identity
        assert read_json(out/'frozen_tasks.json')==selection
        with zipfile.ZipFile(out/'executed_source.zip') as archive:
            assert all(hashlib.sha256(archive.read(p)).hexdigest()==h for p,h in identity['source_hashes'].items())
        assert ledger.data['inflight'] is None and not ledger.data.get('unresolved_call_accounting')
        refs=set(inherited_hashes);admitted=[]
        for e in selection['entries']:
            task=e['task'];assert_frozen(task)
            assert e['split']=='TRAIN' and not e['goal_relabelled'] and not e['inference_uses_teacher']
            for slot in range(4):
                s=read_json(out/'slots'/f'{e["target_id"]}_s{slot}.json')
                assert s['target_id']==e['target_id'] and s['slot']==slot and len(s['attempt_ids'])<=2
                first=None
                for attempt,sid in enumerate(s['attempt_ids']):
                    assert sid==proposal_id(e['target_slot'],slot,attempt)
                    p=read_json(out/'generation'/f'{sid}.json');r=p['candidate'];attempts+=1
                    assert p['target_slot']==e['target_slot'] and p['slot']==slot and p['attempt']==attempt
                    assert p['task_hash']==r['task_hash']==task['task_hash']
                    assert r['method']==selection['sampler'] and r['candidate_id']==slot
                    assert r['seed']==proposal_seed(e['target_slot'],slot,attempt) and not r['private_witness_used']
                    assert r['model_calls']['coupled']==r['repair_coupled_calls']+r['validation_coupled_calls']<=113
                    assert r['repair_coupled_calls']<=110 and r['validation_coupled_calls']<=3
                    if r['status']=='DONE':
                        assert r['actual_network_calls']==16 and r['sampling']['actual_network_calls']==16
                        assert r['wall_s']<=15. and r['integration_intervals']<=15872
                        check_candidate_validation(r,task)
                        assert all(v['decomposition_residual_after']<=1e-10 for v in r['sampling']['steps'])
                    else:
                        assert not r['eligible'] and not p['admitted']
                    expected_admission=bool(r['status']=='DONE' and r['eligible'] and r['reference_hash'] not in refs)
                    assert p['admitted']==expected_admission
                    assert ledger.data['completed'][sid]['status']==p['status']
                    if p['admitted']:
                        assert first is None
                        first=sid;admitted.append(sid);qualified+=1;refs.add(r['reference_hash'])
                assert s['selected_id']==first
        assert attempts<=256 and qualified<=128
        assert attempts==ledger.data['attempts'].get('generation',0)
        rows=data.population(out,ledger)
        assert {r['sample_id'] for r in rows}==set(admitted)
        by_target={e['target_id']:e for e in selection['entries']}
        selected_samples=set()
        for target in selection['execution_target_ids']:
            e=by_target[target];task=e['task'];directory=out/'executions'/target
            s=read_json(directory/'selection.json');o=read_json(directory/'outcome.json');executions+=1
            group=sorted((r for r in rows if r['target_id']==target),key=lambda r:r['sample_id'])
            chosen=group[0] if group else None
            assert s['selected_id']==(chosen['sample_id'] if chosen else None)
            assert s['task_hash']==task['task_hash'] and s['selection_before_execution'] and not s['replacement_after_failure']
            assert ledger.data['completed']['execution_'+target]['status']==o['status']
            if chosen:
                selected_samples.add(chosen['sample_id'])
                if (directory/'report.json').exists():
                    p,steps=audit_physics(directory,task,chosen['reference_hash']);physics_steps+=steps
                    assert p['level']==o['status']
                else:
                    assert o['status'] in ('ERROR','BLOCKED_BUDGET')
                assert chosen['training_eligible']==(o['status']=='EXECUTION_VERIFIED')
                assert (chosen['validation_level']=='EXECUTION_VERIFIED')==(o['status']=='EXECUTION_VERIFIED')
            else:
                assert o['status']=='NO_CANDIDATE'
        assert executions==16 and len(set(selection['execution_target_ids']))==16
        assert sum(bool(by_target[t]['scene']['objects']) for t in selection['execution_target_ids'])==8
        for row in rows:
            e=by_target[row['target_id']]
            assert row['task']==e['task'] and row['condition']==e['task']['condition'] and row['family_id']==e['family_id']
            assert not row['private_teacher_trajectory_used_for_inference']
            with np.load(out/row['arrays']) as a:
                ref=from_arrays(a)
                assert ref.reference_hash==row['reference_hash']
            with np.load(out/row['complete_initial_state']) as a:
                assert InitialState(a['values'],row['task']['contract']['robot_hash']).state_id==row['task']['contract']['initial_state_id']
            if row['sample_id'] not in selected_samples:
                assert row['execution_status']=='NOT_RUN' and row['validation_level']=='MODEL_VALID'
        merged,new=data.index_for(out,index,ledger)
        assert read_json(out/'training_index.json')['records']==merged
        assert read_json(out/'qualified_increment.json')['records']==new
        assert all(e['split']=='TRAIN' and e['execution_status'] in ('NOT_RUN','PASS') for e in new)
        assert sum(e['split']=='TRAIN' for e in merged)==13838+len(new)
        assert sum(e['split']=='VAL' for e in merged)==48
    except Exception as error:
        import traceback
        errors.append({'reason':repr(error),'traceback':traceback.format_exc()})
    return {'status':'PASS' if not errors else 'FAIL','errors':errors,'attempts_checked':attempts,
            'model_admissions_checked':qualified,'execution_units_checked':executions,
            'native_plus_replay_steps':physics_steps,'TEST_read':False,
            'scope':'raw source, admission, grade, state and execution/replay audit; no repeated simulation'}
