"""Read raw evidence once, without rerunning propagation or torque execution."""
import hashlib
import zipfile
import numpy as np
from ff_prior.storage import read_json, digest
from ff_prior.contracts import assert_frozen, MODEL_GATES
from ff_prior.spline import Reference
from ff_prior_complex.tools.audit_completed import audit_physics
from .protocol import METHODS, PROTOCOL, ALLOCATIONS
from .inputs import identity_for


def verify(out, identity):
    errors = []
    count = executions = physics_steps = 0
    try:
        expected, _ = identity_for(identity['source'])
        assert expected == identity, 'frozen source/input identity changed'
        assert read_json(out/'run.json')['identity'] == identity
        with zipfile.ZipFile(out/'executed_source.zip') as archive:
            assert all(hashlib.sha256(archive.read(p)).hexdigest()==h for p,h in identity['source_hashes'].items())
        ledger = read_json(out/'ledger.json')
        assert ledger['inflight'] is None
        assert not ledger.get('unresolved_call_accounting'), 'unresolved interrupted work'
        frozen = read_json(out/'frozen_tasks.json')
        entries = frozen['entries']
        assert frozen['before_any_comparison'] and digest(entries)==identity['tasks_hash']==frozen['digest']
        assert len(entries)==8 and all(e['split']=='TRAIN' and not e['goal_relabelled'] for e in entries)
        assert sum(bool(e['scene']['objects']) for e in entries)==4
        representation = read_json(out/'representation_gate.json')
        diag_calls = 0
        for e in entries:
            assert_frozen(e['task'])
            for stage in ('teacher', 'roundtrip'):
                r = read_json(out/'diagnostics'/e['sample_id']/f'{stage}.json')
                assert r['status']=='DONE', 'incomplete codec diagnostic'
                assert r['task_hash']==e['task']['task_hash']
                check_candidate_validation(r, e['task'])
                assert r['model_calls']['coupled']<=3
                diag_calls += r['model_calls']['coupled']
        assert diag_calls<=48
        if not representation['original_teachers_qualified']:
            raise ValueError('original teacher gate failed; shared dependency BLOCKED')
        assert representation['late4_enabled']==representation['latent_roundtrip_qualified']
        for sj, seed in enumerate(PROTOCOL['sampling_seeds']):
            for ti, e in enumerate(entries):
                sid, task = e['sample_id'], e['task']
                for method in METHODS:
                    records = []
                    for k in range(5):
                        r = read_json(out/'candidates'/str(sj)/sid/method/f'{k:02d}.json')
                        count += 1
                        records.append(r)
                        assert r['method']==method and r['candidate_id']==k and r['seed']==seed+ti*101+k*997
                        assert r['task_hash']==task['task_hash'] and r['split']=='TRAIN'
                        if r['status']=='BLOCKED_REPRESENTATION':
                            assert method=='late4_diffusion' and not representation['late4_enabled'] and not r['eligible']
                            continue
                        assert not r['private_witness_used']
                        assert r['model_calls']['coupled']==r['repair_coupled_calls']+r['validation_coupled_calls']
                        assert 0<=r['repair_coupled_calls']<=110 and 0<=r['validation_coupled_calls']<=3
                        assert r['model_calls']['coupled']<=113
                        if r['status']!='DONE':
                            assert not r['eligible']
                            continue
                        assert r['wall_s']<=15. and r['integration_intervals']<=15872
                        check_candidate_validation(r, task)
                        expected_calls = 16 if method.endswith('diffusion') else 0
                        assert r['actual_network_calls']==r['sampling']['actual_network_calls']==expected_calls
                        if expected_calls:
                            assert len(r['sampling']['steps'])==16
                            assert max(s['decomposition_residual_after'] for s in r['sampling']['steps'])<=1e-10
                        if method=='late4_diffusion':
                            guides = r['sampling']['guidance']
                            assert [g['iteration'] for g in guides]==list(ALLOCATIONS)
                            assert sum(g['coupled_calls'] for g in guides)==r['repair_coupled_calls']
                            assert all(g['coupled_calls']<=ALLOCATIONS[g['iteration']] for g in guides)
                            assert all(g['latent_feedback_max_rad']<=1e-12 for g in guides)
                            assert np.max(abs(np.asarray(r['Z'])-guides[-1]['output_Z']))<=1e-12
                    chosen = min((r for r in records if r['eligible']), key=lambda r:(r['score'],r['candidate_id']), default=None)
                    directory = out/'executions'/str(sj)/sid/method
                    selection = read_json(directory/'selection.json')
                    assert selection['selection_before_execution'] and selection['task_hash']==task['task_hash']
                    assert selection['selected_id']==(chosen['candidate_id'] if chosen else None)
                    outcome = read_json(directory/'outcome.json')
                    executions += 1
                    if chosen is None:
                        assert outcome['status']=='NO_CANDIDATE'
                    elif (directory/'report.json').exists():
                        p, steps = audit_physics(directory, task, chosen['reference_hash'])
                        physics_steps += steps
                        assert outcome['status']==p['level']
                    else:
                        assert outcome['status']=='ERROR', 'missing native physics evidence'
        assert count==320 and executions==64
    except Exception as error:
        import traceback
        errors.append({'reason': repr(error), 'traceback': traceback.format_exc()})
    return {'status': 'PASS' if not errors else 'FAIL', 'errors': errors,
            'candidate_records_checked': count, 'execution_units_checked': executions, 'native_plus_replay_steps': physics_steps,
            'scope': 'raw record/source/selection/native trace audit; not a new simulation or independent scientist review',
            'TEST_read': False}


def check_candidate_validation(r, task):
    assert r['reference_hash']==Reference(r['Z'], task['condition']['initial_features'], task['condition']['duration']).reference_hash
    v = r['validation']
    model = all(v['gates'].get(g,False) for g in MODEL_GATES)
    goal = ('frozen_goal_position_error_m' in v and max(v['frozen_goal_position_error_m'])<=.005
            and max(v['frozen_goal_rotation_error_rad'])<=np.deg2rad(1.))
    assert model==(v['validation_level']=='MODEL_VALID')
    assert bool(goal)==r['frozen_goal_pass'] and r['eligible']==bool(model and goal)
    if model:
        assert v['gates']['geometry'] and v['gates']['geometry_refinement'] and v['gates']['coupled_convergence']
