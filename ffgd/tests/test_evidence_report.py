import json
from ffgd.evidence_report import inventory


def test_inventory_keeps_setup_failures_and_unstarted_candidate_denominator(tmp_path):
    setup=tmp_path/'setup_failure';setup.mkdir()
    (setup/'manifest.json').write_text(json.dumps(dict(status='FAILED_NEEDS_FIX',command=['python','-m','ffgd.cli','plan'],budget={'candidates':1},source_hash='old')))
    (setup/'failure_denominator.json').write_text(json.dumps(dict(requested_candidates=1,sampled=0,seed=13)))
    running=tmp_path/'running';running.mkdir()
    (running/'manifest.json').write_text(json.dumps(dict(status='RUNNING',command=['python','-m','ffgd.cli','plan'],budget={'candidates':5},source_hash='new')))
    (running/'planning_manifest.json').write_text(json.dumps(dict(budget={'seeds':[1,2,3,4,5]})))
    candidate=running/'candidate_000';candidate.mkdir()
    (candidate/'candidate.json').write_text(json.dumps(dict(index=0,seed=1,status='REJECTED',sampled=True,validated=True)))
    report=inventory(tmp_path)
    assert report['consistent'] and len(report['planning_runs'])==2 and len(report['version_groups'])==2
    assert sum(r['requested'] for r in report['planning_runs'])==6
    assert sum(r['sampled'] for r in report['planning_runs'])==1
    assert sum(r['model_valid'] for r in report['planning_runs'])==0
    active=next(r for r in report['planning_runs'] if r['run']=='running')
    assert len(active['candidates'])==5 and all(c['status']=='NOT_RUN' for c in active['candidates'][1:])
    assert active['candidates'][0]['candidate_wall_seconds'] is None


def test_execution_failure_survives_successful_retry_of_same_candidate(tmp_path):
    for name,passed in [('original',False),('refined',True)]:
        p=tmp_path/name;p.mkdir()
        (p/'manifest.json').write_text(json.dumps(dict(status='PASSED' if passed else 'FAILED_NEEDS_FIX',command=['python','-m','ffgd.cli','execute'])))
        (p/'provenance.json').write_text(json.dumps(dict(planning_directory='same',candidate_hash='unchanged',method='GUIDED_DIFFUSION',comparison_only=False)))
        (p/'result.json').write_text(json.dumps(dict(passed=passed,execution=dict(actual_steps=100 if passed else 50,angular_momentum_residual=.0006 if passed else .0012),acceptance=dict(status='EXECUTION_VALID' if passed else 'FAILED',gates={'world_angular_momentum':passed}))))
    report=inventory(tmp_path);rows=report['execution_runs']
    assert report['consistent'] and len(rows)==2
    assert {r['candidate_hash'] for r in rows}=={'unchanged'}
    assert {r['status'] for r in rows}=={'FAILED','EXECUTION_VALID'}
    assert next(r for r in rows if r['run']=='original')['angular_momentum_residual']==.0012
    assert all(r['physics_dt'] is None for r in rows)  # Missing historical fields are never guessed.
