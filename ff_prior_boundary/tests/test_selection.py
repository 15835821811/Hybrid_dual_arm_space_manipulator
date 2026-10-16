from pathlib import Path
from ff_prior.storage import read_json
from ff_prior_boundary.experiment import targets,inputs


def test_three_real_worst_obstacle_cases_are_frozen_without_execution_results():
    source=Path('artifacts/v7_1/qualified_expansion_002');chosen=targets(source)
    assert len(chosen)==3 and len({r['context_id'] for r in chosen})==3
    parent=read_json(source/'audit_selection.json');excluded=set(parent['random_stratified']+parent['risk'])
    rows=[read_json(p) for p in (source/'records').glob('*.json')]
    for r in chosen:
        assert r['sample_id'] not in excluded
        eligible=[v for v in rows if v['context_id']==r['context_id'] and v['sample_id'] not in excluded]
        assert r['model_clearance_m']==min(v['validation']['geometry']['minimum_clearance_m'] for v in eligible)
    assert min(r['model_clearance_m'] for r in chosen)<.001


def test_input_identity_binds_arrays_states_and_immutable_goals():
    _,_,chosen,_,files=inputs('artifacts/v7_1/qualified_expansion_002')
    for r in chosen:
        sid=r['sample_id']
        assert all(p in files for p in [f'records/{sid}.json',f'arrays/{sid}.npz',f'tasks/{sid}/initial_state.npz',f'tasks/{sid}/task.json',f'generation/{sid}.json'])
