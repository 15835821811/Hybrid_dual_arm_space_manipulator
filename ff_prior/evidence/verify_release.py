"""Read actual local artifacts; fail closed. Does not generate robot evidence."""
import argparse
from collections import Counter
from pathlib import Path
import time
import xml.etree.ElementTree as ET
import numpy as np
from scipy.spatial.transform import Rotation
from ff_prior.storage import read_json, atomic_json, digest, file_digest
from ff_prior.pipeline import source_manifest
from ff_prior.spline import from_arrays
from ff_prior.contracts import assert_frozen, public_condition, MODEL_GATES, EXECUTION_GATES
from ff_prior.dataset import split_manifest, Scaler


def verify(run_dir, before_resume, tests):
    started = time.perf_counter()
    run_dir = Path(run_dir)
    run = read_json(run_dir/'run.json')
    ledger = read_json(run_dir/'ledger.json')
    summary = read_json(run_dir/'summary.json')
    assert source_manifest() == run['source_hashes']
    assert digest(run['source_hashes']) == run['identity']['source_hash']
    publication = read_json(run_dir/'publication.json')
    for name, expected in publication['files'].items():
        assert file_digest(run_dir/name) == expected, name
    rows = [read_json(p) for p in sorted((run_dir/'records').glob('*.json'))]
    assert len(rows) == 400 and len({r['sample_id'] for r in rows}) == 400
    valid = [r for r in rows if r['validation_level'] == 'MODEL_VALID']
    rejected = [r for r in rows if r['validation_level'] != 'MODEL_VALID']
    assert len(valid) == 399 and [r['sample_id'] for r in rejected] == ['dev_000083']
    assert rejected[0]['validation']['gates']['geometry_refinement'] is False
    array_refs, poses, base_change, amplitude, length = {}, [], [], [], []
    for row in valid:
        assert all(row['validation']['gates'][k] is True for k in MODEL_GATES)
        assert row['execution_status'] == 'NOT_RUN' and row['data_role'] == 'FF_MODEL_PRIOR'
        assert row['source_hash'] == run['identity']['source_hash']
        assert row['validation']['reference_torque_feasibility'] == 'NOT_EVALUATED'
        assert_frozen(row['task'])
        public_condition(row['condition'])
        with np.load(run_dir/row['arrays']) as a:
            ref = from_arrays(a)
            assert ref.reference_hash == row['reference_hash']
            for key, value in ref.arrays().items():
                np.testing.assert_array_equal(value, a[key])
            np.testing.assert_array_equal(a['prediction_ee'][-1], row['condition']['goals'])
            array_refs[row['sample_id']] = a['Z'].copy()
            poses.append(a['prediction_ee'][-1])
            base_change.append(np.linalg.norm(a['prediction_base'][:, :3]-a['prediction_base'][0, :3], axis=1).max())
            amplitude.append(np.abs(a['view_q']-a['view_q'][0]).max())
            length.append(np.linalg.norm(np.diff(a['view_q'], axis=0), axis=1).sum())
    split = read_json(run_dir/'splits.json')
    assert split_manifest(valid) == split
    dataset = read_json(run_dir/'dataset.json')
    assert {k:len(v) for k,v in dataset['eligible'].items()} == {'TRAIN':150,'VAL':30,'TEST':20,'DEV':199}
    by_id = {r['sample_id']:r for r in valid}
    ids = dataset['eligible']['TRAIN']
    fitted = Scaler.fit([array_refs[i] for i in ids], [by_id[i]['condition'] for i in ids],
        ['TRAIN']*len(ids), run['identity']['robot_hash'],run['identity']['initial_state_id'])
    saved = Scaler.load(run_dir/'learning/scaler.json',run['identity']['robot_hash'],run['identity']['initial_state_id'])
    for name in ('z_mean','z_scale','c_mean','c_scale'):
        np.testing.assert_allclose(getattr(fitted,name),getattr(saved,name),rtol=0,atol=1e-14)
    selection = read_json(run_dir/'audit_selection.json')
    selected = selection['random']+selection['risk']
    assert len(selection['random']) == 16 and len(selection['risk']) == 8
    assert len({by_id[i]['family_id'] for i in selected}) == 24
    assert selection['frozen_before_execution'] and not selection['replacement_after_failure']
    tasks = read_json(run_dir/'benchmark_tasks.json')
    assert len(tasks) == 4 and len({t['source_family'] for t in tasks}) == 4
    assert not set(selection['family_ids']) & {t['source_family'] for t in tasks}
    benchmark = []
    for i, entry in enumerate(tasks):
        assert_frozen(entry['task'])
        for method in summary['benchmark']['methods']:
            p = run_dir/'benchmark'/f'benchmark_{i:03d}_{method}'/'planning.json'
            result = read_json(p)
            assert result['task_hash'] == entry['task']['task_hash']
            assert result['public_conditions_hash'] == digest(entry['task']['condition'])
            assert result['selection_before_execution'] and not result['private_teacher_fields_accessed']
            assert len(result['candidates']) == 5
            eligible = [c for c in result['candidates'] if c['eligible']]
            expected = min(eligible,key=lambda c:(c['score'],c['candidate_id']))['candidate_id'] if eligible else None
            assert expected == result['selected_id']
            for candidate in result['candidates']:
                assert 'error' not in candidate
                sampling = candidate['sampling']
                assert sampling['actual_network_calls'] == (16 if method.startswith('diffusion') else 0)
                guidance = sampling['guidance']
                assert len(guidance) == (0 if method == 'diffusion_no_guidance' else 2)
                for step in guidance:
                    assert step['directions'] == 8 and step['centered_evaluations'] == 16
                    assert step['initial_cost_terms']['coupled_entire_path']
                    assert not step['production_eligible'] and not step['full_exact_gradient']
                if method == 'diffusion_in_denoising_guidance':
                    assert [s['iteration'] for s in guidance] == [14,15]
            benchmark.append(result)
    report_paths = sorted((run_dir/'audits').glob('*/report.json'))+sorted((run_dir/'benchmark').glob('*/execution/report.json'))
    assert len(report_paths) == 25
    reports, trace_hashes = [], {}
    for path in report_paths:
        report = read_json(path)
        assert report['level'] == 'EXECUTION_VERIFIED'
        assert all(report['gates'][g] is True for g in EXECUTION_GATES)
        assert report['replay']['passed'] and not report['replay']['copies_actual_state_each_step']
        assert max(report['replay']['maximum_errors'].values()) <= 1e-10
        assert report['metadata']['robot_state_writes_during_step_loop'] == 0
        assert report['minimum_clearance_m'] >= 0 and report['saturation_fraction'] == 0
        assert not report['training_eligible']
        task = read_json(path.parent/'frozen_task.json')
        assert_frozen(task)
        assert task['task_hash'] == report['task_hash']
        with np.load(path.parent/'native_trace.npz') as a:
            assert digest(a['ctrl']) == report['ctrl_hash']
            assert len(a['time']) == report['native_states'] == len(a['ctrl'])+1
            np.testing.assert_allclose(np.diff(a['time']),.002,atol=2e-12,rtol=0)
            assert np.max(np.linalg.norm(a['base'][:,:3]-a['base'][0,:3],axis=1)) > 0
        trace_hashes[str(path.parent.relative_to(run_dir))] = file_digest(path.parent/'native_trace.npz')
        reports.append(report)
    training = read_json(run_dir/'learning/training.json')
    assert training['status'] == 'PASS' and training['updates'] == 200
    assert training['weights_changed'] and training['reload_max_error'] == 0
    assert training['metadata']['data_quality_scope'] == 'REAL_MODEL_PRIOR'
    assert training['checkpoint_selection'] == 'VAL_MSE_ONLY'
    assert training['train_count'] == 150 and training['val_count'] == 30
    assert training['metadata']['dataset_hash'] == digest(dataset)
    before = read_json(before_resume)
    resume = {k:before[k] == ledger[k] for k in ('attempts','calls','completed')}
    assert all(resume.values()) and not ledger['inflight']
    assert not ledger.get('unresolved_call_accounting')
    for stage in ('main','dev'):
        assert ledger['attempts'][stage] == 200
        for call in ('rrt','sqp','torque_rollout','physics_steps'):
            assert ledger['calls'][f'{stage}.{call}'] == 0
    suites = ET.parse(tests).getroot().findall('testsuite')
    assert sum(int(s.attrib['tests']) for s in suites) == 43
    assert all(int(s.attrib[k]) == 0 for s in suites for k in ('failures','errors','skipped'))
    poses = np.asarray(poses)
    def quantiles(x):
        return dict(zip(('min','p25','median','p75','max'), np.quantile(x,[0,.25,.5,.75,1]).tolist()))
    rotations = np.stack([Rotation.from_matrix(poses[:,i,:3,:3]).magnitude() for i in range(2)],axis=1)
    coverage = {'scope':'399 finite MODEL_VALID proposals; correlated families, no uniform workspace claim',
        'dual_goal_position_min_m':poses[:,:,:3,3].min(axis=0).tolist(),
        'dual_goal_position_max_m':poses[:,:,:3,3].max(axis=0).tolist(),
        'dual_rotation_angle_from_world_identity_rad':[quantiles(rotations[:,i]) for i in range(2)],
        'maximum_joint_amplitude_rad':quantiles(amplitude),'q17_path_length_rad':quantiles(length),
        'maximum_predicted_base_translation_m':quantiles(base_change),
        'task_difficulty':{'benchmark_scope':'4 MODEL_DERIVED DEV tasks, empirical method outcomes only',
                           'outcomes':summary['benchmark']['methods']}}
    output = {'status':'PASS','run':str(run_dir),'source_hash':run['identity']['source_hash'],
        'published_files_verified':len(publication['files']),'model_valid':len(valid),
        'rejected':len(rejected),'native_trace_files_verified':len(reports),'trace_sha256':trace_hashes,
        'resume_no_duplicate_work':resume,'tests_passed':43,'coverage':coverage,
        'physical_extrema':{k:max(r[k] for r in reports) for k in ('tracking_max_rad','manifold_max_rad','maximum_base_translation_m')},
        'minimum_actual_clearance_m':min(r['minimum_clearance_m'] for r in reports),
        'maximum_hold_position_error_m':max(max(r['hold_max_position_error_m']) for r in reports),
        'maximum_hold_rotation_error_rad':max(max(r['hold_max_rotation_error_rad']) for r in reports),
        'network_calls_benchmark':sum(c['sampling']['actual_network_calls'] for r in benchmark for c in r['candidates']),
        'full_path_cost_evaluations':{m:sum(r['full_path_cost_evaluations'] for r in benchmark if r['method']==m) for m in summary['benchmark']['methods']},
        'wall_s':time.perf_counter()-started}
    atomic_json(run_dir/'release_verification.json',output)
    return output


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--run-dir',required=True)
    parser.add_argument('--before-resume',required=True)
    parser.add_argument('--tests',required=True)
    args=parser.parse_args()
    result=verify(args.run_dir,args.before_resume,args.tests)
    print({k:v for k,v in result.items() if k not in ('coverage','trace_sha256')})
