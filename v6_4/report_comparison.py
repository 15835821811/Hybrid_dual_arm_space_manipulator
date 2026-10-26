"""Build a grounded comparison from every frozen test cell, including failures."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from v6_4.contracts import TrajectoryProposal
from v6_4.run_planning import load_task_suite
from v6_4.task_protocol import TaskSpec
from v6_4.trajectory_codec import CubicBSplineCodec

ROOT = Path(__file__).resolve().parents[1]


def _read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _bool(value, name):
    if type(value) is not bool:
        raise ValueError(name + ' must be a JSON boolean')
    return value


def _path(value):
    path = Path(value)
    return (path if path.is_absolute() else ROOT / path).resolve()


def _finite(value, shape, name):
    array = np.asarray(value, dtype=float)
    if array.shape != shape or not np.all(np.isfinite(array)):
        raise ValueError(name + ' has invalid shape or nonfinite values')
    return array


def verify_manifest(directory, filename, *, required=(), inputs=None):
    """Verify every declared raw file, without importing its producer."""
    directory = Path(directory).resolve()
    manifest_path = directory / filename
    records = _read(manifest_path)
    if not isinstance(records, dict) or not records or not set(required) <= set(records):
        raise ValueError('missing required manifest entries: ' + str(manifest_path))
    if inputs is not None:
        inputs.append(manifest_path)
    for name, record in records.items():
        relative = Path(name.replace('\\', '/'))
        path = (directory / relative).resolve()
        if relative.is_absolute() or directory not in path.parents:
            raise ValueError('manifest path escapes artifact directory')
        expected = record if isinstance(record, str) else record['sha256']
        if _sha(path) != expected or (isinstance(record, dict) and path.stat().st_size != record['bytes']):
            raise ValueError('artifact raw identity changed: ' + str(path))
        if inputs is not None:
            inputs.append(path)
    return records


def validate_actual_trial(task, trial, method, *, expected_proposal_sha256=None, inputs=None):
    """Bind saved actual evidence. This performs no physics or geometry queries.

    Historical evaluator/source identities remain historical; they need not equal
    today's source. A missing/failed evaluation never supplies a successful label.
    """
    inputs = inputs if inputs is not None else []
    trial = _path(trial)
    result_path, task_path, identity_path = (trial / name for name in
        ('result.json', 'task.json', 'planning_identity.json'))
    inputs.extend((result_path, task_path, identity_path))
    result, saved_task, identity = _read(result_path), TaskSpec.from_dict(_read(task_path)), _read(identity_path)
    if (saved_task.sha256() != task.sha256() or result.get('task_id') != task.task_id
            or result.get('task_sha256') != task.sha256() or result.get('method') != method
            or identity.get('task_sha256') != task.sha256() or identity.get('method') != method):
        raise ValueError('actual trial task/method/initial declaration differs')
    accepted = _bool(result['proposal_accepted'], 'proposal_accepted')
    success = _bool(result['task_success'], 'task_success')
    complete = _bool(result['complete'], 'complete')
    fallback = _bool(result['fallback_used'], 'fallback_used')
    if fallback or result.get('postprocessing'):
        raise ValueError('repair/fallback is outside the raw comparison cohort')
    proposal = None
    controls = None
    if method == 'fixed_reference':
        if result['raw_proposal_passed'] is not None or expected_proposal_sha256 is not None:
            raise ValueError('fixed reference raw proposal metric must be N/A')
        raw_passed = None
    else:
        raw_passed = _bool(result['raw_proposal_passed'], 'raw_proposal_passed')
        proposal_path = trial / 'raw_proposal.json'
        inputs.append(proposal_path)
        proposal = TrajectoryProposal.from_dict(_read(proposal_path))
        if (proposal.task_id != task.task_id or proposal.task_sha256 != task.sha256()
                or proposal.origin != method or proposal.postprocessing
                or (expected_proposal_sha256 is not None and proposal.sha256() != expected_proposal_sha256)):
            raise ValueError('actual raw proposal does not bind the selected method/task/candidate')
        codec = CubicBSplineCodec(task.initial_planner_q, task.initial_planner_dq)
        controls = codec.decode_free(proposal.free_controls)
        raw_path, gate_path = trial / 'raw_proposal.npz', trial / 'proposal_gate.json'
        if raw_path.exists():
            inputs.append(raw_path)
            with np.load(raw_path, allow_pickle=False) as data:
                if not np.array_equal(data['controls_free'], proposal.free_controls) or not np.array_equal(data['control_points'], controls):
                    raise ValueError('actual raw control file differs from immutable proposal')
        elif accepted or raw_passed:
            raise ValueError('accepted proposal has no raw controls')
        if gate_path.exists():
            inputs.append(gate_path)
            gate = _read(gate_path)
            if (gate.get('task_id') != task.task_id or gate.get('task_sha256') != task.sha256()
                    or gate.get('origin') != method or gate.get('postprocessing')
                    or _bool(gate['raw_passed'], 'gate.raw_passed') != raw_passed
                    or (accepted and not _bool(gate['passed'], 'gate.passed'))):
                raise ValueError('actual proposal gate/source attribution differs')
        elif accepted or raw_passed:
            raise ValueError('accepted proposal has no gate evidence')
        if accepted and not raw_passed:
            raise ValueError('accepted comparison proposal did not pass the raw gate')
    if success and (not accepted or not complete):
        raise ValueError('successful task must have an accepted complete actual execution')
    trace_paths = list((trial / 'traces').glob('*.npz')) + list((trial / 'failures').glob('*partial_trace.npz'))
    counts = []
    for path in trace_paths:
        inputs.append(path)
        with np.load(path, allow_pickle=False) as data:
            counts.append(len(data['torque']))
    steps = max(counts, default=0)
    eval_path = trial / 'evaluation/report.json'
    evaluation = None
    valid = False
    if eval_path.exists():
        inputs.append(eval_path)
        evaluation = _read(eval_path)
        if (evaluation.get('task_id') != task.task_id or evaluation.get('task_sha256') != task.sha256()
                or evaluation.get('model_contract_sha256') != task.model_contract_sha256):
            raise ValueError('independent evaluation belongs to another task/model')
        for key in ('complete', 'evidence_valid', 'task_success'):
            _bool(evaluation[key], 'evaluation.' + key)
        verify_manifest(eval_path.parent, 'manifest.json', required=('report.json',), inputs=inputs)
        if evaluation.get('trace_path'):
            trace = _path(evaluation['trace_path'])
            if (trial not in trace.parents or not result.get('trace_path')
                    or _path(result['trace_path']) != trace or _sha(trace) != evaluation['trace_sha256']):
                raise ValueError('evaluation actual torque trace raw identity/path differs')
            inputs.append(trace)
            with np.load(trace, allow_pickle=False) as data:
                n = len(data['torque'])
                times = _finite(data['time'], (n,), 'actual time')
                _finite(data['torque'], (n, 67), 'actual torque')
                _finite(data['planner_q'], (n, 17), 'actual planner_q')
                if (n == 0 or n > 13500 or not np.allclose(times, .002*np.arange(1,n+1), atol=1e-9, rtol=0.)
                        or not np.array_equal(data['initial_qpos'], task.initial_qpos)
                        or not np.array_equal(data['initial_qvel'], task.initial_qvel)):
                    raise ValueError('actual trace initial state or complete simulation grid differs')
                steps = n
                if evaluation['complete'] != (n == 13500):
                    raise ValueError('evaluation complete flag contradicts actual torque horizon')
                if evaluation['evidence_valid']:
                    verify_manifest(eval_path.parent,'manifest.json',required=('report.json','fresh_replay.npz',
                        'interval_boundaries.jsonl','interval_rows.jsonl'),inputs=inputs)
                    if evaluation['metrics']['physics_steps'] != n:
                        raise ValueError('evaluation physics count contradicts actual trace')
                    fresh = eval_path.parent / 'fresh_replay.npz'
                    if _sha(fresh) != evaluation['fresh_replay_sha256']:
                        raise ValueError('independent fresh replay raw identity changed')
                    with np.load(fresh, allow_pickle=False) as state:
                        for key, shape in (('time',(n+1,)),('q',(n+1,17)),('qpos',(n+1,81)),('qvel',(n+1,79))):
                            _finite(state[key],shape,'fresh.'+key)
                        if (not np.allclose(state['time'], .002*np.arange(n+1),atol=1e-9,rtol=0.)
                                or not np.array_equal(state['qpos'][0], task.initial_qpos)
                                or not np.array_equal(state['qvel'][0], task.initial_qvel)
                                or not np.allclose(state['q'][1:],data['planner_q'],atol=1e-9,rtol=0.)):
                            raise ValueError('fresh replay does not bind the actual initial/grid/planner states')
                    interval = evaluation['independent_interval']
                    for file, key in (('interval_boundaries.jsonl','boundary_sha256'),('interval_rows.jsonl','rows_sha256')):
                        if _sha(eval_path.parent/file) != interval[key]:
                            raise ValueError('independent interval raw identity changed')
                    if method != 'fixed_reference':
                        reference = _path(evaluation['reference_path'])
                        if (trial not in reference.parents or _path(result['reference_path']) != reference
                                or _sha(reference) != evaluation['reference_sha256']):
                            raise ValueError('selected reference path/raw identity differs')
                        inputs.append(reference)
                        with np.load(reference,allow_pickle=False) as ref:
                            if not np.array_equal(ref['control_points'],controls):
                                raise ValueError('selected reference differs from actual raw proposal')
                        clock = _finite(data['generated_reference_time_s'],(n,),'generated reference clock')
                        if not np.allclose(clock,times,atol=1e-9,rtol=0.):
                            raise ValueError('actual reference sample time differs')
                        sampled = codec.sample(controls,np.minimum(27.,np.maximum(0.,clock)))
                        for key, field in (('q','generated_reference_q'),('dq','generated_reference_dq')):
                            if not np.allclose(_finite(data[field],(n,17),field),sampled[key],atol=1e-12,rtol=0.):
                                raise ValueError('actual consumed reference differs from selected proposal')
        elif evaluation['evidence_valid'] or evaluation['task_success']:
            raise ValueError('valid evaluation has no actual torque trace')
        valid = evaluation['evidence_valid']
        supported = evaluation['complete'] and valid and evaluation['task_success']
        if success != supported or complete != evaluation['complete']:
            raise ValueError('actual result flags differ from independent evaluation')
        if supported and not all(evaluation[key]['passed'] is True for key in
            ('task_requirements','execution_contract','independent_interval','native_geometry')):
            raise ValueError('claimed success contradicts independent functional/safety gates')
    elif success or complete:
        raise ValueError('complete/successful actual result has no independent evaluation')
    return {'result':result, 'identity':identity, 'evaluation':evaluation, 'proposal':proposal,
        'proposal_accepted':accepted, 'raw_proposal_passed':raw_passed, 'task_success':success,
        'complete':complete, 'evidence_valid':valid, 'fallback_used':fallback,
        'actual_physics_steps':steps, 'actual_trial_path':trial.as_posix(),
        'trace_sha256':evaluation.get('trace_sha256') if evaluation else None,
        'proposal_sha256':proposal.sha256() if proposal else None}


def validate_selection(task, directory, method, inputs):
    """Check all eight budget slots and reconstruct the common static choice."""
    from v6_4.run_teacher_comparison import SCORE_DEFINITION, static_score
    directory = Path(directory)
    manifest = verify_manifest(directory,'artifact_manifest.json',required=('selection.json','task.json','source_manifest.json'),inputs=inputs)
    selection = _read(directory/'selection.json')
    if (selection.get('task_id') != task.task_id or selection.get('task_sha256') != task.sha256()
            or TaskSpec.from_dict(_read(directory/'task.json')).sha256() != task.sha256()
            or selection['sources_unchanged'] is not True or selection['candidate_count'] != 8
            or selection.get('selection_uses_actual_outcomes') is not False
            or selection.get('score_definition') != SCORE_DEFINITION
            or selection.get('repair_used') is not False or selection.get('fallback_used') is not False
            or len(selection['records']) != 8):
        raise ValueError('invalid frozen all-eight-candidate selection evidence')
    if [r['candidate_index'] for r in selection['records']] != list(range(8)):
        raise ValueError('candidate slots must be unique ordered indices 0..7')
    codec = CubicBSplineCodec(task.initial_planner_q,task.initial_planner_dq)
    for record in selection['records']:
        candidate = directory / f"candidate_{record['candidate_index']:03d}"
        prefix = candidate.name + '/'
        if not {prefix+'record.json',prefix+'gate.json'} <= set(manifest):
            raise ValueError('candidate raw record/gate missing from artifact manifest')
        if _read(candidate/'record.json') != record:
            raise ValueError('selection candidate differs from its raw record')
        gate = _read(candidate/'gate.json')
        for key in ('raw_proposal_passed','gate_passed','eligible_for_selection'):
            _bool(record[key],key)
        if record['raw_proposal_passed'] != _bool(gate['raw_passed'],'gate.raw_passed') or record['gate_passed'] != _bool(gate['passed'],'gate.passed'):
            raise ValueError('candidate raw flags differ from saved gate')
        if record['proposal_sha256'] is None:
            if (record['raw_proposal_passed'] or record['gate_passed'] or record['eligible_for_selection']
                    or (candidate/'proposal.json').exists() or record['origin'] is not None):
                raise ValueError('missing proposal cannot pass or be selected')
            continue
        proposal = TrajectoryProposal.from_dict(_read(candidate/'proposal.json'))
        if not {prefix+'proposal.json',prefix+'control_points.npz'} <= set(manifest):
            raise ValueError('candidate raw proposal/controls missing from artifact manifest')
        if (proposal.sha256() != record['proposal_sha256'] or proposal.task_sha256 != task.sha256()
                or proposal.task_id != task.task_id or proposal.origin != record['origin']
                or list(proposal.postprocessing) != record['postprocessing']):
            raise ValueError('candidate proposal/source attribution differs')
        if proposal.origin != method or proposal.postprocessing:
            raise ValueError('repair/fallback candidate cannot be attributed to raw method')
        controls = codec.decode_free(proposal.free_controls)
        with np.load(candidate/'control_points.npz',allow_pickle=False) as data:
            if not np.array_equal(data['controls_free'],proposal.free_controls) or not np.array_equal(data['control_points'],controls):
                raise ValueError('candidate controls differ from raw proposal')
        prediction_path = candidate/'nominal_prediction.npz'
        if record['raw_proposal_passed'] or record['gate_passed'] or record['eligible_for_selection']:
            if not {prefix+'nominal_prediction.npz',prefix+'reference_identity.json'} <= set(manifest):
                raise ValueError('passing candidate prediction/source missing from artifact manifest')
            if (gate.get('task_id') != task.task_id or gate.get('task_sha256') != task.sha256()
                    or gate.get('origin') != method or gate.get('postprocessing')
                    or _sha(prediction_path) != record.get('nominal_prediction_sha256')):
                raise ValueError('passing candidate lacks bound nominal gate/prediction')
        if record['eligible_for_selection']:
            if not record['raw_proposal_passed'] or not record['gate_passed'] or record['failure']:
                raise ValueError('selected candidate is not an admissible raw proposal')
            with np.load(prediction_path,allow_pickle=False) as data:
                prediction = {k:data[k] for k in data.files}
                if not np.array_equal(prediction['control_points'],controls):
                    raise ValueError('nominal prediction controls differ from candidate')
                score, components = static_score(task,prediction)
            if record['score'] != score or record['score_components'] != components:
                raise ValueError('candidate common static score differs from raw prediction')
    eligible = [r for r in selection['records'] if r['eligible_for_selection']]
    selected = min(eligible,key=lambda r:(r['score'],r['candidate_index']))['candidate_index'] if eligible else None
    first = 0 if selection['records'][0]['eligible_for_selection'] else None
    if selection['selectedIndex'] != selected or selection['selectedIndexK1'] != first:
        raise ValueError('K1/K8 choice differs from common frozen static ranking')
    return selection


def summarize(rows, method, k, raw_records):
    selected = [r for r in rows if r['method'] == method and r['candidate_count'] == k]
    if len(selected) != 3 or len({r['task_id'] for r in selected}) != 3:
        raise ValueError('three distinct tasks required in every method/K denominator')
    accepted = sum(r['proposal_accepted'] is True for r in selected)
    success = sum(r['task_success'] is True for r in selected)
    return {'method': method, 'K': k, 'task_denominator': len(selected),
        'raw_candidate_denominator': len(raw_records) if raw_records is not None else None,
        'raw_candidate_pass_count': sum(r['raw_proposal_passed'] is True for r in raw_records) if raw_records is not None else None,
        'task_available_count': accepted, 'conditional_completed_count': success,
        'conditional_completion_rate': success / accepted if accepted else None,
        'total_completed_count': success, 'total_completion_rate': success / len(selected),
        'fallback_count': sum(bool(r.get('fallback_used', False)) for r in selected)}


def build(tasks_path, fixed_dir, teacher_dir, diffusion_dir, output):
    root = Path(__file__).resolve().parents[1]
    tasks = {t.task_id: t for t in load_task_suite(tasks_path) if t.split == 'test'}
    if len(tasks) != 3 or len({t.family for t in tasks.values()}) != 3:
        raise ValueError('report requires the complete frozen three-family test suite')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    rows, inputs, raw, selections, fixed_identities = [], [Path(tasks_path)], {}, {}, {}
    fixed_dir, teacher_dir, diffusion_dir = map(Path, (fixed_dir, teacher_dir, diffusion_dir))
    for task in tasks.values():
        directory = fixed_dir / task.task_id
        inputs.append(directory / 'result.json')
        result = _read(directory / 'result.json')
        if result.get('task_id') != task.task_id or result.get('task_sha256') != task.sha256() or result.get('method') != 'fixed_reference':
            raise ValueError('fixed baseline directory must bind its distinct declared test task')
        rows.append(dict(result, method='fixed_reference', candidate_count=1,
                         actual_trial_path=directory.resolve().as_posix()))
        fixed_identities[task.task_id] = _read(directory/'planning_identity.json')
    cohort_reports = {}
    for method, directory in (('teacher', teacher_dir), ('diffusion', diffusion_dir)):
        verify_manifest(directory,'artifact_manifest.json',required=('report.json','protocol.json'),inputs=inputs)
        inputs.append(directory / 'report.json')
        cohort_reports[method] = _read(directory / 'report.json')
        if not _bool(cohort_reports[method]['all_sources_unchanged'],'all_sources_unchanged'):
            raise ValueError('executed comparison changed its source freeze: ' + method)
        if method == 'diffusion' and not _bool(cohort_reports[method]['checkpoint_unchanged'],'checkpoint_unchanged'):
            raise ValueError('diffusion checkpoint changed during comparison')
        protocol = _read(directory/'protocol.json')
        if (protocol.get('task_hashes') != {t.task_id:t.sha256() for t in tasks.values()}
                or protocol.get('candidate_count',protocol.get('teacher_starts')) != 8 or protocol.get('repair_used') is not False
                or protocol.get('fallback_used') is not False):
            raise ValueError('comparison protocol differs from complete frozen test budget')
        if protocol.get('task_suite_sha256') != _sha(tasks_path):
            raise ValueError('comparison protocol belongs to another frozen task suite')
        if method == 'diffusion':
            checkpoint = _path(protocol['checkpoint_path'])
            if _sha(checkpoint) != protocol['checkpoint_sha256']:
                raise ValueError('declared diffusion checkpoint raw identity changed')
            inputs.append(checkpoint)
        cohort_rows = cohort_reports[method]['rows']
        expected = {(task_id, k) for task_id in tasks for k in (1, 8)}
        actual = [(r['task_id'], r['candidate_count']) for r in cohort_rows]
        if len(actual) != 6 or set(actual) != expected or any(type(r['candidate_count']) is not int for r in cohort_rows):
            raise ValueError('missing or duplicate comparison cell: ' + method)
        rows.extend(dict(r, method=method) for r in cohort_rows)
        for k in (1, 8):
            raw[method, k] = []
        for task in tasks.values():
            path = directory / task.task_id / 'screening/selection.json'
            inputs.append(path)
            selection = validate_selection(task,path.parent,method,inputs)
            screening_sources = _read(path.parent/'source_manifest.json')
            protocol_sources = {k.replace('\\','/'):v for k,v in protocol['source_sha256'].items()}
            if any(protocol_sources.get(k.replace('\\','/')) != v for k,v in screening_sources.items()):
                raise ValueError('candidate screening source identity differs from declared cohort freeze')
            selections[method,task.task_id] = selection
            raw[method, 1].append(selection['records'][0])
            raw[method, 8].extend(selection['records'])
    details = []
    for row in rows:
        task = tasks[row['task_id']]
        if row['task_sha256'] != task.sha256():
            raise ValueError('comparison changes an immutable test task')
        detail = {key: row.get(key) for key in ('method', 'candidate_count', 'task_id', 'status',
            'proposal_accepted', 'raw_proposal_passed', 'task_success', 'complete', 'new_physics_steps',
            'reused_actual_trial', 'actual_trial_path')}
        for flag in ('proposal_accepted','task_success','complete'):
            _bool(row[flag], 'comparison.'+flag)
        if row['method'] != 'fixed_reference':
            _bool(row['raw_proposal_passed'],'comparison.raw_proposal_passed')
            selection = selections[row['method'],task.task_id]
            selected = selection['selectedIndexK1' if row['candidate_count']==1 else 'selectedIndex']
            if row.get('selectedIndex') != selected:
                raise ValueError('actual cell did not use its common static selection')
            expected_hash = selection['records'][selected]['proposal_sha256'] if selected is not None else None
            if row.get('proposal_sha256') != expected_hash:
                raise ValueError('actual cell proposal SHA differs from statically selected candidate')
            if selected is None and (row['proposal_accepted'] or row['raw_proposal_passed'] or row['task_success'] or row['complete'] or row.get('actual_trial_path')):
                raise ValueError('absent selected proposal cannot acquire actual success/acceptance')
        else:
            expected_hash = None
        detail.update(family=task.family, fallback_used=False, shape_vector_dimension=17)
        detail['actual_physics_steps'] = 0
        if row['task_success'] and not row.get('actual_trial_path'):
            raise ValueError('claimed success has no actual trial')
        if row.get('actual_trial_path'):
            trial = Path(row['actual_trial_path'])
            if not trial.is_absolute():
                trial = root / trial
            verified = validate_actual_trial(task,trial,row['method'],expected_proposal_sha256=expected_hash,inputs=inputs)
            if any(verified['identity'].get(key) != fixed_identities[task.task_id].get(key)
                    for key in ('qp_config','run_config')):
                raise ValueError('actual comparison changed the shared control/safety/simulation configuration')
            for key in ('proposal_accepted','raw_proposal_passed','task_success','complete'):
                if row[key] != verified[key]:
                    raise ValueError('comparison row flags differ from bound actual evidence: '+key)
                detail[key] = verified[key]
            detail['actual_trial_path'] = verified['actual_trial_path']
            detail['actual_physics_steps'] = verified['actual_physics_steps']
            detail['fallback_used'] = verified['fallback_used']
            evaluation = verified['evaluation']
            if evaluation and verified['evidence_valid']:
                detail.update(metrics=evaluation['metrics'], native_geometry=evaluation['native_geometry'],
                    independent_interval=evaluation['independent_interval'], execution_contract=evaluation['execution_contract'],
                    task_requirements=evaluation['task_requirements'], rejection_count=evaluation['rejection_count'],
                    evidence_valid=evaluation['evidence_valid'], evaluation_errors=evaluation['errors'])
                detail['actual_physics_steps'] = int(evaluation['metrics']['physics_steps'])
                shape = evaluation['metrics'].get('actual_to_planned_shape_error_rad', {}).get('rms')
                detail['shape_17d_L2_rms_rad'] = shape
                detail['shape_per_coordinate_rms_rad'] = shape / math.sqrt(17) if shape is not None else None
            if row['method'] != 'fixed_reference':
                reused = _bool(row['reused_actual_trial'],'reused_actual_trial')
                if reused:
                    lineage_dir = (teacher_dir if row['method']=='teacher' else diffusion_dir)/task.task_id/f"K{row['candidate_count']}"
                    lineage = _read(lineage_dir/'reuse_lineage.json')
                    inputs.append(lineage_dir/'reuse_lineage.json')
                    if (_path(lineage['original_trial']) != trial.resolve() or lineage['proposal_sha256'] != expected_hash
                            or lineage['original_task_sha256'] != task.sha256() or lineage['original_method'] != row['method']
                            or lineage['new_physics_steps'] != 0 or row['new_physics_steps'] != 0):
                        raise ValueError('reused actual evidence has invalid lineage')
                    original_names = {p.relative_to(trial).as_posix() for p in trial.rglob('*') if p.is_file()}
                    if set(lineage['original_files']) != original_names:
                        raise ValueError('reuse lineage must bind every original actual artifact')
                    for name, record in lineage['original_files'].items():
                        source = trial/name
                        if _sha(source) != record['sha256'] or source.stat().st_size != record['bytes']:
                            raise ValueError('reused actual original file changed')
                        inputs.append(source)
                elif row['new_physics_steps'] != verified['actual_physics_steps']:
                    raise ValueError('actual cell physics count differs from saved force evidence')
        elif row['proposal_accepted'] or row['complete'] or row.get('new_physics_steps'):
            raise ValueError('actual acceptance/completion/physics count has no bound trial path')
        details.append(detail)
    aggregates = [summarize(details, 'fixed_reference', 1, None)]
    aggregates.extend(summarize(details, method, k, raw[method, k]) for k in (1, 8) for method in ('teacher', 'diffusion'))
    report = {'schema': 'v6_4_grounded_three_method_comparison_v1', 'test_task_count': 3,
        'aggregates': aggregates, 'details': details,
        'fixed_reference_raw_proposal_rate': 'NOT_APPLICABLE: existing Cartesian reference has no learned joint proposal',
        'repair': 'NOT_RUN', 'fallback': 'NOT_RUN',
        'independent_diffusion_actual_execution_count': len({r['actual_trial_path'] for r in details
            if r['method'] == 'diffusion' and r.get('actual_trial_path') and r['actual_physics_steps'] > 0}),
        'diffusion_actual_execution_cell_count': sum(r['method'] == 'diffusion'
            and r['actual_physics_steps'] > 0 for r in details),
        'independent_diffusion_complete_evidence_count': len({r['actual_trial_path'] for r in details
            if r['method'] == 'diffusion' and r.get('actual_trial_path') and r['complete'] and r.get('evidence_valid')}),
        'all_failures_in_task_denominators': True, 'wall_20ms_is_acceptance_gate': False,
        'scope': 'three frozen independent tasks; no hard real time, hardware, robustness, continuous CCD, or broad generalization claim',
        'inputs': [{'path': p.resolve().as_posix(), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}
                   for p in sorted(set(inputs))]}
    (output / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    lines = ['# V6.4-A 三组规划比较', '',
        '固定的三个测试任务使用相同名义机器人、20ms/2ms仿真周期、任务判据与安全执行链。所有规划拒绝均进入总任务分母。', '',
        '| 方法 | K | 原始候选通过 | 可执行任务 | 条件完成 | 总任务完成 |',
        '|---|---:|---:|---:|---:|---:|']
    for a in aggregates:
        rate = 'N/A' if a['raw_candidate_denominator'] is None else f"{a['raw_candidate_pass_count']}/{a['raw_candidate_denominator']}"
        conditional = 'N/A' if not a['task_available_count'] else f"{a['conditional_completed_count']}/{a['task_available_count']}"
        lines.append(f"| {a['method']} | {a['K']} | {rate} | {a['task_available_count']}/{a['task_denominator']} | {conditional} | {a['total_completed_count']}/{a['task_denominator']} |")
    lines.extend(['', '固定参考的原始提案率不适用；其可用性指原Cartesian参考可直接进入相同力矩执行链。K1是候选0，K8使用公开的共同静态路径/基座代价选择，实际执行结果不参与选择。相同提案的K1/K8实际轨迹可以复用，完整来源保存在原件中。', '',
        '所有原始候选、失败、完整实际命令与力矩、独立重放及来源SHA由对应试验目录保存。形状误差的主指标是17维向量L2的RMS，不能当作每关节RMS；报告同时提供除以√17的坐标RMS。', '',
        '本表未运行repair或fallback。墙钟超20ms仍记录为性能诊断，当前结论限于非实时仿真研究。三项测试不支持广泛泛化、硬实时、硬件、模型误差鲁棒性或连续时间几何安全声明。', '',
        '完整统计、逐任务安全/区间/形状/控制干预/基座/力矩与耗时见同目录report.json。'])
    (output / 'REPORT.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    manifest = {p.name: {'sha256': hashlib.sha256(p.read_bytes()).hexdigest(), 'bytes': p.stat().st_size}
                for p in output.iterdir() if p.is_file()}
    (output / 'artifact_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n', encoding='utf-8')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ('tasks', 'fixed', 'teacher', 'diffusion', 'output'):
        parser.add_argument('--'+option, type=Path, required=True)
    args = parser.parse_args()
    result = build(args.tasks, args.fixed, args.teacher, args.diffusion, args.output)
    print(json.dumps({'aggregates': result['aggregates'], 'diffusion_actual_cells': result['independent_diffusion_actual_execution_count']}), flush=True)
