"""Independent saved-media inventory and CSV arithmetic checks; no simulation."""
import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np


def read(p):
    return json.loads(p.read_text(encoding='utf-8-sig'))


def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda: f.read(4*1024*1024), b''):
            h.update(b)
    return h.hexdigest()


def check(ok, message):
    if not ok:
        raise ValueError(message)


def audit(run, media, record):
    # Loads immutable Task/reference and saved arrays only. No model or executor.
    from v6_4.visualization.build_search_aware_warmstart_media import load_actual
    data, manifest = read(media/'dashboard_data.json'), read(media/'visualization_manifest.json')
    checks = read(media/'input_checks.json')
    views = {'overview', 'front', 'side', 'top', 'iso', 'five_view_grid', 'continuum_focus'}
    check(len(data['tasks']) == 4 and set(data['endpoints']) == {'R8','R12','N8','S8','D8'}, 'task/endpoint scope')
    check(len(data['slots']) == 40 and len({(s['task_id'],s['method']) for s in data['slots']}) == 40, 'logical identities')
    check(set(data['views']) == views, 'view set')
    unique = [s for s in data['slots'] if not s.get('alias_of_slot') and s['actual_steps'] > 0]
    check(checks['status'] == 'PASS' and checks['unique_saved_actuals'] == len(unique) == 28, 'unique actuals')
    check(manifest['logical_actual_slots'] == 40 and manifest['unique_actual_media'] == 28 and manifest['video_count'] == 196, 'media counts')
    for name in ('physics_steps_executed', 'geometry_queries', 'QP_solves', 'optimizer_updates', 'new_model_samples'):
        check(manifest[name] == 0, name)
    check(manifest['historical_outputs_overwritten'] is False and manifest['visualization_is_new_acceptance'] is False, 'display scope')
    current = {p.relative_to(media).as_posix() for p in media.rglob('*') if p.is_file() and p.name != 'visualization_manifest.json'}
    check(set(manifest['files']) == current, 'inventory exhaustive')
    for relative, bound in manifest['files'].items():
        p = (media/relative).resolve()
        check(p.is_relative_to(media) and p.stat().st_size == bound['bytes'] and sha(p) == bound['sha256'], relative)
    for name in ('budget_coverage_near_quality.png', 'actual_quality_planning_cost.png'):
        check(data['charts'][name] == 'charts/'+name and sha(media/'charts'/name) == sha(run/'result_tables'/name), name)
    by_key = {(s['task_id'],s['method']):s for s in data['slots']}
    for s in data['slots']:
        if s['actual_steps'] == 0:
            check(s['status'] == 'NO_PLAN' and s.get('media') is None and s.get('figures') is None, 'no fabricated NO_PLAN asset')
            continue
        meta = s['media']
        check(set(meta['videos']) == views and set(meta['video_records']) == views, 'all views')
        check(meta['includes_exact_saved_endpoint'] is True and meta['selected_saved_indices'][-1] == s['actual_steps'], 'exact endpoint')
        check(abs(meta['displayed_saved_end_s']-meta['source_end_s']) <= 1e-9 and abs(meta['source_end_s']-s['actual_steps']*.002) <= 1e-9, 'end clock')
        check(meta['frame_count'] == len(meta['selected_saved_indices']) == len(meta['selected_saved_times_s']), 'frame arrays')
        check(abs(meta['encoded_duration_s']-meta['frame_count']/meta['fps']) <= 1e-9, 'encoded clock')
        for k in ('physics_steps','geometry_queries','QP_solves'):
            check(meta['render_cost'][k] == 0, 'render-only cost')
        check(not meta['render_cost']['forbidden_call_attempts'], 'render guard')
        if s.get('alias_of_slot'):
            original = by_key[(s['task_id'],s['media_source_method'])]
            check(not original.get('alias_of_slot') and s['media'] == original['media'] and s['figures'] == original['figures'], 'strict alias reuse')
    check(data['summary'] == read(run/'summary.json'), 'summary byte content')
    header = 'physical_time_s,continuum_generated_position_error_m,continuum_Task_base_position_error_m,rigid_current_Task_position_error_m,continuum_orientation_error_rad,rigid_orientation_error_rad'
    for s in unique:
        task, _, _, _, replay, reference, base, parity, sources = load_actual(run,s)
        csv_path = media/s['figures']['tracking_csv']
        with csv_path.open(encoding='utf-8') as stream:
            check(stream.readline().strip() == header, 'CSV units/schema')
        observed = np.loadtxt(csv_path, delimiter=',', skiprows=1)
        rigid_target = replay['target_position'] + (replay['target_rotation'] @ np.asarray(task.scenario['grasp_point_target_frame_m']))
        target_c = np.asarray(task.scenario['continuum_target_rotation_world'])
        target_r = replay['target_rotation'] @ np.asarray(task.scenario['grasp_rotation_target_frame'])
        def angle(actual, target):
            relative = np.swapaxes(actual,1,2) @ target
            return np.arccos(np.clip((np.trace(relative,axis1=1,axis2=2)-1)/2,-1,1))
        expected = np.column_stack((replay['time'], np.linalg.norm(replay['continuum_position']-reference,axis=1),
                 np.linalg.norm(replay['continuum_position']-base,axis=1), np.linalg.norm(replay['rigid_position']-rigid_target,axis=1),
                 angle(replay['continuum_rotation'],target_c),angle(replay['rigid_rotation'],target_r)))
        check(observed.shape == expected.shape == (s['actual_steps']+1,6), 'CSV shape')
        check(np.isfinite(observed).all(), 'CSV finite')
        differences = np.max(np.abs(observed-expected),axis=0)
        # arccos is ill-conditioned near zero; retain angle agreement with a
        # 1e-7 rad arithmetic tolerance (no scientific acceptance threshold edit).
        tolerance = np.array([1e-12,1e-12,1e-12,1e-12,1e-7,1e-7])
        check(np.all(differences <= tolerance), 'CSV arithmetic '+s['task_id']+' '+s['method'])
        check(np.max(np.abs(observed[:,0]-np.arange(s['actual_steps']+1)*.002)) <= 1e-9, 'CSV native time')
        record['csv_checks'].append(dict(task=s['task_id'], method=s['method'], rows=len(observed),
             csv=s['figures']['tracking_csv'],sha256=sha(csv_path),max_abs_difference=differences.tolist(),
             tolerance=tolerance.tolist(), source_binding=sources, parity=parity))
    record.update(manifest_sha256=sha(media/'visualization_manifest.json'), files_verified=len(current),
                  logical_slots=40, unique_saved_actuals=28, alias_slots=6, no_plan_slots=6, video_count=196,
                  status='PASS', visual_review='SEPARATE_REQUIRED')


def main():
    p=argparse.ArgumentParser(); p.add_argument('--run',type=Path,required=True); p.add_argument('--media',type=Path,required=True); p.add_argument('--output',type=Path,required=True)
    a=p.parse_args(); check(not a.output.exists(),'new output required')
    a.output.parent.mkdir(parents=True,exist_ok=True)
    record=dict(schema='c3_media_numeric_audit_v1',started_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
                script_sha256=sha(Path(__file__)),argv=[sys.executable,*sys.argv],status='FAIL',csv_checks=[],
                new_physics_steps=0,new_model_samples=0,new_optimizer_updates=0)
    start=time.perf_counter()
    try:
        audit(a.run.resolve(),a.media.resolve(),record)
    except Exception as exc:
        record['error']=repr(exc)
    record.update(ended_utc=dt.datetime.now(dt.timezone.utc).isoformat(),elapsed_s=time.perf_counter()-start)
    with a.output.open('x',encoding='utf-8') as stream:
        json.dump(record,stream,indent=2,ensure_ascii=False)
    print(json.dumps({k:v for k,v in record.items() if k != 'csv_checks'},ensure_ascii=False))
    return 0 if record['status']=='PASS' else 1


if __name__=='__main__':
    sys.exit(main())
