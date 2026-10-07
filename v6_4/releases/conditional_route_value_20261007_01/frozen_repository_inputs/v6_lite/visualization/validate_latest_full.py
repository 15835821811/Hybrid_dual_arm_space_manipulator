"""Independently validate the latest visualization bundle and its frozen inputs."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import json
from pathlib import Path, PureWindowsPath
import re
import shutil
import subprocess
import sys

import numpy as np
from PIL import Image

ROOT = next(p for p in Path(__file__).resolve().parents if (p / '.git').exists() and (p / 'model_test').is_dir())
DEFAULT_OUTPUT = ROOT / 'v6_lite/visualization/latest'
SCENES = {f'v6_lite_scenario_{i:02d}' for i in range(5)}
ROLES = {'overview', 'front', 'side', 'top', 'iso', 'grid', 'focus'}
CATEGORIES = {'nominal', 'conservatism', 'shape', 'velocity', 'inertia', 'wall'}
SOURCE_PINS = {
    'acceptance': ('v6_lite/output/runs/research_acceptance_01/manifest.json',
                   '379b9977a441c540a20cfc50a9e90614a1e0985fa2a5a73bbb7c9690f823eec0',
                   '9cd1831e7c77a340464a669eac3b5aa7cd9c3c3c', 46),
    'inertia': ('v6_lite/output/runs/research_inertia_shadow_resumed_01/manifest.json',
                'ee98249e6ba936278e005943dc942724e57fb5ef2f59d2fe226b0e101d89158e',
                '19ad85c7db81a2e254fd176077b6aae39333a439', 146),
}
VELOCITY_MANIFEST_SHA = '1da995a2b611635db5dd05129dceedc9adf27983e28aa8edae08eec6f846dda0'
AUDIT_OUTPUTS = {'visualization_validation.json', 'validation_contact_sheet.png'}


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path):
    return json.loads(path.read_text(encoding='utf-8'))


def _safe_path(value: str, base: Path = None) -> Path:
    """Records are repository relative, including records written on Windows."""
    if not isinstance(value, str) or not value or '\x00' in value:
        raise ValueError('invalid empty/non-string record path')
    windows = PureWindowsPath(value)
    normalized = value.replace('\\', '/')
    if windows.drive or windows.root or normalized.startswith('/') or '..' in normalized.split('/'):
        raise ValueError('absolute or traversing record path rejected')
    path = (ROOT / normalized).resolve()
    if not path.is_relative_to(ROOT.resolve()):
        raise ValueError('record path escaped repository')
    if base is not None and not path.is_relative_to(base.resolve()):
        raise ValueError('artifact escaped current visualization directory')
    return path


def _record_key(value):
    return _safe_path(value).relative_to(ROOT).as_posix()


def _verify_records(records, check, prefix, base=None):
    if not isinstance(records, list):
        raise ValueError(prefix + ' must be a list of records')
    result = {}
    for record in records:
        if not isinstance(record, dict):
            raise ValueError(prefix + ' record must be an object')
        path = _safe_path(record['path'], base)
        key = path.relative_to(ROOT).as_posix()
        if key in result:
            raise ValueError(prefix + ' has duplicate normalized path')
        expected = record['sha256']
        size = record['bytes']
        if not isinstance(expected, str) or not re.fullmatch(r'[a-f0-9]{64}', expected):
            raise ValueError(prefix + ' record has invalid SHA256')
        if not isinstance(size, int) or isinstance(size, bool) or size < 0:
            raise ValueError(prefix + ' record has invalid byte count')
        check(prefix + ':' + key, path.is_file() and path.stat().st_size == size and _sha(path) == expected)
        result[key] = record
    return result


def _verify_source_pins(manifest, source_records, check):
    verified = {}
    for name, (filename, expected_sha, commit, count) in SOURCE_PINS.items():
        declaration = manifest['inputs'][name]
        record = declaration['manifest_record']
        path = _safe_path(record['path'])
        key = path.relative_to(ROOT).as_posix()
        check('source_pin:' + name, key == filename and _sha(path) == record['sha256'] == expected_sha
              and path.stat().st_size == record['bytes'] and declaration['source_commit'] == commit
              and declaration['verified_file_count'] == count and key in source_records)
        raw = _json(path)
        normalized = {}
        for raw_key, expected_value in raw.items():
            expected = expected_value if isinstance(expected_value, str) else expected_value['sha256']
            relative = raw_key.replace('\\', '/')
            if PureWindowsPath(raw_key).drive or relative.startswith('/') or '..' in relative.split('/'):
                raise ValueError('source manifest contains unsafe path')
            if relative in normalized:
                raise ValueError('source manifest contains duplicate normalized key')
            normalized[relative] = expected
            file = (path.parent / relative).resolve()
            if not file.is_relative_to(path.parent):
                raise ValueError('source manifest escaped its original run')
            source_key = file.relative_to(ROOT).as_posix()
            check(name + ':raw_source:' + relative, file.is_file() and _sha(file) == expected
                  and source_key in source_records and source_records[source_key]['sha256'] == expected)
        check(name + ':exact_source_manifest_count', len(normalized) == count)
        verified[name] = normalized
    return verified


def _ratio(value, total):
    return value == f'{total}/{total}' or (isinstance(value, dict)
           and value.get('passed') == total and not isinstance(value.get('passed'), bool)
           and value.get('total') == total)


def _verify_claims(manifest, check):
    coverage = manifest['coverage']
    check('declared_complete_scene_duration_and_views', coverage['nominal_scenarios'] == 5
          and coverage['nominal_duration_s'] == 27 and coverage['video_count'] == 35
          and set(coverage['views']) == ROLES - {'grid', 'focus'} and len(coverage['views']) == 5)
    check('six_research_categories', set(coverage['research_categories']) == CATEGORIES
          and len(coverage['research_categories']) == 6)
    check('declared_latest_scope_is_complete_saved_evidence', manifest['complete'] is True
          and manifest['physics_recomputed'] is False and coverage['all_pressure_scenarios_included'] is True
          and coverage['all_inertia_runs_included'] == 15 and coverage['old_b2_used_as_current_source'] is False)
    claims = manifest['claim_scope']
    check('original_acceptance_and_deployment_claims', claims['research_acceptance'] == 'PASSED'
          and _ratio(claims['functional_checks'], 25) and _ratio(claims['execution_checks'], 11)
          and claims['wall_deployment'] == 'NOT_MET' and claims['hardware'] == 'NOT_ESTABLISHED'
          and claims['shadow_closed_loop_robustness'] is False
          and claims['velocity_stress'] == 'FAILED_INCOMPLETE'
          and claims['hard_real_time'] is False and claims['continuous_time_collision_guarantee'] is False
          and claims['visualization_is_new_functional_acceptance'] is False)


def _verify_failed_stress(source_records, check):
    manifest_key = 'v6_lite/output/runs/research_velocity_stress_01/manifest.json'
    path = _safe_path(manifest_key)
    check('velocity_original_manifest_pin', manifest_key in source_records
          and _sha(path) == VELOCITY_MANIFEST_SHA == source_records[manifest_key]['sha256'])
    raw_manifest = _json(path)
    normalized = set()
    for name, expected_value in raw_manifest.items():
        relative = name.replace('\\', '/')
        if PureWindowsPath(name).drive or relative.startswith('/') or '..' in relative.split('/'):
            raise ValueError('velocity original manifest contains unsafe path')
        if relative in normalized:
            raise ValueError('velocity original manifest contains duplicate normalized key')
        normalized.add(relative)
        file = (path.parent / relative).resolve()
        if not file.is_relative_to(path.parent):
            raise ValueError('velocity source escaped original run')
        expected = expected_value if isinstance(expected_value, str) else expected_value['sha256']
        key = file.relative_to(ROOT).as_posix()
        check('velocity:registered_raw_source:' + relative, file.is_file() and _sha(file) == expected
              and key in source_records and source_records[key]['sha256'] == expected)
    check('velocity_exact_34_source_files', len(normalized) == 34)
    filename = 'v6_lite/output/runs/research_velocity_stress_01/report.json'
    check('failed_stress_is_explicit_source', filename in source_records)
    report = _json(_safe_path(filename))
    check('failed_stress_remains_incomplete', report['passed'] is False and report['complete'] is False
          and report['delay_or_model_error_robustness_established'] is False)
    trial = report['trial_summary']
    rows = trial['scenes']
    completed = {r['scenario_id'] for r in rows if r['status'] == 'COMPLETED_PASS'}
    rejected = {r['scenario_id'] for r in rows if r['status'] == 'REJECTED_PARTIAL'}
    check('stress_exact_five_attempts_two_complete_three_rejected', len(rows) == 5
          and {r['scenario_id'] for r in rows} == SCENES
          and completed == {'v6_lite_scenario_01', 'v6_lite_scenario_02'}
          and rejected == {'v6_lite_scenario_00', 'v6_lite_scenario_03', 'v6_lite_scenario_04'}
          and trial['all_five_attempts_observed'] and trial['reported_completed_scene_count'] == 2
          and trial['reported_rejected_scene_count'] == 3 and trial['partial_is_full_acceptance'] is False)
    for scene in sorted(SCENES):
        relative = (f'simulation/traces/{scene}.npz' if scene in completed
                    else f'simulation/failures/{scene}_partial_trace.npz')
        key = 'v6_lite/output/runs/research_velocity_stress_01/' + relative
        path = _safe_path(key)
        check('stress_saved_trace:' + scene, path.is_file() and key in source_records)
        with np.load(path, allow_pickle=False) as arrays:
            times = arrays['time']
            check('stress_unextrapolated_saved_horizon:' + scene,
                  times.ndim == 1 and len(times) > 0 and np.all(np.isfinite(times))
                  and np.all(np.diff(times) > 0) and np.allclose(times, np.arange(1, len(times) + 1) * .002, atol=1e-9, rtol=0.)
                  and (len(times) == 13500 if scene in completed else len(times) < 13500))


def _probe_video(path):
    program = shutil.which('ffprobe')
    if not program:
        raise RuntimeError('ffprobe unavailable')
    run = subprocess.run([program, '-v', 'error', '-select_streams', 'v:0', '-show_entries',
                          'stream=width,height,nb_frames,avg_frame_rate,duration:format=duration',
                          '-of', 'json', str(path)], capture_output=True, check=True, timeout=30)
    data = json.loads(run.stdout)
    stream = data['streams'][0]
    return {'width': int(stream['width']), 'height': int(stream['height']),
            'frame_count': int(stream['nb_frames']), 'fps': float(Fraction(stream['avg_frame_rate'])),
            'duration_s': float(stream.get('duration', data['format']['duration']))}


def _decode_frames(path, frame_count):
    program = shutil.which('ffmpeg')
    if not program:
        raise RuntimeError('ffmpeg unavailable')
    indices = [0, frame_count // 2, frame_count - 1]
    expression = '+'.join(f'eq(n\\,{i})' for i in indices)
    command = [program, '-v', 'error', '-threads', '1', '-i', str(path), '-filter_threads', '1',
               '-vf', f'select={expression},scale=320:180', '-vsync', '0', '-frames:v', '3',
               '-f', 'rawvideo', '-pix_fmt', 'rgb24', 'pipe:1']
    run = subprocess.run(command, capture_output=True, check=True, timeout=180)
    expected = 3 * 320 * 180 * 3
    if len(run.stdout) != expected:
        raise ValueError('did not decode all first/middle/final representative frames')
    return np.frombuffer(run.stdout, np.uint8).reshape(3, 180, 320, 3)


def _content_metrics(frames):
    values = np.asarray(frames, dtype=np.float64)
    center = values[:, values.shape[1] // 10:values.shape[1] * 9 // 10,
                    values.shape[2] // 10:values.shape[2] * 9 // 10]
    std = np.std(center, axis=(1, 2, 3))
    colors = [len(np.unique(f.reshape(-1, 3), axis=0)) for f in np.asarray(frames)]
    diffs = [np.abs(values[j] - values[0]) for j in range(1, len(values))]
    return {'central_std': std.tolist(), 'color_counts': colors,
            'maximum_motion_mean_8bit': float(max((d.mean() for d in diffs), default=0.)),
            'maximum_motion_pixel_fraction': float(max(((d.max(axis=2) > 5).mean() for d in diffs), default=0.))}


def _nonblank(metrics):
    return all(x > 1. for x in metrics['central_std']) and all(x > 16 for x in metrics['color_counts'])


def _moving(metrics):
    return metrics['maximum_motion_mean_8bit'] > .02 and metrics['maximum_motion_pixel_fraction'] > .0001


def _video_role(path):
    hits = [role for role in ROLES if re.search(r'(^|[_-])' + role + r'([_-]|$)', path.stem.lower())]
    if len(hits) != 1:
        raise ValueError('video filename does not identify one declared view/grid/focus role')
    return hits[0]


def _verify_nominal(manifest, artifacts, sources, output, check):
    rows = manifest['nominal']['videos']
    check('five_exact_unique_nominal_metadata_rows', len(rows) == 5
          and {r['scenario_id'] for r in rows} == SCENES)
    all_videos, results = [], []
    for row in rows:
        scene = row['scenario_id']
        index = int(scene.rsplit('_', 1)[1])
        source = _safe_path(row['source_trace'])
        expected = ROOT / f'v6_lite/output/runs/research_acceptance_01/simulation/traces/v6_lite_scenario_{index:02d}.npz'
        source_key = source.relative_to(ROOT).as_posix()
        check(scene + ':original_nominal_trace', source == expected and source_key in sources
              and _sha(source) == row['source_trace_sha256'] == sources[source_key]['sha256'])
        check(scene + ':declared_full_source_time_grid', row['frame_count'] == 811
              and row['fps'] == 30 and np.isfinite(row['duration_s'])
              and (row['duration_s'] == 27 or abs(row['duration_s'] - 811 / 30) < .001))
        files = [_safe_path(x, output) for x in row['files']]
        check(scene + ':all_metadata_files_hash_registered', all(p.relative_to(ROOT).as_posix() in artifacts for p in files))
        videos = [p for p in files if p.suffix.lower() == '.mp4']
        roles = [_video_role(p) for p in videos]
        check(scene + ':seven_complete_unique_views', len(videos) == 7 and len(set(videos)) == 7 and set(roles) == ROLES)
        all_videos.extend(videos)
        for path, role in zip(videos, roles):
            info = _probe_video(path)
            check(scene + ':' + role + ':probe_full_video', info['frame_count'] == 811
                  and abs(info['fps'] - 30) < 1e-9 and info['width'] >= 160 and info['height'] >= 120
                  and np.isfinite(info['duration_s']) and abs(info['duration_s'] - 811 / 30) < .004)
            frames = _decode_frames(path, 811)
            metrics = _content_metrics(frames)
            check(scene + ':' + role + ':decoded_nonblank_motion', _nonblank(metrics) and _moving(metrics))
            results.append({'scenario_id': scene, 'role': role, 'path': path.relative_to(ROOT).as_posix(),
                            'probe': info, 'decoded_frame_indices': [0, 405, 810], 'content': metrics})
    actual_videos = {p.resolve() for p in output.rglob('*.mp4') if p.is_file()}
    check('35_videos_actual_glob_exact', len(all_videos) == len(set(all_videos)) == 35 and set(all_videos) == actual_videos)
    return results


def _verify_images(artifacts, output, check):
    results = []
    for key in artifacts:
        path = _safe_path(key, output)
        if path.suffix.lower() not in {'.png', '.jpg', '.jpeg', '.gif', '.webp'}:
            continue
        with Image.open(path) as image:
            count = getattr(image, 'n_frames', 1)
            indices = sorted({0, count // 2, count - 1})
            frames = []
            for i in indices:
                image.seek(i)
                frames.append(np.asarray(image.convert('RGB').resize((320, 180))))
            metrics = _content_metrics(np.asarray(frames))
            check('image_nonblank:' + key, image.width >= 64 and image.height >= 64 and _nonblank(metrics))
            if path.suffix.lower() == '.gif':
                check('gif_animation:' + key, count > 1 and _moving(metrics))
            results.append({'path': key, 'frame_count': count, 'representative_indices': indices, 'content': metrics})
    check('rendered_images_present', bool(results))
    return results


def _verify_rendered_coverage(artifacts, sources, output, check):
    """Check the declared presentation matrix, including byte-copied PCC plots."""
    aggregate_names = {'error_curves.png', 'safety_clearance_summary.png',
                       'full_control_timing.png', 'execution_contract.png',
                       'actual_actions.png', 'base_pose_drift.gif'}
    expected = {output / name for name in aggregate_names}
    check('six_nominal_aggregate_products_registered', all(
        p.is_file() and p.relative_to(ROOT).as_posix() in artifacts for p in expected))
    paths = {output / f'{scene}_tracking_paths_3d.png' for scene in SCENES}
    check('five_nominal_path_plots_exact', paths == {
        p.resolve() for p in output.glob('*_tracking_paths_3d.png') if p.is_file()}
        and all(p.relative_to(ROOT).as_posix() in artifacts for p in paths))
    plot_names = {'distance_comparison.png', 'gradient_comparison.png',
                  'minimum_clearance_comparison.png'}
    pcc = {output / 'pcc_monitor' / scene / 'plots' / name
           for scene in SCENES for name in plot_names}
    check('fifteen_nominal_pcc_plots_exact', pcc == {
        p.resolve() for p in (output / 'pcc_monitor').rglob('*.png') if p.is_file()}
        and all(p.relative_to(ROOT).as_posix() in artifacts for p in pcc))
    for scene in sorted(SCENES):
        for name in sorted(plot_names):
            source_key = f'v6_lite/output/runs/research_acceptance_01/simulation/pcc_monitor/{scene}/plots/{name}'
            destination = output / 'pcc_monitor' / scene / 'plots' / name
            artifact_key = destination.relative_to(ROOT).as_posix()
            check(f'{scene}:original_pcc_plot_copy:{name}', source_key in sources
                  and artifact_key in artifacts and destination.is_file()
                  and _sha(destination) == sources[source_key]['sha256'] == artifacts[artifact_key]['sha256'])


def validate(output: Path, write=True) -> dict:
    output = Path(output).resolve()
    checks, videos, images = [], [], []
    def check(name, passed):
        checks.append({'name': name, 'passed': bool(passed)})
    error = None
    manifest_sha = None
    try:
        if not output.is_relative_to(ROOT.resolve()):
            raise ValueError('validation output must stay inside repository')
        manifest_path = output / 'visualization_manifest.json'
        manifest = _json(manifest_path)
        manifest_sha = _sha(manifest_path)
        check('manifest_schema', manifest['schema'] == 'v6_2_latest_visualization_bundle_v1')
        artifacts = _verify_records(manifest['artifacts'], check, 'artifact', output)
        sources = _verify_records(manifest['source_inputs'], check, 'source')
        generators = _verify_records(manifest['generator_source'], check, 'generator_source')
        check('four_actual_visualization_generator_sources', set(generators) == {
            'v6_lite/visualization/' + name for name in
            ('generate_latest_full.py', 'latest_nominal.py', 'latest_saved_renderer.py', 'latest_studies.py')})
        actual = {p.relative_to(ROOT).as_posix() for p in output.rglob('*') if p.is_file()
                  and p.name != 'visualization_manifest.json' and p.relative_to(output).as_posix() not in AUDIT_OUTPUTS}
        check('actual_current_output_exact_artifact_coverage', set(artifacts) == actual)
        _verify_source_pins(manifest, sources, check)
        _verify_claims(manifest, check)
        _verify_failed_stress(sources, check)
        check('local_interactive_entrypoint', (output / 'index.html').is_file()
              and (output / 'index.html').relative_to(ROOT).as_posix() in artifacts
              and '<html' in (output / 'index.html').read_text(encoding='utf-8').lower())
        check('current_metadata_and_result_data_present', all((output / filename).is_file()
              and (output / filename).relative_to(ROOT).as_posix() in artifacts for filename in
              ('generation_plan.json', 'studies_metadata.json', 'nominal_metadata.json', 'result_data.json')))
        _verify_rendered_coverage(artifacts, sources, output, check)
        # Reject incomplete/hash-invalid bundles before expensive media decoding.
        if any(not x['passed'] for x in checks):
            raise ValueError('source, completeness or claim gate rejected bundle')
        videos = _verify_nominal(manifest, artifacts, sources, output, check)
        images = _verify_images(artifacts, output, check)
    except (ValueError, KeyError, TypeError, OSError, RuntimeError, subprocess.SubprocessError) as exc:
        error = {'type': type(exc).__name__, 'message': str(exc)}
        check('validation_completed_without_exception', False)
    report = {'schema': 'v6_2_latest_visualization_validation_v1', 'created_utc': datetime.now(timezone.utc).isoformat(),
              'passed': bool(checks) and all(x['passed'] for x in checks) and error is None,
              'manifest_sha256': manifest_sha, 'validator_sha256': _sha(Path(__file__)),
              'check_count': len(checks), 'passed_check_count': sum(x['passed'] for x in checks),
              'checks': checks, 'error': error, 'videos': videos, 'images': images,
              'scope': {'independent_of_visualization_producer': True, 'producer_imported': False,
                        'original_experiments_recomputed': False, 'nominal_video_count': len(videos),
                        'decoded_representative_frames_per_video': 3,
                        'pixel_quality_checks_are_heuristics': True, 'controller_acceptance_relabelled': False}}
    if write and output.is_relative_to(ROOT.resolve()) and output.is_dir():
        (output / 'visualization_validation.json').write_text(
            json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = validate(args.output_dir)
    print(json.dumps({k: report[k] for k in ('passed', 'check_count', 'passed_check_count', 'error')}, ensure_ascii=True))
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
