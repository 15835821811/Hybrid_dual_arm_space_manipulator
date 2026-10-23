"""Independent brake-unit read-only delivery; never edits frozen sources or runs.

CLI --check reads the completed static observation coverage only.
CLI --software-check uses synthetic timestamps without opening run artifacts.
CLI --render creates one new output directory after all workers are terminal.
"""
from pathlib import Path
import argparse
import csv
from fractions import Fraction
import html
import json
import os
import shutil
import subprocess
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
import sys
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from ff_prior.storage import read_json, file_digest
from ff_prior_pilot_train.storage import atomic_json

STATIC = ROOT/'artifacts/v7_1/balanced_restore_001'
DYNAMIC = ROOT/'artifacts/v7_1/balanced_brake_001'
COMPLETE_BODY = ROOT/'ff_prior_balanced_delivery/static_body_completion'
PACKAGE = ROOT/'ff_prior_balanced_delivery'
WALL_S = 600
OUTPUT_BYTES = 512*1024**2


def check_static():
    """Source/coverage checks, no new geometry or repeat scientific audit."""
    native = STATIC/'units/u00/execution/native_trace.npz'
    meta = read_json(COMPLETE_BODY/'body_obstacle_metadata.json')
    assert meta['status'] == 'COMPLETE'
    assert meta['source_native_sha256'] == file_digest(native)
    with np.load(native) as n, np.load(COMPLETE_BODY/'body_obstacle_trace.npz') as b:
        nt, bt = n['time'], b['time']
        assert len(nt) == len(bt) == meta['samples'] == meta['expected_samples'] == 13751
        assert np.array_equal(nt, bt)
        assert np.array_equal(b['native_index'], np.arange(len(nt)))
        assert b['distance_m'].shape == (len(nt), 7, len(meta['links']))
        assert b['nearest_geom_id'].shape == b['distance_m'].shape
        assert b['connector'].shape == b['distance_m'].shape+(2, 3)
    return dict(status='PASS', static_native_states=len(nt), body_samples=len(bt),
        every_saved_static_state_has_exact_body_time=True,
        completed_body_metadata_sha256=file_digest(COMPLETE_BODY/'body_obstacle_metadata.json'),
        source_native_sha256=meta['source_native_sha256'], new_native_steps=0,
        new_coupled_calls=0, new_geometry_queries=0,
        scientific_reaudit=False, render_performed=False)


def choose_frame_indices(native_times, diagnostic_times, fps=15):
    """Near-uniform physical time, snapped only to stored native/diagnostic times.

    Uncovered intervals retain actual native frames and explicit missing-body
    flags. No distance or actual state is interpolated.
    """
    times = np.asarray(native_times, dtype=float)
    diag = np.unique(np.asarray(diagnostic_times, dtype=float))
    if not len(times) or np.any(np.diff(times) <= 0):
        raise ValueError('NATIVE_TIME_MUST_STRICTLY_INCREASE')
    legal = []
    for t in diag:
        j = int(np.searchsorted(times, t))
        candidates = [k for k in (j-1, j) if 0 <= k < len(times)]
        k = min(candidates, key=lambda k: abs(times[k]-t))
        if abs(times[k]-t) <= 1e-7:
            legal.append(k)
    legal = np.unique(legal).astype(int)
    duration = float(times[-1]-times[0])
    targets = np.linspace(times[0], times[-1], max(1, int(round(duration*fps))+1))
    chosen = []
    for target in targets:
        k = int(np.argmin(abs(times-target)))
        if len(legal):
            candidate = int(legal[np.argmin(abs(times[legal]-target))])
            if abs(times[candidate]-target) <= .0101:
                k = candidate
        chosen.append(k)
    indices = np.unique(np.r_[0, chosen, len(times)-1]).astype(int)
    return indices, np.isin(indices, legal)


def software_check():
    """Only new-unit bindings and synthetic sampling; no model or saved raw."""
    checks = []
    assert DYNAMIC == ROOT/'artifacts/v7_1/balanced_brake_001'
    checks.append('new_brake_run_binding')
    times = np.arange(27251)*.002
    diag = np.r_[times[:13751], times[13760::10]]
    ids, known = choose_frame_indices(times, diag)
    assert known.all() and ids[0] == 0 and ids[-1] == len(times)-1
    assert abs((times[ids] <= 27.5).sum()/len(ids)-27.5/54.5) < .003
    assert np.max(np.diff(times[ids])) <= .081
    checks.append('mixed_2ms_20ms_samples_preserve_physical_phase_proportions')
    partial = times[times <= 47.16+1e-9]
    ids, known = choose_frame_indices(partial, diag[diag <= partial[-1]+1e-9])
    assert known.all() and partial[ids[-1]] == partial[-1]
    assert np.max(partial[ids]) < 54.5
    checks.append('failed_prefix_does_not_extrapolate_unexecuted_suffix')
    ids, known = choose_frame_indices(times, times[:9912])
    assert (~known).any() and np.all(np.isin(times[ids[known]], times[:9912]))
    checks.append('missing_geometry_marked_without_distance_interpolation')
    ids, known = choose_frame_indices(times, [])
    assert not known.any() and ids[0] == 0 and ids[-1] == len(times)-1
    checks.append('native_prefix_without_geometry_stays_explicitly_uncovered')
    for invalid in ([], [0., 0.], [1., 0.]):
        try:
            choose_frame_indices(invalid, [])
        except ValueError:
            continue
        raise AssertionError('INVALID_NATIVE_TIMES_ACCEPTED')
    checks.append('nonincreasing_or_empty_native_time_rejected')
    return dict(status='PASS', checks=checks, synthetic_timestamps_only=True,
        run_artifacts_opened=False, model_compilations=0,
        new_native_steps=0, new_coupled_calls=0, new_geometry_queries=0,
        render_performed=False)


def assert_workers_terminal():
    from ff_prior_fullchain.job import process_probe
    for status_file in (ROOT/'artifacts/v7_1').glob('*/status.json'):
        status = read_json(status_file)
        if status.get('status') != 'RUNNING':
            continue
        probe = process_probe(status.get('pid', 0), status.get('started_at'))
        if probe.get('alive') is not False:
            raise RuntimeError('DELIVERY_REFUSES_CONCURRENT_OR_UNRESOLVED_WORKER: '+str(status_file))
    for run in (STATIC, DYNAMIC):
        path = run/'status.json'
        if path.exists() and read_json(path).get('status') == 'RUNNING':
            raise RuntimeError('DELIVERY_REQUIRES_RESOLVED_TERMINAL_RUN: '+str(run))


class DeliveryBudget:
    def __init__(self, out):
        self.out = out
        self.started = time.time()
        self.deadline = self.started+WALL_S

    def check(self, reserve=0):
        if time.time() >= self.deadline:
            raise TimeoutError('FINAL_MEDIA_600S_BUDGET')
        total = sum(p.stat().st_size for p in self.out.rglob('*') if p.is_file())
        if total+reserve > OUTPUT_BYTES:
            raise RuntimeError('FINAL_MEDIA_512MIB_OUTPUT_BUDGET')
        return total

    def notify(self, stage, **kw):
        self.check()
        print(json.dumps(dict(stage=stage, elapsed_s=time.time()-self.started, **kw)), flush=True)


def load_episode():
    entry = read_json(ROOT/'ff_prior_balanced_restore/evidence/selection.json')['entries'][0]
    layout = read_json(ROOT/'ff_prior_balanced_scene/evidence/layout.json')
    with np.load(STATIC/'units/u00/execution/native_trace.npz') as z:
        static = {k:z[k] for k in z.files}
    static_status = read_json(STATIC/'units/u00/execution/report.json')['level']
    common = ('time', 'qpos', 'qvel', 'ee', 'base')
    arrays = {k:static[k] for k in common}
    goals = np.repeat(np.asarray(entry['task']['condition']['goals'])[None], len(static['time']), axis=0)
    arrays.update(clearance_m=static['clearance_m'], q17=static['q_actual_17'],
        raw_torque=static['raw_torque'], momentum=static['base_generalized_momentum'])
    diagnostic_dirs = [COMPLETE_BODY]
    dynamic = None
    dynamic_status = 'BLOCKED_NO_DYNAMIC_NATIVE'
    dp = DYNAMIC/'execution/native_trace.npz'
    if dp.exists():
        with np.load(dp) as z:
            dynamic = {k:z[k] for k in z.files}
        result_path = DYNAMIC/'execution/result.json'
        result = read_json(result_path) if result_path.exists() else {}
        dynamic_status = result.get('status', 'PARTIAL_NATIVE_NO_TERMINAL_EXECUTION_REPORT')
        assert len(dynamic['time']) and np.array_equal(static['integration'][-1], dynamic['integration'][0]), 'ACTUAL_HANDOFF_836_NOT_IDENTICAL'
        contract = read_json(ROOT/'ff_prior_balanced_brake/evidence/contract.json')
        assert contract['scene'] == entry['scene']
        assert np.array_equal(contract['waypoints']['waypoint_points_m'], layout['waypoints']['waypoint_points_m'])
        arrays = {k:np.concatenate((static[k], dynamic[k][1:])) for k in common}
        goals = np.concatenate((goals, dynamic['desired'][1:]))
        arrays.update(clearance_m=np.r_[static['clearance_m'], dynamic['clearance'][1:]],
            q17=np.concatenate((static['q_actual_17'], dynamic['q17'][1:])),
            raw_torque=np.concatenate((static['raw_torque'], dynamic['raw_torque'])),
            momentum=np.concatenate((static['base_generalized_momentum'], dynamic['momentum'][1:])))
        diagnostic_dirs.append(DYNAMIC/'execution')
    return entry, layout, static, dynamic, arrays, goals, diagnostic_dirs, static_status, dynamic_status


def _save(fig, out, name, budget):
    import matplotlib.pyplot as plt
    budget.check(16*1024**2)
    fig.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(out/f'{name}.{ext}', dpi=150, bbox_inches='tight')
    plt.close(fig)


def full_episode_figures(out, robot, arrays, goals, series, static, dynamic, budget):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from ff_prior.coupled_predictor import pose_errors
    from ff_prior_balanced_evidence.plots import time_cell_edges
    t = arrays['time']
    pos, rot = pose_errors(arrays['ee'], goals)
    velocity = arrays['qvel'][:,robot.dof_ids]@robot.Bplus.T
    acceleration = np.diff(velocity, axis=0)/np.diff(t)[:,None]
    # Actual nonuniform diagnostic time coordinates, rasterized PDF meshes.
    if series.a:
        edges = time_cell_edges(series.a['time'])
        labels = [link['body_name'] for link in series.meta['links']]
        for oi, obstacle in enumerate(series.meta['obstacles']):
            budget.check(16*1024**2)
            fig, ax = plt.subplots(figsize=(14, 7))
            im = ax.pcolormesh(edges, np.arange(len(labels)+1)-.5,
                series.a['distance_m'][:,oi].T*1000, shading='flat', rasterized=True,
                cmap='viridis', vmin=0, vmax=100)
            stride = max(1, len(labels)//25)
            ax.set_yticks(np.arange(0,len(labels),stride), labels[::stride], fontsize=7)
            ax.set(title=f"{obstacle['object_id']}: exact distance to every collision body", xlabel='absolute simulation time (s)', ylabel='body')
            fig.colorbar(im, ax=ax, label='mm; color display saturates at 100 mm, data unchanged')
            _save(fig, out, f"14_{obstacle['object_id']}_body_distance", budget)
    fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    axes[0,0].plot(t,pos*1000); axes[0,0].axhline(5,color='red',ls='--');axes[0,0].set(title='Actual / target position error; static hold and dynamic gates',ylabel='mm')
    axes[0,1].plot(t,np.rad2deg(rot));axes[0,1].axhline(1,color='red',ls='--');axes[0,1].set(title='Actual / target rotation error',ylabel='degree')
    axes[1,0].plot(t,arrays['clearance_m']*1000);axes[1,0].axhline(0,color='red');axes[1,0].set(title='Whole-machine acceptance clearance',ylabel='mm')
    axes[1,1].plot(t,arrays['base'][:,:3]);axes[1,1].set(title='Continuous actual free-base translation',ylabel='m')
    axes[2,0].plot(t,arrays['base'][:,3:]);axes[2,0].set(title='Continuous actual free-base quaternion')
    axes[2,1].plot(t,arrays['momentum']);axes[2,1].set(title='All six actual base generalized momentum components')
    for ax in axes.flat:
        ax.axvline(static['time'][-1],color='grey',ls=':');ax.set_xlabel('absolute simulation time (s)');ax.grid(alpha=.2)
    _save(fig,out,'15_full_episode_errors_base_momentum',budget)
    fig,axes=plt.subplots(3,1,figsize=(14,10))
    axes[0].plot(t,arrays['q17']);axes[0].set(title='All 17 actual coordinates',ylabel='rad')
    axes[1].plot(t,velocity/robot.spec.planner_velocity_limits);axes[1].set(ylabel='velocity / limit')
    axes[2].plot(t[1:],acceleration/robot.spec.planner_acceleration_limits);axes[2].set(ylabel='acceleration / limit',xlabel='absolute time (s)')
    for ax in axes[1:]:ax.axhline(1,color='red',ls='--');ax.axhline(-1,color='red',ls='--')
    _save(fig,out,'16_full_episode_17_motion',budget)
    fig,axes=plt.subplots(2,1,figsize=(14,8))
    torque=arrays['raw_torque'];axes[0].plot(t[:-1],torque);axes[0].set(title='All 67 actual raw torques',ylabel='N m')
    axes[1].plot(t[:-1],torque/robot.spec.torque_limits);axes[1].axhline(1,color='red',ls='--');axes[1].axhline(-1,color='red',ls='--');axes[1].set(ylabel='raw torque / limit',xlabel='absolute time (s)')
    _save(fig,out,'17_full_episode_67_torque',budget)
    fig,ax=plt.subplots(figsize=(14,4))
    for run, native in ((STATIC/'units/u00',static),(DYNAMIC,dynamic)):
        path=run/'execution/replay_trace.npz'
        if native is not None and path.exists():
            with np.load(path) as replay:
                count=min(len(native['time']),len(replay['integration']))
                ax.plot(native['time'][:count],np.max(abs(native['integration'][:count]-replay['integration'][:count]),axis=1))
    ax.axhline(1e-10,color='red',ls='--');ax.set(title='Recorded independent full 836-state replay differences; no new replay',xlabel='absolute time (s)',ylabel='maximum absolute difference')
    _save(fig,out,'18_full_episode_replay836',budget)
    with (out/'timeseries.csv').open('w',encoding='utf-8',newline='') as f:
        writer=csv.writer(f);writer.writerow(['absolute_time_s','continuum_position_mm','rigid_position_mm','continuum_rotation_deg','rigid_rotation_deg','whole_body_clearance_mm','segment'])
        for i in range(len(t)):
            writer.writerow([t[i],*(1000*pos[i]),*np.rad2deg(rot[i]),1000*arrays['clearance_m'][i],'static' if t[i]<=static['time'][-1]+1e-8 else 'dynamic'])
    with (out/'full_joint_motion.csv').open('w',encoding='utf-8',newline='') as f:
        writer=csv.writer(f);writer.writerow(['absolute_time_s']+[f'q{j}_rad' for j in range(17)]+[f'qd{j}_rad_s' for j in range(17)]+[f'qdd{j}_rad_s2' for j in range(17)])
        for i in range(len(t)):
            writer.writerow([t[i],*arrays['q17'][i],*velocity[i],*(acceleration[i-1] if i else ['']*17)])
    with (out/'full_raw_torque67.csv').open('w',encoding='utf-8',newline='') as f:
        writer=csv.writer(f);writer.writerow(['absolute_time_s']+[f'raw_torque_{j}_Nm' for j in range(67)])
        for ti,row in zip(t[:-1],torque):writer.writerow([ti,*row])
    budget.check()


def _ass_clock(t):
    centi=int(round(float(t)*100))
    return f'{centi//360000}:{centi//6000%60:02d}:{centi//100%60:02d}.{centi%100:02d}'


def timestamp_video(out, video, frame_times, duration, budget):
    """Keep all rendered frames; show their actual timestamps and real-time span."""
    ffmpeg=shutil.which('ffmpeg');ffprobe=shutil.which('ffprobe')
    if not ffmpeg or not ffprobe:raise RuntimeError('FFMPEG_FFPROBE_REQUIRED')
    count=len(frame_times);playback=duration if duration>0 else 1.
    fps=Fraction(count,1)/Fraction(str(round(playback,9)))
    original_fps=video['fps'];scale=playback/(count/original_fps)
    ass=out/(Path(video['file']).stem+'_timestamps.ass')
    header='[Script Info]\nScriptType: v4.00+\nPlayResX: 960\nPlayResY: 720\n[V4+ Styles]\nFormat: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\nStyle: Default,Arial,20,&H00FFFFFF,&H000000FF,&H00000000,&H80000000,0,0,0,0,100,100,0,0,1,2,1,7,16,16,12,1\n[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n'
    lines=[]
    for j,t in enumerate(frame_times):
        lines.append(f'Dialogue: 0,{_ass_clock(j*playback/count)},{_ass_clock((j+1)*playback/count)},Default,,0,0,0,,Recorded t = {t:.3f} s | RGB = XYZ | cyan EE0 / orange EE1\n')
    ass.write_text(header+''.join(lines),encoding='utf-8')
    path=out/video['file'];raw=path.with_name(path.stem+'_rendered.mp4');path.rename(raw)
    try:
        timeout=max(1,int(budget.deadline-time.time()))
        command=[ffmpeg,'-hide_banner','-loglevel','error','-n','-i',raw.name,'-vf',f'setpts={scale:.12g}*PTS,subtitles={ass.name}',
            '-fps_mode','cfr','-r',f'{fps.numerator}/{fps.denominator}','-an','-c:v','libx264','-threads','2','-preset','fast','-crf','23','-pix_fmt','yuv420p','-movflags','+faststart',path.name]
        completed=subprocess.run(command,cwd=out,capture_output=True,timeout=timeout,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        if completed.returncode:raise RuntimeError(completed.stderr.decode('utf-8',errors='replace'))
        probe=subprocess.run([ffprobe,'-v','error','-count_frames','-select_streams','v:0','-show_entries','stream=nb_read_frames,duration,r_frame_rate','-of','json',path.name],cwd=out,capture_output=True,text=True,timeout=max(1,int(budget.deadline-time.time())),creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        if probe.returncode:raise RuntimeError(probe.stderr)
        stream=json.loads(probe.stdout)['streams'][0]
        assert int(stream['nb_read_frames'])==count,'FINAL_VIDEO_FRAME_COUNT_CHANGED'
        assert abs(float(stream['duration'])-playback)<.002,'FINAL_VIDEO_PHYSICAL_TIME_SCALE_MISMATCH'
        raw.unlink()  # Only the newly created duplicate delivery intermediate.
        video.update(fps=float(fps),source_render_fps=original_fps,playback_duration_s=float(stream['duration']),
            decoded_frames=int(stream['nb_read_frames']),timestamp_overlay='actual stored frame time',
            playback_speed='physical-time uniform target frames snapped to exact diagnostic ticks',
            maximum_timestamp_resampling_offset_s=float(np.max(abs(np.asarray(frame_times)-frame_times[0]-np.arange(count)*playback/count))),
            timestamp_ass=ass.name)
    except Exception:
        # Retain both intermediate and any partial output as failure evidence.
        raise
    budget.check()


def render(out):
    assert_workers_terminal()
    check=check_static()
    out=out.resolve()
    allowed=PACKAGE.resolve()
    if not out.is_relative_to(allowed):raise ValueError('DELIVERY_OUTPUT_MUST_STAY_IN_NEW_PACKAGE')
    out.mkdir(parents=True,exist_ok=False)
    budget=DeliveryBudget(out)
    atomic_json(out/'reservation.json',dict(wall_s=WALL_S,output_bytes=OUTPUT_BYTES,
        kind='read_only_saved_state_media',new_native_steps=0,new_coupled_calls=0,new_geometry_queries=0,
        no_writes_to_frozen_sources_or_runs=True,static_check=check))
    try:
        entry,layout,static,dynamic,arrays,goals,dirs,ss,ds=load_episode()
        source_paths=[Path(__file__).resolve(),ROOT/'ff_prior_balanced_evidence/media.py',ROOT/'ff_prior_balanced_evidence/plots.py',
            STATIC/'units/u00/execution/native_trace.npz',STATIC/'units/u00/execution/replay_trace.npz',
            COMPLETE_BODY/'body_obstacle_trace.npz',COMPLETE_BODY/'body_obstacle_metadata.json']
        if dynamic is not None:source_paths.extend([DYNAMIC/'execution/native_trace.npz',DYNAMIC/'execution/body_obstacle_trace.npz'])
        source_hashes={str(p.relative_to(ROOT)):file_digest(p) for p in source_paths if p.exists()}
        atomic_json(out/'sources_before.json',source_hashes)
        for prefix,directory in (('static_',STATIC/'units/u00/visuals'),('dynamic_',DYNAMIC/'visuals')):
            for path in directory.glob('*'):
                if path.suffix in ('.png','.pdf') and path.stem[:2].isdigit() and int(path.stem[:2])<=8:
                    budget.check(path.stat().st_size);shutil.copy2(path,out/(prefix+path.name))
        from ff_prior_balanced_evidence.plots import BodySeries
        from ff_prior_balanced_evidence.media import generate_episode_media
        from ff_prior.robot_adapter import RobotAdapter
        from ff_prior.scenes import SceneSpec
        robot=RobotAdapter(scene=SceneSpec.from_dict(entry['scene']))
        series=BodySeries(dirs)
        frame_indices,known=choose_frame_indices(arrays['time'],series.a['time'] if series.a else [])
        selected={key:np.asarray(arrays[key])[frame_indices] for key in ('time','qpos','qvel','ee','base','clearance_m')}
        atomic_json(out/'frame_selection.json',dict(original_native_indices=frame_indices,
            actual_times=selected['time'],exact_body_diagnostic_available=known,
            policy='Uniform absolute physical-time targets -> nearest exact body-diagnostic native tick within 10.1ms; otherwise actual native frame explicitly without body evidence',
            interpolated_actual_states=False,interpolated_distances=False))
        full_episode_figures(out,robot,arrays,goals,series,static,dynamic,budget)
        del series
        dynamic_summary = read_json(DYNAMIC/'execution/result.json') if dynamic is not None and (DYNAMIC/'execution/result.json').exists() else {}
        waypoint_observations = dynamic_summary.get('original_waypoint_observations', [])
        reached = sum(bool(x['recorded_through_arrival']) for x in waypoint_observations)
        stop_reason = dynamic_summary.get('stop_reason')
        status=f'STATIC {ss}; DYNAMIC {ds}; original waypoints {reached}/7; {stop_reason or ""}'
        title = ('新七障碍修正单元：动态未完成，仅交付实际记录前缀' if ds!='EXECUTION_VERIFIED' else '新七障碍修正单元：完整连续实际记录')
        media=generate_episode_media(out,robot,selected,goals[frame_indices],entry['scene'],
            layout['waypoints']['waypoint_points_m'],dirs,budget.deadline,budget.notify,status,
            title+'：坐标系、轨迹与臂身避障观测',max_frames=len(frame_indices))
        mapped=frame_indices[np.asarray(media['media_frame_native_indices'],dtype=int)]
        assert np.array_equal(arrays['time'][mapped],media['media_frame_times'])
        expected_exact=int(known[np.asarray(media['media_frame_native_indices'],dtype=int)].sum())
        assert media['body_diagnostic_exact_frame_count']==expected_exact,'BODY_FRAME_COVERAGE_ACCOUNTING'
        for video in media['videos']:
            timestamp_video(out,video,media['media_frame_times'],float(arrays['time'][-1]-arrays['time'][0]),budget)
        page=(out/'index.html').read_text(encoding='utf-8')
        gallery=''.join(f'<section><h2>{html.escape(path.stem)}</h2><a href="{path.stem}.pdf">PDF</a><img loading="lazy" src="{path.name}"></section>' for path in sorted(out.glob('dynamic_*.png')))
        note=f'<section><h2>交付边界</h2><p>静态：{html.escape(ss)}；动态：{html.escape(ds)}；原航点到达 {reached}/7。停止原因：{html.escape(str(stop_reason))}。最终动画显示已保存的真实前缀；无新原生步、无额外重放或几何距离查询。时间戳为实际记录时刻，播放按绝对物理时间近均匀取帧；未插值状态或距离。动态是否完成以原始执行报告为准；距离诊断保留原有 COMPLETE/PARTIAL 状态，已执行区间的观测完整性单独记录。</p><p><a href="frame_selection.json">逐帧原生索引与诊断覆盖</a> · <a href="full_joint_motion.csv">全部17坐标、速度、加速度CSV</a> · <a href="full_raw_torque67.csv">全部67力矩CSV</a> · <a href="sources_before.json">准确源文件hash</a></p></section>'
        note += '<section><h2>静态独立原媒体</h2><p><a href="http://127.0.0.1:8771/static/index.html">静态 27.5 秒独立原媒体索引</a>。原版本仅保留 9912 个臂身诊断样本，诊断为 PARTIAL；原文件与原播放速度均保留。完整 13751 个静态臂身样本和准确物理播放时间以本页最终交付为准。</p></section>'
        page=page.replace('</main>',note+gallery+'</main>')
        # Only this new delivery page changes: the frozen template remains intact.
        old_tick="function tick(t){if(playing&&t-last>66){i=(i+1)%D.time.length;last=t;draw()}requestAnimationFrame(tick)}"
        new_tick="let playbackAnchor=null,playbackOrigin=0;function tick(t){if(playing){if(playbackAnchor===null){playbackAnchor=t;playbackOrigin=D.time[i]}let wanted=playbackOrigin+(t-playbackAnchor)/1000;if(wanted>D.time[D.time.length-1]){wanted=D.time[0];playbackAnchor=t;playbackOrigin=wanted}let left=0,right=D.time.length;while(left<right){let middle=(left+right)>>1;if(D.time[middle]<=wanted)left=middle+1;else right=middle}let next=Math.max(0,left-1);if(next!==i){i=next;draw()}}else playbackAnchor=null;requestAnimationFrame(tick)}"
        assert page.count(old_tick)==1,'FROZEN_TEMPLATE_PLAYBACK_HOOK_CHANGED'
        page=page.replace(old_tick,new_tick)
        page=page.replace("s.oninput=()=>{i=+s.value;draw()}","s.oninput=()=>{i=+s.value;playbackAnchor=null;draw()}")
        (out/'index.html').write_text(page,encoding='utf-8')
        if dynamic is None:
            (out/'dynamic_blocked.html').write_text('<meta charset="utf-8"><h1>动态依赖未形成实际记录</h1><p>当前仅交付静态实际证据，不制造动态成功动画。</p><a href="index.html">静态交付</a>',encoding='utf-8')
        hashes_after={str(p.relative_to(ROOT)):file_digest(p) for p in source_paths if p.exists()}
        assert hashes_after==source_hashes,'READ_ONLY_SOURCE_MUTATION'
        media.update(static_status=ss,dynamic_status=ds,
            delivery_unit='balanced_brake_001', source_dynamic_run=str(DYNAMIC.relative_to(ROOT)),
            dynamic_original_waypoints_reached=reached,dynamic_original_waypoints_total=7,
            dynamic_stop_reason=stop_reason,planned_episode_complete=ds=='EXECUTION_VERIFIED',
            actual_prefix_media_complete=len(media['videos'])==2 and expected_exact==len(mapped) and not media['errors'],
            actual_complete_handoff_verified=dynamic is not None,
            original_native_frame_indices=mapped.tolist(),source_hashes=source_hashes,
            source_files_unchanged=True,wall_s=time.time()-budget.started,output_bytes=budget.check(),
            new_native_steps=0,new_coupled_calls=0,new_geometry_queries=0,
            full_static_body_samples=13751,metadata_only_not_scientific_reaudit=True,
            files={p.name:dict(bytes=p.stat().st_size,sha256=file_digest(p)) for p in out.iterdir() if p.is_file()})
        atomic_json(out/'manifest.json',media)
        return dict(status=media['status'],out=str(out),wall_s=media['wall_s'],bytes=media['output_bytes'],videos=media['videos'])
    except Exception as exc:
        atomic_json(out/'delivery_failure.json',dict(status='PARTIAL_DELIVERY',error=repr(exc),wall_s=time.time()-budget.started,
            no_physics_repeated=True,no_frozen_source_changes=True))
        raise


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--check',action='store_true')
    parser.add_argument('--software-check',action='store_true')
    parser.add_argument('--render',action='store_true')
    parser.add_argument('--out',type=Path,default=PACKAGE/'final_brake_001')
    args=parser.parse_args()
    if sum((args.check,args.software_check,args.render))!=1:
        parser.error('Choose exactly one of --check, --software-check or --render')
    result=software_check() if args.software_check else (check_static() if args.check else render(args.out))
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
