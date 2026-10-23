"""Read-only 83.5-second actual chain delivery; never advances dynamics."""
from pathlib import Path
import sys,time,json,html,csv
ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
import numpy as np
from ff_prior.storage import read_json,file_digest
from ff_prior_pilot_train.storage import atomic_json
from ff_prior_satellite_origin.protocol import OUT,PACKAGE,P
from ff_prior_balanced_delivery.tools.brake_media import (load_episode,choose_frame_indices,
    full_episode_figures,timestamp_video,assert_workers_terminal)
from ff_prior_balanced_evidence.media import generate_episode_media
from ff_prior_balanced_evidence.plots import BodySeries
from ff_prior.robot_adapter import RobotAdapter
from ff_prior.scenes import SceneSpec

DEST=PACKAGE/'delivery_001'
class Budget:
    def __init__(self):self.started=time.time();self.deadline=self.started+900
    def check(self,reserve=0):
        if time.time()>self.deadline:raise TimeoutError('MEDIA_TIME_BOUND')
        size=sum(p.stat().st_size for parent in (OUT,PACKAGE) for p in parent.rglob('*') if p.is_file())
        if size+reserve>P['max_bytes']-16*1024**2:raise RuntimeError('TOTAL_UNIT_STORAGE_BOUND')
        return size
    def notify(self,stage,**kw):
        self.check();print(json.dumps(dict(stage=stage,**kw)),flush=True)

def force_figures(out,run,budget):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    a=dict(np.load(run/'execution/satellite_ledger.npz'));s=a['state'];t=a['time']
    meta=read_json(run/'execution/satellite_ledger_metadata.json');target=meta['target_body_id']
    f=a['xfrc_applied'][:,target,:3];linear=s[:,18:21]-s[0,18:21];angular=s[:,21:24]-s[0,21:24]
    fig,axs=plt.subplots(3,2,figsize=(14,11))
    axs[0,0].plot(t[:-1],f);axs[0,0].set(title='Disclosed satellite COM force (world XYZ)',ylabel='N')
    axs[0,1].plot(t,a['impulse']);axs[0,1].plot(t,linear,'--');axs[0,1].set(title='Impulse and measured linear momentum increment',ylabel='N s')
    axs[1,0].plot(t,(s[:,:3]-s[0,:3])*1000);axs[1,0].set(title='Actual satellite displacement',ylabel='mm')
    axs[1,1].plot(t,s[:,12:15]*1000);axs[1,1].set(title='Actual satellite velocity',ylabel='mm/s')
    axs[2,0].plot(t,linear-a['impulse']);axs[2,0].set(title='Linear momentum minus impulse balance',ylabel='N s')
    axs[2,1].plot(t,angular-a['angular_impulse']);axs[2,1].set(title='Angular momentum balance about world origin',ylabel='N m s')
    for ax in axs.flat:ax.axvline(t[0]+2,color='red',ls=':');ax.set_xlabel('absolute time (s)');ax.grid(alpha=.2)
    fig.tight_layout()
    for ext in ('png','pdf'):fig.savefig(out/f'19_physical_origin_ledger.{ext}',dpi=150)
    plt.close(fig)
    native=np.load(run/'execution/native_trace.npz');replay=np.load(run/'execution/replay_trace.npz')
    fig,axs=plt.subplots(2,1,figsize=(14,7))
    axs[0].plot(t,s[:,24]-s[0,24],label='actual kinetic energy increment');axs[0].plot(t,a['work_J'],'--',label='integrated input work');axs[0].legend();axs[0].set(ylabel='J',title='Satellite work-energy ledger')
    ri=replay['integration'];ni=native['integration']
    axs[1].plot(t[:len(ri)],np.max(abs(ni[:len(ri)]-ri),axis=1));axs[1].axhline(1e-10,color='red',ls='--');axs[1].set(title='Origin + coast: full836 independent replay',xlabel='absolute time (s)')
    fig.tight_layout()
    for ext in ('png','pdf'):fig.savefig(out/f'20_origin_energy_full_replay.{ext}',dpi=150)
    plt.close(fig)
    with (out/'satellite_ledger.csv').open('w',encoding='utf-8',newline='') as fh:
        writer=csv.writer(fh);writer.writerow(['absolute_time_s']+[f'satellite_state_{i}' for i in range(25)]+['Jx','Jy','Jz','angular_Jx','angular_Jy','angular_Jz','work_J'])
        for i in range(len(t)):writer.writerow([t[i],*s[i],*a['impulse'][i],*a['angular_impulse'][i],a['work_J'][i]])
    with (out/'satellite_force_inputs.csv').open('w',encoding='utf-8',newline='') as fh:
        writer=csv.writer(fh);writer.writerow(['interval_start_s','interval_end_s','Fx_N','Fy_N','Fz_N','COM_x','COM_y','COM_z'])
        for i in range(len(f)):writer.writerow([t[i],t[i+1],*f[i],*a['force_point_world'][i]])
    budget.check()

def main():
    assert_workers_terminal()
    result=read_json(OUT/'execution/result.json')
    assert read_json(OUT/'status.json')['status']!='RUNNING'
    DEST.mkdir(exist_ok=False);budget=Budget()
    entry,layout,static,dynamic,arrays,goals,dirs,ss,ds=load_episode()
    new=dict(np.load(OUT/'execution/native_trace.npz'))
    assert np.array_equal(dynamic['integration'][-1],new['integration'][0]),'REAL54_5_HANDOFF_NOT_IDENTICAL'
    previous_time=float(arrays['time'][-1]);old_steps=len(arrays['time'])
    for key in ('time','qpos','qvel','ee','base','q17','momentum'):
        arrays[key]=np.concatenate((arrays[key],new[key][1:]))
    arrays['clearance_m']=np.r_[arrays['clearance_m'],new['clearance'][1:]]
    arrays['raw_torque']=np.concatenate((arrays['raw_torque'],new['raw_torque']))
    goals=np.concatenate((goals,new['desired'][1:]))
    dirs.append(OUT/'execution')
    paths=[ROOT/'artifacts/v7_1/balanced_restore_001/units/u00/execution/native_trace.npz',
        ROOT/'artifacts/v7_1/balanced_brake_001/execution/native_trace.npz',
        OUT/'execution/native_trace.npz',OUT/'execution/replay_trace.npz',OUT/'execution/satellite_ledger.npz',
        OUT/'execution/result.json']
    paths += [d/name for d in dirs for name in ('body_obstacle_trace.npz','body_obstacle_metadata.json')]
    before={p.relative_to(ROOT).as_posix():file_digest(p) for p in paths}
    atomic_json(DEST/'sources_before.json',before)
    robot=RobotAdapter(scene=SceneSpec.from_dict(entry['scene']));series=BodySeries(dirs)
    ids,known=choose_frame_indices(arrays['time'],series.a['time'])
    selected={key:arrays[key][ids] for key in ('time','qpos','qvel','ee','base','clearance_m')}
    atomic_json(DEST/'frame_selection.json',dict(original_native_indices=ids,actual_times=selected['time'],exact_body_diagnostic_available=known,interpolated_states=False))
    full_episode_figures(DEST,robot,arrays,goals,series,static,dynamic,budget)
    force_figures(DEST,OUT,budget)
    status=f"APP_DEV; physical origin + coast: {result['status']}; prior seven waypoints 7/7"
    media=generate_episode_media(DEST,robot,selected,goals[ids],entry['scene'],layout['waypoints']['waypoint_points_m'],dirs,budget.deadline,budget.notify,status,
        '真实连续执行：七航点完成后，卫星2秒物理起动＋27秒关机追踪',max_frames=len(ids))
    for video in media['videos']:timestamp_video(DEST,video,media['media_frame_times'],float(arrays['time'][-1]-arrays['time'][0]),budget)
    page=(DEST/'index.html').read_text(encoding='utf-8')
    old_tick="function tick(t){if(playing&&t-last>66){i=(i+1)%D.time.length;last=t;draw()}requestAnimationFrame(tick)}"
    new_tick="let playbackAnchor=null,playbackOrigin=0;function tick(t){if(playing){if(playbackAnchor===null){playbackAnchor=t;playbackOrigin=D.time[i]}let wanted=playbackOrigin+(t-playbackAnchor)/1000;if(wanted>D.time[D.time.length-1]){wanted=D.time[0];playbackAnchor=t;playbackOrigin=wanted}let left=0,right=D.time.length;while(left<right){let middle=(left+right)>>1;if(D.time[middle]<=wanted)left=middle+1;else right=middle}let next=Math.max(0,left-1);if(next!==i){i=next;draw()}}else playbackAnchor=null;requestAnimationFrame(tick)}"
    assert page.count(old_tick)==1
    page=page.replace(old_tick,new_tick).replace("s.oninput=()=>{i=+s.value;draw()}","s.oninput=()=>{i=+s.value;playbackAnchor=null;draw()}")
    note=f'<section><h2>实体运动与接续证据</h2><p>前驱54.5秒已完成原七点；本段从相同完整末态继续。54.5–56.5秒公开COM推力，56.5–83.5秒输入关机。连续体保持W7，刚性臂跟踪卫星实际SE3。结果：{html.escape(result["status"])}。这是恒质量已知外力模型，不包含燃料/排气模拟。</p><p><a href="satellite_ledger.csv">卫星完整动量账本CSV</a> · <a href="satellite_force_inputs.csv">逐步物理推力CSV</a> · <a href="full_joint_motion.csv">17维运动CSV</a> · <a href="full_raw_torque67.csv">67路力矩CSV</a> · <a href="frame_selection.json">实际帧及臂身诊断覆盖</a></p></section>'
    note+=''.join(f'<section><h2>{p.stem}</h2><a href="{p.stem}.pdf">PDF</a><img loading="lazy" src="{p.name}"></section>' for p in sorted(DEST.glob('1[5-9]*.png')))+''.join(f'<section><h2>{p.stem}</h2><img loading="lazy" src="{p.name}"></section>' for p in DEST.glob('20*.png'))
    page=page.replace('</main>',note+'</main>');(DEST/'index.html').write_text(page,encoding='utf-8')
    after={p.relative_to(ROOT).as_posix():file_digest(p) for p in paths};assert before==after
    media.update(source_hashes=before,source_files_unchanged=True,actual_complete_handoff_verified=True,
        actual_duration_s=float(arrays['time'][-1]-arrays['time'][0]),origin_start_time=previous_time,
        original_seven_waypoints='completed by sealed predecessor; not repeated',physical_result=result['status'],
        output_bytes=sum(p.stat().st_size for p in DEST.rglob('*') if p.is_file()),wall_s=time.time()-budget.started,
        full_native_frames=len(arrays['time']),exact_body_frames=int(known.sum()),selected_frames=len(ids),
        new_native_steps=0,new_replay_steps=0,new_coupled_calls=0,new_geometry_queries=0,
        files={p.name:dict(bytes=p.stat().st_size,sha256=file_digest(p)) for p in DEST.iterdir() if p.is_file()})
    atomic_json(DEST/'manifest.json',media)
    print(json.dumps({k:media[k] for k in ('status','actual_duration_s','output_bytes','wall_s','videos','errors')},ensure_ascii=False))
if __name__=='__main__':main()
