"""Only new saved native states can produce execution videos and interactive3D."""
import time,html
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from ff_prior.storage import read_json
from ff_prior_pilot_train.storage import atomic_json
from ff_prior.coupled_predictor import pose_errors
from ff_prior_balanced_evidence.diagnostics import record_saved_trace
from ff_prior_balanced_evidence.media import generate_episode_media
from .protocol import P,ROOT


def deliver(run,robot,entry,progress,budget):
    out=run/'delivery';out.mkdir(exist_ok=False)
    arrays=[];desired=[];body=[];last=None;phases=[]
    for phase in ('A','B','C'):
        path=run/phase/'execution/native_trace.npz'
        if not path.exists():continue
        a=dict(np.load(path));goals=a.get('desired',np.repeat(np.asarray(entry['task']['condition']['goals'])[None],len(a['time']),axis=0))
        if last is not None and not np.array_equal(last,a['integration'][0]):raise ValueError('MEDIA_SEAM_NOT_EXACT')
        start=0 if last is None else 1;last=a['integration'][-1];arrays.append((a,start));desired.append(goals[start:]);phases.append(phase)
        diagnostic=run/phase/'execution'
        if phase=='A':
            diagnostic=run/phase/'body_diagnostics'
            record_saved_trace(robot,entry['scene'],path,diagnostic,control_dt=.02,
                deadline=min(budget.deadline,time.time()+2700),notify=progress)
        body.append(diagnostic)
    if not arrays:
        # Model diagrams are allowed; never animate an unexecuted success.
        images=[]
        for phase in ('A','B','C'):
            models=sorted((run/phase/'planning').glob('guidance_*/center.npz'))
            if not models:continue
            a=np.load(models[-1]);fig=plt.figure();ax=fig.add_subplot(111,projection='3d')
            for arm in range(2):ax.plot(*a['ee'][:,arm,:3,3].T,label=f'arm {arm} MODEL ONLY')
            ax.legend();ax.set_title('MODEL ONLY — no actual execution');name=f'{phase}_model.png';fig.savefig(out/name,dpi=140);plt.close(fig);images.append(name)
        page='<meta charset="utf-8"><h1>新 Diffusion 引导单元：实际执行未形成</h1><p>下游连续追踪 BLOCKED。以下仅为模型诊断；没有新实际视频或三维。</p>'
        page+=''.join(f'<img src="{x}" style="max-width:900px">' for x in images)
        (out/'index.html').write_text(page,encoding='utf-8')
        return dict(status='BLOCKED_NO_NATIVE_MEDIA',model_images=images,index=str(out/'index.html'))
    budget.ensure_output(700*1024**2)
    full={k:np.concatenate([a[k][s:] for a,s in arrays]) for k in ('time','qpos','qvel','ee','base')}
    goals=np.concatenate(desired);t=full['time'];full['clearance_m']=np.concatenate([a.get('clearance_m',a.get('clearance'))[s:] for a,s in arrays])
    pos,rot=pose_errors(full['ee'],goals);v=full['qvel'][:,robot.dof_ids]@robot.Bplus.T
    q=full['qpos'][:,robot.qpos_ids]@robot.Bplus.T
    series=[('position_mm',pos*1000,5),('rotation_deg',np.rad2deg(rot),1),('joint_position',q,None),
        ('joint_velocity_ratio',v/robot.spec.planner_velocity_limits,1),
        ('joint_acceleration_ratio',np.diff(v,axis=0)/.002/robot.spec.planner_acceleration_limits,1),
        ('whole_body_clearance_mm',full['clearance_m']*1000,0),('base_state',full['base'],None)]
    for i,(name,y,gate) in enumerate(series):
        fig,ax=plt.subplots(figsize=(12,5));ax.plot(t[-len(y):],y)
        if gate is not None:ax.axhline(gate,color='red',ls='--')
        ax.set(xlabel='Absolute simulation time (s)',title=name);ax.grid(alpha=.2)
        fig.savefig(out/f'{i+1:02d}_{name}.png',dpi=140);fig.savefig(out/f'{i+1:02d}_{name}.pdf');plt.close(fig)
    for phase,(a,skip) in zip(phases,arrays):
        path=run/phase/'execution';replay=np.load(path/'replay_trace.npz')
        fig,axs=plt.subplots(3,1,figsize=(12,10));tt=a['time']
        axs[0].plot(tt[:-1],a['raw_torque']/robot.spec.torque_limits);axs[0].axhline(1,color='red');axs[0].axhline(-1,color='red');axs[0].set_title('All67 raw torques / original limits')
        mom=a.get('momentum',a.get('base_generalized_momentum'));axs[1].plot(tt,mom);axs[1].set_title('Original generalized momentum — no initial subtraction')
        n=len(replay['integration']);axs[2].plot(tt[:n],abs(a['integration'][:n]-replay['integration']).max(axis=1));axs[2].axhline(1e-10,color='red');axs[2].set_title('All836 replay maximum component difference')
        fig.savefig(out/f'08_{phase}_torque_momentum_replay.png',dpi=140);plt.close(fig)
        ledger=path/'satellite_ledger.npz'
        if ledger.exists():
            l=np.load(ledger);fig,axs=plt.subplots(3,1,figsize=(12,10));axs[0].plot(l['time'][:-1],l['xfrc_applied'].sum(axis=1));axs[1].plot(l['time'],l['impulse']);axs[2].plot(l['time'],l['work_J']);fig.savefig(out/'09_thrust_impulse_work.png',dpi=140);plt.close(fig)
    contract=read_json(ROOT/'ff_prior_balanced_brake/evidence/contract.json')
    result=generate_episode_media(out,robot,full,goals,entry['scene'],contract['waypoints']['waypoint_points_m'],body,
        min(budget.deadline,time.time()+2700),progress,'Saved actual execution; qualification see per-phase reports',
        '冻结 Diffusion 引导：新实际连续链',max_frames=420)
    atomic_json(out/'media.json',result)
    return dict(result,index=str(out/'index.html'),phases=phases,actual_end_s=float(t[-1]))
