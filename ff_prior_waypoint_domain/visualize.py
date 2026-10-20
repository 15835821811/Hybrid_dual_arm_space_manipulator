"""Render saved native states only; no controls or new physical trials."""
from pathlib import Path
import json
import subprocess
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation
from ff_prior.robot_adapter import RobotAdapter
from ff_prior.coupled_predictor import pose_errors
from ff_prior.storage import atomic_json, read_json, file_digest
from .experiment import OUT,ROOT


def plots(directory, a, result):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    t = a["time"]
    p,r = pose_errors(a["ee"],a["desired"])
    fig,ax = plt.subplots(3,3,figsize=(16,11),layout="constrained")
    for arm,name in enumerate(("Continuum","Rigid")):
        for j,c in enumerate(("tab:blue","tab:orange","tab:green")):
            ax[0,arm].plot(t,a["ee"][:,arm,j,3],color=c,label=f"actual {'xyz'[j]}")
            ax[0,arm].plot(t,a["desired"][:,arm,j,3],color=c,ls="--",label=f"desired {'xyz'[j]}")
        ax[0,arm].set_title(name+" world position")
        ax[0,arm].set_ylabel("m")
        ax[0,arm].legend(fontsize=7,ncol=2)
        ax[1,0].plot(t,p[:,arm]*1000,label=name)
        ax[1,1].plot(t,np.rad2deg(r[:,arm]),label=name)
    for j in range(3): ax[0,2].plot(t,(a["base"][:,j]-a["base"][0,j])*1000,label="xyz"[j])
    ax[0,2].set_title("Free base translation"); ax[0,2].set_ylabel("mm"); ax[0,2].legend()
    ax[1,0].axhline(5,color="red",ls="--",label="5 mm gate")
    ax[1,0].set_title("Instantaneous reference tracking"); ax[1,0].set_ylabel("position error (mm)"); ax[1,0].legend()
    ax[1,1].axhline(1,color="red",ls="--",label="1 degree gate")
    ax[1,1].set_title("World orientation tracking"); ax[1,1].set_ylabel("degrees"); ax[1,1].legend()
    rotation = Rotation.from_quat(a["base"][:,[4,5,6,3]])
    angle = np.rad2deg((rotation*rotation[0].inv()).as_rotvec())
    for j in range(3): ax[1,2].plot(t,angle[:,j],label="xyz"[j])
    ax[1,2].set_title("Free base rotation vector"); ax[1,2].set_ylabel("degrees"); ax[1,2].legend()
    ax[2,0].plot(t,a["clearance"]*1000); ax[2,0].axhline(0,color="red",ls="--")
    ax[2,0].set_title("Full-body minimum clearance"); ax[2,0].set_ylabel("mm")
    ax[2,1].plot(t,np.max(abs(a["q17"]-a["qref"]),axis=1)*1000)
    ax[2,1].axhline(20,color="red",ls="--"); ax[2,1].set_title("Joint reference tracking"); ax[2,1].set_ylabel("mrad")
    ratio = np.max(abs(a["raw_torque"])/np.r_[np.full(60,4.),np.full(7,80.)],axis=1)
    ax[2,2].plot(t[1:],ratio); ax[2,2].axhline(1,color="red",ls="--")
    ax[2,2].set_title("Requested torque / physical limit"); ax[2,2].set_ylabel("ratio")
    for x in ax.flat: x.set_xlabel("simulation time (s)"); x.grid(alpha=.2); x.axvline(25.5,color="gray",lw=.5)
    fig.suptitle(str(directory.relative_to(OUT))+" | "+result["status"])
    fig.savefig(directory/"tracking.png",dpi=150); plt.close(fig)
    fig = plt.figure(figsize=(12,5),layout="constrained")
    for arm,name in enumerate(("Continuum","Rigid")):
        x = fig.add_subplot(1,2,arm+1,projection="3d")
        x.plot(*a["ee"][:,arm,:3,3].T,label="actual")
        x.plot(*a["desired"][:,arm,:3,3].T,ls="--",label="desired")
        x.set(xlabel="world x (m)",ylabel="world y (m)",zlabel="world z (m)",title=name)
        x.legend()
    fig.savefig(directory/"world_paths.png",dpi=150); plt.close(fig)
    fig,ax=plt.subplots(2,1,figsize=(12,7),layout="constrained")
    for j in range(10):
        ax[0].plot(t,a["q17"][:,j],lw=.8,label=f"q{j+1}")
    ax[0].axhline(1,color="red",ls="--");ax[0].axhline(-1,color="red",ls="--")
    ax[0].set(title="Continuum work domain (hard bounds -1, +1 rad)",ylabel="rad")
    ax[0].legend(ncol=5,fontsize=8)
    for arm,selection in (("continuum",slice(0,60)),("rigid",slice(60,67))):
        ax[1].plot(t[1:],np.max(abs(a["raw_torque"][:,selection]),axis=1),label=arm)
    ax[1].set(title="Maximum requested actuator torque per arm",ylabel="Nm");ax[1].legend()
    for panel in ax: panel.set_xlabel("time (s)");panel.grid(alpha=.2)
    fig.savefig(directory/"joint_domain_and_torque.png",dpi=150);plt.close(fig)
    columns = np.c_[t,p*1000,np.rad2deg(r),a["clearance"]*1000,a["base"]]
    np.savetxt(directory/"tracking.csv",columns,delimiter=",",
        header="time_s,continuum_position_mm,rigid_position_mm,continuum_rotation_deg,rigid_rotation_deg,clearance_mm,base_x,base_y,base_z,base_qw,base_qx,base_qy,base_qz",comments="")


def video(robot,directory,a):
    if (directory/"visualization_provenance.json").exists():
        old=read_json(directory/"visualization_provenance.json")
        if old["native_sha256"]==file_digest(directory/"native_trace.npz") and old["video_sha256"]==file_digest(directory/"actual_motion.mp4"):
            return old["video_frames"]
    renderer = mujoco.Renderer(robot.model,height=480,width=640)
    d = robot.new_data()
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    points = np.concatenate([a["skeleton_c"].reshape(-1,3),a["skeleton_r"].reshape(-1,3),a["base"][:,:3]])
    camera.lookat[:] = (points.min(0)+points.max(0))/2
    camera.distance = max(3.8,float(np.linalg.norm(points.max(0)-points.min(0))*1.5))
    camera.azimuth, camera.elevation = 120,-25
    target = directory/"actual_motion.mp4"
    command = ["ffmpeg","-y","-loglevel","error","-f","rawvideo","-pix_fmt","rgb24","-s","640x480","-r","15",
        "-i","-","-an","-c:v","libx264","-crf","21","-pix_fmt","yuv420p",str(target)]
    proc = subprocess.Popen(command,stdin=subprocess.PIPE)
    indices = np.unique(np.round(np.linspace(0,len(a["time"])-1,max(2,int(a["time"][-1]*15)+1))).astype(int))
    try:
        for index in indices:
            d.qpos[:],d.qvel[:] = a["qpos"][index],a["qvel"][index]
            robot.forward(d)
            renderer.update_scene(d,camera=camera)
            for arm,color in ((0,[0.,.8,.9,.9]),(1,[1.,.4,.1,.9])):
                geom = renderer.scene.geoms[renderer.scene.ngeom]
                mujoco.mjv_initGeom(geom,mujoco.mjtGeom.mjGEOM_SPHERE,np.array([.013,.013,.013]),
                    a["desired"][index,arm,:3,3],np.eye(3).ravel(),np.array(color))
                renderer.scene.ngeom += 1
            proc.stdin.write(renderer.render().tobytes())
        proc.stdin.close()
        if proc.wait()!=0: raise RuntimeError("video encoding failed")
    finally: renderer.close()
    return len(indices)


def main():
    summary = read_json(OUT/"summary.json")
    robot = RobotAdapter()
    compact = []
    for name,result in summary["results"].items():
        directory = OUT/name
        with np.load(directory/"native_trace.npz") as source: a={k:source[k] for k in source.files}
        plots(directory,a,result)
        count = video(robot,directory,a)
        p,r = pose_errors(a["ee"],a["desired"])
        idx = np.unique(np.r_[np.arange(0,len(a["time"]),100),len(a["time"])-1]).astype(int)
        compact.append(dict(name=name,status=result["status"],time=np.round(a["time"][idx],3).tolist(),
            c=np.round(a["skeleton_c"][idx],5).tolist(),r=np.round(a["skeleton_r"][idx],5).tolist(),
            base=np.round(a["base"][idx,:3],5).tolist(),tip=np.round(a["ee"][idx,:,:3,3],5).tolist(),
            goal=np.round(a["desired"][idx,:,:3,3],5).tolist(),p=np.round(p[idx]*1000,3).tolist(),
            angle=np.round(np.rad2deg(r[idx]),3).tolist(),
            domain=np.round(np.max(abs(a["q17"][idx,:10]),axis=1),5).tolist(),
            clearance=np.round(a["clearance"][idx]*1000,3).tolist()))
        atomic_json(directory/"visualization_provenance.json",dict(native_sha256=file_digest(directory/"native_trace.npz"),
            video_frames=count,video_sha256=file_digest(directory/"actual_motion.mp4"),
            source="saved actual native qpos/qvel; desired markers are reference, not measured tips",
            new_physics_steps=0))
        print("Rendered",name,flush=True)
    atomic_json(OUT/"visual_data.json",compact)
    template = (ROOT/"ff_prior_waypoint_domain/replay_template.html").read_text(encoding="utf-8")
    inline = template.replace("/*DATA*/[]",json.dumps(compact,separators=(",",":")))
    visual_dir=Path("C:/Users/admin/.codex/visualizations/2026/10/20/01a15851-126e-7990-b674-2690d3b375bc")
    (visual_dir/"waypoint-domain.html").write_text(inline,encoding="utf-8")
    cards=[]
    for name,result in summary["results"].items():
        cards.append(f'<h2>{name} — {result["status"]}</h2><video controls width="720" src="{name}/actual_motion.mp4"></video>'
                     f'<p>圆点是期望末端位置。模型运动来自保存的真实力矩仿真状态。<a href="{name}/tracking.csv">完整500 Hz曲线CSV</a></p>'
                     f'<img width="100%" src="{name}/tracking.png"><img width="100%" src="{name}/world_paths.png">'
                     f'<img width="100%" src="{name}/joint_domain_and_torque.png">')
    (OUT/"index.html").write_text('<!doctype html><meta charset="utf-8"><title>非接触双臂动态场景</title>'
        '<style>body{font:16px system-ui;margin:24px auto;max-width:1200px}video,img{max-width:100%}</style>'
        '<h1>非接触接近、跟踪与保持：完整实测可视化</h1>'
        '<p>此为两个开发场景，不是最终TEST或硬件捕获实验。EXECUTION_FAILED的原因见各result.json；动画完整不代表全部物理门通过。</p>'
        +"\n".join(cards),encoding="utf-8")


if __name__=="__main__": main()
