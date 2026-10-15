"""Plot measured reference/native evidence and render one representative video."""
from pathlib import Path
import argparse
import subprocess
import numpy as np
from .storage import read_json,atomic_json
from .coupled_predictor import pose_errors


def visualize(run_dir,video=False):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    run_dir=Path(run_dir)
    selection=read_json(run_dir/"audit_selection.json")
    sample=selection["random"][0]
    row=read_json(run_dir/"records"/(sample+".json"))
    with np.load(run_dir/row["arrays"]) as data:
        reference={k:data[k] for k in data.files}
    with np.load(run_dir/"audits"/sample/"native_trace.npz") as data:
        actual={k:data[k] for k in data.files}
    out=run_dir/"visuals"
    out.mkdir(exist_ok=True)
    t=actual["time"]-actual["time"][0]
    rt=reference["prediction_phase"]*float(reference["T_ref"])
    p,r=pose_errors(actual["ee"],row["condition"]["goals"])
    fig,axes=plt.subplots(2,3,figsize=(13,7),constrained_layout=True)
    for arm in range(2):
        for axis,label in enumerate("xyz"):
            axes[0,arm].plot(rt,reference["prediction_ee"][:,arm,axis,3],ls="--",lw=1,label=f"reference {label}")
            axes[0,arm].plot(t,actual["ee"][:,arm,axis,3],lw=.8,label=f"actual {label}")
        axes[0,arm].set_title(("Continuum","Rigid")[arm]+" world tip (m)")
        axes[0,arm].legend(fontsize=7,ncol=2)
        axes[1,0].semilogy(t,np.maximum(p[:,arm],1e-12),label=("continuum","rigid")[arm])
    for axis,label in enumerate("xyz"):
        axes[0,2].plot(t,1000*(actual["base"][:,axis]-actual["base"][0,axis]),label=label)
        axes[0,2].plot(rt,1000*(reference["prediction_base"][:,axis]-reference["prediction_base"][0,axis]),ls="--",lw=.8)
    axes[0,2].set_title("Free-base translation (mm)"); axes[0,2].legend()
    axes[1,0].set_title("Distance to frozen goals (m)"); axes[1,0].legend()
    axes[1,1].plot(t,actual["clearance_m"]*1000)
    axes[1,1].axhline(0,color="red",lw=.7)
    axes[1,1].set_title("Native full-body clearance (mm)")
    axes[1,2].plot(t,np.max(abs(actual["manifold_residual"]),axis=1))
    axes[1,2].set_title("Native manifold residual (rad)")
    for ax in axes.ravel():
        ax.set_xlabel("time (s)"); ax.grid(alpha=.2)
    fig.suptitle(f"V7.1 measured DEV audit: {sample}; reference dashed, actual solid")
    fig.savefig(out/"reference_vs_actual.png",dpi=160)
    plt.close(fig)
    records=[read_json(p) for p in (run_dir/"records").glob("*.json")]
    valid=[x for x in records if x["validation_level"]=="MODEL_VALID"]
    fig,ax=plt.subplots(1,2,figsize=(10,4),constrained_layout=True)
    for partition,color in (("main","tab:blue"),("dev","tab:orange")):
        goals=np.asarray([x["condition"]["goals"] for x in valid if x["source_partition"]==partition])
        if len(goals):
            for arm in range(2):
                ax[arm].scatter(goals[:,arm,0,3],goals[:,arm,1,3],s=8,alpha=.5,label=partition,color=color)
    for arm in range(2):
        ax[arm].set_title(("Continuum","Rigid")[arm]+" goal coverage, world XY")
        ax[arm].set_xlabel("x (m)"); ax[arm].set_ylabel("y (m)"); ax[arm].legend(); ax[arm].grid(alpha=.2)
    fig.savefig(out/"goal_coverage.png",dpi=160); plt.close(fig)
    evidence={"sample_id":sample,"trace":"audits/"+sample+"/native_trace.npz","video":"NOT_REQUESTED"}
    if video:
        import mujoco
        from .robot_adapter import RobotAdapter
        from .scenes import SceneSpec
        robot=RobotAdapter(scene=SceneSpec.from_dict(read_json(run_dir/"scene.json"))); data=robot.new_data()
        renderer=mujoco.Renderer(robot.model,height=480,width=640)
        camera=mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(camera)
        camera.lookat[:]=[0,0,.2]
        camera.distance=4.2; camera.azimuth=135; camera.elevation=-25
        process=subprocess.Popen(["ffmpeg","-y","-loglevel","error","-f","rawvideo","-pix_fmt","rgb24",
            "-s","640x480","-r","15","-i","-","-an","-vcodec","libx264","-pix_fmt","yuv420p",
            str(out/"representative_actual.mp4")],stdin=subprocess.PIPE)
        try:
            for index in np.linspace(0,len(t)-1,90).round().astype(int):
                data.qpos[:]=actual["qpos"][index]
                data.qvel[:]=actual["qvel"][index]
                mujoco.mj_forward(robot.model,data)
                renderer.update_scene(data,camera=camera)
                process.stdin.write(renderer.render().tobytes())
            process.stdin.close()
            code=process.wait()
            if code: raise RuntimeError(f"ffmpeg exited {code}")
        finally:
            renderer.close()
        evidence["video"]="representative_actual.mp4"
        evidence["video_source"]="native saved qpos replay for visualization only; no physics certification inferred from rendering"
    atomic_json(out/"provenance.json",evidence)
    return evidence


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--run-dir",required=True)
    parser.add_argument("--video",action="store_true")
    args=parser.parse_args()
    print(visualize(args.run_dir,args.video))
