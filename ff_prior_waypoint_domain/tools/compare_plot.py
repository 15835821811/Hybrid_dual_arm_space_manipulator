from .analyze import ROOT,OUT,OLD
from ff_prior.coupled_predictor import pose_errors
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    fig,axes=plt.subplots(3,2,figsize=(13,10),layout="constrained")
    styles=[(OLD,"task_controller","old baseline","tab:blue",":"),
        (OUT,"task_controller","fixed baseline","tab:blue","-"),
        (OLD,"diffusion_null_prior","old prior","tab:orange",":"),
        (OUT,"diffusion_null_prior","fixed prior","tab:orange","-")]
    for col,scene in enumerate(("stationary","drifting")):
        for directory,method,label,color,linestyle in styles:
            with np.load(directory/scene/method/"native_trace.npz") as a:
                p,_=pose_errors(a["ee"],a["desired"]);t=a["time"]
                axes[0,col].plot(t,np.max(abs(a["q17"][:,:10]),axis=1),label=label,color=color,ls=linestyle)
                for arm in range(2):
                    axes[arm+1,col].semilogy(t,np.maximum(p[:,arm]*1000,1e-6),label=label,color=color,ls=linestyle)
        axes[0,col].axhline(1,color="red",ls="--");axes[0,col].set(title=scene,ylabel="max continuum |q| (rad)")
        for row,arm in ((1,"Continuum"),(2,"Rigid")):
            axes[row,col].axhline(5,color="red",ls="--")
            axes[row,col].set_ylabel(arm+" position error (mm)")
        for row in range(3):axes[row,col].grid(alpha=.2);axes[row,col].set_xlabel("simulation time (s)")
    axes[0,0].legend(ncol=2,fontsize=8)
    fig.savefig(OUT/"repair_comparison.png",dpi=160);plt.close(fig)
    index=OUT/"index.html"
    text=index.read_text(encoding="utf-8")
    text += '<h2>修复前后配对曲线</h2><p>虚线旧版、实线修复版；蓝色基线、橙色先验。相同冻结任务与先验；低于1rad工作域门不等于先验具有优势。</p><img width="100%" src="repair_comparison.png">'
    index.write_text(text,encoding="utf-8")


if __name__=="__main__":main()
