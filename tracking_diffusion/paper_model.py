"""B-spline conditional Transformer; constraints enter during reverse sampling.

The paper's conditional AdaLN / DDIM / late guidance workflow is retained.
Continuous v-prediction is an explicit numerical adaptation of noise prediction;
this is an engineering implementation for the dual-arm model, not a reproduction.
"""
import os
for _name in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ[_name] = "1"

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn

from .model import schedule
from .pipeline import write_json
from .spline import SplineChunk,basis_spline


class Block(nn.Module):
    def __init__(self, width=128):
        super().__init__()
        self.norm1 = nn.LayerNorm(width, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(width, elementwise_affine=False)
        self.attention = nn.MultiheadAttention(width, 4, batch_first=True)
        self.mlp = nn.Sequential(nn.Linear(width, 4*width), nn.GELU(), nn.Linear(4*width, width))
        self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(width, 6*width))
        nn.init.zeros_(self.modulation[-1].weight)
        nn.init.zeros_(self.modulation[-1].bias)

    def forward(self, x, condition):
        s1, a1, g1, s2, a2, g2 = self.modulation(condition)[:,None].chunk(6, dim=-1)
        z = self.norm1(x)*(1+a1)+s1
        x = x+g1*self.attention(z,z,z,need_weights=False)[0]
        return x+g2*self.mlp(self.norm2(x)*(1+a2)+s2)


class SplineTransformer(nn.Module):
    def __init__(self, condition_dim=136, width=128, tokens=6):
        super().__init__()
        self.input = nn.Linear(17,width)
        self.position = nn.Parameter(torch.randn(1,tokens,width)*.02)
        self.condition = nn.Sequential(nn.Linear(condition_dim,width),nn.SiLU(),nn.Linear(width,width))
        self.time = nn.Sequential(nn.Linear(64,width),nn.SiLU(),nn.Linear(width,width))
        self.blocks = nn.ModuleList([Block(width) for _ in range(3)])
        self.output = nn.Sequential(nn.LayerNorm(width),nn.Linear(width,17))
        self.register_buffer("frequency",torch.exp(torch.linspace(0,-math.log(10000),32)))

    def forward(self,x,t,condition):
        phase=t[:,None]*1000*self.frequency[None]
        c=self.condition(condition)+self.time(torch.cat([phase.sin(),phase.cos()],-1))
        h=self.input(x)+self.position
        for block in self.blocks:
            h=block(h,c)
        return self.output(h)


class CudaGraphModel:
    """Replay the same trained forward pass with fixed-size sampling inputs."""
    def __init__(self,model,batch,condition_dim):
        self.x=torch.zeros((batch,6,17),device="cuda")
        self.t=torch.full((batch,),.5,device="cuda")
        self.c=torch.zeros((batch,condition_dim),device="cuda")
        stream=torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream),torch.no_grad():
            for _ in range(3): model(self.x,self.t,self.c)
        torch.cuda.current_stream().wait_stream(stream)
        self.graph=torch.cuda.CUDAGraph()
        with torch.no_grad(),torch.cuda.graph(self.graph):
            self.output=model(self.x,self.t,self.c)

    def __call__(self,x,t,condition):
        self.x.copy_(x)
        self.t.copy_(t)
        self.c.copy_(condition)
        self.graph.replay()
        return self.output


@torch.no_grad()
def sample(model,condition,generator,steps=60,guidance=None,on_step=None,on_clean=None,callback_stride=1):
    x=torch.randn((len(condition),6,17),device=condition.device,generator=generator)
    times=torch.linspace(1.,0.,steps+1,device=condition.device)
    if on_step:
        on_step(0,steps,x.detach().cpu().numpy())
    for k in range(steps):
        a,s=schedule(times[k].expand(len(x)))
        a,s=a[:,:,None],s[:,:,None]
        v=model(x,times[k].expand(len(x)),condition)
        clean=a*x-s*v
        eps=s*x+a*v
        # Apply task/constraint gradients to the clean estimate in the final
        # 30% of DDIM steps, then propagate that estimate into the next sample.
        if guidance is not None and k >= steps-int(math.ceil(.3*steps)):
            if on_clean and k == steps-int(math.ceil(.3*steps)):
                on_clean("before_late_guidance",clean.detach().cpu().numpy())
            with torch.enable_grad():
                clean=guidance(clean, (k+1-steps*.7)/(steps*.3))
        an,sn=schedule(times[k+1].expand(len(x)))
        x=an[:,:,None]*clean+sn[:,:,None]*eps
        if on_step and ((k+1)%callback_stride==0 or k==steps-1):
            on_step(k+1,steps,x.detach().cpu().numpy())
    return x


def train(dataset,output,steps=6000,seed=42,device="cuda",startup_fraction=0.,prefix_velocity_loss=0.):
    output.mkdir(parents=True,exist_ok=True)
    if (output/"model.pt").exists():
        raise FileExistsError("Use a new training run directory")
    from .evaluation import snapshot
    snapshot(output)
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    with np.load(dataset,allow_pickle=False) as data:
        tm,vm=data["split"]==0,data["split"]==1
        ct,cv=data["condition"][tm],data["condition"][vm]
        xt,xv=data["free"][tm],data["free"][vm]
        train_seeds=np.unique(data["seed"][tm]).tolist()
        val_seeds=np.unique(data["seed"][vm]).tolist()
        metadata=json.loads(data["metadata"].item())
        train_times=data["time"][tm]
        train_task_ids=data["seed"][tm]
        validation_times=data["time"][vm]
        train_horizons=data["horizon"][tm]
    if not len(ct) or not len(cv):
        raise ValueError("Both task-separated train and validation splits are needed")
    cm,cs=ct.mean(0),np.maximum(ct.std(0),.01)
    xm,xs=xt.mean(0),np.maximum(xt.std(0),.02)
    tensor=lambda value:torch.as_tensor(value,dtype=torch.float32,device=device)
    c,x,vc,vx=map(tensor,[(ct-cm)/cs,(xt-xm)/xs,(cv-cm)/cs,(xv-xm)/xs])
    if prefix_velocity_loss<0: raise ValueError("Prefix loss weight must be nonnegative")
    if prefix_velocity_loss:
        hs=np.arange(1,6)*.1
        derivatives=np.array([basis_spline(8)(np.minimum(np.arange(1,6)*.02,h)/h,nu=1)[:,2:] for h in hs])
        derivative_basis=tensor(derivatives)
        horizon_ids=torch.as_tensor(np.rint(train_horizons*10).astype(int)-1,device=device)
    model=SplineTransformer(ct.shape[1]).to(device)
    ema=copy.deepcopy(model).eval()
    optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-5)
    rng=torch.Generator(device=device).manual_seed(seed+10)
    vrng=torch.Generator(device=device).manual_seed(seed+20)
    if not 0 <= startup_fraction < 1:
        raise ValueError("startup_fraction must lie in [0,1)")
    groups=[]
    if startup_fraction:
        for is_start in (True,False):
            members=[np.flatnonzero((train_task_ids==task)&((train_times<1.-1e-8)==is_start)) for task in train_seeds]
            if len({len(m) for m in members})!=1 or not len(members[0]):
                raise ValueError("Task-uniform sampling requires complete equal-size window coverage")
            groups.append(torch.as_tensor(np.stack(members),device=device))
    ids=torch.linspace(0,len(vx)-1,min(256,len(vx)),device=device).long()
    vn=torch.randn(vx[ids].shape,device=device,generator=vrng)
    vt=torch.rand(len(ids),device=device,generator=vrng)
    startup_ids=torch.as_tensor(np.flatnonzero(validation_times<1.-1e-8),device=device)
    startup_ids=startup_ids[torch.linspace(0,len(startup_ids)-1,min(256,len(startup_ids)),device=device).long()]
    startup_noise=torch.randn(vx[startup_ids].shape,device=device,generator=torch.Generator(device=device).manual_seed(seed+21))
    startup_t=torch.rand(len(startup_ids),device=device,generator=torch.Generator(device=device).manual_seed(seed+22))
    history,best,best_state=[],float("inf"),None
    started=time.perf_counter()
    metadata.update({"network":"3-layer 4-head AdaLN Transformer, width 128",
                     "prediction":"continuous cosine v", "paper_workflow":True})
    write_json(output/"training_config.json",{"dataset":str(dataset.resolve()),
        "dataset_sha256":hashlib.sha256(dataset.read_bytes()).hexdigest(),"steps":steps,
        "seed":seed,"device":device,"train_tasks":train_seeds,"validation_tasks":val_seeds,
        "train_windows":len(x),"validation_windows":len(vx),"metadata":metadata,"startup_fraction":startup_fraction,
        "prefix_velocity_loss_weight":prefix_velocity_loss})
    for step in range(1,steps+1):
        if startup_fraction:
            selected=[]
            for members,count in zip(groups,(round(128*startup_fraction),128-round(128*startup_fraction))):
                ti=torch.randint(len(members),(count,),device=device,generator=rng)
                wi=torch.randint(members.shape[1],(count,),device=device,generator=rng)
                selected.append(members[ti,wi])
            rows=torch.cat(selected)
        else:
            rows=torch.randint(len(x),(128,),device=device,generator=rng)
        clean,cond=x[rows],c[rows]
        t=torch.rand(len(rows),device=device,generator=rng)
        noise=torch.randn(clean.shape,device=device,generator=rng)
        a,s=schedule(t)
        a,s=a[:,:,None],s[:,:,None]
        verror=model(a*clean+s*noise,t,cond)-(a*noise-s*clean)
        loss=verror.square().mean()
        if prefix_velocity_loss:
            # qdot error = B'(t/h) * delta_free; fixed boundary controls cancel.
            # Division by per-coordinate output_scale yields this normalized
            # prefix velocity error. No future measured execution state is read.
            prefix_error=torch.einsum("bpk,bkj->bpj",derivative_basis[horizon_ids[rows]],s*verror)
            loss=loss+prefix_velocity_loss*prefix_error.square().mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(),1.)
        optimizer.step()
        for group in optimizer.param_groups:
            group["lr"]=3e-4*(.1+.9*.5*(1+math.cos(math.pi*step/steps)))
        with torch.no_grad():
            decay=min(.995,(step+1)/(step+10))
            for pe,p in zip(ema.parameters(),model.parameters()):
                pe.lerp_(p,1-decay)
        if step==1 or step%200==0 or step==steps:
            with torch.no_grad():
                a,s=schedule(vt)
                a,s=a[:,:,None],s[:,:,None]
                vl=(ema(a*vx[ids]+s*vn,vt,vc[ids])-(a*vn-s*vx[ids])).square().mean().item()
                sa,ss=schedule(startup_t)
                sa,ss=sa[:,:,None],ss[:,:,None]
                svl=(ema(sa*vx[startup_ids]+ss*startup_noise,startup_t,vc[startup_ids])-(sa*startup_noise-ss*vx[startup_ids])).square().mean().item()
            item={"step":step,"train_v_mse":loss.item(),"validation_v_mse":vl,
                  "startup_validation_v_mse":svl,"elapsed_s":time.perf_counter()-started}
            history.append(item)
            write_json(output/"progress.json",{"status":"training","history":history})
            print(json.dumps(item),flush=True)
            if vl<best:
                best,best_state=vl,copy.deepcopy(ema.state_dict())
    ema.load_state_dict(best_state)
    with torch.no_grad():
        generated=sample(ema,vc[ids],torch.Generator(device=device).manual_seed(seed+30))
        rmse=float(torch.sqrt(((generated-vx[ids])*tensor(xs)).square().mean()))
    ckpt={"version":3,"metadata":metadata,"state_dict":best_state,"condition_dim":ct.shape[1],
          "condition_mean":cm,"condition_scale":cs,"output_mean":xm,"output_scale":xs,
          "train_seeds":train_seeds,"validation_seeds":val_seeds,"history":history,
          "training_seed":seed,"training_steps":steps,"startup_fraction":startup_fraction,
          "prefix_velocity_loss_weight":prefix_velocity_loss}
    torch.save(ckpt,output/"model.tmp.pt")
    os.replace(output/"model.tmp.pt",output/"model.pt")
    report={"status":"trained","train_tasks":len(train_seeds),"validation_tasks":len(val_seeds),
            "train_windows":len(x),"validation_windows":len(vx),"steps":steps,
            "best_validation_v_mse":best,"sampling_control_rmse_rad_per_s":rmse,
            "wall_seconds":time.perf_counter()-started,"history":history,"startup_fraction":startup_fraction,
            "prefix_velocity_loss_weight":prefix_velocity_loss}
    write_json(output/"training_report.json",report)
    write_json(output/"progress.json",report)
    return report


class PaperPolicy:
    chunk_type=SplineChunk

    def __init__(self,path,device="cuda",seed=73,candidates=5,steps=60,guidance=True,
                 projection=None,scoring="legacy",continuity_weight=0.,diagnostic=False,cuda_graph=False):
        torch.set_num_threads(1)
        ckpt=torch.load(path,map_location=device,weights_only=False)
        if ckpt["version"]!=3:
            raise ValueError("Expected a paper-workflow Transformer checkpoint")
        self.model=SplineTransformer(ckpt["condition_dim"]).to(device).eval()
        self.model.load_state_dict(ckpt["state_dict"])
        self.device,self.count,self.steps=device,candidates,steps
        self.generator=torch.Generator(device=device).manual_seed(seed)
        self.cm,self.cs=ckpt["condition_mean"],ckpt["condition_scale"]
        self.xm,self.xs=ckpt["output_mean"],ckpt["output_scale"]
        self.metadata=dict(ckpt["metadata"],sampling_seed=seed,candidates=candidates,
                           sampling_steps=steps,guidance_enabled=guidance)
        self.task_domain=self.metadata["task_domain"]
        self.horizon=self.metadata["horizon"]
        self.train_seeds,self.val_seeds=ckpt["train_seeds"],ckpt["validation_seeds"]
        self.guidance_enabled=guidance
        self.projection_enabled=guidance if projection is None else projection
        self.scoring=scoring
        self.continuity_weight=continuity_weight
        self.diagnostic=diagnostic
        self.previous_acceleration=None
        self.last_stages={}
        self.telemetry_stride=1
        self.sampling_model=self.model
        if cuda_graph:
            if device!="cuda": raise ValueError("CUDA graph requires CUDA device")
            self.sampling_model=CudaGraphModel(self.model,candidates,ckpt["condition_dim"])
        self.metadata["cuda_graph"]=cuda_graph
        self.metadata.update(projection_enabled=self.projection_enabled,scoring=scoring,continuity_weight=continuity_weight)
        self.env=None
        self.last_guidance=[]

    def bind_environment(self,env):
        self.env=env

    def predict(self,condition,horizon=None,time=None,on_step=None):
        h=self.horizon if horizon is None else horizon
        condition=np.r_[condition,h].astype(np.float32)
        c=torch.as_tensor(np.repeat(((condition-self.cm)/self.cs)[None],self.count,axis=0),device=self.device)
        guide=None
        if self.guidance_enabled or self.projection_enabled:
            from .guidance import ConstraintGuidance
            if self.env is None:
                raise ValueError("Guided generation requires the measured robot environment")
            guide=ConstraintGuidance(self.env,h,self.xm,self.xs,self.device,
                project=self.projection_enabled,diagnostic=self.diagnostic)
        self.last_stages={}
        def clean_stage(name,value):
            self.last_stages[name]=value*self.xs+self.xm
        def callback(step,total,value):
            on_step(step,total,value*self.xs+self.xm)
        x=sample(self.sampling_model,c,self.generator,self.steps,guide if self.guidance_enabled else None,
            callback if on_step else None,clean_stage if self.diagnostic else None,self.telemetry_stride).cpu().numpy()
        if self.projection_enabled and not self.guidance_enabled:
            free=x*self.xs+self.xm
            corrected=np.array([guide.project_joint_bounds(f) for f in free])
            x=(corrected-self.xm)/self.xs
            if self.diagnostic:
                self.last_stages.update(after_gradient=free,after_projection=corrected)
        if guide and self.diagnostic:
            self.last_stages.update(guide.stages)
        self.last_guidance=guide.history if guide else []
        return x*self.xs+self.xm


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--steps",type=int,default=6000)
    parser.add_argument("--seed",type=int,default=42)
    parser.add_argument("--device",default="cuda")
    parser.add_argument("--startup-fraction",type=float,default=0.)
    parser.add_argument("--prefix-velocity-loss",type=float,default=0.)
    args=parser.parse_args()
    train(args.dataset,args.output,args.steps,args.seed,args.device,args.startup_fraction,args.prefix_velocity_loss)


if __name__=="__main__":
    main()
