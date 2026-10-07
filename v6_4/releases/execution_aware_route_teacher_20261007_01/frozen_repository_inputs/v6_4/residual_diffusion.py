"""One fixed-budget, masked 12-D residual DDPM; no simulation or proposal repair.

The inherited cosine100 / internal-v / DDIM20 convention is retained. Formal
training is a separate explicit call and cannot run without successful TRAIN
and VAL evidence. Generated meter coefficients are never clipped/projected.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn

from .dataset import object_sha, sha256
from .diffusion_model import (build_noise_schedule, noise_schedule_identity,
                              objective_identity, parameterization_identity, timestep_embedding)
from .residual_dataset import (ConditionNormalizer, ResidualNormalizer, _read, _write,
                                load_residual_dataset, validate_z)


@dataclass(frozen=True)
class ResidualDiffusionConfig:
    condition_dim: int
    latent_dim: int = 12
    hidden_dim: int = 128
    hidden_layers: int = 2
    diffusion_steps: int = 100
    ddim_steps: int = 20
    noise_schedule: str = "cosine"
    parameterization: str = "epsilon_residual"
    objective: str = "v_mse"
    cosine_s: float = .008
    beta_cap: float = .999
    beta_start: float = 1e-4
    beta_end: float = .02

    def __post_init__(self):
        if self.condition_dim < 1 or (self.latent_dim, self.hidden_dim, self.hidden_layers, self.diffusion_steps, self.ddim_steps) != (12,128,2,100,20):
            raise ValueError("fixed B.2 architecture: 12 latent, two 128 hidden layers, cosine100/DDIM20")
        if (self.noise_schedule, self.parameterization, self.objective, self.cosine_s, self.beta_cap) != ("cosine", "epsilon_residual", "v_mse", .008, .999):
            raise ValueError("fixed inherited noise and loss convention")

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, value):
        return cls(**value)


def _mask(mask, batch, device):
    m = torch.as_tensor(mask, device=device)
    if m.shape == (batch, 6):
        m = m.repeat_interleave(2, dim=-1)
    if m.shape != (batch, 12) or m.dtype != torch.bool or torch.any(~m.any(-1)):
        raise ValueError("nonempty boolean interval mask [B,6] or dimension mask [B,12] required")
    if torch.any(m[:, 0::2] != m[:, 1::2]):
        raise ValueError("both transverse coefficients share the interval mask")
    return m


class ResidualDDPM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.denoiser = nn.Sequential(nn.Linear(12+128+config.condition_dim+12, 128), nn.SiLU(),
                                     nn.Linear(128,128), nn.SiLU(), nn.Linear(128,12))
        betas, alpha_bars = build_noise_schedule(config)
        self.register_buffer("betas", betas)
        self.register_buffer("alpha_bars", alpha_bars)

    def _inputs(self, x, timesteps, condition, mask):
        b = x.shape[0]
        if x.shape != (b,12) or condition.shape != (b,self.config.condition_dim) or not torch.isfinite(x).all() or not torch.isfinite(condition).all():
            raise ValueError("finite latent [B,12] and declared condition [B,C] required")
        if timesteps.shape != (b,) or timesteps.dtype not in (torch.int32,torch.int64) or torch.any((timesteps<0)|(timesteps>=100)):
            raise ValueError("integer timestep outside cosine100")
        return _mask(mask,b,x.device)

    def forward(self, noisy, timesteps, condition, mask):
        m = self._inputs(noisy,timesteps,condition,mask)
        x = torch.where(m,noisy,0.)
        return self.denoiser(torch.cat([x,timestep_embedding(timesteps,128),condition,m.to(x.dtype)],-1))*m

    def q_sample(self, x0, timesteps, epsilon, mask):
        m = _mask(mask,len(x0),x0.device)
        if x0.shape != (len(x0),12) or epsilon.shape != x0.shape or not torch.isfinite(x0).all() or not torch.isfinite(epsilon).all():
            raise ValueError("finite matching 12-D clean/noise required")
        if torch.any(x0[~m] != 0):
            raise ValueError("invalid clean latent must be exactly zero")
        if timesteps.shape != (len(x0),) or timesteps.dtype not in (torch.int32,torch.int64) or torch.any((timesteps<0)|(timesteps>=100)):
            raise ValueError("integer timestep outside cosine100")
        a = self.alpha_bars[timesteps,None]
        return torch.where(m,a.sqrt()*x0+(1-a).sqrt()*epsilon,0.)

    def prediction(self, noisy, timesteps, condition, mask):
        m = self._inputs(noisy,timesteps,condition,mask)
        xt = torch.where(m,noisy,0.)
        v = self(xt,timesteps,condition,m)
        a = self.alpha_bars[timesteps,None]
        epsilon = (1-a).sqrt()*xt+a.sqrt()*v
        x0 = a.sqrt()*xt-(1-a).sqrt()*v
        return {"v": v, "epsilon": torch.where(m,epsilon,0.), "x0": torch.where(m,x0,0.)}

    def loss(self, x0, timesteps, epsilon, condition, mask, *, reduction="mean"):
        m = _mask(mask,len(x0),x0.device)
        xt = self.q_sample(x0,timesteps,epsilon,m)
        pred = self(xt,timesteps,condition,m)
        a = self.alpha_bars[timesteps,None]
        target = torch.where(m,a.sqrt()*epsilon-(1-a).sqrt()*x0,0.)
        losses = ((pred-target).square()*m).sum(-1)/m.sum(-1)
        if reduction not in ("mean","none"):
            raise ValueError("unsupported reduction")
        return losses.mean() if reduction == "mean" else losses

    @torch.no_grad()
    def sample_ddim(self, condition, mask, *, initial_noise):
        b = len(condition)
        m = _mask(mask,b,condition.device)
        x = torch.as_tensor(initial_noise,device=condition.device,dtype=condition.dtype)
        if x.shape != (b,12) or not torch.isfinite(x).all():
            raise ValueError("finite declared initial noise [B,12] required")
        x = torch.where(m,x,0.)
        steps = torch.linspace(99,0,20,device=x.device).round().to(torch.long)
        for i,t in enumerate(steps):
            ts = torch.full((b,),int(t),device=x.device,dtype=torch.long)
            prediction = self.prediction(x,ts,condition,m)
            previous = self.alpha_bars[steps[i+1]] if i+1<len(steps) else x.new_tensor(1.)
            x = torch.where(m,previous.sqrt()*prediction["x0"]+(1-previous).sqrt()*prediction["epsilon"],0.)
        return x


TRAINING_DEFAULTS = {"seed":64201,"draw_seed":64202,"validation_seed":64203,
    "optimizer_updates":4000,"batch_size":32,"validate_every":250,"curve_every":50,
    "validation_draws_per_successful_VAL_reference":16,
    "optimizer":{"name":"AdamW","lr":1e-4,"weight_decay":.01,"gradient_norm_clip":1.}}


def _state_sha(model):
    h = hashlib.sha256()
    for name,tensor in sorted(model.state_dict().items()):
        h.update(name.encode()+b"\0"); h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def _source_identity():
    root = Path(__file__).resolve().parents[1]
    paths = ("v6_4/residual_dataset.py","v6_4/residual_diffusion.py","v6_4/task_anchored_reference.py",
             "v6_4/diffusion_model.py","v6_4/dataset.py","v6_4/task_protocol.py",
             "v6_4/reference_adapter.py","v6_lite/irregular_waypoints.py")
    return {str(root/p):sha256(root/p) for p in paths}


def _runtime(device):
    return {"torch":torch.__version__,"cuda_runtime":torch.version.cuda,"cudnn":torch.backends.cudnn.version(),
            "device":str(device),"gpu":torch.cuda.get_device_name(device) if torch.device(device).type=="cuda" else None,
            "matmul_allow_tf32":torch.backends.cuda.matmul.allow_tf32,"cudnn_allow_tf32":torch.backends.cudnn.allow_tf32,
            "deterministic_algorithms":torch.are_deterministic_algorithms_enabled(),
            "cudnn_deterministic":torch.backends.cudnn.deterministic,"cudnn_benchmark":torch.backends.cudnn.benchmark,
            "CUBLAS_WORKSPACE_CONFIG":os.environ.get("CUBLAS_WORKSPACE_CONFIG")}


def _configure():
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG",":4096:8")
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    torch.use_deterministic_algorithms(True)


def prepare_training(dataset_manifest, output, *, plan=None, device="cpu"):
    """Freeze protocol/scalers/draws only. Never calls an optimizer or physics."""
    output=Path(output)
    if (output/"training_config.json").exists() or (output/"training_status.json").exists():
        raise FileExistsError("training preparation already exists")
    ds=load_residual_dataset(dataset_manifest)
    if not len(ds.indices("train")) or not len(ds.indices("val")):
        status={"status":"TRAINING_NOT_RUN","reason":"successful nonzero full-task TRAIN and VAL labels required",
                "counts":ds.manifest["counts"],"data_status":ds.manifest["data_status"],"checkpoint_created":False}
        _write(output/"training_status.json",status)
        return status
    protocol=json.loads(json.dumps(TRAINING_DEFAULTS))
    p=_read(plan) if plan is not None else {}
    supplied=p.get("training",p)
    for key,value in TRAINING_DEFAULTS.items():
        if key in supplied and supplied[key] != value:
            raise ValueError(f"frozen B.2 training field differs: {key}")
    train=ds.indices("train")
    tids=sorted({ds.samples[i]["task_id"] for i in train})
    cn=ConditionNormalizer.fit([(ds.tasks[t],ds.definitions[t]) for t in tids])
    zn=ResidualNormalizer.fit(ds.z_m[train],ds.masks[train])
    _write(output/"condition_normalizer.json",cn.to_dict())
    _write(output/"residual_normalizer.json",zn.to_dict())
    gen=torch.Generator(device="cpu").manual_seed(protocol["validation_seed"])
    refs=np.repeat(ds.indices("val"),16)
    times=torch.randint(100,(len(refs),),generator=gen).numpy()
    epsilon=torch.randn((len(refs),12),generator=gen).numpy()
    masks=np.repeat(ds.masks[refs],2,axis=1)
    epsilon=np.where(masks,epsilon,0.)
    output.mkdir(parents=True,exist_ok=True)
    np.savez(output/"validation_draws.npz",reference_indices=refs,timesteps=times,epsilon=epsilon)
    _configure()
    config={"schema":"v6_4_residual_training_protocol_v1",**protocol,
            "model":ResidualDiffusionConfig(len(cn.mean)).to_dict(),"device":str(device),"runtime":_runtime(device),
            "dataset_manifest_path":str(Path(dataset_manifest).resolve()),"dataset_manifest_sha256":sha256(dataset_manifest),
            "source_files":_source_identity(),"plan_sha256":object_sha(p),
            "normalizer_sha256":{n:sha256(output/n) for n in ("condition_normalizer.json","residual_normalizer.json")},
            "validation_draws_sha256":sha256(output/"validation_draws.npz"),
            "sampling_distribution":"uniform successful TRAIN task, then uniform successful reference within that task",
            "validation_reduction":"mean fixed draws per reference, mean references per task, mean VAL tasks",
            "selection":"minimum fixed VAL v_mse; exact ties select earlier update",
            "fresh_initialization":True,"old_weights_loaded":False,"data_status":ds.manifest["data_status"],
            "noise":noise_schedule_identity(ResidualDiffusionConfig(len(cn.mean))),
            "parameterization":parameterization_identity(ResidualDiffusionConfig(len(cn.mean))),
            "objective":objective_identity(ResidualDiffusionConfig(len(cn.mean))),
            "physics_steps":0,"postprocessing":"none; inverse TRAIN scaler only; downstream plan rejects invalid bounds"}
    _write(output/"training_config.json",config)
    return {"status":"READY","config":config,"checkpoint_created":False}


def load_training_bundle(output):
    output=Path(output)
    c=_read(output/"training_config.json")
    if sha256(c["dataset_manifest_path"]) != c["dataset_manifest_sha256"]:
        raise ValueError("dataset manifest changed after training freeze")
    for path,expected in c["source_files"].items():
        if sha256(path)!=expected:
            raise ValueError(f"training source changed: {path}")
    for name,expected in c["normalizer_sha256"].items():
        if sha256(output/name)!=expected:
            raise ValueError("normalizer changed")
    if sha256(output/"validation_draws.npz")!=c["validation_draws_sha256"]:
        raise ValueError("fixed VAL draws changed")
    ds=load_residual_dataset(c["dataset_manifest_path"])
    return {"config":c,"dataset":ds,"condition_normalizer":ConditionNormalizer.from_dict(_read(output/"condition_normalizer.json")),
            "residual_normalizer":ResidualNormalizer.from_dict(_read(output/"residual_normalizer.json")),
            "validation_draws":dict(np.load(output/"validation_draws.npz",allow_pickle=False))}


def make_conditions(tasks,definitions,normalizer,device="cpu"):
    return torch.as_tensor(np.stack([normalizer.transform(t,d) for t,d in zip(tasks,definitions)]),device=device)


@torch.no_grad()
def _validation(model,clean,conditions,masks,draws,ds,device):
    refs=draws["reference_indices"]
    losses=[]
    for start in range(0,len(refs),256):
        idx=torch.as_tensor(refs[start:start+256],device=device)
        t=torch.as_tensor(draws["timesteps"][start:start+256],device=device,dtype=torch.long)
        eps=torch.as_tensor(draws["epsilon"][start:start+256],device=device)
        losses.extend(model.loss(clean[idx],t,eps,conditions[idx],masks[idx],reduction="none").cpu().tolist())
    by_ref={int(i):float(np.mean(np.asarray(losses)[refs==i])) for i in np.unique(refs)}
    by_task={tid:float(np.mean([v for i,v in by_ref.items() if ds.samples[i]["task_id"]==tid]))
             for tid in sorted({ds.samples[i]["task_id"] for i in refs})}
    return float(np.mean(list(by_task.values()))),{"per_reference":by_ref,"per_task":by_task}


def train_model(output):
    """Exactly one fresh 4000-update run; root invokes after source freeze."""
    output=Path(output)
    if (output/"training_status.json").exists() and _read(output/"training_status.json")["status"]=="TRAINING_NOT_RUN":
        return _read(output/"training_status.json")
    model_dir=output/"model"
    if model_dir.exists():
        raise FileExistsError("run already started; implicit resume/retraining is forbidden")
    bundle=load_training_bundle(output)
    c,ds=bundle["config"],bundle["dataset"]
    _configure(); device=torch.device(c["device"])
    if _runtime(device)!=c["runtime"]:
        raise ValueError("runtime differs from prepared protocol")
    torch.manual_seed(c["seed"])
    if device.type=="cuda": torch.cuda.manual_seed_all(c["seed"])
    model=ResidualDDPM(ResidualDiffusionConfig.from_dict(c["model"])).to(device)
    initial_sha=_state_sha(model)
    opt=torch.optim.AdamW(model.parameters(),lr=c["optimizer"]["lr"],weight_decay=c["optimizer"]["weight_decay"])
    cn,zn=bundle["condition_normalizer"],bundle["residual_normalizer"]
    clean=torch.as_tensor(np.stack([zn.normalize(z,m).reshape(12) for z,m in zip(ds.z_m,ds.masks)]),device=device,dtype=torch.float32)
    conditions=make_conditions([ds.tasks[s["task_id"]] for s in ds.samples],[ds.definitions[s["task_id"]] for s in ds.samples],cn,device)
    masks=torch.as_tensor(ds.masks,device=device)
    tids=sorted({ds.samples[i]["task_id"] for i in ds.indices("train")})
    by_task={tid:[int(i) for i in ds.indices("train") if ds.samples[i]["task_id"]==tid] for tid in tids}
    gen=torch.Generator(device="cpu").manual_seed(c["draw_seed"])
    model_dir.mkdir(parents=True)
    curves=model_dir/"curves.jsonl"
    config_sha=sha256(output/"training_config.json")
    draw_sha=hashlib.sha256(); exposures={s["sample_id"]:0 for s in ds.samples}; best=float("inf"); selected=None
    start=time.perf_counter()
    last_loss=None

    def checkpoint(update,val,details):
        return {"schema":"v6_4_residual_ddpm_checkpoint_v1","model_config":c["model"],"state_dict":model.state_dict(),
                "training_config":c,"training_config_sha256":config_sha,"condition_normalizer":cn.to_dict(),
                "residual_normalizer":zn.to_dict(),"update":update,"validation_loss":val,"validation_details":details,
                "selection":c["selection"],"initial_state_sha256":initial_sha,"state_sha256":_state_sha(model),
                "seed":c["seed"],"fresh_initialization":True,"old_weights_loaded":False,
                "training_draws_sha256":draw_sha.hexdigest(),"sample_exposures":dict(exposures),"physics_steps":0}

    def log(value):
        with curves.open("a",encoding="utf-8") as f: f.write(json.dumps(value,sort_keys=True,allow_nan=False)+"\n")

    model.train()
    for update in range(1,c["optimizer_updates"]+1):
        task_draw=torch.randint(len(tids),(c["batch_size"],),generator=gen).tolist()
        refs=np.asarray([by_task[tids[j]][int(torch.randint(len(by_task[tids[j]]),(1,),generator=gen))] for j in task_draw],dtype=np.int64)
        times=torch.randint(100,(c["batch_size"],),generator=gen)
        eps=torch.randn((c["batch_size"],12),generator=gen)
        eps*=torch.as_tensor(np.repeat(ds.masks[refs],2,axis=1))
        for a in (refs,times.numpy(),eps.numpy()): draw_sha.update(a.tobytes())
        for i in refs: exposures[ds.samples[i]["sample_id"]]+=1
        idx=torch.as_tensor(refs,device=device)
        opt.zero_grad(set_to_none=True)
        loss=model.loss(clean[idx],times.to(device),eps.to(device),conditions[idx],masks[idx])
        if not torch.isfinite(loss): raise FloatingPointError("nonfinite training loss")
        loss.backward()
        grad=torch.nn.utils.clip_grad_norm_(model.parameters(),c["optimizer"]["gradient_norm_clip"],error_if_nonfinite=True)
        opt.step(); last_loss=float(loss.detach())
        if update%c["curve_every"]==0:
            log({"kind":"train","update":update,"loss":last_loss,"gradient_norm_before_clip":float(grad),"elapsed_s":time.perf_counter()-start})
        if update%c["validate_every"]==0:
            load_training_bundle(output)
            if sha256(output/"training_config.json")!=config_sha: raise ValueError("training config changed live")
            model.eval()
            val,details=_validation(model,clean,conditions,masks,bundle["validation_draws"],ds,device)
            if not np.isfinite(val): raise FloatingPointError("nonfinite fixed VAL loss")
            improved=val<best
            if improved:
                best,selected=val,update
                torch.save(checkpoint(update,val,details),model_dir/"selected.pt")
            log({"kind":"validation","update":update,"loss":val,**details,"selected":improved,"elapsed_s":time.perf_counter()-start})
            model.train()
    if device.type=="cuda": torch.cuda.synchronize(device)
    load_training_bundle(output)
    final=checkpoint(c["optimizer_updates"],val,details)
    final["optimizer_state_dict"]=opt.state_dict()
    torch.save(final,model_dir/"last.pt")
    report={"status":"COMPLETED","optimizer_updates":c["optimizer_updates"],"batch_size":c["batch_size"],
            "sample_draws":sum(exposures.values()),"sample_exposures":exposures,"selected_update":selected,
            "selected_validation_loss":best,"final_training_loss":last_loss,"elapsed_s":time.perf_counter()-start,
            "initial_state_sha256":initial_sha,"final_state_sha256":_state_sha(model),"training_draws_sha256":draw_sha.hexdigest(),
            "selected_checkpoint_sha256":sha256(model_dir/"selected.pt"),"last_checkpoint_sha256":sha256(model_dir/"last.pt"),
            "training_config_sha256":config_sha,"runtime":c["runtime"],"data_status":c["data_status"],
            "fresh_initialization":True,"old_weights_loaded":False,"physics_steps":0,"test_used_for_selection":False}
    _write(model_dir/"training_report.json",report)
    return report


class ResidualSampler:
    def __init__(self,checkpoint,device="cpu"):
        _configure()
        self.checkpoint_path=Path(checkpoint)
        self.checkpoint_sha256=sha256(checkpoint)
        self.checkpoint=torch.load(checkpoint,map_location="cpu",weights_only=False)
        c=self.checkpoint
        if c["schema"]!="v6_4_residual_ddpm_checkpoint_v1": raise ValueError("unsupported residual checkpoint")
        for path,expected in c["training_config"]["source_files"].items():
            if sha256(path)!=expected: raise ValueError("checkpoint production source identity changed")
        self.device=torch.device(device)
        self.model=ResidualDDPM(ResidualDiffusionConfig.from_dict(c["model_config"])).to(self.device)
        self.model.load_state_dict(c["state_dict"],strict=True); self.model.eval()
        if _state_sha(self.model)!=c["state_sha256"]: raise ValueError("checkpoint tensor identity mismatch")
        self.condition_normalizer=ConditionNormalizer.from_dict(c["condition_normalizer"])
        self.residual_normalizer=ResidualNormalizer.from_dict(c["residual_normalizer"])

    def condition(self,task,definition):
        return make_conditions([task],[definition],self.condition_normalizer,self.device)

    @torch.no_grad()
    def sample(self,task,definition,initial_noise_np):
        from .task_anchored_reference import TaskAnchoredResidualPlan
        TaskAnchoredResidualPlan.from_definition(definition,np.zeros((6,2)))
        noise=np.asarray(initial_noise_np,dtype=np.float32)
        if noise.shape not in ((12,),(6,2)) or not np.isfinite(noise).all(): raise ValueError("finite initial latent12 required")
        mask=np.asarray(definition["interval_mask"],dtype=bool)
        if not definition.get("applicable") or not mask.any(): raise ValueError("NOT_APPLICABLE")
        condition=self.condition(task,definition)
        if self.device.type=="cuda": torch.cuda.synchronize(self.device)
        start=time.perf_counter()
        normalized=self.model.sample_ddim(condition,torch.as_tensor(mask[None],device=self.device),initial_noise=noise.reshape(1,12))
        if self.device.type=="cuda": torch.cuda.synchronize(self.device)
        inference=time.perf_counter()-start
        z=self.residual_normalizer.inverse(normalized.cpu().numpy().astype(np.float64).reshape(6,2),mask)
        return z,{"method":"E2","checkpoint_sha256":self.checkpoint_sha256,"selected_update":self.checkpoint["update"],
                  "selected_validation_loss":self.checkpoint["validation_loss"],"inference_wall_s":inference,
                  "training_config_sha256":self.checkpoint["training_config_sha256"],
                  "noise_schedule_identity":self.checkpoint["training_config"]["noise"],
                  "parameterization":self.checkpoint["training_config"]["parameterization"],
                  "objective":self.checkpoint["training_config"]["objective"],
                  "initial_noise_sha256":hashlib.sha256(noise.tobytes()).hexdigest(),
                  "masked_initial_noise_sha256":hashlib.sha256(np.where(np.repeat(mask,2),noise.reshape(12),0.).tobytes()).hexdigest(),
                  "condition_scaler_sha256":object_sha(self.checkpoint["condition_normalizer"]),
                  "residual_scaler_sha256":object_sha(self.checkpoint["residual_normalizer"]),
                  "postprocessing":"inverse TRAIN residual scaler only; no clip/project/fallback","physics_steps":0}


def load_residual_sampler(checkpoint,device="cpu"):
    return ResidualSampler(checkpoint,device)


def main():
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument("action",choices=("prepare","train"))
    parser.add_argument("--output",required=True)
    parser.add_argument("--dataset")
    parser.add_argument("--plan")
    parser.add_argument("--device",default="cpu")
    args=parser.parse_args()
    if args.action=="prepare":
        if not args.dataset: parser.error("--dataset required for prepare")
        result=prepare_training(args.dataset,args.output,plan=args.plan,device=args.device)
    else: result=train_model(args.output)
    print(json.dumps(result,sort_keys=True,allow_nan=False))


if __name__=="__main__": main()
