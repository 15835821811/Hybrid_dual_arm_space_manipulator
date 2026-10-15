"""Small epsilon Transformer, TRAIN-only updates and exact-call DDIM."""
from pathlib import Path
import os
import time
import numpy as np
import torch
from torch import nn
from .contracts import SCHEMA, assert_frozen
from .storage import digest, atomic_json, read_json
from .dataset import Scaler


class NoiseTransformer(nn.Module):
    def __init__(self,condition_dim,hidden=128,layers=4,heads=4):
        super().__init__()
        self.input = nn.Linear(17,hidden)
        self.condition = nn.Linear(condition_dim,hidden)
        self.time = nn.Sequential(nn.Linear(32,hidden),nn.SiLU(),nn.Linear(hidden,hidden))
        self.position = nn.Parameter(torch.randn(1,19,hidden)*.01)
        block = nn.TransformerEncoderLayer(hidden,heads,hidden*4,dropout=0,batch_first=True,activation="gelu",norm_first=True)
        self.blocks = nn.TransformerEncoder(block,layers,enable_nested_tensor=False)
        self.output = nn.Linear(hidden,17)
        self.calls = 0

    def forward(self,z,t,condition):
        self.calls += 1
        freq = torch.exp(torch.arange(16,device=z.device)*(-np.log(10000)/15))
        phase = t[:,None]*freq[None]
        emb = self.time(torch.cat([phase.sin(),phase.cos()],dim=1))
        x = self.input(z)+self.position+emb[:,None]+self.condition(condition)[:,None]
        return self.output(self.blocks(x))


def schedule():
    return torch.cumprod(1-torch.linspace(1e-4,.02,1000),dim=0)


def weight_hash(model):
    import hashlib
    h = hashlib.sha256()
    for name,p in model.state_dict().items():
        h.update(name.encode())
        h.update(p.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def save_torch(path,payload):
    path = Path(path)
    tmp = path.with_name(path.name+".tmp")
    torch.save(payload,tmp)
    os.replace(tmp,path)


def train(run_dir,records,dataset,robot_hash,initial_state_id,updates=200,seed=20261015,software_fixture=False):
    """Resumes optimizer/RNG/update count; never trains on VAL, TEST or DEV."""
    setup_wall,setup_cpu = time.perf_counter(),time.process_time()
    run_dir = Path(run_dir)
    out = run_dir/"learning"
    out.mkdir(exist_ok=True)
    if dataset["status"] != "READY":
        result = {"status":"BLOCKED","reason":dataset["status"]}
        atomic_json(out/"training.json",result)
        return result
    by_id = {r["sample_id"]:r for r in records}
    if not software_fixture:
        for sample_id in dataset["eligible"]["TRAIN"]+dataset["eligible"]["VAL"]:
            row=by_id[sample_id]
            if row.get("validation_level")!="MODEL_VALID" or row.get("data_role")!="FF_MODEL_PRIOR" or row.get("execution_status") not in ("NOT_RUN","PASS"):
                raise ValueError("unqualified/known-failed training record")
    def load(ids):
        return np.stack([np.load(run_dir/by_id[i]["arrays"])["Z"] for i in ids]),[by_id[i]["condition"] for i in ids]
    z,conditions = load(dataset["eligible"]["TRAIN"])
    zv,cv = load(dataset["eligible"]["VAL"])
    scaler = Scaler.fit(z,conditions,["TRAIN"]*len(z),robot_hash,initial_state_id)
    scaler_path = out/"scaler.json"
    if scaler_path.exists():
        old = Scaler.load(scaler_path,robot_hash,initial_state_id)
        if not np.array_equal(old.z_mean,scaler.z_mean):
            raise ValueError("frozen training data changed")
    else:
        scaler.save(scaler_path)
    torch.set_num_threads(2)
    torch.manual_seed(seed)
    x = torch.tensor(scaler.z(z),dtype=torch.float32)
    c = torch.tensor(np.stack([scaler.condition(v) for v in conditions]),dtype=torch.float32)
    xv = torch.tensor(scaler.z(zv),dtype=torch.float32)
    vc = torch.tensor(np.stack([scaler.condition(v) for v in cv]),dtype=torch.float32)
    model = NoiseTransformer(c.shape[1])
    optimizer = torch.optim.AdamW(model.parameters(),lr=2e-4)
    original = weight_hash(model)
    metadata = {"robot_hash":robot_hash,"initial_state_id":initial_state_id,"schema_hash":digest(SCHEMA),
                "dataset_hash":digest(dataset),"scaler_hash":read_json(scaler_path)["hash"],
                "condition_dim":c.shape[1],"seed":seed,"hidden":128,"layers":4,"heads":4,
                "data_quality_scope":"SOFTWARE_FIXTURE_ONLY" if software_fixture else "REAL_MODEL_PRIOR"}
    start,best,history,previous_wall,previous_cpu,previous_calls = 0,float("inf"),[],0.,0.,0
    if (out/"progress.pt").exists():
        saved = torch.load(out/"progress.pt",weights_only=False,map_location="cpu")
        if saved["metadata"] != metadata:
            raise ValueError("training resume schema/data identity mismatch")
        model.load_state_dict(saved["weights"])
        optimizer.load_state_dict(saved["optimizer"])
        torch.set_rng_state(saved["rng"])
        start,best,history,original,previous_wall = (saved[k] for k in ("updates","best","history","original_hash","wall_s"))
        previous_cpu = saved.get("cpu_s",0.)
        previous_calls = saved.get("network_calls",0)
        if (out/"training.json").exists():
            prior_report=read_json(out/"training.json")
            previous_wall=max(previous_wall,prior_report.get("wall_s",0.))
            previous_cpu=max(previous_cpu,prior_report.get("cpu_s",0.))
    alpha = schedule()
    # Fixed VAL noise is generated independently of training RNG.
    vg = torch.Generator().manual_seed(seed+10)
    vt = torch.randint(0,1000,(len(xv),),generator=vg)
    vn = torch.randn(xv.shape,generator=vg)
    wall = time.perf_counter()
    for step in range(start,updates):
        model.train()
        ids = torch.randint(0,len(x),(min(16,len(x)),))
        t = torch.randint(0,1000,(len(ids),))
        noise = torch.randn_like(x[ids])
        a = alpha[t,None,None]
        prediction = model(a.sqrt()*x[ids]+(1-a).sqrt()*noise,t,c[ids])
        loss = torch.mean((prediction-noise)**2)
        if not torch.isfinite(loss):
            raise RuntimeError("nonfinite training loss")
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(),1.)
        optimizer.step()
        model.eval()
        with torch.no_grad():
            av = alpha[vt,None,None]
            val_loss = float(torch.mean((model(av.sqrt()*xv+(1-av).sqrt()*vn,vt,vc)-vn)**2))
        history.append({"update":step+1,"train_mse":float(loss.detach()),"val_mse":val_loss})
        if val_loss < best:
            best = val_loss
            save_torch(out/"model.pt",{"metadata":metadata,"weights":model.state_dict(),"update":step+1,"val_mse":best})
        save_torch(out/"progress.pt",{"metadata":metadata,"weights":model.state_dict(),"optimizer":optimizer.state_dict(),
            "rng":torch.get_rng_state(),"updates":step+1,"best":best,"history":history,
            "original_hash":original,"wall_s":previous_wall+time.perf_counter()-setup_wall,
            "cpu_s":previous_cpu+time.process_time()-setup_cpu,"network_calls":previous_calls+model.calls})
    changed = weight_hash(model) != original
    training_forward_calls=previous_calls+model.calls
    model,loaded_scaler = load_checkpoint(out,robot_hash,initial_state_id)
    probe = torch.zeros(1,19,17)
    cc = torch.tensor(loaded_scaler.condition(conditions[0]),dtype=torch.float32)[None]
    with torch.no_grad():
        y = model(probe,torch.tensor([123]),cc)
    reloaded,_ = load_checkpoint(out,robot_hash,initial_state_id)
    with torch.no_grad():
        reload_error = float(torch.max(abs(y-reloaded(probe,torch.tensor([123]),cc))))
    result = {"status":"PASS" if changed and reload_error == 0 else "FAIL","updates":updates,
              "weights_changed":changed,"initial_weight_hash":original,"selected_weight_hash":weight_hash(model),
              "reload_max_error":reload_error,"checkpoint_selection":"VAL_MSE_ONLY","history":history,
              "wall_s":previous_wall+time.perf_counter()-setup_wall,"cpu_s":previous_cpu+time.process_time()-setup_cpu,
              "train_count":len(x),"val_count":len(xv),"training_and_val_forward_calls":training_forward_calls,
              "reload_forward_calls":model.calls+reloaded.calls,
              "diffusion_benefit":"NOT_ESTABLISHED","metadata":metadata}
    atomic_json(out/"training.json",result)
    return result


def load_checkpoint(directory,robot_hash,initial_state_id):
    directory = Path(directory)
    scaler = Scaler.load(directory/"scaler.json",robot_hash,initial_state_id)
    saved = torch.load(directory/"model.pt",weights_only=False,map_location="cpu")
    metadata = saved["metadata"]
    if any(metadata.get(k) != v for k,v in {"robot_hash":robot_hash,"initial_state_id":initial_state_id,
        "schema_hash":digest(SCHEMA),"scaler_hash":read_json(directory/"scaler.json")["hash"]}.items()):
        raise ValueError("checkpoint schema/robot/scaler mismatch")
    model = NoiseTransformer(metadata["condition_dim"],metadata["hidden"],metadata["layers"],metadata["heads"])
    model.load_state_dict(saved["weights"])
    model.eval()
    return model,scaler


def ddim(model,scaler,task,steps,seed,guidance=None):
    assert_frozen(task)
    if not 1 <= steps <= 1000:
        raise ValueError("DDIM steps must be between 1 and 1000")
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn((1,19,17),generator=generator)
    c = torch.tensor(scaler.condition(task["condition"]),dtype=torch.float32)[None]
    alpha = schedule()
    times = np.linspace(999,0,steps).round().astype(int)
    start_calls = model.calls
    guidance_log = []
    for i,t in enumerate(times):
        with torch.no_grad():
            epsilon = model(x,torch.tensor([int(t)]),c)
            a = alpha[t]
            x0 = (x-(1-a).sqrt()*epsilon)/a.sqrt()
        if guidance is not None and i >= steps-2:
            z,info = guidance(scaler.inverse_z(x0[0].numpy()),i)
            x0 = torch.tensor(scaler.z(z),dtype=torch.float32)[None]
            guidance_log.append(info)
        next_a = alpha[times[i+1]] if i+1 < steps else torch.tensor(1.)
        x = next_a.sqrt()*x0 + (1-next_a).sqrt()*epsilon
    actual_calls = model.calls-start_calls
    if actual_calls != steps:
        raise RuntimeError("DDIM network-call contract violation")
    assert_frozen(task)
    return scaler.inverse_z(x[0].numpy()),{"requested_steps":steps,"actual_network_calls":actual_calls,
        "sampling_seed":seed,"guidance_inside_reverse_process":bool(guidance_log),"guidance":guidance_log}
