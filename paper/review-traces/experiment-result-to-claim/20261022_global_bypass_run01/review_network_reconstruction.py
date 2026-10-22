"""Neural-only reconstruction from immutable checkpoint and saved public setup.
No project modules, simulator, optimizer or training is imported/invoked.
"""
from pathlib import Path
import hashlib,json
import numpy as np
import torch
from torch import nn
TRACE=Path(__file__).resolve().parent;ROOT=TRACE.parents[3];RUN=ROOT/'artifacts/v7_1/global_bypass_001'
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def digest(x):return hashlib.sha256(json.dumps(x,default=lambda a:a.tolist(),sort_keys=True,separators=(',',':')).encode()).hexdigest()
class Low(nn.Module):
    def __init__(self):
        super().__init__();self.net=nn.Sequential(nn.Linear(199,256),nn.SiLU(),nn.Linear(256,256),nn.SiLU(),nn.Linear(256,256),nn.SiLU(),nn.Linear(256,102));self.calls=0
    def forward(self,x,t,c):
        self.calls+=1;freq=torch.exp(torch.arange(16)*(-np.log(10000)/15));phase=t[:,None]*freq[None]
        return self.net(torch.cat([x,c,phase.sin(),phase.cos()],1))
torch.set_num_threads(2)
torch.manual_seed(20264601)
model=Low();path=ROOT/'artifacts/v7_1/sample_alignment_001/learning/aligned/model.pt';checkpoint=torch.load(path,map_location='cpu',weights_only=False);train=read(path.parent/'training.json')
model.load_state_dict(checkpoint['weights']);model.eval();h=hashlib.sha256()
for k,v in model.state_dict().items():h.update(k.encode());h.update(v.detach().numpy().tobytes())
codec=read(ROOT/'artifacts/v7_1/control_transfer_bounded_001/codec.json');arr={k:np.asarray(codec[k]) for k in ('basis','feature_mean','feature_scale','ridge','latent_mean','latent_scale')}
assert hashlib.sha256(path.read_bytes()).hexdigest()==train['checkpoint_sha256']
assert checkpoint['metadata']==train['metadata']
assert h.hexdigest()==train['trained_weight_hash']
ident=read(RUN/'run.json')['identity'];condition=ident['selection']['entries'][0]['task']['condition'];g=np.asarray(condition['goals']);q0=np.asarray(condition['initial_features'])
public=np.r_[q0,g[:,:3,3].ravel(),g[:,:3,:2].transpose(0,2,1).ravel(),condition['duration'],np.asarray(condition['pose_mask']).ravel()]
alpha=torch.cumprod(1-torch.linspace(1e-4,.02,1000),dim=0).double();times=np.linspace(999,0,16).round().astype(int);rows=[]
with torch.inference_mode():
    for spec in ident['selection']['units']:
        ud=RUN/'units'/spec['id'];setup=read(ud/'public_setup.json');base=np.asarray(setup['base']);J=np.asarray(setup['jacobian']);frame=np.asarray(setup['frame']);amp=setup['bow_amplitude_m']
        c=(np.r_[public,base[-1]-q0]-arr['feature_mean'])/arr['feature_scale'];center=np.r_[1.,c]@arr['ridge'];ct=torch.tensor(c,dtype=torch.float32)[None]
        for slot,seed in enumerate(spec['proposal_seeds']):
            pd=ud/'candidates'/f'{slot:02d}'
            with np.load(pd/'controls.npz') as z:saved={k:z[k] for k in z.files}
            before=model.calls;detail_max=0.
            if spec['method']=='diffusion_geometry':
                x=torch.cat((torch.randn((1,102),generator=torch.Generator().manual_seed(seed)),torch.randn((1,221),generator=torch.Generator().manual_seed(seed+971003))),1).double()
                for i,t in enumerate(times):
                    v=model(x[:,:102].float(),torch.full((1,),int(t)),ct).double();a=alpha[t]
                    clean=torch.cat((a.sqrt()*x[:,:102]-(1-a).sqrt()*v,torch.zeros((1,221),dtype=torch.float64)),1)
                    epsilon=(x-a.sqrt()*clean)/(1-a).sqrt();na=alpha[times[i+1]] if i<15 else x.new_tensor(1.)
                    x=na.sqrt()*clean+(1-na).sqrt()*epsilon
                latent=x[0].numpy();detail_max=float(abs(latent[102:]).max());raw=base+arr['basis']@(latent*arr['latent_scale']+arr['latent_mean']+center).reshape(19,17)
            else:raw=base.copy()
            direction=np.zeros(17)
            if slot:
                angle=np.random.default_rng(seed-slot).uniform(0,2*np.pi);v=np.cos(angle)*frame[0]+np.sin(angle)*frame[1];w=-np.sin(angle)*frame[0]+np.cos(angle)*frame[1]
                error=np.r_[np.zeros(6),[v,-v,w,-w][slot-1]*amp,np.zeros(3)]
                direction=np.linalg.solve(J.T@J+1e-4*np.eye(17),J.T@error);direction*=min(1,.3/max(abs(direction).max(),1e-12))
            phase=np.r_[np.linspace(0,1,20)[1:-1],1.];bow=np.sin(np.pi*phase)**2;bow[-1]=0;proposal=raw+bow[:,None]*direction
            rawerr=float(abs(raw-saved['network_raw']).max());properr=float(abs(proposal-saved['proposal']).max())
            rows.append({'unit':spec['id'],'slot':slot,'seed':seed,'method':spec['method'],'model_calls':model.calls-before,'raw_max_abs_error_rad':rawerr,'proposal_max_abs_error_rad':properr,'detail_latent_max_abs':detail_max,'passed':rawerr<1e-12 and properr<1e-9})
result={'status':'PASS' if all(r['passed'] for r in rows) else 'FAIL','scope':'neural and public proposal algebra only; no physical evaluation','checkpoint_sha256':train['checkpoint_sha256'],'weights_sha256':h.hexdigest(),'feature_dimensions':len(c),'old_network_output_dimensions':102,'full_codec_output_dimensions':323,'timesteps':times.tolist(),'new_training_updates':0,'physics_calls':0,'rows':rows,'total_low_model_forward_calls':model.calls,'note':'Zero detail output follows frozen TransferDenoiser zero final layer; this reconstruction directly supplies those identically zero outputs.'}
(TRACE/'network_reconstruction.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');print(json.dumps(result,ensure_ascii=False))
