
import hashlib,json,os,sys
from pathlib import Path
stage,release,original = map(lambda x:Path(x).resolve(),sys.argv[1:4])
def guard(event,args):
    if event=='open' and isinstance(args[0],(str,bytes,os.PathLike)):
        p=Path(os.fsdecode(args[0])).resolve()
        if p==original or p.is_relative_to(original):
            raise RuntimeError('Original repository fallback forbidden: '+str(p))
sys.addaudithook(guard)
from v6_4.visualization.export_search_aware_release import PortableResolver
r=PortableResolver(release); result=r.verify_all(); r.verify_sources(); dataset=r.load_dataset()
import torch
def refuse(*a,**k): raise RuntimeError('Forward forbidden in loading smoke')
torch.nn.Module._call_impl=refuse
models={}
for name in ('D','S'):
    s=r.load_selected_sampler(name,device='cpu')
    assert s.sample_units==0 and not s.model.training and all(not p.requires_grad for p in s.model.parameters())
    models[name]=s.identity
loaded={name:str(Path(module.__file__).resolve()) for name,module in sys.modules.items()
        if name.split('.')[0] in ('v6_4','v6_lite','model_test') and getattr(module,'__file__',None)}
assert loaded and all(Path(path).is_relative_to(stage) for path in loaded.values())
print(json.dumps(dict(status='PASS',portable=result,models=models,dataset_labels=len(dataset.samples),
     frozen_imports=loaded,original_repository_access_forbidden=True,model_forward_forbidden=True,
     new_physics_steps=0,new_DDIM_samples=0,new_optimizer_updates=0),ensure_ascii=False))
