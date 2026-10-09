"""Run portable dataset/weight loading with the bundled frozen producer code.

The publication may contain later maintenance. This creates an isolated copy of
the frozen sources plus the delivery resolver, never modifies the release, and
performs no model forward, sampling, physics or optimizer update.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import os

CHILD = r'''
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
'''


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--release',type=Path,required=True)
    parser.add_argument('--stage',type=Path,required=True)
    args=parser.parse_args()
    repo=Path(__file__).resolve().parents[2]
    if args.stage.exists():
        raise FileExistsError('Use a fresh staging directory; prior evidence is preserved')
    shutil.copytree(args.release/'frozen_source',args.stage)
    helper=repo/'v6_4/visualization/export_search_aware_release.py'
    target=args.stage/'v6_4/visualization/export_search_aware_release.py'
    target.parent.mkdir(exist_ok=True,parents=True); shutil.copyfile(helper,target)
    child=args.stage/'_load_frozen_models.py';child.write_text(CHILD,encoding='utf-8')
    env=os.environ.copy();env['PYTHONPATH']=str(args.stage.resolve())
    command=[sys.executable,'-B','-X','utf8',str(child.resolve()),str(args.stage.resolve()),str(args.release.resolve()),str(repo)]
    result=subprocess.run(command,cwd=args.stage,env=env,capture_output=True,text=True,encoding='utf-8')
    (args.stage/'stdout.txt').write_text(result.stdout,encoding='utf-8')
    (args.stage/'stderr.txt').write_text(result.stderr,encoding='utf-8')
    receipt=dict(argv=command,exit_code=result.returncode,helper_sha256=hashlib.sha256(helper.read_bytes()).hexdigest(),
                 child_sha256=hashlib.sha256(child.read_bytes()).hexdigest(),status='PASS' if result.returncode==0 else 'FAIL')
    (args.stage/'load_receipt.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
    print(result.stdout);print(result.stderr,file=sys.stderr);print(json.dumps(receipt))
    return result.returncode


if __name__=='__main__':
    sys.exit(main())
