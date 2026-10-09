"""Verify relocated portable bytes and load frozen models without forward calls."""
import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time


def main():
    p=argparse.ArgumentParser(); p.add_argument('--release',type=Path,required=True)
    p.add_argument('--destination',type=Path,required=True); p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.destination.exists() or a.output.exists():
        raise FileExistsError('Fresh relocation and audit output required')
    record=dict(schema='c3_portable_load_audit_v1',started_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
        argv=[sys.executable,*sys.argv],script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        source_release=str(a.release.resolve()),relocated_release=str(a.destination.resolve()),status='FAIL')
    start=time.perf_counter()
    try:
        shutil.copytree(a.release,a.destination)
        from v6_4.visualization.export_search_aware_release import PortableResolver
        r=PortableResolver(a.destination)
        record['verification']=r.verify_all()
        r.verify_sources()
        dataset=r.load_dataset()
        record['dataset_type']=type(dataset).__name__
        # A hard forward guard ensures this verification adds no inference.
        import torch
        old_call=torch.nn.Module._call_impl
        def refuse(*args,**kwargs):
            raise RuntimeError('Model forward forbidden during portable loading audit')
        torch.nn.Module._call_impl=refuse
        try:
            models={}
            for name in ('D','S'):
                sampler=r.load_selected_sampler(name,device='cpu')
                assert sampler.sample_units==0 and not sampler.model.training
                assert all(not x.requires_grad for x in sampler.model.parameters())
                assert sampler.identity['checkpoint_update']==4000
                models[name]=sampler.identity
            record['models']=models
        finally:
            torch.nn.Module._call_impl=old_call
        unavailable=next((original for original,row in r.mapping.items() if not row.get('available')),None)
        assert unavailable is not None
        try:
            r.resolve(unavailable)
        except FileNotFoundError:
            record['local_only_source_refused']=True
        else:
            raise AssertionError('local-only source must not resolve from original drive')
        record.update(status='PASS',new_physics_steps=0,new_model_forwards=0,new_DDIM_samples=0,
                      new_optimizer_updates=0,source_manifest_sha256=hashlib.sha256((a.release/'release_manifest.json').read_bytes()).hexdigest())
    except Exception as exc:
        record['error']=repr(exc)
    record.update(ended_utc=dt.datetime.now(dt.timezone.utc).isoformat(),elapsed_s=time.perf_counter()-start)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('x',encoding='utf-8') as f:
        json.dump(record,f,indent=2,ensure_ascii=False)
    print(json.dumps(record,ensure_ascii=False))
    return 0 if record['status']=='PASS' else 1


if __name__=='__main__':
    sys.exit(main())
