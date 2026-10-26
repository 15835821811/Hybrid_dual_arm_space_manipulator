"""One saved-result arithmetic pass; never generates a new model/physical trajectory."""
from pathlib import Path
import json,time,hashlib,subprocess,zipfile
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'artifacts/v7_1/guided_diffusion_chain_001'
OUT=ROOT/'ff_prior_guided_chain_closeout/evidence'


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


def main():
    status=json.loads((RUN/'status.json').read_text(encoding='utf-8'))
    if status['status']=='RUNNING':raise RuntimeError('TERMINAL_ONLY')
    result_path=OUT/'saved_checks.json'
    if result_path.exists():raise RuntimeError('SAVED_RESULT_CHECK_ALREADY_DONE')
    rows=[]
    def check(name,ok,**info):rows.append(dict(name=name,passed=bool(ok),**info))
    identity=json.loads((RUN/'run.json').read_text(encoding='utf-8'))['identity']
    for rel,sha in {**identity['files'],**identity['source_hashes']}.items():
        check('identity:'+rel,hashlib.sha256((ROOT/rel).read_bytes()).hexdigest()==sha)
    # Frozen schedule arithmetic only: no checkpoint loading or neural forward.
    alphas=torch.cumprod(1-torch.linspace(1e-4,.02,1000),dim=0).double().numpy()
    ddim_times=torch.linspace(999,0,16).round().long().numpy()
    archive=RUN/'executed_source.zip'
    if archive.exists():
        with zipfile.ZipFile(archive) as z:
            for rel,sha in identity['source_hashes'].items():
                check('archive:'+rel,hashlib.sha256(z.read(rel)).hexdigest()==sha)
    neural_steps=0;guides=0;fd_columns=0;diagnostics=[]
    for phase in ('A','B','C'):
        folder=RUN/phase/'planning'
        records={}
        if (folder/'sampler.json').exists():
            records={r['step']:r for r in json.loads((folder/'sampler.json').read_text(encoding='utf-8'))['steps']}
        for path in sorted(folder.glob('ddim_*.npz')):
            a=np.load(path);j=int(path.stem.split('_')[-1]);neural_steps+=1
            r=records.get(j,dict(alpha=float(alphas[ddim_times[j]]),next_alpha=float(alphas[ddim_times[j+1]]) if j<15 else 1.))
            alpha=r['alpha'];next_a=r['next_alpha']
            clean=np.sqrt(alpha)*a['xt']-np.sqrt(1-alpha)*a['v']
            eps=(a['xt']-np.sqrt(alpha)*a['encoded_corrected'])/np.sqrt(1-alpha)
            following=np.sqrt(next_a)*a['encoded_corrected']+np.sqrt(1-next_a)*eps
            for name,x,y in [('clean',clean,a['clean']),('epsilon',eps,a['epsilon']),('next_xt',following,a['next_xt'])]:
                err=float(abs(x-y).max());check(f'{phase}/ddim{j}/{name}',err<=1e-10,maximum_error=err)
        for path in sorted(folder.glob('guidance_*/jacobian.npz')):
            a=np.load(path);guides+=1;fd_columns+=len(a['steps'])
            J=(a['perturbed_residual']-a['residual']).T/a['steps']
            check(str(path.relative_to(RUN))+'/all_J_columns',np.array_equal(J,a['jacobian']),columns=J.shape[1],rows=J.shape[0])
            probes=json.loads((path.parent/'directional_checks.json').read_text(encoding='utf-8'))
            nworld=json.loads((path.parent/'center.json').read_text(encoding='utf-8'))['metrics']['world_rows']
            bad_rows=set()
            for j,p in enumerate(probes['probes']):
                values=np.asarray(p['values']);h=1e-4
                central=(values[0]-values[1])/(2*h);half=(values[2]-values[3])/h
                linear=J@np.asarray(p['direction']);tol=.02*(1+abs(half))
                check(str(path.parent.relative_to(RUN))+f'/probe{j}/central',np.array_equal(central,p['central']))
                check(str(path.parent.relative_to(RUN))+f'/probe{j}/half',np.array_equal(half,p['half']))
                diff=float(abs(linear-np.asarray(p['linear'])).max())
                check(str(path.parent.relative_to(RUN))+f'/probe{j}/linear',diff<=1e-9,maximum_error=diff)
                passed=bool(np.all(abs(central-half)<=tol) and np.all(abs(linear-half)<=tol))
                check(str(path.parent.relative_to(RUN))+f'/probe{j}/verdict',passed==p['passed'],reported_passed=p['passed'])
                bad=np.flatnonzero((abs(central-half)>tol)|(abs(linear-half)>tol));bad_rows.update(bad.tolist())
                details=[]
                for k in bad:
                    kind='world' if k<nworld else 'geometry' if k<nworld+len(a['active']) else 'torque'
                    rid=a['active'][k-nworld].tolist() if kind=='geometry' else a['torque'][k-nworld-len(a['active'])].tolist() if kind=='torque' else int(k)
                    details.append(dict(row=int(k),kind=kind,identity=rid,central=float(central[k]),half=float(half[k]),linear=float(linear[k]),tolerance=float(tol[k]),central_half_error=float(abs(central[k]-half[k])),linear_half_error=float(abs(linear[k]-half[k]))))
                diagnostics.append(dict(phase=phase,guide=path.parent.name,probe=j,failed_rows=details,failed_count=len(details),central_half_failures=int(np.sum(abs(central-half)>tol)),linear_half_failures=int(np.sum(abs(linear-half)>tol))))
            # Inspect the existing coordinate perturbations at the failed rows;
            # do not run smaller finite differences or new geometry queries.
            for k in sorted(bad_rows):
                delta=a['perturbed_residual'][:,k]-a['residual'][k]
                diagnostics.append(dict(phase=phase,guide=path.parent.name,row=k,nominal_residual=float(a['residual'][k]),coordinate_delta_min=float(delta.min()),coordinate_delta_max=float(delta.max()),nonzero_coordinate_columns=int(np.count_nonzero(delta)),largest_columns=np.argsort(abs(delta))[-10:].tolist()))
        for path in sorted(folder.glob('guidance_*/*.npz')):
            if path.name=='jacobian.npz':continue
            a=np.load(path)
            if 'near' not in a:continue
            near=a['near'];selected=a['selected'];omitted=a['omitted'];pairs=a['pair_ids'];rad=a['geom_rbound'];centers=a['geom']
            check(str(path.relative_to(RUN))+'/all_near_partition',np.array_equal(np.sort(np.r_[selected,omitted]),np.arange(len(near))))
            check(str(path.relative_to(RUN))+'/near_pair_identity',not len(near) or np.array_equal(pairs[near[:,4].astype(int)],near[:,1:3].astype(int)))
            minima=[]
            for i,t in enumerate(a['time']):
                lower=np.linalg.norm(centers[i,pairs[:,0]]-centers[i,pairs[:,1]],axis=1)-rad[pairs[:,0]]-rad[pairs[:,1]]
                rows_at=near[near[:,0]==t];idx=rows_at[:,4].astype(int)
                check(str(path.relative_to(RUN))+f'/state{i}/exact_query_coverage',np.array_equal(np.sort(idx),np.flatnonzero(lower<=.008)))
                lower[idx]=rows_at[:,3];minima.append(float(lower.min()))
            check(str(path.relative_to(RUN))+'/all_minima',np.array_equal(minima,a['minimum_path']))
        actual=RUN/phase/'execution/native_trace.npz';replay=RUN/phase/'execution/replay_trace.npz'
        if actual.exists() and replay.exists():
            a=np.load(actual);b=np.load(replay);n=len(a['integration']);nr=len(b['integration'])
            check(phase+'/full_replay_length',n==nr,actual=n,replay=nr)
            if n==nr:
                delta=float(abs(a['integration']-b['integration']).max())
                check(phase+'/all836_replay',delta<=1e-10,maximum_error=delta,components=int(a['integration'].size))
    write(result_path,dict(status='PASS' if all(r['passed'] for r in rows) else 'UNRESOLVED',checks=rows,count=len(rows),
        passed=sum(r['passed'] for r in rows),failed=sum(not r['passed'] for r in rows),
        saved_complete_ddim_steps=neural_steps,saved_guidance_jacobians=guides,all_saved_fd_columns=fd_columns,
        new_neural=0,new_coupled=0,new_native=0,new_geometry_queries=0,
        limitation='Saved arithmetic only; no reconstruction of unarchived mass matrices or all836 states of each FD propagation'))
    write(OUT/'derivative_diagnosis.json',dict(source='All saved directional probes and coordinate perturbations; no new model or physics',diagnostics=diagnostics))
    print(json.dumps(dict(checks=len(rows),failed=sum(not r['passed'] for r in rows))))


if __name__=='__main__':main()
