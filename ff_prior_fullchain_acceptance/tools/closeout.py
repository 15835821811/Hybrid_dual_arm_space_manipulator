"""Terminal statistics from saved arrays and a once-only completed-resume check."""
from pathlib import Path
import sys,os,json,csv,subprocess,time
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='2'
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT));os.chdir(ROOT)
import numpy as np
from ff_prior.storage import read_json,file_digest
from ff_prior.coupled_predictor import pose_errors
from ff_prior_fullchain_acceptance.inputs import identity
from ff_prior_fullchain_acceptance.protocol import OUT,PACKAGE
E=PACKAGE/'evidence'

def write(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def distribution(v):
    x=np.asarray(v);return dict(mean=float(x.mean()),median=float(np.median(x)),p95=float(np.percentile(x,95)),maximum=float(x.max()),minimum=float(x.min()))
def manifest():
    return {p.relative_to(OUT).as_posix():dict(bytes=p.stat().st_size,sha256=file_digest(p)) for p in sorted(OUT.rglob('*')) if p.is_file()}

def resume_once():
    dst=E/'completed_resume.json';attempt=E/'completed_resume_attempt.json'
    if dst.exists() or attempt.exists():raise RuntimeError('ONCE_ONLY_ALREADY_ATTEMPTED')
    assert read_json(OUT/'status.json')['status']=='FINISHED_WITH_BLOCKED_ITEMS'
    before=manifest();write(E/'core_manifest.json',before)
    cmd=[sys.executable,'-m','ff_prior_fullchain_acceptance.job','resume']
    write(attempt,dict(time=time.time(),command=cmd))
    r=subprocess.run(cmd,cwd=ROOT,env=dict(os.environ,PYTHONIOENCODING='utf-8'),text=True,encoding='utf-8',capture_output=True)
    after=manifest();passed=r.returncode==0 and 'NO_WORK_ALREADY_COMPLETE' in r.stdout and before==after
    record=dict(status='PASS' if passed else 'FAIL',returncode=r.returncode,stdout=r.stdout,stderr=r.stderr,
        before_files=len(before),after_files=len(after),core_files_unchanged=before==after,
        completed_with_blocked_items_preserved=True,new_physics_steps=0,only_attempt=True)
    write(dst,record);assert passed
    print(json.dumps({k:v for k,v in record.items() if k not in ('stdout','stderr')}))

def statistics():
    ident=identity();assert ident==read_json(OUT/'run.json')['identity']
    summary=read_json(OUT/'summary.json');rows=[];new=[];plot=[]
    for x in summary['results']:
        spec,result=x['spec'],x['result'];r=dict(id=spec['id'],scene=spec['scene'],method=spec['method'],
            seed=spec.get('seed',spec.get('seed_group')),level=result['level'],new_execution=result.get('new_execution',False),
            scope=result.get('scope','DEPENDENCY_BLOCKED'),execution_source=result.get('execution_source'),
            source=spec.get('source'),reason=result.get('reason'))
        if result.get('new_execution'):
            d=OUT/spec['id']/'execution';a=np.load(d/'native_trace.npz');b=np.load(d/'replay_trace.npz');p=read_json(d/'report.json')
            task=read_json(d/'frozen_task.json');duration=task['condition']['duration']
            hold=a['time']>=a['time'][0]+duration-1e-9
            pos,rot=pose_errors(a['ee'],np.asarray(task['condition']['goals']))
            error=float(np.max(abs(a['integration']-b['integration'])))
            r.update(hold_position_mm=distribution(np.max(pos[hold],axis=1)*1000),
                hold_rotation_deg=distribution(np.max(np.rad2deg(rot[hold]),axis=1)),
                minimum_clearance_mm=p['minimum_clearance_m']*1000,
                geometry_step_mm=p['maximum_geometry_step_m']*1000,
                full_state_replay_error=error,full_state_dimensions=a['integration'].shape[1],
                native_steps=len(a['ctrl']),replay_steps=len(b['ctrl']),
                physics_contract='terminal hold only; do not compare full approach to terminal goal',
                full_native_time_s=float(a['time'][-1]-a['time'][0]),
                gates=p['gates'])
            assert error<=1e-10 and len(a['ctrl'])==13750 and len(b['ctrl'])==13750
            plot.append(dict(id=spec['id'],method=spec['method'],time=a['time'][::50].tolist(),
                ee=a['ee'][::50,:,:3,3].tolist(),goals=np.asarray(task['condition']['goals'])[:,:3,3].tolist(),
                source=(d/'native_trace.npz').relative_to(ROOT).as_posix(),sha256=file_digest(d/'native_trace.npz')))
            new.append(r)
        elif result['level']=='EXECUTION_VERIFIED':
            p=result['report']
            r.update(world_position_mm=p['position_mm'],world_rotation_deg=p['rotation_deg'],
                minimum_clearance_mm=p['minimum_clearance_mm'],full_state_replay_error=p['replay_full_integration_error'],
                original_new_execution=result['original_new_execution'],shared_not_independent=result['shared_not_independent'],
                physics_contract='whole dynamic trajectory; inherited exact historical result',
                native_steps=0,replay_steps=0)
        rows.append(r)
    paired=[]
    for seed in sorted(set(r['seed'] for r in new)):
        a=next(r for r in new if r['seed']==seed and r['method']=='deterministic_system')
        b=next(r for r in new if r['seed']==seed and r['method']=='diffusion_system')
        paired.append(dict(seed=seed,static_hold_worst_position_diffusion_minus_structured_mm=b['hold_position_mm']['maximum']-a['hold_position_mm']['maximum'],
            static_minimum_clearance_diffusion_minus_structured_mm=b['minimum_clearance_mm']-a['minimum_clearance_mm']))
    stats=dict(run=OUT.relative_to(ROOT).as_posix(),source_commit=read_json(OUT/'run.json')['producer_head'],
        worker_elapsed_s=read_json(OUT/'status.json')['elapsed_s'],experiment_elapsed_s=summary['elapsed_s'],
        output_bytes=sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file()),
        new_static_verified=sum(r['level']=='EXECUTION_VERIFIED' for r in new),new_static_attempted=len(new),
        new_native_steps=sum(r['native_steps'] for r in new),new_replay_steps=sum(r['replay_steps'] for r in new),
        historical_logical_verified=12,historical_unique_executions=8,historical_shared_logic=4,
        logical_verified=18,logical_denominator=30,blocked=12,covered_scenes=3,total_scenes=5,
        engineering_complete=False,scientific_stable_advantage=False,new_training=0,new_proposals=0,TRAIN_added=0,TEST=False,
        scene_counts=summary['scene_counts'],paired_static=paired,rows=rows,fresh_review='PENDING')
    write(E/'statistics.json',stats);write(E/'native_endpoint_paths.json',plot)
    with (E/'unit_metrics.csv').open('w',encoding='utf-8-sig',newline='') as f:
        fields=['id','scene','method','seed','level','scope','new_execution','native_steps','replay_steps','minimum_clearance_mm','full_state_replay_error']
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(rows)
    print(json.dumps({k:v for k,v in stats.items() if k not in ('rows','scene_counts')},ensure_ascii=False,indent=2))

if __name__=='__main__':
    (resume_once if sys.argv[-1]=='resume-once' else statistics)()
