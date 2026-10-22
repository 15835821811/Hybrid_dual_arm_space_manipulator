"""Terminal read-only recovery: saved arrays, accounting, and refusal to restart.

Never changes the FAILED run or its frozen source. No additional physics.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
import time
for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[name]='2'  # Exact frozen numerical environment for saved DLS reconstruction.
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT));os.chdir(ROOT)
import numpy as np
from ff_prior.storage import read_json,digest,file_digest
from ff_prior.coupled_predictor import pose_errors
from ff_prior_static_goal_seed.inputs import identity
from ff_prior_static_goal_seed.protocol import OUT,PACKAGE,P
E=PACKAGE/'evidence'

def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')

def manifest():
    return {p.relative_to(OUT).as_posix():{'bytes':p.stat().st_size,'sha256':file_digest(p)}
        for p in sorted(OUT.rglob('*')) if p.is_file()}

def resume_once():
    dst=E/'completed_resume.json';attempt=E/'completed_resume_attempt.json'
    if dst.exists() or attempt.exists():raise RuntimeError('ONCE_ONLY_GUARD_ALREADY_ATTEMPTED')
    assert read_json(OUT/'status.json')['status']=='FAILED'
    before=manifest();write(E/'core_manifest.json',before)
    command=[sys.executable,'-m','ff_prior_static_goal_seed.job','resume']
    write(attempt,{'time':time.time(),'command':command,'expected':'FAILED_RUN_RESUME_REFUSED'})
    r=subprocess.run(command,cwd=ROOT,env=dict(os.environ,PYTHONIOENCODING='utf-8'),
        text=True,encoding='utf-8',capture_output=True)
    after=manifest()
    ok=(r.returncode!=0 and 'Unknown work is never automatically replayed' in r.stderr and before==after)
    record={'status':'PASS' if ok else 'FAIL','scope':'FAILED run refuses restart; not a completed-run success',
        'worker_status_preserved':'FAILED','returncode':r.returncode,'stdout':r.stdout,'stderr':r.stderr,
        'only_attempt':True,'before_files':len(before),'after_files':len(after),'core_files_unchanged':before==after,
        'expected_refusal':ok,'new_physics_steps':0}
    write(dst,record);assert ok
    print(json.dumps({k:v for k,v in record.items() if k not in ('stdout','stderr')},ensure_ascii=False))

def dist(v):
    x=np.asarray(v,dtype=float)
    return dict(mean=float(x.mean()),median=float(np.median(x)),p95=float(np.percentile(x,95)),minimum=float(x.min()),maximum=float(x.max()))

def metrics(ee,goal):
    pos,rot=pose_errors(np.asarray(ee),np.asarray(goal))
    return dict(position_mm=[float(x*1000) for x in pos],rotation_deg=[float(x) for x in np.rad2deg(rot)],
        maximum_position_mm=float(pos.max()*1000),maximum_rotation_deg=float(np.rad2deg(rot).max()),
        goal_pass=bool(max(pos)<=.005 and max(rot)<=np.deg2rad(1)))

def statistics():
    ident=identity();assert ident==read_json(OUT/'run.json')['identity']
    from ff_prior_static_goal_seed.tools.audit_saved_results_v1 import verify
    audit=verify(OUT,ident)
    audit.update(original_worker_status='FAILED',original_verifier_error_preserved=True,
        recovery_scope='separate-version saved-evidence audit; no physical rerun',new_physics_steps=0)
    audit['algebra_threads']=2
    write(E/'raw_audit_recovered_v2.json',audit)
    rows=[];units=[];repair_rows=[];task=ident['selection']['entries'][0]['task'];goal=task['condition']['goals']
    for spec in ident['selection']['units']:
        d=OUT/'units'/spec['id'];result=read_json(d/'result.json')
        for candidate in result['candidates']:
            pd=d/'candidates'/f'{candidate["id"]:02d}'
            a=np.load(pd/'goal_seed/initial.npz');b=np.load(pd/'goal_seed/result.npz')
            raw=np.load(pd/'raw/trace.npz');seeded=np.load(pd/'seeded/trace.npz');screen=read_json(pd/'seeded/screen.json')
            prov=read_json(pd/'provenance.json');seed=read_json(pd/'goal_seed/result.json')
            row={'unit':spec['id'],'method':spec['method'],'seed':spec['seed'],'candidate':candidate['id'],
                'selected_for_repair':candidate['selected_for_repair'],'final_status':candidate['status'],
                'MODEL_VALID':candidate['eligible'],'network_calls':prov['network_calls'],
                'prepare_scale':prov['preparation']['scale'],'prepare_unchanged':prov['preparation']['unchanged'],
                'raw':metrics(raw['ee'][-1],goal),'prepared':metrics(a['ee'],goal),'seeded':metrics(b['ee'],goal),
                'seed_Q':seed['Q'],'seed_iterations':seed['iterations'],'seed_stop':seed['stop'],
                'seeded_clearance_mm':float(screen['metrics']['minimum_clearance_m']*1000),
                'seeded_geometry_step_mm':float(screen['geometry_step_m']*1000),
                'final':None}
            if candidate['selected_for_repair']:
                v=read_json(pd/'validated/validation.json');pred=np.load(pd/'validated/prediction.npz')
                end=read_json(pd/'repair/result.json')
                row['final']=dict(metrics(pred['ee'][-1],goal),clearance_mm=v['validation']['geometry']['minimum_clearance_m']*1000,
                    geometry_step_mm=v['validation']['geometry']['maximum_geometry_step_m']*1000,
                    failed_gates=[k for k,x in v['validation']['gates'].items() if not x],
                    worst_pair=v['validation']['geometry']['coarse']['pair'],
                    numerical_passed=v['validation']['numerical']['passed'],iterations=end['iterations'],stop=end['stop'])
                for rp in sorted((pd/'repair/iterations').glob('*/record.json')):
                    x=read_json(rp);rr={'unit':spec['id'],'method':spec['method'],'iteration':int(rp.parent.name),
                        'accepted':x['accepted'],'trial_count':len(x['trials']),
                        'solver_returned_feasible':x['solver'].get('returned_feasible'),
                        'solver_success':x['solver'].get('success')}
                    for key in ('radius','trust_radius','radius_before','radius_after','radius_next'):
                        if key in x:rr[key]=x[key]
                    accepted=[t for t in x['trials'] if t.get('accepted')]
                    if accepted:rr['accepted_metrics']={k:accepted[0].get(k) for k in ('Q','minimum_clearance_m','collision_violation','scale','ratio')}
                    rr['evaluated_rejected_trials']=sum(t.get('status')=='EVALUATED' and not t.get('accepted') for t in x['trials'])
                    rr['goal_violating_rejected_trials']=sum(t.get('status')=='EVALUATED' and t.get('Q',0)>1 and not t.get('accepted') for t in x['trials'])
                    repair_rows.append(rr)
                units.append({'unit':spec['id'],'method':spec['method'],'seed':spec['seed'],
                    'candidate':candidate['id'],'final':row['final'],'calls':read_json(d/'calls.json'),
                    'execution_level':result['execution_level'],'candidate_denominator':5,'execution_denominator':1})
            rows.append(row)
    methods={}
    for method in P['methods']:
        selected=[r for r in rows if r['method']==method];full=[r['final'] for r in selected if r['final']]
        methods[method]={'candidate_denominator':15,'fully_validated':len(full),'screen_only':15-len(full),
            'MODEL_VALID':sum(r['MODEL_VALID'] for r in selected),'execution_attempted':0,'EXECUTION_VERIFIED':0,
            'execution_opportunities':3,'network_calls':sum(r['network_calls'] for r in selected),
            'stage_metrics':{stage:{'goal_pass':sum(r[stage]['goal_pass'] for r in selected),
                'position_mm':dist([r[stage]['maximum_position_mm'] for r in selected]),
                'rotation_deg':dist([r[stage]['maximum_rotation_deg'] for r in selected])} for stage in ('raw','prepared','seeded')},
            'seeded_clearance_mm':dist([r['seeded_clearance_mm'] for r in selected]),
            'final_goal_pass':sum(r['goal_pass'] for r in full),
            'final_position_mm':dist([r['maximum_position_mm'] for r in full]),
            'final_rotation_deg':dist([r['maximum_rotation_deg'] for r in full]),
            'final_clearance_mm':dist([r['clearance_mm'] for r in full]),
            'seeded_nonpenetration_at_screen_grid':sum(r['seeded_clearance_mm']>=0 for r in selected)}
    paired=[]
    for seed in P['seeds']:
        a=next(u for u in units if u['seed']==seed and u['method']=='structured_geometry')
        b=next(u for u in units if u['seed']==seed and u['method']=='diffusion_geometry')
        paired.append({'seed':seed,'diffusion_minus_structure_position_mm':b['final']['maximum_position_mm']-a['final']['maximum_position_mm'],
            'diffusion_minus_structure_rotation_deg':b['final']['maximum_rotation_deg']-a['final']['maximum_rotation_deg'],
            'diffusion_minus_structure_clearance_mm':b['final']['clearance_mm']-a['final']['clearance_mm']})
    summary={'run':OUT.relative_to(ROOT).as_posix(),'source_commit':read_json(OUT/'run.json')['producer_head'],
        'original_worker_status':'FAILED','original_failure':read_json(OUT/'failure.json')['reason'],
        'recovered_raw_audit':audit['status'],'all_units_done':len(units)==6,'candidate_denominator':30,
        'fully_validated':6,'screen_only':24,'execution_opportunities':6,'native_steps':0,'replay_steps':0,
        'TRAIN_added':0,'training_updates':0,'TEST_run':False,'engineering_five_obstacle_gate':False,
        'diffusion_advantage':'NOT_ESTABLISHED','elapsed_s':read_json(OUT/'status.json')['elapsed_s'],
        'budget':{k:v for k,v in read_json(OUT/'budget.json').items() if k!='limits'},
        'output_bytes':sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file()),
        'methods':methods,'units':units,'paired':paired,'fresh_review':'PENDING',
        'qualification_note':'24 screened only; six fully validated all geometry failed; no execution result exists'}
    write(E/'statistics.json',summary);write(E/'candidate_layers.json',rows);write(E/'repair_iterations.json',repair_rows)
    with (E/'candidate_layers.csv').open('w',newline='',encoding='utf-8-sig') as f:
        fields=['unit','method','seed','candidate','selected_for_repair','final_status','MODEL_VALID','network_calls','prepare_scale',
            'raw_mm','prepared_mm','seeded_mm','seeded_deg','seeded_clearance_mm','final_mm','final_deg','final_clearance_mm']
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for row in rows:
            v={k:row[k] for k in fields if k in row}
            for stage in ('raw','prepared','seeded'):v[stage+'_mm']=row[stage]['maximum_position_mm']
            v['seeded_deg']=row['seeded']['maximum_rotation_deg']
            for key,metric in [('final_mm','maximum_position_mm'),('final_deg','maximum_rotation_deg'),('final_clearance_mm','clearance_mm')]:
                v[key]=row['final'][metric] if row['final'] else None
            w.writerow(v)
    print(json.dumps({'raw_audit':audit['status'],'checks':len(audit['checks']),'failed_checks':[k for k,v in audit['checks'].items() if not v],
        'methods':methods,'paired':paired,'worker_status':'FAILED','physical_reruns':0},ensure_ascii=False,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=('statistics','resume-once'));a=p.parse_args()
    (resume_once if a.action=='resume-once' else statistics)()
