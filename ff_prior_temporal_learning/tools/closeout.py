"""Read-only terminal evidence statistics; no physics or optimizer updates.

The explicit --resume-once mode invokes only the completed-run guard once,
records its output, and verifies every run file remains unchanged.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)
import numpy as np
from ff_prior.coupled_predictor import pose_errors
from ff_prior.robot_adapter import physical_spec
from ff_prior_dynamic_margin.representation import design

RUN = ROOT / 'artifacts/v7_1/temporal_learning_001'
EVIDENCE = ROOT / 'ff_prior_temporal_learning/evidence'

def read(p):
    return json.loads(p.read_text(encoding='utf-8'))

def write(p, value):
    p.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')

def manifest():
    rows = {}
    for p in sorted(RUN.rglob('*')):
        if p.is_file():
            h = hashlib.sha256()
            with p.open('rb') as f:
                for b in iter(lambda: f.read(2**20), b''): h.update(b)
            rows[p.relative_to(RUN).as_posix()] = dict(bytes=p.stat().st_size, sha256=h.hexdigest())
    return rows

def resume_once():
    destination = EVIDENCE/'completed_resume.json'
    attempt = EVIDENCE/'completed_resume_attempt.json'
    if destination.exists() or attempt.exists():
        raise RuntimeError('Once-only verification already attempted; inspect saved evidence, do not rerun')
    assert read(RUN/'status.json')['status'] in ('FINISHED', 'FINISHED_WITH_BLOCKED_ITEMS')
    before = manifest()
    write(EVIDENCE/'core_manifest.json', before)
    write(attempt, dict(started=time.time(), files=len(before), command=[sys.executable, '-m', 'ff_prior_temporal_learning.job', 'resume']))
    r = subprocess.run([sys.executable, '-m', 'ff_prior_temporal_learning.job', 'resume'], cwd=ROOT,
        env=dict(os.environ, PYTHONIOENCODING='utf-8'), capture_output=True, text=True, encoding='utf-8')
    after = manifest()
    record = dict(returncode=r.returncode, stdout=r.stdout, stderr=r.stderr,
                  before_files=len(before), after_files=len(after), core_files_unchanged=before==after,
                  only_attempt=True, timestamp=time.time())
    write(destination, record)
    assert r.returncode == 0 and before == after and 'NO_WORK_ALREADY_COMPLETE' in r.stdout
    print(json.dumps({k:v for k,v in record.items() if k not in ('stdout','stderr')}))

def dist(x):
    x=np.asarray(x)
    return dict(mean=float(x.mean()), median=float(np.median(x)), p95=float(np.percentile(x,95)), maximum=float(x.max()))

def statistics():
    s=read(RUN/'summary.json'); spec=physical_spec(); _,_,d1,d2=design(27.)
    val=np.load(RUN/'validation_encoded.npz'); codec=read(RUN/'codec.json')
    rows=[]; native_steps=0; replay_steps=0
    for e in s['results']:
        r=e['result']; directory=RUN/e['unit']
        source=ROOT/r['reused_from'] if not r.get('new_execution') and r.get('reused_from') else directory
        row={k:v for k,v in e.items() if k!='result'}
        row.update(status=r['status'], new_execution=r.get('new_execution',False),
            failed_gates=[k for k,v in r.get('gates',{}).items() if not v],
            stop_reason=r.get('stop_reason'), steps=r.get('steps',0),
            complete=bool(r.get('gates',{}).get('complete_native_time',False)),
            metric_scope='observed prefix only if incomplete; missing remainder is not scored as success')
        if not (source/'native_trace.npz').exists():
            rows.append(row);continue
        with np.load(source/'native_trace.npz') as n, np.load(source/'replay_trace.npz') as rep:
            pos,rot=pose_errors(n['ee'],n['desired']); pos*=1000;rot=np.rad2deg(rot)
            row.update(position_mm=dist(pos.max(axis=1)),rotation_deg=dist(rot.max(axis=1)),
                position_per_arm_mm={k:[float(v) for v in f(pos)] for k,f in {'mean':lambda x:x.mean(0),'p95':lambda x:np.percentile(x,95,axis=0),'maximum':lambda x:x.max(0),'final':lambda x:x[-1]}.items()},
                position_gate_failed_frames=int(np.sum(pos.max(1)>5)),rotation_gate_failed_frames=int(np.sum(rot.max(1)>1)),
                observed_frames=len(pos),observed_duration_s=float(n['time'][-1]),
                replay_full_state_max=float(abs(n['integration']-rep['integration']).max()),
                controls_equal=bool(np.array_equal(n['ctrl'],rep['ctrl'])),
                minimum_clearance_mm=float(n['clearance'].min()*1000),
                reference_velocity_ratio=float(np.max(abs(n['reference_qd'])/spec.planner_velocity_limits)),
                reference_acceleration_ratio=float(np.max(abs(n['reference_qdd'])/spec.planner_acceleration_limits)),
                q17_second_difference_acceleration_ratio=float(np.max(abs(np.diff(n['q17'],n=2,axis=0)/.002**2)/spec.planner_acceleration_limits)))
            # q17 finite differences are descriptive; audit uses low-level qvel and Bplus for the actual gate.
            if row['new_execution']:
                native_steps+=len(n['ctrl']);replay_steps+=len(rep['ctrl'])
        p=read(source/'prepare.json');row['prepare']=p
        with np.load(source/'proposal.npz') as raw, np.load(source/'prepared.npz') as prepared:
            c=raw['raw_coefficients'];cp=prepared['coefficients']
            row['raw_derivative_hull'] = dict(velocity_ratio=float(np.max(abs(d1@c)/spec.planner_velocity_limits)),
                acceleration_ratio=float(np.max(abs(d2@c)/spec.planner_acceleration_limits)))
            if e['split']=='VAL':
                index=int(e['unit'][1]); target=val['x'][index]
                xp=(cp[2:]-c[0]-np.asarray(codec['mean']))/np.asarray(codec['scale'])
                row['coefficient_MSE_to_qualified_VAL']=dict(raw=float(np.mean((raw['normalized_output']-target)**2)),prepared=float(np.mean((xp-target)**2)))
        if (source/'feedback.json').exists():
            f=read(source/'feedback.json')
            row['feedback']={key:dist([v[key] for v in f]) for key in ['world_correction_norm','constraint_correction_norm']}
            row['feedback']['preferred_norm']=dist([np.linalg.norm(v['sequence_preferred']) for v in f])
        rows.append(row)
    paired=[]
    for scene in ['stationary','drifting']:
        for group in range(3):
            pair=[x for x in rows if x['split']=='DEV' and x['scene']==scene and x['seed_group']==group]
            a=next(x for x in pair if x['method']=='deterministic_sequence');b=next(x for x in pair if x['method']=='temporal_diffusion')
            paired.append(dict(scene=scene,seed_group=group,deterministic=a['unit'],diffusion=b['unit'],
                position_mm_delta={k:b['position_mm'][k]-a['position_mm'][k] for k in a['position_mm']},
                rotation_deg_delta={k:b['rotation_deg'][k]-a['rotation_deg'][k] for k in a['rotation_deg']}))
    groups={}
    for split in ['VAL','DEV']:
        for method in ['deterministic_sequence','temporal_diffusion']:
            group=[x for x in rows if x['split']==split and x['method']==method]
            groups[split+'/'+method]=dict(logical=len(group),verified=sum(x['status']=='EXECUTION_VERIFIED' for x in group),
                actual_executions=sum(x['new_execution'] for x in group),
                unit_max_position_mm=dist([x['position_mm']['maximum'] for x in group]),
                unit_max_rotation_deg=dist([x['rotation_deg']['maximum'] for x in group]),
                truncated=sum(not x['complete'] for x in group),
                averaging='per-unit maximum, not pooled time; deterministic DEV reuses are not independent replications')
    out=dict(run=RUN.relative_to(ROOT).as_posix(),source_commit='681965a0',counts=s['counts'],groups=groups,units=rows,paired_DEV=paired,
        actual_native_steps=native_steps,actual_replay_steps=replay_steps,training={m:{k:v for k,v in t.items() if k!='history'} for m,t in s['training']['methods'].items()},
        source_summary_sha256=hashlib.sha256((RUN/'summary.json').read_bytes()).hexdigest(),
        no_new_physics_or_optimization=True,zero_missing_units=len(rows)==20)
    write(EVIDENCE/'completed_statistics.json',out)
    print(json.dumps(dict(groups=groups,native_steps=native_steps,replay_steps=replay_steps,paired=paired),ensure_ascii=False,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--resume-once',action='store_true');a=p.parse_args()
    resume_once() if a.resume_once else statistics()
