$ErrorActionPreference = 'Stop'
$previousPythonEncoding = $env:PYTHONIOENCODING
Push-Location -LiteralPath (Split-Path $PSScriptRoot -Parent)
try {
    $env:PYTHONIOENCODING = 'utf-8'
    @'
from pathlib import Path
from collections import Counter
import numpy as np
from ff_prior.storage import read_json, atomic_json
from ff_prior.spline import Reference

root=Path.cwd(); out=root/'artifacts/v7_1/qualified_modes_smoke_001'
assert read_json(out/'status.json')['status']=='FINISHED'
summary=read_json(out/'summary.json'); entries=read_json(out/'frozen_tasks.json')['evaluation_entries']
methods=list(summary['methods']); rows={}; quality={}; paired={}
def describe(values):
    a=np.asarray(values,dtype=float)
    return {'min':float(a.min()),'median':float(np.median(a)),'mean':float(a.mean()),'max':float(a.max())}
for method in methods:
    rows[method]=[]; selected=[]; task_cost={}; diversities=[]
    for entry in entries:
        sid=entry['sample_id']; condition=entry['task']['condition']
        candidates=[read_json(out/'candidates'/sid/method/f'{k:02d}.json') for k in range(5)]
        rows[method].extend(candidates)
        task_cost[sid]={'coupled':sum(r['model_calls']['coupled'] for r in candidates),'wall_s':sum(r['wall_s'] for r in candidates)}
        z=np.array([r['Z'] for r in candidates])
        diversities.append(float(np.mean([np.sqrt(np.mean((z[i]-z[j])**2)) for i in range(5) for j in range(i)])))
        d=out/'executions'/sid/method
        choice=read_json(d/'selection.json')['selected_id']
        r=next(r for r in candidates if r['candidate_id']==choice)
        ref=Reference(r['Z'],condition['initial_features'],condition['duration'])
        times=np.linspace(0,ref.duration,1025); q=ref.evaluate(times); acc=ref.evaluate(times,2)
        report=read_json(d/'report.json')
        with np.load(d/'native_trace.npz') as trace:
            dt=np.diff(trace['time']); ctrl=trace['ctrl']
            effort=float(np.sum(np.sum(ctrl**2,axis=1)*dt))
            actual_length=float(np.linalg.norm(np.diff(trace['q_actual_17'],axis=0),axis=1).sum())
        selected.append({'task':sid,'joint_path_length_rad':float(np.linalg.norm(np.diff(q,axis=0),axis=1).sum()),
            'actual_joint_path_length_rad':actual_length,'joint_acceleration_squared_integral_rad2_s3':float(np.trapz(np.sum(acc**2,axis=1),times)),
            'torque_squared_integral_Nm2_s':effort,'minimum_clearance_mm':report['minimum_clearance_m']*1000,
            'hold_position_max_mm':max(report['hold_max_position_error_m'])*1000,
            'hold_rotation_max_deg':float(np.rad2deg(max(report['hold_max_rotation_error_rad']))),
            'tracking_max_rad':report['tracking_max_rad'],'base_translation_max_mm':report['maximum_base_translation_m']*1000})
    quality[method]={'selected':selected,'metrics':{k:describe([r[k] for r in selected]) for k in selected[0] if k!='task'},
        'per_task_cost':task_cost,'final_candidate_pairwise_control_rms_rad':describe(diversities),
        'repair_query_histogram':dict(Counter(r['repair_coupled_calls'] for r in rows[method]))}
goal='goal_loss_diffusion'
for m in methods:
    if m==goal:continue
    a,b=summary['methods'][goal],summary['methods'][m]
    paired[m]={'goal_model_percent_change':{k:100*(a[k]/b[k]-1) for k in ('coupled_queries','candidate_wall_s','integration_intervals')},
        'task_query_wins':sum(quality[goal]['per_task_cost'][sid]['coupled']<v['coupled'] for sid,v in quality[m]['per_task_cost'].items()),
        'task_query_ties':sum(quality[goal]['per_task_cost'][sid]['coupled']==v['coupled'] for sid,v in quality[m]['per_task_cost'].items()),
        'candidate_query_wins':sum(a['model_calls']['coupled']<b['model_calls']['coupled'] for a,b in zip(rows[goal],rows[m])),
        'candidate_query_ties':sum(a['model_calls']['coupled']==b['model_calls']['coupled'] for a,b in zip(rows[goal],rows[m]))}
result={'status':'PASS','scope':'posthoc descriptive metrics of saved references and native execution traces; no new model/physics calls; not preregistered superiority gates',
    'methods':quality,'paired':paired,'caveats':['10 development tasks, one training seed, one timing run; no inferential significance claim',
    'Final candidate diversity includes nonlinear repair; not raw diffusion diversity or homotopy certification',
    'Torque squared integral is an effort proxy, not mechanical energy or electrical power',
    'Joint derivatives are reference-space diagnostics, not Cartesian smoothness or safety certifications']}
atomic_json(root/'ff_prior_modes/evidence/saved_trajectory_diagnostics.json',result)
print({m:v['goal_model_percent_change'] for m,v in paired.items()})
'@ | & .\.venv\Scripts\python.exe -
    if ($LASTEXITCODE -ne 0) { throw "Saved-result analysis failed: $LASTEXITCODE" }
} finally {
    Pop-Location
    $env:PYTHONIOENCODING = $previousPythonEncoding
}
