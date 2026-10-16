param([string]$Output = 'artifacts/v7_1/train_residual_probe_001')
$ErrorActionPreference = 'Stop'
$previousEncoding = $env:PYTHONIOENCODING
$previousOutput = $env:FF_PROBE_ANALYSIS_OUTPUT
Push-Location -LiteralPath (Split-Path $PSScriptRoot -Parent)
try {
    $env:PYTHONIOENCODING = 'utf-8'
    $env:FF_PROBE_ANALYSIS_OUTPUT = $Output
    @'
from pathlib import Path
import os
import numpy as np
from ff_prior.storage import read_json,atomic_json
from ff_prior_probe.reporting import diversity

root=Path.cwd();out=root/os.environ['FF_PROBE_ANALYSIS_OUTPUT']
assert read_json(out/'status.json')['status'] in ('FINISHED','FINISHED_WITH_BLOCKED_ITEMS')
summary=read_json(out/'summary.json');entries=read_json(out/'frozen_tasks.json')['evaluation_entries']
def stats(values):
    a=np.asarray(values,dtype=float)
    return {'mean':float(a.mean()),'median':float(np.median(a)),'min':float(a.min()),'max':float(a.max())} if len(a) else None
rows={m:{(e['sample_id'],k):read_json(out/'candidates'/e['sample_id']/m/f'{k:02d}.json') for e in entries for k in range(5)} for m in summary['methods']}
metrics={};paired={};nonlinearity=[];loss_pairs=[]
for m,data in rows.items():
    complete=[r for r in data.values() if r.get('sampling')]
    norm=[np.linalg.norm(np.asarray(r['sampling']['guidance'][-1]['initial_goal_error']).reshape(4,3),axis=1) for r in complete]
    raw=[r['sampling']['residual_probe'] for r in complete]
    ds=[diversity([r['sampling']['residual_probe']['raw_scaled'] for (sid,k),r in data.items() if sid==e['sample_id'] and r.get('sampling')]) for e in entries]
    metrics[m]={'observed_proposals':len(complete),'unscaled_residual_rms_rad':stats([np.sqrt(np.mean((np.asarray(p['raw_unscaled'])-p['mean'])**2)) for p in raw if p['mean'] is not None]),
        'scaled_residual_rms_rad':stats([np.sqrt(np.mean((np.asarray(p['raw_scaled'])-p['mean'])**2)) for p in raw if p['mean'] is not None]),
        'raw_scaled_pairwise_rms_per_task_rad':stats([d['pairwise_rms_rad'] for d in ds]),
        'pre_repair_position_mm':stats([v[:2].max()*5 for v in norm]),'pre_repair_rotation_deg':stats([v[2:].max() for v in norm])}
for prefix in ('v','g'):
    for suffix in ('025','050'):
        m=prefix+suffix;full=prefix+'100';a,b=summary['methods'][m],summary['methods'][full]
        paired[m]={'relative_to':full,'query_percent_change':100*(a['coupled_queries']/b['coupled_queries']-1),
            'time_percent_change':100*(a['candidate_wall_s']/b['candidate_wall_s']-1),
            'eligible_difference':a['eligible']-b['eligible'],
            'final_pairwise_diversity_retained':a['mean_pairwise_eligible_rms_rad']/b['mean_pairwise_eligible_rms_rad'] if b['mean_pairwise_eligible_rms_rad'] else None,
            'query_wins':sum(rows[m][key]['model_calls']['coupled']<r['model_calls']['coupled'] for key,r in rows[full].items()),
            'query_ties':sum(rows[m][key]['model_calls']['coupled']==r['model_calls']['coupled'] for key,r in rows[full].items())}
        amplitude=.25 if suffix=='025' else .5
        for key,r in rows[m].items():
            zero,one=rows['mean_zero'][key],rows[full][key]
            if not all(v.get('sampling') for v in (r,zero,one)):continue
            errors=[np.asarray(v['sampling']['guidance'][-1]['initial_goal_error']) for v in (r,zero,one)]
            actual,e0,e1=errors;linear=e0+amplitude*(e1-e0)
            nonlinearity.append({'method':m,'task':key[0],'candidate':key[1],
                'normalized_pose_vector_relative_interpolation_error':float(np.linalg.norm(actual-linear)/max(np.linalg.norm(actual-e0),1e-9))})
for key,r in rows['g100'].items():
    v=rows['v100'][key];zero=rows['mean_zero'][key]
    if not all(x.get('sampling') for x in (r,v,zero)):continue
    pg,pv=[x['sampling']['residual_probe'] for x in (r,v)]
    dg=np.asarray(pg['raw_unscaled'])-pg['mean'];dv=np.asarray(pv['raw_unscaled'])-pv['mean']
    rg=float(np.sqrt(np.mean(dg**2)));rv=float(np.sqrt(np.mean(dv**2)))
    e0=np.asarray(zero['sampling']['guidance'][-1]['initial_goal_error'])
    eg=np.asarray(r['sampling']['guidance'][-1]['initial_goal_error'])-e0
    ev=np.asarray(v['sampling']['guidance'][-1]['initial_goal_error'])-e0
    loss_pairs.append({'task':key[0],'candidate':key[1],'goal_vs_v_residual_amplitude_ratio':rg/rv if rv else None,
        'goal_vs_v_pose_change_per_residual_rms_ratio':float((np.linalg.norm(eg)/rg)/(np.linalg.norm(ev)/rv)) if rg and rv and np.linalg.norm(ev) else None,
        'residual_cosine_similarity':float(np.sum(dg*dv)/max(np.linalg.norm(dg)*np.linalg.norm(dv),1e-12))})
result={'status':'PASS','methods':metrics,'paired_amplitudes':paired,'interpolation_diagnostics':nonlinearity,
    'goal_loss_vs_v_same_seed':loss_pairs,'goal_loss_vs_v_summaries':{k:stats([r[k] for r in loss_pairs if r[k] is not None]) for k in ('goal_vs_v_residual_amplitude_ratio','goal_vs_v_pose_change_per_residual_rms_ratio','residual_cosine_similarity')},
    'interpolation_summary_by_method':{m:stats([r['normalized_pose_vector_relative_interpolation_error'] for r in nonlinearity if r['method']==m]) for m in paired},
    'additional_model_or_physics_calls':0,'scope':'saved TRAIN observations; interpolation includes common smoothing/bounds and coupled propagation; not isolated dynamics nonlinearity or independent generalization',
    'gate_redefinition':False}
atomic_json(root/'ff_prior_probe/evidence/mechanism_diagnostics.json',result)
print({'paired_amplitudes':paired,'loss_summary':result['goal_loss_vs_v_summaries']})
'@ | & .\.venv\Scripts\python.exe -
    if ($LASTEXITCODE -ne 0) { throw "Saved mechanism analysis failed: $LASTEXITCODE" }
} finally {
    Pop-Location
    $env:PYTHONIOENCODING = $previousEncoding
    $env:FF_PROBE_ANALYSIS_OUTPUT = $previousOutput
}
