param([string]$Output = 'artifacts/v7_1/qualified_expansion_002')
$ErrorActionPreference = 'Stop'
$previousEncoding = $env:PYTHONIOENCODING
$previousOutput = $env:FF_EXPAND_ANALYSIS_OUTPUT
Push-Location -LiteralPath (Split-Path $PSScriptRoot -Parent)
try {
    $env:PYTHONIOENCODING = 'utf-8'
    $env:FF_EXPAND_ANALYSIS_OUTPUT = $Output
    @'
from pathlib import Path
from collections import Counter
import os
import numpy as np
from ff_prior.storage import read_json,atomic_json,file_digest,digest
from ff_prior.spline import from_arrays
from ff_prior.initial_state import STATE

out=Path(os.environ['FF_EXPAND_ANALYSIS_OUTPUT']).resolve()
status=read_json(out/'status.json');ledger=read_json(out/'ledger.json')
rows=[];pending=[];motion=[];goals=[];initials=[];proofs=[];failures=Counter();scales=Counter();durations=Counter()
for p in sorted((out/'records').glob('*.json')):
    r=read_json(p);sid=r['sample_id']
    if ledger['completed'].get(sid,{}).get('status')!='ADMITTED':pending.append(sid);continue
    proof=read_json(out/r['generation_proof'])
    assert proof['admitted'] and proof['validation_result']['eligible']
    assert r['arrays_sha256']==file_digest(out/r['arrays'])
    assert r['initial_state_sha256']==file_digest(out/r['complete_initial_state'])
    with np.load(out/r['arrays']) as a:
        ref=from_arrays(a);assert ref.reference_hash==r['reference_hash']
        for name,order in [('view_q',0),('view_qd',1),('view_qdd',2)]:
            assert np.allclose(a[name],ref.evaluate(a['view_time'],order),rtol=0,atol=1e-12)
        motion.append(float(np.max(abs(ref.control-ref.q0))))
    with np.load(out/r['complete_initial_state']) as a:
        assert digest({'values':a['values'],'model_hash':r['task']['contract']['robot_hash'],'state_mask':int(STATE)})==r['task']['contract']['initial_state_id']
    rows.append(r);initials.append(r['condition']['initial_features']);goals.append(np.asarray(r['condition']['goals'])[:,:3,3])
    scales[str(proof['provenance']['motion_scale'])]+=1;durations[str(r['condition']['duration'])]+=1
    v=r['validation'];proofs.append([max(v['frozen_goal_position_error_m']),max(v['frozen_goal_rotation_error_rad']),v['geometry']['minimum_clearance_m'],v['geometry']['maximum_geometry_step_m']])
for p in (out/'generation').glob('*.json'):
    r=read_json(p)
    if not r['admitted']:
        failures[r.get('reason','UNKNOWN')]+=1
def bounds(values):
    a=np.asarray(values);return {'minimum':a.min(0).tolist(),'maximum':a.max(0).tolist()} if len(a) else None
result={'status':'PASS_SAVED_ARRAYS_ONLY','run_status_at_read':status['status'],'run_stage_at_read':status['stage'],
    'audited_saved_rows':len(rows),'pending_generation_commit':pending,'known_execution_failures':sum(r['execution_status']=='FAIL' for r in rows),
    'execution_levels':dict(Counter(r['validation_level'] for r in rows)),
    'contexts':dict(Counter(r['context_id'] for r in rows)),
    'new_empty_scene_families':len({r['family_id'] for r in rows if r['family_id'].startswith('expand_family_')}),
    'distinct_initial_states':len({r['task']['contract']['initial_state_id'] for r in rows}),
    'distinct_world_tasks':len({r['task']['task_hash'] for r in rows}),
    'external_obstacle_rows':sum(bool(r['scene']['objects']) for r in rows),'motion_scale_counts':dict(scales),'duration_counts':dict(durations),
    'initial_q17_range_rad':bounds(initials),'maximum_control_excursion_rad':bounds(motion),'world_goal_position_range_m':bounds(goals),
    'saved_gate_statistics_columns':['max_dual_position_error_m','max_dual_rotation_error_rad','minimum_clearance_m','maximum_geometry_step_m'],
    'saved_gate_statistics':bounds(proofs),'generation_rejection_reasons':dict(failures),
    'actual_array_and_state_hashes_checked':True,'saved_training_views_match_shared_decoder':True,
    'new_model_physics_calls':0,'completed_run_claim':False,'new_obstacle_scene_or_homotopy_claim':False,
    'note':'Live snapshot if worker is running; admission is MODEL_VALID only unless individually executed. Distinct tasks/controls do not establish useful route diversity.'}
target=Path('ff_prior_expand/evidence/saved_coverage_diagnostics.json');atomic_json(target,result)
print({'saved_report':str(target),'rows_checked':len(rows),'contexts':result['contexts'],'levels':result['execution_levels'],'new_empty_families':result['new_empty_scene_families'],'rejections':dict(failures)})
'@ | & '.\.venv\Scripts\python.exe' -
    if ($LASTEXITCODE -ne 0) { throw "Saved-array audit failed: $LASTEXITCODE" }
} finally {
    Pop-Location
    $env:PYTHONIOENCODING = $previousEncoding
    $env:FF_EXPAND_ANALYSIS_OUTPUT = $previousOutput
}
