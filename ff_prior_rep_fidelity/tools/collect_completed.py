"""Read saved evidence only; no simulation, optimization or model fitting."""
import shutil
import numpy as np
from ff_prior.storage import read_json,file_digest
from ff_prior_pilot_train.storage import atomic_json
from ff_prior_rep_fidelity.protocol import ROOT,PACKAGE,OUT

def main():
    dest=PACKAGE/'evidence/completed';dest.mkdir(exist_ok=True)
    summary=read_json(OUT/'summary.json');rows=[]
    manifest=read_json(OUT/'candidate_manifest.json')['candidates']
    for candidate in manifest:
        name=candidate['id'];d=OUT/'candidates'/name
        result=read_json(d/'validation.json');v=result['validation']
        with np.load(d/'representation.npz') as a:
            teacher=a['teacher'];output=a['output'];basis=a['basis']
            intermediates={k:float(np.max(abs(a[k]-teacher))) for k in
                           ('raw','smoothed','clipped','projected','output') if k in a}
            endpoint_delta=float(np.max(abs(output[-1]-teacher[-1])))
            reconstruction={'control_max_abs_rad':float(np.max(abs(output-teacher))),
                'control_rms_rad':float(np.sqrt(np.mean((output-teacher)**2))),
                'terminal_control_max_abs_rad':endpoint_delta,
                'basis_orthogonality_max':float(np.max(abs(basis.T@basis-np.eye(basis.shape[1])))),
                'intermediate_control_error_rad':intermediates}
        rows.append({'id':name,'rank':candidate['rank'],'dimensions':candidate['dimensions'],
            'mode':candidate['mode'],'reference_hash':candidate['reference_hash'],
            'model_eligible':result['eligible'],'frozen_goal_pass':result['frozen_goal_pass'],
            'position_mm':(1000*np.asarray(v['frozen_goal_position_error_m'])).tolist(),
            'rotation_deg':np.rad2deg(v['frozen_goal_rotation_error_rad']).tolist(),
            'minimum_clearance_mm':v['geometry']['minimum_clearance_m']*1000,
            'final_intervals':result['validation_grids'][-1]['intervals'],
            'failed_gates':[k for k,value in v['gates'].items() if not value],
            'quick_gates':candidate['quick_gates'],'prepare_metadata':candidate['metadata'],
            'reconstruction':reconstruction,
            'validation_origin':read_json(d/'validation_origin.json'),
            'execution_origin':read_json(d/'execution_origin.json')})
    execution=read_json(OUT/'candidates/r19_raw/execution/report.json')
    stats={'run':'rep_fidelity_001','status':read_json(OUT/'status.json'),
        'rows':rows,'audit':summary['raw_audit'],'budget':summary['budget'],
        'logical_candidates':len(rows),'unique_references':len({r['reference_hash'] for r in rows}),
        'logical_model_passes':sum(r['model_eligible'] for r in rows),
        'unique_model_passes':len({r['reference_hash'] for r in rows if r['model_eligible']}),
        'new_execution':{k:execution[k] for k in ('level','gates','replay','minimum_clearance_m',
          'hold_max_position_error_m','hold_max_rotation_error_rad','final_position_error_m',
          'final_rotation_error_rad','checked_states','native_states','metadata')},
        'no_new_TRAIN':True,'no_training':True,'no_TEST':True}
    atomic_json(dest/'statistics.json',stats)
    for filename in ('summary.json','independent_verification.json','execution_selection.json','ledger.json','status.json','budget.json','initial_state.json'):
        shutil.copy2(OUT/filename,dest/filename)
    for row in rows:
        for filename in ('validation.json','validation_origin.json','execution_origin.json','candidate.json'):
            source=OUT/'candidates'/row['id']/filename
            target=dest/'candidates'/row['id']/filename;target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(source,target)
    target=dest/'execution';target.mkdir(exist_ok=True)
    for filename in ('report.json','initial_state.json','outcome.json','selection.json'):
        shutil.copy2(OUT/'candidates/r19_raw/execution'/filename,target/filename)
    atomic_json(dest/'raw_manifest.json',{'files':{p.relative_to(OUT).as_posix():file_digest(p)
        for p in sorted(OUT.rglob('*')) if p.is_file()},'large_arrays_remain_local':True})
    print('candidates',len(rows),'unique',stats['unique_references'],'model_passes',stats['unique_model_passes'])
    for r in rows:
        print(r['id'],max(r['position_mm']),max(r['rotation_deg']),r['minimum_clearance_mm'],
              r['final_intervals'],r['failed_gates'],r['reconstruction']['terminal_control_max_abs_rad'])

if __name__=='__main__':main()
