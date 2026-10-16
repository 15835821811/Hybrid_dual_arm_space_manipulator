"""Post-hoc numerical diagnosis only: never modifies formal candidate results."""
from pathlib import Path
import time
import numpy as np
from ff_prior.storage import read_json,atomic_json,file_digest
from ff_prior.pipeline import audit,acceptance
from ff_prior.spline import Reference
from ff_prior.coupled_predictor import predict_coupled,pose_errors
from ff_prior.model_gates import validate_model
from ff_prior_loop.data import runtime

CONTRACT={'stage':'POSTHOC_DIAGNOSIS','selection':'first 3 trained_joint candidates in frozen task/id order with goal pass and only geometry_refinement failed',
    'max_cases':3,'coupled_intervals':[512,1024],'max_additional_coupled_queries':6,'wall_budget_s':120,
    'formal_counts_changed':False,'training_changed':False,'goal_changed':False,'physical_thresholds_changed':False,
    'actual_execution':False,'not_part_of_paired_candidate_budget':True,'purpose':'separate finite-grid obstruction from physical collision/goal failure'}


def main():
    out=Path('artifacts/v7_1/qualified_guidance_loop_001'); dest=out/'posthoc_resolution'
    if (dest/'results.json').exists(): raise ValueError('diagnosis already exists; refuse rerun')
    assert read_json(out/'status.json')['status'] in ('FINISHED','FINISHED_WITH_BLOCKED_ITEMS')
    tasks=read_json(out/'frozen_tasks.json')['evaluation_entries']; selected=[]
    for e in tasks:
        for p in sorted((out/'candidates'/e['sample_id']/'trained_diffusion_joint').glob('*.json')):
            r=read_json(p)
            failed=[k for k,v in r.get('validation',{}).get('gates',{}).items() if not v]
            if r.get('frozen_goal_pass') and failed==['geometry_refinement']:
                selected.append((e,p,r))
    selected=selected[:3]
    protected=[out/'summary.json',out/'expansion_decision.json']+list((out/'candidates').rglob('*.json'))
    hashes={str(p):file_digest(p) for p in protected}
    atomic_json(dest/'plan.json',{'contract':CONTRACT,'selected':[str(p) for _,p,_ in selected],'formal_hashes_before':hashes})
    config=read_json('artifacts/v7_1/smoke_002/run.json')['config']; _,canonical,initial,_=audit(config); profile=acceptance(config)
    deadline=time.time()+120; results=[]; total=0
    for e,p,r in selected:
        rr,ii,cc,_=runtime(e,canonical,initial); task=e['task']; c=task['condition']; before=rr.calls.copy()
        ref=Reference(r['Z'],c['initial_features'],c['duration']); coarse=predict_coupled(ii,ref,rr,512,deadline)
        fine=predict_coupled(ii,ref,rr,1024,deadline); pos,rot=pose_errors(fine['ee'][-1],coarse['ee'][-1])
        body=np.linalg.norm(fine['body'][::2]-coarse['body'],axis=-1)
        centers=np.linalg.norm(fine['geom'][::2]-coarse['geom'],axis=-1)
        rotations=(fine['geom_rotation'][::2]-coarse['geom_rotation']).reshape(513,rr.model.ngeom,3,3)
        extent=float(max(body.max(),(centers+np.linalg.norm(rotations,axis=(-1,-2))*rr.model.geom_rbound).max()))
        comparison={'coarse':512,'fine':1024,'terminal_position_m':pos,'terminal_rotation_rad':rot,'aligned_body_extent_m':extent,
            'passed':bool(max(pos)<=.001 and max(rot)<=np.deg2rad(.2) and extent<=.001)}
        validation=validate_model(ref,fine,profile,rr,ii,cc,{'passed':comparison['passed'],'comparisons':[comparison]},deadline)
        gp,gr=pose_errors(fine['ee'][-1],c['goals'])
        calls={k:rr.calls[k]-before[k] for k in before}; total+=calls['coupled']
        results.append({'candidate':str(p),'task_hash':task['task_hash'],'original_reference_hash':r['reference_hash'],
            'original_max_geometry_step_m':r['validation']['geometry']['maximum_geometry_step_m'],
            'diagnostic_1024_validation':validation,'position_error_m':gp,'rotation_error_rad':gr,
            'diagnostic_model_and_goal_pass':validation['validation_level']=='MODEL_VALID' and max(gp)<=.005 and max(gr)<=np.deg2rad(1),
            'model_calls':calls,'formal_status_remains':r['validation']['validation_level'],'EXECUTION_VERIFIED':False})
    assert all(file_digest(p)==h for p,h in hashes.items()) and total<=6
    atomic_json(dest/'results.json',{'contract':CONTRACT,'results':results,'additional_coupled_queries':total,
        'formal_records_unchanged':True,'expanded':False})
    print({'cases':len(results),'diagnostic_model_goal_pass':sum(r['diagnostic_model_and_goal_pass'] for r in results),'extra_coupled':total})


if __name__=='__main__': main()
