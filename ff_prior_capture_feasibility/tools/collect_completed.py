"""Saved-evidence extraction; no new simulation, optimization or training."""
from pathlib import Path
from collections import Counter
import numpy as np
from ff_prior.storage import read_json, file_digest
from ff_prior_pilot_train.storage import atomic_json
from ff_prior_capture_feasibility.protocol import OUT,PACKAGE,P

def main():
    rows=[]; solvers=[]; trials=[]; linear=[]
    for d in sorted((OUT/'teachers').iterdir()):
        r=read_json(d/'result.json'); v=read_json(d/'validation.json'); initial=read_json(d/'initial.json')
        iters=[read_json(p) for p in sorted((d/'iterations').glob('*/record.json'))]
        for a in iters:
            solvers.append(a['solver']); trials.extend(a['trials'])
            linear.append({'seed':d.name,'iteration':a['iteration'],'zero_step_pose_feasible':a['before']['Q']<=P['solver_goal_fraction'],
                           'returned_linear_pose_feasible':a['solver']['linear_pose_feasible'],'success':a['solver']['success']})
        vv=v['validation']
        rows.append({'seed':d.name,'iterations':len(iters),'accepted':sum(a['accepted'] for a in iters),'initial_Q':initial['Q'],
                     'initial_clearance_mm':initial['minimum_clearance_m']*1000,'search_final_Q':r['Q'],
                     'search_clearance_mm':r['minimum_clearance_m']*1000,'eligible':v['eligible'],'goal_pass':v['frozen_goal_pass'],
                     'position_mm':np.asarray(vv['frozen_goal_position_error_m'])*1000,'rotation_deg':np.rad2deg(vv['frozen_goal_rotation_error_rad']),
                     'validation_clearance_mm':vv['geometry']['minimum_clearance_m']*1000,
                     'geometry_step_mm':vv['geometry']['maximum_geometry_step_m']*1000,
                     'failing_gates':[k for k,val in vv['gates'].items() if not val],
                     'validation_grids':v['validation_grids'],'numerical':vv['numerical']})
    legacy=read_json(OUT/'legacy_local/repair.json'); lv=read_json(OUT/'legacy_local/validation.json')
    ledger=read_json(OUT/'ledger.json')
    result={'status':read_json(OUT/'status.json'),'raw_audit':read_json(OUT/'independent_verification.json'),
            'teachers':rows,'model_qualified':sum(r['eligible'] for r in rows),'pose_qualified':sum(r['goal_pass'] for r in rows),
            'teacher_iterations':len(solvers),'accepted_updates':sum(r['accepted'] for r in rows),
            'solver_success':sum(s['success'] for s in solvers),'linear_pose_feasible_return':sum(s['linear_pose_feasible'] for s in solvers),
            'zero_feasible_solver_return_infeasible':sum(r['zero_step_pose_feasible'] and not r['returned_linear_pose_feasible'] for r in linear),
            'linear_zero_check':linear,'trial_count':len(trials),'quick_rejects':sum(t['status']=='QUICK_REJECTED' for t in trials),
            'failing_gate_counts':dict(Counter(k for r in rows for k in r['failing_gates'])),
            'local':read_json(OUT/'local_subspaces/result.json'),
            'legacy':{'coupled':legacy['used'],'validation':lv['validation'],'goal_pass':lv['frozen_goal_pass'],
                      'stages':[{'iteration':s['iteration'],'coupled_calls':s['coupled_calls'],
                                 'solvers':[h['solver'] for h in s['history']],
                                 'accepted':sum(t.get('accepted',False) for h in s['history'] for t in h['trials'])} for s in legacy['stages']]},
            'ledger':ledger,'infeasibility_proven':False,'TRAIN_added':0,'actual_executions':0,'TEST':False}
    atomic_json(PACKAGE/'evidence/completed/compact.json',result)
    hashes={p.relative_to(OUT).as_posix():file_digest(p) for p in sorted(OUT.rglob('*')) if p.is_file()}
    atomic_json(PACKAGE/'evidence/completed/raw_manifest.json',{'root':str(OUT),'files':hashes})
    print({k:result[k] for k in ('model_qualified','pose_qualified','teacher_iterations','accepted_updates','solver_success','linear_pose_feasible_return','zero_feasible_solver_return_infeasible','trial_count','quick_rejects','failing_gate_counts')})
if __name__=='__main__':main()
