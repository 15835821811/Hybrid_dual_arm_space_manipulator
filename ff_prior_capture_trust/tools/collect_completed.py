"""Terminal-state evidence extraction only; no physics or optimization."""
from pathlib import Path
import numpy as np
from ff_prior.storage import read_json,file_digest
from ff_prior_pilot_train.storage import atomic_json
from ff_prior_capture_trust.protocol import ROOT,PACKAGE,OUT,PREVIOUS

def main():
    status=read_json(OUT/'status.json');assert status['status']!='RUNNING'
    rows=[]
    for d in sorted((OUT/'teachers').iterdir()):
        if not (d/'validation.json').exists():continue
        v=read_json(d/'validation.json');r=read_json(d/'result.json');old=read_json(PREVIOUS/'teachers'/d.name/'validation.json')
        records=[]
        for p in sorted((d/'iterations').glob('*/record.json')):
            a=read_json(p);s=a['solver']
            with np.load(p.with_name('linearization.npz')) as arrays:
                sparse_nodes=len(arrays['phase']);dense_nodes=len(arrays['dense_phase'])
            records.append({'iteration':a['iteration'],'accepted':a['accepted'],'before':a['before'],
                'radius_rad':a['radius_rad'],'next_radius_rad':a['next_radius_rad'],
                'no_nonzero_feasible_step':a['no_nonzero_feasible_step'],'solver_message':s['message'],
                'restored_callback_count':sum(c['restoration_alpha']<1 for c in s['callbacks']),
                'callback_count':len(s['callbacks']),
                'success':s['success'],'solved_feasible':s['solved_feasible'],'returned_feasible':s['returned_feasible'],
                'initial_residual':s['initial_residual'],'solved_residual':s['solved_residual'],'returned_residual':s['returned_residual'],
                'returned_source':s['returned_source'],'initial_objective':s['initial_objective'],'returned_objective':s['returned_objective'],
                'dense_nodes':dense_nodes,'sparse_nodes':sparse_nodes,'active_pairs':len(a['active_pairs']),
                'sparse_agreement':a['sparse_agreement'],'trials':a['trials']})
        rows.append({'id':d.name,'eligible':v['eligible'],'goal_pass':v['frozen_goal_pass'],
            'validation':v['validation'],'validation_grids':v['validation_grids'],'old_validation':old['validation'],
            'stop':r['stop'],'iterations':records})
    representations={p.parent.name:read_json(p) for p in (OUT/'representation').glob('*/validation.json')}
    executions={str(p.parent.relative_to(OUT)):read_json(p) for p in OUT.glob('**/outcome.json')}
    result={'status':status,'sanity':read_json(OUT/'sanity_uniform.json'),'teachers':rows,'representation':representations,
      'executions':executions,'audit':read_json(OUT/'independent_verification.json') if (OUT/'independent_verification.json').exists() else None,
      'budget':read_json(OUT/'budget.json'),'ledger':read_json(OUT/'ledger.json'),'TRAIN_added':0,'TEST':False}
    atomic_json(PACKAGE/'evidence/completed/compact.json',result)
    files={p.relative_to(OUT).as_posix():file_digest(p) for p in sorted(OUT.rglob('*')) if p.is_file()}
    atomic_json(PACKAGE/'evidence/completed/raw_manifest.json',{'root':str(OUT),'files':files})
    print({'status':status['status'],'rows':len(rows),'eligible':sum(r['eligible'] for r in rows),'executions':executions,'files':len(files)})
if __name__=='__main__':main()
