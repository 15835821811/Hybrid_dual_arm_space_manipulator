"""Compact raw M1 statistics; never evaluates a model or physical simulator."""
import numpy as np
from ff_prior.storage import read_json,file_digest
from ff_prior_pilot_train.storage import atomic_json
from ff_prior_full_control.tools.collect_completed import candidate,execution,stats,dist
from ff_prior_control_transfer.protocol import ROOT,OUT,PACKAGE,P


def main():
    sel=read_json(OUT/'frozen_tasks.json');codec=read_json(OUT/'codec.json');basis=np.asarray(codec['basis']);rows=[];paired=[];recon=[];ex=[]
    fixed=[(r,seed+j*997) for j,r in enumerate(sel['diagnostic']) for seed in P['diagnostic_seeds']]
    for j,(task,seed) in enumerate(fixed):
        pair={}
        for phase in ('before','after'):
            d=OUT/'TRAIN_diagnostic'/phase/f'{j:02d}'
            r=candidate(d/'validation.json','TRAIN',phase,j%2,task['sample_id'],j)
            ident=read_json(d/'identity.json');assert ident['seed']==seed and ident['task_hash']==task['task']['task_hash']
            with np.load(d/'controls.npz') as a:
                error=a['raw']-a['teacher'];modes=basis.T@error
                r.update(raw_teacher_rmse_rad=float(np.sqrt(np.mean(error**2))),
                    raw_teacher_max_rad=float(abs(error).max()),low_error_rms_rad=float(np.sqrt(np.mean(modes[:6]**2))),
                    detail_error_rms_rad=float(np.sqrt(np.mean(modes[6:]**2))),
                    raw_prepare_max_rad=float(abs(a['raw']-a['prepared']).max()))
            r.update(seed=seed,Q=max(r['position_mm']/5,r['rotation_deg']) if r['position_mm'] is not None else None,
                position_pass=r['position_mm'] is not None and r['position_mm']<=5,
                rotation_pass=r['rotation_deg'] is not None and r['rotation_deg']<=1)
            rows.append(r);pair[phase]=r
        a,b=pair['before'],pair['after'];paired.append(dict(sample_id=task['sample_id'],seed=seed,
            before_Q=a['Q'],after_Q=b['Q'],before_eligible=a['eligible'],after_eligible=b['eligible'],
            before_position_mm=a['position_mm'],after_position_mm=b['position_mm'],
            before_rotation_deg=a['rotation_deg'],after_rotation_deg=b['rotation_deg'],
            teacher_rmse_before=a['raw_teacher_rmse_rad'],teacher_rmse_after=b['raw_teacher_rmse_rad']))
    for task in sel['reconstruction']:
        sid=task['sample_id'];d=OUT/'reconstruction'/sid;r=candidate(d/'validation.json','TRAIN','reconstruction',0,sid,0)
        r['preparation']=read_json(d/'preparation.json');recon.append(r)
        if sid in sel['reconstruction_execution_ids']:ex.append(execution(d/'execution','TRAIN','reconstruction',0,sid))
    phases={}
    for phase in ('before','after'):
        group=[r for r in rows if r['method']==phase];summary=stats(group)
        summary.update(position_pass=sum(r['position_pass'] for r in group),rotation_pass=sum(r['rotation_pass'] for r in group),
            Q=dist([r['Q'] for r in group]),teacher_control_rmse_rad=dist([r['raw_teacher_rmse_rad'] for r in group]),
            teacher_control_max_rad=dist([r['raw_teacher_max_rad'] for r in group]),
            low_error_rms_rad=dist([r['low_error_rms_rad'] for r in group]),detail_error_rms_rad=dist([r['detail_error_rms_rad'] for r in group]))
        phases[phase]=summary
    g=read_json(OUT/'TRAIN_diagnostic/gate.json');ledger=read_json(OUT/'ledger.json')
    status=read_json(OUT/'status.json');assert status['status']=='FINISHED_WITH_BLOCKED_ITEMS'
    assert not (OUT/'learning').exists() and not (OUT/'unguided').exists() and not (OUT/'comparison').exists()
    result=dict(run=OUT.name,status=status,migration=read_json(OUT/'TRAIN_diagnostic/migration.json'),gate=g,
        reconstruction=stats(recon,ex),reconstruction_rows=recon,executions=ex,diagnostic=phases,pairs=paired,
        loss_relative_reduction=1-g['after_loss']/g['before_loss'],Q_median_relative_reduction=1-g['after']['median_Q']/g['before']['median_Q'],
        Q_improved=sum(p['after_Q']<p['before_Q'] for p in paired),Q_worsened=sum(p['after_Q']>p['before_Q'] for p in paired),
        previous_qualified_lost=[p for p in paired if p['before_eligible'] and not p['after_eligible']],
        downstream=dict(candidate_training='BLOCKED_NOT_RUN',VAL='BLOCKED_NOT_RUN',DEV='BLOCKED_NOT_RUN',APPLICATION_DEV='BLOCKED_NOT_RUN',new_candidates=0),
        ledger={k:v for k,v in ledger.items() if k!='completed'},completed_units=len(ledger['completed']),budget=read_json(OUT/'budget.json'),
        audit=read_json(OUT/'independent_verification.json'),detail_limit_normalized=dist(codec['detail_limit']),detail_scale_rad=dist(codec['latent_scale'][102:]),
        raw_sources={p.relative_to(ROOT).as_posix():file_digest(p) for p in [OUT/'status.json',OUT/'summary.json',OUT/'TRAIN_diagnostic/gate.json',OUT/'TRAIN_diagnostic/migration.json',OUT/'ledger.json']})
    atomic_json(PACKAGE/'evidence/completed/statistics.json',result)
    atomic_json(PACKAGE/'evidence/completed/diagnostic_rows.json',rows)
    print({k:result[k] for k in ['loss_relative_reduction','Q_median_relative_reduction','Q_improved','Q_worsened','previous_qualified_lost']})


if __name__=='__main__':main()
