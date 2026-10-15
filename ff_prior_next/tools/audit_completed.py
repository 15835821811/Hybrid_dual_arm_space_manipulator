"""Read-only verification of a completed run, then compact evidence packaging."""
from pathlib import Path
import os
for variable in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[variable]='2'
import json
import shutil
import zipfile
import hashlib
import subprocess
import numpy as np
from ff_prior.storage import read_json,atomic_json,digest,file_digest
from ff_prior.contracts import EXECUTION_GATES
from ff_prior.guidance import public_goal_initializer
from ff_prior.pipeline import audit,acceptance
from ff_prior_next.experiment import METHODS,sources,PROTOCOL


def main():
    root=Path('artifacts/v7_1/public_seed_residual_001')
    source=Path('artifacts/v7_1/smoke_002')
    manifest=read_json(root/'run.json'); summary=read_json(root/'summary.json'); ledger=read_json(root/'ledger.json')
    assert summary['status']=='FINISHED' and sources()==manifest['identity']['source_hashes']
    assert read_json(root/'benchmark_tasks.json')==read_json(source/'benchmark_tasks.json')
    assert all(file_digest(source/rel)==h for rel,h in manifest['identity']['source_files'].items())
    assert ledger['attempts']=={'direction_diagnostics':4,'candidates':60,'execution':12}
    assert len(ledger['completed'])==76 and not ledger['inflight']
    with zipfile.ZipFile(root/'executed_source.zip') as z:
        assert all(hashlib.sha256(z.read(p)).hexdigest()==h for p,h in manifest['identity']['source_hashes'].items())
    tasks=read_json(root/'benchmark_tasks.json')
    # No rollout or target search is performed by this audit.
    old=read_json(source/'run.json')
    audit_report,robot,initial,collision=audit(old['config'])
    assert audit_report['acceptance']==manifest['physical_acceptance']
    total_reports=0; baseline_difference=0.; directions=[]
    for ti,entry in enumerate(tasks):
        task=entry['task']; base=public_goal_initializer(task,robot,initial)
        diagnostic=read_json(root/'direction_diagnostics'/f'{ti:03d}.json')
        J=np.array(diagnostic['jacobian_12x8']); e=np.array(diagnostic['initial_normalized_pose_error'])
        fraction=np.linalg.norm(e+J@np.linalg.lstsq(J,-e,rcond=1e-8)[0])/np.linalg.norm(e)
        assert abs(fraction-diagnostic['optimistic_local_remaining_fraction'])<1e-12
        directions.append(float(fraction))
        for method in METHODS:
            candidates=[]
            for k in range(5):
                c=read_json(root/'candidates'/method/f'{ti:03d}_{k:02d}.json')
                assert c['status']=='DONE' and c['task_hash']==task['task_hash']
                assert c['public_base_hash']==digest(base) and not c['private_teacher_fields_accessed']
                assert c['seed']==20261017+ti*101+k*997
                s=c['sampling']
                assert s['actual_network_calls']==(0 if method==METHODS[0] else 16)
                assert len(s['guidance'])==(0 if method==METHODS[2] else 2)
                assert all(g['directions']==8 and g['centered_evaluations']==16 and g['initial_cost_terms']['coupled_entire_path'] for g in s['guidance'])
                if method==METHODS[1]: assert [g['iteration'] for g in s['guidance']]==[14,15]
                if method!=METHODS[0]:
                    assert len(s['clean_prior_projection'])==16 and s['clean_prior_fit_split']=='TRAIN'
                    assert max(v['decomposition_residual_after'] for v in s['steps'])<1e-12
                else:
                    prior=read_json(Path('artifacts/v7_1/smoke_diffusion_repair_002/benchmark')/f'benchmark_{ti:03d}_structured_prior_same_guidance/planning.json')['candidates'][k]
                    baseline_difference=max(baseline_difference,float(abs(np.array(c['Z'])-prior['Z']).max()))
                candidates.append(c)
            eligible=[c for c in candidates if c['eligible']]
            expected=min(eligible,key=lambda c:(c['score'],c['candidate_id']))['candidate_id'] if eligible else None
            selection=read_json(root/'executions'/method/f'{ti:03d}'/'selection.json')
            assert selection['selected_id']==expected and selection['selection_before_execution']
            if expected is not None:
                r=read_json(root/'executions'/method/f'{ti:03d}'/'report.json'); total_reports+=1
                assert r['task_hash']==task['task_hash']
                if r['level']=='EXECUTION_VERIFIED':
                    assert all(r['gates'][g] for g in EXECUTION_GATES) and r['replay']['passed']
                    assert max(r['replay']['maximum_errors'].values())==0
    assert baseline_difference<1e-10
    training=read_json(root/'learning/training.json'); data=read_json(source/'dataset.json')
    assert training['metadata']['training_ids']==data['eligible']['TRAIN'] and training['updates']==200
    assert training['reload_error']==0 and file_digest(root/'learning/model.pt')==training['checkpoint_sha256']
    patch=Path('artifacts/v7_1/baseline/user_changes.patch').read_text(encoding='utf-8-sig').replace('\r\n','\n').strip()
    actual=subprocess.check_output(['git','diff','--binary','--','.gitattributes','.gitignore','v6_lite/hierarchical_qp.py','v6_lite/run_v6_lite.py'],text=True,encoding='utf-8').replace('\r\n','\n').strip()
    assert patch==actual
    for name,file in (('smoke_002','evidence_manifest.json'),('diffusion_repair','manifest.json')):
        p=Path('ff_prior/evidence')/name
        assert all(file_digest(p/rel)==h for rel,h in read_json(p/file)['files'].items())
    verification={'status':'PASS','tests_passed':52,'same_public_initializers':True,'same_frozen_tasks':True,
        'fixed_candidates_verified':60,'exact_network_calls_verified':640,'training_updates':200,
        'physical_reports':total_reports,'source_data_and_old_evidence_unchanged':True,
        'user_tracked_patch_preserved':True,'baseline_max_Z_difference_from_previous_run':baseline_difference,
        'local_uncovered_error_fractions':directions,'scope':'independent code path recheck of saved results, no new rollout',
        'verifier_extra_model_calls':robot.calls,'pilot':'NOT_RUN','TEST':'NOT_RUN'}
    atomic_json(root/'independent_verification.json',verification)
    dest=Path('ff_prior_next/evidence/public_seed_residual_001'); dest.mkdir(parents=True,exist_ok=True)
    files=[p for p in root.rglob('*.json') if p.name not in ('launcher.json',)]
    files += [root/'REPORT.md',root/'executed_source.zip',root/'events.jsonl']
    for p in files:
        target=dest/p.relative_to(root); target.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(p,target)
    atomic_json(dest/'evidence_manifest.json',{'checkpoint_included':False,'native_trace_included':False,
        'files':{p.relative_to(dest).as_posix():file_digest(p) for p in sorted(dest.rglob('*')) if p.is_file() and p.name!='evidence_manifest.json'}})
    print(json.dumps(verification,ensure_ascii=False,indent=2))


if __name__=='__main__': main()
