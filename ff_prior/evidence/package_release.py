"""Package compact measured evidence; never copy full datasets/checkpoints."""
from pathlib import Path
import shutil
import zipfile
import numpy as np
from ff_prior.storage import read_json, atomic_json, file_digest


def package():
    run=Path('artifacts/v7_1/smoke_002')
    out=Path('ff_prior/evidence/smoke_002')
    out.mkdir(parents=True,exist_ok=True)
    names=['summary.json','benchmark.json','ledger.json','run.json','baseline_audit.json',
           'dataset.json','splits.json','audit_selection.json','benchmark_tasks.json','scene.json',
           'pair_policy.json','initial_state.npz','publication.json','failures.json',
           'release_verification.json','learning/scaler.json','learning/training.json',
           'records/main_000000.json','records/dev_000004.json','records/dev_000083.json']
    names += [p.relative_to(run).as_posix() for p in (run/'audits').glob('*/*.json')]
    names += [p.relative_to(run).as_posix() for p in (run/'benchmark').rglob('*.json')]
    names += [p.relative_to(run).as_posix() for p in (run/'visuals').glob('*')]
    for name in names:
        dest=out/name
        dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(run/name,dest)
    shutil.copyfile('artifacts/v7_1/tests_final.xml',out/'tests.xml')
    shutil.copyfile('artifacts/v7_1/geometry_counterexamples.json',out/'geometry_counterexamples.json')
    shutil.copyfile('artifacts/v7_1/resume_002_before.json',out/'resume_before.json')
    manifest=read_json(run/'run.json')
    with zipfile.ZipFile(out/'executed_source.zip','w',zipfile.ZIP_DEFLATED) as z:
        for name,expected in manifest['source_hashes'].items():
            assert file_digest(name)==expected
            z.write(name,name)
        z.write('configs/ff_prior/v7_1.proposed.yaml','configs/ff_prior/v7_1.proposed.yaml')
    sample='dev_000004'
    with np.load(run/f'arrays/{sample}.npz') as a:
        np.savez_compressed(out/'reference_sample.npz',**{k:a[k] for k in a.files if k in
            ['Z','C','knots','degree','T_ref','T_hold','view_time','view_q','view_qd','view_qdd',
             'prediction_base','prediction_ee','prediction_phase']})
    with np.load(run/f'audits/{sample}/native_trace.npz') as a, np.load(run/'initial_state.npz') as state:
        indices=np.linspace(0,len(a['time'])-1,101,dtype=int)
        np.savez_compressed(out/'portable_sample.npz',ctrl=a['ctrl'],initial_state=state['values'],
            checkpoint_indices=indices,**{k:a[k][indices] for k in ('qpos','qvel','qacc_warmstart','act','time')})
    actual=read_json(run/f'audits/{sample}/report.json')
    atomic_json(out/'portable_sample.json',{'sample_id':sample,'model_hash':actual['scene_model_hash'],
        'initial_state_id':actual['initial_state_id'],'ctrl_hash':actual['ctrl_hash'],
        'source_native_trace_sha256':file_digest(run/f'audits/{sample}/native_trace.npz'),
        'scope':'full 5250 controls and initial state; 101 checkpoints; full trace stays local'})
    first=Path('artifacts/v7_1/smoke_001')
    old=read_json('ff_prior/evidence/smoke_001/summary.json')
    new=read_json(run/'summary.json')
    def hashes(p):
        return {r['reference_hash'] for f in (p/'records').glob('*.json')
                if (r:=read_json(f))['validation_level']=='MODEL_VALID'}
    first_refs,final_refs=hashes(first),hashes(run)
    costs={'canonical_run':'smoke_002','engineering_repeat':'smoke_001',
        'proposal_attempts_both_runs':800,'accepted_records_both_runs':798,
        'unique_accepted_reference_hashes':len(first_refs|final_refs),
        'repeated_reference_hashes':len(first_refs&final_refs),
        'physical_executions_both_runs':50,'independent_full_replays_both_runs':50,
        'generation_and_measured_stage_wall_s_both_runs':sum(old['resource_wall_s'].values())+sum(new['resource_wall_s'].values()),
        'training_wall_s_both_runs':old['training']['wall_s']+new['training']['wall_s'],
        'stage_cpu_s_both_runs':sum(old['resource_cpu_s'].values())+sum(new['resource_cpu_s'].values()),
        'training_cpu_s_both_runs':old['training']['cpu_s']+new['training']['cpu_s'],
        'first_run_missing_counters':['explicit mj_forward','dataset split stage wall/cpu'],
        'timing_scope':'Measured pipeline stages and training; excludes setup/model compile, engineering tests, reporting/rendering, release audit and portable replay. Not full task elapsed time.',
        'storage_bytes_local_runs':{p.name:sum(f.stat().st_size for f in p.rglob('*') if f.is_file()) for p in (first,run)}}
    atomic_json(out/'cumulative_cost.json',costs)
    checklist=read_json('configs/ff_prior/acceptance_checklist.json')
    evidence={
        'workspace_and_contract_audit':['baseline_audit.json','run.json','../../../docs/V7_1_BASELINE_AUDIT.md'],
        'shared_spline_decoder':['tests.xml','release_verification.json','reference_sample.npz'],
        'bounded_low_cost_generator':['ledger.json','release_verification.json'],
        'geometry_counterexamples':['tests.xml','geometry_counterexamples.json'],
        'coupled_predictor':['tests.xml','records/main_000000.json'],
        'numerical_convergence':['records/main_000000.json','records/dev_000083.json','tests.xml'],
        'goal_provenance':['tests.xml','release_verification.json'],
        'validation_levels':['tests.xml','release_verification.json'],
        'split_and_condition_audit':['splits.json','release_verification.json'],
        'normalizer_audit':['learning/scaler.json','release_verification.json'],
        'budget_resume_integrity':['tests.xml','resume_before.json','release_verification.json'],
        'physical_execution':['audits/dev_000004/report.json','release_verification.json'],
        'independent_replay':['audits/dev_000004/report.json','release_verification.json'],
        'native_state_whole_body_check':['audits/dev_000004/report.json','release_verification.json'],
        'audit_selection_integrity':['audit_selection.json','release_verification.json'],
        'training_smoke':['learning/training.json','release_verification.json'],
        'in_denoising_guidance':['benchmark/benchmark_000_diffusion_in_denoising_guidance/planning.json','release_verification.json'],
        'matched_minimal_comparison':['benchmark.json','release_verification.json'],
        'strict_five_candidate_protocol':['tests.xml'],
        'held_out_benchmark':['benchmark.json'],
        'cost_and_scope_report':['cumulative_cost.json','summary.json','../../../docs/V7_1_OFFLINE_COST_REPORT.md']}
    checklist.update(schema_version='ff_prior_v7_1_measured_acceptance',package_type='measured_smoke_evidence',
        repository_implementation_status='IMPLEMENTED_DEV_SCOPE',robot_tests_status='43_PASSED_INCLUDING_NATIVE_PHYSICS',
        measured_output_counts={'main_attempted':200,'main_valid':200,'dev_attempted':200,'dev_valid':199,
            'rejected':1,'offline_executions_and_replays_each':24,'benchmark_executions_and_replays_each':1})
    for item in checklist['checks']:
        item['status']='NOT_RUN' if item['phase']=='STUDY' else 'PASS'
        item['evidence']=evidence[item['id']]
        if item['phase']=='STUDY':
            item['reason']='Explicit study/strict/final TEST execution was not authorized by default; software guard and quota tests passed.'
        if item['phase']=='M5':
            item['claim_limit']='Implementation and DEV experiment completed; diffusion benefit NOT_ESTABLISHED.'
    atomic_json(out/'acceptance_results.json',checklist)
    atomic_json(out/'evidence_manifest.json',{'files':{str(p.relative_to(out)).replace('\\','/'):file_digest(p)
        for p in sorted(out.rglob('*')) if p.is_file() and p.name!='evidence_manifest.json'},
        'large_data_included':False,'checkpoint_included':False,'canonical_run':str(run)})
    print(costs)
    print('compact evidence bytes',sum(p.stat().st_size for p in out.rglob('*') if p.is_file()))


if __name__=='__main__':
    package()
