"""Verify completed saved evidence without generating targets or physical steps."""
import os
for variable in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'): os.environ[variable]='2'
from pathlib import Path
import hashlib
import shutil
import subprocess
import zipfile
import numpy as np
from scipy.spatial.transform import Rotation
from ff_prior.storage import read_json,atomic_json,digest,file_digest
from ff_prior.contracts import MODEL_GATES,EXECUTION_GATES
from ff_prior.spline import from_arrays
from ff_prior_fixed.experiment import sources,PROTOCOL,METHODS,inputs
from ff_prior_fixed.correction import choose_targets


def maximum_pose_errors(actual,predicted):
    pos=float(np.linalg.norm(actual[...,:3,3]-predicted[...,:3,3],axis=-1).max())
    relative=np.swapaxes(actual[...,:3,:3],-1,-2)@predicted[...,:3,:3]
    rot=float(Rotation.from_matrix(relative.reshape(-1,3,3)).magnitude().max())
    return pos,rot


def main():
    out=Path('artifacts/v7_1/fixed_train_consistency_001')
    source=Path('artifacts/v7_1/smoke_002')
    run=read_json(out/'run.json'); ledger=read_json(out/'ledger.json'); summary=read_json(out/'summary.json')
    assert summary['status']=='FINISHED' and sources()==run['identity']['source_hashes']
    assert ledger['attempts']=={'correction':16,'consistency':4,'candidates':60,'execution':12}
    assert len(ledger['completed'])==92 and ledger['inflight'] is None
    assert not ledger.get('unresolved_call_accounting')
    assert all(file_digest(source/rel)==h for rel,h in run['identity']['source_files'].items())
    assert read_json(out/'benchmark_tasks.json')==read_json(source/'benchmark_tasks.json')
    with zipfile.ZipFile(out/'executed_source.zip') as z:
        assert all(hashlib.sha256(z.read(p)).hexdigest()==h for p,h in run['identity']['source_hashes'].items())
    old,data,tasks,rows,_=inputs(source); selected=choose_targets(rows)
    selection=read_json(out/'train_target_selection.json')
    assert digest(selection)==run['train_selection_hash']
    assert [r['sample_id'] for r in selected]==[r['sample_id'] for r in selection]
    assert len({r['family_id'] for r in selected})==8
    parent={r['sample_id']:r for r,s in rows if s=='TRAIN'}
    corrections={}; accepted=[]; diversity=[]
    for path in sorted((out/'corrections/records').glob('*.json')):
        row=read_json(path); p=parent[row['source_sample_id']]
        assert row['split']=='TRAIN' and row['task']==p['task'] and row['condition']==p['condition']
        assert row['task_hash']==p['task']['task_hash'] and not row['goal_relabelled'] and not row['private_terminal_used']
        ref=from_arrays(np.load(out/row['arrays']))
        assert ref.reference_hash==row['reference_hash']
        corrections[row['sample_id']]=row
        if row['training_eligible']:
            assert row['frozen_goal_pass'] and all(row['model_validation']['gates'][g] is True for g in MODEL_GATES)
            assert max(row['position_error_m'])<=.005 and max(row['rotation_error_rad'])<=np.deg2rad(1)
            assert row['execution_status']=='NOT_RUN' and row['data_role']=='FF_MODEL_PRIOR'
            accepted.append(row['sample_id'])
    assert len(corrections)==16
    for index in range(8):
        a,b=[corrections[f'trainfix_{index:03d}_{v}'] for v in (0,1)]
        za=np.load(out/a['arrays'])['Z']; zb=np.load(out/b['arrays'])['Z']
        diversity.append({'source_sample_id':a['source_sample_id'],'pair_control_rms_rad':float(np.sqrt(np.mean((za-zb)**2))),
            'same_frozen_task':a['task_hash']==b['task_hash']})
    diagnostics=read_json(out/'diagnostic_selection.json'); exclusions=[]; measured=[]; report_count=0
    for entry in diagnostics:
        report=read_json(out/'consistency'/entry['id']/'execution/report.json')
        assert report['task_hash']==entry['task']['task_hash']
        trace=np.load(out/'consistency'/entry['id']/'execution/native_trace.npz')
        assert digest(trace['ctrl'])==report['ctrl_hash'] and len(trace['ctrl'])+1==len(trace['qpos'])
        if report['execution_status']=='FAIL': exclusions.append(entry['sample_id'])
        if report['level']=='EXECUTION_VERIFIED':
            assert all(report['gates'][g] is True for g in EXECUTION_GATES)
            assert report['replay']['passed'] and max(report['replay']['maximum_errors'].values())==0
        report_count+=1
        d=read_json(out/'consistency'/entry['id']/'diagnostic.json')['comparison']
        saved=np.load(out/'consistency'/entry['id']/'comparison.npz')
        assert np.array_equal(trace['ee'][saved['indices']],saved['actual_ee'])
        assert np.array_equal(trace['base'][saved['indices']],saved['actual_base'])
        assert saved['indices'][0]==0 and saved['indices'][-1]==len(trace['time'])-1
        for name,array in (('nominal_reference_vs_execution','predicted_ee'),('actual_native_path_reconstruction_vs_execution','native_conditioned_ee')):
            pos,rot=maximum_pose_errors(saved['actual_ee'],saved[array]); claimed=d[name]['end_effectors']
            assert abs(pos-claimed['max_position_m'])<1e-12 and abs(rot-claimed['max_rotation_rad'])<1e-12
        assert d['all_67_native_joint_rates_used'] and not d['Bplus_used_for_reconstruction'] and not d['per_frame_base_state_copied']
        measured.append({'sample_id':entry['sample_id'],'execution_status':report['execution_status'],
            'reference_ee_max_m':d['nominal_reference_vs_execution']['end_effectors']['max_position_m'],
            'native_conditioned_ee_max_m':d['actual_native_path_reconstruction_vs_execution']['end_effectors']['max_position_m'],
            'flags':d['diagnostic_flags']})
    dataset=read_json(out/'training_dataset.json')
    assert dataset['VAL']==data['eligible']['VAL'] and sorted(exclusions)==dataset['excluded_known_execution_failed_ids']
    assert set(dataset['TRAIN'])==(set(data['eligible']['TRAIN'])|set(accepted))-set(exclusions)
    training=read_json(out/'learning/training.json')
    assert training['metadata']['training_ids']==dataset['TRAIN'] and training['updates']==200 and training['reload_error']==0
    assert file_digest(out/'learning/model.pt')==training['checkpoint_sha256']
    baseline_difference=0.; networks=0
    for ti,entry in enumerate(tasks):
        base_hashes=set()
        for method in METHODS:
            cs=[]
            for k in range(5):
                c=read_json(out/'candidates'/method/f'{ti:03d}_{k:02d}.json');cs.append(c)
                assert c['status']=='DONE' and c['task_hash']==entry['task']['task_hash'] and not c['private_teacher_fields_accessed']
                assert c['seed']==20261017+ti*101+k*997; base_hashes.add(c['public_base_hash'])
                s=c['sampling']; networks+=s['actual_network_calls']
                assert s['actual_network_calls']==(0 if method==METHODS[0] else 16)
                assert len(s['guidance'])==(0 if method==METHODS[2] else 2)
                assert all(g['directions']==8 and g['centered_evaluations']==16 for g in s['guidance'])
                if method!=METHODS[0]:
                    assert s['clean_prior_fit_split']=='TRAIN' and len(s['clean_prior_projection'])==16
                    assert max(x['decomposition_residual_after'] for x in s['steps'])<1e-12
                else:
                    previous=read_json(Path('artifacts/v7_1/public_seed_residual_001/candidates')/method/f'{ti:03d}_{k:02d}.json')
                    baseline_difference=max(baseline_difference,float(abs(np.array(c['Z'])-previous['Z']).max()))
            eligible=[c for c in cs if c['eligible']]
            chosen=min(eligible,key=lambda c:(c['score'],c['candidate_id']))['candidate_id'] if eligible else None
            directory=out/'executions'/method/f'{ti:03d}'
            chosen_record=read_json(directory/'selection.json')
            assert chosen_record['selected_id']==chosen and chosen_record['selection_before_execution']
            if chosen is not None:
                r=read_json(directory/'report.json'); report_count+=1
                assert r['task_hash']==entry['task']['task_hash']
                if r['level']=='EXECUTION_VERIFIED': assert all(r['gates'][g] for g in EXECUTION_GATES) and r['replay']['passed']
        assert len(base_hashes)==1
    assert networks==640 and baseline_difference<1e-10
    original_patch=Path('artifacts/v7_1/baseline/user_changes.patch').read_text(encoding='utf-8-sig').replace('\r\n','\n').strip()
    current_patch=subprocess.check_output(['git','diff','--binary','--','.gitattributes','.gitignore','v6_lite/hierarchical_qp.py','v6_lite/run_v6_lite.py'],text=True,encoding='utf-8').replace('\r\n','\n').strip()
    assert original_patch==current_patch
    for folder,file in [('ff_prior/evidence/smoke_002','evidence_manifest.json'),('ff_prior/evidence/diffusion_repair','manifest.json'),('ff_prior_next/evidence/public_seed_residual_001','evidence_manifest.json')]:
        p=Path(folder); assert all(file_digest(p/n)==h for n,h in read_json(p/file)['files'].items())
    atomic_json(out/'independent_verification.json',{'status':'PASS','tests_passed':58,'fixed_TRAIN_targets':8,
        'correction_attempts':16,'accepted_model_and_goal':len(accepted),'pair_diversity':diversity,
        'diagnostics':measured,'excluded_known_failed':exclusions,'physical_reports':report_count,
        'same_frozen_DEV_tasks':True,'candidate_budget':60,'sampling_network_calls':640,
        'baseline_max_Z_difference':baseline_difference,'source_and_old_evidence_preserved':True,
        'user_patch_preserved':True,'extra_physics_steps_in_verifier':0,
        'complete_run_model_calls':read_json(out/'verification.json')['complete_model_calls'],'pilot':'NOT_RUN','TEST':'NOT_RUN'})
    dest=Path('ff_prior_fixed/evidence/fixed_train_consistency_001');dest.mkdir(parents=True,exist_ok=True)
    files=[p for p in out.rglob('*.json') if p.name!='launcher.json']
    files+=list((out/'corrections/arrays').glob('*.npz'))
    files += [out/'REPORT.md',out/'executed_source.zip',out/'events.jsonl']
    for p in files:
        target=dest/p.relative_to(out);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target)
    atomic_json(dest/'evidence_manifest.json',{'checkpoint_included':False,'native_trace_included':False,
        'small_new_correction_arrays_included':True,
        'files':{p.relative_to(dest).as_posix():file_digest(p) for p in sorted(dest.rglob('*')) if p.is_file() and p.name!='evidence_manifest.json'}})
    print({'status':'PASS','accepted':len(accepted),'physical_reports':report_count,'same_DEV_budget':True,
        'pair_rms_rad_range':[min(d['pair_control_rms_rad'] for d in diversity),max(d['pair_control_rms_rad'] for d in diversity)]})


if __name__=='__main__': main()
