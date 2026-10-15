"""TRAIN correction -> execution consistency -> original frozen DEV smoke."""
from pathlib import Path
import subprocess
import time
import zipfile
import numpy as np
from ff_prior.pipeline import audit,acceptance
from ff_prior.storage import read_json,atomic_json,atomic_npz,digest,file_digest,Ledger,RunLock
from ff_prior.contracts import assert_task_compatible
from ff_prior.guidance import public_goal_initializer
from ff_prior.spline import Reference
from ff_prior.physical_audit import audit_execution
from ff_prior_next import learning
from ff_prior_next.experiment import inputs as original_inputs,sources as previous_sources,candidate,METHODS
from .correction import choose_targets,correct
from .consistency import analyze
from .reporting import write_report

ROOT=Path(__file__).resolve().parents[1]
PROTOCOL={'version':'v71_fixed_train_consistency_1','profile':'smoke','train_targets':8,
    'target_selection':'hash_order_one_per_TRAIN_family','corrections_per_target':2,'max_correction_attempts':16,
    'correction_iterations':8,'correction_directions':17,'correction_difference_rad':1e-4,
    'correction_max_line_trials':4,'correction_phase_intervals':128,
    'diagnostic_targets':2,'diagnostic_roles':['original','first_eligible_correction'],
    'max_diagnostic_executions':4,'reference_diagnostic_intervals_max':1000,
    'native_reconstruction':'all_67_joints_midpoint_base_integration_without_base_state_injection',
    'tasks':4,'candidates_per_task':5,'methods':list(METHODS),'training_updates':200,'training_seed':20261015,
    'ddim_calls':16,'guidance_updates':2,'directions':8,'max_benchmark_selected_executions':12,
    'wall_budget_s':1800,'pilot':False,'TEST':False,'no_failure_replacement':True,'no_parameter_sweep':True,
    'exclude_known_execution_failed_labels':True,'benefit_criterion':'guided verified tasks > structured verified tasks; exploratory DEV only'}


def sources():
    return dict(previous_sources(),**{p.relative_to(ROOT).as_posix():file_digest(p)
        for p in sorted((ROOT/'ff_prior_fixed').glob('*.py'))})


def inputs(source):
    old,data,tasks,rows,files=original_inputs(source)
    files['splits.json']=file_digest(Path(source)/'splits.json')
    return old,data,tasks,rows,files


def verified_training_rows(rows,augmented,excluded):
    result=[(r,s) for r,s in rows if r['sample_id'] not in excluded]
    for row in augmented:
        if row['sample_id'] in excluded: continue
        if row.get('split')!='TRAIN' or not row.get('training_eligible') or not row.get('frozen_goal_pass'):
            raise ValueError('unqualified augmented row')
        if row['validation_level']!='MODEL_VALID' or row['execution_status']=='FAIL':
            raise ValueError('failed model/execution augmented row')
        result.append((row,'TRAIN'))
    if not any(s=='TRAIN' for _,s in result) or not any(s=='VAL' for _,s in result):
        raise ValueError('training blocked after known-failure exclusion')
    if any(s not in ('TRAIN','VAL') for _,s in result): raise ValueError('DEV/TEST leakage')
    return result


def run(source,out,progress,resume=False,previous_elapsed=0.):
    source,out=Path(source).resolve(),Path(out).resolve()
    if source==out or source in out.parents or out in source.parents: raise ValueError('output must be independent')
    old,data,tasks,rows,files=inputs(source)
    identity={'protocol':PROTOCOL,'source':str(source),'source_files':files,'tasks_hash':digest(tasks),'source_hashes':sources()}
    with RunLock(out):
        if (out/'run.json').exists() and (not resume or read_json(out/'run.json')['identity']!=identity):
            raise ValueError('run identity conflict')
        proof=read_json(ROOT/'ff_prior_fixed/evidence/software_tests.json')
        if proof['status']!='PASS' or proof['source_hashes']!=sources(): raise ValueError('software proof missing or stale')
        atomic_json(out/'software_tests.json',proof)
        audit_report,robot,initial,collision=audit(old['config']); profile=acceptance(old['config'])
        if audit_report['acceptance']!=read_json(source/'baseline_audit.json')['acceptance']: raise ValueError('physical contract changed')
        for entry in tasks: assert_task_compatible(entry['task'],robot,initial,collision,profile)
        targets=choose_targets(rows)
        selection=[{'sample_id':r['sample_id'],'family_id':r['family_id'],'task_hash':r['task']['task_hash'],
                    'condition_hash':digest(r['condition'])} for r in targets]
        if not (out/'run.json').exists():
            atomic_json(out/'run.json',{'identity':identity,'physical_acceptance':audit_report['acceptance'],
                'robot_hash':initial.model_hash,'initial_state_id':initial.state_id,
                'train_selection_hash':digest(selection),'producer_head':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()})
            atomic_json(out/'train_target_selection.json',selection); atomic_json(out/'benchmark_tasks.json',tasks)
            with zipfile.ZipFile(out/'executed_source.zip','w',zipfile.ZIP_DEFLATED) as archive:
                for path in identity['source_hashes']: archive.write(ROOT/path,path)
        if read_json(out/'train_target_selection.json')!=selection: raise ValueError('frozen TRAIN selection changed')
        ledger=Ledger(out,identity)
        if (out/'summary.json').exists() and read_json(out/'summary.json')['status']=='FINISHED': return read_json(out/'summary.json')
        deadline=time.time()+max(0,1800-previous_elapsed)
        splits=read_json(source/'splits.json')['records']
        for ti,row in enumerate(targets):
            assert_task_compatible(row['task'],robot,initial,collision,profile)
            for variant in range(2):
                sid=f'trainfix_{ti:03d}_{variant}'; key='correction_'+sid
                if key in ledger.data['completed']: continue
                ledger.begin(key,'correction',1800,16); before=robot.calls.copy()
                progress('train_correction',target=ti+1,variant=variant+1,total_targets=8,
                    completed_attempts=ledger.data['attempts']['correction']-1,total_attempts=16)
                try:
                    z,result=correct(row['task'],row['sample_id'],'TRAIN',variant,robot,initial,collision,profile,deadline)
                    arrays=f'corrections/arrays/{sid}.npz'; atomic_npz(out/arrays,**Reference(z,row['condition']['initial_features'],row['condition']['duration']).arrays())
                    result.update(sample_id=sid,arrays=arrays,condition=row['condition'],task=row['task'],
                        family_id=row['family_id'],family_group=splits[row['sample_id']]['family_group'],
                        parent_sample_id=row['sample_id'],source_partition='train_correction')
                except Exception as error:
                    result={'status':'ERROR','sample_id':sid,'training_eligible':False,'source_sample_id':row['sample_id'],
                        'task_hash':row['task']['task_hash'],'split':'TRAIN','reason':repr(error)}
                atomic_json(out/'corrections/records'/f'{sid}.json',result)
                ledger.finish({'status':result['status'],'kind':'correction','sample_id':sid},
                    {k:robot.calls[k]-before[k] for k in before})
        augmented=[read_json(p) for p in sorted((out/'corrections/records').glob('*.json'))]
        augmented=[r for r in augmented if r['training_eligible']]
        # Preselect all diagnostics before any of their executions. No replacement.
        diagnostic_selection=[]
        for ti,row in enumerate(targets[:2]):
            diagnostic_selection.append({'id':f'{ti:03d}_original','sample_id':row['sample_id'],
                'task':row['task'],'arrays':str(source/row['arrays']),'role':'original'})
            options=[r for r in augmented if r['parent_sample_id']==row['sample_id']]
            chosen=min(options,key=lambda r:r['variant']) if options else None
            diagnostic_selection.append({'id':f'{ti:03d}_corrected','sample_id':chosen['sample_id'] if chosen else None,
                'task':row['task'],'arrays':str(out/chosen['arrays']) if chosen else None,'role':'first_eligible_correction'})
        if (out/'diagnostic_selection.json').exists() and read_json(out/'diagnostic_selection.json')!=diagnostic_selection:
            raise ValueError('diagnostic selection changed')
        atomic_json(out/'diagnostic_selection.json',diagnostic_selection)
        for entry in diagnostic_selection:
            key='consistency_'+entry['id']; directory=out/'consistency'/entry['id']
            if key in ledger.data['completed']: continue
            ledger.begin(key,'consistency',1800,4); before=robot.calls.copy()
            progress('prediction_execution_consistency',diagnostic=entry['id'],sample_id=entry['sample_id'])
            result={'status':'BLOCKED','sample_id':entry['sample_id'],'task_hash':entry['task']['task_hash'],'execution_status':'NOT_RUN'}
            try:
                if entry['arrays'] is None: raise RuntimeError('NO_ELIGIBLE_TRAIN_CORRECTION_FOR_FROZEN_DIAGNOSTIC')
                z=np.load(entry['arrays'])['Z']; task=entry['task']; ref=Reference(z,task['condition']['initial_features'],task['condition']['duration'])
                execution=audit_execution(task,profile,robot,initial,ref,directory/'execution',deadline)
                result['execution_status']=execution['execution_status']; result['execution_level']=execution['level']
                arrays=np.load(directory/'execution/native_trace.npz')
                result['comparison']=analyze(robot,initial,ref,arrays,directory/'comparison.npz',deadline)
                result['status']='MEASURED'
            except Exception as error: result['reason']=repr(error)
            atomic_json(directory/'diagnostic.json',result)
            ledger.finish({'status':result['status'],'kind':'consistency','sample_id':entry['sample_id'],
                'execution_status':result['execution_status']},{k:robot.calls[k]-before[k] for k in before})
        excluded=set()
        for entry in diagnostic_selection:
            report=out/'consistency'/entry['id']/'execution/report.json'
            if report.exists() and read_json(report)['execution_status']=='FAIL': excluded.add(entry['sample_id'])
        training_rows=verified_training_rows(rows,augmented,excluded)
        atomic_json(out/'training_dataset.json',{'TRAIN':[r['sample_id'] for r,s in training_rows if s=='TRAIN'],
            'VAL':[r['sample_id'] for r,s in training_rows if s=='VAL'],'new_eligible_ids':[r['sample_id'] for r in augmented if r['sample_id'] not in excluded],
            'excluded_known_execution_failed_ids':sorted(excluded),'TEST_arrays_read':False,'lineage_inherited_from_TRAIN':True})
        z=[]; bases=[]; conditions=[]; ids=[]; labels=[]
        prep_wall,prep_cpu=time.perf_counter(),time.process_time(); before=robot.calls.copy()
        progress('prepare_augmented_training',training_rows=len(training_rows))
        for row,split in training_rows:
            path=out/row['arrays'] if row.get('source_partition')=='train_correction' else source/row['arrays']
            z.append(np.load(path)['Z']); bases.append(public_goal_initializer(row['task'],robot,initial))
            conditions.append(row['condition']); ids.append(row['sample_id']); labels.append(split)
        old_prep=read_json(out/'preparation.json') if (out/'preparation.json').exists() else {}
        atomic_json(out/'preparation.json',{'calls':{k:robot.calls[k]-before[k]+old_prep.get('calls',{}).get(k,0) for k in before},
            'wall_s':time.perf_counter()-prep_wall+old_prep.get('wall_s',0),'cpu_s':time.process_time()-prep_cpu+old_prep.get('cpu_s',0)})
        training_ok=False
        try:
            result=learning.train(out/'learning',z,bases,conditions,ids,labels,initial.model_hash,initial.state_id,progress,deadline)
            training_ok=result['status']=='PASS'
        except Exception as error: atomic_json(out/'training_failure.json',{'status':'BLOCKED','reason':repr(error)})
        bundle=learning.load(out/'learning',initial.model_hash,initial.state_id) if training_ok else None
        for ti,entry in enumerate(tasks):
            task=entry['task']; base=public_goal_initializer(task,robot,initial)
            for method in METHODS:
                candidates=[]
                for k in range(5):
                    key=f'candidate_{ti}_{method}_{k}'; path=out/'candidates'/method/f'{ti:03d}_{k:02d}.json'
                    if key in ledger.data['completed']:
                        if path.exists(): candidates.append(read_json(path))
                        continue
                    ledger.begin(key,'candidates',1800,60); before=robot.calls.copy()
                    progress('dev_smoke',task=ti+1,method=method,candidate=k+1,completed_candidates=ledger.data['attempts']['candidates']-1,total_candidates=60)
                    try:
                        if time.time()>=deadline: raise RuntimeError('WALL_BUDGET_EXHAUSTED')
                        if method!=METHODS[0] and bundle is None: raise RuntimeError('BLOCKED_TRAINING')
                        result=candidate(method,k,task,base,20261017+ti*101+k*997,bundle,robot,initial,collision,profile,deadline)
                    except Exception as error: result={'status':'ERROR','reason':repr(error),'eligible':False,'candidate_id':k,'task_hash':task['task_hash']}
                    atomic_json(path,result); candidates.append(result)
                    ledger.finish({'status':result['status'],'method':method,'kind':'candidate'},
                        {n:robot.calls[n]-before[n] for n in before})
                key=f'execution_{ti}_{method}'
                if key in ledger.data['completed']: continue
                eligible=[c for c in candidates if c.get('eligible')]
                chosen=min(eligible,key=lambda c:(c['score'],c['candidate_id'])) if eligible else None
                directory=out/'executions'/method/f'{ti:03d}'
                atomic_json(directory/'selection.json',{'task_hash':task['task_hash'],'selected_id':chosen['candidate_id'] if chosen else None,
                    'selection_before_execution':True,'rule':'minimum normalized pose error among model+frozen-goal eligible'})
                ledger.begin(key,'execution',1800,12); before=robot.calls.copy()
                progress('dev_execution',task=ti+1,method=method,selected_id=chosen['candidate_id'] if chosen else None)
                try:
                    if chosen:
                        ref=Reference(np.array(chosen['Z']),task['condition']['initial_features'],task['condition']['duration'])
                        result=audit_execution(task,profile,robot,initial,ref,directory,deadline); status=result['level']
                    else: status='NO_CANDIDATE'
                    result={'status':status,'method':method,'kind':'execution'}
                except Exception as error: result={'status':'EXECUTION_FAILED','reason':repr(error),'method':method,'kind':'execution'}
                ledger.finish(result,{n:robot.calls[n]-before[n] for n in before})
        if not all(file_digest(source/p)==h for p,h in files.items()): raise ValueError('immutable source input changed')
        if sources()!=identity['source_hashes']: raise ValueError('runtime source changed')
        atomic_json(out/'verification.json',{'source_inputs_unchanged':True,'source_code_unchanged':True,
            'same_frozen_DEV_tasks':read_json(out/'benchmark_tasks.json')==tasks,'TRAIN_goals_not_relabelled':True,
            'physical_profile_unchanged':True,'pilot':'NOT_RUN','TEST':'NOT_RUN','complete_model_calls':robot.calls})
        incomplete=not training_ok or (out/'interruption.json').exists() or ledger.data.get('unresolved_call_accounting') or any(
            v['status'] in ('ERROR','BLOCKED','INTERRUPTED') for v in ledger.data['completed'].values())
        status='FINISHED_WITH_BLOCKED_ITEMS' if incomplete else 'FINISHED'
        progress('report')
        return write_report(out,ledger,status)
