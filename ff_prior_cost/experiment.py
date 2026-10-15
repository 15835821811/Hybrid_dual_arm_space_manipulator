"""One bounded background job: diagnose, distill, freeze, compare, execute."""
from pathlib import Path
import subprocess
import time
import zipfile
import numpy as np
from ff_prior.pipeline import audit, acceptance
from ff_prior.storage import atomic_json, read_json, file_digest, digest, Ledger, RunLock
from ff_prior.contracts import assert_frozen, assert_task_compatible
from ff_prior.guidance import public_goal_initializer
from ff_prior.model_gates import evaluate_reference
from ff_prior.coupled_predictor import predict_coupled, pose_errors
from ff_prior_fixed.correction import pose_vector
from ff_prior.physical_audit import audit_execution
from ff_prior.spline import Reference
from ff_prior_next import learning
from ff_prior_goal.experiment import inputs, sources as previous_sources, validation_targets, CHECKPOINT
from .projection import Projector, CONTRACT
from .sampling import sample
from . import training

ROOT = Path(__file__).resolve().parents[1]
METHODS = ('structured_final', 'frozen_diffusion_two', 'frozen_diffusion_final', 'trained_diffusion_final')
PROTOCOL = {'version': 'v71_equal_budget_residual_smoke_1', 'profile': 'smoke',
    'methods': list(METHODS), 'projection': CONTRACT, 'training': training.CONTRACT,
    'train_teachers': 30, 'val_teachers': 6, 'diagnostic_TRAIN': 6, 'diagnostic_VAL': 6,
    'VAL_targets': 6, 'DEV_targets': 4, 'candidates_per_task': 5, 'ddim_steps': 16,
    'max_candidates': 200, 'max_selected_executions': 40, 'wall_budget_s': 1800,
    'seed_DEV': 20261017, 'seed_VAL': 20261023, 'seed_diagnostic': 20261031,
    'teacher': 'public initializer then smooth and <=110-query fixed-world-goal repair; no private terminal',
    'learning_gain': 'no lower eligibility/execution, >=20% fewer total candidate coupled queries and no greater candidate wall than frozen_final on VAL and DEV',
    'diffusion_necessity': 'same conditions against structured_final; otherwise default structured_final',
    'selection': 'checkpoint by fixed VAL clean loss; recommended method by VAL only, frozen before DEV',
    'pilot': False, 'TEST': False, 'study': False, 'no_replacement': True,
    'physical_gates_changed': False, 'frozen_goal_relabel': False}


def sources():
    return dict(previous_sources(), **{p.relative_to(ROOT).as_posix(): file_digest(p)
        for p in sorted((ROOT/'ff_prior_cost').glob('*.py'))})


def train_targets(rows):
    pool = sorted((r for r,s in rows if s == 'TRAIN'), key=lambda r:digest({'cost_teacher_v1':r['sample_id']}))
    selected = []; families = set()
    for row in pool:
        if row['family_id'] in families: continue
        selected.append({'sample_id':row['sample_id'], 'family_id':row['family_id'], 'task':row['task']})
        families.add(row['family_id'])
    if len(selected) != 30: raise ValueError('expected original 30 TRAIN families')
    return selected


def validate(z, task, robot, initial, collision, profile, deadline):
    c = task['condition']; ref = Reference(z,c['initial_features'],c['duration'])
    pred, gates = evaluate_reference(robot,initial,ref,collision,profile,deadline)
    goal = False; score = None
    if pred is not None:
        pos,rot = pose_errors(pred['ee'][-1],c['goals'])
        goal = bool(np.all(pos <= profile.position_error_m) and np.all(rot <= profile.rotation_error_rad))
        score = float(np.sum((pos/profile.position_error_m)**2)+np.sum((rot/profile.rotation_error_rad)**2))
        gates.update(frozen_goal_position_error_m=pos,frozen_goal_rotation_error_rad=rot)
    return {'validation':gates,'frozen_goal_pass':goal,'eligible':goal and gates['validation_level']=='MODEL_VALID',
            'score':score,'reference_hash':ref.reference_hash}


def candidate(method,index,task,base,seed,bundle,robot,initial,collision,profile,deadline):
    assert_frozen(task); before=robot.calls.copy(); wall=time.perf_counter(); cpu=time.process_time()
    projector=Projector(task,robot,initial,collision,profile,deadline,
        schedule=(14,15) if method=='frozen_diffusion_two' else (15,))
    if method=='structured_final':
        z=base+.015*np.sin(np.linspace(0,np.pi,19))[:,None]*np.random.default_rng(seed).normal(size=(1,17))
        z,info=projector(z,15); sampling={'actual_network_calls':0,'guidance':[info]}
    else:
        if bundle is None: raise RuntimeError('TRAINED_CHECKPOINT_BLOCKED')
        z,sampling=sample(bundle,base,task,seed,projector)
    repair_calls=robot.calls['coupled']-before['coupled']
    result=validate(z,task,robot,initial,collision,profile,deadline)
    calls={k:robot.calls[k]-before[k] for k in before}
    if repair_calls>110 or calls['coupled']-repair_calls>3 or calls['coupled']>113:
        raise RuntimeError('EQUAL_QUERY_BUDGET_VIOLATION')
    return dict(result,status='DONE',method=method,candidate_id=index,seed=seed,task_hash=task['task_hash'],
        public_base_hash=digest(base),Z=z,sampling=sampling,model_calls=calls,
        repair_coupled_calls=repair_calls,validation_coupled_calls=calls['coupled']-repair_calls,
        shared_coupled_ceiling=113,wall_s=time.perf_counter()-wall,cpu_s=time.process_time()-cpu,
        private_teacher_fields_accessed=False)


def run(source,out,progress,resume=False,previous_elapsed=0.):
    from .reporting import write_report, choose_on_val
    source,out=Path(source).resolve(),Path(out).resolve()
    if source==out or source in out.parents or out in source.parents: raise ValueError('independent output required')
    old,data,dev,rows,files=inputs(source)
    identity={'protocol':PROTOCOL,'source':str(source),'source_files':files,'tasks_hash':digest(dev),'source_hashes':sources()}
    with RunLock(out):
        if (out/'run.json').exists() and (not resume or read_json(out/'run.json')['identity']!=identity):
            raise ValueError('run identity conflict')
        proof=read_json(ROOT/'ff_prior_cost/evidence/software_tests.json')
        if proof['status']!='PASS' or proof['source_hashes']!=sources(): raise ValueError('software proof stale')
        if proof['junit_sha256']!=file_digest(ROOT/'ff_prior_cost/evidence/tests.xml'): raise ValueError('test proof changed')
        audit_report,robot,initial,collision=audit(old['config']); profile=acceptance(old['config'])
        if audit_report['acceptance']!=read_json(source/'baseline_audit.json')['acceptance']: raise ValueError('physical gates changed')
        train=train_targets(rows); groups={'VAL':validation_targets(rows),'DEV':dev}
        for e in train+groups['VAL']+dev: assert_task_compatible(e['task'],robot,initial,collision,profile)
        if not (out/'run.json').exists():
            atomic_json(out/'run.json',{'identity':identity,'acceptance':audit_report['acceptance'],
                'robot_hash':initial.model_hash,'initial_state_id':initial.state_id,
                'producer_head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()})
            atomic_json(out/'frozen_tasks.json',groups); atomic_json(out/'train_targets.json',train)
            atomic_json(out/'software_tests.json',proof)
            with zipfile.ZipFile(out/'executed_source.zip','w',zipfile.ZIP_DEFLATED) as archive:
                for p in identity['source_hashes']: archive.write(ROOT/p,p)
        if read_json(out/'frozen_tasks.json')!=groups or read_json(out/'train_targets.json')!=train:
            raise ValueError('frozen target conflict')
        ledger=Ledger(out,identity)
        if (out/'summary.json').exists() and read_json(out/'summary.json')['status']=='FINISHED': return read_json(out/'summary.json')
        deadline=time.time()+max(0,1800-previous_elapsed)
        bundle=learning.load(CHECKPOINT,initial.model_hash,initial.state_id)
        # Only TRAIN and VAL participate in diagnosis. Probes are separate from benchmark budgets.
        for split,entries in (('TRAIN',train[:6]),('VAL',groups['VAL'])):
            for ti,e in enumerate(entries):
                key=f'diagnostic_{split}_{ti}'; path=out/'diagnostics'/split/f'{ti:03d}.json'
                if key in ledger.data['completed']: continue
                ledger.begin(key,'diagnostic',1800,12); before=robot.calls.copy()
                progress('step_diagnosis',split=split,task=ti+1,total=6)
                try:
                    task=e['task']; c=task['condition']; base=public_goal_initializer(task,robot,initial); probes=[]
                    def observe(z,iteration,stage):
                        p=predict_coupled(initial,Reference(z,c['initial_features'],c['duration']),robot,128,deadline)
                        pos,rot=pose_errors(p['ee'][-1],c['goals'])
                        probes.append({'iteration':iteration,'stage':stage,'position_error_m':pos,
                            'rotation_error_rad':rot,'normalized_pose_error':pose_vector(p['ee'][-1],c['goals']),
                            'rms_from_base_rad':float(np.sqrt(np.mean((z-base)**2))), 'Z_hash':digest(z)})
                    observe(base,-1,'public_initializer')
                    proj=Projector(task,robot,initial,collision,profile,deadline,(14,15),observe)
                    z,sampling=sample(bundle,base,task,20261031+ti*101,proj,observe)
                    result={'status':'MEASURED','split':split,'task_hash':task['task_hash'],'probes':probes,
                        'sampling':sampling,'diagnostic_extra_coupled_queries':len(probes),
                        'repair_coupled_queries':proj.used,'MODEL_VALID_claimed':False}
                except Exception as error: result={'status':'ERROR','reason':repr(error)}
                result['model_calls']={k:robot.calls[k]-before[k] for k in before}
                atomic_json(path,result); ledger.finish({'status':result['status']},result['model_calls'])
        # One minimal public-initializer repair per existing TRAIN/VAL family.
        for split,entries in (('TRAIN',train),('VAL',groups['VAL'])):
            for ti,e in enumerate(entries):
                key=f'teacher_{split}_{ti}'; path=out/'teachers'/split/f'{ti:03d}.json'
                if key in ledger.data['completed']: continue
                ledger.begin(key,'teacher',1800,36); before=robot.calls.copy()
                progress('fixed_goal_teacher',split=split,task=ti+1,total=len(entries))
                try:
                    task=e['task']; base=public_goal_initializer(task,robot,initial)
                    projector=Projector(task,robot,initial,collision,profile,deadline)
                    z,info=projector(base,15)
                    result=dict(validate(z,task,robot,initial,collision,profile,deadline),status='DONE',split=split,
                        sample_id=f'cost_{split}_{ti:03d}',parent_sample_id=e['sample_id'],family_id=e['family_id'],
                        task=task,base=base,Z=z,repair=info,execution_evidence='NOT_EXECUTED',
                        training_eligible=split=='TRAIN')
                    result['training_eligible']=result['training_eligible'] and result['eligible']
                except Exception as error: result={'status':'ERROR','reason':repr(error),'eligible':False,'split':split}
                atomic_json(path,result); ledger.finish({'status':result['status'],'eligible':result['eligible']},
                    {k:robot.calls[k]-before[k] for k in before})
        key='training'; trained_bundle=None
        if key not in ledger.data['completed']:
            ledger.begin(key,'training',1800,1); progress('training_preflight')
            try:
                records=[read_json(p) for p in sorted((out/'teachers').glob('*/*.json'))]
                result=training.train(out/'learning',records,CHECKPOINT,initial.model_hash,initial.state_id,progress,deadline)
            except Exception as error:
                result={'status':'BLOCKED','reason':repr(error)}; atomic_json(out/'learning/training.json',result)
            ledger.finish({'status':result['status']})
        if (out/'learning/training.json').exists() and read_json(out/'learning/training.json')['status']=='PASS':
            trained_bundle=training.load(out/'learning',CHECKPOINT,initial.model_hash,initial.state_id)
        for split,entries in groups.items():
            if split=='DEV':
                selected=choose_on_val(out)
                frozen={'selected_method':selected,'selection_split':'VAL','DEV_results_used':False,
                    'protocol_hash':digest(PROTOCOL),'trained_checkpoint_sha256':file_digest(out/'learning/model.pt') if trained_bundle else None}
                if (out/'pre_DEV_freeze.json').exists():
                    if read_json(out/'pre_DEV_freeze.json')!=frozen: raise ValueError('selection mutated')
                else: atomic_json(out/'pre_DEV_freeze.json',frozen)
                progress('protocol_frozen_before_DEV',selected_method=selected)
            for ti,e in enumerate(entries):
                task=e['task']; base=public_goal_initializer(task,robot,initial)
                for method in METHODS:
                    candidates=[]
                    for k in range(5):
                        key=f'candidate_{split}_{ti}_{method}_{k}'; path=out/'candidates'/split/method/f'{ti:03d}_{k:02d}.json'
                        if key in ledger.data['completed']:
                            if path.exists(): candidates.append(read_json(path))
                            continue
                        ledger.begin(key,'candidates',1800,200); before=robot.calls.copy()
                        progress('sampling',split=split,task=ti+1,method=method,candidate=k+1,total=200,
                            completed=ledger.data['attempts']['candidates']-1)
                        try:
                            seed=(20261017 if split=='DEV' else 20261023)+ti*101+k*997
                            result=candidate(method,k,task,base,seed,trained_bundle if method==METHODS[-1] else bundle,
                                robot,initial,collision,profile,deadline)
                        except Exception as error: result={'status':'BLOCKED' if 'CHECKPOINT_BLOCKED' in str(error) else 'ERROR',
                            'reason':repr(error),'eligible':False,'candidate_id':k,'task_hash':task['task_hash'],'method':method}
                        atomic_json(path,result); candidates.append(result)
                        ledger.finish({'status':result['status'],'split':split,'method':method},
                            {n:robot.calls[n]-before[n] for n in before})
                    key=f'execution_{split}_{ti}_{method}'; directory=out/'executions'/split/method/f'{ti:03d}'
                    if key in ledger.data['completed']: continue
                    eligible=[r for r in candidates if r.get('eligible')]
                    chosen=min(eligible,key=lambda r:(r['score'],r['candidate_id'])) if eligible else None
                    atomic_json(directory/'selection.json',{'task_hash':task['task_hash'],
                        'selected_id':chosen['candidate_id'] if chosen else None,'selection_before_execution':True,
                        'rule':'lowest normalized pose error; no retry/replacement'})
                    ledger.begin(key,'execution',1800,40); before=robot.calls.copy()
                    progress('torque_execution',split=split,task=ti+1,method=method,selected=chosen['candidate_id'] if chosen else None)
                    try:
                        if chosen:
                            ref=Reference(chosen['Z'],task['condition']['initial_features'],task['condition']['duration'])
                            physical=audit_execution(task,profile,robot,initial,ref,directory,deadline); state=physical['level']
                        else: state='NO_CANDIDATE'
                        result={'status':state,'split':split,'method':method}
                    except Exception as error: result={'status':'EXECUTION_FAILED','reason':repr(error),'split':split,'method':method}
                    ledger.finish(result,{n:robot.calls[n]-before[n] for n in before})
                write_report(out,ledger,'RUNNING')
        if any(file_digest(source/p)!=h for p,h in files.items()): raise ValueError('input mutated')
        if sources()!=identity['source_hashes']: raise ValueError('source mutated')
        atomic_json(out/'verification.json',{'source_inputs_unchanged':True,'source_code_unchanged':True,
            'same_frozen_DEV':groups['DEV']==dev,'goals_not_relabelled':True,'physical_profile_unchanged':True,
            'TEST_arrays_read':False,'complete_model_calls':robot.calls})
        incomplete=(out/'interruption.json').exists() or ledger.data.get('unresolved_call_accounting') or any(
            r['status'] in ('BLOCKED','ERROR','INTERRUPTED') for r in ledger.data['completed'].values())
        return write_report(out,ledger,'FINISHED_WITH_BLOCKED_ITEMS' if incomplete else 'FINISHED')
