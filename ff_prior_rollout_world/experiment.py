import time,subprocess,zipfile
import numpy as np
import torch
from ff_prior.storage import read_json,digest,RunLock
from ff_prior_pilot_train.storage import atomic_json,atomic_npz,Ledger
from ff_prior.pipeline import audit,acceptance
from ff_prior_origin_learn_bounded.runtime import TaskOriginSceneCache
from ff_prior.guidance import public_goal_initializer
from ff_prior.spline import Reference
from ff_prior_rep_fidelity.validation import validate as dense_validate
from ff_prior_rep_fidelity.physical_audit import audit_execution
from ff_prior.initial_state import STATE
from .sampling import unguided_sample
from .world_training import diagnostic as diagnostic_run
from ff_prior_feasible.validation import validate
from .protocol import ROOT,PACKAGE,OUT,SOURCE,BASE,REUSE,P
from .inputs import identity
from .data import load as load_data
from .sampling import FullProjector,candidate
from . import learning

class Budget:
    def __init__(self,out,deadline):
        self.out,self.deadline=out,deadline
        self.total=read_json(out/'budget.json')['reserved_coupled_upper'] if (out/'budget.json').exists() else 0
        self.integrations=0;self.last=0.
    def check(self):
        if time.time()>=self.deadline:raise RuntimeError('WALL_BUDGET_EXHAUSTED')
        if time.time()-self.last>30:
            if sum(p.stat().st_size for p in self.out.rglob('*') if p.is_file())>P['output_bytes_cap']:raise RuntimeError('STORAGE_LIMIT')
            self.last=time.time()
    def reserve(self,n):
        self.check()
        if self.total+n>P['coupled_cap']:raise RuntimeError('QUERY_LIMIT')
        self.total+=n;atomic_json(self.out/'budget.json',{'reserved_coupled_upper':self.total,'limit':P['coupled_cap']})
    def consume(self,n):self.check();self.integrations+=n

def assert_resume_safe(out):
    if (out/'ledger.json').exists():
        l=read_json(out/'ledger.json')
        if l['inflight'] is not None or l.get('unresolved_call_accounting'):raise ValueError('UNKNOWN_WORK_NOT_REPLAYABLE')
    for p in out.glob('learning/*/journal.json'):
        if read_json(p)['status']!='COMMITTED':raise ValueError('unknown training segment')

def run(out=OUT,progress=lambda *a,**k:None,previous_elapsed=0.):
    deadline=time.time()+max(0,P['wall_s']-previous_elapsed)
    from .audit_results import verify,report
    assert_resume_safe(out);ident=identity();torch.set_num_threads(2)
    with RunLock(out):
        assert read_json(PACKAGE/'evidence/software_tests.json')['source_hashes']==ident['source_hashes']
        if (out/'run.json').exists():assert read_json(out/'run.json')['identity']==ident
        else:
            atomic_json(out/'run.json',{'identity':ident,'producer_head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()})
            with zipfile.ZipFile(out/'executed_source.zip','w',zipfile.ZIP_DEFLATED) as z:
                for p in ident['source_hashes']:z.write(ROOT/p,p)
        physical,canonical,initial,_=audit(read_json(BASE/'run.json')['config'])
        old_physical=read_json(SOURCE/'physical_contract.json')
        assert all(digest(physical[k])==digest(old_physical[k]) for k in ('acceptance','identity','initial_state_id','initial_state_mask'))
        profile=acceptance(read_json(BASE/'run.json')['config'])
        atomic_json(out/'physical_contract.json',physical)
        cache=TaskOriginSceneCache(canonical,initial,profile,max_entries=16,max_scene_models=6)
        progress('M0_TRAIN_only_codec')
        codec,arrays,rows,data=load_data();selection=ident['selection']
        trainhashes={rows[i]['task']['task_hash'] for i in data['tr']}
        assert not trainhashes&set(selection['forbidden_TRAIN_task_hashes'])
        atomic_json(out/'codec.json',codec.state)
        atomic_json(out/'training_input.json',{'TRAIN':14222,'VAL':48,'TRAIN_ids':[arrays[3][i] for i in data['tr']],
            'VAL_ids':[arrays[3][i] for i in data['va']],'roundtrip_max_rad':data['reconstruction_max_rad'],
            'TRAIN_task_hashes':sorted(trainhashes),'new_data':0,'TEST':False})
        atomic_json(out/'frozen_tasks.json',selection)
        ledger=Ledger(out,{'identity_hash':digest(ident)});budget=Budget(out,deadline)
        def unit(key,stage,robot,action):
            if key in ledger.data['completed']:return
            ledger.begin(key,stage,P['wall_s'],1000);before=robot.calls.copy();budget.check()
            action();budget.check();ledger.finish({'status':'DONE'},{k:robot.calls[k]-before[k] for k in before})
        def execute(directory,task,z,rr,ii):
            r=audit_execution(task,profile,rr,ii,Reference(z,task['condition']['initial_features'],task['condition']['duration']),directory,budget.deadline)
            atomic_json(directory/'outcome.json',{'status':r['level']});return r
        gate=dict(read_json(REUSE/'reconstruction_gate.json'),evidence_reused=True,new_executions=0,
                  source=REUSE.relative_to(ROOT).as_posix())
        atomic_json(out/'reconstruction_gate.json',gate)
        diagnostic=diagnostic_run(out,selection,codec,arrays,rows,data,cache,profile,ledger,budget,progress) if gate['status']=='PASS' else {'status':'BLOCKED','reason':'RECONSTRUCTION_GATE'}
        learning_gate=dict(gate,status='PASS' if gate['status']=='PASS' and diagnostic['status']=='PASS' else 'BLOCKED',learnability=diagnostic)
        atomic_json(out/'learning_gate.json',learning_gate)
        if learning_gate['status']=='PASS':
            learning.train(out,codec,arrays,data,ledger,budget,progress)
            from .world_training import refine_candidate
            refine_candidate(out,selection,codec,arrays,rows,data,cache,profile,ledger,budget,progress)
            bundle=learning.load(out/'learning/full')
            for sj,seed in enumerate(P['val_seeds']):
                for j,i in enumerate(data['va']):
                    row=rows[i];rr,ii,cc,_=cache.get(row);task=row['task'];base=arrays[1][i]
                    def evaluate_unguided():
                        budget.reserve(3);progress('M3_full16_unguided_VAL',seed_group=sj,target=j)
                        raw,steps=unguided_sample(bundle,base,task,seed+j*997)
                        z,prep=FullProjector(task,rr,ii,cc,profile,budget.deadline,codec=codec,base=base).prepare(raw)
                        r=validate(z,task,rr,ii,cc,profile,budget.deadline)
                        r.update(raw_Z=raw,prepared_Z=z,preparation=prep,steps=steps,seed=seed+j*997,task_hash=task['task_hash'],sample_id=arrays[3][i],physics_guidance_calls=0)
                        atomic_json(out/'unguided'/str(sj)/f'{j:02d}.json',r)
                    unit(f'unguided_{sj}_{j}','unguided',rr,evaluate_unguided)
            groups=[('DEV',sj,ti,e,seed+ti*101) for sj,seed in enumerate(P['sampling_seeds']) for ti,e in enumerate(selection['entries'])]
            groups += [('APPLICATION_DEV',0,ti,e,e['seed']) for ti,e in enumerate(selection['application'])]
            for scope,sj,ti,e,seed in groups:
                task=e['task'];sid=e['sample_id'];rr,ii,cc,_=cache.get(e);base=public_goal_initializer(task,rr,ii)
                directory=out/'comparison'/scope/str(sj)/sid;records=[]
                for k in range(5):
                    path=directory/'candidates'/f'{k:02d}.json'
                    def evaluate_candidate():
                        budget.reserve(113);progress('M4_same_budget_comparison',scope=scope,seed=sj,task=ti,candidate=k)
                        r=candidate('late4_diffusion',k,task,base,seed+k*997,bundle,rr,ii,cc,profile,budget.deadline)
                        r.update(method='world323',sample_id=sid,seed_group=sj,split=scope);atomic_json(path,r)
                    unit(f'candidate_{scope}_{sj}_{sid}_{k}','candidate',rr,evaluate_candidate);records.append(read_json(path))
                chosen=min((r for r in records if r['eligible']),key=lambda r:(r['score'],r['candidate_id']),default=None)
                def execute_candidate():
                    ex=directory/'execution';atomic_json(ex/'selection.json',{'before_execution':True,'selected_id':None if chosen is None else chosen['candidate_id'],
                        'task_hash':task['task_hash'],'candidate_pool_digest':digest(records)})
                    progress('M4_native_torque_and_replay',scope=scope,sample_id=sid,seed=sj)
                    if chosen:execute(ex,task,chosen['Z'],rr,ii)
                    else:atomic_json(ex/'outcome.json',{'status':'NO_CANDIDATE'})
                unit(f'execute_{scope}_{sj}_{sid}','execution',rr,execute_candidate)
        progress('M5_saved_evidence_audit')
        checked=verify(out,ident);atomic_json(out/'independent_verification.json',checked)
        return report(out,checked)
