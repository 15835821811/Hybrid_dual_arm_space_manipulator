import time, subprocess, zipfile
import numpy as np
from ff_prior.storage import read_json,digest,RunLock
from ff_prior_pilot_train.storage import atomic_json,atomic_npz,Ledger
from ff_prior.pipeline import audit,acceptance
from ff_prior.contracts import assert_frozen,assert_task_compatible
from ff_prior_origin_learn_bounded.runtime import TaskOriginSceneCache
from ff_prior.guidance import public_goal_initializer
from ff_prior.spline import Reference
from ff_prior.proposal_generator import quick_gates
from ff_prior.initial_state import STATE
from .protocol import ROOT,OUT,PACKAGE,PREVIOUS,SOURCE,BASE,P
from .inputs import identity
from .representation import fixed_basis,project,prepare
from .validation import validate
from .physical_audit import audit_execution

class Budget:
    def __init__(self, out, robot, deadline):
        self.out, self.robot, self.deadline = out, robot, deadline
        self.used = read_json(out/"ledger.json").get("calls", {}) if (out/"ledger.json").exists() else {}
        self.previous = sum(v for k, v in self.used.items() if k.endswith(".coupled"))
        self.start = robot.calls["coupled"]
        self.last_size_check = 0
        self.integrations = read_json(out/"budget.json")["integration_intervals"] if (out/"budget.json").exists() else 0
    def check(self, upcoming=0):
        if time.time() >= self.deadline:
            raise RuntimeError("WALL_BUDGET_EXHAUSTED")
        if self.previous + self.robot.calls["coupled"]-self.start + upcoming > P["coupled_cap"]:
            raise RuntimeError("COUPLED_BUDGET_EXHAUSTED")
        if time.time()-self.last_size_check > 30:
            if sum(p.stat().st_size for p in self.out.rglob("*") if p.is_file()) > P["output_bytes_cap"]:
                raise RuntimeError("OUTPUT_BUDGET_EXHAUSTED")
            self.last_size_check = time.time()

    def consume(self, intervals):
        self.check(1)
        if self.integrations+intervals>P['integration_cap']:raise RuntimeError('INTEGRATION_BUDGET_EXHAUSTED')
        self.integrations+=intervals
    def can_fit(self, queries, intervals):
        return (self.previous+self.robot.calls['coupled']-self.start+queries<=P['coupled_cap']
            and self.integrations+intervals<=P['integration_cap'] and time.time()+60<self.deadline)
    def save(self):
        atomic_json(self.out/'budget.json',{'integration_intervals':self.integrations,
            'actual_coupled':self.previous+self.robot.calls['coupled']-self.start,'limits':P})

def assert_resume_safe(out):
    if (out/"ledger.json").exists():
        r = read_json(out/"ledger.json")
        if r["inflight"] is not None or r.get("unresolved_call_accounting"):
            raise ValueError("UNKNOWN_WORK_NOT_REPLAYABLE")

def run(out=OUT,progress=lambda *a,**k:None,previous_elapsed=0.):
    out.mkdir(parents=True,exist_ok=True);assert_resume_safe(out);ident=identity()
    from .audit_results import verify,report
    with RunLock(out):
        assert read_json(PACKAGE/'evidence/software_tests.json')['source_hashes']==ident['source_hashes']
        if (out/'run.json').exists():assert read_json(out/'run.json')['identity']==ident
        else:
            atomic_json(out/'run.json',{'identity':ident,'producer_head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()})
            with zipfile.ZipFile(out/'executed_source.zip','w',zipfile.ZIP_DEFLATED) as archive:
                for p in ident['source_hashes']:archive.write(ROOT/p,p)
        config=read_json(BASE/'run.json')['config'];physical,canonical,initial,_=audit(config)
        old=read_json(SOURCE/'physical_contract.json')
        assert physical['acceptance']==old['acceptance'] and digest(physical['identity'])==digest(old['identity']) and physical['initial_state_id']==old['initial_state_id']
        profile=acceptance(config);entry=ident['selection']['entry'];task=entry['task']
        cache=TaskOriginSceneCache(canonical,initial,profile,max_entries=1,max_scene_models=1)
        rr,ii,cc,proof=cache.get(entry);assert_task_compatible(task,rr,ii,cc,profile);assert_frozen(task)
        atomic_json(out/'frozen_task.json',task);atomic_json(out/'physical_contract.json',physical)
        atomic_json(out/'scene_initial_audit.json',proof)
        atomic_npz(out/'initial_state.npz',values=ii.values)
        atomic_json(out/'initial_state.json',{'state_id':ii.state_id,'model_hash':ii.model_hash,'state_mask':int(STATE)})
        atomic_json(out/'physical_limits.json',{'lower':rr.lower,'upper':rr.upper,'geom_rbound':rr.model.geom_rbound,
            'B':rr.B,'limited':rr.model.jnt_limited[rr.joints],'ranges':rr.model.jnt_range[rr.joints],
            'q0':rr.spec.planner_zero,'velocity':rr.spec.planner_velocity_limits,'acceleration':rr.spec.planner_acceleration_limits})
        base=public_goal_initializer(task,rr,ii);teacher=np.asarray(ident['selection']['teacher']['Z'])
        original_basis=np.asarray(read_json(SOURCE/'learning/aligned/codec.json')['basis'])
        atomic_npz(out/'representation_inputs.npz',base=base,teacher=teacher,original_basis=original_basis)
        deadline=time.time()+max(0,P['wall_s']-previous_elapsed)
        ledger=Ledger(out,{'identity_hash':digest(ident)});budget=Budget(out,rr,deadline)
        def unit(key,stage,action):
            if key in ledger.data['completed']:return
            ledger.begin(key,stage,P['wall_s'],64);before=rr.calls.copy();budget.check()
            action();budget.check();ledger.finish({'status':'DONE'},{k:rr.calls[k]-before[k] for k in before});budget.save()
        c=task['condition']
        def quick(z):return all(quick_gates(rr,Reference(z,c['initial_features'],c['duration'])).values())
        candidates=[]
        for rank in P['ranks']:
            basis=fixed_basis(original_basis,rank);raw=project(teacher,base,basis)
            for mode in P['modes']:
                name=f'r{rank:02d}_{mode}';d=out/'candidates'/name
                value,details=prepare(raw,base,basis,c['initial_features'],rr.lower,rr.upper,quick,mode)
                arrays={'teacher':teacher,'base':base,'basis':basis,'raw':raw,'output':value}
                arrays.update({k:v for k,v in details.items() if isinstance(v,np.ndarray)})
                atomic_npz(d/'representation.npz',**arrays)
                metadata={k:v for k,v in details.items() if not isinstance(v,np.ndarray)}
                ref=Reference(value,c['initial_features'],c['duration'])
                row={'id':name,'rank':rank,'dimensions':rank*17,'mode':mode,'Z':value,'reference_hash':ref.reference_hash,
                     'task_hash':task['task_hash'],'metadata':metadata,'quick_gates':quick_gates(rr,ref),
                     'teacher_max_control_difference_rad':float(abs(value-teacher).max()),
                     'basis_fitted_to_DEV':False,'training_eligible':False}
                atomic_json(d/'candidate.json',row);candidates.append(row)
        # Freeze exact deterministic candidate controls before evaluating any new path.
        atomic_json(out/'candidate_manifest.json',{'before_model_evaluation':True,'candidates':candidates,'digest':digest(candidates)})
        prior={}
        for d in [PREVIOUS/'teachers'/P['teacher'],PREVIOUS/'representation/pure_codec']:
            r=read_json(d/'validation.json');prior[r['reference_hash']]=(d,r)
        canonical={};owners=[]
        for row in candidates:
            name=row['id'];d=out/'candidates'/name;ref=row['reference_hash']
            if ref in canonical:
                src=out/'candidates'/canonical[ref];r=read_json(src/'validation.json')
                atomic_json(d/'validation.json',r)
                atomic_json(d/'validation_origin.json',{'kind':'same_run_exact_duplicate','owner':canonical[ref],'reference_hash':ref})
                continue
            canonical[ref]=name;owners.append(name)
            def evaluate(row=row,d=d,ref=ref):
                progress('representation_validation',candidate=row['id'])
                if ref in prior:
                    source,r=prior[ref]
                    assert np.array_equal(r['Z'],row['Z']) and r['task_hash']==task['task_hash']
                    atomic_json(d/'validation.json',r)
                    atomic_json(d/'validation_origin.json',{'kind':'historical_exact_identity','source':source.relative_to(ROOT).as_posix(),'reference_hash':ref})
                else:
                    validate(d,row['Z'],task,rr,ii,cc,profile,budget)
                    atomic_json(d/'validation_origin.json',{'kind':'new_model_validation','reference_hash':ref})
            unit('validate_'+name,'model_validation',evaluate)
        selected=[n for n in owners if read_json(out/'candidates'/n/'validation.json')['eligible']]
        assert len(selected)<=P['execution_limit']
        selection={'ids':selected,'before_execution':True,'order':'rank_then_mode_exact_hash_dedup','candidate_digest':digest(candidates)}
        if (out/'execution_selection.json').exists():assert read_json(out/'execution_selection.json')==selection
        else:atomic_json(out/'execution_selection.json',selection)
        for name in owners:
            d=out/'candidates'/name;r=read_json(d/'validation.json')
            def execute(name=name,d=d,r=r):
                ex=d/'execution'
                atomic_json(ex/'selection.json',{'before_execution':True,'task_hash':task['task_hash'],'reference_hash':r['reference_hash'],'eligible':r['eligible']})
                if name not in selected:
                    atomic_json(ex/'outcome.json',{'status':'NO_CANDIDATE'});return
                progress('native_torque_and_replay',candidate=name)
                result=audit_execution(task,profile,rr,ii,Reference(r['Z'],c['initial_features'],c['duration']),ex,budget.deadline)
                result['goal_source']='frozen_original_APPLICATION_DEV_world_goal'
                atomic_json(ex/'report.json',result);atomic_json(ex/'outcome.json',{'status':result['level']})
            unit('execute_'+name,'selected_execution',execute)
        for row in candidates:
            name=row['id'];owner=canonical[row['reference_hash']]
            atomic_json(out/'candidates'/name/'execution_origin.json',{'owner':owner,'kind':'fresh_execution_owner' if name==owner else 'same_run_exact_duplicate',
                'status':read_json(out/'candidates'/owner/'execution/outcome.json')['status']})
        assert_frozen(task);progress('saved_evidence_audit')
        checked=verify(out,ident);atomic_json(out/'independent_verification.json',checked)
        return report(out,checked)
