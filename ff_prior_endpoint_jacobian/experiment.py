import time,subprocess,zipfile
from pathlib import Path
from ff_prior.storage import read_json,digest,RunLock,file_digest
from ff_prior_pilot_train.storage import atomic_json,Ledger
from ff_prior.pipeline import audit,acceptance
from ff_prior_origin_learn_bounded.runtime import TaskOriginSceneCache
from ff_prior_delivery_train.learning import load as warm_load
from .protocol import ROOT,CONTROL,WARM,BASE,DEFAULT_OUT,PROTOCOL as P
from .inputs import identity
from . import data,learning,comparison

class Budget:
    def __init__(self,out,deadline):self.out=out;self.total=0;self.stage_used=0;self.deadline=deadline;self.global_deadline=deadline;self.name='preflight';self.limit=0
    def stage(self,name,limit,seconds=None):
        self.name=name;self.stage_used=0;self.limit=limit;self.deadline=min(self.global_deadline,time.time()+seconds) if seconds else self.global_deadline
    def reserve(self,n):
        if time.time()>=self.deadline or self.total+n>P['total_queries'] or self.stage_used+n>self.limit:raise RuntimeError('BOUNDED_QUERY_OR_TIME_LIMIT')
        if sum(p.stat().st_size for p in self.out.rglob('*') if p.is_file())>=P['max_output_bytes']:raise RuntimeError('STORAGE_LIMIT')
        self.total+=n;self.stage_used+=n
        atomic_json(self.out/'query_budget.json',{'reserved_coupled_upper':self.total,'stage':self.name,'stage_reserved':self.stage_used,'stage_limit':self.limit,'unknown_work_not_refunded':True})

def run(out,progress):
    from .audit_results import verify
    from .reporting import report
    out=Path(out);ident=identity();deadline=time.time()+P['wall_budget_s']
    with RunLock(out):
        proof=read_json(ROOT/'ff_prior_endpoint_jacobian/evidence/software_tests.json')
        if proof['status']!='PASS' or proof['source_hashes']!=ident['source_hashes']:raise ValueError('software proof stale')
        atomic_json(out/'run.json',{'identity':ident,'producer_head':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'TEST':False})
        with zipfile.ZipFile(out/'executed_source.zip','w',zipfile.ZIP_DEFLATED) as z:
            for name in ident['source_hashes']:z.write(ROOT/name,name)
        config=read_json(BASE/'run.json')['config'];physical,canonical,initial,_=audit(config);profile=acceptance(config)
        if physical['acceptance']!=read_json(CONTROL/'physical_contract.json')['acceptance']:raise ValueError('physical gate changed')
        atomic_json(out/'physical_contract.json',physical)
        cache=TaskOriginSceneCache(canonical,initial,profile,max_entries=16,max_scene_models=4)
        _,codec=warm_load(WARM/'learning/goal_loss');progress('load_exact_bound_TRAIN_cache')
        arrays,rows,G,d=data.load(codec)
        atomic_json(out/'training_input.json',{'TRAIN':14222,'VAL':48,'source':str(CONTROL),'data_hash':read_json(CONTROL/'learning/goal_loss/training.json')['metadata']['data_hash'],'new_data':0})
        ledger=Ledger(out,{'identity_hash':digest(ident)});budget=Budget(out,deadline)
        atomic_json(out/'reuse_manifest.json',{'control':str(CONTROL),'methods':P['reuse_methods'],'candidates':540,'executions':108,'status':'IDENTITY_BOUND','new_physics':False})
        budget.stage('M1',P['M1_queries'],P['M1_wall_s']);diag=learning.diagnostic(out,ident['selection'],codec,arrays,rows,G,d,cache,ledger,budget,progress)
        if diag['status']=='PASS':
            budget.stage('M2',27000,9000.);learning.train(out,ident['selection'],codec,arrays,rows,G,d,cache,ledger,budget,progress)
            bundle=learning.load(out/'learning');budget.stage('unguided',576);comparison.unguided(out,bundle,arrays,rows,d,cache,profile,ledger,budget,progress)
            budget.stage('DEV',20340);comparison.dev(out,bundle,cache,profile,ledger,budget,progress)
        else:atomic_json(out/'dependencies.json',{'M2':'BLOCKED_M1','M3_unguided':'BLOCKED_M1','M3_DEV':'BLOCKED_M1','no_dependent_training_or_physics':True})
        atomic_json(out/'runtime_cache.json',cache.stats());progress('read_only_new_raw_evidence_audit')
        audit_result=verify(out,ident,ledger);atomic_json(out/'independent_verification.json',audit_result)
        return report(out)
