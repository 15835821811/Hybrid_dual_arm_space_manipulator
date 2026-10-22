"""One read-only check of supplied final report/decision/resume evidence.

This script never imports or invokes the job or resume entry points.
"""
from pathlib import Path
import json, gzip, hashlib
from datetime import datetime,timezone
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3];RUN=ROOT/'artifacts/v7_1/world_steer_001'
records=[];checks={}
def read(rel):
    p=ROOT/rel;b=p.read_bytes();records.append({'path':str(p),'sha256':hashlib.sha256(b).hexdigest(),'bytes':len(b),'mtime_utc':datetime.fromtimestamp(p.stat().st_mtime,timezone.utc).isoformat()});return b
report=read('ff_prior_world_steer/docs/ANALYSIS.md').decode('utf-8-sig')
decision=read('ff_prior_fullchain/docs/AFTER_WORLD_STEER_DECISION.md').decode('utf-8-sig')
receipt=json.loads(read('ff_prior_world_steer/evidence/completed_resume.json'))
attempt=json.loads(read('ff_prior_world_steer/evidence/completed_resume_attempt.json'))
manifest=[]
for side in ['before','after']:
    spec=receipt[side];b=read(spec['path']);checks[side+'_compressed_hash']=hashlib.sha256(b).hexdigest()==spec['sha256'] and len(b)==spec['bytes'];manifest.append(json.loads(gzip.decompress(b)))
before,after=manifest
checks['before_after_identity']=before==after
checks['manifest_file_count']=len(after)==receipt['core_files']==40730
checks['manifest_bytes']=sum(x['size'] for x in after.values())==receipt['core_bytes']==585993194
checks['current_file_names']=set(p.relative_to(RUN).as_posix() for p in RUN.rglob('*') if p.is_file())==set(after)
current_bad=[]
for name,expected in after.items():
    b=(RUN/name).read_bytes()
    if len(b)!=expected['size'] or hashlib.sha256(b).hexdigest()!=expected['sha256']:current_bad.append(name)
checks['current_40730_hashes_and_sizes']=not current_bad
outs={}
for side in ['stdout','stderr']:
    spec=receipt[side];b=read(spec['path']);checks[side+'_raw_hash']=hashlib.sha256(b).hexdigest()==spec['sha256'] and len(b)==spec['bytes'];outs[side]=b
checks['raw_resume_no_work']=json.loads(outs['stdout'])['resume']=='NO_WORK_ALREADY_COMPLETE' and outs['stderr']==b''
checks['single_call_receipt']=receipt['calls']==1 and receipt['returncode']==0 and receipt['status']=='PASS' and attempt['repeat_forbidden']
checks['before_manifest_reserved']=attempt['before']==receipt['before'] and attempt['files']==40730
checks['report_numbers_and_scope']=all(s in report for s in ['533,440','12,436','4,558','2,276','37,414','5,948,992','未运行','NO_WORK_ALREADY_COMPLETE','provisional'])
checks['decision_bounded_and_not_started']=all(s in decision for s in ['待测假设','先实现','未完成前不得启动','20860','22586','不能只按q合并','最终原生执行依旧从原真实初态开始'])
checks['proposed_call_arithmetic']=(4000+2800+80+40)*3+100==20860<22586
checks['proposed_native_ceiling_arithmetic']=3*27500+3*27000==163500<=165000
result={'status':'PASS' if all(checks.values()) else 'FAIL','scope':'one final supplied-report supplement, no job or resume call','checks':checks,'metadata_assertions':len(checks),'additional_core_file_hash_comparisons':len(after),'current_changed_files':current_bad,'inputs':records,'new_physics_calls':0,'new_resume_calls':0,'planned_closeout_receipt':'ff_prior_fullchain/evidence/world_steer_closeout.json','planned_receipt_status':'not reviewed; executor stated it will be created after main commit; no completion asserted','reviewer_interpretation':'Reports accurately distinguish C1 failure to obtain a route from unexecuted downstream stages. The 20,860-call proposal is arithmetic-only, not demonstrated cost or authorization. Freeze worst-case geometry/coupled/wall/storage and seed-prefix selection before new work; do not infer effectiveness or budget sufficiency.'}
(HERE/'supplement.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({'status':result['status'],'metadata_assertions':len(checks),'core_file_hash_comparisons':len(after),'failures':[k for k,v in checks.items() if not v]}))
