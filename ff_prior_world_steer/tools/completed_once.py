"""One completed-run no-op check; persist manifests BEFORE invoking resume."""
from pathlib import Path
import json,hashlib,gzip,subprocess,os,time,sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
RUN=ROOT/'artifacts/v7_1/world_steer_001'
EVIDENCE=ROOT/'ff_prior_world_steer/evidence'
RECEIPT=EVIDENCE/'completed_resume.json'
ATTEMPT=EVIDENCE/'completed_resume_attempt.json'
MANIFESTS=ROOT/'artifacts/review_manifests/world_steer_001'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for part in iter(lambda:f.read(1024*1024),b''):h.update(part)
 return h.hexdigest()
def inventory():
 return {p.relative_to(RUN).as_posix():{'size':p.stat().st_size,'sha256':sha(p)}
         for p in sorted(RUN.rglob('*')) if p.is_file()}
def write(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def save_manifest(name,data):
 p=MANIFESTS/name
 with gzip.open(p,'wt',encoding='utf-8') as f:json.dump(data,f,ensure_ascii=False,sort_keys=True)
 return {'path':p.relative_to(ROOT).as_posix(),'sha256':sha(p),'bytes':p.stat().st_size}
assert not RECEIPT.exists() and not ATTEMPT.exists(),'NO_REPEAT_RESUME'
s=json.loads((RUN/'status.json').read_text(encoding='utf-8'))
assert s['status'] in ('FINISHED','FINISHED_WITH_BLOCKED_ITEMS')
from ff_prior_fullchain.job import process_probe
assert process_probe(s['pid'],s['started_at'])['alive'] is False
MANIFESTS.mkdir(parents=True,exist_ok=False)
before=inventory();bm=save_manifest('before.json.gz',before)
write(ATTEMPT,{'status':'CALL_RESERVED_ONCE','before':bm,'files':len(before),'time':time.time(),'repeat_forbidden':True})
result=subprocess.run([str(ROOT/'.venv/Scripts/python.exe'),'-m','ff_prior_world_steer.job','resume'],
 cwd=ROOT,env=dict(os.environ,PYTHONIOENCODING='utf-8',PYTHONUNBUFFERED='1'),
 stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=120)
stdout=EVIDENCE/'completed_resume.stdout.bin';stderr=EVIDENCE/'completed_resume.stderr.bin'
stdout.write_bytes(result.stdout);stderr.write_bytes(result.stderr)
after=inventory();am=save_manifest('after.json.gz',after)
decoded=result.stdout.decode('utf-8')
reply=json.loads(decoded)
same=before==after
receipt={'status':'PASS' if same and result.returncode==0 and reply.get('resume')=='NO_WORK_ALREADY_COMPLETE' else 'FAIL',
 'run':RUN.relative_to(ROOT).as_posix(),'calls':1,'returncode':result.returncode,'resume_result':reply.get('resume'),
 'before':bm,'after':am,'core_files':len(before),'core_bytes':sum(v['size'] for v in before.values()),
 'same_file_list':set(before)==set(after),'same_hash_and_size':same,
 'stdout':{'path':stdout.relative_to(ROOT).as_posix(),'sha256':sha(stdout),'bytes':len(result.stdout)},
 'stderr':{'path':stderr.relative_to(ROOT).as_posix(),'sha256':sha(stderr),'bytes':len(result.stderr)},
 'changed_files':[k for k in before.keys()|after.keys() if before.get(k)!=after.get(k)],
 'repeat_forbidden':True,'new_physics_requested':False}
write(RECEIPT,receipt)
print(json.dumps(receipt,ensure_ascii=True,indent=2))

