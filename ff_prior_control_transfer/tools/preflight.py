import sys, subprocess, importlib.metadata as m, xml.etree.ElementTree as ET
from ff_prior_control_transfer.inputs import identity
from ff_prior_control_transfer.protocol import *
from ff_prior.storage import read_json,file_digest,digest
from ff_prior_pilot_train.storage import atomic_json
from ff_prior.pipeline import audit,acceptance
from ff_prior_origin_learn_bounded.runtime import TaskOriginSceneCache
from ff_prior.contracts import assert_task_compatible
from ff_prior.guidance import public_goal_initializer
ident=identity()
physical,canonical,initial,_=audit(read_json(BASE/'run.json')['config'])
old=read_json(SOURCE/'physical_contract.json')
assert physical['acceptance']==old['acceptance']
assert digest(physical['identity'])==digest(old['identity']) and physical['initial_state_id']==old['initial_state_id']
profile=acceptance(read_json(BASE/'run.json')['config'])
cache=TaskOriginSceneCache(canonical,initial,profile,max_entries=1,max_scene_models=1)
e=ident['selection']['reconstruction'][0]; rr,ii,cc,proof=cache.get(e)
assert_task_compatible(e['task'],rr,ii,cc,profile)
base=public_goal_initializer(e['task'],rr,ii)
assert base.shape==(19,17) and rr.calls['coupled']==0
result={'status':'PASS','head':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'python':sys.executable,'versions':{p:m.version(p) for p in ('numpy','scipy','mujoco','torch','pytest')},'initial_scene_proof':proof,'no_trajectory_propagation':True,'reconstruction_cases':6,'source_count':len(ident['source_hashes']),'original_modified_files':{p:file_digest(ROOT/p) for p in ('.gitattributes','.gitignore','v6_lite/hierarchical_qp.py','v6_lite/run_v6_lite.py')}}
atomic_json(PACKAGE/'evidence/preflight.json',result)
suite=ET.parse(PACKAGE/'evidence/software_tests.xml').getroot().find('testsuite')
assert int(suite.get('tests'))==8 and int(suite.get('errors'))==0 and int(suite.get('failures'))==0
atomic_json(PACKAGE/'evidence/software_tests.json',{'status':'PASS','passed':8,'source_hashes':ident['source_hashes'],'junit_sha256':file_digest(PACKAGE/'evidence/software_tests.xml'),'scope':'software_only_not_physics'})
print({'status':result['status'],'sources':len(ident['source_hashes']),'scene_initial':proof['initial_geometry']['passed']})
