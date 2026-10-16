"""Import only committed exact-task derivatives from the immutable failed run."""
import ast
import inspect
import shutil
import zipfile
from pathlib import Path
import numpy as np
from ff_prior.storage import read_json,file_digest,digest
from ff_prior.recovery import process_is_alive
from .storage import atomic_json

ROOT=Path(__file__).resolve().parents[1]
MANIFEST=ROOT/'ff_prior_pilot_train/evidence/recovery_001/failed_run_manifest.json'


def function_ast(source,name):
    return ast.dump(next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name==name),include_attributes=False)


def import_completed(out,arrays,rows,codec,ledger,identity):
    from . import sensitivity,learning
    out=Path(out);manifest=read_json(MANIFEST);source=Path(manifest['source'])
    if source==out:raise ValueError('failed output must remain immutable')
    if process_is_alive(read_json(source/'status.json')['pid']):raise ValueError('reuse source still alive')
    if any(file_digest(source/p)!=h for p,h in manifest['files'].items()):raise ValueError('failed run evidence changed')
    old=read_json(source/'run.json')['identity']
    for key in ('source','source_files','tasks_hash'):
        if old[key]!=identity[key]:raise ValueError('reuse input identity changed: '+key)
    for p,h in old['source_hashes'].items():
        if not p.startswith('ff_prior_pilot_train/') and identity['source_hashes'].get(p)!=h:
            raise ValueError('physics dependency changed: '+p)
    if old['protocol']['sensitivity']!=sensitivity.CONTRACT:raise ValueError('sensitivity contract changed')
    if digest(codec.state)!=digest(read_json(source/'learning/codec.json')):raise ValueError('reuse codec changed')
    # Orchestration/I/O may change; numerical derivative and codec functions may not.
    with zipfile.ZipFile(source/'executed_source.zip') as archive:
        for module,names in ((sensitivity,('directions','coefficient_map','build')),(learning,('fit_codec',))):
            previous=archive.read('ff_prior_pilot_train/'+Path(module.__file__).name).decode('utf-8-sig')
            current=inspect.getsource(module)
            for name in names:
                if function_ast(previous,name)!=function_ast(current,name):raise ValueError('numerical function changed: '+name)
    z,_,_,ids,splits,_=arrays;anchors={}
    for i,(r,s) in enumerate(zip(rows,splits)):
        if s=='TRAIN':anchors.setdefault(r['task']['task_hash'],i)
    prior=read_json(source/'ledger.json');imported=[]
    for work,result in prior['completed'].items():
        if result!={'stage':'sensitivity','status':'PASS'}:raise ValueError('nonqualified cache record')
        key=work.removeprefix('sensitivity_');i=anchors[key]
        jp=source/'sensitivity'/f'{key}.json';npz=jp.with_suffix('.npz');proof=read_json(jp)
        if not (proof['status']=='PASS' and proof['task_hash']==key and proof['reference_hash']==digest(z[i]) and proof['anchor_sample_id']==ids[i] and proof['surrogate_scope']==sensitivity.CONTRACT and proof['calls']['coupled']==39):raise ValueError('cache task/reference mismatch')
        if not all(c['passed'] and c['relative_difference']<=.02 for c in proof['half_step_checks']) or proof['mixed_direction_relative_error']>.05:raise ValueError('cache derivative validation failed')
        with np.load(npz) as a:
            if a['G'].shape!=(12,codec.dim) or not np.isfinite(a['G']).all():raise ValueError('invalid cached matrix')
        target=out/'sensitivity';target.mkdir(exist_ok=True)
        for p in (jp,npz):
            dst=target/p.name
            if dst.exists():
                if file_digest(dst)!=file_digest(p):raise ValueError('reuse output conflict')
            else:shutil.copyfile(p,dst)
        if work not in ledger.data['completed']:
            ledger.begin(work,'sensitivity',14400,sensitivity.CONTRACT['max_anchors'])
            ledger.finish({'status':'PASS','reused_from':str(source)},proof['calls'])
        imported.append(key)
    report={'status':'PASS','source':str(source),'manifest_sha256':file_digest(MANIFEST),'anchors':len(imported),
        'reused_keys':imported,'inherited_elapsed_s':manifest['elapsed_s'],'reused_coupled_calls':39*len(imported),
        'previous_failed_attempt':manifest['inflight'],'previous_failed_attempt_extra_calls':manifest['failed_attempt_extra_calls'],
        'previous_failure_erased':False,'new_physics_calls_for_import':0}
    atomic_json(out/'sensitivity_reuse.json',report)
    return report
