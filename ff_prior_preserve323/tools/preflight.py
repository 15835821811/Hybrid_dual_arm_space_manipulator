"""Once-only binding freeze and software checks before background launch."""
import os
import platform
import subprocess
import sys
import torch
import numpy
import mujoco
from ff_prior.storage import read_json,file_digest
from ff_prior_pilot_train.storage import atomic_json
from ff_prior_preserve323.inputs import freeze,identity,sources
from ff_prior_preserve323.protocol import ROOT,PACKAGE,OUT,REUSE,P

assert not OUT.exists(),'experiment already exists'
assert not (PACKAGE/'evidence/preflight.json').exists(),'preflight already complete'
frozen=freeze() if not (PACKAGE/'evidence/selection.json').exists() else {'already_frozen':True}
old=read_json(ROOT/'ff_prior_control_transfer/evidence/preflight.json')['original_modified_files']
assert all(file_digest(ROOT/p)==h for p,h in old.items())
r=subprocess.run([sys.executable,'-m','pytest','ff_prior_preserve323/test_preserve.py','-q'],cwd=ROOT,capture_output=True,text=True,encoding='utf-8',env=dict(os.environ,PYTHONIOENCODING='utf-8'))
(PACKAGE/'evidence/software_tests.txt').write_text(r.stdout+r.stderr,encoding='utf-8')
assert r.returncode==0,(r.stdout,r.stderr)
ident=identity()
assert len(ident['selection']['fit_pool'])==len(ident['selection']['check_pool'])==16
atomic_json(PACKAGE/'evidence/software_tests.json',dict(status='PASS',returncode=0,source_hashes=ident['source_hashes'],
    stdout=r.stdout,scope='software and neural derivative tests; synthetic residual fixture is not physical evidence'))
commands={}
for name,args in [('head',['git','rev-parse','HEAD']),('branch',['git','branch','--show-current']),
    ('python_processes',['powershell','-NoProfile','-Command',"Get-CimInstance Win32_Process -Filter \"Name = 'python.exe'\" | Select-Object ProcessId,CommandLine | ConvertTo-Json -Compress"])]:
    x=subprocess.run(args,cwd=ROOT,capture_output=True,text=True,encoding='utf-8',errors='replace');commands[name]=x.stdout.strip()
atomic_json(PACKAGE/'evidence/preflight.json',dict(head=commands['head'],branch=commands['branch'],commands=commands,
    original_modified_files=old,python=sys.version,python_executable=sys.executable,torch=torch.__version__,numpy=numpy.__version__,mujoco=mujoco.__version__,
    platform=platform.platform(),frozen=frozen,source_files=len(ident['source_hashes']),bound_files=len(ident['binding']['files']),
    reuse_status=read_json(REUSE/'status.json')['status'],reuse_gate=read_json(REUSE/'reconstruction_gate.json'),
    reused_teacher_executions=3,new_teacher_executions=0,no_new_data=True,no_TEST=True,protocol=P))
print('PASS: frozen inputs, software tests, old evidence identities and user-file preservation')
