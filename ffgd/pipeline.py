"""Bounded serial experiment queue, not a programming agent or future chat.

Wait for the identified existing data worker, run the three concrete M3 units,
and stop at the first failure. Never relabel milestones or launch retries.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import psutil
from .orchestration import ROOT,ORCH,atomic_json,utc,read_state


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--dataset-run',required=True);args=parser.parse_args()
    dataset=Path(args.dataset_run).resolve();state=read_state();owner=state['active_unit']['process_identity'] if state['active_unit'] else None
    if owner is None:raise RuntimeError('expected an identified active dataset worker')
    worker=json.loads((dataset/'manifest.json').read_text(encoding='utf-8'))
    if owner!=worker['process_identity']:raise RuntimeError('dataset owner identity mismatch')
    configpaths=[ROOT/'configs/ffgd/train_debug32.json',ROOT/'configs/ffgd/train_pilot.json']
    frozen={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in configpaths}
    out=ORCH/f'm3_queue_{os.getpid()}';out.mkdir(exist_ok=False)
    own=psutil.Process();record=dict(status='WAITING_FOR_DATA',start=utc(),pid=own.pid,create_time=own.create_time(),host=platform.node(),
                                   command=own.cmdline(),dataset=str(dataset),expected_owner=owner,config_hashes=frozen,units=[])
    atomic_json(out/'queue.json',record);print(json.dumps(dict(queue=str(out),pid=own.pid,status=record['status'])),flush=True)
    try:
        deadline=time.monotonic()+worker['budget']['max_wall_seconds']+300
        while True:
            manifest=json.loads((dataset/'manifest.json').read_text(encoding='utf-8'))
            if manifest['status']!='RUNNING':break
            process=psutil.Process(owner['pid'])
            if abs(process.create_time()-owner['create_time'])>.01 or process.cmdline()!=owner['command']:raise RuntimeError('DATA_WORKER_IDENTITY_CHANGED')
            if time.monotonic()>deadline:raise RuntimeError('QUEUE_WAIT_BUDGET_EXHAUSTED')
            time.sleep(10)
        if manifest['status']!='PASSED':raise RuntimeError('DATA_WORKER_FAILED')
        data=json.loads((dataset/'dataset_manifest.json').read_text(encoding='utf-8'))
        if data['accepted']!=2000:raise RuntimeError('PILOT_SIZE_NOT_MET')
        def launch(name,config):
            command=[sys.executable,'-m','ffgd.cli',name,'--config',str(config)]
            record['status']='RUNNING_'+name.upper();record['current_command']=command;atomic_json(out/'queue.json',record)
            existing=set((ROOT/'artifacts/ffgd/runs').iterdir())
            logfile=out/f'{len(record["units"]):02d}_{name}.log'
            with logfile.open('w',encoding='utf-8') as log:
                child=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
                record['child_pid']=child.pid;record['child_create_time']=psutil.Process(child.pid).create_time();atomic_json(out/'queue.json',record)
                code=child.wait()
            created=set((ROOT/'artifacts/ffgd/runs').iterdir())-existing
            # Match exact argv because a lightweight test may have made another run.
            matches=[]
            for folder in created:
                try:m=json.loads((folder/'manifest.json').read_text(encoding='utf-8'))
                except FileNotFoundError:continue
                if m['command'][3:]==command[3:]:matches.append(folder)
            record['units'].append(dict(command=command,exit_code=code,log=str(logfile),runs=[str(p) for p in matches]));atomic_json(out/'queue.json',record)
            if code!=0:raise RuntimeError('QUEUED_UNIT_FAILED: '+name)
            if len(matches)!=1:raise RuntimeError('QUEUED_UNIT_RUN_ID_AMBIGUOUS')
            return matches[0]
        for p in configpaths:
            if hashlib.sha256(p.read_bytes()).hexdigest()!=frozen[str(p)]:raise RuntimeError('QUEUED_CONFIG_CHANGED')
        debug=launch('train',configpaths[0]);main=launch('train',configpaths[1])
        result=json.loads((main/'result.json').read_text(encoding='utf-8'))
        check=out/'learning_checks.json';atomic_json(check,dict(checkpoint=result['checkpoint'],dataset=str(dataset),device='cuda'))
        checks=launch('learning-checks',check)
        record.update(status='FINISHED_REQUIRES_AGENT_REVIEW',end=utc(),training_run=str(main),learning_checks_run=str(checks),child_pid=None)
        atomic_json(out/'queue.json',record);print(json.dumps(record),flush=True)
        return 0
    except Exception as ex:
        record.update(status='FAILED_NEEDS_AGENT_REVIEW',error=str(ex),end=utc());atomic_json(out/'queue.json',record)
        raise


if __name__=='__main__':sys.exit(main())
