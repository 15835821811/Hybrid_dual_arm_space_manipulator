"""Detached bounded worker, terminal status and conservative no-repeat resume."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from ff_prior.storage import read_json
from ff_prior_pilot_train.storage import atomic_json
from ff_prior_coverage_train.job import Monitor, status, TERMINAL
from ff_prior.recovery import recover_dead_lock, process_is_alive
from .protocol import ROOT, SOURCE, DEFAULT_OUT


def completed_no_work(out):
    from .inputs import identity_for
    frozen = read_json(out/'run.json')['identity']
    if identity_for(frozen['source'])[0] != frozen:
        raise ValueError('completed run source identity changed')
    summary = read_json(out/'summary.json')
    if summary['status'] not in ('FINISHED', 'FINISHED_WITH_BLOCKED_ITEMS'):
        raise ValueError('summary not complete')
    return dict(status(out), resume='NO_WORK_ALREADY_COMPLETE')


def worker(source, out, resume=False):
    out = Path(out).resolve()
    previous = read_json(out/'status.json')['elapsed_s'] if resume and (out/'status.json').exists() else 0.
    with Monitor(out, previous) as monitor:
        try:
            from .experiment import run
            result = run(source, out, monitor.update, resume, previous)
            monitor.finish(result)
            return 0 if result['status']=='FINISHED' else 2
        except Exception as error:
            traceback.print_exc()
            failure = {'status': 'FAILED', 'reason': repr(error), 'traceback': traceback.format_exc()}
            atomic_json(out/'failure.json', failure)
            previous_report = (out/'REPORT.md').read_text(encoding='utf-8') if (out/'REPORT.md').exists() else ''
            (out/'REPORT.md').write_text('# E256T 未完成\n\n'+repr(error)+'\n\n状态 FAILED；未完成不能写为通过，已有结果保留。\n\n'+previous_report, encoding='utf-8')
            monitor.finish(failure)
            return 1


def launch(source=SOURCE, out=DEFAULT_OUT, resume=False):
    source, out = Path(source).resolve(), Path(out).resolve()
    if source != SOURCE.resolve() or out != DEFAULT_OUT.resolve():
        raise ValueError('only frozen source and independent E256T output are allowed')
    if not resume:
        out.mkdir(parents=True, exist_ok=False)
    elif not (out/'run.json').exists():
        raise ValueError('resume requires frozen run identity')
    guard = out/'launch.guard'
    fd = os.open(guard, os.O_WRONLY|os.O_CREAT|os.O_EXCL)
    try:
        state = status(out)
        if state.get('owner_alive') or state.get('status')=='RUNNING':
            raise RuntimeError('worker alive; duplicate launch refused')
        if (out/'run.lock').exists() and process_is_alive(read_json(out/'run.lock')['pid']):
            raise RuntimeError('run lock owner alive; duplicate launch refused')
        if resume and state.get('status') in ('FINISHED','FINISHED_WITH_BLOCKED_ITEMS'):
            return completed_no_work(out)
        if resume and state.get('status') not in TERMINAL:
            atomic_json(out/'interruption.json', {'previous_status': state, 'unknown_uncheckpointed_work': True,
                'policy': 'consume reserved units; never repeat unknown physics'})
        if (out/'run.lock').exists():
            recover_dead_lock(out)
        argv = [sys.executable, '-u', '-m', 'ff_prior_expand256_train.job', 'worker', '--source', str(source), '--output', str(out)]
        if resume:
            argv.append('--resume')
        env = dict(os.environ, PYTHONUNBUFFERED='1', PYTHONIOENCODING='utf-8',
                   OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
        flags = {'creationflags': subprocess.CREATE_NO_WINDOW|subprocess.CREATE_NEW_PROCESS_GROUP} if os.name=='nt' else {'start_new_session': True}
        with (out/'worker.stdout.log').open('ab') as stdout, (out/'worker.stderr.log').open('ab') as stderr:
            proc = subprocess.Popen(argv, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr, env=env, **flags)
        atomic_json(out/'launcher.json', {'pid': proc.pid, 'argv': argv, 'started': time.time(), 'detached': True})
        for _ in range(100):
            if (out/'run.lock').exists() or proc.poll() is not None:
                break
            time.sleep(.1)
        return {'status': 'LAUNCHED', 'launcher_pid': proc.pid, 'worker': status(out), 'output': str(out)}
    finally:
        os.close(fd)
        guard.unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('start','resume','status','report','worker','await-source'))
    parser.add_argument('--source', default=str(SOURCE))
    parser.add_argument('--output', default=str(DEFAULT_OUT))
    parser.add_argument('--resume', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    os.chdir(ROOT)
    if args.action=='await-source':
        from .chain import start
        print(json.dumps(start(),ensure_ascii=False,indent=2));return 0
    if args.action=='worker':
        return worker(args.source, args.output, args.resume)
    if args.action in ('start','resume'):
        result = launch(args.source, args.output, args.action=='resume')
    elif args.action=='status':
        from .protocol import CHAIN_OUT
        result = {'training':status(args.output),'dependency_chain':status(CHAIN_OUT),'expansion':status(SOURCE)}
    elif (Path(args.output)/'REPORT.md').exists():
        print((Path(args.output)/'REPORT.md').read_text(encoding='utf-8'))
        return 0
    else:
        result = {'status': 'REPORT_NOT_READY', 'progress': status(args.output)}
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__=='__main__':
    sys.exit(main())
