"""Honest CLI: unimplemented stages exit nonzero rather than emit fake PASS."""
import argparse
import json
import subprocess
import sys
from .orchestration import ROOT,read_state,unit


def main(argv=None):
    p=argparse.ArgumentParser(prog='python -m ffgd.cli')
    sub=p.add_subparsers(dest='command',required=True)
    for name in ('doctor','status','next'):sub.add_parser(name)
    t=sub.add_parser('test');t.add_argument('--suite',choices=['architecture','physics','gradients'],required=True);t.add_argument('--samples',type=int,default=32)
    for name in ('data','train','plan','validate','execute','replay','benchmark','report','resume'):
        s=sub.add_parser(name);s.add_argument('--config');s.add_argument('--run');s.add_argument('--task');s.add_argument('--unit')
    args=p.parse_args(argv)
    if args.command in ('status','next'):
        s=read_state();print(json.dumps(s if args.command=='status' else s['next_action'],indent=2,ensure_ascii=False));return 0
    if args.command=='doctor':
        from .calibration import doctor
        return unit('doctor',doctor,dict(max_wall_seconds=180,max_workers=1))
    if args.command=='test':
        if args.suite=='architecture':
            def software(path):
                r=subprocess.run([sys.executable,'-m','pytest','ffgd/tests','-q',f'--junitxml={path / "tests.xml"}'],cwd=ROOT,capture_output=True,text=True)
                (path/'output.txt').write_text(r.stdout+r.stderr,encoding='utf-8');print(r.stdout+r.stderr)
                return dict(passed=r.returncode==0,exit_code=r.returncode,scope='implemented software contracts only',ARCH_01_to_10='NOT_RUN')
            return unit('software',software,dict(max_wall_seconds=120))
        if args.suite=='physics':
            if args.samples<1: p.error('samples must be positive')
            from .calibration import physics
            return unit('physics',lambda path:physics(path,args.samples),dict(samples=args.samples,max_wall_seconds=600))
        if args.suite=='gradients':
            from .calibration import gradients
            return unit('path_gradients',lambda path:gradients(path,3),dict(cases=3,directions=8,fd_steps=3,max_wall_seconds=900))
    p.error(f'{args.command} is NOT_IMPLEMENTED; follow earliest dependency in next')


if __name__=='__main__':sys.exit(main())
