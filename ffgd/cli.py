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
    t=sub.add_parser('test');t.add_argument('--suite',choices=['architecture','physics','gradients','geometry','energy','execution','momentum','distance-precision','convex','feature-switch','full-space'],required=True);t.add_argument('--samples',type=int)
    for name in ('data','train','plan','validate','execute','replay','benchmark','report','resume'):
        s=sub.add_parser(name);s.add_argument('--config');s.add_argument('--run');s.add_argument('--task');s.add_argument('--unit')
    args=p.parse_args(argv)
    if args.command in ('status','next'):
        s=read_state();print(json.dumps(s if args.command=='status' else s['next_action'],indent=2,ensure_ascii=False));return 0
    if args.command=='doctor':
        from .calibration import doctor
        return unit('doctor',doctor,dict(max_wall_seconds=180,max_workers=1))
    if args.command=='data':
        from .data import generate
        from pathlib import Path
        if not args.config:p.error('data requires --config')
        config=json.loads(Path(args.config).read_text(encoding='utf-8-sig'))
        return unit('prior_data',lambda path:generate(path,config,args.run),dict(max_wall_seconds=config['max_wall_seconds'],max_attempts=config['max_attempts']))
    if args.command=='train':
        from .training import train
        from pathlib import Path
        if not args.config:p.error('train requires --config')
        config=json.loads(Path(args.config).read_text(encoding='utf-8-sig'))
        return unit('training',lambda path:train(path,config,args.run),dict(max_wall_seconds=config['max_wall_seconds'],updates=config['updates']))
    if args.command=='test':
        if args.samples is None:args.samples=3 if args.suite in ('gradients','energy','full-space') else 32
        if args.samples<1:p.error('samples must be positive')
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
            return unit('path_gradients',lambda path:gradients(path,args.samples),dict(cases=args.samples,directions=8,fd_steps=3,max_wall_seconds=900))
        if args.suite in ('geometry','energy'):
            from .calibration_constraints import geometry_audit,energy_gradients
            function=geometry_audit if args.suite=='geometry' else energy_gradients
            count=args.samples
            return unit(args.suite,lambda path:function(path,count),dict(cases=count,max_wall_seconds=1800))
        if args.suite=='execution':
            from .calibration_execution import run
            return unit('execution_calibration',run,dict(max_wall_seconds=600,native_steps=1000))
        if args.suite=='momentum':
            from .calibration_momentum import run
            return unit('momentum_sweep',run,dict(max_wall_seconds=600,native_steps=21000))
        if args.suite=='distance-precision':
            from .calibration_distance_precision import run
            return unit('distance_precision',run,dict(max_wall_seconds=180,cases=2,tolerances=3))
        if args.suite=='convex':
            from .calibration_convex import run
            return unit('convex_geometry',lambda path:run(path,args.samples),dict(max_wall_seconds=180,cases=args.samples))
        if args.suite=='feature-switch':
            from .calibration_feature_switch import run
            return unit('feature_switch',run,dict(max_wall_seconds=180,cases=1,epsilon_steps=6))
        if args.suite=='full-space':
            from .calibration_constraints import energy_gradients
            return unit('full_space_gradients',lambda path:energy_gradients(path,args.samples,24),dict(max_wall_seconds=900,cases=args.samples,free_dim=323))
    p.error(f'{args.command} is NOT_IMPLEMENTED; follow earliest dependency in next')


if __name__=='__main__':sys.exit(main())
