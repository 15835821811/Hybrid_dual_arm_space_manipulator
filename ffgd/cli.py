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
    t=sub.add_parser('test');t.add_argument('--suite',choices=['architecture','physics','gradients','geometry','energy','execution','momentum','distance-precision','convex','feature-switch','full-space','compiled'],required=True);t.add_argument('--samples',type=int)
    for name in ('data','train','learning-checks','architecture-audit','architecture-io','guidance-diagnostic','plan','validate','execute','replay','benchmark','report','resume'):
        s=sub.add_parser(name);s.add_argument('--config');s.add_argument('--run');s.add_argument('--task');s.add_argument('--unit')
    args=p.parse_args(argv)
    if args.command in ('status','next'):
        s=read_state();print(json.dumps(s if args.command=='status' else s['next_action'],indent=2,ensure_ascii=False));return 0
    if args.command=='resume':
        from .recovery import resume
        result=resume();print(json.dumps(result,indent=2,ensure_ascii=False));return 0 if result['ready'] else 1
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
    if args.command=='learning-checks':
        from .learning_checks import run
        from pathlib import Path
        if not args.config:p.error('learning-checks requires --config')
        config=json.loads(Path(args.config).read_text(encoding='utf-8-sig'))
        return unit('learning_checks',lambda path:run(path,config),dict(max_wall_seconds=300,samples=5,steps=60))
    if args.command=='architecture-audit':
        from .architecture_audit import audit
        from pathlib import Path
        if not args.config:p.error('architecture-audit requires --config')
        config=json.loads(Path(args.config).read_text(encoding='utf-8-sig'))
        return unit('architecture_audit',lambda path:audit(path,config),dict(max_wall_seconds=600))
    if args.command=='architecture-io':
        from .calibration_io import run
        from pathlib import Path
        if not args.config:p.error('architecture-io requires --config')
        config=json.loads(Path(args.config).read_text(encoding='utf-8-sig'))
        return unit('software',lambda path:run(path,config),dict(max_wall_seconds=30))
    if args.command=='guidance-diagnostic':
        from .calibration_guidance import run
        from pathlib import Path
        if not args.config:p.error('guidance-diagnostic requires --config')
        config=json.loads(Path(args.config).read_text(encoding='utf-8-sig'))
        return unit('guidance_diagnostic',lambda path:run(path,config),dict(max_wall_seconds=180,one_step_rays=5))
    if args.command=='plan':
        from .problem_io import load_problem,planning_config
        from .api import plan
        from pathlib import Path
        if not args.config or not args.task:p.error('plan requires --config and --task')
        config,budget=planning_config(args.config)
        def planning(path):
            robot,snapshot,task,problem=load_problem(args.task)
            (path/'submitted_problem.json').write_text(json.dumps(problem,indent=2),encoding='utf-8')
            problem['initial']={'snapshot_file':'initial_state.npz'}
            (path/'problem.json').write_text(json.dumps(problem,indent=2),encoding='utf-8')
            profile=json.loads(Path(config['profile']).read_text(encoding='utf-8-sig'))
            from .architecture_guard import LegacySentinel
            from .orchestration import atomic_json
            sentinel=LegacySentinel(allowed_data_reads=[config['checkpoint'],Path(config['checkpoint']).with_suffix('.identity.json')],output_directory=path)
            try:
                with sentinel:
                    candidate,result=plan(robot,snapshot,task,config['checkpoint'],budget,profile,path,config.get('device','cpu'),config.get('disabled_guide_terms',()),config.get('guide_weights'),config.get('physics_backend','eager'))
            except Exception as error:
                if not (path/'planning_result.json').exists():
                    records=[]
                    for index,seed in enumerate(budget.seeds):
                        record_file=path/f'candidate_{index:03d}'/'candidate.json'
                        record=json.loads(record_file.read_text()) if record_file.exists() else dict(index=index,seed=seed,sampled=False,validated=False,status='NOT_RUN_SETUP_FAILURE')
                        if record['status']=='STARTED':record.update(status='INCOMPLETE_EXPERIMENT_FAILURE')
                        records.append(record)
                    atomic_json(path/'planning_result.json',dict(status='EXPERIMENT_FAILED',requested_candidates=len(budget.seeds),records=records,
                                sampled=sum(r['sampled'] for r in records),valid=0,selected_index=None,error=str(error)))
                raise
            finally:atomic_json(path/'architecture_sentinel.json',sentinel.report())
            return dict(passed=candidate is not None,**result)
        return unit('planning',planning,dict(max_wall_seconds=budget.max_wall_seconds+60,candidates=len(budget.seeds),steps=budget.sampling.steps))
    if args.command=='execute':
        from .execution_acceptance import execute_selected
        if not args.run:p.error('execute requires --run planning-directory')
        return unit('selected_execution',lambda path:execute_selected(path,args.run),dict(max_wall_seconds=900))
    if args.command=='validate':
        from .artifact_checks import validate_saved
        if not args.run:p.error('validate requires --run planning-directory; optional --unit candidate-index')
        return unit('independent_validation',lambda path:validate_saved(path,args.run,args.unit),dict(max_wall_seconds=600))
    if args.command=='replay':
        from .artifact_checks import replay_saved
        if not args.run:p.error('replay requires --run execution-directory')
        return unit('independent_replay',lambda path:replay_saved(path,args.run),dict(max_wall_seconds=600))
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
        if args.suite=='compiled':
            from .calibration import physics
            from .calibration_constraints import energy_gradients
            def compiled(path):
                (path/'physics').mkdir();(path/'gradients').mkdir()
                a=physics(path/'physics',32,True);b=energy_gradients(path/'gradients',1,24,True)
                return dict(passed=a['passed'] and b['passed'],physics=a,gradients=b,scope='optional exact-graph TorchScript CPU backend calibration')
            return unit('compiled_calibration',compiled,dict(max_wall_seconds=600,native_cases=32,full_free_gradient_cases=1))
    p.error(f'{args.command} is NOT_IMPLEMENTED; follow earliest dependency in next')


if __name__=='__main__':sys.exit(main())
