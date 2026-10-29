"""Honest CLI: unimplemented stages exit nonzero rather than emit fake PASS."""
import argparse
import json
import subprocess
import sys
from .orchestration import ROOT,read_state,unit


def main(argv=None):
    p=argparse.ArgumentParser(prog='python -m ffgd.cli')
    sub=p.add_subparsers(dest='command',required=True)
    for name in ('doctor','status','next','init'):sub.add_parser(name)
    t=sub.add_parser('test');t.add_argument('--suite',choices=['architecture','physics','gradients','geometry','energy','execution','momentum','distance-precision','convex','feature-switch','full-space','full-space-peak','full-space-bodyframe','full-space-joint-peak','full-space-batched-path','full-space-origin','compiled'],required=True);t.add_argument('--samples',type=int)
    t.add_argument('--planning-margins')
    t.add_argument('--free-coordinate-schema',choices=['physical_position_v1','duration_squared_v1'],default='physical_position_v1')
    t.add_argument('--task-objective',choices=['center_squared','tolerance_squared_hinge'],default='center_squared')
    for name in ('data','train','learning-checks','architecture-audit','architecture-io','guidance-diagnostic','condition-diagnostic','weight-diagnostic','guard-diagnostic','collision-diagnostic','mass-rows-diagnostic','full-state-diagnostic','momentum-diagnostic','plan','validate','execute','replay','validate-chain','continue-state','benchmark','report','resume'):
        s=sub.add_parser(name);s.add_argument('--config');s.add_argument('--run');s.add_argument('--task');s.add_argument('--unit')
    args=p.parse_args(argv)
    if args.command=='init':
        from .orchestration import initialize
        print(json.dumps(initialize(),indent=2,ensure_ascii=False));return 0
    if args.command in ('status','next'):
        s=read_state();print(json.dumps(s if args.command=='status' else s['next_action'],indent=2,ensure_ascii=False));return 0
    if args.command=='resume':
        from .recovery import resume
        result=resume();print(json.dumps(result,indent=2,ensure_ascii=False));return 0 if result['ready'] else 1
    if args.command=='report':
        from .evidence_report import run
        return unit('evidence_report',run,dict(max_wall_seconds=30))
    if args.command=='validate-chain':
        from pathlib import Path
        from .artifact_checks import inspect_chain
        if not args.config:p.error('validate-chain requires --config')
        config=json.loads(Path(args.config).read_text(encoding='utf-8-sig'))
        return unit('execution_state_audit',lambda path:inspect_chain(path,config),dict(max_wall_seconds=config.get('max_wall_seconds',600)))
    if args.command=='benchmark':
        from pathlib import Path
        if not args.config:p.error('benchmark requires an explicit comparison config')
        config=json.loads(Path(args.config).read_text(encoding='utf-8-sig'))
        planning=json.loads(Path(config['planning_config']).read_text(encoding='utf-8-sig'))
        from .baselines.compare import run
        return unit('baseline_comparison',lambda path:run(path,config),dict(max_wall_seconds=planning['budget']['max_wall_seconds']+60,
                    candidates=len(planning['budget']['seeds']),method=config['method'],reserved_cpus=planning['budget'].get('cpu_threads',1)))
    if args.command=='doctor':
        from .calibration import doctor
        return unit('doctor',doctor,dict(max_wall_seconds=180,max_workers=1))
    if args.command=='data':
        from .data import generate,reencode_origin,merge_prior,reencode_duration_coordinates,reencode_free_terminal
        from pathlib import Path
        if not args.config:p.error('data requires --config')
        config=json.loads(Path(args.config).read_text(encoding='utf-8-sig'))
        if config.get('operation')=='reencode_free_terminal':
            if args.run:p.error('free-terminal reencoding does not support --run resume')
            return unit('prior_free_terminal_reencoding',lambda path:reencode_free_terminal(path,config),dict(max_wall_seconds=config['max_wall_seconds'],new_physical_trajectories=0))
        if config.get('operation')=='reencode_duration_coordinates':
            if args.run:p.error('duration reencoding does not support --run resume')
            return unit('prior_duration_reencoding',lambda path:reencode_duration_coordinates(path,config),dict(max_wall_seconds=config['max_wall_seconds'],new_physical_trajectories=0))
        if config.get('operation')=='merge_prior':
            if args.run:p.error('prior merge does not support --run resume')
            return unit('prior_union',lambda path:merge_prior(path,config),dict(max_wall_seconds=config['max_wall_seconds'],new_physical_trajectories=0))
        if config.get('operation')=='reencode_boundary_origin':
            if args.run:p.error('boundary-origin reencoding does not support --run resume')
            return unit('prior_coordinate_reencoding',lambda path:reencode_origin(path,config),dict(max_wall_seconds=config['max_wall_seconds'],new_physical_trajectories=0))
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
        diagnostic_budget=(dict(transition_checks=len(config['transition_steps']),next_network_calls=2*len(config['transition_steps']))
                           if config.get('transition_steps') else dict(one_step_rays=len(config.get('coefficients',config.get('lengths',[.1,.25,.5,1.,2.])))))
        return unit('guidance_diagnostic',lambda path:run(path,config),dict(max_wall_seconds=180,**diagnostic_budget))
    if args.command=='condition-diagnostic':
        from .calibration_condition import run
        from pathlib import Path
        if not args.config:p.error('condition-diagnostic requires --config')
        config=json.loads(Path(args.config).read_text(encoding='utf-8-sig'))
        return unit('condition_diagnostic',lambda path:run(path,config),dict(max_wall_seconds=180,samples=4 if config.get('mode')=='train_mean_factorial' else 2,steps=60))
    if args.command=='weight-diagnostic':
        from .calibration_weights import run
        from pathlib import Path
        if not args.config:p.error('weight-diagnostic requires --config')
        config=json.loads(Path(args.config).read_text(encoding='utf-8-sig'))
        return unit('weight_diagnostic',lambda path:run(path,config),dict(max_wall_seconds=180,weight_profiles=len(config.get('constraint_weights',[1,10,100,1000]))*len(config.get('step_caps',[None]))))
    if args.command=='guard-diagnostic':
        from .calibration_guard import run
        from pathlib import Path
        if not args.config:p.error('guard-diagnostic requires --config')
        config=json.loads(Path(args.config).read_text(encoding='utf-8-sig'))
        return unit('guard_diagnostic',lambda path:run(path,config),dict(max_wall_seconds=300,physical_energy_gradient_jacobian_passes=4))
    if args.command=='collision-diagnostic':
        from .calibration_collision_failure import run
        from pathlib import Path
        if not args.config:p.error('collision-diagnostic requires --config')
        config=json.loads(Path(args.config).read_text(encoding='utf-8-sig'))
        return unit('collision_diagnostic',lambda path:run(path,config),dict(max_wall_seconds=180,grids=2))
    if args.command=='momentum-diagnostic':
        from pathlib import Path
        from .calibration_momentum import recorded_input_sweep
        if not args.config:p.error('momentum-diagnostic requires --config')
        config=json.loads(Path(args.config).read_text(encoding='utf-8-sig'))
        return unit('recorded_momentum_sweep',lambda path:recorded_input_sweep(path,config),dict(max_wall_seconds=600,factors=[1,2,4]))
    if args.command=='mass-rows-diagnostic':
        from .calibration_mass_rows import run
        from pathlib import Path
        if not args.config:p.error('mass-rows-diagnostic requires --config')
        config=json.loads(Path(args.config).read_text(encoding='utf-8-sig'))
        return unit('mass_rows_diagnostic',lambda path:run(path,config),dict(max_wall_seconds=300,whole_path_gradient_jacobian_passes=len(config.get('thread_counts',[1]*4)),reserved_cpus=max(config.get('thread_counts',[1]))))
    if args.command=='full-state-diagnostic':
        from pathlib import Path
        from .calibration_fullstate import run
        if not args.config:p.error('full-state-diagnostic requires --config')
        config=json.loads(Path(args.config).read_text(encoding='utf-8-sig'))
        budget=config.get('max_wall_seconds',300)
        if not isinstance(budget,(int,float)) or not 0<budget<=1800:p.error('full-state diagnostic budget must be in (0,1800] seconds')
        return unit('full_state_dynamics',lambda path:run(path,config),dict(max_wall_seconds=budget))
    if args.command=='plan':
        from pathlib import Path
        if not args.config or not args.task:p.error('plan requires --config and --task')
        # Discover only explicitly declared input files before installing the
        # sentinel. Algorithm imports and robot/task loading stay inside it.
        config=json.loads(Path(args.config).read_text(encoding='utf-8-sig'))
        submitted=json.loads(Path(args.task).read_text(encoding='utf-8-sig'))
        seeds=tuple(config['budget'].get('seeds',(13001,13002,13003,13004,13005)))
        def planning(path):
            from .architecture_guard import LegacySentinel
            from .orchestration import atomic_json
            allowed=[args.config,args.task,config['profile'],config['checkpoint'],Path(config['checkpoint']).with_suffix('.identity.json')]
            if 'snapshot_file' in submitted.get('initial',{}):allowed.append(Path(args.task).parent/submitted['initial']['snapshot_file'])
            sentinel=LegacySentinel(allowed_data_reads=allowed,output_directory=path)
            try:
                with sentinel:
                    from .problem_io import load_problem,planning_config
                    from .api import plan
                    loaded,budget=planning_config(args.config)
                    if loaded!=config:raise ValueError('PLANNING_CONFIG_CHANGED_DURING_SETUP')
                    robot,snapshot,task,problem=load_problem(args.task)
                    if problem!=submitted:raise ValueError('PUBLIC_TASK_CHANGED_DURING_SETUP')
                    atomic_json(path/'submitted_problem.json',problem)
                    problem['initial']={'snapshot_file':'initial_state.npz'}
                    atomic_json(path/'problem.json',problem)
                    profile=json.loads(Path(config['profile']).read_text(encoding='utf-8-sig'))
                    candidate,result=plan(robot,snapshot,task,config['checkpoint'],budget,profile,path,config.get('device','cpu'),config.get('disabled_guide_terms',()),config.get('guide_weights'),config.get('physics_backend','eager'),config.get('collision_aggregation','pair_mean'),config.get('joint_peak_penalty',False),config.get('task_objective','center_squared'),config.get('planning_margins'),config.get('compiled_full_step',False))
            except Exception as error:
                from .architecture_guard import ArchitectureViolation
                result_file=path/'planning_result.json'
                if isinstance(error,ArchitectureViolation) and result_file.exists():
                    rejected=json.loads(result_file.read_text(encoding='utf-8'))
                    rejected['model_selection_before_architecture_rejection']=dict(status=rejected['status'],selected_index=rejected.get('selected_index'))
                    rejected.update(status='ARCHITECTURE_REJECTED',selected_index=None,architecture_rejected=True,error=str(error))
                    atomic_json(result_file,rejected)
                elif not result_file.exists():
                    records=[]
                    for index,seed in enumerate(seeds):
                        record_file=path/f'candidate_{index:03d}'/'candidate.json'
                        record=json.loads(record_file.read_text()) if record_file.exists() else dict(index=index,seed=seed,sampled=False,validated=False,status='NOT_RUN_EXPERIMENT_FAILURE')
                        if record['status']=='STARTED':record.update(status='INCOMPLETE_EXPERIMENT_FAILURE')
                        records.append(record)
                    atomic_json(path/'planning_result.json',dict(status='EXPERIMENT_FAILED',requested_candidates=len(seeds),records=records,
                                sampled=sum(r['sampled'] for r in records),valid=0,selected_index=None,error=str(error)))
                raise
            finally:atomic_json(path/'architecture_sentinel.json',sentinel.report())
            return dict(passed=candidate is not None,**result)
        return unit('planning',planning,dict(max_wall_seconds=config['budget'].get('max_wall_seconds',1800)+60,
                    candidates=len(seeds),steps=config['budget'].get('sampling',{}).get('steps',60),reserved_cpus=config['budget'].get('cpu_threads',1)))
    if args.command=='continue-state':
        from .problem_io import export_continuation
        if not args.run or not args.task:p.error('continue-state requires --run execution-directory and --task next-public-task')
        index=-1 if args.unit is None else int(args.unit)
        return unit('continuation_state',lambda path:export_continuation(path,args.run,args.task,index),dict(max_wall_seconds=60))
    if args.command=='execute':
        from .execution_acceptance import execute_selected
        from pathlib import Path
        if not args.run:p.error('execute requires --run planning-directory')
        config=json.loads(Path(args.config).read_text(encoding='utf-8-sig')) if args.config else {}
        return unit('selected_execution',lambda path:execute_selected(path,args.run,config),dict(max_wall_seconds=900,physics_substeps=config.get('physics_substeps',1)))
    if args.command=='validate':
        from .artifact_checks import validate_saved,validate_execution_states,compare_tracking_prediction
        if not args.run:p.error('validate requires --run planning-directory; optional --unit candidate-index')
        if args.unit=='tracking-prediction':
            return unit('tracking_prediction_diagnostic',lambda path:compare_tracking_prediction(path,args.run),dict(max_wall_seconds=600,native_execution_steps=0))
        if args.unit=='actual-states':
            from pathlib import Path
            tracking_profile=json.loads(Path(args.config).read_text(encoding='utf-8-sig')) if args.config else None
            return unit('execution_state_audit',lambda path:validate_execution_states(path,args.run,tracking_profile),dict(max_wall_seconds=60,native_integration_steps=0))
        return unit('independent_validation',lambda path:validate_saved(path,args.run,args.unit),dict(max_wall_seconds=600))
    if args.command=='replay':
        from .artifact_checks import replay_saved
        if not args.run:p.error('replay requires --run execution-directory')
        return unit('independent_replay',lambda path:replay_saved(path,args.run),dict(max_wall_seconds=600))
    if args.command=='test':
        if (args.task_objective!='center_squared' or args.planning_margins or args.free_coordinate_schema!='physical_position_v1') and args.suite!='full-space-origin':p.error('experimental task objective/coordinates require full-space-origin calibration')
        from pathlib import Path
        margins=json.loads(Path(args.planning_margins).read_text(encoding='utf-8')) if args.planning_margins else None
        if args.samples is None:args.samples=3 if args.suite in ('gradients','energy','full-space','full-space-peak','full-space-bodyframe','full-space-joint-peak','full-space-batched-path','full-space-origin') else 32
        if args.samples<1:p.error('samples must be positive')
        if args.suite=='architecture':
            def software(path):
                r=subprocess.run([sys.executable,'-m','pytest','ffgd/tests','-q',f'--junitxml={path / "tests.xml"}'],cwd=ROOT,capture_output=True,text=True)
                (path/'output.txt').write_text(r.stdout+r.stderr,encoding='utf-8');print(r.stdout+r.stderr)
                return dict(passed=r.returncode==0,exit_code=r.returncode,scope='implemented software contracts only',ARCH_01_to_10='NOT_RUN')
            return unit('software',software,dict(max_wall_seconds=300))
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
        if args.suite=='full-space-origin':
            from .calibration_constraints import energy_gradients
            return unit('full_space_origin_gradients',lambda path:energy_gradients(path,args.samples,24,collision_aggregation='pair_mean_plus_max',bodyframe_backend=True,joint_peak_penalty=True,full_path_backend=True,boundary_origin='initial_position',task_objective=args.task_objective,planning_margins=margins,free_coordinate_schema=args.free_coordinate_schema),dict(max_wall_seconds=900,cases=args.samples,free_dim=323))
        if args.suite in ('full-space','full-space-peak','full-space-bodyframe','full-space-joint-peak','full-space-batched-path'):
            from .calibration_constraints import energy_gradients
            aggregation='pair_mean' if args.suite=='full-space' else 'pair_mean_plus_max'
            name={'full-space':'full_space_gradients','full-space-peak':'full_space_peak_gradients','full-space-bodyframe':'full_space_bodyframe_gradients','full-space-joint-peak':'full_space_joint_peak_gradients','full-space-batched-path':'full_space_batched_path_gradients'}[args.suite]
            return unit(name,lambda path:energy_gradients(path,args.samples,24,collision_aggregation=aggregation,bodyframe_backend=args.suite in ('full-space-bodyframe','full-space-joint-peak','full-space-batched-path'),joint_peak_penalty=args.suite in ('full-space-joint-peak','full-space-batched-path'),full_path_backend=args.suite=='full-space-batched-path'),dict(max_wall_seconds=900,cases=args.samples,free_dim=323))
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
