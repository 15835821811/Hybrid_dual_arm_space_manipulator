import sys
import ast
from ff_prior.storage import read_json, file_digest, digest
from ff_prior_pilot_train.storage import atomic_json
from .protocol import ROOT, PACKAGE, OLD, MODEL, GATE, P

def sources():
    pending = list(PACKAGE.glob('*.py')); found = set()
    def add(module):
        parts=module.split('.')
        for i in range(1,len(parts)+1):
            p=ROOT.joinpath(*parts[:i])
            for path in (p.with_suffix('.py'),p/'__init__.py'):
                if path.is_file() and path not in found: pending.append(path)
    while pending:
        p=pending.pop()
        if p in found: continue
        found.add(p);pkg=p.relative_to(ROOT).with_suffix('').parts[:-1]
        for node in ast.walk(ast.parse(p.read_text(encoding='utf-8-sig'))):
            if isinstance(node,ast.Import):
                for name in node.names: add(name.name)
            elif isinstance(node,ast.ImportFrom):
                prefix=list(pkg[:len(pkg)-node.level+1]) if node.level else []
                module='.'.join(prefix+([node.module] if node.module else []));add(module)
                for name in node.names:
                    if name.name!='*':add(module+'.'+name.name)
    found.add(PACKAGE/'job.ps1')
    found.add(PACKAGE/'docs/EXPERIMENT_PLAN.md')
    return {p.relative_to(ROOT).as_posix():file_digest(p) for p in sorted(found)}

def freeze():
    import numpy as np
    from ff_prior.robot_adapter import RobotAdapter
    from .execution import Environment,task_identity
    from ff_prior_waypoint_domain.adapter import Environment as Original
    target=PACKAGE/'evidence/selection.json'
    if target.exists():raise ValueError('once-only freeze')
    receipt=ROOT/'ff_prior_fullchain/evidence/dynamic_margin_closeout.json'
    gate=read_json(receipt)
    if not (gate.get('closed') and gate.get('P2a_gate_passed')):raise ValueError('P2a review not closed')
    paths=[receipt,GATE/'summary.json',GATE/'run.json',
           ROOT/'paper/review-traces/experiment-result-to-claim/20261022_dynamic_margin_run01/verdict.json',
           MODEL/'model.pt',MODEL/'codec.json',OLD/'frozen_protocol.json',OLD/'binding.json']
    paths.extend(p for p in (ROOT/'dual_arm_space_robot_2026').rglob('*')
                 if p.is_file() and p.suffix in ('.urdf','.stl','.STL','.obj','.dae','.xml'))
    units=[];initial_checks=[]
    for scene,task_seed in P['task_seeds'].items():
        rr=RobotAdapter();env=Environment(rr,scene=='drifting',task_seed)
        original=Original(rr,scene=='drifting',task_seed)
        task=task_identity(env,scene,task_seed)
        old=OLD/scene/'task_controller/task.json';paths.append(old)
        if digest(task)!=digest(read_json(old)):raise ValueError('original task identity mismatch')
        np.testing.assert_array_equal(env.initial.values,original.initial.values)
        for key in env.history:np.testing.assert_array_equal(env.history[key],original.history[key])
        if len(env.integration_history)!=1 or len(env.collision.states)!=1:raise ValueError('initial recorder incomplete')
        initial_checks.append(dict(scene=scene,task_hash=digest(task),observer_initial_exact=True,
                                   physics_steps=0,complete_state_size=len(env.initial.values)))
        for group,seed in enumerate(P['proposal_seeds'][scene]):
            for method in ('task_controller','diffusion_null_prior'):
                units.append(dict(id=f'u{len(units):02}',scene=scene,task_seed=task_seed,
                    task_hash=digest(task),method=method,proposal_seed=seed,seed_group=group,
                    baseline_has_seed_input=False,split='APPLICATION_DEV'))
        del env,original,rr
    atomic_json(PACKAGE/'evidence/binding.json',dict(files={p.relative_to(ROOT).as_posix():file_digest(p) for p in paths}))
    atomic_json(target,dict(units=units,logical_units=12,distinct_tasks=2,unique_planned_executions=8,
        baseline_reuse_units=4,initial_observer_checks=initial_checks,before_any_new_sampling_or_execution=True))
    return dict(logical_units=12,unique_executions=8,binding_files=len(paths),initial_observer_checks=initial_checks)

def identity():
    binding=read_json(PACKAGE/'evidence/binding.json')
    for path,h in binding['files'].items():
        if file_digest(ROOT/path)!=h:raise ValueError('input changed: '+path)
    return dict(protocol=P,selection=read_json(PACKAGE/'evidence/selection.json'),
                binding=binding,source_hashes=sources())

if __name__=='__main__':
    import json
    print(json.dumps(freeze() if sys.argv[-1]=='freeze' else identity()))

