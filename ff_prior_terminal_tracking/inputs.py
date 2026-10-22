"""Immutable public task plus actual predecessor terminal state binding."""
import ast
import numpy as np
from ff_prior.storage import read_json,file_digest,digest
from ff_prior_pilot_train.storage import atomic_json,atomic_npz
from .protocol import ROOT,PACKAGE,OUT,SOURCE,DYNAMIC,P

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
    assert not (PACKAGE/'evidence/contract.json').exists(),'Freeze only once'
    report=read_json(SOURCE/'units/u00/execution/report.json')
    assert report['level']=='EXECUTION_VERIFIED' and all(report['gates'].values())
    entry=read_json(ROOT/'ff_prior_diffusion_demo/evidence/selection.json')['entries'][0]
    native=SOURCE/'units/u00/execution/native_trace.npz';a=np.load(native)
    terminal={k:a[k][-1].copy() for k in ('integration','qpos','qvel','time','reference_q','reference_qd')}
    from ff_prior.robot_adapter import RobotAdapter
    from ff_prior.scenes import SceneSpec
    from ff_prior.initial_state import InitialState
    from tracking_diffusion.environment import free_slices
    import mujoco
    robot=RobotAdapter(scene=SceneSpec.from_dict(entry['scene']))
    assert robot.identity()['scene_model_hash']==report['scene_model_hash']
    data=InitialState(terminal['integration'],report['scene_model_hash']).reset(robot)
    target=robot.id(mujoco.mjtObj.mjOBJ_BODY,'target_satellite')
    rt=data.xmat[target].reshape(3,3);pt=data.xpos[target]
    _,tv=free_slices(robot.model,robot.spec.target_free_joint_name)
    old=read_json(DYNAMIC);w=dict(old['waypoints']);goals=np.asarray(entry['task']['condition']['goals'])
    # Bridge starts at the commanded static goal, never the measured terminal error.
    w['initial_position_w']=goals[0,:3,3].tolist()
    contract=dict(schema='actual_terminal_seven_waypoint_composition_v1',split='APPLICATION_DEV_COMPOSITION',
        predecessor_task_hash=entry['task']['task_hash'],scene=entry['scene'],scene_model_hash=report['scene_model_hash'],
        source_native_sha256=file_digest(native),source_terminal_index=len(a['time'])-1,
        absolute_start_time=float(terminal['time']),static_goals=goals,waypoints=w,
        original_waypoints_sha256=digest(old['waypoints']['waypoint_points_m']),
        grasp_local=rt.T@(goals[1,:3,3]-pt),grasp_rotation_local=rt.T@goals[1,:3,:3],
        target_velocity=data.qvel[tv].copy(),target_velocity_changed=False,moving_satellite_claim=False,
        original_dynamic_DEV_claim=False,controller='existing_v7_domain_whole_body_feedback',
        network_role='inherited_D1_diffusion_generated_static_path; dynamic_controller_not_a_new_diffusion_model')
    contract['contract_hash']=digest(contract)
    atomic_npz(PACKAGE/'evidence/terminal.npz',**terminal)
    atomic_json(PACKAGE/'evidence/contract.json',contract)
    inputs={**read_json(ROOT/'ff_prior_diffusion_demo/evidence/binding.json')['files']}
    for f in (native,SOURCE/'units/u00/execution/report.json',SOURCE/'run.json',DYNAMIC,
              PACKAGE/'evidence/terminal.npz',PACKAGE/'evidence/contract.json',
              ROOT/'ff_prior_fullchain/evidence/diffusion_demo_closeout.json'):
        inputs[f.relative_to(ROOT).as_posix()]=file_digest(f)
    atomic_json(PACKAGE/'evidence/binding.json',{'files':inputs})
    return dict(inputs=len(inputs),contract_hash=contract['contract_hash'],extra_physics_steps=0)

def identity():
    binding=read_json(PACKAGE/'evidence/binding.json')
    for path,h in binding['files'].items():
        if file_digest(ROOT/path)!=h:raise ValueError('FROZEN_INPUT_CHANGED: '+path)
    c=read_json(PACKAGE/'evidence/contract.json');raw=dict(c);given=raw.pop('contract_hash')
    assert digest(raw)==given
    return dict(protocol=P,source_hashes=sources(),binding=binding,contract=c)

if __name__=='__main__':
    import sys,json
    print(json.dumps(freeze() if sys.argv[-1]=='freeze' else identity(),ensure_ascii=True))

