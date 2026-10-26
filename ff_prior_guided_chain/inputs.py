"""Bind original public inputs and exact sources, never old successful trajectories."""
import ast,copy,hashlib,subprocess,zipfile,importlib.metadata,time
import numpy as np
import mujoco
from ff_prior.storage import read_json,file_digest,digest
from ff_prior_pilot_train.storage import atomic_json
from ff_prior.robot_adapter import RobotAdapter
from ff_prior.scenes import SceneSpec
from ff_prior.initial_state import InitialState,STATE
from ff_prior_loop.runtime import PlanningOriginSpec
from ff_prior.geometry import CollisionBackend
from ff_prior.contracts import frozen_task
from .protocol import ROOT,PACKAGE,P


def runtime():
    old=read_json(ROOT/'ff_prior_balanced_restore/evidence/selection.json')['entries'][0]
    layout=read_json(ROOT/'ff_prior_balanced_scene/evidence/layout.json')
    robot=RobotAdapter(scene=SceneSpec.from_dict(layout['scene']))
    robot.model.opt.integrator=mujoco.mjtIntegrator.mjINT_RK4
    robot.spec=PlanningOriginSpec(robot.spec,old['task']['condition']['initial_features'])
    path=ROOT/P['initial'];assert file_digest(path)==P['initial_sha256']
    with np.load(path) as a:values=a['values'].copy()
    assert mujoco.mj_stateSize(robot.model,STATE)==len(values)==836
    initial=InitialState(values,robot.identity()['scene_model_hash'])
    collision=CollisionBackend(robot)
    contract=dict(initial_state_id=initial.state_id,robot_hash=initial.model_hash,pair_policy_hash=collision.policy_hash,
        acceptance=old['task']['contract']['acceptance'])
    entry=dict(scene=layout['scene'],task=frozen_task(old['task']['condition'],contract))
    return robot,initial,entry


def sources():
    pending=list(PACKAGE.rglob('*.py'));found=set()
    def add(module):
        parts=module.split('.')
        for i in range(1,len(parts)+1):
            p=ROOT.joinpath(*parts[:i])
            for path in (p.with_suffix('.py'),p/'__init__.py'):
                if path.is_file() and path not in found:pending.append(path)
    while pending:
        p=pending.pop()
        if p in found:continue
        found.add(p);pkg=p.relative_to(ROOT).with_suffix('').parts[:-1]
        for node in ast.walk(ast.parse(p.read_text(encoding='utf-8-sig'))):
            if isinstance(node,ast.Import):
                for item in node.names:add(item.name)
            elif isinstance(node,ast.ImportFrom):
                prefix=list(pkg[:len(pkg)-node.level+1]) if node.level else []
                module='.'.join(prefix+([node.module] if node.module else []));add(module)
                for item in node.names:
                    if item.name!='*':add(module+'.'+item.name)
    found.update([PACKAGE/'docs/IMPLEMENTATION_AND_FREEZE.md',PACKAGE/'job.ps1'])
    return {p.relative_to(ROOT).as_posix():file_digest(p) for p in sorted(found)}


def input_files():
    paths=[ROOT/P['initial'],ROOT/'ff_prior_balanced_restore/evidence/selection.json',
        ROOT/'ff_prior_balanced_scene/evidence/layout.json',ROOT/'ff_prior_balanced_brake/evidence/contract.json',
        ROOT/'ff_prior_satellite_origin/evidence/contract.json',ROOT/P['temporal_codec'],
        ROOT/P['temporal_model']/'model.pt',PACKAGE/'evidence/authorization.json',PACKAGE/'evidence/budget.json',
        ROOT/'ff_prior_guided_chain_plan/evidence/implementation_spec.json']
    paths.extend(ROOT/P['static_model']/name for name in ('model.pt','codec.json','training.json'))
    paths.extend(p for p in (ROOT/'dual_arm_space_robot_2026').rglob('*') if p.is_file() and p.suffix.lower() in ('.urdf','.stl','.obj','.dae','.xml'))
    return {p.relative_to(ROOT).as_posix():file_digest(p) for p in sorted(set(paths))}


def versions():
    return {n:importlib.metadata.version(n) for n in ('numpy','scipy','torch','mujoco','clarabel','matplotlib')}


def freeze():
    path=PACKAGE/'evidence/frozen.json';assert not path.exists(),'FREEZE_ONCE'
    auth=read_json(PACKAGE/'evidence/authorization.json')
    assert auth['physical_trial_authorized'] and time.time()-auth['started_at']<P['implementation_s'],'IMPLEMENTATION_RESERVE_EXHAUSTED'
    tests=read_json(PACKAGE/'evidence/software_tests.json');assert tests['status']=='PASS'
    src=sources();assert src==tests['source_hashes'],'SOURCE_CHANGED_AFTER_SOFTWARE_CHECK'
    cost=read_json(PACKAGE/'evidence/cost.json');assert cost['within_authorized_stop_envelope']
    value=dict(protocol=P,source_hashes=src,files=input_files(),versions=versions(),
        git_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        software_sha256=file_digest(PACKAGE/'evidence/software_tests.json'),cost_sha256=file_digest(PACKAGE/'evidence/cost.json'),
        actual_model_hash=tests['model_hash'],original_numeric_initial_sha256=P['initial_sha256'])
    atomic_json(path,value)
    with zipfile.ZipFile(PACKAGE/'evidence/frozen_source.zip','w',zipfile.ZIP_DEFLATED) as z:
        for f in src:z.write(ROOT/f,f)
    return dict(status='FROZEN',source_files=len(src),input_files=len(value['files']),identity_hash=digest(value))


def identity():
    v=read_json(PACKAGE/'evidence/frozen.json')
    assert v['protocol']==P and v['source_hashes']==sources(),'FROZEN_SOURCE_CHANGED'
    assert v['versions']==versions(),'FROZEN_DEPENDENCY_CHANGED'
    for f,h in v['files'].items():
        if file_digest(ROOT/f)!=h:raise ValueError('FROZEN_INPUT_CHANGED:'+f)
    assert file_digest(PACKAGE/'evidence/software_tests.json')==v['software_sha256']
    assert file_digest(PACKAGE/'evidence/cost.json')==v['cost_sha256']
    return v
