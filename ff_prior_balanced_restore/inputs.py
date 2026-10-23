"""Bind one new scene, the exact numerical initial836, and one saved323 guess."""
import ast
import copy
import numpy as np
import mujoco
from ff_prior.storage import read_json,file_digest,digest
from ff_prior.contracts import assert_frozen,frozen_task,AcceptanceConfig
from ff_prior.robot_adapter import RobotAdapter
from ff_prior.scenes import SceneSpec
from ff_prior.initial_state import InitialState,STATE
from ff_prior.geometry import CollisionBackend
from ff_prior_loop.runtime import PlanningOriginSpec
from ff_prior_pilot_train.storage import atomic_json,atomic_npz
from .protocol import ROOT,PACKAGE,OLD,P


def sources():
    pending=list(PACKAGE.rglob('*.py'))+list((ROOT/'ff_prior_balanced_tracking').rglob('*.py'))+list((ROOT/'ff_prior_balanced_evidence').rglob('*.py'));found=set()
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
                for n in node.names:add(n.name)
            elif isinstance(node,ast.ImportFrom):
                prefix=list(pkg[:len(pkg)-node.level+1]) if node.level else []
                module='.'.join(prefix+([node.module] if node.module else []));add(module)
                for n in node.names:
                    if n.name!='*':add(module+'.'+n.name)
    found.update([PACKAGE/'job.ps1',PACKAGE/'docs/EXPERIMENT_PLAN.md',ROOT/'ff_prior_balanced_tracking/job.ps1',ROOT/'ff_prior_balanced_tracking/docs/EXPERIMENT_PLAN.md',ROOT/'ff_prior_balanced_evidence/README.md'])
    return {p.relative_to(ROOT).as_posix():file_digest(p) for p in sorted(found)}


def runtime(entry,values):
    robot=RobotAdapter(scene=SceneSpec.from_dict(entry['scene']))
    robot.spec=PlanningOriginSpec(robot.spec,entry['task']['condition']['initial_features'])
    assert mujoco.mj_stateSize(robot.model,STATE)==836==len(values)
    initial=InitialState(np.asarray(values).copy(),robot.identity()['scene_model_hash'])
    return robot,initial,CollisionBackend(robot)


def freeze():
    assert not (PACKAGE/'evidence/selection.json').exists(),'FREEZE_ONCE_ONLY'
    c=read_json(ROOT/'ff_prior_fullchain/control.json')
    assert c['next_phase']=='D3_AUTHORIZED_BALANCED_STATIC_IMPLEMENT_AND_FREEZE'
    auth=read_json(PACKAGE/'evidence/authorization.json');assert auth['physical_trial_authorized']
    closed=read_json(ROOT/'ff_prior_fullchain/evidence/horizon_restore_closeout.json')
    assert closed['closed'] and closed['commit']
    old=read_json(ROOT/'ff_prior_horizon_restore/evidence/selection.json')
    prior=old['entries'][0];assert_frozen(prior['task'])
    layout=read_json(ROOT/'ff_prior_balanced_scene/evidence/layout.json')
    report=read_json(OLD/'units/u00/validated/validation.json')
    assert report['eligible']
    with np.load(OLD/'units/u00/execution/initial_state.npz') as a:values=a['values'].copy()
    entry=dict(sample_id=P['task_id'],split='APP_DEV',scene=layout['scene'],task=copy.deepcopy(prior['task']))
    robot,initial,collision=runtime(entry,values)
    assert initial.model_hash==layout['new_scene_model_hash'] and initial.state_id==layout['new_initial_state_id']
    contract=dict(initial_state_id=initial.state_id,robot_hash=initial.model_hash,
        pair_policy_hash=collision.policy_hash,acceptance=prior['task']['contract']['acceptance'])
    entry['task']=frozen_task(prior['task']['condition'],contract)
    assert entry['task']['task_hash']!=prior['task']['task_hash']
    assert len(entry['scene']['objects'])==7
    z=np.asarray(report['Z']);assert z.shape==(19,17) and np.isfinite(z).all()
    atomic_npz(PACKAGE/'evidence/initial_state.npz',values=values)
    atomic_npz(PACKAGE/'evidence/guess.npz',Z=z)
    unit=dict(id='u00',task_id=P['task_id'],method='historical_diffusion_full323_newscene_restoration',
        proposal_seed=old['units'][0]['proposal_seed'],seed=old['units'][0]['seed'],
        source_run='horizon_restore_001',source_reference_hash=report['reference_hash'],
        source_qualification='MODEL_VALID_IN_OLD_SCENE_ONLY',new_scene_qualification='UNQUALIFIED_GUESS',
        guess_file='ff_prior_balanced_restore/evidence/guess.npz',candidate_denominator=1,execution_denominator=1)
    selection=dict(entries=[entry],units=[unit],candidate_denominator=1,execution_denominator=1,
        before_new_physics=True,new_network_samples=0,original_numeric_initial_state_preserved=True,
        original_static_conditions_preserved=True,original_waypoints_sha256=layout['public_waypoints_sha256'],
        source_neural_lineage=old['units'][0],scope='one APP_DEV balanced2R5C composition; no comparison or TEST')
    paths={ROOT/p for p in read_json(ROOT/'ff_prior_horizon_restore/evidence/binding.json')['files']}
    paths.update([OLD/'units/u00/validated/validation.json',OLD/'units/u00/execution/initial_state.npz',
        OLD/'units/u00/execution/report.json',OLD/'run.json',
        ROOT/'ff_prior_horizon_restore/evidence/selection.json',ROOT/'ff_prior_fullchain/evidence/horizon_restore_closeout.json',
        ROOT/'ff_prior_balanced_scene/evidence/layout.json',ROOT/'ff_prior_balanced_scene/evidence/budget_proposal.json',
        ROOT/'artifacts/v7_1/dynamic_handoff_001/u00/task.json',
        PACKAGE/'evidence/authorization.json',PACKAGE/'evidence/budget_derivation.json',
        PACKAGE/'evidence/initial_state.npz',PACKAGE/'evidence/guess.npz'])
    assert all(robot.calls[k]==0 for k in ('physics_steps','coupled','distance','forward','fk','mass_matrix'))
    atomic_json(PACKAGE/'evidence/binding.json',dict(files={p.relative_to(ROOT).as_posix():file_digest(p) for p in sorted(paths)}))
    atomic_json(PACKAGE/'evidence/selection.json',selection)
    return dict(inputs=len(paths),task_hash=entry['task']['task_hash'],new_scene_model_hash=initial.model_hash,
        new_initial_state_id=initial.state_id,new_physics=0,new_neural=0)


def identity():
    binding=read_json(PACKAGE/'evidence/binding.json')
    for p,h in binding['files'].items():
        if file_digest(ROOT/p)!=h:raise ValueError('FROZEN_INPUT_CHANGED: '+p)
    selection=read_json(PACKAGE/'evidence/selection.json')
    assert len(selection['units'])==1 and selection['candidate_denominator']==1
    for e in selection['entries']:assert_frozen(e['task'])
    return dict(protocol=P,source_hashes=sources(),selection=selection,binding=binding)


if __name__=='__main__':
    import sys,json
    print(json.dumps(freeze() if sys.argv[-1]=='freeze' else identity(),ensure_ascii=True))
