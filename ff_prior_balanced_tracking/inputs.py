"""Frozen public task plus the verified same-scene actual terminal binding."""
import ast
import numpy as np
from ff_prior.storage import read_json,file_digest,digest
from ff_prior_pilot_train.storage import atomic_json,atomic_npz
from .protocol import ROOT,PACKAGE,SOURCE,DYNAMIC,P


def sources():
    pending=list(PACKAGE.glob('*.py'));found=set()
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
                for name in node.names:add(name.name)
            elif isinstance(node,ast.ImportFrom):
                prefix=list(pkg[:len(pkg)-node.level+1]) if node.level else []
                module='.'.join(prefix+([node.module] if node.module else []));add(module)
                for name in node.names:
                    if name.name!='*':add(module+'.'+name.name)
    found.add(PACKAGE/'job.ps1');found.add(PACKAGE/'docs/EXPERIMENT_PLAN.md')
    return {p.relative_to(ROOT).as_posix():file_digest(p) for p in sorted(found)}


def freeze():
    assert not (PACKAGE/'evidence/contract.json').exists(),'FREEZE_ONLY_ONCE'
    control=read_json(ROOT/'ff_prior_fullchain/control.json')
    assert control['next_phase']=='D3_AUTHORIZED_BALANCED_TERMINAL_TRACKING_FREEZE','BALANCED_DYNAMIC_SCOPE_NOT_READY'
    authorization=ROOT/'ff_prior_balanced_restore/evidence/authorization.json'
    assert authorization.is_file(),'MISSING_NEW_FULLCHAIN_AUTHORIZATION'
    auth=read_json(authorization)
    assert auth['physical_trial_authorized'] and auth['dynamic_run']=='balanced_tracking_001'
    assert auth['candidate_denominator']==1 and not any(auth[k] for k in ('training','expansion','TEST','advantage_comparison'))
    # Both stages were implemented before the static source freeze. Terminal
    # binding must not become an opportunity to change either implementation.
    static_sources=read_json(SOURCE/'run.json')['identity']['source_hashes']
    for path,expected in static_sources.items():
        assert file_digest(ROOT/path)==expected,'STATIC_FROZEN_SOURCE_CHANGED: '+path
    dynamic_sources=sources()
    for path,expected in dynamic_sources.items():
        assert path in static_sources,'DYNAMIC_SOURCE_NOT_IN_STATIC_BUNDLE: '+path
        assert static_sources[path]==expected,'DYNAMIC_SOURCE_DIFFERS_FROM_STATIC_BUNDLE: '+path
    closed_path=ROOT/'ff_prior_fullchain/evidence/balanced_restore_closeout.json'
    closed=read_json(closed_path)
    assert closed['closed'] and closed['commit'] and closed['run']==SOURCE.relative_to(ROOT).as_posix()
    assert (closed['model_valid'],closed['executed'],closed['replay_verified'])==(1,1,1),'STATIC_CHAIN_INCOMPLETE'
    verdict=read_json(ROOT/closed['independent_review'])
    assert verdict['claims']['C1']['claim_supported']=='yes','STATIC_REVIEW_NOT_CONFIRMED'
    report_path=SOURCE/'units/u00/execution/report.json';report=read_json(report_path)
    assert report['level']=='EXECUTION_VERIFIED' and all(report['gates'].values())
    validation_path=SOURCE/'units/u00/validated/validation.json';validation=read_json(validation_path)
    assert validation['eligible'] and validation['validation']['validation_level']=='MODEL_VALID'
    assert all(validation['validation']['gates'].values()),'STATIC_MODEL_NOT_VALID'
    selection_path=ROOT/'ff_prior_balanced_restore/evidence/selection.json'
    entry=read_json(selection_path)['entries'][0]
    layout_path=ROOT/'ff_prior_balanced_scene/evidence/layout.json';layout=read_json(layout_path)
    assert digest(entry['scene'])==digest(layout['scene']),'SEVEN_SCENE_IDENTITY_MISMATCH'
    assert [o['object_id'] for o in entry['scene']['objects']]==['R1','R2','C1','C2','C3','C4','C5']
    native=SOURCE/'units/u00/execution/native_trace.npz'
    replay_path=SOURCE/'units/u00/execution/replay_trace.npz'
    with np.load(native,allow_pickle=False) as a,np.load(replay_path,allow_pickle=False) as replay:
        assert a['integration'].shape==(13751,836),'STATIC_NATIVE_LENGTH_MISMATCH'
        assert replay['integration'].shape==a['integration'].shape
        assert np.max(np.abs(a['integration']-replay['integration']))<=P['replay_atol'],'STATIC_FULL_REPLAY_MISMATCH'
        terminal={k:a[k][-1].copy() for k in ('integration','qpos','qvel','time','reference_q','reference_qd')}
        terminal_index=len(a['time'])-1
    assert digest(terminal['integration'])==closed['terminal_state_digest'],'CLOSEOUT_TERMINAL_ID_MISMATCH'
    from ff_prior.robot_adapter import RobotAdapter
    from ff_prior.scenes import SceneSpec
    from ff_prior.initial_state import InitialState
    from tracking_diffusion.environment import free_slices
    from ff_prior_dynamic_reference.execution import integration
    import mujoco
    robot=RobotAdapter(scene=SceneSpec.from_dict(entry['scene']))
    assert robot.identity()['scene_model_hash']==report['scene_model_hash']==layout['new_scene_model_hash']
    data=InitialState(terminal['integration'],report['scene_model_hash']).reset(robot)
    assert np.array_equal(integration(robot,data),terminal['integration'])
    assert np.array_equal(data.qpos,terminal['qpos']) and np.array_equal(data.qvel,terminal['qvel'])
    assert data.time==float(terminal['time']) and abs(data.time-27.5)<1e-8
    target=robot.id(mujoco.mjtObj.mjOBJ_BODY,'target_satellite')
    rt=data.xmat[target].reshape(3,3);pt=data.xpos[target]
    _,tv=free_slices(robot.model,robot.spec.target_free_joint_name)
    old=read_json(DYNAMIC);w=dict(old['waypoints']);goals=np.asarray(entry['task']['condition']['goals'])
    assert len(w['waypoint_points_m'])==7 and w['transition_duration_s']==4.5 and w['path_duration_s']==21.
    assert np.isclose(np.sum(w['segment_durations_s']),21.,atol=1e-12,rtol=0.)
    assert digest(w['waypoint_points_m'])==digest(layout['waypoints']['waypoint_points_m'])
    # Bridge starts at the commanded static goal, never a measured error relabel.
    w['initial_position_w']=goals[0,:3,3].tolist()
    contract=dict(schema='balanced_same_scene_actual_terminal_composition_v1',split='APP_DEV',
        predecessor_task_hash=entry['task']['task_hash'],scene=entry['scene'],scene_model_hash=report['scene_model_hash'],
        source_native_sha256=file_digest(native),source_replay_sha256=file_digest(replay_path),source_terminal_index=terminal_index,
        source_terminal_digest=digest(terminal['integration']),source_static_review=closed['independent_review'],
        static_source_bundle_hash=digest(static_sources),dynamic_source_bundle_hash=digest(dynamic_sources),
        both_stages_sources_unchanged_since_static_run=True,
        source_static_closeout=closed_path.relative_to(ROOT).as_posix(),absolute_start_time=float(terminal['time']),
        static_goals=goals,waypoints=w,original_waypoints_sha256=digest(old['waypoints']['waypoint_points_m']),
        original_waypoint_contract_sha256=digest(old['waypoints']),
        grasp_local=rt.T@(goals[1,:3,3]-pt),grasp_rotation_local=rt.T@goals[1,:3,:3],
        target_velocity=data.qvel[tv].copy(),target_velocity_changed=False,moving_satellite_claim=False,
        same_seven_obstacles_preserved=True,original_dynamic_DEV_claim=False,
        controller='existing_v7_domain_whole_body_feedback_with_complete_constraint_observation',
        reference=dict(profile=P['bypass_profile'],segments=P['bypass_segments'],axis_world=P['bypass_axis_world'],
            amplitude_m=P['bypass_amplitude_m'],waypoint_positions_rotations_arrival_times_unchanged=True),
        network_role='historical_Diffusion_initial_guess_plus_full323_static_restoration; dynamic_public_C2_reference_and_feedback',
        no_new_neural_sampling=True)
    contract['contract_hash']=digest(contract)
    atomic_npz(PACKAGE/'evidence/terminal.npz',**terminal)
    atomic_json(PACKAGE/'evidence/contract.json',contract)
    files=dict(read_json(ROOT/'ff_prior_balanced_restore/evidence/binding.json')['files'])
    for path in (native,replay_path,report_path,validation_path,SOURCE/'run.json',DYNAMIC,selection_path,layout_path,
                 PACKAGE/'evidence/terminal.npz',PACKAGE/'evidence/contract.json',closed_path,
                 ROOT/closed['independent_review'],authorization):
        files[path.relative_to(ROOT).as_posix()]=file_digest(path)
    atomic_json(PACKAGE/'evidence/binding.json',{'files':files})
    return dict(inputs=len(files),contract_hash=contract['contract_hash'],extra_coupled=0,extra_native_steps=0)


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
