from dataclasses import asdict
import os
import time
import numpy as np
import pytest
from ff_prior.storage import digest, atomic_json
from ff_prior.spline import Reference, decode
from ff_prior.coupled_predictor import predict_coupled, pose_errors
from ff_prior.contracts import frozen_task, AcceptanceConfig
from ff_prior.guidance import public_goal_initializer
from ff_prior.model_gates import evaluate_reference
from ff_prior_goal.projection import GoalProjector, smooth, CONTRACT
from ff_prior_goal.experiment import candidate, validation_targets, CHECKPOINT
from ff_prior_next import learning


def reachable_task(physical):
    robot, initial, collision = physical; profile = AcceptanceConfig()
    q0 = robot.spec.planner_zero
    s = np.r_[np.linspace(0, 1, 20)[1:-1], 1.]
    z = q0+(s*s*(3-2*s))[:, None]*np.linspace(-.04, .04, 17)[None]
    p = predict_coupled(initial, Reference(z, q0, 10.), robot, 256)
    c = {'initial_features': q0, 'goals': p['ee'][-1], 'duration': 10., 'pose_mask': np.ones((2, 6))}
    task = frozen_task(c, {'initial_state_id': initial.state_id, 'robot_hash': initial.model_hash,
        'pair_policy_hash': collision.policy_hash, 'acceptance': asdict(profile)})
    return task, profile


def test_curvature_regularization_preserves_all_endpoint_constraints():
    rng = np.random.default_rng(1); q0 = rng.normal(size=17); z = rng.normal(size=(19, 17))
    result = smooth(z, q0)
    assert np.array_equal(result[-1], z[-1])
    assert np.linalg.norm(np.diff(decode(result, q0), n=2, axis=0)) < np.linalg.norm(np.diff(decode(z, q0), n=2, axis=0))
    ref = Reference(result, q0, 10.)
    assert np.allclose(ref.evaluate(0), q0, atol=1e-14, rtol=0)
    assert np.allclose(ref.evaluate(10.), z[-1], atol=1e-14, rtol=0)
    assert np.max(abs(ref.evaluate([0., 10.], 1))) < 1e-12
    assert np.max(abs(ref.evaluate([0., 10.], 2))) < 1e-12


def test_real_full_path_goal_projection_and_model_validation(physical):
    robot, initial, collision = physical; task, profile = reachable_task(physical)
    before = digest(task); base = public_goal_initializer(task, robot, initial)
    projector = GoalProjector(task, robot, initial, collision, profile, time.time()+60)
    z = base+.01*np.sin(np.linspace(0, np.pi, 19))[:, None]*np.ones((1, 17))
    for i in (14, 15):
        z, info = projector(z, i)
        assert info['coupled_calls'] <= 1+4*(34+4)
    p, gate = evaluate_reference(robot, initial, Reference(z, task['condition']['initial_features'], 10.), collision, profile)
    pos, rot = pose_errors(p['ee'][-1], task['condition']['goals'])
    assert max(pos) < .005 and max(rot) < np.deg2rad(1)
    assert gate['validation_level'] == 'MODEL_VALID' and digest(task) == before


def test_unreachable_goal_does_not_become_success(physical):
    robot, initial, collision = physical; task, profile = reachable_task(physical)
    c = dict(task['condition']); c['goals'] = np.array(c['goals']); c['goals'][:, :3, 3] += 10
    task = frozen_task(c, task['contract']); before = digest(task)
    projector = GoalProjector(task, robot, initial, collision, profile, time.time()+60)
    z, info = projector(np.repeat(robot.spec.planner_zero[None], 19, axis=0), 15)
    assert max(info['position_error_m']) > .005 and digest(task) == before
    assert len(info['history']) <= 4
    with pytest.raises(ValueError): projector(np.full((19, 17), np.nan), 15)


def test_real_checkpoint_denoising_preserves_budget_and_goal(physical):
    robot, initial, collision = physical; task, profile = reachable_task(physical)
    bundle = learning.load(CHECKPOINT, initial.model_hash, initial.state_id)
    base = public_goal_initializer(task, robot, initial)
    result = candidate('diffusion_projected', 0, task, base, 20261028, bundle,
                       robot, initial, collision, profile, time.time()+60)
    assert result['sampling']['actual_network_calls'] == 16
    assert [g['iteration'] for g in result['sampling']['guidance']] == [14, 15]
    assert all(s['decomposition_residual_after'] < 1e-10 for s in result['sampling']['steps'])
    assert result['eligible'] and result['task_hash'] == task['task_hash']


def test_val_target_selection_is_stable_and_excludes_training_dev():
    rows = [({'sample_id': str(i), 'family_id': str(i//2), 'task': {'id': i}}, 'VAL') for i in range(12)]
    rows += [({'sample_id': 'forbidden', 'family_id': 'no', 'task': {}}, 'DEV')]
    assert validation_targets(rows) == validation_targets(list(reversed(rows)))
    assert len({r['family_id'] for r in validation_targets(rows)}) == 6


def test_detached_job_duplicate_guard(tmp_path):
    from ff_prior_goal.job import Monitor, status, launch
    atomic_json(tmp_path/'run.json', {'fixture': True})
    with Monitor(tmp_path) as monitor:
        assert status(tmp_path)['owner_alive']
        with pytest.raises(RuntimeError, match='alive'): launch(tmp_path/'source', tmp_path, True)
        monitor.finish({'status': 'FINISHED'})
    assert not (tmp_path/'launch.guard').exists()
