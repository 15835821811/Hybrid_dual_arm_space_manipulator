"""Public-state initializer and generic workspace bows; no stored answer paths."""
import numpy as np
import torch
import mujoco
from ff_prior.storage import read_json, digest, file_digest
from ff_prior.guidance import public_goal_initializer
from ff_prior.spline import Reference
from ff_prior.proposal_generator import quick_gates
from ff_prior_control_transfer.codec import FullCodec
from ff_prior_control_transfer.model import TransferDenoiser, rollout
from .protocol import SOURCE, CODEC, P


def load_bundle():
    torch.set_num_threads(P['cpu_threads'])
    torch.manual_seed(P['detail_initialization_seed'])
    codec = FullCodec(read_json(CODEC))
    model = TransferDenoiser(codec).eval()
    assert model.detail[-1].weight.count_nonzero() == 0
    assert model.detail[-1].bias.count_nonzero() == 0
    assert codec.dim == 323 and codec.state['detail_fit_split'] == 'TRAIN'
    return model, codec


def transverse_frame(displacement):
    axis = np.asarray(displacement, dtype=float)
    axis = axis/max(np.linalg.norm(axis), 1e-12)
    if np.linalg.norm(axis) < .5:
        axis = np.array([0., 0., 1.])
    helper = np.eye(3)[np.argmin(abs(axis))]
    first = np.cross(axis, helper); first /= np.linalg.norm(first)
    return first, np.cross(axis, first)


def public_setup(entry, robot, initial):
    task = entry['task']
    assert np.array_equal(task['condition']['initial_features'], robot.spec.planner_zero)
    base = public_goal_initializer(task, robot, initial)
    data = initial.reset(robot); robot.kinematics(data, mass=True)
    M = robot.mass(data); a, b = robot.dof_ids, robot.base_dofs
    tangent = np.zeros((robot.model.nv, 17)); tangent[a] = robot.B
    tangent[b] = -np.linalg.solve(M[np.ix_(b,b)], M[np.ix_(b,a)]@robot.B)
    poses = robot.ee(data); J = []
    for i, body in enumerate((robot.tip_c, robot.tip_r)):
        jp, jr = np.zeros((3,robot.model.nv)), np.zeros((3,robot.model.nv))
        mujoco.mj_jac(robot.model, data, jp, jr, poses[i,:3,3], body)
        J.extend([jp@tangent, jr@tangent])
    J = np.vstack(J)
    displacement = np.asarray(task['condition']['goals'])[1,:3,3]-poses[1,:3,3]
    frame = transverse_frame(displacement)
    amplitude = max(float(o['size'][0]) for o in entry['scene']['objects'])+P['bow_margin_m']
    return base, J, frame, amplitude


def prepare(raw, base, task, robot):
    """Preserve a legal full path exactly; shrink only a proposed reference."""
    raw = np.asarray(raw, dtype=float)
    if raw.shape != (19,17) or not np.isfinite(raw).all():
        raise ValueError('INVALID_PROPOSAL_ARRAY')
    c = task['condition']
    for scale in (1., .5, .25, .125, .0625, .03125, 0.):
        z = raw.copy() if scale == 1 else base+scale*(raw-base)
        gates = quick_gates(robot, Reference(z, c['initial_features'], c['duration']))
        if all(gates.values()):
            return z, {'scale': scale, 'unchanged': np.array_equal(raw,z), 'gates': gates,
                       'policy': 'preserve_feasible_no_smoothing_or_clip'}
    return raw.copy(), {'scale': None, 'unchanged': True, 'gates': gates,
                        'policy': 'REJECTED_NO_FEASIBLE_BLEND'}


def make(method, slot, seed, task, setup, bundle):
    base, J, frame, amplitude = setup
    model, codec = bundle
    before = model.calls
    if method == 'diffusion_geometry':
        condition = torch.tensor(codec.condition(task['condition'], base), dtype=torch.float32)[None]
        with torch.no_grad():
            latent = rollout(model, condition, [seed])[0].numpy()
        raw = codec.decode(codec.coefficients(latent, codec.center(task['condition'], base)), base)
        assert model.calls-before == 16
    elif method == 'structured_geometry':
        raw = base.copy()
    else:
        raise ValueError(method)
    network_raw = raw.copy()
    direction = np.zeros(17)
    if slot:
        # Rotate the transverse directions by the frozen unit seed. No known
        # successful joint, sign, trajectory or solution is used to set the bow.
        angle = np.random.default_rng(seed-slot).uniform(0,2*np.pi)
        v = np.cos(angle)*frame[0]+np.sin(angle)*frame[1]
        w = -np.sin(angle)*frame[0]+np.cos(angle)*frame[1]
        desired = [v,-v,w,-w][slot-1]*amplitude
        error = np.r_[np.zeros(6), desired, np.zeros(3)]
        direction = np.linalg.solve(J.T@J+1e-4*np.eye(17), J.T@error)
        direction *= min(1., P['bow_control_cap_rad']/max(abs(direction).max(),1e-12))
        phase = np.r_[np.linspace(0,1,20)[1:-1],1.]
        bow = np.sin(np.pi*phase)**2; bow[-1] = 0.
        raw = raw+bow[:,None]*direction
    return network_raw, raw, {'method': method, 'slot': slot, 'sampling_seed': seed,
        'network_calls': model.calls-before, 'network_obstacle_conditioned': False,
        'network_raw_hash': digest(network_raw), 'proposal_hash': digest(raw),
        'public_base_hash': digest(base), 'bow_direction_rad': direction,
        'bow_amplitude_m': amplitude if slot else 0., 'private_solution_used': False,
        'source': 'old_aligned_low102_zero_detail_full323' if method=='diffusion_geometry' else 'public_generalized_jacobian',
        'fallback_to_other_method': False, 'new_training': False}
