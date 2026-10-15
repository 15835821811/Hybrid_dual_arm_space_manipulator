"""Five bounded smooth proposal families; no inverse planner or rollout."""
import numpy as np
from .spline import Reference

FAMILIES = ("smooth_baseline", "correlated_low_frequency_perturbation", "continuum_leads",
            "rigid_arm_leads", "joint_group_motion_allocation")


def propose(robot, family_id, variant, master_seed):
    seed = int(master_seed) + int(family_id) * 1009 + int(variant) * 31
    rng = np.random.default_rng(seed)
    q0 = robot.spec.planner_zero
    # Scale all perturbations by audited domain; never wrap PCC components.
    amplitude = rng.uniform(0.015, 0.045) * (robot.upper - robot.lower)
    terminal = q0 + rng.uniform(-1, 1, 17) * amplitude
    s = np.linspace(0, 1, 20)[1:-1]
    h = s*s*(3-2*s)
    phase = np.repeat(h[:, None], 17, axis=1)
    if variant in (2, 3):
        lead = 1 - (1 - h)**1.5
        lag = h**1.5
        phase[:, :10], phase[:, 10:] = (lead[:,None], lag[:,None]) if variant == 2 else (lag[:,None], lead[:,None])
    internal = q0 + phase * (terminal - q0)
    if variant == 1:
        for frequency in (1, 2):
            internal += np.sin(np.pi*frequency*s)[:, None] * rng.normal(0, 0.10, (1,17)) * amplitude
    if variant == 4:
        allocation = rng.uniform(-0.4, 0.4, 17)
        internal += np.sin(np.pi*s)[:,None] * allocation * amplitude
    z = np.vstack([internal, terminal])
    for duration in (10., 15., 20., 25., 29.5):
        ref = Reference(z, q0, duration)
        if motion_valid(robot, ref):
            return ref, {"seed": seed, "proposal_type": FAMILIES[variant], "q_terminal_seed": terminal}
    return None, {"seed": seed, "proposal_type": FAMILIES[variant], "reason": "DURATION_SET_EXHAUSTED"}


def motion_valid(robot, ref):
    for order, limit in ((1, robot.spec.planner_velocity_limits), (2, robot.spec.planner_acceleration_limits)):
        lo, hi = ref.bounds(order)
        if np.any(np.maximum(abs(lo), abs(hi)) > limit + 1e-12):
            return False
    return True


def quick_gates(robot, ref):
    lo, hi = ref.bounds()
    low = ref.control @ robot.B.T
    limited = robot.model.jnt_limited[robot.joints].astype(bool)
    ranges = robot.model.jnt_range[robot.joints]
    boundary = np.r_[ref.evaluate(0,1),ref.evaluate(0,2),ref.evaluate(ref.duration,1),ref.evaluate(ref.duration,2)]
    return {"finite": bool(np.isfinite(ref.control).all()),
            "domain": bool(np.all(lo >= robot.lower) and np.all(hi <= robot.upper)),
            "native_range": bool(np.all(low[:,limited] >= ranges[limited,0]) and np.all(low[:,limited] <= ranges[limited,1])),
            "boundary": bool(np.max(abs(boundary)) < 1e-10 and np.array_equal(ref.control[0], robot.spec.planner_zero)),
            "motion": motion_valid(robot, ref), "time_hold": bool(ref.duration + ref.hold <= 30 and ref.hold == .5)}
