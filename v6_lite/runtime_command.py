"""C.1 source-bound command validity. No backup or latency robustness claim."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass

import mujoco
import numpy as np


def state_id(model, data):
    state = np.empty(mujoco.mj_stateSize(model, mujoco.mjtState.mjSTATE_INTEGRATION))
    mujoco.mj_getState(model, data, state, mujoco.mjtState.mjSTATE_INTEGRATION)
    return hashlib.sha256(state.tobytes()).hexdigest()


def partition_id(partition):
    return hashlib.sha256(json.dumps(
        [leaf.interval_id for leaf in partition.leaves], separators=(",", ":")
    ).encode()).hexdigest()


def model_id(model, source_hash):
    """Bind source identity and mutable compiled dynamics/geometry settings."""
    digest = hashlib.sha256(source_hash.encode())
    for name in ("body_mass", "body_inertia", "body_pos", "body_quat", "body_ipos", "body_iquat",
                 "dof_damping", "dof_armature", "dof_frictionloss", "jnt_stiffness",
                 "jnt_axis", "jnt_pos", "jnt_range", "geom_pos", "geom_quat", "geom_size",
                 "geom_type", "geom_contype", "geom_conaffinity", "actuator_gear",
                 "actuator_gainprm", "actuator_biasprm", "actuator_dynprm", "actuator_ctrlrange"):
        digest.update(getattr(model, name).tobytes())
    digest.update(np.asarray([model.opt.timestep, int(model.opt.integrator),
        int(model.opt.solver), int(model.opt.disableflags), int(model.opt.enableflags),
        model.opt.iterations, model.opt.tolerance, model.opt.density, model.opt.viscosity]).tobytes())
    digest.update(model.opt.gravity.tobytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class CommandCertificate:
    command_id: int
    source_state_id: str
    source_model_hash: str
    source_partition_id: str
    command_sha256: str
    state_acquisition_time: float
    target_acquisition_time: float
    planned_execution_start: float
    planned_execution_start_simulation_s: float
    solve_finished_time: float
    validation_finished_time: float
    valid_until: float
    maximum_supported_state_age: float
    scope: str = "exact_source_start_then_ten_declared_model_microstates"
    clock: str = "monotonic_seconds; simulation time is a separate coordinate"
    validity_clock: str = "wall_monotonic"
    simulation_valid_until: float | None = None

    @classmethod
    def issue(cls, *, command_id, source_state_id, source_model_hash,
              source_partition_id, command, acquired, source_simulation_s,
              solve_finished, validated, period_s=.020, policy="wall_deadline"):
        if policy not in ("wall_deadline", "offline_replay", "research_simulation"):
            raise ValueError("unsupported command clock policy")
        research = policy == "research_simulation"
        return cls(command_id, source_state_id, source_model_hash,
                   source_partition_id, hashlib.sha256(command.tobytes()).hexdigest(),
                   acquired, acquired, acquired, source_simulation_s,
                   solve_finished, validated, acquired + period_s, period_s,
                   clock=("simulation seconds for admission; monotonic seconds for compute diagnostics"
                          if research else "monotonic_seconds; simulation time is a separate coordinate"),
                   validity_clock="simulation_time" if research else "wall_monotonic",
                   simulation_valid_until=source_simulation_s+period_s if research else None)

    def to_dict(self):
        return asdict(self)


class DispatchGate:
    def __init__(self):
        self.last_command_id = -1
        self.next_substep = 0

    def check(self, certificate, *, now, model_hash, partition_hash, command,
              observed_state_id, observed_simulation_s, substep=0,
              microstate_matches=True, policy="wall_deadline"):
        c = certificate
        clocks = (now, c.state_acquisition_time, c.target_acquisition_time,
                  c.planned_execution_start, c.solve_finished_time,
                  c.validation_finished_time, c.valid_until,
                  c.maximum_supported_state_age, c.planned_execution_start_simulation_s,
                  observed_simulation_s)
        reason = None
        if not all(math.isfinite(v) for v in clocks):
            reason = "NONFINITE_TIME"
        elif policy not in ("wall_deadline", "offline_replay", "research_simulation"):
            reason = "UNSUPPORTED_CLOCK_POLICY"
        elif c.validity_clock != ("simulation_time" if policy == "research_simulation" else "wall_monotonic"):
            reason = "CERTIFICATE_CLOCK_POLICY_MISMATCH"
        elif not (c.state_acquisition_time <= c.solve_finished_time
                  <= c.validation_finished_time <= now):
            reason = "INVALID_TIME_ORDER"
        elif (c.valid_until != c.state_acquisition_time + c.maximum_supported_state_age
              or c.planned_execution_start != c.state_acquisition_time
              or c.maximum_supported_state_age != .020):
            reason = "INVALID_VALIDITY_RANGE"
        elif model_hash != c.source_model_hash:
            reason = "MODEL_ID_MISMATCH"
        elif partition_hash != c.source_partition_id:
            reason = "PARTITION_ID_MISMATCH"
        elif hashlib.sha256(command.tobytes()).hexdigest() != c.command_sha256:
            reason = "COMMAND_PAYLOAD_MISMATCH"
        elif policy == "research_simulation" and (
                c.simulation_valid_until is None or not math.isfinite(c.simulation_valid_until)
                or abs(c.simulation_valid_until-c.planned_execution_start_simulation_s
                       - c.maximum_supported_state_age) > 1e-12):
            reason = "INVALID_SIMULATION_VALIDITY_RANGE"
        elif policy == "research_simulation" and observed_simulation_s >= c.simulation_valid_until:
            reason = "EXPIRED_COMMAND_SIMULATION_TIME"
        elif policy == "wall_deadline" and now > c.valid_until:
            reason = "EXPIRED_COMMAND_WALL_CLOCK"
        elif now < c.target_acquisition_time:
            reason = "FUTURE_TARGET_TIMESTAMP"
        elif policy == "wall_deadline" and now - c.target_acquisition_time > c.maximum_supported_state_age:
            reason = "STALE_TARGET_WALL_CLOCK"
        elif substep == 0 and c.command_id <= self.last_command_id:
            reason = "COMMAND_OUT_OF_ORDER"
        elif substep > 0 and (c.command_id != self.last_command_id or substep != self.next_substep):
            reason = "SERVO_SUBSTEP_OUT_OF_ORDER"
        elif substep not in range(10):
            reason = "VALIDATED_RAMP_EXHAUSTED"
        elif substep == 0 and observed_state_id != c.source_state_id:
            reason = "SOURCE_STATE_ID_MISMATCH"
        elif abs(observed_simulation_s - (c.planned_execution_start_simulation_s + .002 * substep)) > 1e-9:
            reason = "VALIDATED_START_TIME_MISMATCH"
        elif not microstate_matches:
            reason = "VALIDATED_START_STATE_MISMATCH"
        if reason is None:
            self.last_command_id = c.command_id
            self.next_substep = substep + 1
        return {"accepted": reason is None, "reason": reason,
                "policy": policy, "wall_clock_current": now <= c.valid_until,
                "validity_clock": c.validity_clock,
                "simulation_state_age_s": observed_simulation_s-c.planned_execution_start_simulation_s,
                "simulation_valid_until": c.simulation_valid_until,
                "wall_deployment_certified": False,
                "state_age_s": now - c.state_acquisition_time,
                "target_age_s": now - c.target_acquisition_time,
                "dispatch_time": now, "servo_substep": substep,
                "continuation_guaranteed_on_reject": False}
