"""C.1.1 immutable packets and bounded fixed-boundary handoff.

All clocks are supplied by the caller. Virtual-clock tests use the same gate
as the wall executor. Supported states retain the existing 1e-9 tolerance.
There is no fallback control and an exhausted predecessor cannot continue.
"""
from dataclasses import dataclass
import hashlib
import math
import numpy as np

PERIOD = .020
SERVO_PERIOD = .002
STATE_TOL = 1e-9


def frozen_array(value):
    # bytes backing prevents even setflags(write=True) from mutating a packet.
    a = np.ascontiguousarray(value, dtype=np.float64)
    return np.frombuffer(a.tobytes(), dtype=np.float64).reshape(a.shape)


def array_id(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def payload_id(states, torques, endpoint, reference_end):
    digest = hashlib.sha256()
    for value in (states, torques, endpoint, reference_end):
        digest.update(np.ascontiguousarray(value, dtype=np.float64).tobytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class HandoffCertificate:
    command_id: int
    predecessor_command_id: int
    source_state_id: str
    source_acquisition_time: float
    planning_release: float
    solve_started: float
    solve_finished: float
    validation_finished: float
    publish_deadline: float
    execution_start: float
    execution_end: float
    execution_start_simulation_s: float
    model_hash: str
    controller_config_hash: str
    partition_id: str
    predicted_start_state_id: str
    predecessor_prediction_id: str
    payload_sha256: str
    start_admission_condition: str = "full_integration_state_linf_le_1e-9"
    scope: str = "declared_model_ten_microsteps_including_moving_target"


@dataclass(frozen=True)
class CommandPacket:
    certificate: HandoffCertificate
    integration_states: np.ndarray
    torques: np.ndarray
    endpoint_velocity: np.ndarray
    reference_end: np.ndarray

    @classmethod
    def create(cls, certificate, states, torques, endpoint, reference_end):
        if (np.asarray(states).ndim != 2 or len(states) != 11
                or np.asarray(torques).shape != (10, 67)
                or np.asarray(endpoint).shape != (17,)
                or np.asarray(reference_end).shape != (17,)):
            raise ValueError("a packet must contain exactly ten validated updates")
        if not all(np.all(np.isfinite(x)) for x in (states, torques, endpoint, reference_end)):
            raise ValueError("nonfinite packet payload")
        return cls(certificate, *(frozen_array(x) for x in
                   (states, torques, endpoint, reference_end)))

    def immutable_after_transport(self):
        return self.create(self.certificate, self.integration_states, self.torques,
                           self.endpoint_velocity, self.reference_end)


class HandoffBuffer:
    """Single executor owns both slots, so publication/switch are atomic.

    A planner uses a one-request/one-result transport. No asynchronous writer
    mutates these slots. Rejected publication leaves any valid active segment
    usable only until its existing end. No candidate queue is maintained.
    """
    def __init__(self, model_hash, config_hash):
        self.model_hash = model_hash
        self.config_hash = config_hash
        self.active = self.pending = None
        self.last_id = -1
        self.next_substep = 0
        self.pending_published_at = None

    def publish(self, packet, now, *, source_state_id, partition_id):
        c = packet.certificate
        reason = None
        times = (now, c.source_acquisition_time, c.planning_release, c.solve_started,
                 c.solve_finished, c.validation_finished, c.publish_deadline,
                 c.execution_start, c.execution_end)
        predecessor = self.active.certificate.command_id if self.active else -1
        if not all(math.isfinite(x) for x in times):
            reason = "NONFINITE_TIME"
        elif not (c.source_acquisition_time <= c.planning_release <= c.solve_started
                  <= c.solve_finished <= c.validation_finished <= now
                  <= c.publish_deadline <= c.execution_start):
            reason = "LATE_OR_INVALID_PUBLICATION"
        elif abs(c.execution_end - c.execution_start - PERIOD) > 1e-9:
            reason = "INVALID_EXECUTION_COVERAGE"
        elif c.command_id != self.last_id + 1 or self.pending is not None:
            reason = "DUPLICATE_OR_OUT_OF_ORDER"
        elif c.predecessor_command_id != predecessor:
            reason = "PREDECESSOR_MISMATCH"
        elif c.model_hash != self.model_hash:
            reason = "MODEL_MISMATCH"
        elif c.controller_config_hash != self.config_hash:
            reason = "CONFIG_MISMATCH"
        elif c.partition_id != partition_id:
            reason = "PARTITION_MISMATCH"
        elif c.source_state_id != source_state_id:
            reason = "SOURCE_STATE_MISMATCH"
        elif array_id(packet.integration_states[0]) != c.predicted_start_state_id:
            reason = "PREDICTED_STATE_PAYLOAD_MISMATCH"
        elif c.payload_sha256 != payload_id(packet.integration_states, packet.torques,
                                           packet.endpoint_velocity, packet.reference_end):
            reason = "COMMAND_PAYLOAD_MISMATCH"
        elif c.start_admission_condition != "full_integration_state_linf_le_1e-9":
            reason = "UNSUPPORTED_START_ADMISSION_CONDITION"
        elif self.active is not None and (
            c.predecessor_prediction_id != array_id(self.active.integration_states[-1])
            or np.max(np.abs(packet.integration_states[0] - self.active.integration_states[-1])) > STATE_TOL
            or abs(c.execution_start - self.active.certificate.execution_end) > 1e-9):
            reason = "PREDECESSOR_PREDICTION_MISMATCH"
        if reason is None:
            self.pending = packet.immutable_after_transport()
            self.pending_published_at = now
        return {"accepted": reason is None, "reason": reason, "time": now,
                "publish_slack_s": c.publish_deadline - now}

    def handoff(self, now, observed_state, simulation_s):
        p = self.pending
        reason = None
        if p is None:
            reason = "NO_VALID_CONTINUATION"
        else:
            c = p.certificate
            if (not math.isfinite(now) or not math.isfinite(simulation_s)
                    or np.asarray(observed_state).shape != p.integration_states[0].shape
                    or not np.all(np.isfinite(observed_state))):
                reason = "NONFINITE_OR_MALFORMED_OBSERVED_STATE"
            elif not c.execution_start <= now < c.execution_start + SERVO_PERIOD:
                reason = "HANDOFF_OUTSIDE_FIRST_UPDATE_WINDOW"
            elif abs(simulation_s - c.execution_start_simulation_s) > STATE_TOL:
                reason = "HANDOFF_TIME_MISMATCH"
            elif np.max(np.abs(observed_state - p.integration_states[0])) > STATE_TOL:
                reason = "HANDOFF_STATE_MISMATCH"
            elif self.active is not None and self.next_substep != 10:
                reason = "PREDECESSOR_NOT_EXECUTED"
        if reason is None:
            self.active, self.pending = p, None
            self.last_id = p.certificate.command_id
            self.next_substep = 0
        elif p is not None:
            self.pending = None
        return {"accepted": reason is None, "reason": reason, "time": now,
                "continuation_guaranteed": reason is None}

    def servo(self, now, observed_state, simulation_s, substep):
        p = self.active
        reason = None
        if p is None:
            reason = "NO_VALID_CONTINUATION"
        else:
            c = p.certificate
            scheduled = c.execution_start + substep * SERVO_PERIOD
            if (not math.isfinite(now) or not math.isfinite(simulation_s)
                    or np.asarray(observed_state).shape != p.integration_states[0].shape
                    or not np.all(np.isfinite(observed_state))):
                reason = "NONFINITE_OR_MALFORMED_OBSERVED_STATE"
            elif substep != self.next_substep or substep not in range(10):
                reason = "VALIDATED_RAMP_EXHAUSTED_OR_OUT_OF_ORDER"
            elif not scheduled <= now < min(scheduled + SERVO_PERIOD, c.execution_end):
                reason = "MISSED_SERVO_WINDOW"
            elif abs(simulation_s - (c.execution_start_simulation_s + substep * SERVO_PERIOD)) > STATE_TOL:
                reason = "SERVO_SIMULATION_TIME_MISMATCH"
            elif np.max(np.abs(observed_state - p.integration_states[substep])) > STATE_TOL:
                reason = "SERVO_STATE_MISMATCH"
        if reason is None:
            self.next_substep += 1
        return {"accepted": reason is None, "reason": reason, "time": now,
                "servo_substep": substep, "continuation_guaranteed": reason is None}
