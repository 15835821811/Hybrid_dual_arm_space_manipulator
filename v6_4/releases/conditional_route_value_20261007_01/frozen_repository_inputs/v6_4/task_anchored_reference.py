"""Cartesian route residuals with immutable task anchors and no joint proposal.

Reference-time partitions below are unrelated to PCC arc-length partitions.
Only a continuum position/velocity reference changes; the inherited provider
still obtains rigid target transport from current physical-time feedback.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json

import numpy as np

from v6_4.task_protocol import TaskSpec, canonical_json
from v6_4.reference_adapter import CartesianPassThroughReferenceProvider
from v6_lite.irregular_waypoints import IrregularWaypointTarget

REPRESENTATION_VERSION = "task_anchored_cartesian_residual_v1"
PLAN_SCHEMA = "task_anchored_residual_plan_v1"
BASE_REFERENCE_ID = "original_cartesian_passthrough_physical_time_home_v1"
MAX_INTERVALS = 6
LATENT_DIM = 12
COEFFICIENT_NORM_BOUND_M = .020
SUPPORT_CUTOFF_S = 23.98
FREE_PATH_POLICY = "free_intermediate_path_between_fixed_requirements"
REFERENCE_MODE = REPRESENTATION_VERSION


def _hash(value):
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _definition_hash(definition):
    return _hash({k: v for k, v in definition.items() if k != "definition_sha256"})


def _target(task):
    source = task.scenario["continuum_target"]
    if (source.get("mode") != "irregular_waypoints"
            or source.get("reference_profile") != "minimum_jerk_c2"):
        raise ValueError("unsupported base Cartesian reference")
    initial = np.asarray(source["initial_position_w"], dtype=float)
    points = np.asarray(source["waypoint_points_m"], dtype=float)
    durations = np.asarray(source["segment_durations_s"], dtype=float)
    transition, path_duration = float(source["transition_duration_s"]), float(source["path_duration_s"])
    rotation = np.asarray(source["target_rotation_world"], dtype=float)
    if (initial.shape != (3,) or points.ndim != 2 or points.shape[1] != 3
            or len(points) < 2 or durations.shape != (len(points)-1,)
            or rotation.shape != (3, 3) or not np.all(np.isfinite(initial))
            or not np.all(np.isfinite(points)) or not np.all(np.isfinite(durations))
            or not np.all(np.isfinite(rotation)) or np.any(durations <= 0.)
            or not np.isfinite(transition) or transition <= 0.
            or not np.isfinite(path_duration) or path_duration <= 0.
            or not np.isclose(durations.sum(), path_duration, atol=1e-10, rtol=0.)
            or transition+path_duration > task.duration_s):
        raise ValueError("invalid declared minimum-jerk Cartesian source")
    return IrregularWaypointTarget(initial, points, rotation, transition, path_duration, durations, {})


def _time_array(time_s):
    times = np.asarray(time_s, dtype=float)
    if (not np.all(np.isfinite(times)) or np.any(times < -1e-9)
            or np.any(times > 27.+1e-9)):
        raise ValueError("reference time must lie within the declared physical horizon")
    # Only the original provider's <=1 ns endpoint clock roundoff is admitted.
    return np.minimum(27., np.maximum(0., times))


def _base_kinematics(target, time_s):
    """Use the original p/v implementation; differentiate its same polynomial."""
    times = _time_array(time_s)
    output = []
    cumulative = np.r_[0., np.cumsum(target.segment_durations_s)]
    for time in times.reshape(-1):
        p, v = target.sample(float(time))
        if time < target.transition_duration_s:
            duration = target.transition_duration_s
            u = float(time)/duration
            delta = target.waypoint_points_w[0]-target.initial_position_w
        elif time >= target.path_end_s:
            output.append((p, v, np.zeros(3)))
            continue
        else:
            elapsed = float(time)-target.transition_duration_s
            index = min(int(np.searchsorted(cumulative, elapsed, side="right")-1), len(cumulative)-2)
            duration = float(target.segment_durations_s[index])
            u = (elapsed-cumulative[index])/duration
            delta = target.waypoint_points_w[index+1]-target.waypoint_points_w[index]
        acceleration = 60.*u*(1.-u)*(1.-2.*u)*delta/duration**2
        output.append((p, v, acceleration))
    if not output:
        return tuple(np.empty(times.shape+(3,)) for _ in range(3))
    values = np.asarray(output).reshape(times.shape+(3, 3))
    return tuple(values[..., i, :] for i in range(3))


def _merge_closed(intervals):
    merged = []
    for lower, upper in sorted(intervals):
        if merged and lower <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], upper)
        else:
            merged.append([float(lower), float(upper)])
    return merged


def build_reference_definition(task: TaskSpec) -> dict:
    """Derive a deterministic definition using declared inputs only.

    Every TaskPoint window is protected as a closed set. The earliest six
    complement components are selected *before* degenerate chords are disabled;
    disabled slots are never replaced by later components.
    """
    if not isinstance(task, TaskSpec):
        raise TypeError("a frozen TaskSpec is required")
    target = _target(task)
    hold = [float(target.path_end_s), task.duration_s]
    windows = [{"point_id": p.point_id, "arm": p.arm, "time_s": p.time_s,
                "time_window_s": list(p.time_window_s),
                "position_and_orientation_required": True,
                "specified_velocity_requirement": None} for p in task.requirements]
    protected = _merge_closed([[0., 0.], [SUPPORT_CUTOFF_S, task.duration_s], hold]
                              + [list(p.time_window_s) for p in task.requirements])
    contract = {"schema": "task_derived_protected_time_contract_v1",
                "task_sha256": task.sha256(), "path_freedom": task.path_freedom,
                "task_point_windows": windows, "initial_time_s": 0.,
                "final_reference_freeze_s": [SUPPORT_CUTOFF_S, task.duration_s],
                "base_reference_hold_intervals_s": [hold],
                "additional_strict_path_intervals_s": None,
                "additional_hold_intervals_s": None,
                "additional_specified_velocity_requirements": None,
                "extra_requirements_status": "NONE_IN_SUPPORTED_TASKSPEC_V1"}
    times = np.zeros((MAX_INTERVALS, 2))
    bases = np.zeros((MAX_INTERVALS, 3, 2))
    mask = np.zeros(MAX_INTERVALS, dtype=bool)
    disabled, reasons, candidates = [], [], []
    if task.path_freedom != FREE_PATH_POLICY:
        reasons.append("UNSUPPORTED_OR_STRICT_PATH_FREEDOM")
        contract["extra_requirements_status"] = "UNKNOWN_OR_STRICT_PATH_POLICY_NOT_APPLICABLE"
    else:
        cursor = 0.
        for lower, upper in protected:
            if lower > cursor:
                candidates.append([cursor, lower])
            cursor = max(cursor, upper)
        for index, interval in enumerate(candidates[:MAX_INTERVALS]):
            lower, upper = interval
            times[index] = interval
            p0, _ = target.sample(lower)
            p1, _ = target.sample(upper)
            chord = p1-p0
            norm = float(np.linalg.norm(chord))
            if norm <= 1e-12:
                disabled.append({"slot": index, "interval_s": interval,
                                 "reason": "DEGENERATE_BASE_REFERENCE_CHORD", "chord_norm_m": norm})
                continue
            direction = chord/norm
            axis_index = int(np.argmin(np.abs(direction)))
            axis = np.eye(3)[axis_index]
            e1 = axis-direction*np.dot(axis, direction)
            e1 /= np.linalg.norm(e1)
            e2 = np.cross(direction, e1)
            bases[index] = np.column_stack([e1, e2])
            mask[index] = True
    if not np.any(mask):
        reasons.append("NO_LEGAL_NONDEGENERATE_RESIDUAL_INTERVAL")
    base_identity = {"base_reference_id": BASE_REFERENCE_ID,
                     "task_sha256": task.sha256(), "scenario": task.scenario,
                     "home_posture_q": list(task.initial_planner_q),
                     "home_posture_dq": [0.]*17, "time_mapping": "identity_physical_time",
                     "joint_terminal_progress_repair": False,
                     "rigid_transport": "current_target_feedback_including_grasp_offset_twist"}
    definition = {"schema": REPRESENTATION_VERSION, "representation_version": REPRESENTATION_VERSION,
                  "task_id": task.task_id, "task_sha256": task.sha256(),
                  "base_reference_id": BASE_REFERENCE_ID, "base_reference_sha256": _hash(base_identity),
                  "base_reference_version": BASE_REFERENCE_ID,
                  "frame": "world", "time_mapping": "identity_physical_time",
                  "intervals_s": times.tolist(), "transverse_bases": bases.tolist(),
                  "interval_mask": mask.tolist(), "coefficient_norm_bound_m": COEFFICIENT_NORM_BOUND_M,
                  "support_cutoff_s": SUPPORT_CUTOFF_S, "protected_time_intervals_s": protected,
                  "protected_time_contract": contract, "partition_kind": "reference_time_not_PCC_arc_length",
                  "interval_selection": "earliest_six_complement_components_then_disable_degenerate_without_replacement",
                  "candidate_component_count": len(candidates), "disabled_intervals": disabled,
                  "applicable": bool(np.any(mask)), "status": "APPLICABLE" if np.any(mask) else "NOT_APPLICABLE",
                  "reasons": reasons}
    definition["definition_sha256"] = _definition_hash(definition)
    return definition


def _validated_definition(value):
    if not isinstance(value, dict):
        raise ValueError("residual definition must be a JSON object")
    definition = json.loads(canonical_json(value))
    if (definition.get("schema") != REPRESENTATION_VERSION
            or definition.get("representation_version") != REPRESENTATION_VERSION
            or definition.get("definition_sha256") != _definition_hash(definition)
            or definition.get("frame") != "world"
            or definition.get("time_mapping") != "identity_physical_time"
            or definition.get("base_reference_id") != BASE_REFERENCE_ID
            or definition.get("base_reference_version") != BASE_REFERENCE_ID
            or definition.get("coefficient_norm_bound_m") != COEFFICIENT_NORM_BOUND_M
            or definition.get("support_cutoff_s") != SUPPORT_CUTOFF_S):
        raise ValueError("invalid residual definition schema, identity or frozen settings")
    if definition.get("applicable") is not True or definition.get("status") != "APPLICABLE":
        raise ValueError("NOT_APPLICABLE: "+str(definition.get("reasons")))
    raw_mask = definition.get("interval_mask")
    if not isinstance(raw_mask, list) or len(raw_mask) != MAX_INTERVALS or any(type(v) is not bool for v in raw_mask):
        raise ValueError("interval mask must contain six explicit booleans")
    mask = np.asarray(raw_mask)
    intervals = np.asarray(definition["intervals_s"], dtype=float)
    bases = np.asarray(definition["transverse_bases"], dtype=float)
    if (intervals.shape != (6, 2) or bases.shape != (6, 3, 2)
            or not np.all(np.isfinite(intervals)) or not np.all(np.isfinite(bases))
            or not np.any(mask) or np.any(bases[~mask] != 0.)):
        raise ValueError("invalid residual interval/basis arrays")
    previous_end = 0.
    for interval, basis, active in zip(intervals, bases, mask):
        if not active:
            continue
        lower, upper = interval
        if not 0. <= lower < upper <= SUPPORT_CUTOFF_S or lower < previous_end:
            raise ValueError("active residual intervals must be ordered, disjoint and before 24 s")
        if not np.allclose(basis.T@basis, np.eye(2), atol=1e-12, rtol=0.):
            raise ValueError("world transverse basis must be orthonormal")
        for start, end in definition["protected_time_intervals_s"]:
            if max(lower, start) < min(upper, end) or lower < start == end < upper:
                raise ValueError("residual interior intersects protected time")
        previous_end = upper
    return definition


@dataclass(frozen=True)
class TaskAnchoredResidualPlan:
    _definition_json: str
    _coefficients: tuple

    @classmethod
    def from_definition(cls, definition, z_m):
        definition = _validated_definition(definition)
        if np.asarray(z_m).dtype.kind not in "fiu":
            raise ValueError("z_m must contain real numeric coefficients")
        coefficients = np.asarray(z_m, dtype=float)
        mask = np.asarray(definition["interval_mask"], dtype=bool)
        if coefficients.shape != (6, 2) or not np.all(np.isfinite(coefficients)):
            raise ValueError("z_m must be finite with shape (6,2)")
        if np.any(coefficients[~mask] != 0.):
            raise ValueError("mask-out coefficients must be exactly zero")
        if np.any(np.linalg.norm(coefficients, axis=1) > COEFFICIENT_NORM_BOUND_M):
            raise ValueError("coefficient two-dimensional norm exceeds 0.020 m")
        return cls(canonical_json(definition), tuple(tuple(float(x) for x in row) for row in coefficients))

    @classmethod
    def from_dict(cls, value):
        if (not isinstance(value, dict)
                or set(value) != {"schema", "representation_version", "definition", "z_m"}
                or value.get("schema") != PLAN_SCHEMA
                or value.get("representation_version") != REPRESENTATION_VERSION):
            raise ValueError("unsupported residual plan schema")
        return cls.from_definition(value["definition"], value["z_m"])

    @property
    def definition(self):
        return json.loads(self._definition_json)

    @property
    def z_m(self):
        values = np.asarray(self._coefficients, dtype=float)
        values.setflags(write=False)
        return values

    @property
    def interval_mask(self):
        values = np.asarray(self.definition["interval_mask"], dtype=bool)
        values.setflags(write=False)
        return values

    def to_dict(self):
        return {"schema": PLAN_SCHEMA, "representation_version": REPRESENTATION_VERSION,
                "definition": self.definition, "z_m": self.z_m.tolist()}

    def sha256(self):
        return _hash(self.to_dict())

    def offset_kinematics(self, time_s):
        """Return world dp [m], dv [m/s], da [m/s²], scalar or array time."""
        times = _time_array(time_s)
        outputs = [np.zeros(times.shape+(3,)) for _ in range(3)]
        definition = self.definition
        for interval, basis, coefficients, mask in zip(definition["intervals_s"],
                definition["transverse_bases"], self._coefficients, definition["interval_mask"]):
            if not mask or not any(coefficients):
                continue
            lower, upper = interval
            duration = upper-lower
            inside = (times > lower) & (times < upper)
            u = np.where(inside, (times-lower)/duration, 0.)
            phi = 64.*u**3*(1.-u)**3
            rate = 192.*u**2*(1.-u)**2*(1.-2.*u)/duration
            acceleration = 384.*u*(1.-u)*(1.-5.*u+5.*u**2)/duration**2
            delta = np.asarray(basis)@np.asarray(coefficients)
            for output, weight in zip(outputs, (phi, rate, acceleration)):
                output += np.where(inside, weight, 0.)[..., None]*delta
        return tuple(outputs)


class TaskAnchoredResidualReferenceProvider:
    def __init__(self, task: TaskSpec, plan: TaskAnchoredResidualPlan):
        if not isinstance(plan, TaskAnchoredResidualPlan):
            raise TypeError("a validated TaskAnchoredResidualPlan is required")
        if canonical_json(plan.definition) != canonical_json(build_reference_definition(task)):
            raise ValueError("residual definition differs from the deterministic frozen TaskSpec")
        self.task, self.plan = task, plan
        self._target = _target(task)
        self._base = CartesianPassThroughReferenceProvider(task)
        self._zero = not np.any(plan.z_m)
        # Compatibility-only *home posture preference*, never a nominal q path
        # for whole-body geometry or old joint-control-point proposal gates.
        self.controls = self._base.controls
        self.prediction = dict(self._base.prediction)
        self.prediction["reference_mode"] = np.asarray(REFERENCE_MODE)
        self.prediction["residual_plan_json"] = np.asarray(canonical_json(plan.to_dict()))
        self.metadata = {"schema": "task_anchored_residual_provider_v1",
            "task_sha256": task.sha256(), "plan_sha256": plan.sha256(),
            "definition_sha256": plan.definition["definition_sha256"],
            "base_reference_id": BASE_REFERENCE_ID, "reference_mode": REFERENCE_MODE,
            "frame": "world", "time_mapping": "identity_physical_time",
            "joint_terminal_progress_repair": False, "home_posture_only": True,
            "nominal_joint_reference_geometry": "N/A_NEW_REPRESENTATION",
            "zero_residual_direct_passthrough": self._zero,
            "future_actual_trace_read": False, "physics_steps_executed": 0}

    def prepare(self, spec, model, initial_data, scenario):
        self._base.prepare(spec, model, initial_data, scenario)
        return self

    def sample(self, time_s):
        base = self._base.sample(time_s)
        if self._zero:
            return base
        offset, velocity, _ = self.plan.offset_kinematics(time_s)
        if not np.any(offset) and not np.any(velocity):
            return base
        base["continuum_target_position"] = base["continuum_target_position"]+offset
        base["continuum_target_velocity"] = base["continuum_target_velocity"]+velocity
        return base

    def offset_kinematics(self, time_s):
        return self.plan.offset_kinematics(time_s)

    def continuum_kinematics(self, time_s):
        """Pure declared-source p/v/a; no current or future actual-state input."""
        base = _base_kinematics(self._target, time_s)
        residual = self.plan.offset_kinematics(time_s)
        return tuple(a+b for a, b in zip(base, residual))


def reference_precheck(task, plan, cartesian_speed_limit_m_s=.24):
    """Reference-level gates only, using an analytic conservative speed bound.

    The 0.24 m/s default is the original continuum task-command cap. This is
    neither a proof of joint feasibility nor any whole-body geometry query.
    """
    checks, errors, points = {}, [], []
    result = {"schema": "task_anchored_reference_precheck_v1", "task_id": task.task_id,
              "task_sha256": task.sha256(), "plan_sha256": None,
              "checks": checks, "errors": errors, "reference_task_requirements": points,
              "joint_control_point_checks": "N/A_NEW_REPRESENTATION",
              "nominal_joint_reference_geometry": "N/A_NO_JOINT_REFERENCE",
              "whole_body_geometry": "NOT_RUN", "physics_steps": 0,
              "native_distance_queries": 0, "future_actual_trace_read": False,
              "reference_check_is_actual_Task_or_safety_acceptance": False}
    try:
        if not np.isfinite(cartesian_speed_limit_m_s) or cartesian_speed_limit_m_s <= 0.:
            raise ValueError("reference Cartesian speed cap must be positive m/s")
        definition = build_reference_definition(task)
        checks["representation_applicable"] = definition["applicable"]
        if not definition["applicable"]:
            raise ValueError("NOT_APPLICABLE: "+str(definition["reasons"]))
        if not isinstance(plan, TaskAnchoredResidualPlan):
            raise TypeError("validated residual plan required")
        checked = TaskAnchoredResidualPlan.from_dict(plan.to_dict())
        result["plan_sha256"] = checked.sha256()
        checks["definition_bound_to_TaskSpec"] = checked.definition == definition
        if not checks["definition_bound_to_TaskSpec"]:
            raise ValueError("plan is not bound to this TaskSpec definition")
        checks["finite_format_mask_and_amplitude"] = True
        provider = TaskAnchoredResidualReferenceProvider(task, checked)
        target = provider._target
        residual = checked.z_m
        u = (5.-np.sqrt(5.))/10.
        max_phi_rate = 192.*u**2*(1.-u)**2*(1.-2.*u)
        deltas = np.vstack([target.waypoint_points_w[0]-target.initial_position_w,
                            np.diff(target.waypoint_points_w, axis=0)])
        durations = np.r_[target.transition_duration_s, target.segment_durations_s]
        base_bound = float(np.max(1.875*np.linalg.norm(deltas, axis=1)/durations))
        residual_bound = max([float(max_phi_rate*np.linalg.norm(z)/(b-a))
            for (a,b),z,mask in zip(definition["intervals_s"], residual, definition["interval_mask"])
            if mask]+[0.])
        upper_bound = base_bound+residual_bound  # supports are pairwise disjoint
        checks["reference_velocity_conservative_bound"] = upper_bound <= cartesian_speed_limit_m_s
        grid = np.arange(13501)*task.physics_period_s
        positions, velocity, _ = provider.continuum_kinematics(grid)
        dp, _, _ = checked.offset_kinematics(grid)
        result.update({"cartesian_speed_limit_m_s": float(cartesian_speed_limit_m_s),
            "reference_speed_upper_bound_m_s": upper_bound,
            "base_reference_speed_upper_bound_m_s": base_bound,
            "residual_speed_upper_bound_m_s": residual_bound,
            "reference_speed_sampled_max_m_s": float(np.max(np.linalg.norm(velocity, axis=1))),
            "reference_velocity_check_scope": "analytic triangle-inequality upper bound, with 13501 physical-time grid diagnostics; not joint velocity feasibility",
            "reference_offset_peak_sampled_m": float(np.max(np.linalg.norm(dp, axis=1))),
            "reference_offset_analytic_peak_m": float(np.max(np.linalg.norm(residual, axis=1))),
            "nonzero_reference": bool(np.any(residual)), "grid_count": len(grid)})
        for point in task.requirements:
            if point.arm == "continuum" and point.frame == "world":
                desired, _, _ = provider.continuum_kinematics(point.time_s)
                position_error = float(np.linalg.norm(desired-np.asarray(point.position_m)))
                rotation = np.asarray(task.scenario["continuum_target_rotation_world"])
            elif point.arm == "rigid" and point.frame == "target":
                position_error = float(np.linalg.norm(np.asarray(task.scenario["grasp_point_target_frame_m"])-point.position_m))
                rotation = np.asarray(task.scenario["grasp_rotation_target_frame"])
            else:
                raise ValueError("reference Task precheck has unsupported arm/frame combination")
            cosine = (np.sum(rotation*np.asarray(point.rotation).reshape(3,3))-1.)*.5
            orientation_error = float(np.arccos(np.clip(cosine, -1., 1.)))
            p, v, a = checked.offset_kinematics(np.asarray(point.time_window_s))
            preserved = bool(np.all(p == 0.) and np.all(v == 0.) and np.all(a == 0.))
            passed = position_error <= point.position_tolerance_m and orientation_error <= point.orientation_tolerance_rad and preserved
            points.append({"point_id": point.point_id, "arm": point.arm,
                "nominal_time_s": point.time_s, "protected_window_s": list(point.time_window_s),
                "position_error_m": position_error, "orientation_error_rad": orientation_error,
                "protected_window_residual_exact_zero": preserved, "passed": bool(passed)})
        checks["reference_Task_at_declared_anchor_times"] = all(p["passed"] for p in points)
        checks["protected_closed_windows_and_hold_preserved"] = all(p["protected_window_residual_exact_zero"] for p in points)
        result["reference_task_passed"] = checks["reference_Task_at_declared_anchor_times"]
        result["reference_velocity_passed"] = checks["reference_velocity_conservative_bound"]
    except (ValueError, TypeError, KeyError, IndexError) as error:
        errors.append({"type": type(error).__name__, "message": str(error)})
    result["passed"] = bool(checks) and all(checks.values()) and not errors
    result["status"] = "REFERENCE_ACCEPTED" if result["passed"] else "REFERENCE_REJECTED"
    return result
