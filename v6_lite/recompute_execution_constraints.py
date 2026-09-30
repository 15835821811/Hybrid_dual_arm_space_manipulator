"""Rebuild V6.2-A.1 safety rows from native replay states without QP.solve.

This verifier intentionally does not read logged constraint matrices or use the
controller's row-construction methods. It shares the declared geometry models
and configuration, then recomputes the reaction map, witnesses, target drift,
instantaneous rows, and affine next-start rows from MuJoCo state.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import mujoco
import numpy as np

from model_test.robot_model_spec_v5 import RobotModelSpecV5
from model_test.whole_body_verifier_v5 import (
    CollisionPair, WholeBodyCollisionVerifier, WholeBodyVerificationConfig,
    WorkspaceSphere,
)
from v6_lite.continuum_model_spec import (
    CONTINUUM_ACTUATED_DOF, default_continuum_model_spec,
)
from v6_lite.continuum_shape_model import ContinuumShapeModel, transform_from_free_qpos
from v6_lite.execution_ramp import ramp_mean_weights, ramp_velocity
from v6_lite.hierarchical_qp import HierarchicalQPConfig, free_joint_slices, joint_addresses
from v6_lite.pcc_clearance import PCCClearanceEvaluator
from v6_lite.run_v6_lite import default_v6_lite_robot_spec
from v6_lite.shape_clearance import (
    build_continuum_capsule_envelopes, minimum_capsule_clearance,
    target_box_from_mujoco,
)


@dataclass(frozen=True)
class RecomputedRows:
    matrix: np.ndarray
    lower: np.ndarray
    sources: tuple[str, ...]
    distances_m: np.ndarray
    target_drifts_m_s: np.ndarray
    barrier_gains_s_inv: np.ndarray
    reaction_map_residual: float


class ReplayConstraintBuilder:
    """Independent row assembly using the same physical proxy definitions."""

    def __init__(self, spec: RobotModelSpecV5, model: mujoco.MjModel,
                 pairs: Sequence[CollisionPair], config: HierarchicalQPConfig):
        self.spec, self.model, self.pairs, self.config = spec, model, tuple(pairs), config
        self.qpos_ids, self.dof_ids = joint_addresses(model, spec)
        self.base_qpos_slice, self.base_dof_slice = free_joint_slices(model, spec.base_joint_name)
        _target_qpos, self.target_dof_slice = free_joint_slices(model, spec.target_free_joint_name)
        self.shape_spec = default_continuum_model_spec(spec)
        self.shape_model = ContinuumShapeModel(self.shape_spec)
        target_geom_id = int(mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_GEOM, "target_satellite_collision"
        ))
        if target_geom_id < 0:
            raise ValueError("target satellite collision OBB is missing")
        self.target_geom_id = target_geom_id
        self.target_body_id = int(model.geom_bodyid[target_geom_id])
        self.base_body_id = int(mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, self.shape_spec.base_body_name
        ))
        self.pcc = (
            PCCClearanceEvaluator(
                self.shape_model,
                coarse_samples_per_segment=config.pcc_coarse_samples_per_segment,
                active_segment_count=config.pcc_active_segment_count,
                refinement_max_iterations=config.pcc_refinement_max_iterations,
            ) if config.enable_pcc_cbf else None
        )
        self.capsules = (
            build_continuum_capsule_envelopes(model, self.shape_spec)
            if config.enable_capsule_cbf else None
        )

    def _point_jacobian(self, data: mujoco.MjData, body_id: int,
                        point: np.ndarray) -> np.ndarray:
        result = np.zeros((3, self.model.nv), dtype=np.float64)
        if body_id:
            mujoco.mj_jac(self.model, data, result, None, point, body_id)
        return result

    def _reaction_map(self, data: mujoco.MjData) -> tuple[np.ndarray, float]:
        full_mass = np.zeros((self.model.nv, self.model.nv), dtype=np.float64)
        mujoco.mj_fullM(self.model, full_mass, data.qM)
        base_ids = np.arange(self.base_dof_slice.start, self.base_dof_slice.stop)
        mass_bb = full_mass[np.ix_(base_ids, base_ids)]
        mass_ba = full_mass[np.ix_(base_ids, self.dof_ids)]
        arm_map = self.spec.planner_to_low_level
        base_map = -np.linalg.solve(mass_bb, mass_ba @ arm_map)
        generalized_map = np.zeros((self.model.nv, 17), dtype=np.float64)
        generalized_map[base_ids] = base_map
        generalized_map[self.dof_ids] = arm_map
        residual = float(np.linalg.norm(mass_bb @ base_map + mass_ba @ arm_map))
        return generalized_map, residual

    def build(self, data: mujoco.MjData) -> RecomputedRows:
        mujoco.mj_forward(self.model, data)
        return self._build_forwarded(data)

    def _minimum_capsule(self, data: mujoco.MjData, target_box):
        return minimum_capsule_clearance(
            self.model, data, self.capsules, target_box,
            spec=self.shape_spec, compute_planner_gradient=False,
        )

    def _build_forwarded(self, data: mujoco.MjData) -> RecomputedRows:
        """Assemble rows when mj_forward has already prepared this state."""
        generalized_map, reaction_residual = self._reaction_map(data)
        target_qvel = np.zeros(self.model.nv, dtype=np.float64)
        target_qvel[self.target_dof_slice] = data.qvel[self.target_dof_slice]
        rows: list[np.ndarray] = []
        lowers: list[float] = []
        sources: list[str] = []
        distances: list[float] = []
        drifts: list[float] = []
        gains: list[float] = []

        def append(source: str, distance: float, gradient: np.ndarray,
                   drift: float, gain: float, safe: float) -> None:
            rows.append(np.asarray(gradient, dtype=np.float64))
            lowers.append(-gain * (distance - safe) - drift)
            sources.append(source)
            distances.append(distance)
            drifts.append(drift)
            gains.append(gain)

        fromto = np.zeros(6, dtype=np.float64)
        for pair in self.pairs:
            geom_a, geom_b = int(pair.geom_a), int(pair.geom_b)
            distance = float(mujoco.mj_geomDistance(
                self.model, data, geom_a, geom_b,
                self.config.clearance_query_max_m, fromto,
            ))
            if distance > self.config.clearance_activation_m:
                continue
            first, second = fromto[:3].copy(), fromto[3:].copy()
            if int(self.model.geom_type[geom_a]) <= int(self.model.geom_type[geom_b]):
                point_a, point_b = first, second
            else:
                point_a, point_b = second, first
            direction = point_a - point_b
            norm = float(np.linalg.norm(direction))
            if norm <= 1e-10:
                continue
            normal = direction / norm
            body_a = int(self.model.geom_bodyid[geom_a])
            body_b = int(self.model.geom_bodyid[geom_b])
            relative_jacobian = (
                self._point_jacobian(data, body_a, point_a)
                - self._point_jacobian(data, body_b, point_b)
            )
            row = normal @ relative_jacobian @ generalized_map
            drift = float(normal @ relative_jacobian @ target_qvel)
            safe = (
                self.config.rigid_target_clearance_safe_m
                if pair.pair_class == "rigid_target"
                else self.config.clearance_safe_m
            )
            append(
                f"mujoco:{pair.pair_class}:{pair.geom_a_name}:{pair.geom_b_name}",
                distance, row, drift, self.config.clearance_barrier_gain, safe,
            )

        if self.pcc is not None or self.capsules is not None:
            target_box = target_box_from_mujoco(self.model, data, self.target_geom_id)
        if self.pcc is not None:
            low_level = np.asarray(data.qpos[self.qpos_ids[:CONTINUUM_ACTUATED_DOF]])
            projection = self.shape_spec.project_actual_configuration(low_level)
            base_transform = transform_from_free_qpos(data.qpos[self.base_qpos_slice])
            result = self.pcc.evaluate(
                projection.planner_configuration, base_transform, target_box,
            )
            if result.distance <= self.config.pcc_clearance_activation_m:
                base_jacobian = self._point_jacobian(
                    data, self.base_body_id, result.centerline_point,
                )
                row = np.asarray(result.normal @ base_jacobian @ generalized_map).copy()
                row[:10] += result.gradient
                target_jacobian = self._point_jacobian(
                    data, self.target_body_id, result.point_on_obb,
                )
                drift = float(-result.normal @ target_jacobian @ target_qvel)
                append(
                    f"pcc:segment_{result.segment_id + 1}", result.distance,
                    row, drift, self.config.pcc_clearance_barrier_gain,
                    self.config.pcc_clearance_safe_m,
                )
        if self.capsules is not None:
            result = self._minimum_capsule(data, target_box)
            if result.signed_distance_m <= self.config.capsule_clearance_activation_m:
                body_id = int(self.model.geom_bodyid[result.source_index])
                arm_jacobian = self._point_jacobian(data, body_id, result.point_on_arm)
                row = result.normal_box_to_arm @ arm_jacobian @ generalized_map
                target_jacobian = self._point_jacobian(
                    data, self.target_body_id, result.point_on_box,
                )
                drift = float(-result.normal_box_to_arm @ target_jacobian @ target_qvel)
                append(
                    f"capsule:{result.source_name}", result.signed_distance_m,
                    row, drift, self.config.capsule_clearance_barrier_gain,
                    self.config.capsule_clearance_safe_m,
                )
        matrix = np.vstack(rows) if rows else np.zeros((0, 17), dtype=np.float64)
        return RecomputedRows(
            matrix, np.asarray(lowers), tuple(sources), np.asarray(distances),
            np.asarray(drifts), np.asarray(gains), reaction_residual,
        )


def tick_residuals(rows: RecomputedRows, old: np.ndarray, endpoint: np.ndarray,
                   config: HierarchicalQPConfig) -> dict[str, Any]:
    """Recompute every ramp and frozen next-start row from fresh geometry."""
    a, lower = rows.matrix, rows.lower
    if not len(lower):
        raise ValueError("replay state has no active clearance rows")
    old = np.asarray(old, dtype=np.float64)
    endpoint = np.asarray(endpoint, dtype=np.float64)
    starts = a @ old - lower
    ends = a @ endpoint - lower
    ramp_min = min(
        float(np.min(a @ ramp_velocity(old, endpoint, step) - lower))
        for step in range(11)
    )
    old_weight, new_weight = ramp_mean_weights()
    gain_dt = rows.barrier_gains_s_inv * config.task_period_s
    lookahead_matrix = a * (1.0 + gain_dt * new_weight)[:, None]
    lookahead_lower = lower - gain_dt * (
        old_weight * (a @ old) + rows.target_drifts_m_s
    ) + config.lookahead_model_margin_m_s
    lookahead = lookahead_matrix @ endpoint - lookahead_lower
    categories = {name: 0 for name in ("mujoco", "pcc", "capsule")}
    for source, slack in zip(rows.sources, lookahead):
        if slack <= 2e-5:
            categories[source.split(":", 1)[0]] += 1
    return {
        "start_min_m_s": float(np.min(starts)),
        "end_min_m_s": float(np.min(ends)),
        "ramp_min_m_s": ramp_min,
        "lookahead_min_m_s": float(np.min(lookahead)),
        "worst_ramp_source": rows.sources[int(np.argmin(starts))] if np.min(starts) <= np.min(ends)
        else rows.sources[int(np.argmin(ends))],
        "worst_lookahead_source": rows.sources[int(np.argmin(lookahead))],
        "lookahead_binding_by_source": categories,
        "reaction_map_residual": rows.reaction_map_residual,
        "row_count": len(rows.sources),
    }


def assess_tick(recomputed: dict[str, Any], logged_ramp: float,
                logged_lookahead: float, *, tolerance: float,
                agreement_tolerance: float = 1e-7) -> dict[str, Any]:
    """A forged positive log cannot override a reconstructed violation."""
    actual_ramp = float(recomputed["ramp_min_m_s"])
    actual_lookahead = float(recomputed["lookahead_min_m_s"])
    return {
        "feasible": actual_ramp >= -tolerance and actual_lookahead >= -tolerance,
        "log_agrees": (
            abs(actual_ramp - logged_ramp) <= agreement_tolerance
            and abs(actual_lookahead - logged_lookahead) <= agreement_tolerance
        ),
    }


def _obstacles(scenario: dict[str, Any]) -> tuple[WorkspaceSphere, ...]:
    return tuple(WorkspaceSphere(
        name=item["name"], center=np.asarray(item["center_w"]),
        radius=float(item["radius_m"]),
    ) for item in scenario["workspace_obstacles"])


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_recomputed_execution(output_dir: Path, *,
                                  report_path: Path | None = None) -> dict[str, Any]:
    """Replay torques and independently rebuild 50 Hz constraint residuals."""
    output_dir = Path(output_dir)
    metrics_path = output_dir / "v6_lite_metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    spec = default_v6_lite_robot_spec()
    config = HierarchicalQPConfig(**metrics["qp_config"])
    run_config = metrics["run_config"]
    reports = []
    all_feasible = True
    all_agree = True
    all_states_match = True
    all_trace_hashes_match = True
    for scenario_result in metrics["scenarios"]:
        verifier = WholeBodyCollisionVerifier(
            spec, _obstacles(scenario_result["scenario"]),
            WholeBodyVerificationConfig(
                minimum_clearance=run_config["whole_body_minimum_clearance_m"],
                query_distance_max=2.5,
                adaptive_subdivisions=run_config["verification_subdivisions"],
                self_collision_ancestor_exclusion_depth=3,
                include_target_satellite_pairs=True,
            ),
        )
        model = verifier.model
        model.geom_contype[:] = 0
        model.geom_conaffinity[:] = 0
        data = mujoco.MjData(model)
        builder = ReplayConstraintBuilder(spec, model, verifier.pairs, config)
        path = Path(scenario_result["trace"]["path"])
        trace_sha256 = _sha256(path)
        trace_hash_matches = trace_sha256 == scenario_result["trace"]["sha256"]
        all_trace_hashes_match &= trace_hash_matches
        with np.load(path, allow_pickle=False) as trace:
            initial_qpos = trace["initial_qpos"].copy()
            initial_qvel = trace["initial_qvel"].copy()
            torque = trace["torque"].copy()
            selected = trace["task_selected_command"].copy()
            logged_ramp = trace["task_ramp_clearance_min_slack_m_s"].copy()
            logged_lookahead = trace["task_lookahead_min_slack_m_s"].copy()
            task_qpos = trace["task_qpos"].copy()
            task_time = trace["task_time"].copy()
        data.qpos[:] = initial_qpos
        data.qvel[:] = initial_qvel
        data.ctrl[:] = 0.0
        mujoco.mj_forward(model, data)
        if len(torque) != len(selected) * 10:
            raise ValueError(f"trace task/physics tick mismatch: {path}")
        minima = {name: float("inf") for name in (
            "start", "end", "ramp", "lookahead",
        )}
        max_log_error = 0.0
        max_state_error = 0.0
        max_reaction_map_residual = 0.0
        binding = {name: 0 for name in ("mujoco", "pcc", "capsule")}
        violating_ticks: list[dict[str, Any]] = []
        disagreed_ticks: list[dict[str, Any]] = []
        for step, control in enumerate(torque):
            if step % 10 == 0:
                tick = step // 10
                max_state_error = max(max_state_error, float(np.max(np.abs(
                    data.qpos - task_qpos[tick]
                ))), abs(float(data.time - task_time[tick])))
                rows = builder.build(data)
                old = np.zeros(17) if tick == 0 else selected[tick - 1]
                result = tick_residuals(rows, old, selected[tick], config)
                max_reaction_map_residual = max(
                    max_reaction_map_residual, result["reaction_map_residual"]
                )
                assessment = assess_tick(
                    result, float(logged_ramp[tick]), float(logged_lookahead[tick]),
                    tolerance=config.clearance_rate_tolerance_m_s,
                )
                for name in minima:
                    minima[name] = min(minima[name], result[f"{name}_min_m_s"])
                max_log_error = max(max_log_error,
                                    abs(result["ramp_min_m_s"] - logged_ramp[tick]),
                                    abs(result["lookahead_min_m_s"] - logged_lookahead[tick]))
                for name in binding:
                    binding[name] += result["lookahead_binding_by_source"][name]
                if not assessment["feasible"]:
                    violating_ticks.append({"tick": tick, "time_s": float(data.time), **result})
                if not assessment["log_agrees"]:
                    disagreed_ticks.append({"tick": tick, "time_s": float(data.time), **result})
            data.ctrl[:] = control
            mujoco.mj_step(model, data)
        scenario_feasible = not violating_ticks
        scenario_agrees = not disagreed_ticks
        scenario_state_matches = max_state_error <= 1e-8
        all_feasible &= scenario_feasible
        all_agree &= scenario_agrees
        all_states_match &= scenario_state_matches
        reports.append({
            "scenario_id": scenario_result["scenario"]["scenario_id"],
            "task_ticks": len(selected),
            "physics_steps": len(torque),
            "recomputed_minimum_residuals_m_s": minima,
            "max_logged_residual_disagreement_m_s": max_log_error,
            "max_replay_state_disagreement": max_state_error,
            "max_reaction_map_residual": max_reaction_map_residual,
            "trace_sha256": trace_sha256,
            "trace_hash_matches": trace_hash_matches,
            "lookahead_binding_by_source": binding,
            "violating_ticks": violating_ticks[:10],
            "violating_tick_count": len(violating_ticks),
            "disagreed_ticks": disagreed_ticks[:10],
            "disagreed_tick_count": len(disagreed_ticks),
            "passed": scenario_feasible and scenario_agrees and scenario_state_matches
            and trace_hash_matches,
        })
    checks = {
        "five_scenarios": len(reports) == 5,
        "native_torque_replay_states_match": all_states_match,
        "recomputed_ramp_and_next_start_feasible": all_feasible,
        "recomputed_residuals_match_logged_diagnostics": all_agree,
        "trace_hashes_match": all_trace_hashes_match,
    }
    report = {
        "evidence_type": "independent_native_replay_constraint_recomputation",
        "contract_version": metrics["contract_version"],
        "metrics_sha256": _sha256(metrics_path),
        "model_identity": metrics["robot_model_identity"],
        "run_config": run_config,
        "qp_config": metrics["qp_config"],
        "passed": all(checks.values()),
        "checks": checks,
        "scenarios": reports,
        "method": "native_mj_step_then_independent_geometry_and_reaction_map_rebuild_no_qp_solve",
    }
    destination = output_dir / "recomputed_execution_validation.json" if report_path is None else report_path
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False),
                           encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--report-path", type=Path)
    args = parser.parse_args()
    report = validate_recomputed_execution(args.output_dir, report_path=args.report_path)
    print(json.dumps({"passed": report["passed"], "checks": report["checks"]},
                     ensure_ascii=False, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
