"""Independent whole-body collision verification for the V5 robot contract.

The verifier deliberately does not inspect ``data.ncon`` and does not depend on
contact response.  It compiles a geometry-only copy of the audited robot, adds
the scenario obstacles as static MuJoCo geoms, and calls ``mj_geomDistance`` for
an explicit, named set of robot/environment, moving-target, and self-collision
pairs.

The result is a dense-discrete certificate, not a continuous-time certificate.
Optional adaptive subdivision uses MuJoCo's configuration-space integration to
expose between-sample collisions while preserving free-joint quaternion
semantics.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

import mujoco
import numpy as np

try:
    from .robot_model_spec_v5 import RobotModelSpecV5
except ImportError:  # pragma: no cover - direct script execution compatibility
    from robot_model_spec_v5 import RobotModelSpecV5


@dataclass(frozen=True)
class WorkspaceSphere:
    name: str
    center: np.ndarray
    radius: float

    def validate(self) -> None:
        center = np.asarray(self.center, dtype=np.float64)
        if not self.name:
            raise ValueError("workspace sphere name cannot be empty")
        if center.shape != (3,) or np.any(~np.isfinite(center)):
            raise ValueError("workspace sphere center must be finite with shape (3,)")
        if not np.isfinite(self.radius) or self.radius <= 0.0:
            raise ValueError("workspace sphere radius must be finite and positive")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "name": self.name,
            "center_w": np.asarray(self.center, dtype=np.float64).tolist(),
            "radius_m": float(self.radius),
        }


@dataclass(frozen=True)
class CollisionPair:
    pair_class: str
    geom_a: int
    geom_b: int
    geom_a_name: str
    geom_b_name: str
    body_a_name: str
    body_b_name: str


@dataclass(frozen=True)
class WholeBodyVerificationConfig:
    minimum_clearance: float = 0.0
    query_distance_max: float = 2.5
    adaptive_subdivisions: int = 4
    self_collision_ancestor_exclusion_depth: int = 3
    # Target-satellite pairs are opt-in so the shared V5 verifier keeps its
    # historical pair policy and hashes.  V6-lite enables this explicitly.
    include_target_satellite_pairs: bool = False
    # The fixed grasp pose places the rigid end link and its dedicated 10 mm
    # contact cube inside the global 25 mm clearance margin.  Only these two
    # explicitly named terminal-assembly geoms may be exempted from
    # rigid-arm/target clearance.  The continuum arm is never exempted.
    intentional_target_contact_geom_names: tuple[str, ...] = (
        "collision_0072",
        "collision_0073",
    )

    def validate(self) -> None:
        if self.minimum_clearance < 0.0:
            raise ValueError("minimum_clearance cannot be negative")
        if self.query_distance_max <= self.minimum_clearance:
            raise ValueError("query_distance_max must exceed minimum_clearance")
        if self.adaptive_subdivisions < 1:
            raise ValueError("adaptive_subdivisions must be at least one")
        if self.self_collision_ancestor_exclusion_depth < 1:
            raise ValueError("self-collision exclusion depth must be positive")
        if not isinstance(self.include_target_satellite_pairs, bool):
            raise ValueError("include_target_satellite_pairs must be boolean")
        names = self.intentional_target_contact_geom_names
        if any(not isinstance(name, str) or not name for name in names):
            raise ValueError(
                "intentional target-contact geom names must be nonempty strings"
            )
        if len(set(names)) != len(names):
            raise ValueError("intentional target-contact geom names must be unique")


@dataclass(frozen=True)
class WholeBodyVerificationReport:
    feasible: bool
    minimum_clearance: float
    minimum_by_class: Mapping[str, float]
    minimum_pair: Mapping[str, Any]
    violation_count: int
    checked_state_count: int
    supplied_sample_count: int
    adaptive_subdivisions: int
    pair_count: int
    query_count: int
    truncated_query_count: int
    time_scope: str
    continuous_time_certified: bool
    uses_mj_geom_distance: bool
    uses_contact_list: bool
    nativeccd_claimed: bool
    pair_policy_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class WholeBodyCollisionVerifier:
    """Signed-distance verifier for the arms, base, target, and obstacles."""

    _CONTINUUM_ROOT = "base_link"
    _CONTINUUM_TIP = "end_effector"
    _RIGID_ROOT = "base_link_r"
    _RIGID_TIP = "end_effector_r"
    _RIGID_TARGET_CONTACT_BODIES = ("end_link", "end_effector_r")
    _TARGET_BODY = "target_satellite"
    _TARGET_COLLISION_GEOM = "target_satellite_collision"
    _BASE_BODIES = ("base_of_satelltte", "driving_box")
    _MOUNT_BODIES = ("base_link", "base_link_r")

    def __init__(
        self,
        robot_spec: RobotModelSpecV5,
        obstacles: Sequence[WorkspaceSphere] = (),
        config: WholeBodyVerificationConfig = WholeBodyVerificationConfig(),
    ) -> None:
        robot_spec.validate()
        config.validate()
        self.robot_spec = robot_spec
        self.obstacles = tuple(obstacles)
        self.config = config
        for obstacle in self.obstacles:
            obstacle.validate()
        self.model = self._compile_geometry_model()
        self.data = mujoco.MjData(self.model)
        self.pairs = self._build_pairs()
        if not self.pairs:
            raise RuntimeError("whole-body collision pair policy produced no pairs")
        self._pair_policy_sha256 = self._hash_pair_policy()

    @staticmethod
    def _name(model: mujoco.MjModel, object_type: mujoco.mjtObj, index: int) -> str:
        return mujoco.mj_id2name(model, object_type, index) or f"unnamed_{index}"

    def _compile_geometry_model(self) -> mujoco.MjModel:
        spec = mujoco.MjSpec.from_file(str(self.robot_spec.source_urdf))
        # Match the V5 runtime compilation.  Adding motors before compilation
        # also prevents MuJoCo's URDF static-body fusion from changing the body
        # identities used by the audited pair policy.
        spec.option.gravity[:] = np.zeros(3)
        spec.option.timestep = self.robot_spec.simulation_timestep
        spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
        for joint_name, armature, damping, torque_limit in zip(
            self.robot_spec.low_level_joint_names,
            self.robot_spec.joint_armature,
            self.robot_spec.joint_damping,
            self.robot_spec.torque_limits,
        ):
            joint = spec.joint(joint_name)
            if joint is None:
                raise RuntimeError(f"cannot configure missing joint {joint_name!r}")
            joint.armature = float(armature)
            joint.damping = float(damping)
            actuator = spec.add_actuator(name=f"v5_torque_{joint_name}", target=joint_name)
            actuator.trntype = mujoco.mjtTrn.mjTRN_JOINT
            actuator.gaintype = mujoco.mjtGain.mjGAIN_FIXED
            actuator.gainprm[0] = 1.0
            actuator.biastype = mujoco.mjtBias.mjBIAS_NONE
            actuator.gear[0] = 1.0
            actuator.ctrllimited = True
            actuator.ctrlrange[:] = (-float(torque_limit), float(torque_limit))
            actuator.forcelimited = True
            actuator.forcerange[:] = (-float(torque_limit), float(torque_limit))
        for obstacle_index, obstacle in enumerate(self.obstacles):
            # Prefixing prevents a scenario-provided name from shadowing a
            # robot geom.  These are fixed world geoms used only for distance
            # queries; no contact-based conclusion is drawn from them.
            spec.worldbody.add_geom(
                name=f"v5_workspace_sphere_{obstacle_index:03d}_{obstacle.name}",
                type=mujoco.mjtGeom.mjGEOM_SPHERE,
                size=[float(obstacle.radius), 0.0, 0.0],
                pos=np.asarray(obstacle.center, dtype=np.float64).tolist(),
                contype=1,
                conaffinity=1,
            )
        model = spec.compile()
        if (model.nq, model.nv) != (81, 79):
            raise RuntimeError(
                f"geometry model state spaces changed: nq={model.nq}, nv={model.nv}"
            )
        return model

    def _body_id(self, name: str) -> int:
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
        if body_id < 0:
            raise ValueError(f"missing body {name!r} in geometry model")
        return int(body_id)

    def _descendant_body_ids(self, root_name: str, tip_name: str) -> set[int]:
        root = self._body_id(root_name)
        tip = self._body_id(tip_name)
        values = set()
        current = tip
        while current != 0:
            values.add(current)
            if current == root:
                return values
            current = int(self.model.body_parentid[current])
        raise ValueError(f"{tip_name!r} is not a descendant of {root_name!r}")

    def _collision_geoms(self, body_ids: set[int]) -> list[int]:
        values = []
        for geom_id in range(self.model.ngeom):
            if int(self.model.geom_bodyid[geom_id]) not in body_ids:
                continue
            if not (
                int(self.model.geom_contype[geom_id])
                or int(self.model.geom_conaffinity[geom_id])
            ):
                continue
            values.append(geom_id)
        return values

    def _geom_id(self, name: str) -> int:
        geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name)
        if geom_id < 0:
            raise ValueError(f"missing geom {name!r} in geometry model")
        return int(geom_id)

    def _intentional_target_contact_geom_ids(
        self, rigid_geoms: Sequence[int]
    ) -> set[int]:
        """Resolve and tightly validate the rigid-tip contact whitelist.

        A typo must fail closed, and a caller cannot exempt an arbitrary rigid
        link (or the target itself) merely by supplying a geom name.  Every
        exempted geom must belong to the two-body terminal grasp assembly.
        """

        rigid_geom_set = set(rigid_geoms)
        allowed_body_ids = {
            self._body_id(name) for name in self._RIGID_TARGET_CONTACT_BODIES
        }
        whitelist: set[int] = set()
        for geom_name in self.config.intentional_target_contact_geom_names:
            geom_id = self._geom_id(geom_name)
            if geom_id not in rigid_geom_set:
                raise ValueError(
                    f"intentional target-contact geom {geom_name!r} is not a "
                    "rigid-arm collision geom"
                )
            body_id = int(self.model.geom_bodyid[geom_id])
            if body_id not in allowed_body_ids:
                body_name = self._name(
                    self.model, mujoco.mjtObj.mjOBJ_BODY, body_id
                )
                raise ValueError(
                    f"intentional target-contact geom {geom_name!r} belongs to "
                    f"non-terminal grasp-assembly body {body_name!r}"
                )
            whitelist.add(geom_id)
        return whitelist

    def _ancestor_distance(self, first: int, second: int) -> int | None:
        ancestors: dict[int, int] = {}
        current = first
        distance = 0
        while True:
            ancestors[current] = distance
            if current == 0:
                break
            current = int(self.model.body_parentid[current])
            distance += 1
        current = second
        distance = 0
        while True:
            if current in ancestors:
                return ancestors[current] + distance
            if current == 0:
                return None
            current = int(self.model.body_parentid[current])
            distance += 1

    def _pair(self, pair_class: str, geom_a: int, geom_b: int) -> CollisionPair:
        body_a = int(self.model.geom_bodyid[geom_a])
        body_b = int(self.model.geom_bodyid[geom_b])
        return CollisionPair(
            pair_class=pair_class,
            geom_a=geom_a,
            geom_b=geom_b,
            geom_a_name=self._name(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom_a),
            geom_b_name=self._name(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom_b),
            body_a_name=self._name(self.model, mujoco.mjtObj.mjOBJ_BODY, body_a),
            body_b_name=self._name(self.model, mujoco.mjtObj.mjOBJ_BODY, body_b),
        )

    def _self_pairs(self, geoms: Sequence[int], pair_class: str) -> list[CollisionPair]:
        pairs = []
        exclusion = self.config.self_collision_ancestor_exclusion_depth
        for index, geom_a in enumerate(geoms):
            body_a = int(self.model.geom_bodyid[geom_a])
            for geom_b in geoms[index + 1 :]:
                body_b = int(self.model.geom_bodyid[geom_b])
                graph_distance = self._ancestor_distance(body_a, body_b)
                if graph_distance is not None and graph_distance <= exclusion:
                    continue
                pairs.append(self._pair(pair_class, geom_a, geom_b))
        return pairs

    def _build_pairs(self) -> tuple[CollisionPair, ...]:
        continuum_bodies = self._descendant_body_ids(
            self._CONTINUUM_ROOT, self._CONTINUUM_TIP
        )
        rigid_bodies = self._descendant_body_ids(self._RIGID_ROOT, self._RIGID_TIP)
        base_bodies = {self._body_id(name) for name in self._BASE_BODIES}
        mount_bodies = {self._body_id(name) for name in self._MOUNT_BODIES}
        continuum_geoms = self._collision_geoms(continuum_bodies)
        rigid_geoms = self._collision_geoms(rigid_bodies)
        base_geoms = self._collision_geoms(base_bodies)
        obstacle_geoms = []
        for obstacle_index, obstacle in enumerate(self.obstacles):
            geom_id = mujoco.mj_name2id(
                self.model,
                mujoco.mjtObj.mjOBJ_GEOM,
                f"v5_workspace_sphere_{obstacle_index:03d}_{obstacle.name}",
            )
            if geom_id < 0:
                raise RuntimeError(f"compiled obstacle geom missing for {obstacle.name!r}")
            obstacle_geoms.append(int(geom_id))

        pairs: list[CollisionPair] = []
        pairs.extend(self._self_pairs(continuum_geoms, "continuum_self"))
        pairs.extend(self._self_pairs(rigid_geoms, "rigid_self"))
        pairs.extend(
            self._pair("arm_arm", geom_a, geom_b)
            for geom_a in continuum_geoms
            for geom_b in rigid_geoms
        )
        # The two branch roots are deliberately mounted in/on the base.  Only
        # these named mount bodies are excluded; distal arm/base pairs remain.
        distal_arm_geoms = [
            geom_id
            for geom_id in continuum_geoms + rigid_geoms
            if int(self.model.geom_bodyid[geom_id]) not in mount_bodies
        ]
        pairs.extend(
            self._pair("arm_base", geom_a, geom_b)
            for geom_a in distal_arm_geoms
            for geom_b in base_geoms
        )
        pairs.extend(
            self._pair("arm_obstacle", geom_a, geom_b)
            for geom_a in continuum_geoms + rigid_geoms
            for geom_b in obstacle_geoms
        )
        pairs.extend(
            self._pair("base_obstacle", geom_a, geom_b)
            for geom_a in base_geoms
            for geom_b in obstacle_geoms
        )
        if self.config.include_target_satellite_pairs:
            target_body_id = self._body_id(self._TARGET_BODY)
            target_geom = self._geom_id(self._TARGET_COLLISION_GEOM)
            if int(self.model.geom_bodyid[target_geom]) != target_body_id:
                raise RuntimeError(
                    f"{self._TARGET_COLLISION_GEOM!r} is not attached to "
                    f"{self._TARGET_BODY!r}"
                )
            intentional_target_contact_geoms = (
                self._intentional_target_contact_geom_ids(rigid_geoms)
            )
            # The target is a moving 0.4 m box in the audited URDF.  V6-lite
            # treats it as a hard clearance object even when runtime collision
            # response is disabled; only the named terminal grasp geoms are
            # exempted from rigid-arm/target distance pairs.
            pairs.extend(
                self._pair("continuum_target", geom_a, target_geom)
                for geom_a in continuum_geoms
            )
            pairs.extend(
                self._pair("rigid_target", geom_a, target_geom)
                for geom_a in rigid_geoms
                if geom_a not in intentional_target_contact_geoms
            )
            pairs.extend(
                self._pair("base_target", geom_a, target_geom)
                for geom_a in base_geoms
            )
        unique: dict[tuple[int, int, str], CollisionPair] = {}
        for pair in pairs:
            key = (min(pair.geom_a, pair.geom_b), max(pair.geom_a, pair.geom_b), pair.pair_class)
            unique[key] = pair
        return tuple(unique[key] for key in sorted(unique))

    def pairs_for_model(
        self, model: mujoco.MjModel
    ) -> tuple[CollisionPair, ...]:
        """Return this named pair policy with IDs resolved in ``model``.

        MuJoCo geom IDs are compilation-order dependent.  Callers using a
        model other than ``self.model`` must therefore remap by the audited
        names instead of reusing the verifier's integer IDs.  Body-name checks
        make a superficially matching geom name fail closed if it was attached
        to a different body in the destination model.
        """

        remapped: list[CollisionPair] = []
        for pair in self.pairs:
            geom_a = int(
                mujoco.mj_name2id(
                    model, mujoco.mjtObj.mjOBJ_GEOM, pair.geom_a_name
                )
            )
            geom_b = int(
                mujoco.mj_name2id(
                    model, mujoco.mjtObj.mjOBJ_GEOM, pair.geom_b_name
                )
            )
            if geom_a < 0 or geom_b < 0:
                missing = pair.geom_a_name if geom_a < 0 else pair.geom_b_name
                raise ValueError(
                    f"cannot remap collision policy: missing geom {missing!r}"
                )
            body_a = int(model.geom_bodyid[geom_a])
            body_b = int(model.geom_bodyid[geom_b])
            body_a_name = self._name(
                model, mujoco.mjtObj.mjOBJ_BODY, body_a
            )
            body_b_name = self._name(
                model, mujoco.mjtObj.mjOBJ_BODY, body_b
            )
            if body_a_name != pair.body_a_name or body_b_name != pair.body_b_name:
                raise ValueError(
                    "cannot remap collision policy because a geom/body "
                    f"attachment changed for {pair.geom_a_name!r} or "
                    f"{pair.geom_b_name!r}"
                )
            remapped.append(
                CollisionPair(
                    pair_class=pair.pair_class,
                    geom_a=geom_a,
                    geom_b=geom_b,
                    geom_a_name=pair.geom_a_name,
                    geom_b_name=pair.geom_b_name,
                    body_a_name=body_a_name,
                    body_b_name=body_b_name,
                )
            )
        return tuple(remapped)

    def _hash_pair_policy(self) -> str:
        if self.config.include_target_satellite_pairs:
            config_payload = asdict(self.config)
        else:
            # Preserve byte-for-byte V5 policy identities.  Target-pair
            # options did not exist in the historical V5 contract and have no
            # effect when the feature is disabled, so they must not perturb
            # legacy provenance hashes.
            config_payload = {
                "minimum_clearance": self.config.minimum_clearance,
                "query_distance_max": self.config.query_distance_max,
                "adaptive_subdivisions": self.config.adaptive_subdivisions,
                "self_collision_ancestor_exclusion_depth": (
                    self.config.self_collision_ancestor_exclusion_depth
                ),
            }
        payload = {
            "config": config_payload,
            "pairs": [
                {
                    "class": pair.pair_class,
                    "geom_a": pair.geom_a_name,
                    "geom_b": pair.geom_b_name,
                    "body_a": pair.body_a_name,
                    "body_b": pair.body_b_name,
                }
                for pair in self.pairs
            ],
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()

    def _interpolated_states(self, qpos: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        subdivisions = self.config.adaptive_subdivisions
        if subdivisions == 1 or qpos.shape[0] < 2:
            return qpos.copy(), np.arange(qpos.shape[0], dtype=np.float64)
        states = []
        phases = []
        delta = np.empty(self.model.nv, dtype=np.float64)
        for sample_index in range(qpos.shape[0] - 1):
            start = qpos[sample_index]
            stop = qpos[sample_index + 1]
            mujoco.mj_differentiatePos(self.model, delta, 1.0, start, stop)
            for subdivision in range(subdivisions):
                fraction = subdivision / subdivisions
                value = start.copy()
                mujoco.mj_integratePos(self.model, value, delta, fraction)
                states.append(value)
                phases.append(sample_index + fraction)
        states.append(qpos[-1].copy())
        phases.append(float(qpos.shape[0] - 1))
        return np.asarray(states), np.asarray(phases)

    def verify_qpos_sequence(self, qpos: np.ndarray) -> WholeBodyVerificationReport:
        qpos = np.asarray(qpos, dtype=np.float64)
        if qpos.ndim != 2 or qpos.shape[1] != self.model.nq:
            raise ValueError(
                f"qpos must have shape [time, {self.model.nq}], got {qpos.shape}"
            )
        if qpos.shape[0] == 0 or np.any(~np.isfinite(qpos)):
            raise ValueError("qpos sequence must be nonempty and finite")
        states, phases = self._interpolated_states(qpos)
        minimum = float("inf")
        minimum_by_class: dict[str, float] = {}
        minimum_pair: dict[str, Any] = {}
        violation_count = 0
        truncated = 0
        fromto = np.empty(6, dtype=np.float64)
        for state_index, state in enumerate(states):
            self.data.qpos[:] = state
            self.data.qvel[:] = 0.0
            mujoco.mj_forward(self.model, self.data)
            for pair in self.pairs:
                distance = float(
                    mujoco.mj_geomDistance(
                        self.model,
                        self.data,
                        pair.geom_a,
                        pair.geom_b,
                        self.config.query_distance_max,
                        fromto,
                    )
                )
                if distance >= self.config.query_distance_max - 1e-12:
                    truncated += 1
                    # MuJoCo may return either distmax or DBL_MAX when the
                    # broad phase proves the pair is farther than distmax.  In
                    # both cases the evidence is only the lower bound distmax.
                    distance = self.config.query_distance_max
                current_class_minimum = minimum_by_class.get(pair.pair_class, float("inf"))
                if distance < current_class_minimum:
                    minimum_by_class[pair.pair_class] = distance
                if distance < self.config.minimum_clearance:
                    violation_count += 1
                if distance < minimum:
                    minimum = distance
                    minimum_pair = {
                        "pair_class": pair.pair_class,
                        "geom_a": pair.geom_a_name,
                        "geom_b": pair.geom_b_name,
                        "body_a": pair.body_a_name,
                        "body_b": pair.body_b_name,
                        "state_index": int(state_index),
                        "source_interval_phase": float(phases[state_index]),
                        "signed_distance_m": distance,
                        "fromto": fromto.tolist(),
                    }
        query_count = len(states) * len(self.pairs)
        time_scope = (
            "supplied_samples_only"
            if self.config.adaptive_subdivisions == 1
            else "dense_discrete_with_configuration_space_subdivision"
        )
        return WholeBodyVerificationReport(
            feasible=bool(minimum >= self.config.minimum_clearance),
            minimum_clearance=minimum,
            minimum_by_class=minimum_by_class,
            minimum_pair=minimum_pair,
            violation_count=int(violation_count),
            checked_state_count=len(states),
            supplied_sample_count=qpos.shape[0],
            adaptive_subdivisions=self.config.adaptive_subdivisions,
            pair_count=len(self.pairs),
            query_count=query_count,
            truncated_query_count=truncated,
            time_scope=time_scope,
            continuous_time_certified=False,
            uses_mj_geom_distance=True,
            uses_contact_list=False,
            nativeccd_claimed=False,
            pair_policy_sha256=self._pair_policy_sha256,
        )

    def planner_sequence_to_qpos(
        self,
        planner_q: np.ndarray,
        *,
        initial_qpos: np.ndarray | None = None,
    ) -> np.ndarray:
        planner_q = np.asarray(planner_q, dtype=np.float64)
        if planner_q.ndim != 2 or planner_q.shape[1] != 17:
            raise ValueError("planner_q must have shape [time, 17]")
        if initial_qpos is None:
            base = np.asarray(self.model.qpos0, dtype=np.float64).copy()
        else:
            base = np.asarray(initial_qpos, dtype=np.float64).copy()
        if base.shape != (self.model.nq,):
            raise ValueError(f"initial_qpos must have shape ({self.model.nq},)")
        qpos_ids = []
        for joint_name in self.robot_spec.low_level_joint_names:
            joint_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name
            )
            qpos_ids.append(int(self.model.jnt_qposadr[joint_id]))
        sequence = np.repeat(base[None, :], planner_q.shape[0], axis=0)
        sequence[:, qpos_ids] = self.robot_spec.encode_position(planner_q)
        return sequence
