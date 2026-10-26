"""Frozen task definitions for planning, independent of generated references.

All twists in the condition are world-frame linear then angular velocity.
Task requirements use their explicitly named world or target frame.  Intermediate
routes are free; the saved old time-parametrized curve is not the success target.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import numpy as np

TASK_SCHEMA = "v6_4_task_protocol_v1"
FAMILIES = ("end_effector_detour", "mid_arm_detour", "multiple_routes")
SPLITS = ("train", "val", "test")


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _vector(value, size, name):
    a = np.asarray(value, dtype=float)
    if a.shape != (size,) or not np.all(np.isfinite(a)):
        raise ValueError(f"{name} must be finite with shape ({size},)")
    return tuple(float(x) for x in a)


def _rotation(value):
    r = np.asarray(value, dtype=float).reshape(3, 3)
    if not np.all(np.isfinite(r)) or not np.allclose(r.T @ r, np.eye(3), atol=1e-9, rtol=0.) or abs(np.linalg.det(r)-1) > 1e-9:
        raise ValueError("task orientation must be a proper rotation")
    return tuple(float(x) for x in r.flat)


@dataclass(frozen=True)
class TaskPoint:
    point_id: str
    arm: str
    frame: str
    position_m: tuple[float, ...]
    rotation: tuple[float, ...]
    time_s: float
    time_window_s: tuple[float, float]
    position_tolerance_m: float
    orientation_tolerance_rad: float
    kind: str = "waypoint"

    def __post_init__(self):
        if not self.point_id or self.arm not in ("rigid", "continuum") or self.frame not in ("world", "target") or self.kind not in ("waypoint", "terminal"):
            raise ValueError("invalid task point identity/frame/kind")
        object.__setattr__(self, "position_m", _vector(self.position_m, 3, "position_m"))
        object.__setattr__(self, "rotation", _rotation(self.rotation))
        object.__setattr__(self, "time_window_s", _vector(self.time_window_s, 2, "time_window_s"))
        values = (self.time_s, self.position_tolerance_m, self.orientation_tolerance_rad)
        if not all(np.isfinite(x) for x in values) or self.position_tolerance_m <= 0 or not 0 < self.orientation_tolerance_rad <= np.pi:
            raise ValueError("invalid task point time/tolerances")
        if not 0 <= self.time_window_s[0] <= self.time_s <= self.time_window_s[1]:
            raise ValueError("nominal task point time must lie within its window")

    def to_dict(self):
        return {"point_id": self.point_id, "arm": self.arm, "frame": self.frame,
                "position_m": list(self.position_m), "rotation": list(self.rotation),
                "time_s": self.time_s, "time_window_s": list(self.time_window_s),
                "position_tolerance_m": self.position_tolerance_m,
                "orientation_tolerance_rad": self.orientation_tolerance_rad, "kind": self.kind}

    @classmethod
    def from_dict(cls, value):
        return cls(**value)


@dataclass(frozen=True)
class TaskSpec:
    task_id: str
    group_id: str
    split: str
    family: str
    seed: int
    scenario_json: str
    initial_qpos: tuple[float, ...]
    initial_qvel: tuple[float, ...]
    initial_planner_q: tuple[float, ...]
    initial_planner_dq: tuple[float, ...]
    base_pose: tuple[float, ...]
    base_twist: tuple[float, ...]
    target_pose: tuple[float, ...]
    target_twist: tuple[float, ...]
    requirements: tuple[TaskPoint, ...]
    model_contract_sha256: str
    path_freedom: str = "free_intermediate_path_between_fixed_requirements"
    layout_json: str = "{}"
    duration_s: float = 27.
    task_period_s: float = .020
    physics_period_s: float = .002

    def __post_init__(self):
        if not self.task_id or not self.group_id or self.split not in SPLITS or self.family not in FAMILIES:
            raise ValueError("invalid task identity/family/split")
        if not isinstance(self.seed, int) or isinstance(self.seed, bool) or self.seed < 0:
            raise ValueError("task seed must be a nonnegative integer")
        if (self.duration_s, self.task_period_s, self.physics_period_s) != (27., .020, .002):
            raise ValueError("V6.4-A requires a 27 s, 50/500 Hz simulation protocol")
        if not re.fullmatch("[0-9a-f]{64}", self.model_contract_sha256):
            raise ValueError("model contract SHA256 missing")
        scenario = json.loads(self.scenario_json)
        if not isinstance(scenario, dict) or not isinstance(scenario.get("workspace_obstacles"), list):
            raise ValueError("task scenario and obstacle declarations missing")
        object.__setattr__(self, "scenario_json", canonical_json(scenario))
        layout=json.loads(self.layout_json)
        if not isinstance(layout,dict):
            raise ValueError('task layout provenance must be an object')
        object.__setattr__(self, 'layout_json', canonical_json(layout))
        for field, size in (("initial_qpos",81), ("initial_qvel",79), ("initial_planner_q",17),
                            ("initial_planner_dq",17), ("base_pose",7), ("base_twist",6),
                            ("target_pose",7), ("target_twist",6)):
            object.__setattr__(self, field, _vector(getattr(self, field), size, field))
        for name in ("base_pose", "target_pose"):
            if abs(np.linalg.norm(getattr(self, name)[3:])-1.) > 1e-9:
                raise ValueError("initial pose quaternion must be normalized")
        points = tuple(self.requirements)
        if not points or not all(isinstance(p, TaskPoint) for p in points) or len({p.point_id for p in points}) != len(points):
            raise ValueError("task requirements must be nonempty unique TaskPoints")
        if any(p.time_window_s[1] > self.duration_s for p in points):
            raise ValueError("task requirement exceeds declared horizon")
        if {p.arm for p in points if p.kind == "terminal"} != {"rigid", "continuum"}:
            raise ValueError("both arms need independent terminal requirements")
        if any(p.time_s != self.duration_s or p.time_window_s[1] != self.duration_s for p in points if p.kind == "terminal"):
            raise ValueError("terminal requirements must name the declared final time")
        object.__setattr__(self, "requirements", points)

    @property
    def scenario(self):
        return json.loads(self.scenario_json)

    def to_dict(self):
        return {"schema": TASK_SCHEMA, "task_id":self.task_id, "group_id":self.group_id,
                "split":self.split, "family":self.family, "seed":self.seed, "scenario":self.scenario,
                **{k:list(getattr(self,k)) for k in ("initial_qpos","initial_qvel","initial_planner_q",
                   "initial_planner_dq","base_pose","base_twist","target_pose","target_twist")},
                "requirements":[p.to_dict() for p in self.requirements],
                "model_contract_sha256":self.model_contract_sha256, "path_freedom":self.path_freedom,
                "layout":json.loads(self.layout_json),
                "duration_s":self.duration_s, "task_period_s":self.task_period_s,
                "physics_period_s":self.physics_period_s, "twist_convention":"world_linear_then_angular"}

    def sha256(self):
        return hashlib.sha256(canonical_json(self.to_dict()).encode("utf-8")).hexdigest()

    @classmethod
    def from_dict(cls, value):
        value = dict(value)
        if value.pop("schema", TASK_SCHEMA) != TASK_SCHEMA or value.pop("twist_convention", "world_linear_then_angular") != "world_linear_then_angular":
            raise ValueError("unsupported task schema/twist convention")
        value["scenario_json"] = canonical_json(value.pop("scenario"))
        value['layout_json']=canonical_json(value.pop('layout',{}))
        value["requirements"] = tuple(TaskPoint.from_dict(p) for p in value["requirements"])
        return cls(**value)


def validate_task_splits(tasks):
    """All variants, windows and routes from one source task stay in one split."""
    tasks = tuple(tasks)
    if not tasks or len({t.task_id for t in tasks}) != len(tasks):
        raise ValueError("tasks must be nonempty with unique IDs")
    groups = {}
    initial_conditions = {}
    for t in tasks:
        if groups.setdefault(t.group_id, t.split) != t.split:
            raise ValueError("source task group leaks across splits")
        # Same source scene/initial state cannot silently receive a new group ID.
        source_scene=t.scenario
        source_scene.pop('scenario_id',None)
        source_key = canonical_json({"scenario":source_scene,"qpos":t.initial_qpos,"qvel":t.initial_qvel})
        if initial_conditions.setdefault(source_key, t.split) != t.split:
            raise ValueError("identical source task leaks across splits")
    return {split:[t.task_id for t in tasks if t.split == split] for split in SPLITS}


def task_from_scenario(spec, scenario, *, task_id, group_id, family, split, requirements=None, layout_diagnostics=None):
    """Freeze declared initial state using named joints; no future replay input."""
    import mujoco
    from model_test.whole_body_verifier_v5 import WholeBodyCollisionVerifier, WholeBodyVerificationConfig
    from v6_lite.hierarchical_qp import free_joint_slices, joint_addresses
    verifier = WholeBodyCollisionVerifier(spec, scenario.obstacles, WholeBodyVerificationConfig(
        minimum_clearance=.005, query_distance_max=2.5, adaptive_subdivisions=4,
        self_collision_ancestor_exclusion_depth=3, include_target_satellite_pairs=True))
    model = verifier.model
    data = mujoco.MjData(model)
    qp, qv = joint_addresses(model, spec)
    bp, bv = free_joint_slices(model, spec.base_joint_name)
    tp, tv = free_joint_slices(model, spec.target_free_joint_name)
    data.qpos[qp] = spec.encode_position(spec.planner_zero)
    data.qvel[:] = 0.
    data.qpos[tp.start:tp.start+3] += scenario.target_satellite_position_shift_m
    data.qvel[tv.start:tv.start+3] = scenario.target_satellite_linear_velocity_m_s
    data.qvel[tv.start+3:tv.stop] = scenario.target_satellite_angular_velocity_rad_s
    mujoco.mj_forward(model, data)
    twists=[]
    for joint in (spec.base_joint_name,spec.target_free_joint_name):
        jid=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,joint)
        velocity=np.zeros(6)
        mujoco.mj_objectVelocity(model,data,mujoco.mjtObj.mjOBJ_BODY,int(model.jnt_bodyid[jid]),velocity,0)
        twists.append(tuple(np.r_[velocity[3:],velocity[:3]]))
    if requirements is None:
        target=scenario.continuum_target
        arrival=np.r_[target.path_start_s,target.path_start_s+np.cumsum(target.segment_durations_s)]
        requirements=[]
        for i,(point,t) in enumerate(zip(target.waypoint_points_w,arrival)):
            terminal=i==len(arrival)-1
            time=27. if terminal else float(t)
            requirements.append(TaskPoint(f"continuum_{i:02d}","continuum","world",tuple(point),
                tuple(scenario.continuum_target_rotation_world.flat),time,
                (25.5,27.) if terminal else (max(0.,time-.35),min(27.,time+.35)),
                .00018 if terminal else .002,np.deg2rad(.25),"terminal" if terminal else "waypoint"))
        requirements.append(TaskPoint("rigid_terminal","rigid","target",tuple(scenario.grasp_point_target_frame_m),
            tuple(scenario.grasp_rotation_target_frame.flat),27.,(25.5,27.),.00010,np.deg2rad(.25),"terminal"))
    return TaskSpec(task_id,group_id,split,family,scenario.seed,canonical_json(scenario.to_dict()),
        tuple(data.qpos),tuple(data.qvel),tuple(spec.decode_position(data.qpos[qp])),
        tuple(spec.decode_velocity(data.qvel[qv])),tuple(data.qpos[bp]),twists[0],
        tuple(data.qpos[tp]),twists[1],tuple(requirements),spec.runtime_contract_sha256(),
        layout_json=canonical_json(layout_diagnostics or {}))


def _mid_arm_layout(spec, scenario):
    """Locate the actual middle module and preserve a signed safe initial gap."""
    import mujoco
    from model_test.whole_body_verifier_v5 import WholeBodyCollisionVerifier,WholeBodyVerificationConfig,WorkspaceSphere
    from v6_lite.hierarchical_qp import joint_addresses,free_joint_slices
    verifier=WholeBodyCollisionVerifier(spec,scenario.obstacles,WholeBodyVerificationConfig(
        minimum_clearance=.005,query_distance_max=2.5,adaptive_subdivisions=1,
        self_collision_ancestor_exclusion_depth=3,include_target_satellite_pairs=True))
    model=verifier.model;data=mujoco.MjData(model)
    qp,_=joint_addresses(model,spec);tp,tv=free_joint_slices(model,spec.target_free_joint_name)
    data.qpos[qp]=spec.encode_position(spec.planner_zero)
    data.qpos[tp.start:tp.start+3]+=scenario.target_satellite_position_shift_m
    data.qvel[tv.start:tv.start+3]=scenario.target_satellite_linear_velocity_m_s
    data.qvel[tv.start+3:tv.stop]=scenario.target_satellite_angular_velocity_rad_s
    mujoco.mj_forward(model,data)
    joint_name=spec.low_level_joint_names[29]  # second scalar joint of module 15 / 30.
    jid=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,joint_name)
    body_id=int(model.jnt_bodyid[jid])
    continuum_geoms={p.geom_a for p in verifier.pairs if p.pair_class=='continuum_target'}
    middle_geoms=[i for i in continuum_geoms if int(model.geom_bodyid[i])==body_id]
    if not middle_geoms:
        raise ValueError('declared middle module has no retained collision geometry')
    middle_point=np.asarray(data.xpos[body_id]).copy()
    middle_rotation=np.asarray(data.xmat[body_id]).reshape(3,3).copy()
    obstacle_id=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_GEOM,
        f'v5_workspace_sphere_001_{scenario.obstacles[1].name}')
    if obstacle_id<0:
        raise ValueError('middle-arm obstacle mapping missing')
    selected=None
    for offset in (.075,.090,.110,.140):
        for normal in (middle_rotation[:,1],-middle_rotation[:,1],middle_rotation[:,2],-middle_rotation[:,2]):
            center=middle_point+offset*normal
            model.geom_pos[obstacle_id]=center
            report=verifier.verify_qpos_sequence(np.asarray(data.qpos)[None,:]).to_dict()
            if report['feasible']:
                mujoco.mj_forward(model,data)
                distances=[float(mujoco.mj_geomDistance(model,data,i,obstacle_id,2.5,np.zeros(6))) for i in middle_geoms]
                if min(distances)>=.005:
                    selected=(center.copy(),normal.copy(),report,distances);break
        if selected is not None:break
    if selected is None:
        raise ValueError('cannot freeze an initially safe physical middle-arm sphere layout')
    center,normal,report,distances=selected
    spheres=list(scenario.obstacles)
    spheres[1]=WorkspaceSphere(spheres[1].name,center,spheres[1].radius)
    return replace(scenario,obstacles=tuple(spheres)),{
        'definition':'actual_module_15_collision_body_sphere_offset_at_initial_state',
        'module_index_1_based':15,'module_count':30,'joint_name':joint_name,
        'body_name':mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_BODY,body_id),
        'geom_names':[mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_GEOM,i) for i in middle_geoms],
        'body_position_world_m':middle_point.tolist(),'offset_normal_world':normal.tolist(),
        'sphere_center_world_m':center.tolist(),'sphere_radius_m':float(spheres[1].radius),
        'initial_native_middle_geom_clearance_m':distances,
        'initial_whole_body_minimum_clearance_m':report['minimum_clearance'],
        'initial_state_geometry_feasible':True,'trajectory_feasibility_established':False}


def generate_task_suite(spec, counts=(2,1,1), seed=20261002):
    """Small actual task groups in all three families; sizes remain configurable.

    The task source is a new seeded scene, not a view/window of an old trace.
    Family layouts alter existing sphere locations without changing their radii.
    These are proposed tasks, not certified feasible demonstrations.
    """
    from model_test.whole_body_verifier_v5 import WorkspaceSphere
    from v6_lite.run_v6_lite import V6LiteRunConfig, build_scenarios
    if len(counts)!=3 or any(not isinstance(x,int) or isinstance(x,bool) or x<1 for x in counts):
        raise ValueError("declare positive train/val/test task counts per family")
    scenes=build_scenarios(spec,V6LiteRunConfig(scenario_count=3*sum(counts),seed=seed))
    tasks=[]; index=0
    for family in FAMILIES:
        for split,count in zip(SPLITS,counts):
            for local in range(count):
                scene=scenes[index]
                spheres=list(scene.obstacles)
                # Frozen, reproducible layout variations. Feasibility is measured
                # later by the teacher/closed loop, never asserted by this factory.
                shift={"end_effector_detour":(0.,-.020,.010),
                       "mid_arm_detour":(0.,0.,0.),
                       "multiple_routes":(.010,-.010,.025)}[family]
                spheres[1]=WorkspaceSphere(spheres[1].name,np.asarray(spheres[1].center)+shift,spheres[1].radius)
                sid=f"v6_4_{family}_{split}_{local:03d}"
                scene=replace(scene,scenario_id=sid,obstacles=tuple(spheres))
                layout={'definition':'independent_seeded_end_effector_sphere_layout',
                        'trajectory_feasibility_established':False,
                        'intermediate_route_side_is_not_a_task_requirement':True}
                if family=='mid_arm_detour':
                    scene,layout=_mid_arm_layout(spec,scene)
                if family=='multiple_routes':
                    layout['definition']='two_sided_route_choice_about_frozen_workspace_spheres'
                    layout['route_side_labels']=['positive_lateral','negative_lateral']
                    layout['route_difference_measured_by_teacher_and_actual_execution']=True
                tasks.append(task_from_scenario(spec,scene,task_id=sid,group_id=sid,family=family,split=split,layout_diagnostics=layout))
                index+=1
    validate_task_splits(tasks)
    return tuple(tasks)


def freeze_task_suite(tasks, output: Path):
    output=Path(output)
    if output.exists():
        raise FileExistsError("task protocol output already exists")
    splits=validate_task_splits(tasks)
    payload={"schema":TASK_SCHEMA,"tasks":[t.to_dict() for t in tasks],
             "task_hashes":{t.task_id:t.sha256() for t in tasks},"splits":splits,
             "family_counts":{f:sum(t.family==f for t in tasks) for f in FAMILIES},
             "declared_not_feasibility_certified":True}
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(payload,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    return payload
