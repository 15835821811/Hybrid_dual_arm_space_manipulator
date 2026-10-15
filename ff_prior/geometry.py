"""Explicit topology-based pair policy, with no legacy grasp exemption.

The policy excludes only rigidly welded geometry, direct articulated neighbours
(after collapsing fixed links), and the explicitly enumerated adjacent two-axis
continuum modules. Every exclusion is materialized with names and a reason.
Distances never use the sparse contact list. Physical contacts remain enabled.
"""
from __future__ import annotations

import mujoco
import numpy as np
from .storage import digest


class CollisionBackend:
    def __init__(self, robot):
        self.robot, self.model = robot, robot.model
        if mujoco.__version__ != "3.3.2":
            raise RuntimeError("distance backend is audited only with pinned MuJoCo 3.3.2")
        m = self.model
        self.geoms = [i for i in range(m.ngeom) if m.geom_contype[i] or m.geom_conaffinity[i]]
        self.pairs, self.exclusions = [], []
        module_neighbours = {frozenset(("base_link", "link_1"))}
        module_neighbours.update(frozenset((f"link_{i}", f"link_{i+1}")) for i in range(1, 30))
        for index, a in enumerate(self.geoms):
            ba = int(m.geom_bodyid[a])
            wa = int(m.body_weldid[ba])
            for b in self.geoms[index + 1:]:
                bb = int(m.geom_bodyid[b])
                wb = int(m.body_weldid[bb])
                names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, v) for v in (ba, bb)]
                reason = None
                if wa == wb:
                    reason = "same_rigid_assembly"
                elif m.body_weldid[m.body_parentid[wa]] == wb or m.body_weldid[m.body_parentid[wb]] == wa:
                    # World is never an articulated neighbour of the robot.
                    if wa != 0 and wb != 0:
                        reason = "direct_articulated_connection"
                if frozenset(names) in module_neighbours:
                    reason = "adjacent_two_axis_continuum_modules"
                record = self.describe(a, b)
                if reason:
                    self.exclusions.append(dict(record, reason=reason))
                else:
                    self.pairs.append((a, b))
        self.pairs = np.asarray(self.pairs, dtype=int).reshape(-1, 2)
        self.policy = {"version": "explicit_topology_v1", "pairs": [self.describe(*p) for p in self.pairs],
                       "exclusions": self.exclusions, "intentional_contact_geom_allowlist": [],
                       "target_pairs_enabled": True, "continuous_time_certified": False}
        self.policy_hash = digest(self.policy)

    def describe(self, a, b):
        m = self.model
        return {"geom_a": mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, int(a)),
                "geom_b": mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, int(b)),
                "body_a": mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, int(m.geom_bodyid[a])),
                "body_b": mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, int(m.geom_bodyid[b]))}

    def scan(self, data, threshold=0.0, exhaustive=False):
        m = self.model
        a, b = self.pairs.T
        # Bounding spheres supply a conservative lower bound, used only to
        # prune pairs that cannot beat the current exact minimum.
        lower = np.linalg.norm(data.geom_xpos[a] - data.geom_xpos[b], axis=1) - m.geom_rbound[a] - m.geom_rbound[b]
        order = np.argsort(lower)
        minimum, pair, queries, violations = float("inf"), None, 0, []
        for k in order:
            if not exhaustive and lower[k] > max(minimum, threshold):
                break
            ga, gb = self.pairs[k]
            distance = float(mujoco.mj_geomDistance(m, data, int(ga), int(gb), 2.5, None))
            queries += 1
            if distance < minimum:
                minimum, pair = distance, self.describe(ga, gb)
            if distance < threshold:
                violations.append(dict(self.describe(ga, gb), distance_m=distance))
        self.robot.calls["distance"] += queries
        return {"minimum_clearance_m": minimum, "minimum_pair": pair,
                "passed": minimum >= threshold, "threshold_m": threshold,
                "violations": violations, "query_count": queries, "pair_count": len(self.pairs),
                "query_distance_max_m": 2.5, "minimum_is_capped": minimum >= 2.5,
                "pair_policy_hash": self.policy_hash, "continuous_time_certified": False}
