"""Regressions for the V6-lite target-satellite collision pair policy."""

from __future__ import annotations

import unittest

import mujoco
import numpy as np

from model_test.whole_body_verifier_v5 import (
    WholeBodyCollisionVerifier,
    WholeBodyVerificationConfig,
)
from v6_lite.run_v6_lite import default_v6_lite_robot_spec


class TargetSatelliteCollisionPolicyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.spec = default_v6_lite_robot_spec()
        cls.verifier = WholeBodyCollisionVerifier(
            cls.spec,
            (),
            WholeBodyVerificationConfig(
                minimum_clearance=0.005,
                adaptive_subdivisions=1,
                include_target_satellite_pairs=True,
            ),
        )

    def test_target_pairs_are_opt_in_to_preserve_shared_v5_policy(self) -> None:
        verifier = WholeBodyCollisionVerifier(
            self.spec,
            (),
            WholeBodyVerificationConfig(adaptive_subdivisions=1),
        )
        target_classes = {"continuum_target", "rigid_target", "base_target"}
        self.assertTrue(target_classes.isdisjoint(
            pair.pair_class for pair in verifier.pairs
        ))
        self.assertEqual(len(verifier.pairs), 2698)
        self.assertEqual(
            verifier._pair_policy_sha256,
            "44082350db74a23abde75f46230016f23272d8f60abcb055449d8e60a734a6ec",
        )

    def test_all_continuum_geoms_and_noncontact_rigid_geoms_pair_with_target(
        self,
    ) -> None:
        continuum_bodies = self.verifier._descendant_body_ids(
            self.verifier._CONTINUUM_ROOT,
            self.verifier._CONTINUUM_TIP,
        )
        rigid_bodies = self.verifier._descendant_body_ids(
            self.verifier._RIGID_ROOT,
            self.verifier._RIGID_TIP,
        )
        continuum_geoms = set(self.verifier._collision_geoms(continuum_bodies))
        rigid_geoms = set(self.verifier._collision_geoms(rigid_bodies))
        base_geoms = set(
            self.verifier._collision_geoms(
                {
                    self.verifier._body_id(name)
                    for name in self.verifier._BASE_BODIES
                }
            )
        )

        continuum_pairs = {
            pair.geom_a: pair
            for pair in self.verifier.pairs
            if pair.pair_class == "continuum_target"
        }
        rigid_pairs = {
            pair.geom_a: pair
            for pair in self.verifier.pairs
            if pair.pair_class == "rigid_target"
        }
        base_pairs = {
            pair.geom_a: pair
            for pair in self.verifier.pairs
            if pair.pair_class == "base_target"
        }
        target_geom = self.verifier._geom_id("target_satellite_collision")
        intentional_contacts = {
            self.verifier._geom_id("collision_0072"),
            self.verifier._geom_id("collision_0073"),
        }

        self.assertEqual(set(continuum_pairs), continuum_geoms)
        self.assertEqual(set(rigid_pairs), rigid_geoms - intentional_contacts)
        self.assertEqual(set(base_pairs), base_geoms)
        self.assertTrue(intentional_contacts.isdisjoint(rigid_pairs))
        self.assertTrue(
            all(pair.geom_b == target_geom for pair in continuum_pairs.values())
        )
        self.assertTrue(all(pair.geom_b == target_geom for pair in rigid_pairs.values()))
        self.assertTrue(all(pair.geom_b == target_geom for pair in base_pairs.values()))

    def test_target_penetration_is_rejected_without_contact_response(self) -> None:
        model = self.verifier.model
        data = mujoco.MjData(model)
        data.qpos[:] = model.qpos0
        mujoco.mj_forward(model, data)

        distal_geom = self.verifier._geom_id("collision_0063")
        target_joint = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_JOINT,
            self.spec.target_free_joint_name,
        )
        target_qpos_address = int(model.jnt_qposadr[target_joint])
        data.qpos[target_qpos_address : target_qpos_address + 3] = np.asarray(
            data.geom_xpos[distal_geom], dtype=np.float64
        )

        # Verification is signed-distance based and does not inspect the
        # contact list, so disabling collision response cannot hide overlap.
        original_contype = np.asarray(model.geom_contype).copy()
        original_conaffinity = np.asarray(model.geom_conaffinity).copy()
        try:
            model.geom_contype[:] = 0
            model.geom_conaffinity[:] = 0
            report = self.verifier.verify_qpos_sequence(data.qpos[None, :])
        finally:
            model.geom_contype[:] = original_contype
            model.geom_conaffinity[:] = original_conaffinity

        self.assertFalse(report.feasible)
        self.assertLess(report.minimum_by_class["continuum_target"], 0.0)
        self.assertTrue(report.uses_mj_geom_distance)
        self.assertFalse(report.uses_contact_list)

    def test_pair_ids_are_resolved_from_names_for_the_destination_model(self) -> None:
        remapped = self.verifier.pairs_for_model(self.verifier.model)
        self.assertEqual(len(remapped), len(self.verifier.pairs))
        for pair in remapped:
            self.assertEqual(
                pair.geom_a,
                mujoco.mj_name2id(
                    self.verifier.model,
                    mujoco.mjtObj.mjOBJ_GEOM,
                    pair.geom_a_name,
                ),
            )
            self.assertEqual(
                pair.geom_b,
                mujoco.mj_name2id(
                    self.verifier.model,
                    mujoco.mjtObj.mjOBJ_GEOM,
                    pair.geom_b_name,
                ),
            )

    def test_nonterminal_geom_cannot_be_whitelisted_for_target_contact(self) -> None:
        with self.assertRaisesRegex(ValueError, "non-terminal grasp-assembly body"):
            WholeBodyCollisionVerifier(
                self.spec,
                (),
                WholeBodyVerificationConfig(
                    adaptive_subdivisions=1,
                    include_target_satellite_pairs=True,
                    intentional_target_contact_geom_names=("collision_0071",),
                ),
            )


if __name__ == "__main__":
    unittest.main()
