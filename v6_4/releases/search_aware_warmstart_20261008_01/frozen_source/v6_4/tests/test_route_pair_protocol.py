"""B.3 invariants: fixed finite budget and no alternate residual interval."""
import unittest
from unittest.mock import patch

import numpy as np

from v6_4.route_pair_protocol import fixed_route_protocol, route_coefficients


class RoutePairProtocolTests(unittest.TestCase):
    def test_inactive_key_interval_is_not_replaced_with_another_active_slot(self):
        definition = {"interval_mask": [True, True, False, True, True, True]}
        with self.assertRaisesRegex(ValueError, "slot 2 is disabled"):
            route_coefficients(definition, "z_plus")
        np.testing.assert_array_equal(route_coefficients(definition, "z0"), np.zeros((6, 2)))

    def test_teacher_modes_use_only_the_predeclared_coordinate_and_amplitude(self):
        definition = {"interval_mask": [True]*6}
        for mode, expected in (("z0", [0., 0.]), ("z_plus", [.012, 0.]),
                               ("z_minus", [-.012, 0.]), ("z_perp", [0., .010])):
            z = route_coefficients(definition, mode)
            np.testing.assert_array_equal(z[2], expected)
            np.testing.assert_array_equal(np.delete(z, 2, axis=0), np.zeros((5, 2)))
            self.assertTrue(np.all(np.linalg.norm(z, axis=1) <= .020))
        with self.assertRaises(ValueError):
            route_coefficients(definition, "retry_alternate_interval")

    def test_protocol_keeps_pairs_whole_and_pilot_outside_formal_counts(self):
        with patch("mujoco.mj_step", side_effect=AssertionError("physics forbidden")):
            plan = fixed_route_protocol()
        self.assertEqual(plan["mother_roles"], ["pilot", "train", "train", "train", "val", "test", "test"])
        self.assertEqual(plan["split_task_counts"], {"train": 6, "val": 2, "test": 4})
        self.assertEqual(sum(plan["split_task_counts"].values()), 12)
        self.assertEqual(plan["budget"]["total_new_actual_slots_max"], 6+32+20)
        self.assertFalse(plan["route_direction_label_is_condition_input"])
        self.assertFalse(plan["position_search_or_failed_task_replacement"])


if __name__ == "__main__":
    unittest.main()
