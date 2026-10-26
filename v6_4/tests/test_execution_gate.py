"""Neither an engineering label nor a raw label may bypass a failed gate."""
import unittest

from v6_4.run_planning import gate_allows_execution


class ExecutionGateTests(unittest.TestCase):
    def test_failed_full_gate_cannot_execute_in_either_mode(self):
        for raw in (False, True):
            for required in (False, True):
                self.assertFalse(gate_allows_execution({'passed': False, 'raw_passed': raw},
                                                       require_raw_gate=required))

    def test_repaired_pass_needs_explicit_engineering_mode(self):
        repaired = {'passed': True, 'raw_passed': False}
        self.assertFalse(gate_allows_execution(repaired))
        self.assertTrue(gate_allows_execution(repaired, require_raw_gate=False))

    def test_valid_direct_raw_pass_remains_usable(self):
        self.assertTrue(gate_allows_execution({'passed': True, 'raw_passed': True}))


if __name__ == '__main__':
    unittest.main()
