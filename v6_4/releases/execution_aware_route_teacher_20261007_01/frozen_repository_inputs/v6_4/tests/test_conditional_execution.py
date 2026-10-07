import unittest

from v6_4.conditional_execution import ExecutionCostLedger


class CostForwardingTests(unittest.TestCase):
    def test_forwarded_object_arguments_return_and_exception(self):
        ledger = ExecutionCostLedger()
        token, answer = object(), object()
        observed = []
        def original(*args, **kwargs):
            observed.append((args, kwargs))
            return answer
        wrapped = ledger.forward_counted(original, "mj_step")
        with ledger.scope("actual"):
            self.assertIs(wrapped(token, mode=token), answer)
        self.assertIs(observed[0][0][0], token)
        self.assertIs(observed[0][1]["mode"], token)
        self.assertEqual(ledger.physics_steps("actual"), 1)
        error = ValueError("sentinel")
        def raising():
            raise error
        with self.assertRaises(ValueError) as caught:
            ledger.forward_counted(raising, "mj_step2")()
        self.assertIs(caught.exception, error)
        self.assertEqual(ledger.counts[("setup", "mj_step2", "raised")], 1)
        self.assertEqual(ledger.physics_steps("setup"), 0)

    def test_partial_preview_counts_and_restores_outer_scope(self):
        ledger = ExecutionCostLedger()
        step = ledger.forward_counted(lambda: None, "mj_step2")
        def partial():
            for _ in range(4):
                step()
            raise RuntimeError("rejected partial preview")
        with ledger.scope("actual"):
            with self.assertRaises(RuntimeError):
                ledger.forward_preview(partial)()
            self.assertEqual(ledger.phase, "actual")
        self.assertEqual(ledger.preview_records[0]["completed_physics_steps"], 4)
        self.assertEqual(ledger.physics_steps("actual"), 0)
        self.assertEqual(ledger.physics_steps("private_preview"), 4)


if __name__ == "__main__":
    unittest.main()
