"""Bookkeeping only: no simulator model, propagation, or native steps."""
import tempfile
import time
import unittest
from pathlib import Path
from .native_budget import NativeBudget,NativeBudgetError
from ff_prior.storage import read_json


class NativeLimits(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.out=Path(self.tmp.name)
        self.b=NativeBudget(self.out,time.time()+60,steps_each=2,total_cap=4)

    def tearDown(self):self.tmp.cleanup()

    def complete_execution(self):
        self.b.begin_phase('execution',2)
        for _ in range(2):self.b.before_step('execution');self.b.after_step('execution')
        self.b.finish_phase('execution')

    def test_reservation_is_not_execution(self):
        self.b.begin_phase('execution',2)
        x=read_json(self.out/'native_budget.json')
        self.assertEqual(x['reserved_upper_bound']['execution'],2)
        self.assertEqual(x['completed_observed']['execution'],0)

    def test_phase_cap_before_extra_attempt(self):
        self.complete_execution()
        with self.assertRaises(NativeBudgetError):self.b.before_step('execution')
        self.assertEqual(self.b.attempted['execution'],2)

    def test_full_execution_and_replay_counts(self):
        self.complete_execution();self.b.begin_phase('replay',2)
        for _ in range(2):self.b.before_step('replay');self.b.after_step('replay')
        self.b.finish_phase('replay')
        self.assertEqual(sum(self.b.completed.values()),4)

    def test_unknown_attempt_blocks_replay(self):
        self.b.begin_phase('execution',2);self.b.before_step('execution');self.b.finish_phase('execution')
        with self.assertRaises(NativeBudgetError):self.b.begin_phase('replay',1)
        self.assertIsNotNone(read_json(self.out/'native_budget.json')['pending'])

    def test_cannot_reopen_saved_budget(self):
        with self.assertRaises(NativeBudgetError):NativeBudget(self.out,time.time()+60)

    def test_deadline_refuses_before_attempt(self):
        self.b.begin_phase('execution',2);self.b.deadline=0
        with self.assertRaises(NativeBudgetError):self.b.before_step('execution')
        self.assertEqual(self.b.attempted['execution'],0)

    def test_no_phase_reentry(self):
        self.complete_execution()
        with self.assertRaises(NativeBudgetError):self.b.begin_phase('execution',2)

    def test_replay_not_longer_than_actual(self):
        self.b.begin_phase('execution',2);self.b.before_step('execution');self.b.after_step('execution');self.b.finish_phase('execution')
        with self.assertRaises(NativeBudgetError):self.b.begin_phase('replay',2)


if __name__=='__main__':unittest.main()
