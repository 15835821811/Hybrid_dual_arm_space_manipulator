"""Timeline accounting and startup tails must remain auditable."""

import unittest

from v6_lite.runtime_timing import CycleTimeline, latency_summary


class TimingTests(unittest.TestCase):
    def test_phases_cover_dispatch_without_overlap(self):
        timeline = CycleTimeline(7, 0.14)
        for name in ("refresh", "solve", "dispatch"):
            timeline.mark(name)
        record = timeline.record()
        self.assertAlmostEqual(sum(p["wall_s"] for p in record["phases"]),
                               record["dispatch_latency_s"])
        for left, right in zip(record["phases"], record["phases"][1:]):
            self.assertEqual(left["end_ns"], right["start_ns"])

    def test_startup_and_consecutive_misses_are_retained(self):
        summary = latency_summary([.08, .01, .021, .022, .01])
        self.assertEqual(summary["first_cycle_ms"], 80)
        self.assertEqual(summary["max_ms"], 80)
        self.assertEqual(summary["over_deadline_count"], 3)
        self.assertEqual(summary["longest_consecutive_over_deadline"], 2)
        self.assertFalse(summary["passed"])
        self.assertEqual(summary["steady_after_first_cycle"]["count"], 4)
        self.assertEqual(latency_summary([.01] * 1350)["count"], 1350)
