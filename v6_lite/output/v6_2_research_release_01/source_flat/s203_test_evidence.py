"""Ensure published numbers are generated from intact historical JSON."""

from __future__ import annotations

import unittest

from v6_lite.generate_evidence import ROOT, check_historical_artifacts, render_markdown


class EvidenceTests(unittest.TestCase):
    def test_historical_integrity_is_not_labeled_replay(self):
        result = check_historical_artifacts()
        self.assertTrue(result["passed"])
        self.assertFalse(result["native_mujoco_replay_performed"])
        self.assertEqual(result["evidence_type"], "historical_artifact_integrity_only")

    def test_generated_markdown_matches_artifacts(self):
        expected = render_markdown()
        actual = (ROOT / "docs" / "V6_2A_EVIDENCE.md").read_text(encoding="utf-8")
        self.assertEqual(actual, expected)
        self.assertIn("0.004418652734", actual)


if __name__ == "__main__":
    unittest.main()
