"""C.2 accounting and sealing tests; no candidate physics."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from v6_4.evaluate_preference_warmstart import _first_match, task_rows, make_stream, verify_selections
from v6_4.route_optimizer_protocol import write, read, sha


def row(I=.1, L=1., d=.031, good=True):
    return {"prediction_admissible": good, "prediction_metrics": {"I_support": I, "L_full": L, "d_support": d}}


class PipelineTests(unittest.TestCase):
    def test_first_match_keeps_missing_oracle_and_censored_separate(self):
        oracle = {"I_support": .1, "L_full": 1.}
        self.assertEqual(_first_match([], None, "A")["status"], "N/A_R12_NO_PLAN")
        result = _first_match([row(good=False), row(I=.102)], oracle, "A")
        self.assertIsNone(result["slot"])
        self.assertEqual(result["right_censored_budget"], 2)
        self.assertEqual(_first_match([row(good=False), row(I=.1009, L=1.004)], oracle, "A")["slot"], 2)

    def test_B_near_length_never_relaxes_clearance(self):
        oracle = {"I_support": .1, "L_full": 1.}
        self.assertIsNone(_first_match([row(d=.029999)], oracle, "B")["slot"])
        self.assertEqual(_first_match([row(d=.030)], oracle, "B")["slot"], 1)

    def test_external_learning_split_drives_task_selection(self):
        with tempfile.TemporaryDirectory() as tmp:
            write(Path(tmp) / "plan.json", {"tasks": [
                {"task_id": "old", "split": "test", "learning_split": "train"},
                {"task_id": "new", "split": "test"}]})
            self.assertEqual([t["task_id"] for t in task_rows(tmp, "train")], ["old"])
            self.assertEqual([t["task_id"] for t in task_rows(tmp, "test")], ["new"])

    def test_streams_share_configuration_bytes_but_no_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ("plan.json", "source_identity.json", "frozen_execution_config.json", "frozen_run_config.json"):
                write(root / name, {"name": name})
            frozen = {"task_id": "test"}
            first, second = make_stream(root, frozen, "N"), make_stream(root, frozen, "D")
            self.assertNotEqual(first, second)
            self.assertEqual(sha(first / "source_identity.json"), sha(second / "source_identity.json"))
            write(first / "private_result.json", {"quality": 123})
            self.assertFalse((second / "private_result.json").exists())

    def test_selection_seal_detects_change_before_any_execution(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write(root / "model_freeze.json", {"checkpoint": "one"})
            write(root / "selected.json", {"plan": "first"})
            write(root / "sealed_selections" / "all_selections.json", {
                "model_freeze_sha256": sha(root / "model_freeze.json"),
                "files": {"selected.json": sha(root / "selected.json")}})
            with patch("v6_4.preference_warmstart_protocol.verify_frozen", return_value={}), patch(
                    "v6_4.evaluate_preference_warmstart.verify_model_freeze", return_value={}):
                verify_selections(root)
                (root / "selected.json").write_text('{"plan":"changed"}')
                with self.assertRaises(ValueError):
                    verify_selections(root)


if __name__ == "__main__":
    unittest.main()
