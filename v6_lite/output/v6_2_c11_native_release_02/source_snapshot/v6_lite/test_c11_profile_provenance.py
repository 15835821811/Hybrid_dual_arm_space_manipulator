"""Profiles must remain bound to the source and logs actually exercised."""

import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest import mock

from v6_lite import run_test_profiles as profiles


class ProfileProvenanceTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="c11_provenance_test_")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name) / "repo"
        (self.root / "v6_lite").mkdir(parents=True)
        (self.root / "model_test").mkdir()
        self.original = b"value = 1\r\n"
        (self.root / "v6_lite/a.py").write_bytes(self.original)
        (self.root / "model_test/b.py").write_bytes(b"value = 2\n")
        self.git("init", "--quiet")
        self.git("add", ".")
        self.commit("initial fixture")

    def git(self, *args):
        return subprocess.check_output(["git", *args], cwd=self.root, stderr=subprocess.PIPE).decode().strip()

    def commit(self, message):
        return self.git("-c", "user.name=Profile Provenance Test", "-c",
                        "user.email=profile-provenance@example.invalid", "-c",
                        "commit.gpgsign=false", "commit", "--quiet", "-m", message)

    def run_fixture(self, action):
        # Execute an actual selected test, while isolating discovery and immutable
        # historical IDs from this deliberately tiny temporary Git repository.
        output = self.root / "profile_output"
        suite = unittest.TestSuite([unittest.FunctionTestCase(action)])
        stdout = io.StringIO()
        with mock.patch.object(profiles, "__file__", str(self.root / "v6_lite/run_test_profiles.py")), \
                mock.patch.object(profiles, "historical_ids", return_value=[]), \
                mock.patch.object(profiles.unittest.defaultTestLoader, "discover", return_value=suite), \
                contextlib.redirect_stdout(stdout):
            passed = profiles.run("current", output)
        self.last_stdout = json.loads(stdout.getvalue())
        return passed, json.loads((output / "report.json").read_text(encoding="utf-8")), output

    def test_snapshot_records_raw_and_normalized_hashes_and_excludes_generated_sources(self):
        for name in ("output", "__pycache__", "cache", ".pytest_cache"):
            generated = self.root / "v6_lite" / name
            generated.mkdir()
            (generated / "ignored.py").write_text("generated = True\n", encoding="utf-8")
        before = profiles.current_source_snapshot(self.root)
        self.assertEqual(set(before["files"]), {"v6_lite/a.py", "model_test/b.py"})
        record = before["files"]["v6_lite/a.py"]
        self.assertEqual(record["sha256_raw"], hashlib.sha256(self.original).hexdigest())
        self.assertEqual(record["sha256_lf_normalized"], hashlib.sha256(b"value = 1\n").hexdigest())
        self.assertEqual(before["git_commit"], self.git("rev-parse", "HEAD"))
        self.assertTrue(before["worktree_dirty"])
        self.assertFalse(before["tracked_worktree_dirty"])
        (self.root / "v6_lite/a.py").write_bytes(b"value = 3\n")
        after = profiles.current_source_snapshot(self.root)
        self.assertTrue(after["tracked_worktree_dirty"])
        self.assertEqual(profiles.source_provenance(before, after)["source_changes"]["modified"],
                         ["v6_lite/a.py"])

    def test_successful_profile_binds_unchanged_source_and_exact_log(self):
        passed, report, output = self.run_fixture(lambda: None)
        self.assertTrue(passed)
        self.assertTrue(report["current"]["passed"])
        self.assertEqual(self.last_stdout["current"]["tests"], 1)
        self.assertEqual(self.last_stdout["report"], (output / "report.json").as_posix())
        self.assertNotIn("source_provenance", self.last_stdout)
        provenance = report["source_provenance"]
        self.assertTrue(provenance["source_unchanged"])
        self.assertEqual(provenance["before"]["git_commit"], self.git("rev-parse", "HEAD"))
        self.assertEqual(provenance["before"]["files"], provenance["after"]["files"])
        self.assertFalse(provenance["historical_expected_hashes_derived_from_current_source"])
        log = (output / report["current"]["log_file"]).read_bytes()
        self.assertEqual(report["current"]["log_sha256_raw"], hashlib.sha256(log).hexdigest())
        self.assertEqual(report["current"]["log_size_bytes"], len(log))
        self.assertGreater(report["current"]["elapsed_seconds"], 0)
        self.assertGreaterEqual(report["elapsed_seconds"], report["current"]["elapsed_seconds"])

    def test_passing_tests_cannot_hide_added_removed_or_modified_source(self):
        def mutate_sources():
            (self.root / "v6_lite/a.py").write_bytes(b"value = 3\n")
            (self.root / "model_test/b.py").unlink()
            (self.root / "model_test/new.py").write_bytes(b"value = 4\n")

        passed, report, _ = self.run_fixture(mutate_sources)
        self.assertTrue(report["current"]["passed"])
        self.assertFalse(passed)
        self.assertFalse(report["passed"])
        self.assertEqual(report["source_provenance"]["source_changes"], {
            "added": ["model_test/new.py"], "removed": ["model_test/b.py"],
            "modified": ["v6_lite/a.py"],
        })

    def test_commit_change_fails_even_when_python_source_is_identical(self):
        def change_commit():
            (self.root / "notes.txt").write_text("new commit\n", encoding="utf-8")
            self.git("add", "notes.txt")
            self.commit("changed identity during tests")

        passed, report, _ = self.run_fixture(change_commit)
        self.assertTrue(report["current"]["passed"])
        self.assertFalse(passed)
        self.assertFalse(report["source_provenance"]["git_commit_unchanged"])
        self.assertEqual(report["source_provenance"]["before"]["files"],
                         report["source_provenance"]["after"]["files"])

    def test_restored_source_bytes_still_record_a_write_during_tests(self):
        def change_then_restore():
            path = self.root / "v6_lite/a.py"
            original_time = path.stat().st_mtime_ns
            path.write_bytes(b"value = 9\n")
            path.write_bytes(self.original)
            os.utime(path, ns=(original_time, original_time + 1_000_000_000))

        passed, report, _ = self.run_fixture(change_then_restore)
        self.assertTrue(report["current"]["passed"])
        self.assertFalse(passed)
        provenance = report["source_provenance"]
        self.assertEqual(provenance["before"]["files"]["v6_lite/a.py"]["sha256_raw"],
                         provenance["after"]["files"]["v6_lite/a.py"]["sha256_raw"])
        self.assertEqual(provenance["source_changes"]["modified"], ["v6_lite/a.py"])

    def test_inventory_does_not_follow_a_link_to_another_source_tree(self):
        target = Path(self.temporary.name) / "external"
        target.mkdir()
        (target / "must_not_be_inventory.py").write_text("outside = True\n", encoding="utf-8")
        link = self.root / "v6_lite" / "fixture_link"
        if os.name == "nt":
            def quote(path):
                return "'" + str(path).replace("'", "''") + "'"
            subprocess.run(["powershell", "-NoProfile", "-Command",
                            "New-Item -ItemType Junction -Path " + quote(link) +
                            " -Target " + quote(target)], check=True, capture_output=True)
            self.assertTrue(link.is_junction())
        else:
            link.symlink_to(target, target_is_directory=True)
        snapshot = profiles.current_source_snapshot(self.root)
        self.assertEqual(set(snapshot["files"]), {"v6_lite/a.py", "model_test/b.py"})
        self.assertEqual(snapshot["capture_errors"], [])


if __name__ == "__main__":
    unittest.main()
