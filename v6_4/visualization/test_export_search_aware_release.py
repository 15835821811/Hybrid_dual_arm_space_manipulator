"""Exporter finality and idempotency audits on tiny stdlib mock evidence.

No scientific run is exported. Retained-release tests mock portable-byte
verification to isolate the source/report freshness contract.
"""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from . import export_search_aware_release as exporter


class ReleaseFinalityTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(); self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name); self.run = self.root / "mock_run"; self.release = self.root / "mock_release"
        self.run.mkdir(); self.release.mkdir()
        self.identity = {"source_sha256": {}, "protected_artifacts": {}, "algorithm_producer_commit": "mock",
                         "base_publication_commit": "mock-base"}
        self.save("source_identity.json", self.identity)
        self.portable_check = patch.object(exporter.PortableResolver, "verify_all", return_value={"mock": True}).start()
        self.addCleanup(patch.stopall)

    def save(self, relative, value):
        path = self.run / relative; path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, allow_nan=False), encoding="utf8")
        return path

    def numerical_complete(self):
        self.save("plan.json", {"mock": True})
        self.save("validation/protocol_validation.json", {"all_terminal": True, "actual_logical_slots": {"val": 20, "test": 40}})
        for model in ("D", "S"):
            hashes = {}
            for update in (250, 4000):
                path = self.run / "models" / model / f"checkpoint_{update:04d}.pt"
                path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(b"mock; never loaded")
                hashes[str(update)] = exporter.sha(path)
            self.save(f"models/{model}/training_report.json", {"training_executed": True,
                "optimizer_updates_total": 4000, "checkpoint_sha256": hashes})
        self.save("model_freeze.json", {"artifacts": {}})
        self.save("model_selection.json", {"mock": True})
        self.save("actual_complete.json", {"mock": True})
        for phase in ("closed_loop_val", "frozen_test"):
            self.save(phase + "/sealed_selections/all_selections.json", {"files": {}})

    def complete_report(self):
        summary = {"status": {key: True for key in ("research_execution_completed", "independent_test_completed",
            "closed_loop_val_completed", "training_completed_D", "training_completed_S")},
            "evidence_verification": {"complete": True}}
        self.save("summary.json", summary)
        (self.run / "REPORT.md").write_text("Synthetic report; no experiment executed.\n", encoding="utf8")
        for name in exporter.REPORT_TABLES:
            path = self.run / "result_tables" / name; path.parent.mkdir(exist_ok=True)
            path.write_text("synthetic mock\n", encoding="utf8")
        self.reseal_report()

    def reseal_report(self):
        tables = self.run / "result_tables"
        self.save("result_tables/report_artifact_identity.json", {
            "schema": exporter.REPORT_SCHEMA,
            "reporter_source_sha256": exporter.sha(Path(exporter.__file__).with_name("report_search_aware_warmstart.py")),
            "root_reports": {name: exporter.sha(self.run / name) for name in ("REPORT.md", "summary.json")},
            "derived_artifacts": {p.relative_to(self.run).as_posix(): exporter.sha(p) for p in tables.iterdir()
                                  if p.name != "report_artifact_identity.json"},
            "input_files": {str((self.run / name).resolve()): exporter.sha(self.run / name) for name in exporter.REPORT_INPUTS}})

    def retained(self, stage):
        inventory = exporter._run_inventory(self.run)
        value = {"schema": exporter.RELEASE_SCHEMA, "stage": stage, "source_run": str(self.run),
            "source_identity_sha256": exporter.sha(self.run / "source_identity.json"), "inventory": inventory,
            "source_run_inventory_sha256": exporter._inventory_sha(inventory),
            "report_artifact_identity_sha256": exporter.sha(self.run / "result_tables/report_artifact_identity.json") if stage == "final" else None}
        exporter.immutable_json(self.release / "release_manifest.json", value)
        exporter.immutable_json(self.release / "portable_paths.json", {})
        return value

    def test_partial_cannot_be_promoted_in_place_even_before_final_inputs_exist(self):
        self.retained("partial")
        with self.assertRaisesRegex(ValueError, "partial release cannot be reused as final; use a fresh destination"):
            exporter.export_release(self.run, self.release, require_complete=True)
        self.portable_check.assert_not_called()

    def test_unchanged_partial_reuse_is_explicitly_partial(self):
        value = self.retained("partial")
        self.assertEqual(exporter.export_release(self.run, self.release, require_complete=False), value)
        self.portable_check.assert_called_once()

    def test_unchanged_final_requires_current_complete_report(self):
        self.numerical_complete(); self.complete_report(); value = self.retained("final")
        self.assertEqual(exporter.export_release(self.run, self.release), value)
        self.portable_check.assert_called_once()

    def test_completed_numerics_without_report_cannot_export_final(self):
        self.numerical_complete()
        with self.assertRaisesRegex(ValueError, "current complete report and provenance"):
            exporter.export_release(self.run, self.release)
        self.assertFalse((self.release / "release_manifest.json").exists())

    def test_incomplete_report_status_or_unverified_evidence_cannot_be_final(self):
        self.numerical_complete(); self.complete_report()
        for key in ("research_execution_completed", "independent_test_completed", "closed_loop_val_completed", "training_completed_D", "training_completed_S"):
            summary = exporter.read(self.run / "summary.json"); summary["status"][key] = False
            self.save("summary.json", summary); self.reseal_report()
            with self.assertRaisesRegex(ValueError, "report completion backed by verified current evidence"):
                exporter.verify_inputs(self.run)
            summary["status"][key] = True; self.save("summary.json", summary)
        summary["evidence_verification"]["complete"] = False
        self.save("summary.json", summary); self.reseal_report()
        with self.assertRaises(ValueError): exporter.verify_inputs(self.run)

    def test_stale_root_report_and_derived_table_are_rejected(self):
        self.numerical_complete(); self.complete_report(); self.retained("final")
        original = (self.run / "REPORT.md").read_text()
        (self.run / "REPORT.md").write_text("changed report")
        with self.assertRaisesRegex(ValueError, "root report changed"):
            exporter.export_release(self.run, self.release)
        (self.run / "REPORT.md").write_text(original)
        (self.run / "result_tables/endpoint_summary.csv").write_text("changed derived data")
        with self.assertRaisesRegex(ValueError, "derived report artifact changed"):
            exporter.export_release(self.run, self.release)

    def test_report_input_hash_and_required_input_coverage_are_enforced(self):
        self.numerical_complete(); self.complete_report()
        validation = exporter.read(self.run / "validation/protocol_validation.json"); validation["extra"] = 1
        self.save("validation/protocol_validation.json", validation)
        with self.assertRaisesRegex(ValueError, "report scientific input changed"):
            exporter.verify_inputs(self.run)
        self.reseal_report()
        path = self.run / "result_tables/report_artifact_identity.json"; identity = exporter.read(path)
        identity["input_files"].pop(str((self.run / "actual_complete.json").resolve()))
        self.save("result_tables/report_artifact_identity.json", identity)
        with self.assertRaisesRegex(ValueError, "required final scientific input identities"):
            exporter.verify_inputs(self.run)

    def test_new_valid_report_cannot_make_existing_final_snapshot_current(self):
        self.numerical_complete(); self.complete_report(); self.retained("final")
        (self.run / "REPORT.md").write_text("A newly generated valid synthetic report.\n")
        self.reseal_report()
        with self.assertRaisesRegex(ValueError, "source-run inventory is stale; use a fresh destination"):
            exporter.export_release(self.run, self.release)

    def test_added_deleted_and_changed_omitted_source_files_invalidate_reuse(self):
        self.numerical_complete(); self.complete_report()
        raw = self.run / "private_prediction/traces/mock.npz"; raw.parent.mkdir(parents=True); raw.write_bytes(b"mock0")
        self.retained("final")
        for value in (b"mock1", None):
            if value is None: raw.unlink()
            else: raw.write_bytes(value)
            with self.assertRaisesRegex(ValueError, "source-run inventory is stale"):
                exporter.export_release(self.run, self.release)
        raw.write_bytes(b"mock0"); (self.run / "later.md").write_text("new file")
        with self.assertRaisesRegex(ValueError, "source-run inventory is stale"):
            exporter.export_release(self.run, self.release)

    def test_allow_partial_does_not_weaken_a_retained_final(self):
        self.numerical_complete(); self.complete_report(); self.retained("final")
        (self.run / "REPORT.md").unlink()
        with self.assertRaisesRegex(ValueError, "current complete report and provenance"):
            exporter.export_release(self.run, self.release, require_complete=False)

    def test_declared_derived_inventory_must_match_current_tables(self):
        self.numerical_complete(); self.complete_report()
        (self.run / "result_tables/undeclared.csv").write_text("new derived artifact")
        with self.assertRaisesRegex(ValueError, "artifact inventory is incomplete or stale"):
            exporter.verify_inputs(self.run)


class PortableCoverageTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(); self.addCleanup(self.temporary.cleanup)
        self.release = Path(self.temporary.name) / "release"; self.release.mkdir()
        self.original = "E:/unavailable_original_run/source_identity.json"
        source = self.release / "frozen_source/pkg/producer.py"; source.parent.mkdir(parents=True)
        source.write_bytes(b"# synthetic producer\n")
        self.source_sha = exporter.sha(source)
        identity = self.release / "snapshot/source_identity.json"; identity.parent.mkdir()
        identity.write_text(json.dumps({"source_sha256": {"pkg/producer.py": self.source_sha}}), encoding="utf8")
        inventory = [{"path": "source_identity.json", "source_path": self.original,
            "sha256": exporter.sha(identity), "size": identity.stat().st_size, "copied": True,
            "availability": "portable_bytes", "reason": "portable_scientific_receipt"}]
        self.value = {"schema": exporter.RELEASE_SCHEMA, "inventory": inventory,
                      "source_run_inventory_sha256": exporter._inventory_sha(inventory)}
        self.mapping = {self.original: {"available": True, "path": "snapshot/source_identity.json",
            "sha256": exporter.sha(identity), "size": identity.stat().st_size},
            "E:/unavailable_repository/pkg/producer.py": {"available": True,
                "path": "frozen_source/pkg/producer.py", "sha256": self.source_sha, "size": source.stat().st_size}}
        self.save("release_manifest.json", self.value)
        self.save("portable_paths.json", self.mapping)
        self.save("frozen_source_manifest.json", {"source_sha256": {"pkg/producer.py": self.source_sha}})

    def save(self, name, value):
        (self.release / name).write_text(json.dumps(value), encoding="utf8")

    def test_valid_portable_coverage_requires_no_original_drive_access(self):
        self.assertTrue(exporter.PortableResolver(self.release).verify_all()["portable_bytes_verified"])

    def test_missing_or_wrong_copied_mapping_rejected(self):
        self.mapping.pop(self.original); self.save("portable_paths.json", self.mapping)
        with self.assertRaises(ValueError): exporter.PortableResolver(self.release).verify_all()

    def test_missing_or_truncated_frozen_source_manifest_rejected(self):
        (self.release / "frozen_source_manifest.json").unlink()
        with self.assertRaisesRegex(ValueError, "source manifest missing"):
            exporter.PortableResolver(self.release).verify_all()
        self.save("frozen_source_manifest.json", {"source_sha256": {}})
        with self.assertRaisesRegex(ValueError, "snapshotted execution identity"):
            exporter.PortableResolver(self.release).verify_all()

    def test_frozen_source_cannot_be_marked_unavailable(self):
        self.mapping["E:/unavailable_repository/pkg/producer.py"]["available"] = False
        self.save("portable_paths.json", self.mapping)
        with self.assertRaisesRegex(ValueError, "frozen producer source mapping changed"):
            exporter.PortableResolver(self.release).verify_all()


if __name__ == "__main__":
    unittest.main()
