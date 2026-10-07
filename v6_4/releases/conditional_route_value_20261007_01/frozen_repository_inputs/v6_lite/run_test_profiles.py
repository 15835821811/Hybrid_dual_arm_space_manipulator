"""Separate current tests from reconstructed immutable historical evidence.

The historical goldens and tests are unchanged. Sources come from fixed Git
objects, never from the current file's hash. The reconstruction overrides are
the exact A.1 files named by the saved reports. Full profile runs both groups.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from datetime import datetime, timezone

BASE = "fbfa9f62a0855cf19485d5929e90b1c2fc7647d4"
FROZEN = {
    "v6_lite/hierarchical_qp.py": ("9bfd3946729268627778e1a82575610c68147005", "1e2ae02348fadb231827a720b0cdfb1b0fd3aa8393f068d766c7145b91087a65"),
    "v6_lite/run_v6_lite.py": ("1167a6bf255c8b8c9340c78e02f37e8e454566d9", "16106e7e1b0f24718126969f89da9dbaecb39595fc0e418c8390c4fda05a4727"),
    "v6_lite/recompute_execution_constraints.py": ("1167a6bf255c8b8c9340c78e02f37e8e454566d9", "8c5c6d9490ce090bc45af9831d530b3b7ba0eeacbb87d715d79819a53521b4c8"),
    "v6_lite/audit_b2_discrete_servo_counterfactual.py": ("74258ea3b0ead403088284280aee428d145d7822", "5bf0aebc95cb20340a46bcb772909fa0307d695b0f88904b2e4cc515c4018519"),
}
RAW_EOL = {
    "v6_lite/continuum_shape_model.py": "acb5f8dce38e835483f31492b45a7e8181a98469868e4f54ad36e04c49ca8e2c",
    "v6_lite/shape_clearance.py": "e573c59d333ee714ddfb1d5920a87a2fd5de613a4138153aca9ef0a951821fde",
}


def historical_ids(root):
    # Selection itself is frozen; editing a current report cannot hide tests.
    old = json.loads(subprocess.check_output(["git", "show",
        BASE + ":v6_lite/output/v6_2_c1_release/acceptance_summary.json"], cwd=root))
    return ["v6_lite." + x.split("(", 1)[1].rstrip(")") for x in old["regression"]["failed_test_names"]]


def flatten(suite):
    for item in suite:
        if isinstance(item, unittest.TestSuite):
            yield from flatten(item)
        else:
            yield item


def sha(raw):
    return hashlib.sha256(raw.replace(b"\r\n", b"\n")).hexdigest()


def current_source_snapshot(root):
    """Record current inputs for provenance, never historical expectations."""
    errors = []
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root).decode().strip()
        status = subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=all"],
                                         cwd=root).decode("utf-8", errors="replace").splitlines()
        git = {"git_commit": commit, "worktree_dirty": bool(status),
               "tracked_worktree_dirty": any(not line.startswith("??") for line in status),
               "status_porcelain": status}
    except (OSError, subprocess.CalledProcessError) as error:
        git = {"git_commit": None, "worktree_dirty": None,
               "tracked_worktree_dirty": None, "status_porcelain": None}
        errors.append("Git identity unavailable: " + str(error))

    excluded = {"output", "__pycache__", "cache", "caches", ".cache",
                ".pytest_cache", ".mypy_cache", ".ruff_cache", ".git"}
    files = {}

    def linked(path):
        # followlinks=False alone does not exclude Windows junctions.
        return path.is_symlink() or (hasattr(path, "is_junction") and path.is_junction())

    def walk_error(error):
        errors.append("Source inventory unavailable: " + str(error))

    for package in ("v6_lite", "model_test"):
        package_root = root / package
        if not package_root.is_dir() or linked(package_root):
            errors.append("Source package unavailable or linked: " + package)
            continue
        for directory, children, names in os.walk(package_root, followlinks=False, onerror=walk_error):
            children[:] = sorted(name for name in children
                                 if name not in excluded and not linked(Path(directory) / name))
            for name in sorted(names):
                path = Path(directory) / name
                if not name.endswith(".py") or linked(path):
                    continue
                relative = path.relative_to(root).as_posix()
                try:
                    stat_before = path.stat()
                    raw = path.read_bytes()
                    stat_after = path.stat()
                except OSError as error:
                    errors.append("Source unreadable: " + relative + ": " + str(error))
                    continue
                if (stat_before.st_mtime_ns, stat_before.st_size, stat_before.st_ino) != (
                        stat_after.st_mtime_ns, stat_after.st_size, stat_after.st_ino):
                    errors.append("Source changed while capturing: " + relative)
                files[relative] = {"size_bytes": len(raw),
                                   "mtime_ns": stat_after.st_mtime_ns,
                                   "sha256_raw": hashlib.sha256(raw).hexdigest(),
                                   "sha256_lf_normalized": sha(raw)}
    inventory = json.dumps(files, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {"captured_utc": datetime.now(timezone.utc).isoformat(), **git,
            "files": files, "inventory_sha256": hashlib.sha256(inventory).hexdigest(),
            "capture_errors": errors}


def source_provenance(before, after):
    old, new = before["files"], after["files"]
    changes = {"added": sorted(new.keys() - old.keys()),
               "removed": sorted(old.keys() - new.keys()),
               "modified": sorted(name for name in old.keys() & new.keys() if old[name] != new[name])}
    complete = not before["capture_errors"] and not after["capture_errors"]
    same_commit = before["git_commit"] == after["git_commit"]
    unchanged = complete and same_commit and not any(changes.values())
    return {"schema": "c11_current_test_source_provenance_v1", "before": before, "after": after,
            "capture_complete": complete, "git_commit_unchanged": same_commit,
            "source_changes": changes, "source_unchanged": unchanged,
            "historical_expected_hashes_derived_from_current_source": False}


def test_log_record(path, elapsed):
    raw = path.read_bytes()
    return {"elapsed_seconds": elapsed, "log_file": path.name,
            "log_size_bytes": len(raw), "log_sha256_raw": hashlib.sha256(raw).hexdigest()}


def snapshot(root, destination):
    destination.mkdir(parents=True, exist_ok=True)
    names = subprocess.check_output(["git", "ls-tree", "-r", "--name-only", BASE], cwd=root).decode().splitlines()
    manifest = {"base_commit": BASE, "source_overrides": FROZEN, "files": {}}
    for name in names:
        if not name.endswith(".py") or "/output/" in name:
            continue
        commit = FROZEN.get(name, (BASE, None))[0]
        raw = subprocess.check_output(["git", "show", f"{commit}:{name}"], cwd=root)
        expected = FROZEN.get(name, (BASE, None))[1]
        if expected is not None and sha(raw) != expected:
            raise ValueError(f"frozen Git object no longer matches golden: {name}")
        if name in RAW_EOL:
            raw = raw.replace(b"\r\n", b"\n").replace(b"\n", b"\r\n")
            if hashlib.sha256(raw).hexdigest() != RAW_EOL[name]:
                raise ValueError(f"historical raw CRLF golden changed: {name}")
        path = destination / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
        manifest["files"][name] = {"commit": commit, "sha256_lf_normalized": sha(raw),
            "sha256_raw": hashlib.sha256(raw).hexdigest(), "historical_raw_eol": name in RAW_EOL}
    link = destination / "v6_lite/output"
    target = (root / "v6_lite/output").resolve(strict=True)
    try:
        os.symlink(target, link, target_is_directory=True)
    except OSError:
        def quote(p):
            return "'" + str(p).replace("'", "''") + "'"
        subprocess.run(["pwsh", "-NoProfile", "-Command",
            "New-Item -ItemType Junction -Path " + quote(link) + " -Target " + quote(target)],
            check=True, capture_output=True)
    (destination / "frozen_source_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def run(profile, output):
    root = Path(__file__).resolve().parent.parent
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    before = current_source_snapshot(root)
    ids = historical_ids(root)
    report = {"profile": profile, "historical_test_ids": ids, "passed": True,
              "historical_goldens_modified": False, "branch_name_used_for_selection": False}
    if profile in ("current", "all"):
        discovered = unittest.defaultTestLoader.discover(str(root / "v6_lite"), "test_*.py", str(root))
        tests = list(flatten(discovered))
        if not set(ids).issubset({t.id() for t in tests}):
            raise ValueError("declared historical tests disappeared from discovery")
        current = unittest.TestSuite(t for t in tests if t.id() not in ids)
        log = output / "current_tests.log"
        test_started = time.perf_counter()
        with log.open("w", encoding="utf-8") as stream:
            result = unittest.TextTestRunner(stream=stream, verbosity=2).run(current)
        report["current"] = {"tests": result.testsRun, "failures": len(result.failures),
            "errors": len(result.errors), "passed": result.wasSuccessful(),
            **test_log_record(log, time.perf_counter() - test_started)}
        report["passed"] &= result.wasSuccessful()
    if profile in ("historical", "all"):
        # Keep snapshot paths below Windows MAX_PATH for unchanged historical
        # tests. The immutable source manifest is also saved with the report.
        frozen_root = Path(tempfile.mkdtemp(prefix="c11_frozen_"))
        manifest = snapshot(root, frozen_root)
        (output / "frozen_source_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        test_started = time.perf_counter()
        completed = subprocess.run([sys.executable, "-m", "unittest", "-v", *ids],
            cwd=frozen_root, capture_output=True, text=True, encoding="utf-8", errors="replace")
        elapsed = time.perf_counter() - test_started
        log = output / "historical_tests.log"
        log.write_text(completed.stdout + completed.stderr, encoding="utf-8")
        report["historical"] = {"tests_selected": len(ids), "exit_code": completed.returncode,
            "snapshot": frozen_root.as_posix(), "passed": completed.returncode == 0,
            **test_log_record(log, elapsed)}
        report["passed"] &= completed.returncode == 0
    report["source_provenance"] = source_provenance(before, current_source_snapshot(root))
    report["elapsed_seconds"] = time.perf_counter() - started
    report["passed"] &= report["source_provenance"]["source_unchanged"]
    report_path = output / "report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    summary = {"profile": profile, "passed": report["passed"],
               "git_commit": before["git_commit"],
               "source_unchanged": report["source_provenance"]["source_unchanged"],
               "report": report_path.as_posix()}
    for group, keys in (("current", ("tests", "failures", "errors", "passed")),
                        ("historical", ("tests_selected", "exit_code", "passed"))):
        if group in report:
            summary[group] = {key: report[group][key] for key in keys}
    print(json.dumps(summary, indent=2))
    return report["passed"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("current", "historical", "all"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    raise SystemExit(0 if run(args.profile, args.output_dir) else 1)
