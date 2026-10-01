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
import unittest

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
    ids = historical_ids(root)
    report = {"profile": profile, "historical_test_ids": ids, "passed": True,
              "historical_goldens_modified": False, "branch_name_used_for_selection": False}
    if profile in ("current", "all"):
        discovered = unittest.defaultTestLoader.discover(str(root / "v6_lite"), "test_*.py", str(root))
        tests = list(flatten(discovered))
        if not set(ids).issubset({t.id() for t in tests}):
            raise ValueError("declared historical tests disappeared from discovery")
        current = unittest.TestSuite(t for t in tests if t.id() not in ids)
        with (output / "current_tests.log").open("w", encoding="utf-8") as stream:
            result = unittest.TextTestRunner(stream=stream, verbosity=2).run(current)
        report["current"] = {"tests": result.testsRun, "failures": len(result.failures),
            "errors": len(result.errors), "passed": result.wasSuccessful()}
        report["passed"] &= result.wasSuccessful()
    if profile in ("historical", "all"):
        # Keep snapshot paths below Windows MAX_PATH for unchanged historical
        # tests. The immutable source manifest is also saved with the report.
        frozen_root = Path(tempfile.mkdtemp(prefix="c11_frozen_"))
        manifest = snapshot(root, frozen_root)
        (output / "frozen_source_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        completed = subprocess.run([sys.executable, "-m", "unittest", "-v", *ids],
            cwd=frozen_root, capture_output=True, text=True, encoding="utf-8", errors="replace")
        (output / "historical_tests.log").write_text(completed.stdout + completed.stderr, encoding="utf-8")
        report["historical"] = {"tests_selected": len(ids), "exit_code": completed.returncode,
            "snapshot": frozen_root.as_posix(), "passed": completed.returncode == 0}
        report["passed"] &= completed.returncode == 0
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report["passed"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("current", "historical", "all"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    raise SystemExit(0 if run(args.profile, args.output_dir) else 1)
