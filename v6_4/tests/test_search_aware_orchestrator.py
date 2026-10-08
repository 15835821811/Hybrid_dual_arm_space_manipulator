"""C.3 declarations, bounds and nonphysical orchestration checks."""
from pathlib import Path

import pytest

from v6_4 import search_aware_warmstart_experiment as c3


def split_fixture():
    rows = []
    for split, groups in (("train", 3), ("val", 1), ("test", 2)):
        for group in range(groups):
            for side in ("plus", "minus"):
                mother = f"{split}_{group}"
                rows.append({"task_sha256": mother + side, "mother_id": mother,
                    "mother_source_sha256": mother + "sha", "split": split,
                    "historical": split == "train"})
    return {"tasks": rows}


def test_exact_split_and_mother_isolation():
    manifest = split_fixture()
    assert len(c3.validate_splits(manifest)) == 12
    manifest["tasks"][-1]["mother_source_sha256"] = manifest["tasks"][0]["mother_source_sha256"]
    with pytest.raises(ValueError, match="leakage"):
        c3.validate_splits(manifest)


def test_budget_resume_does_not_grant_more_work(tmp_path):
    ledger = c3.BudgetLedger(tmp_path)
    first = ledger.reserve("teacher_candidate_slots", "pair1", 8, {"frozen": True})
    assert ledger.reserve("teacher_candidate_slots", "pair1", 8, {"frozen": True}) == first
    assert ledger.totals()["candidate_slots_total"] == 8
    with pytest.raises(ValueError, match="changed"):
        ledger.reserve("teacher_candidate_slots", "pair1", 9, {"frozen": True})
    for index in range(1, 12):
        ledger.reserve("teacher_candidate_slots", "pair" + str(index + 1), 8)
    with pytest.raises(ValueError, match="exhausted"):
        ledger.reserve("teacher_candidate_slots", "extra", 1)


def test_new_stage_budgets_and_fixed_ddim():
    assert sum(c3.LIMITS[k] for k in ("teacher_candidate_slots", "val_candidate_slots", "test_candidate_slots")) == 328
    assert c3.LIMITS["val_actual_slots"] + c3.LIMITS["test_actual_slots"] == 60
    assert sum(c3.LIMITS[k] for k in c3.LIMITS if "ddim" in k) == 32
    assert c3.CHECKPOINTS == ("D250", "D4000", "S250", "S4000")


def test_stream_directory_isolation():
    frozen = {"task_id": "t"}
    paths = {c3.stream_path(Path("run"), frozen, method, "test") for method in c3.METHODS}
    paths.update(c3.stream_path(Path("run"), frozen, pair, "teacher") for pair in ("T_local", "T_transfer"))
    assert len(paths) == 6
    assert c3.stream_path(Path("run"), frozen, "D250", "val") not in paths


def test_test_search_requires_freeze_before_initialization(tmp_path, monkeypatch):
    monkeypatch.setattr(c3, "verify_run", lambda run: {})
    called = []
    monkeypatch.setattr(c3, "_initializers", lambda *a: called.append(a))
    with pytest.raises(FileNotFoundError):
        c3.test_search(tmp_path)
    assert not called
