"""Software gate boundary checks. These are not robot experiment results."""
from .gates import engineering_gate


def main():
    good = dict(completed=180, eligible=144, selected_outcomes=36, execution_verified=29, candidate_errors=0)
    assert engineering_gate(good, 180, 36, True, [])["status"] == "PASS"
    assert engineering_gate(dict(good, eligible=143), 180, 36, True, [])["status"] == "BLOCKED"
    assert engineering_gate(dict(good, execution_verified=28), 180, 36, True, [])["status"] == "BLOCKED"
    assert engineering_gate(dict(good, selected_outcomes=35), 180, 36, True, [])["status"] == "BLOCKED"
    assert engineering_gate(good, 180, 36, False, [])["status"] == "BLOCKED"
    assert engineering_gate(good, 180, 36, True, ["missing trace"])["status"] == "BLOCKED"
    assert engineering_gate(dict(good, candidate_errors=1), 180, 36, True, [])["status"] == "BLOCKED"
    try:
        engineering_gate({}, 180, 36, True, [])
    except ValueError:
        pass
    else:
        raise AssertionError("missing evidence incorrectly accepted")
    print("8 software policy checks PASS; no physical experiment performed")


if __name__ == "__main__":
    main()
