"""Prospective engineering continuation gates; never relabel individual samples."""
import math


def engineering_gate(metrics, expected_candidates, expected_units, audit_passed, unresolved_errors):
    required = ("completed", "eligible", "selected_outcomes", "execution_verified", "candidate_errors")
    if expected_candidates <= 0 or expected_units <= 0:
        raise ValueError("positive frozen budgets required")
    if any(k not in metrics for k in required):
        raise ValueError("missing raw counts")
    if any(type(metrics[k]) is not int or metrics[k] < 0 for k in required):
        raise ValueError("invalid raw counts")
    if not (metrics["eligible"] <= metrics["completed"] <= expected_candidates):
        raise ValueError("invalid candidate denominator")
    if not (metrics["execution_verified"] <= metrics["selected_outcomes"] <= expected_units):
        raise ValueError("invalid execution denominator")
    if metrics["execution_verified"] > metrics["eligible"]:
        raise ValueError("execution cannot exceed eligible candidates")
    checks = {
        "all_candidates_accounted": metrics["completed"] == expected_candidates,
        "all_preselected_units_accounted": metrics["selected_outcomes"] == expected_units,
        "candidate_qualification": metrics["eligible"] >= math.ceil(.8 * expected_candidates),
        "execution_and_replay": metrics["execution_verified"] >= math.ceil(.8 * expected_units),
        "no_candidate_implementation_errors": metrics["candidate_errors"] == 0,
        "audit_passed": audit_passed is True,
        "no_unresolved_errors": unresolved_errors == [] or unresolved_errors == {},
    }
    return {"status": "PASS" if all(checks.values()) else "BLOCKED",
            "scope": "engineering continuation only; no sample certification or superiority claim",
            "checks": checks, "candidate_denominator": expected_candidates,
            "execution_denominator": expected_units,
            "minimum_eligible": math.ceil(.8 * expected_candidates),
            "minimum_execution_verified": math.ceil(.8 * expected_units)}
