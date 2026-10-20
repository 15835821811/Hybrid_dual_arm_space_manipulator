"""Additional saved-record diagnostics; no physics or optimization calls."""
import json
from collections import Counter
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from ff_prior_sample_alignment.tools.closeout import require_terminal, write, OUT
from ff_prior_endpoint_jacobian.statistics import distribution

require_terminal()
def read(p):
    return json.loads(p.read_text(encoding="utf-8"))

answer = {"application": {}, "sampling_pairing": {}, "budget": {}, "world_events": {}}
for directory in sorted((OUT / "application").glob("*/*")):
    records = [read(p) for p in sorted((directory / "candidates").glob("*.json"))]
    if not records:
        continue
    details, fallbacks, solvers = [], 0, []
    for r in records:
        v = r["validation"]
        geom = v["geometry"]
        details.append({
            "candidate_id": r["candidate_id"], "eligible": r["eligible"],
            "frozen_goal_pass": r["frozen_goal_pass"],
            "failed_gates": [k for k, ok in v["gates"].items() if not ok],
            "minimum_clearance_mm": 1000 * geom["minimum_clearance_m"],
            "geometry_step_mm": 1000 * geom["maximum_geometry_step_m"],
            "worst_pair": geom["coarse"]["pair"], "wall_s": r["wall_s"],
            "calls": r["model_calls"]["coupled"],
        })
        for guide in r.get("sampling", {}).get("guidance", []):
            for event in guide.get("history", []):
                solver = event.get("solver", {})
                solvers.append(solver)
                fallbacks += bool(solver.get("fallback_goal_only"))
    answer["application"]["/".join(directory.parts[-2:])] = {
        "candidate_details": details, "guidance_solver_records": len(solvers),
        "goal_only_fallbacks": fallbacks,
        "solver_messages": dict(Counter(s.get("message", "none") for s in solvers)),
    }
for sj in range(2):
    pairs = [(read(OUT/"unguided"/str(sj)/"plain"/f"{j:02d}.json"),
              read(OUT/"unguided"/str(sj)/"aligned"/f"{j:02d}.json")) for j in range(48)]
    answer["sampling_pairing"][str(sj)] = dict(Counter(
        f"plain{int(a['eligible'])}_aligned{int(b['eligible'])}" for a, b in pairs))
paths = list((OUT/"candidates").glob("*/*/*/*.json"))
paths += list((OUT/"application").glob("*/*/candidates/*.json"))
records = [read(p) for p in paths]
answer["budget"] = {
    "new_candidates": len(records),
    "model_validation_levels": dict(Counter(r["validation"]["validation_level"] for r in records)),
    "all_gate_eligible": sum(r["eligible"] for r in records),
    "maximum_candidate_wall_s": max(r["wall_s"] for r in records),
    "candidate_wall_above_15s": sum(r["wall_s"] > 15 for r in records),
    "maximum_candidate_coupled": max(r["model_calls"]["coupled"] for r in records),
    "maximum_reported_integration_intervals": max(r.get("integration_intervals", 0) for r in records),
}
gradients = [read(p) for p in (OUT/"learning/aligned/updates").glob("*/gradient.json")]
answer["world_events"] = {
    "count": len(gradients),
    "status": dict(Counter(r["status"] for r in gradients)),
    "half_relative_error": distribution([h["relative_error"] for r in gradients for h in r["half_checks"]]),
    "prepared_branch_max_rad": max(r["preparation"]["differentiable_prepare_max_rad"] for r in gradients),
    "shrink": dict(Counter(str(r["preparation"]["coefficient_shrink"]) for r in gradients)),
    "finite": all(np.isfinite(r["true_loss"]) for r in gradients),
}
write("additional_diagnostics.json", answer)
print(json.dumps(answer, ensure_ascii=False))
