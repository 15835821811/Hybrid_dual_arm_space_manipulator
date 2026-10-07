"""Plot the frozen latest studies without compiling a model or running physics.

The returned JSON tables use every declared case/run.  Display curves may be
decimated, while statistics and scope checks always use full saved samples.
This module intentionally does not reuse historical B.2 visualization output.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


REPO = Path(__file__).resolve().parents[2]
# __file__ is v6_lite/visualization/latest_studies.py.
RUNS = REPO / "v6_lite" / "output" / "runs"
COLORS = {0.95: "#d97706", 1.0: "#2563eb", 1.05: "#08916d"}
SAFE, FAILED, UNKNOWN = "#16836b", "#cc4b42", "#da9b32"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError("latest study evidence: " + message)


def _plain(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, np.ndarray):
        return _plain(value.tolist())
    if isinstance(value, np.generic):
        return _plain(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


class _Evidence:
    def __init__(self) -> None:
        self.refs: set[Path] = set()

    def path(self, name: str) -> Path:
        path = (RUNS / name).resolve()
        _require(path.is_relative_to(RUNS.resolve()) and path.is_file(), name + " missing")
        self.refs.add(path)
        return path

    def json(self, name: str) -> Any:
        return json.loads(self.path(name).read_text(encoding="utf-8"))

    def jsonl(self, name: str) -> list[dict]:
        return [json.loads(line) for line in self.path(name).read_text(encoding="utf-8").splitlines() if line.strip()]

    def npz(self, name: str, keys: tuple[str, ...]) -> dict[str, np.ndarray]:
        with np.load(self.path(name), allow_pickle=False) as arrays:
            _require(all(k in arrays for k in keys), name + " missing array")
            return {key: arrays[key].copy() for key in keys}


def _curve(ax, x, y, *, limit: int = 1800, **kwargs):
    x, y = np.asarray(x), np.asarray(y)
    _require(len(x) == len(y), "curve clock/array length mismatch")
    # Retain endpoints and gaps.  Plotting is display-only; no statistic uses it.
    indices = np.unique(np.linspace(0, len(x) - 1, min(limit, len(x)), dtype=int))
    if np.issubdtype(y.dtype, np.floating):
        invalid = ~np.isfinite(y)
        transitions = np.flatnonzero(invalid[1:] != invalid[:-1])
        indices = np.unique(np.r_[indices, transitions, transitions + 1])
    return ax.plot(x[indices], y[indices], **kwargs)


def _grid(axes) -> None:
    for ax in np.asarray(axes).flat:
        ax.grid(True, alpha=.2, linewidth=.5)
        ax.spines[["top", "right"]].set_visible(False)


def generate_studies(output: Path) -> dict:
    """Generate PNG/SVG study panels and complete finite-population JSON tables.

    Returns artifacts, source_refs, data, coverage and HTML-ready panels.
    Existing filenames are refused; the caller owns the new output directory.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output = Path(output).resolve()
    _require(not Path(__file__).resolve().is_relative_to(output), "output would include helper source")
    folder = output / "studies"
    folder.mkdir(parents=True, exist_ok=True)
    evidence = _Evidence()
    artifacts: list[Path] = []
    panels: list[dict] = []
    coverage: list[dict] = []
    data: dict[str, Any] = {
        "schema": "latest_finite_study_visualization_tables_v1",
        "visualization_is_new_physics": False,
        "scope": "Frozen finite research; old wall/hardware failures and limits retained.",
        "curve_display_decimation_max_regular_samples": 1800,
        "statistics_use_full_saved_samples": True,
    }

    def save(fig, name: str, category: str, title: str, caption: str) -> None:
        fig.suptitle(title, fontsize=15, fontweight="bold", y=.99)
        fig.text(.015, .012, caption, fontsize=8, va="bottom", color="#334155")
        fig.tight_layout(rect=(0, .085, 1, .965), h_pad=1.7, w_pad=1.8)
        for ext in ("png", "svg"):
            path = folder / f"{name}.{ext}"
            _require(not path.exists(), "refuse existing visualization " + str(path))
            fig.savefig(path, dpi=170, facecolor="white")
            artifacts.append(path)
        panels.append({"path": folder / f"{name}.png", "category": category,
                       "title": title, "caption": caption})
        plt.close(fig)

    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 9,
                         "axes.titlesize": 11, "axes.labelsize": 9,
                         "legend.fontsize": 8, "svg.fonttype": "none"}):
        # Conservatism: full original population, then only the declared UNKNOWN subset.
        cp = "research_conservatism_01/"
        cr = evidence.json(cp + "report.json")
        baseline = evidence.jsonl(cp + "baseline.jsonl")
        refinement = evidence.jsonl(cp + "refinement.jsonl")
        selections = evidence.json(cp + "selections.json")
        evidence.json(cp + "plan.json")
        evidence.json(cp + "artifact_manifest.json")
        _require(cr["complete"] and cr["evidence_valid"], "conservatism is not complete")
        _require(len(baseline) == 1024 and {r["index"] for r in baseline} == set(range(1024)), "1024 baseline identities")
        _require(len(refinement) == 126 and len({r["index"] for r in refinement}) == 18, "18 x 7 refinement identities")
        _require(not any(r["actual_query_truncated"] for r in baseline), "actual baseline distance censored")
        states = ["PROXY_CLEARANCE_AT_LEAST_GATE", "UNKNOWN_CROSSES_GATE", "PROXY_CLEARANCE_BELOW_GATE"]
        counts = cr["status_by_actual_gate"]
        fig, axes = plt.subplots(1, 3, figsize=(16, 5.7))
        x = np.arange(3)
        good = [counts[s]["actual_safe"] for s in states]
        bad = [counts[s]["actual_below_gate"] for s in states]
        axes[0].bar(x, good, color=SAFE, label="Actual >= 5 mm")
        axes[0].bar(x, bad, bottom=good, color=FAILED, label="Actual < 5 mm")
        for i, (g, b) in enumerate(zip(good, bad)):
            axes[0].text(i, g / 2, str(g), ha="center", va="center", color="white" if g else "black")
            if b:
                axes[0].text(i, g + b / 2, str(b), ha="center", va="center", color="white")
        axes[0].set(xticks=x, xticklabels=["SAFE", "UNKNOWN", "BELOW"], ylabel="Cases", title="Full 1,024-case population")
        axes[0].legend(loc="upper left")
        for state, color, label in zip(states, [SAFE, UNKNOWN, FAILED], ["SAFE", "UNKNOWN", "BELOW"]):
            rows = [r for r in baseline if r["proxy_status"] == state]
            axes[1].scatter([r["upper_m"] * 1000 for r in rows], [r["actual_mujoco_distance_m"] * 1000 for r in rows], s=12, alpha=.65, color=color, label=label)
        axes[1].axhline(5, color="black", linestyle="--", linewidth=1)
        axes[1].axvline(5, color="black", linestyle="--", linewidth=1)
        axes[1].set(xlabel="PCC upper bound (mm)", ylabel="Direct MuJoCo distance (mm)", title="Proxy and physical collision geometry")
        axes[1].legend()
        axes[2].bar(["Numerical UNKNOWN", "Proxy already BELOW"], [18, 172], color=[UNKNOWN, FAILED])
        for i, value in enumerate([18, 172]):
            axes[2].text(i, value + 3, str(value), ha="center")
        axes[2].set(ylabel="Actually-safe rejected cases", title="190 empirical false rejections")
        _grid(axes)
        save(fig, "conservatism_population", "conservatism", "Conservatism: all 1,024 frozen configurations",
             "Source 2d76555 | B1 budget 31, fixed 5 mm gate and calibrated radii. Distal continuum geometry to target OBB only.\n190 = 172 proxy BELOW + 18 UNKNOWN; false-safe 0. Offline cases are not closed-loop task failures.")

        variants = cr["subset_variants"]
        _require(len(variants) == 7 and all(v["sample_count"] == 18 for v in variants), "seven subset arms")
        labels = [f"B1\n{v['point_budget']}/none" if v["family"] == "B1_distance_bounds" else f"Decision\n{v['point_budget']}/{v['leaf_budget']}" for v in variants]
        fig, axes = plt.subplots(1, 3, figsize=(16, 5.7))
        x = np.arange(7)
        nsafe = [v["status_counts"].get(states[0], 0) for v in variants]
        nunknown = [v["status_counts"].get(states[1], 0) for v in variants]
        axes[0].bar(x, nsafe, color=SAFE, label="SAFE")
        axes[0].bar(x, nunknown, bottom=nsafe, color=UNKNOWN, label="UNKNOWN")
        axes[0].set(xticks=x, xticklabels=labels, ylim=(0, 20), ylabel="Cases / 18", title="No larger-budget classification benefit")
        axes[0].legend()
        axes[1].bar(x - .18, [v["point_count"]["max"] for v in variants], width=.36, label="Max evaluated points", color="#507aab")
        axes[1].bar(x + .18, [v["leaf_count"]["max"] for v in variants], width=.36, label="Max leaves", color="#98a9b7")
        axes[1].set(xticks=x, xticklabels=labels, ylabel="Count", title="Observed use, not configured budget")
        axes[1].legend()
        latency_groups = [[r["elapsed_ms"] for r in refinement if r["family"] == v["family"] and r["point_budget"] == v["point_budget"] and r["leaf_budget"] == v["leaf_budget"]] for v in variants]
        axes[2].boxplot(latency_groups, labels=labels, showfliers=True)
        axes[2].set(ylabel="Offline query wall time (ms)", title="Only the 18 baseline UNKNOWN cases")
        _grid(axes)
        save(fig, "conservatism_subset_budgets", "conservatism", "Finite budget diagnosis: 18 UNKNOWN cases, seven arms",
             "Source 2d76555 | B1: 13 SAFE + 5 UNKNOWN, max 155 points, tolerance stop. Decision 255/128: 18 SAFE, max 133 points / 69 leaves.\nOnline decision over all 1,024 cases was NOT_RUN. Subset timing is not population p95; early-stop gap is not an accuracy curve.")
        data["conservatism"] = {"source_commit": cr["git_commit"], "counts": cr["baseline_counts"], "status_by_actual_gate": counts,
            "selections": selections, "subset_variants": variants,
            "baseline_1024": [{k: r[k] for k in ("index", "lower_m", "upper_m", "gap_m", "proxy_status", "actual_mujoco_distance_m", "evaluation_count", "actual_query_truncated")} for r in baseline],
            "refinement_126": [{k: r.get(k) for k in ("index", "family", "point_budget", "leaf_budget", "proxy_status", "lower_m", "upper_m", "gap_m", "point_count", "leaf_count", "elapsed_ms", "tolerance_met", "budget_exhausted")} for r in refinement],
            "online_full_1024": "NOT_RUN", "closed_loop_claim": False}
        coverage.append({"category": "conservatism", "population": 1024, "refinement_cases": 18, "refinement_queries": 126, "source_commit": cr["git_commit"]})

        # Shape: preserve the fixed 172-case subset and the indicator/exact-distance distinction.
        sp = "research_shape_decomposition_01/"
        sr = evidence.json(sp + "report.json")
        cases = evidence.jsonl(sp + "cases.jsonl")
        evidence.json(sp + "plan.json")
        evidence.json(sp + "artifact_manifest.json")
        _require(sr["complete"] and sr["evidence_valid"] and len(cases) == 172 and len({r["index"] for r in cases}) == 172, "shape 172 complete cases")
        _require(all(not r["actual_query_truncated"] for r in cases), "shape actual distances censored")
        fig, axes = plt.subplots(2, 2, figsize=(14, 9))
        ax = axes[0, 0]
        ax.bar(["Original PCC", "Same-R discrete chain", "Actual chain capsules"], [0, 25, 171], color=SAFE, label="SAFE")
        ax.bar(["Original PCC", "Same-R discrete chain", "Actual chain capsules"], [172, 147, 1], bottom=[0, 25, 171], color=FAILED, label="BELOW")
        ax.set(ylabel="Cases / 172", title="Fixed actual-safe / proxy-BELOW subset")
        ax.legend()
        order = sorted(cases, key=lambda r: r["actual_mujoco_distance_m"])
        xx = np.arange(172)
        for key, label, color in [("actual_mujoco_distance_m", "Direct MuJoCo", "#111827"), ("original_upper_m", "PCC upper bound", FAILED), ("same_radius_discrete_chain_clearance_indicator_m", "Same-R chain indicator", "#4c74bd"), ("capsule_clearance_indicator_m", "Capsule indicator", SAFE)]:
            axes[0, 1].plot(xx, [r[key] * 1000 for r in order], label=label, color=color, linewidth=1.2)
        axes[0, 1].axhline(5, color="black", linestyle="--", linewidth=1)
        axes[0, 1].set(xlabel="Cases sorted by actual distance", ylabel="mm", title="Distance / clearance indicators")
        axes[0, 1].legend()
        margins = np.array([r["minimum_sufficient_radius_margin_m"] * 1000 for r in cases])
        axes[1, 0].hist(margins, bins=18, color="#6374b8", edgecolor="white")
        axes[1, 0].axvline(0, color=FAILED, linestyle="--")
        axes[1, 0].set(xlabel="Original-protocol sufficient radius margin (mm)", ylabel="Cases", title=f"172 positive margins; minimum {margins.min():.3f} mm")
        axes[1, 1].scatter([r["refusal_witness"]["position_discrepancy_norm_m"] * 1000 for r in cases], [r["refusal_witness"]["directional_distance_effect_m"] * 1000 for r in cases], color="#507aab", s=20, alpha=.75)
        axes[1, 1].axhline(0, color="#64748b", linestyle="--")
        axes[1, 1].set(xlabel="PCC / discrete witness position difference (mm)", ylabel="Signed effect on point-to-OBB distance (mm)", title="Position difference is not the distance effect")
        _grid(axes)
        save(fig, "shape_decomposition", "shape", "Shape and envelope diagnosis: all 172 selected configurations",
             "Source 32c3b04 | Fixed original radii and 5 mm gate; 25 same-R classification flips are static cases, not closed-loop successes.\nSegment/capsule negative unsafe indicators are not exact penetration depths. Envelope support is finite and state-local, not a global bound.")
        chosen = [min(cases, key=lambda r: r["minimum_sufficient_radius_margin_m"]), next(r for r in cases if r["index"] == 791)]
        fig = plt.figure(figsize=(14, 6.3))
        for j, row in enumerate(chosen):
            ax = fig.add_subplot(1, 2, j + 1, projection="3d")
            for m in row["chain_modules"]:
                pts = np.array([m["start_world"], m["end_world"]])
                ax.plot(*pts.T, color="#507aab", linewidth=2)
            signs = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)])
            corners = (signs * np.asarray(row["target_half_extents_m"])) @ np.asarray(row["target_rotation"]).T + row["target_center_m"]
            for a in range(8):
                for b in range(a + 1, 8):
                    if np.sum(signs[a] != signs[b]) == 1:
                        ax.plot(*corners[[a, b]].T, color="#64748b", linewidth=1)
            witness = row["refusal_witness"]
            for key, color, label in [("pcc_world", FAILED, "PCC witness"), ("discrete_world", SAFE, "Same-s discrete witness")]:
                point = np.asarray(witness[key])
                ax.scatter(*point, color=color, s=42, label=label)
            ax.set(xlabel="World x (m)", ylabel="World y (m)", zlabel="World z (m)", title=f"Input index {row['index']} | actual {row['actual_mujoco_distance_m']*1000:.3f} mm")
            span = np.ptp(np.r_[corners, np.array([m["start_world"] for m in row["chain_modules"]])], axis=0)
            ax.set_box_aspect(np.maximum(span, .05))
            ax.legend(loc="upper left")
        save(fig, "shape_saved_witnesses", "shape", "Saved geometric witnesses: two declared illustrations",
             "Source 32c3b04 | All-case statistics appear separately. Left: minimum envelope margin; right: the sole capsule BELOW case (index 791).\nLines show saved discrete module centerlines; colored points show saved witnesses. No reconstructed surface or exact penetrating depth is claimed.")
        data["shape"] = {"source_commit": sr["source_commit"], "case_count": 172,
            "same_radius_chain_status_counts": sr["same_radius_chain_status_counts"], "capsule_gate_status_counts": sr["capsule_gate_status_counts"],
            "cases_172": [{k: r[k] for k in ("index", "actual_mujoco_distance_m", "original_lower_m", "original_upper_m", "same_radius_discrete_chain_clearance_indicator_m", "same_radius_discrete_chain_gate_status", "capsule_clearance_indicator_m", "capsule_gate_status", "minimum_sufficient_radius_margin_m", "refusal_witness")} for r in cases],
            "illustrated_indices": [r["index"] for r in chosen], "scope": "Fixed 172 distal continuum-to-OBB configurations; original radii unchanged."}
        coverage.append({"category": "shape", "cases": 172, "source_commit": sr["source_commit"], "illustrations_are_population": False})

        # Nominal traces supply frozen references and a clearly separate comparator.
        nprefix = "research_acceptance_01/"
        nr = evidence.json(nprefix + "report.json")
        nm = evidence.json(nprefix + "simulation/v6_lite_metrics.json")
        evidence.json(nprefix + "manifest.json")
        _require(nr["passed"] and nr["complete"] and len(nm["scenarios"]) == 5, "nominal five-scene completion")
        nominal = {}
        nominal_keys = ("time", "rigid_error", "continuum_error", "continuum_target", "task_time")
        for scene in range(5):
            nominal[scene] = evidence.npz(nprefix + f"simulation/traces/v6_lite_scenario_{scene:02}.npz", nominal_keys)
            _require(len(nominal[scene]["time"]) == 13500 and len(nominal[scene]["task_time"]) == 1350, "nominal scene counts")

        # Pressure: deliberately load both the completed and the partial trace directories.
        vp = "research_velocity_stress_01/"
        vr = evidence.json(vp + "report.json")
        evidence.json(vp + "manifest.json")
        evidence.json(vp + "plan.json")
        evidence.json(vp + "scenario_definitions.json")
        vtr = evidence.json(vp + "simulation/research_trial_report.json")
        scalar = evidence.json("research_velocity_failure_scalar_independent_01/verification.json")
        evidence.json("research_velocity_failure_feasibility_02/report.json")
        _require(not vr["passed"] and not vr["complete"] and vtr["all_predeclared_scenes_attempted"], "pressure failure scope changed")
        _require(scalar["evidence_valid"] and len(scalar["certificates"]) == 3, "three independent scalar certificates")
        stress, scene_rows = {}, []
        for scene in range(5):
            completed = scene in (1, 2)
            tail = f"simulation/traces/v6_lite_scenario_{scene:02}.npz" if completed else f"simulation/failures/v6_lite_scenario_{scene:02}_partial_trace.npz"
            stress[scene] = evidence.npz(vp + tail, ("time", "rigid_error", "continuum_error", "task_time", "task_success", "task_solver_status", "task_iterations", "task_selected_clearance_min_slack_m_s"))
            t = stress[scene]["time"]
            _require(len(t) == {0: 7330, 1: 13500, 2: 13500, 3: 6840, 4: 7160}[scene], "pressure saved length")
            _require(np.all(np.isfinite(t)) and np.all(np.diff(t) > 0), "pressure clock")
            row = {"scene": scene, "status": "COMPLETED" if completed else "REJECTED_PARTIAL", "saved_physics_steps": len(t),
                   "last_saved_time_s": float(t[-1]), "declared_duration_s": 27., "planning_attempts": len(stress[scene]["task_time"]),
                   "trace": str(evidence.path(vp + tail).relative_to(REPO)), "unexecuted_after_rejection": not completed}
            if not completed:
                evidence.json(vp + f"simulation/failures/v6_lite_scenario_{scene:02}_counterexample.json")
                evidence.json(vp + f"simulation/failures/v6_lite_scenario_{scene:02}_execution_failure.json")
            scene_rows.append(row)
        _require(sum(r["saved_physics_steps"] for r in scene_rows) == 48330, "pressure aggregate physics count")
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        axes[0].barh(np.arange(5), [27] * 5, color="#e2e8f0", label="Declared 27 s horizon")
        durations = [r["last_saved_time_s"] for r in scene_rows]
        axes[0].barh(np.arange(5), durations, color=[SAFE if r["status"] == "COMPLETED" else FAILED for r in scene_rows])
        for i, row in enumerate(scene_rows):
            axes[0].text(.3, i, f"{row['status']} | {durations[i]:.2f} s | {row['saved_physics_steps']:,} steps", va="center", color="white", fontsize=9)
        axes[0].set(yticks=range(5), yticklabels=[f"Scene {i:02}" for i in range(5)], xlim=(0, 28), xlabel="Executed simulation time (s)", title="Five attempted; two completed, three rejected")
        axes[0].legend(loc="lower right")
        certs = scalar["certificates"]
        low = [r["tolerance_adjusted_lower"] for r in certs]
        high = [r["tolerance_adjusted_upper"] for r in certs]
        xx = np.arange(3)
        axes[1].scatter(xx, low, color=FAILED, marker="^", s=55, label="Required lower bound")
        axes[1].scatter(xx, high, color="#2563eb", marker="v", s=55, label="Required upper bound")
        for i, r in enumerate(certs):
            axes[1].plot([i, i], [high[i], low[i]], color=FAILED, linewidth=2)
            axes[1].annotate(f"empty gap {r['positive_empty_intersection_gap']:.6f}", (i, (low[i]+high[i])/2), xytext=(7, 0), textcoords="offset points", fontsize=8)
        axes[1].set(xticks=xx, xticklabels=[r["scenario_id"].replace("v6_lite_scenario_", "Scene ") for r in certs], xlim=(-.4, 3.), ylabel="theta9 endpoint velocity (rad/s)", title="Three frozen states: incompatible original constraints")
        axes[1].legend(loc="lower left")
        _grid(axes)
        save(fig, "velocity_completion_and_constraints", "velocity", "Target translation speed x2: FAILED / INCOMPLETE",
             "Source 9c7120e | All five declared scenes retained. Full cohort 25/11/interval and timing verdict: NOT_RUN.\nScalar certificates keep both original 1e-4 rad/s tolerances; conclusions concern the three recorded states, not all possible x2 trajectories.")
        fig, axes = plt.subplots(2, 5, figsize=(18, 8), sharex=True, sharey="row")
        for scene in range(5):
            row, traces = scene_rows[scene], stress[scene]
            for ai, key in enumerate(("rigid_error", "continuum_error")):
                ax = axes[ai, scene]
                _curve(ax, nominal[scene]["time"], nominal[scene][key] * 1000, color="#94a3b8", label="Nominal x1 (9cd1831)", linewidth=1)
                _curve(ax, traces["time"], traces[key] * 1000, color=SAFE if scene in (1, 2) else FAILED, label="x2 completed" if scene in (1, 2) else "x2 partial", linewidth=1.2)
                ax.set(xlim=(0, 27), yscale="log", xlabel="Simulation time (s)")
                if scene not in (1, 2):
                    ax.axvline(row["last_saved_time_s"], color=FAILED, linestyle="--", linewidth=1)
                    ax.axvspan(row["last_saved_time_s"], 27, color="#e2e8f0", alpha=.6)
                if ai == 0:
                    ax.set_title(f"Scene {scene:02} | {row['status']}")
                if scene == 0:
                    ax.set_ylabel(("Rigid" if ai == 0 else "Continuum") + " saved error (mm, log)")
                    ax.legend(loc="upper right")
        _grid(axes)
        save(fig, "velocity_all_five_error_histories", "velocity", "All five speed-stress histories, including the refused trials",
             "Sources x1 9cd1831 / x2 9c7120e | Curves use each producer's saved post-step cache errors. Gray tail is unexecuted, not interpolated.\nPressure task logs include a final rejected planning attempt; it is not an executed servo step. Display curves decimated; no full-cohort pass is inferred.")
        data["velocity"] = {"source_commit": "9c7120e8ca8711cd58dc65471e58763c965148ff", "passed": False, "complete": False,
            "all_scenes_attempted": True, "saved_physics_steps": 48330, "scenes": scene_rows,
            "constraint_certificates": certs, "full_cohort_25_11_interval": "NOT_RUN", "full_cohort_performance": "NOT_RUN_INCOMPLETE_COHORT"}
        coverage.append({"category": "velocity", "scenes": 5, "completed": 2, "partial_rejected": 3, "saved_physics_steps": 48330, "source_commit": data["velocity"]["source_commit"]})

        # Inertia final evidence is the resumed 15-group set, not the original interrupted directory.
        ip = "research_inertia_shadow_resumed_01/"
        ir = evidence.json(ip + "report.json")
        im = evidence.json(ip + "manifest.json")
        replay = evidence.json(ip + "replay_report.json")
        evidence.json(ip + "resumption_plan.json")
        evidence.json("research_inertia_shadow_analysis_01/summary.json")
        evidence.json("research_inertia_shadow_independent_audit_03/report.json")
        evidence.json("research_inertia_shadow_evidence_audit_02/report.json")
        _require(ir["complete"] and ir["evidence_valid"] and ir["status"] == "DIAGNOSTIC_COMPLETE", "inertia final status")
        expected = {(i, a) for i in range(5) for a in (.95, 1., 1.05)}
        _require(len(ir["runs"]) == 15 and {(r["scenario_index"], r["alpha"]) for r in ir["runs"]} == expected, "all 15 unique inertia groups")
        _require(ir["physics_steps_completed"] == 202500 and len(im) == 146 and len(replay["runs"]) == 15, "inertia scope counts")
        ik, target, whole, run_rows = {}, {}, {}, {}
        for row in ir["runs"]:
            scene, alpha, key = row["scenario_index"], row["alpha"], row["key"]
            prefix = ip + "observations/" + key + "/"
            ik[scene, alpha] = evidence.npz(prefix + "fresh_kinematics.npz", ("time", "rigid_tip", "continuum_tip", "rigid_target", "rigid_rotation", "continuum_rotation"))
            target[scene, alpha] = evidence.npz(prefix + "robot_target_500hz.npz", ("minimum_m", "minimum_censored_lower_bound", "physics_grid_phase", "complete", "completed_state_count", "completed_query_count"))
            whole[scene, alpha] = evidence.npz(prefix + "whole_body_dense.npz", ("minimum_m", "minimum_censored_lower_bound", "physics_grid_phase", "complete", "completed_state_count", "completed_query_count"))
            # These files are the saved replay source; reading metadata does not integrate physics.
            trajectory = evidence.npz(ip + "replays/" + key + "/trajectory.npz", ("time", "scenario_index", "alpha"))
            _require(len(trajectory["time"]) == 13501 and int(trajectory["scenario_index"]) == scene and float(trajectory["alpha"]) == alpha, "inertia saved trajectory identity")
            arrays = ik[scene, alpha]
            _require(len(arrays["time"]) == 13501 and np.allclose(arrays["time"][1:], nominal[scene]["time"], rtol=0, atol=1e-12), "fresh / original reference clock")
            for obs, nstate, npair in [(target[scene, alpha], 13501, 75), (whole[scene, alpha], 5401, 2927)]:
                _require(bool(obs["complete"]) and int(obs["completed_state_count"]) == nstate and int(obs["completed_query_count"]) == nstate*npair and len(obs["minimum_m"]) == nstate, "inertia observer complete scope")
            run_rows[scene, alpha] = row
        alphas = [1., .95, 1.05]
        fig, axes = plt.subplots(2, 2, figsize=(14, 9))
        heatmaps = [
            ("Rigid final error (mm)", lambda r: r["fresh_kinematics"]["original_phase_window_diagnostic_statistics"]["rigid_final_position_error_m"]*1000),
            ("Continuum path RMSE, 4.5-25.5 s (mm)", lambda r: r["fresh_kinematics"]["original_phase_window_diagnostic_statistics"]["continuum_original_path_window_error_m"]["rmse"]*1000),
            ("Target minimum: all 2 ms states (mm)", lambda r: r["robot_target_500hz"]["minimum_clearance_m"]*1000),
            ("Whole-body minimum: dense configurations (mm)", lambda r: r["whole_body_dense_discrete"]["minimum_clearance_m"]*1000),
        ]
        for ax, (title, value) in zip(axes.flat, heatmaps):
            matrix = np.array([[value(run_rows[s, a]) for s in range(5)] for a in alphas])
            image = ax.imshow(matrix, aspect="auto", cmap="YlGnBu")
            for i in range(3):
                for j in range(5):
                    ax.text(j, i, f"{matrix[i,j]:.4f}", ha="center", va="center", color="white" if matrix[i,j] > (matrix.max()+matrix.min())/2 else "#111827", fontsize=10)
            ax.set(xticks=range(5), xticklabels=[f"Scene {i:02}" for i in range(5)], yticks=range(3), yticklabels=[f"alpha {a:.2f}" for a in alphas], title=title)
            fig.colorbar(image, ax=ax, shrink=.85)
        save(fig, "inertia_all_15_summary", "inertia", "Inertia sensitivity: 15 full groups, 202,500 same-torque steps",
             "Source 19ad85c; input 9cd1831 | Only principal body inertia scaled, 73 robot bodies. Five nominal runs reproduce nine saved fields exactly.\nFresh-current-state diagnostic errors differ from original cached trace metrics. Not a new 25/11/interval acceptance, feedback robustness or hardware test.")

        fig, axes = plt.subplots(2, 5, figsize=(18, 8), sharex=True, sharey="row")
        for s in range(5):
            for ai, (group, label) in enumerate([(target, "Robot-target minimum (mm)"), (whole, "Whole-body minimum (mm)")]):
                ax = axes[ai, s]
                for a in alphas:
                    obs = group[s, a]
                    # Show lower bounds separately; they cannot silently become exact values.
                    values = np.where(obs["minimum_censored_lower_bound"], np.nan, obs["minimum_m"]*1000)
                    _curve(ax, obs["physics_grid_phase"]*.002, values, color=COLORS[a], label=f"alpha {a:.2f}", linewidth=1.1)
                    c = np.flatnonzero(obs["minimum_censored_lower_bound"])
                    if len(c):
                        ii = c[np.unique(np.linspace(0, len(c)-1, min(100, len(c)), dtype=int))]
                        ax.scatter(obs["physics_grid_phase"][ii]*.002, obs["minimum_m"][ii]*1000, color=COLORS[a], marker="^", s=8, alpha=.35)
                ax.axhline(5, color=FAILED, linestyle="--", linewidth=1)
                ax.set(xlim=(0, 27), xlabel="Simulation time (s)", yscale="log")
                if ai == 0:
                    ax.set_title(f"Scene {s:02} | all three alpha")
                if s == 0:
                    ax.set_ylabel(label + ", log")
                    ax.legend(loc="upper right")
        _grid(axes)
        save(fig, "inertia_all_15_clearance_histories", "inertia", "All 15 inertia groups: two declared geometry scopes",
             "Source 19ad85c | Target: 13,501 actual states x 75 pairs/run. Whole body: 5,401 interpolated configurations x 2,927 pairs/run.\nTriangles indicate censored lower bounds; solid curves use uncensored minima. Whole-body scope is not every-2ms/all-pairs or continuous-time CCD.")

        fig, axes = plt.subplots(2, 5, figsize=(18, 8), sharex=True, sharey="row")
        tracking_rows = []
        for s in range(5):
            for a in alphas:
                arr = ik[s, a]
                rigid = np.linalg.norm(arr["rigid_tip"][1:] - arr["rigid_target"][1:], axis=1)*1000
                continuum = np.linalg.norm(arr["continuum_tip"][1:] - nominal[s]["continuum_target"], axis=1)*1000
                for ai, curve in enumerate([rigid, continuum]):
                    _curve(axes[ai, s], arr["time"][1:], curve, color=COLORS[a], label=f"alpha {a:.2f}", linewidth=1.1)
                phase = run_rows[s, a]["fresh_kinematics"]["original_phase_window_diagnostic_statistics"]
                path_mask = (arr["time"][1:] >= 4.5) & (arr["time"][1:] <= 25.5)
                measured_rmse_m = float(np.sqrt(np.mean((continuum[path_mask] / 1000)**2)))
                measured_final_m = float(np.linalg.norm(arr["rigid_tip"][-1] - arr["rigid_target"][-1]))
                _require(int(path_mask.sum()) == phase["continuum_original_path_window_error_m"]["count"] and
                         np.isclose(measured_rmse_m, phase["continuum_original_path_window_error_m"]["rmse"], rtol=0, atol=1e-14) and
                         np.isclose(measured_final_m, phase["rigid_final_position_error_m"], rtol=0, atol=1e-14),
                         "fresh plotted errors disagree with frozen diagnostic statistics")
                tracking_rows.append({"scene": s, "alpha": a, "rigid_final_m": phase["rigid_final_position_error_m"],
                                      "continuum_path": phase["continuum_original_path_window_error_m"], "rigid_last_1_5s": phase["rigid_original_last_1_5s_position_error_m"]})
            for ai in range(2):
                ax = axes[ai, s]
                ax.set(xlim=(0, 27), yscale="log", xlabel="Simulation time (s)")
                ax.axhline(.1 if ai == 0 else .18, linestyle="--", color="#64748b", linewidth=.8)
                if ai == 1:
                    ax.axvspan(4.5, 25.5, color="#e2e8f0", alpha=.25)
                if ai == 0:
                    ax.set_title(f"Scene {s:02}")
                if s == 0:
                    ax.set_ylabel(("Rigid" if ai == 0 else "Continuum") + " fresh error (mm, log)")
                    ax.legend(loc="upper right")
        _grid(axes)
        save(fig, "inertia_all_15_tracking_histories", "inertia", "All 15 fresh-state tracking histories",
             "Source 19ad85c | No new feedback / QP / runtime executor; the same frozen torque is applied open loop. Initial state excluded from errors.\nDashed 0.10 / 0.18 mm lines are original nominal diagnostic comparisons, not per-sample pass gates. Continuum path statistics use the shaded 4.5-25.5 s window.")

        fig, axes = plt.subplots(2, 5, figsize=(18, 8), sharex=True, sharey="row")
        fig_clear, clear_axes = plt.subplots(1, 5, figsize=(18, 5), sharex=True, sharey=True)
        paired_tables = []
        for s in range(5):
            ref = ik[s, 1.]
            for a in (.95, 1.05):
                arr = ik[s, a]
                for ai, key in enumerate(["rigid_tip", "continuum_tip"]):
                    delta = np.linalg.norm(arr[key] - ref[key], axis=1)*1000
                    _curve(axes[ai, s], arr["time"], delta, color=COLORS[a], label=f"alpha {a:.2f} - 1.00", linewidth=1.1)
                obs, nobs = target[s, a], target[s, 1.]
                exact = ~(obs["minimum_censored_lower_bound"] | nobs["minimum_censored_lower_bound"])
                delta = np.where(exact, (obs["minimum_m"]-nobs["minimum_m"])*1000, np.nan)
                _curve(clear_axes[s], arr["time"], delta, color=COLORS[a], label=f"alpha {a:.2f}", linewidth=1.1)
                table = next(r for r in ir["paired_vs_nominal"] if r["scenario_index"] == s and r["alpha"] == a)
                _require(int(exact.sum()) == table["paired_exact_clearance_count"], "paired exact count")
                _require(np.isclose(float(np.min(delta[exact])) / 1000, table["paired_exact_target_clearance_delta_m"]["min"], rtol=0, atol=1e-14) and
                         np.isclose(float(np.max(delta[exact])) / 1000, table["paired_exact_target_clearance_delta_m"]["max"], rtol=0, atol=1e-14),
                         "plotted paired clearance disagrees with frozen exact response")
                paired_tables.append(table)
            for ai in range(2):
                ax = axes[ai, s]
                ax.set(xlim=(0, 27), xlabel="Simulation time (s)")
                if ai == 0:
                    ax.set_title(f"Scene {s:02}")
                if s == 0:
                    ax.set_ylabel(("Rigid" if ai == 0 else "Continuum") + " tip delta vs nominal (mm)")
                    ax.legend(loc="upper left")
            clear_axes[s].axhline(0, color="#64748b", linewidth=.8)
            clear_axes[s].set(title=f"Scene {s:02}", xlim=(0, 27), xlabel="Simulation time (s)")
            if s == 0:
                clear_axes[s].set_ylabel("Exact target minimum delta (mm)")
                clear_axes[s].legend()
        _grid(axes)
        _grid(clear_axes)
        save(fig, "inertia_all_10_paired_tip_responses", "inertia", "Ten paired execution responses across all five scenes",
             "Source 19ad85c | Each perturbed full trajectory compared with its own alpha=1.00 fresh-state trajectory. These are tip offsets, not target errors.\nNo run selection. JSON also retains low-level joint, base pose and tip orientation response statistics for all ten pairs. Display curves decimated.")
        save(fig_clear, "inertia_all_10_paired_clearance_responses", "inertia", "Paired clearance response: censor-aware differences",
             "Source 19ad85c | Difference is plotted only when both target minima are uncensored; gaps are retained.\nTen pairs: 131,555 exact states, 3,455 excluded; total 135,010. Exact range -5.967862 to +5.203110 mm. Not a robust uncertainty bound.")
        data["inertia"] = {"source_commit": "19ad85c7db81a2e254fd176077b6aae39333a439", "status": ir["status"],
            "physics_steps_completed": 202500, "runs": 15, "source_directory": str((RUNS / ip).relative_to(REPO)),
            "final_manifest_items": 146, "resumption": ir["resumption"],
            "nominal_parity": "5 scenes x 9 saved fields, zero residual", "tracking_by_scene_alpha": tracking_rows,
            "geometry_by_scene_alpha": [{"scene": r["scenario_index"], "alpha": r["alpha"], "target": r["robot_target_500hz"], "whole_body": r["whole_body_dense_discrete"]} for r in ir["runs"]],
            "paired_10": paired_tables, "limitations": ir["limitations"],
            "independent_scope": "100 nominal prefix physics steps; 254045 representative geometry queries, not full independent replay",
            "all_pair_raw_distance_values_saved": False,
            "error_cache_scope": "Fresh observer errors differ from original nominal mj_step cached trace errors."}
        coverage.append({"category": "inertia", "scenes": 5, "alphas": alphas, "complete_runs": 15, "paired_responses": 10,
                         "target_states_per_run": 13501, "whole_body_configs_per_run": 5401, "observations_reused": 14, "observations_new": 1, "new_physics_on_resumption": 0})

        # Compute statistics and native wall continuation have separate populations and verdicts.
        wp = "c11_qualified_default_01/"
        wr = evidence.json(wp + "report.json")
        wplan = evidence.json(wp + "plan.json")
        wtrial = evidence.json(wp + "five/wall_trial_report.json")
        _require(not wr["passed"] and wtrial["all_predeclared_scenes_attempted"] and len(wtrial["failed_scenarios"]) == 4, "old wall failure changed")
        wall_rows = []
        for s in range(5):
            if s == 1:
                row = {"scene": s, "status": "COMPLETED", "executed_steps": 13500, "executed_time_s": 27., "rejected_wake_lateness_ms": None}
            else:
                failure = evidence.json(wp + f"five/failures/v6_lite_scenario_{s:02}_interval_failure.json")
                attempt = failure["rejected_servo_attempt"]
                _require(not failure["rejected_step_executed"], "wall rejected physical step was executed")
                row = {"scene": s, "status": failure["failure_reason"], "executed_steps": failure["physics_steps_executed"],
                       "executed_time_s": failure["time_s"], "rejected_wake_lateness_ms": (attempt["actual_start"]-attempt["scheduled"])*1000,
                       "rejected_step_executed": False}
            wall_rows.append(row)
        perf = nr["computational_performance"]
        _require(perf["evidence_valid"] and perf["performance_target_met"] and not perf["startup_samples_excluded"], "nominal performance scope")
        perf_rows = perf["scenes"]
        fig, axes = plt.subplots(2, 2, figsize=(14, 9))
        xx = np.arange(5)
        for off, stat, color in [(-.25, "p95_ms", "#2563eb"), (0, "p99_ms", "#7c80c9"), (.25, "max_ms", "#a8b3c1")]:
            axes[0, 0].bar(xx+off, [r["dispatch"][stat] for r in perf_rows], width=.25, label=stat.replace("_ms", ""), color=color)
        axes[0, 0].axhline(20, color=FAILED, linestyle="--", label="20 ms sampling target")
        axes[0, 0].set(xticks=xx, xticklabels=[f"{i:02}" for i in range(5)], ylabel="Dispatch wall compute (ms)", title="Nominal research: p95 target PASSED only")
        axes[0, 0].legend()
        axes[0, 1].bar(xx, [r["dispatch"]["over_deadline_count"] for r in perf_rows], color=UNKNOWN)
        axes[0, 1].set(xticks=xx, xticklabels=[f"{i:02}" for i in range(5)], ylabel="Cycles > 20 ms / 1,350 each", title="107 / 6,750 exceedances; first cycle retained")
        axes[1, 0].barh(xx, [27]*5, color="#e2e8f0")
        axes[1, 0].barh(xx, [r["executed_time_s"] for r in wall_rows], color=[SAFE if r["status"] == "COMPLETED" else FAILED for r in wall_rows])
        for i, row in enumerate(wall_rows):
            axes[1, 0].text(.3, i, f"{row['executed_time_s']:.3f}s | {row['status']}", va="center", color="white", fontsize=8)
        axes[1, 0].set(yticks=xx, yticklabels=[f"Scene {i:02}" for i in range(5)], xlim=(0,28), xlabel="Native executed simulation time (s)", title="C11 native/HIGH/wall: NOT_MET")
        failed = [r for r in wall_rows if r["status"] != "COMPLETED"]
        axes[1, 1].bar([f"{r['scene']:02}" for r in failed], [r["rejected_wake_lateness_ms"] for r in failed], color=FAILED)
        axes[1, 1].set(xlabel="Refused scene", ylabel="Rejected wake lateness (ms)", title="Four rejected next steps were not executed")
        _grid(axes)
        save(fig, "nominal_compute_and_c11_wall_status", "wall", "Research compute and native wall continuation: separate verdicts",
             "Nominal source 9cd1831: simulation PASSED (25 functional + 11 contract + interval), p95 sampling target PASSED; engineering 16ms/p99 20ms NOT_MET.\nWall source c2b5077: one of five complete; formal repeats/new native 26/11/interval not complete. Admin DEFERRED/platform BLOCKED; hard RT and hardware NOT_ESTABLISHED.")
        data["nominal"] = {"source_commit": "9cd1831e7c77a340464a669eac3b5aa7cd9c3c3c", "status": "PASSED",
            "physics_steps": 67500, "planning_cycles": 6750, "functional_evidence_checks": 25, "contract_checks": 11,
            "independent_interval_boundaries": 6755, "independent_interval_rows": 12350, "metrics": nm["aggregate_metrics"],
            "compute_performance": perf, "wall_claim": False}
        data["wall"] = {"source_commit": wplan["source_commit"], "status": "NOT_MET", "five_scenes": wall_rows,
            "native_five_completed": 1, "native_five_rejected": 4, "formal_repeats": "NOT_STARTED_NATIVE",
            "new_complete_native_26_11_interval": "NOT_MET", "administrator": "DEFERRED", "administrator_platform": "BLOCKED",
            "hard_realtime": "NOT_ESTABLISHED", "hardware": "NOT_ESTABLISHED", "safety_backup": "NOT_ESTABLISHED"}
        coverage.append({"category": "wall", "native_five_scenes": 5, "completed": 1, "rejected": 4,
                         "source_commit": wplan["source_commit"], "nominal_timing_source_commit": data["nominal"]["source_commit"]})

    data["coverage"] = coverage
    data["panels"] = [{**p, "path": str(p["path"].relative_to(output))} for p in panels]
    data = _plain(data)
    table = folder / "latest_studies_data.json"
    _require(not table.exists(), "refuse existing JSON table")
    table.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    artifacts.append(table)
    return {"artifacts": artifacts, "source_refs": sorted(evidence.refs), "data": data, "coverage": coverage, "panels": panels}
