"""Existing-data audit. No data generation, TEST access, or physical simulation."""
import os
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"
import collections
import hashlib
import html
import json
from pathlib import Path
import time
import traceback
import numpy as np
from scipy.interpolate import CubicHermiteSpline

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/v7_1/waypoint_direct_audit_001"
START = time.time()


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write(name, value):
    p = OUT / name
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(p)


def status(stage, **extra):
    write("status.json", dict(status="RUNNING", stage=stage, pid=os.getpid(),
          heartbeat=time.time(), elapsed_s=time.time()-START, **extra))


def inventory():
    """Only explicitly TRAIN/validation paths; never open TEST episode results."""
    entries, sources = [], {}
    for folder in ("paper100_v1", "paper500_v2"):
        manifest = ROOT / "tracking_output" / folder / "manifest.json"
        sources[str(manifest.relative_to(ROOT))] = digest(manifest)
        for e in read(manifest)["tasks"]:
            if e["split"] not in ("train", "validation"):
                continue
            p = ROOT / e.get("teacher_path", f"tracking_output/{folder}/teachers/{e['id']}")
            entries.append((p, e["split"], e["task"]["seed"], e["id"]))
    local = ROOT / "tracking_output/local10mm_v1"
    split_path = local / "paper_data_20/split_manifest.json"
    manifest_path = local / "pilot_manifest.json"
    sources[str(split_path.relative_to(ROOT))] = digest(split_path)
    sources[str(manifest_path.relative_to(ROOT))] = digest(manifest_path)
    tasks = {e["id"]: e for e in read(manifest_path)["tasks"]}
    for split, ids in read(split_path)["split"].items():
        if split not in ("train", "validation"):
            continue
        for ident in ids:
            for base in sorted(local.glob("teacher_a*")):
                p = base / ident
                if (p / "result.json").exists():
                    entries.append((p, split, tasks[ident]["task"]["seed"], ident))
    report_path = ROOT / "tracking_output/ui_pilot/data/dataset_report.json"
    sources[str(report_path.relative_to(ROOT))] = digest(report_path)
    for e in read(report_path)["episodes"]:
        if e["split"] in (0, 1):
            entries.append((ROOT / Path(e["result"]).parent,
                            "train" if e["split"] == 0 else "validation",
                            e["seed"], f"ui_{e['seed']}"))
    # Preserve split conflicts and distinct versions; deduplicate identical paths only.
    unique = {}
    for p, split, seed, ident in entries:
        unique.setdefault(str(p.resolve()), (p, split, seed, ident))
    return sorted(unique.values(), key=lambda x: str(x[0])), sources


def audit_episode(entry, fit=False):
    path, split, seed, ident = entry
    row = dict(path=str(path.relative_to(ROOT)), split=split, seed=seed, id=ident)
    rp, tp = path / "result.json", path / "trace.npz"
    if not rp.exists() or not tp.exists():
        return dict(row, available=False, eligible=False, reasons=["MISSING_RESULT_OR_TRACE"])
    r = read(rp)
    row.update(available=True, result_sha256=digest(rp), trace_sha256=digest(tp),
               old_admitted=bool(r.get("accepted_for_teacher_dataset")),
               domain=r.get("task", {}).get("domain"),
               old_runtime_contract=r.get("model_runtime_contract_sha256"))
    reasons = []
    if not row["old_admitted"]:
        reasons.append("OLD_ADMISSION_FAILED")
    with np.load(tp, allow_pickle=False) as z:
        row["fields"] = {k: list(z[k].shape) for k in z.files}
        t, q, dq = z["time"], z["q"], z["dq"]
        row["sampled_continuum_max_rad"] = float(np.max(np.abs(q[:, :10])))
        row["sampled_reference_max_rad"] = float(np.max(np.abs(z["q_reference"][:, :10])))
        row["duration_s"] = float(t[-1]-t[0])
        row["dt_max_s"] = float(np.max(np.diff(t)))
        row["finite"] = bool(np.isfinite(q).all() and np.isfinite(dq).all())
        # Legacy native free target is the final six generalized velocities.
        # Cross-check with declared motion; unknown remains unknown.
        parameters = r.get("task_parameters", {})
        motion = parameters.get("target_velocity")
        row["declared_target_velocity"] = motion
        row["target_motion"] = ("stationary" if np.max(np.abs(motion)) < 1e-12 else "moving") if motion is not None else "unknown"
        row["native_initial_tail_velocity"] = z["qvel"][0, -6:].tolist()
        if row["sampled_continuum_max_rad"] > 1 or row["sampled_reference_max_rad"] > 1:
            reasons.append("OBSERVED_CONTINUUM_DOMAIN_VIOLATION")
        if row["duration_s"] < 27-1e-6 or not row["finite"]:
            reasons.append("INCOMPLETE_OR_NONFINITE")
        # These are evidence gaps, not assertions of physical failure.
        if not any(k in z.files for k in ("integration_state", "initial_integration_state")):
            reasons.append("COMPLETE_INITIAL_INTEGRATION_STATE_NOT_IN_TRACE")
        if not any(k in z.files for k in ("ctrl", "torque", "controls")):
            reasons.append("NATIVE_TORQUE_HISTORY_NOT_IN_TRACE")
        row["sidecar_files"] = sorted(p.name for p in path.iterdir() if p.is_file())
        if fit:
            idx = np.r_[np.arange(0, len(t)-1, 5), len(t)-1]
            spline = CubicHermiteSpline(t[idx], q[idx], dq[idx])
            error = np.abs(spline(t)-q)
            row["offline_representation"] = dict(
                label="READ_ONLY_COORDINATE_FIT_NOT_PHYSICS_VALIDATION",
                knot_count=len(idx), max_coordinate_error_rad=float(error.max()),
                rmse_rad=float(np.sqrt(np.mean(error**2))),
                original_initial_velocity_error=float(np.max(np.abs(spline(t[0], 1)-dq[0]))))
    reasons.append("LEGACY_CONTRACT_NOT_REQUALIFIED_TO_CURRENT_V7")
    row.update(eligible=False, reasons=reasons)
    return row


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / "status.json").exists():
        raise RuntimeError("Existing run is immutable; use job.ps1 status/report.")
    status("inventory")
    entries, sources = inventory()
    write("freeze.json", dict(protocol_sha256=digest(ROOT/"paper/waypoint_direct/EXPERIMENT_PLAN.md"),
          worker_sha256=digest(Path(__file__)), manifests=sources,
          paths=[str(e[0].relative_to(ROOT)) for e in entries],
          excluded_splits=["test", "internal_test", "diagnostic"], training_updates=0))
    rows = []
    fitted = collections.Counter()
    for i, entry in enumerate(entries):
        if time.time()-START > 3600:
            raise TimeoutError("B0 one-hour cap")
        split = entry[1]
        fit = fitted[split] < (32 if split == "train" else 8)
        row = audit_episode(entry, fit)
        if "offline_representation" in row:
            fitted[split] += 1
        rows.append(row)
        status("read_only_source_audit", completed=i+1, total=len(entries))
    write("episodes.json", rows)
    available = [r for r in rows if r["available"]]
    reasons = collections.Counter(x for r in rows for x in r["reasons"])
    summary = dict(status="BLOCKED", cause="NO_CURRENT_CONTRACT_QUALIFIED_DYNAMIC_TRAIN_VAL",
                   source_records=len(rows), available_records=len(available),
                   unique_task_seeds=len(set(r["seed"] for r in available)),
                   old_admitted_records=sum(r["old_admitted"] for r in available),
                   current_qualified_train=0, current_qualified_val=0,
                   reason_counts=dict(reasons),
                   split_motion=dict(collections.Counter(r["split"]+"/"+r["target_motion"] for r in available)),
                   representation_diagnostics=dict(fitted), actual_training_updates=0,
                   new_physical_executions=0, new_replays=0, new_DEV_candidates=0,
                   stages=dict(B0="AUDITED_QUALIFICATION_BLOCKED", representation_execution="BLOCKED",
                               B1="BLOCKED", B2="BLOCKED", B3="BLOCKED", B4="REPORT"))
    write("summary.json", summary)
    changed = [r["path"] for r in available if digest(ROOT/r["path"]/"trace.npz") != r["trace_sha256"]
               or digest(ROOT/r["path"]/"result.json") != r["result_sha256"]]
    write("preservation.json", dict(files_checked=len(available)*2, changed=changed))
    if changed:
        raise RuntimeError("Source changed during read-only audit; freeze invalid")
    # This is an audit visualization, never an animation of unexecuted candidates.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].bar(["legacy admitted", "V7 qualified"], [summary["old_admitted_records"], 0])
    ax[0].set_title("Records; legacy admission is not V7 certification")
    values = [r["sampled_continuum_max_rad"] for r in available]
    ax[1].hist(values, bins=25)
    ax[1].axvline(1, color="red", label="1 rad hard limit")
    ax[1].set_title("Saved TRAIN/VAL state samples (not 500 Hz audit)")
    ax[1].legend()
    fig.tight_layout()
    fig.savefig(OUT/"data_audit.png", dpi=140)
    plt.close(fig)
    table = "".join(f"<tr><td>{html.escape(r['path'])}</td><td>{r['split']}</td>"
                    f"<td>{r.get('sampled_continuum_max_rad','missing')}</td><td>{html.escape(', '.join(r['reasons']))}</td></tr>" for r in rows)
    (OUT/"index.html").write_text('<!doctype html><meta charset="utf-8"><title>动态数据资格审计</title>'
        '<style>body{font:16px sans-serif;margin:2em}table{border-collapse:collapse}td{border:1px solid #ddd;padding:8px}img{max-width:100%}</style>'
        '<h1>动态数据资格审计：下游 BLOCKED</h1><p>本轮没有训练或物理执行；下图是旧 TRAIN/VAL 的只读检查。</p>'
        '<img src="data_audit.png"><p><a href="../waypoint_domain_002/index.html">历史修复轮全部回放（不是本轮成果）</a></p>'
        '<p><a href="summary.json">摘要</a> · <a href="episodes.json">逐条证据</a></p>'
        '<table><tr><th>来源</th><th>分割</th><th>采样状态最大rad</th><th>排除/待认证原因</th></tr>'+table+'</table>', encoding="utf-8")
    write("status.json", dict(status="BLOCKED", stage="B0_COMPLETE", pid=os.getpid(),
          heartbeat=time.time(), elapsed_s=time.time()-START, downstream="NOT_STARTED"))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        OUT.mkdir(parents=True, exist_ok=True)
        write("failure.json", dict(traceback=traceback.format_exc()))
        write("status.json", dict(status="FAILED", pid=os.getpid(), heartbeat=time.time()))
        raise
