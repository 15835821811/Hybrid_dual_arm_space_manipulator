"""Independently check native tracking values, aliases, media and frozen evidence."""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from v6_4.route_optimizer_protocol import read, sha
from v6_4.visualization.build_preference_warmstart_media import write
from v6_4.visualization import build_c4a_media as media


def check(run, output):
    payload = read(output / "dashboard_data.json")
    manifest = read(output / "visualization_manifest.json")
    assert len(payload["slots"]) == 12 and manifest["video_count"] == 49
    checked_files = 0
    for name, entry in manifest["files"].items():
        assert sha(output / name) == entry["sha256"], name
        checked_files += 1
    original = read(ROOT / "docs/V6_4_C4A_DELIVERY_MANIFEST.json")
    for name, expected in original["files_sha256"].items():
        assert sha(ROOT / name) == expected, name
    inventory = read(ROOT / "v6_4/releases/c4a_architecture_audit_20261010_01/raw_inventory_sha256.json")
    for row in inventory["all_files"]:
        assert sha(run / row["path"]) == row["sha256"], row["path"]
    rows = []
    for slot in payload["slots"]:
        source = media.resolve_unique(payload["slots"], slot["task_id"], slot["method"])
        assert slot["media"] == source["media"] and slot["figures"] == source["figures"]
        if slot.get("alias_of_slot"):
            continue
        task, _, _, _, replay, reference, base, parity, _ = media.load_actual(run, slot)
        rigid_ref = replay["target_position"] + np.einsum("nij,j->ni", replay["target_rotation"], task.scenario["grasp_point_target_frame_m"])
        c_ref = np.broadcast_to(np.asarray(task.scenario["continuum_target_rotation_world"]), replay["continuum_rotation"].shape)
        r_ref = replay["target_rotation"] @ np.asarray(task.scenario["grasp_rotation_target_frame"])
        def angle(a, b):
            relative = np.swapaxes(a, 1, 2) @ b
            return np.arccos(np.clip((np.trace(relative, axis1=1, axis2=2) - 1) / 2, -1, 1))
        expected = np.column_stack((replay["time"], np.linalg.norm(replay["continuum_position"]-reference, axis=1),
            np.linalg.norm(replay["continuum_position"]-base, axis=1), np.linalg.norm(replay["rigid_position"]-rigid_ref, axis=1),
            angle(replay["continuum_rotation"], c_ref), angle(replay["rigid_rotation"], r_ref)))
        saved = np.loadtxt(output / slot["figures"]["tracking_csv"], delimiter=",", skiprows=1)
        assert saved.shape == (13501, 6) and np.isfinite(saved).all()
        assert np.allclose(saved, expected, atol=2e-8, rtol=0), slot["method"]
        meta = slot["media"]
        assert meta["schema"] == media.SCHEMA and meta["render_state_kind"] == "ACTUAL_TRACE_SAVED_STATES"
        assert meta["frame_count"] == 406 and meta["includes_exact_saved_endpoint"]
        assert abs(meta["displayed_saved_end_s"] - 27) < 1e-9
        for name in ("physics_steps", "geometry_queries", "QP_solves"):
            assert meta["render_cost"][name] == 0
        assert meta["render_cost"]["forbidden_call_attempts"] == []
        rows.append({"task_id": slot["task_id"], "method": slot["method"], "native_rows": len(saved),
            "max_abs_difference_per_column": np.max(np.abs(saved-expected), axis=0).tolist(),
            "max_continuum_generated_error_mm": float(saved[:, 1].max()*1000),
            "max_rigid_grasp_error_mm": float(saved[:, 3].max()*1000), "saved_state_parity": parity})
    assert len(rows) == 7
    return {"status": "PASS", "created_utc": datetime.now(timezone.utc).isoformat(),
            "media_files_verified": checked_files, "original_delivery_files_unchanged": len(original["files_sha256"]),
            "raw_evidence_files_unchanged": len(inventory["all_files"]), "logical_slots": 12,
            "strict_alias_slots": 5, "unique_actuals": 7, "videos": 49, "checks": rows,
            "physics_steps": 0, "new_model_samples": 0, "training_updates": 0}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--media", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    result = check(a.run.resolve(), a.media.resolve())
    write(a.output, result)
    print({k: v for k, v in result.items() if k != "checks"})
