"""Qualification gates for the predeclared full wall-clock task horizons."""
import hashlib
import json
import math


def qualify_scenes(scenes, expected_count):
    expected_ids = {f"v6_lite_scenario_{index:02d}" for index in range(expected_count)}
    records = []
    for scene in scenes:
        scene_id = scene.get("scenario", {}).get("scenario_id")
        rates = scene.get("metrics", {}).get("rates_and_latency", {})
        checks = {"original_scene_checks": scene.get("passed") is True}
        for name, expected in (("physics_steps", 13500), ("task_ticks", 1350),
                               ("torque_update_count", 13500), ("physics_hz", 500), ("task_hz", 50)):
            checks[name] = rates.get(name) == expected
        latency = {}
        for name in ("dispatch", "acquisition_to_first_application"):
            value = rates.get(name, {})
            p95, first = value.get("p95_ms"), value.get("first_cycle_ms")
            valid = lambda number: (isinstance(number, (int, float))
                                    and not isinstance(number, bool) and math.isfinite(number))
            checks[f"{name}_full_count"] = value.get("count") == 1350
            checks[f"{name}_startup_retained"] = valid(first) and first >= 0
            checks[f"{name}_p95"] = valid(p95) and 0 <= p95 <= 20
            latency[name] = {"p95_ms": p95, "first_cycle_ms": first, "count": value.get("count")}
        records.append({"scenario_id": scene_id, "checks": checks, "latency": latency,
                        "passed": all(checks.values())})
    complete = (len(scenes) == expected_count
                and {record["scenario_id"] for record in records} == expected_ids)
    return {"expected_scenes": expected_count, "complete_scene_horizons": complete,
            "scenes": records, "passed": complete and all(record["passed"] for record in records),
            "physical_validation_pending": True}


def qualify_directory(path, expected_count):
    metrics_path = path / ("single_scene_metrics.json" if expected_count == 1 else "v6_lite_metrics.json")
    try:
        raw = metrics_path.read_bytes()
        payload = json.loads(raw)
        scenes = [payload] if expected_count == 1 else payload["scenarios"]
        report = qualify_scenes(scenes, expected_count)
        report["summary_checks_passed"] = payload.get("passed") is True
        report["passed"] &= report["summary_checks_passed"]
        report["metrics_file"] = metrics_path.as_posix()
        report["metrics_sha256"] = hashlib.sha256(raw).hexdigest()
        return report
    except (OSError, ValueError, TypeError, KeyError, AttributeError) as error:
        return {"expected_scenes": expected_count, "passed": False,
                "reason": "MISSING_OR_INVALID_FULL_HORIZON_METRICS", "error": str(error),
                "physical_validation_pending": True}
