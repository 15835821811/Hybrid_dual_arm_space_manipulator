"""One full wall scene before the original locked five-scene suite."""
import argparse
from pathlib import Path
from v6_lite.hierarchical_qp import HierarchicalQPConfig
from v6_lite.run_evidence import start_run, finish_run, fail_run
from v6_lite.run_v6_lite import (V6LiteRunConfig, build_scenarios,
    default_v6_lite_robot_spec, run_scenario, _write_json)


def run(output):
    cfg = V6LiteRunConfig(pcc_mode="bounded_interval_pcc")
    qp = HierarchicalQPConfig(enable_capsule_cbf=True)
    spec = default_v6_lite_robot_spec()
    scenario = build_scenarios(spec, cfg)[0]
    metadata = start_run(output, run_config=cfg, qp_config=qp, spec=spec, scenarios=(scenario,))
    try:
        result = run_scenario(spec, cfg, qp, scenario, output / "traces")
        _write_json(output / "single_scene_metrics.json", result)
        finish_run(output, metadata, passed=result["passed"], summary=result["checks"])
        print(result["checks"], flush=True)
        return result["passed"]
    except BaseException as error:
        fail_run(output, metadata, error, scenario_id=scenario.scenario_id)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    raise SystemExit(0 if run(args.output_dir) else 1)
