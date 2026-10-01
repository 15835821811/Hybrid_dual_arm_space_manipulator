"""C.1 monotonic acquisition-to-first-torque timelines; no control arithmetic."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np


TRACE_SCHEMA_VERSION = "v6_2_a1_ramp_aware_qp"
TIMING_SCHEMA_VERSION = "v6_2_c1_dispatch_timeline_v1"


def runtime_identity(pcc_mode, spec, model=None, *, dispatch_clock_policy=None, wall_executor_backend=None):
    return {
        "trace_schema_version": TRACE_SCHEMA_VERSION,
        "controller_version": ("v6_2_research_simulation_bounded_interval_pcc" if
                               pcc_mode == "bounded_interval_pcc" and dispatch_clock_policy == "research_simulation" else
                               "v6_2_c11_native_wall_handoff" if pcc_mode == "bounded_interval_pcc"
                               and dispatch_clock_policy == "wall_deadline" and wall_executor_backend == "native" else
                               "v6_2_c11_wall_handoff" if pcc_mode == "bounded_interval_pcc"
                               and dispatch_clock_policy == "wall_deadline" else
                               "v6_2_c1_bounded_interval_pcc" if
                               pcc_mode == "bounded_interval_pcc" else
                               "v6_2_a1_legacy_pcc"),
        "pcc_mode": pcc_mode,
        "execution_mode": dispatch_clock_policy or "historical_not_specified",
        "command_validity_clock": ("simulation_time" if dispatch_clock_policy == "research_simulation"
                                   else "wall_monotonic"),
        "performance_is_acceptance_gate": dispatch_clock_policy != "research_simulation",
        "wall_deployment_certified": False,
        "servo_law_version": ("b2_implicitfast_compensated_torque_v1" if
                              pcc_mode == "bounded_interval_pcc" else
                              "a1_model_based_servo_torque_v1"),
        "integrator": int(model.opt.integrator) if model is not None else 3,
        "integrator_name": "implicitfast",
        "model_runtime_contract_sha256": spec.runtime_contract_sha256(),
        "model_source_bundle_sha256": spec.source_bundle_sha256(),
    }


class CycleTimeline:
    """Sequential, nonoverlapping phases, measured on wall and thread clocks."""

    def __init__(self, command_id, state_time_s):
        self.command_id = command_id
        self.state_time_s = state_time_s
        self.started = self.last = time.perf_counter_ns()
        self.cpu_started = self.cpu_last = time.thread_time_ns()
        self.phases = []

    def mark(self, name):
        wall, cpu = time.perf_counter_ns(), time.thread_time_ns()
        self.phases.append({"name": name,
                            "start_ns": self.last, "end_ns": wall,
                            "wall_s": (wall - self.last) * 1e-9,
                            "thread_cpu_s": (cpu - self.cpu_last) * 1e-9})
        self.last, self.cpu_last = wall, cpu

    def record(self, **diagnostic):
        return {
            "schema": TIMING_SCHEMA_VERSION,
            "command_id": self.command_id,
            "source_simulation_time_s": self.state_time_s,
            "state_acquisition_monotonic_ns": self.started,
            "actual_dispatch_monotonic_ns": self.last,
            "dispatch_latency_s": (self.last - self.started) * 1e-9,
            "thread_cpu_s": (self.cpu_last - self.cpu_started) * 1e-9,
            "phases": [dict(phase) for phase in self.phases],
            "nested_measurements": {
                "qp_solver_latency": "qp_assembly_and_solve",
                "interval_preview_latency": ["ten_step_preview", "next_start_check"],
                "algorithm_full_latency": "task_tick_started_before_timeline_to_before_constraint_snapshot",
            },
            **diagnostic,
        }


def write_timelines(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, allow_nan=False) + "\n")


def latency_summary(samples, deadline_s=0.020, *, include_steady=True):
    samples = np.asarray(samples, dtype=float)
    if not len(samples):
        return {"count": 0, "p95_ms": None, "passed": False}
    longest = streak = 0
    for late in samples > deadline_s:
        streak = streak + 1 if late else 0
        longest = max(longest, streak)
    return {
        "count": len(samples), "p50_ms": float(np.median(samples) * 1e3),
        "p95_ms": float(np.percentile(samples, 95) * 1e3),
        "p99_ms": float(np.percentile(samples, 99) * 1e3),
        "max_ms": float(np.max(samples) * 1e3),
        "over_deadline_count": int(np.count_nonzero(samples > deadline_s)),
        "longest_consecutive_over_deadline": longest,
        "first_cycle_ms": float(samples[0] * 1e3),
        "steady_after_first_cycle": latency_summary(
            samples[1:], deadline_s, include_steady=False)
        if len(samples) > 1 and include_steady else None,
        "passed": bool(np.percentile(samples, 95) <= deadline_s),
    }
