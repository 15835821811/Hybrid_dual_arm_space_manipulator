"""B.3 cost accounting around the unmodified B.2 execution and evidence path.

Wrappers forward the original arguments and return objects without modifying
model/data, commands, tolerances, or exceptions. Their Python bookkeeping is
included in measured wall time; wall time remains a research diagnostic.
"""
from __future__ import annotations

from collections import Counter
from contextlib import contextmanager
import functools
import time

import mujoco


class ExecutionCostLedger:
    def __init__(self):
        self.phase = "setup"
        self.counts = Counter()
        self.preview_records = []
        self.started = time.perf_counter()

    @contextmanager
    def scope(self, phase):
        previous = self.phase
        self.phase = phase
        try:
            yield
        finally:
            self.phase = previous

    def forward_counted(self, function, operation):
        @functools.wraps(function)
        def call(*args, **kwargs):
            phase = self.phase
            self.counts[(phase, operation, "started")] += 1
            try:
                result = function(*args, **kwargs)
            except BaseException:
                self.counts[(phase, operation, "raised")] += 1
                raise
            self.counts[(phase, operation, "returned")] += 1
            return result
        return call

    def forward_scoped(self, function, phase):
        @functools.wraps(function)
        def call(*args, **kwargs):
            with self.scope(phase):
                return function(*args, **kwargs)
        return call

    def forward_preview(self, function):
        @functools.wraps(function)
        def call(*args, **kwargs):
            before = self.physics_steps("private_preview")
            row = {"call_index": len(self.preview_records), "status": "STARTED"}
            self.preview_records.append(row)
            try:
                with self.scope("private_preview"):
                    result = function(*args, **kwargs)
                row.update(status="RETURNED", coverage_states=len(result[0]["coverage"]))
                return result
            except BaseException as error:
                row.update(status="RAISED", exception_type=type(error).__name__)
                raise
            finally:
                row["completed_physics_steps"] = self.physics_steps("private_preview") - before
        return call

    def physics_steps(self, phase):
        return sum(self.counts[(phase, op, "returned")] for op in ("mj_step", "mj_step2"))

    def to_dict(self):
        phases = sorted({key[0] for key in self.counts} | {"actual", "private_preview", "independent_torque_replay"})
        operations = ("mj_step", "mj_step2", "mj_geomDistance")
        return {
            "schema": "v64_b3_execution_cost_ledger_v1",
            "phase_counts": {p: {op: {state: self.counts[(p, op, state)] for state in ("started", "returned", "raised")}
                                 for op in operations} for p in phases},
            "actual_physics_steps": self.physics_steps("actual"),
            "private_preview_physics_steps": self.physics_steps("private_preview"),
            "independent_saved_torque_replay_steps": self.physics_steps("independent_torque_replay"),
            "preview_calls": len(self.preview_records), "preview_records": self.preview_records,
            "native_geometry_query_calls": sum(v for (p, op, state), v in self.counts.items()
                                                if op == "mj_geomDistance" and state == "returned"),
            "geometry_query_scope": "Native mj_geomDistance calls, separated by phase; analytic PCC query work is reported by the original interval evidence, not mislabeled as native distance calls.",
            "instrumentation": "Transparent argument/return/exception forwarding; additional Python bookkeeping included in wall time.",
            "physics_step_count_scope": "Successfully returned mj_step or mj_step2 integrations; raised calls reported separately without assuming how far a native failure progressed.",
            "control_behavior_modified": False, "wall_20ms_is_gate": False,
            "elapsed_wall_s": time.perf_counter() - self.started,
        }

    @contextmanager
    def installed(self):
        from v6_4 import residual_execution as execution
        from v6_lite import b2_interval_runtime
        patches = [
            (mujoco, "mj_step", self.forward_counted(mujoco.mj_step, "mj_step")),
            (mujoco, "mj_step2", self.forward_counted(mujoco.mj_step2, "mj_step2")),
            (mujoco, "mj_geomDistance", self.forward_counted(mujoco.mj_geomDistance, "mj_geomDistance")),
            (b2_interval_runtime, "preview_ramp", self.forward_preview(b2_interval_runtime.preview_ramp)),
            (execution, "run_synchronous_scenario", self.forward_scoped(execution.run_synchronous_scenario, "actual")),
            (execution, "_replay", self.forward_scoped(execution._replay, "independent_torque_replay")),
            (execution, "reference_consumption_binding", self.forward_scoped(execution.reference_consumption_binding, "reference_binding")),
            (execution, "evaluate_residual", self.forward_scoped(execution.evaluate_residual, "independent_evaluation")),
        ]
        originals = [(obj, name, getattr(obj, name)) for obj, name, replacement in patches]
        try:
            for obj, name, replacement in patches:
                setattr(obj, name, replacement)
            yield self
        finally:
            for obj, name, original in reversed(originals):
                setattr(obj, name, original)
