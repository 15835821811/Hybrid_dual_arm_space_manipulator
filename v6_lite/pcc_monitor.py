"""50 Hz PCC/capsule/MuJoCo comparison monitor for V6.1-B.

The monitor consumes diagnostics already computed by the single online QP;
it performs no extra geometry queries and therefore cannot change the command.
It writes a machine-readable comparison plus three compact plots covering
distance, gradient disagreement, and minimum clearance.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

if TYPE_CHECKING:
    from v6_lite.hierarchical_qp import HierarchicalQPResult


@dataclass(frozen=True)
class PCCMonitorSample:
    time: float
    d_pcc: float
    d_capsule: float
    d_mujoco: float
    distance_error: float
    gradient_error: float
    capsule_gradient_error: float
    pcc_constraint_active_count: int
    capsule_constraint_active_count: int
    pcc_binding_constraint_count: int
    capsule_binding_constraint_count: int
    pcc_avoidance_intervention: float
    pcc_segment_id: int
    pcc_arc_length_m: float
    shape_clearance_latency_s: float


def _json_scalar(value: Any) -> Any:
    if isinstance(value, (float, np.floating)):
        result = float(value)
        return result if math.isfinite(result) else None
    if isinstance(value, (int, np.integer)):
        return int(value)
    return value


def _finite_min(values: np.ndarray) -> float | None:
    finite = values[np.isfinite(values)]
    return float(np.min(finite)) if finite.size else None


class PCCMonitor:
    """Collect and serialize shape-aware safety diagnostics at task rate."""

    def __init__(self, *, scenario_id: str) -> None:
        self.scenario_id = str(scenario_id)
        self.samples: list[PCCMonitorSample] = []

    def record(self, time_s: float, result: HierarchicalQPResult) -> None:
        self.samples.append(
            PCCMonitorSample(
                time=float(time_s),
                d_pcc=float(result.pcc_clearance_m),
                d_capsule=float(result.capsule_clearance_m),
                d_mujoco=float(result.mujoco_continuum_target_clearance_m),
                distance_error=float(result.pcc_mujoco_distance_error_m),
                gradient_error=float(result.pcc_mujoco_gradient_error_norm),
                capsule_gradient_error=float(
                    result.capsule_mujoco_gradient_error_norm
                ),
                pcc_constraint_active_count=int(
                    result.pcc_constraint_active_count
                ),
                capsule_constraint_active_count=int(
                    result.capsule_constraint_active_count
                ),
                pcc_binding_constraint_count=int(
                    result.pcc_binding_constraint_count
                ),
                capsule_binding_constraint_count=int(
                    result.capsule_binding_constraint_count
                ),
                pcc_avoidance_intervention=float(
                    result.pcc_avoidance_intervention
                ),
                pcc_segment_id=int(result.pcc_closest_segment_id),
                pcc_arc_length_m=float(result.pcc_closest_arclength_m),
                shape_clearance_latency_s=float(
                    result.shape_clearance_latency_s
                ),
            )
        )

    def to_dict(self) -> dict[str, Any]:
        rows = [
            {key: _json_scalar(value) for key, value in asdict(item).items()}
            for item in self.samples
        ]
        if not self.samples:
            return {
                "scenario_id": self.scenario_id,
                "sample_count": 0,
                "samples": [],
            }
        pcc = np.asarray([item.d_pcc for item in self.samples])
        capsule = np.asarray([item.d_capsule for item in self.samples])
        mujoco_distance = np.asarray([item.d_mujoco for item in self.samples])
        gradient = np.asarray([item.gradient_error for item in self.samples])
        latency = np.asarray(
            [item.shape_clearance_latency_s for item in self.samples]
        )
        return {
            "scenario_id": self.scenario_id,
            "sample_count": len(self.samples),
            "minimum_clearance_m": {
                "pcc": _finite_min(pcc),
                "capsule": _finite_min(capsule),
                "mujoco": _finite_min(mujoco_distance),
            },
            "pcc_constraint_active_count": int(
                sum(item.pcc_constraint_active_count for item in self.samples)
            ),
            "pcc_binding_constraint_count": int(
                sum(item.pcc_binding_constraint_count for item in self.samples)
            ),
            "capsule_constraint_active_count": int(
                sum(item.capsule_constraint_active_count for item in self.samples)
            ),
            "capsule_binding_constraint_count": int(
                sum(item.capsule_binding_constraint_count for item in self.samples)
            ),
            "pcc_avoidance_intervention_max": float(
                max(item.pcc_avoidance_intervention for item in self.samples)
            ),
            "gradient_error_norm_max": (
                float(np.max(gradient[np.isfinite(gradient)]))
                if np.any(np.isfinite(gradient))
                else None
            ),
            "shape_clearance_latency_ms": {
                "p50": float(1e3 * np.percentile(latency, 50.0)),
                "p95": float(1e3 * np.percentile(latency, 95.0)),
                "max": float(1e3 * np.max(latency)),
            },
            "samples": rows,
        }

    def write(self, output_dir: Path) -> dict[str, Any]:
        output = Path(output_dir)
        plot_dir = output / "plots"
        plot_dir.mkdir(parents=True, exist_ok=True)
        payload = self.to_dict()
        comparison_path = output / "clearance_compare.json"
        with comparison_path.open("w", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False)
        if self.samples:
            self._write_plots(plot_dir)
        return payload

    def _write_plots(self, plot_dir: Path) -> None:
        time_s = np.asarray([item.time for item in self.samples])
        pcc = np.asarray([item.d_pcc for item in self.samples])
        capsule = np.asarray([item.d_capsule for item in self.samples])
        mujoco_distance = np.asarray([item.d_mujoco for item in self.samples])

        figure, axis = plt.subplots(figsize=(9.0, 4.8), constrained_layout=True)
        axis.plot(time_s, 1e3 * pcc, label="PCC tube")
        axis.plot(time_s, 1e3 * capsule, label="Discrete capsules")
        axis.plot(time_s, 1e3 * mujoco_distance, label="MuJoCo geoms")
        axis.axhline(5.0, color="black", linestyle="--", linewidth=1.0)
        axis.set(xlabel="Time [s]", ylabel="Signed clearance [mm]")
        axis.grid(True, alpha=0.3)
        axis.legend()
        figure.savefig(plot_dir / "distance_comparison.png", dpi=180)
        plt.close(figure)

        figure, axis = plt.subplots(figsize=(9.0, 4.8), constrained_layout=True)
        axis.plot(
            time_s,
            [item.gradient_error for item in self.samples],
            label="PCC vs MuJoCo",
        )
        axis.plot(
            time_s,
            [item.capsule_gradient_error for item in self.samples],
            label="Capsule vs MuJoCo",
        )
        axis.set(xlabel="Time [s]", ylabel="Gradient difference norm")
        axis.grid(True, alpha=0.3)
        axis.legend()
        figure.savefig(plot_dir / "gradient_comparison.png", dpi=180)
        plt.close(figure)

        minima = [
            _finite_min(pcc),
            _finite_min(capsule),
            _finite_min(mujoco_distance),
        ]
        values = [np.nan if item is None else 1e3 * item for item in minima]
        figure, axis = plt.subplots(figsize=(7.0, 4.8), constrained_layout=True)
        axis.bar(["PCC", "Capsule", "MuJoCo"], values)
        axis.axhline(5.0, color="black", linestyle="--", linewidth=1.0)
        axis.set_ylabel("Minimum signed clearance [mm]")
        axis.grid(True, axis="y", alpha=0.3)
        figure.savefig(plot_dir / "minimum_clearance_comparison.png", dpi=180)
        plt.close(figure)
