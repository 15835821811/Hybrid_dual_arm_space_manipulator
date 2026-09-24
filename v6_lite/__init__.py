"""V6-lite deterministic dual-arm tracking and avoidance controller."""

from .hierarchical_qp import HierarchicalQPConfig, HierarchicalQPResult, HierarchicalVelocityQP
from .continuum_jacobian import compute_position_jacobian
from .continuum_shape_model import ContinuumPoint, ContinuumShapeModel
from .pcc_clearance import PCCClearanceEvaluator, PCCClearanceResult

__all__ = [
    "HierarchicalQPConfig",
    "HierarchicalQPResult",
    "HierarchicalVelocityQP",
    "ContinuumPoint",
    "ContinuumShapeModel",
    "compute_position_jacobian",
    "PCCClearanceEvaluator",
    "PCCClearanceResult",
]
