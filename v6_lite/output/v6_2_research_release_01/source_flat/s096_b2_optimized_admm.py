"""Private 17D ADMM variant that reuses residual matrix products per step.

The objective, constraints, penalties, warm starts, stopping thresholds and
rho adaptation match the original weighted velocity QP. This class is for
parity and timing trials, not the production online controller.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import cho_factor, cho_solve

from v6_lite.audit_b2_private_qp_sphere import _ScreenQP


class OptimizedScreenQP(_ScreenQP):
    def _solve_qp_admm(self, hessian, linear, matrix, lower, upper,
                       initial, initial_dual=None):
        cfg = self.config
        rho = cfg.admm_rho
        sigma = cfg.admm_sigma
        gram = matrix.T @ matrix
        identity = np.eye(17)
        system = hessian + sigma * identity + rho * gram
        factor = cho_factor(system, lower=True, check_finite=False)
        value = np.asarray(initial, dtype=np.float64).copy()
        product = matrix @ value
        dual = (np.zeros(matrix.shape[0], dtype=np.float64)
                if initial_dual is None else
                np.asarray(initial_dual, dtype=np.float64).copy())
        if dual.shape != (matrix.shape[0],) or np.any(~np.isfinite(dual)):
            raise ValueError("ADMM dual warm start has the wrong shape or is non-finite")
        auxiliary = np.minimum(np.maximum(product + dual / rho, lower), upper)
        linear_scale = float(np.max(np.abs(linear)))
        status = "maximum_iterations"
        for iteration in range(1, cfg.qp_max_iterations + 1):
            rhs = sigma * value - linear + matrix.T @ (rho * auxiliary - dual)
            value = cho_solve(factor, rhs, check_finite=False)
            product = matrix @ value
            previous_auxiliary = auxiliary
            relaxed = (cfg.admm_relaxation * product
                       + (1.0 - cfg.admm_relaxation) * previous_auxiliary)
            auxiliary = np.minimum(
                np.maximum(relaxed + dual / rho, lower), upper)
            dual += rho * (relaxed - auxiliary)
            primal_residual = float(np.max(np.abs(product - auxiliary)))
            hessian_product = hessian @ value
            transpose_dual = matrix.T @ dual
            dual_residual = float(np.max(np.abs(
                hessian_product + linear + transpose_dual)))
            primal_scale = max(
                1.0, float(np.max(np.abs(product))),
                float(np.max(np.abs(auxiliary))))
            dual_scale = max(
                1.0, float(np.max(np.abs(hessian_product))),
                float(np.max(np.abs(transpose_dual))), linear_scale)
            if (primal_residual <= cfg.qp_ftol * primal_scale
                    and dual_residual <= 5.0 * cfg.qp_ftol * dual_scale):
                status = "solved"
                break
            if iteration % 100 == 0 and iteration < cfg.qp_max_iterations:
                primal_ratio = primal_residual / (cfg.qp_ftol * primal_scale)
                dual_ratio = dual_residual / (5.0 * cfg.qp_ftol * dual_scale)
                next_rho = rho
                if dual_ratio > 3.0 * primal_ratio:
                    next_rho = max(rho / 5.0, 0.1)
                elif primal_ratio > 3.0 * dual_ratio:
                    next_rho = min(rho * 5.0, 500.0)
                if next_rho != rho:
                    rho = next_rho
                    system = hessian + sigma * identity + rho * gram
                    factor = cho_factor(system, lower=True, check_finite=False)
        feasibility = np.minimum(product - lower, upper - product)
        feasible = bool(float(np.min(feasibility)) >= -cfg.feasibility_tolerance)
        return value, feasible, status, iteration, dual
