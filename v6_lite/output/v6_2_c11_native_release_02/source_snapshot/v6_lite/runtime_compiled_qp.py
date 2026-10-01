"""C.1.1 compiled execution of the same ADMM recurrence and stopping gates.

No fast-math, row reduction, relaxed tolerance, or truncated iteration count.
Triangular-solve arithmetic may differ at roundoff from SciPy; this is a
declared numerical implementation change requiring new physical acceptance.
Compilation is exercised before fresh startup state acquisition.
"""
import numpy as np
from numba import njit
from v6_lite.b2_interval_online_optimized import OptimizedBoundedIntervalVelocityQP


@njit(cache=True, fastmath=False)
def _triangular_solve(factor, rhs):
    n = len(rhs)
    result = np.empty(n)
    for i in range(n):
        value = rhs[i]
        for j in range(i):
            value -= factor[i, j] * result[j]
        result[i] = value / factor[i, i]
    for i in range(n - 1, -1, -1):
        value = result[i]
        for j in range(i + 1, n):
            value -= factor[j, i] * result[j]
        result[i] = value / factor[i, i]
    return result


@njit(cache=True, fastmath=False)
def solve_admm(hessian, linear, matrix, lower, upper, initial, initial_dual,
               rho, sigma, relaxation, ftol, max_iterations, feasibility_tolerance):
    gram = matrix.T @ matrix
    identity = np.eye(17)
    factor = np.linalg.cholesky(hessian + sigma * identity + rho * gram)
    value = initial.copy()
    product = matrix @ value
    dual = initial_dual.copy()
    auxiliary = np.minimum(np.maximum(product + dual / rho, lower), upper)
    linear_scale = np.max(np.abs(linear))
    solved = False
    updates = 0
    for iteration in range(1, max_iterations + 1):
        rhs = sigma * value - linear + matrix.T @ (rho * auxiliary - dual)
        value = _triangular_solve(factor, rhs)
        product = matrix @ value
        previous_auxiliary = auxiliary
        relaxed = relaxation * product + (1.0 - relaxation) * previous_auxiliary
        auxiliary = np.minimum(np.maximum(relaxed + dual / rho, lower), upper)
        dual += rho * (relaxed - auxiliary)
        primal_residual = np.max(np.abs(product - auxiliary))
        hessian_product = hessian @ value
        transpose_dual = matrix.T @ dual
        dual_residual = np.max(np.abs(hessian_product + linear + transpose_dual))
        primal_scale = max(1.0, np.max(np.abs(product)), np.max(np.abs(auxiliary)))
        dual_scale = max(1.0, np.max(np.abs(hessian_product)), np.max(np.abs(transpose_dual)), linear_scale)
        if primal_residual <= ftol * primal_scale and dual_residual <= 5.0 * ftol * dual_scale:
            solved = True
            break
        if iteration % 100 == 0 and iteration < max_iterations:
            primal_ratio = primal_residual / (ftol * primal_scale)
            dual_ratio = dual_residual / (5.0 * ftol * dual_scale)
            next_rho = rho
            if dual_ratio > 3.0 * primal_ratio:
                next_rho = max(rho / 5.0, .1)
            elif primal_ratio > 3.0 * dual_ratio:
                next_rho = min(rho * 5.0, 500.)
            if next_rho != rho:
                updates += 1
                rho = next_rho
                factor = np.linalg.cholesky(hessian + sigma * identity + rho * gram)
    feasibility = np.minimum(product - lower, upper - product)
    return value, np.min(feasibility) >= -feasibility_tolerance, solved, iteration, dual, updates


class CompiledBoundedIntervalVelocityQP(OptimizedBoundedIntervalVelocityQP):
    def _solve_qp_admm(self, hessian, linear, matrix, lower, upper, initial, initial_dual=None):
        original_rows = len(lower)
        matrix = np.vstack((matrix, np.eye(17)[:10]))
        lower = np.concatenate((lower, self._domain_rate_lower))
        upper = np.concatenate((upper, self._domain_rate_upper))
        dual = np.zeros(original_rows + 10) if initial_dual is None else np.concatenate((initial_dual, np.zeros(10)))
        if dual.shape != (len(lower),) or not np.all(np.isfinite(dual)):
            raise ValueError("ADMM dual warm start has the wrong shape or is non-finite")
        cfg = self.config
        value, feasible, solved, iteration, dual, updates = solve_admm(
            np.ascontiguousarray(hessian), np.ascontiguousarray(linear), matrix,
            lower, upper, np.asarray(initial).copy(), dual, cfg.admm_rho,
            cfg.admm_sigma, cfg.admm_relaxation, cfg.qp_ftol, cfg.qp_max_iterations,
            cfg.feasibility_tolerance)
        self.last_penalty_update_count = int(updates)
        return value, bool(feasible), "solved" if solved else "maximum_iterations", int(iteration), dual[:original_rows]
