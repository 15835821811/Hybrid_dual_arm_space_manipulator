"""Private SPD solve for the 17D weighted-QP diagnostic unconstrained point.

The weighted objective, constraint assembly, ADMM solver and action contract
are unchanged. HierarchicalQP.solve uses this unconstrained point only to
report PCC intervention; its candidate starts from the previous command.
This class is never selected by the production controller.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import cho_factor, cho_solve

from v6_lite.b2_optimized_admm import OptimizedScreenQP


CHOLESKY_TELEMETRY: list[dict] = []


class CholeskyUnconstrainedScreenQP(OptimizedScreenQP):
    def solve(self, *args, **kwargs):
        original_solve = np.linalg.solve
        call_count = 0
        fallback_count = 0
        maximum_residual = 0.0

        def private_spd_solve(matrix, rhs):
            nonlocal call_count, fallback_count, maximum_residual
            if np.shape(matrix) != (17, 17) or np.shape(rhs) != (17,):
                return original_solve(matrix, rhs)
            call_count += 1
            try:
                factor = cho_factor(matrix, lower=True, check_finite=False)
                value = cho_solve(factor, rhs, check_finite=False)
            except np.linalg.LinAlgError:
                fallback_count += 1
                value = original_solve(matrix, rhs)
            maximum_residual = max(maximum_residual,
                                   float(np.max(np.abs(matrix @ value - rhs))))
            return value

        try:
            np.linalg.solve = private_spd_solve
            result = super().solve(*args, **kwargs)
        finally:
            np.linalg.solve = original_solve
        CHOLESKY_TELEMETRY.append({
            "call_count": call_count,
            "fallback_count": fallback_count,
            "maximum_linear_residual": maximum_residual,
        })
        return result
