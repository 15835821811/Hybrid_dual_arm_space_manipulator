"""Check constrained ADMM outcomes and the unchanged acceptance tolerances."""
import unittest
import numpy as np
from threadpoolctl import threadpool_limits
from v6_lite.hierarchical_qp import HierarchicalQPConfig
from v6_lite.b2_interval_online_optimized import OptimizedBoundedIntervalVelocityQP
from v6_lite.runtime_compiled_qp import CompiledBoundedIntervalVelocityQP


class CompiledQPTests(unittest.TestCase):
    def test_active_constraints_warm_duals_and_penalty_updates_match_reference(self):
        with threadpool_limits(1):
            rng = np.random.default_rng(20261001)
            for scale in (1., 10000.):
                matrix = np.vstack((np.eye(17), rng.normal(size=(18, 17))))
                hessian = np.diag(np.linspace(1., scale, 17))
                linear = -hessian @ rng.uniform(-.5, .5, 17)
                lower, upper = np.full(35, -.12), np.full(35, .12)
                reference = object.__new__(OptimizedBoundedIntervalVelocityQP)
                compiled = object.__new__(CompiledBoundedIntervalVelocityQP)
                for obj in (reference, compiled):
                    obj.config = HierarchicalQPConfig()
                    obj._domain_rate_lower = np.full(10, -.2)
                    obj._domain_rate_upper = np.full(10, .2)
                for dual in (None, np.zeros(35)):
                    a = reference._solve_qp_admm(hessian, linear, matrix, lower, upper, np.zeros(17), dual)
                    b = compiled._solve_qp_admm(hessian, linear, matrix, lower, upper, np.zeros(17), dual)
                    self.assertEqual(a[1:4], b[1:4])
                    np.testing.assert_allclose(a[0], b[0], atol=1e-9, rtol=0.)
                    self.assertEqual(reference.last_penalty_update_count, compiled.last_penalty_update_count)
                    self.assertTrue(np.min(matrix @ b[0] - lower) >= -compiled.config.feasibility_tolerance)
                    self.assertTrue(np.min(upper - matrix @ b[0]) >= -compiled.config.feasibility_tolerance)


if __name__ == "__main__":
    unittest.main()
