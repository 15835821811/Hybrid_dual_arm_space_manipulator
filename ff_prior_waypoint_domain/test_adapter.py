import unittest
import numpy as np
from .adapter import velocity_bounds


class DomainTests(unittest.TestCase):
    def bounds(self,q,reference,previous):
        return velocity_bounds(q,reference,previous,-np.ones(17),np.ones(17),np.ones(17),np.ones(17)*2)

    def test_ten_substeps_satisfy_reference_domain(self):
        rng=np.random.default_rng(20261020)
        feasible=0
        for _ in range(1000):
            q=rng.uniform(-.95,.95,17);ref=q+rng.uniform(-.005,.005,17);prev=rng.uniform(-.15,.15,17)
            lo,hi=self.bounds(q,ref,prev)
            if np.any(lo>hi):continue
            feasible+=1
            for velocity in (lo,hi,(lo+hi)/2):
                p=ref.copy()
                for k in range(1,11):
                    p+=.002*(prev+k/10*(velocity-prev))
                    self.assertTrue(np.all(p>=-.96-1e-12) and np.all(p<=.96+1e-12))
        self.assertGreater(feasible,900)

    def test_actual_state_near_upper_bound_forces_inward(self):
        q=np.full(17,.97);lo,hi=self.bounds(q,q,np.zeros(17))
        self.assertTrue(np.all(hi<0))

    def test_reference_near_bound_constrains_even_if_actual_interior(self):
        lo,hi=self.bounds(np.zeros(17),np.full(17,.9599),np.zeros(17))
        self.assertTrue(np.all(hi<.01))

    def test_infeasible_outward_momentum_is_not_clipped_position(self):
        lo,hi=self.bounds(np.full(17,.999),np.full(17,.999),np.full(17,.5))
        self.assertTrue(np.any(lo>hi))


if __name__=="__main__":unittest.main()
