"""Synthetic arithmetic checks only: no robot, geometry query or dynamics."""
import unittest
import numpy as np
from ff_prior_waypoint_domain.adapter import velocity_bounds
from .braking import (braking_velocity_bounds,stopping_distance,emergency_profile,
    TICK_S,SUBSTEP_S,SUBSTEPS,ALPHA,BETA,TRACKING_ENVELOPE_RAD)


class BrakingArithmetic(unittest.TestCase):
    def test_exact_ten_substep_ramp_coefficients(self):
        old=np.array([-.4,.2,.8]);new=np.array([.3,-.1,.5])
        direct=sum(SUBSTEP_S*(old+k/SUBSTEPS*(new-old)) for k in range(1,SUBSTEPS+1))
        np.testing.assert_allclose(direct,ALPHA*old+BETA*new,atol=1e-17)
        self.assertNotEqual(ALPHA,BETA)

    def test_exact_stopping_sum_including_fractional_final_tick(self):
        for a in (.3,.5,1.2):
            for u in (0.,.005,.01,.045,.12,.3454632498508976,-.12,-.347):
                value=u;total=0.
                while abs(value)>1e-15:
                    end=np.sign(value)*max(abs(value)-a*TICK_S,0.)
                    for k in range(1,SUBSTEPS+1):
                        total+=SUBSTEP_S*(value+k/SUBSTEPS*(end-value))
                    value=end
                self.assertAlmostEqual(float(stopping_distance(u,a)),total,places=14)

    def test_monotone_future_barrier_envelope(self):
        r=np.array([0.,.1,-.3]);v=np.array([.4,-.4,.2]);a=np.array([.5,.3,1.])
        candidates=np.linspace(-.6,.6,81)
        mins=[];maxs=[]
        for x in candidates:
            _,_,b=emergency_profile(r,v,np.full(3,x),a,102)
            mins.append(b.min(axis=0));maxs.append(b.max(axis=0))
        self.assertTrue(np.all(np.diff(mins,axis=0)>=-1e-15))
        self.assertTrue(np.all(np.diff(maxs,axis=0)>=-1e-15))

    def test_tightens_only_and_preserves_each_original_term(self):
        q=np.array([.69,-.69,.05]);r=q.copy();v=np.array([.4,-.4,.01])
        lower=np.full(3,-1.);upper=np.ones(3);vmax=np.full(3,.7);acc=np.full(3,.5)
        oldlo,oldhi=velocity_bounds(q,r,v,lower,upper,vmax,acc)
        lo,hi,d=braking_velocity_bounds(q,r,v,lower,upper,vmax,acc)
        self.assertTrue(np.all(lo>=oldlo));self.assertTrue(np.all(hi<=oldhi))
        c=d['original_components']
        rebuiltlo=np.maximum.reduce([c['velocity_lower'],c['measured_barrier_lower'],
            c['acceleration_lower'],c['reference_substep_lower'].max(axis=0)])
        rebuilthi=np.minimum.reduce([c['velocity_upper'],c['measured_barrier_upper'],
            c['acceleration_upper'],c['reference_substep_upper'].min(axis=0)])
        np.testing.assert_array_equal(rebuiltlo,oldlo);np.testing.assert_array_equal(rebuilthi,oldhi)

    def test_preempts_future_failure_and_all_emergency_ticks_obey_old_bounds(self):
        lower=np.array([-1.]);upper=np.array([1.]);vmax=np.array([.7]);acc=np.array([.5])
        r=np.array([1.-.04-.02-.18]);q=r.copy();previous=np.array([.4])
        lo,hi,d=braking_velocity_bounds(q,r,previous,lower,upper,vmax,acc)
        self.assertLessEqual(lo[0],hi[0]);self.assertLess(hi[0],d['original_upper'][0])
        for candidate in (lo,hi):
            position,velocity,_=emergency_profile(r,previous,candidate,acc,d['future_ticks_checked']-1)
            for j,reference in enumerate(position):
                for error in (-TRACKING_ENVELOPE_RAD,TRACKING_ENVELOPE_RAD):
                    l,h=velocity_bounds(reference+error,reference,velocity[j],lower,upper,vmax,acc)
                    self.assertTrue(np.all(velocity[j+1]>=l-2e-14))
                    self.assertTrue(np.all(velocity[j+1]<=h+2e-14))
            self.assertTrue(np.all(position>=lower+.04-2e-14))
            self.assertTrue(np.all(position<=upper-.04+2e-14))

    def test_actual_failure_interval_cannot_be_repaired_retroactively(self):
        # The two recorded index5 limits are instantiated algebraically, not
        # used as a robot initial state or a new evaluation of the failed run.
        recorded_lo=.3454632498508976;recorded_hi=.3447324683300863
        lower=np.array([-1.]);upper=np.array([1.]);a=np.array([.5]);vmax=np.array([.7])
        previous=np.array([recorded_lo+a[0]*TICK_S]);q=np.array([1.-.04-recorded_hi/4])
        lo,hi,d=braking_velocity_bounds(q,q,previous,lower,upper,vmax,a)
        self.assertAlmostEqual(d['original_lower'][0],recorded_lo,places=14)
        self.assertAlmostEqual(d['original_upper'][0],recorded_hi,places=14)
        self.assertGreater(lo[0],hi[0]);self.assertFalse(d['original_constraints_relaxed'])

    def test_symmetry_and_no_mutation(self):
        r=np.array([.77,.0]);q=r+np.array([.001,-.001]);v=np.array([.4,.03])
        lower=np.full(2,-1.);upper=np.ones(2);vmax=np.full(2,.7);a=np.full(2,.5)
        inputs=[q,r,v,lower,upper,vmax,a];before=[x.copy() for x in inputs]
        lo,hi,_=braking_velocity_bounds(*inputs)
        ml,mh,_=braking_velocity_bounds(-q,-r,-v,-upper,-lower,vmax,a)
        np.testing.assert_allclose(lo,-mh,atol=3e-16);np.testing.assert_allclose(hi,-ml,atol=3e-16)
        for x,y in zip(inputs,before):np.testing.assert_array_equal(x,y)

    def test_tracking_prerequisite_violation_stays_visible(self):
        lo,hi,d=braking_velocity_bounds(np.array([.021]),np.array([0.]),np.array([0.]),
            np.array([-1.]),np.array([1.]),np.array([.7]),np.array([.5]))
        self.assertGreater(lo[0],hi[0]);self.assertFalse(d['current_tracking_envelope_satisfied'][0])
        self.assertFalse(d['geometry_torque_task_feasibility_guaranteed'])

    def test_position_margin_alone_is_not_the_measured_barrier_certificate(self):
        # A joint can have enough room to stop, yet violate the old measured
        # barrier at an intermediate braking tick. The envelope tests both.
        _,_,b=emergency_profile(np.array([0.]),np.array([0.]),np.array([.2]),np.array([.5]),41)
        position,_,_=emergency_profile(np.array([0.]),np.array([0.]),np.array([.2]),np.array([.5]),41)
        self.assertGreater(b.max(),position[-1,0])


if __name__=='__main__':unittest.main()
