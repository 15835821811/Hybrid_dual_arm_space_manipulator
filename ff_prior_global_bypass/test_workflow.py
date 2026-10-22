"""Software checks only; none certify a robot path."""
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from .tree import blend,minimum_duration,Node,Piecewise,collect,nearest,fit323
from .budget import Budget,SliceBudget,BudgetExhausted
from .protocol import P,OUT,ROOT
from .experiment import assert_resume_safe
from ff_prior_pilot_train.storage import atomic_json
from ff_prior.storage import read_json


class Tests(unittest.TestCase):
    def test_quintic_endpoint_and_derivative(self):
        self.assertEqual(blend(0.),0.);self.assertEqual(blend(1.),1.)
        for d in (1,2):self.assertTrue(np.array_equal(blend(np.array([0.,1.]),d),[0.,0.]))
        s=np.linspace(.01,.99,100);h=1e-5
        self.assertLess(np.max(abs((blend(s+h)-blend(s-h))/(2*h)-blend(s,1))),1e-8)
        self.assertLess(np.max(abs((blend(s+h,1)-blend(s-h,1))/(2*h)-blend(s,2))),1e-7)

    def test_actual_time_scale_respects_motion(self):
        rng=np.random.default_rng(9)
        for _ in range(20):
            dq=rng.normal(size=17);v=np.linspace(.1,1.,17);a=np.linspace(.2,2.,17)
            T=minimum_duration(dq,v,a);s=np.linspace(0,1,5000)
            self.assertTrue(np.all(abs(blend(s,1)[:,None]*dq/T)<=v))
            self.assertTrue(np.all(abs(blend(s,2)[:,None]*dq/T**2)<=a))

    def test_path_time_and_full_parent_identity(self):
        def node(value,t,parent):return Node(np.full(17,value),np.array([t]),np.array([t]),np.zeros(6),t,parent,t/2,np.zeros((2,4,4)),parent)
        ns=[node(0.,0.,-1),node(.1,2.,0),node(-.05,5.,1)]
        ns[1].duration=2.;ns[2].duration=3.
        path=collect(ns,2);self.assertEqual(len(path),3)
        ref=Piecewise(path,5.)
        self.assertTrue(np.allclose(ref.evaluate([0.,2.,5.]),np.array([0.,.1,-.05])[:,None]))
        for d in (1,2):self.assertLess(np.max(abs(ref.evaluate([0.,2.,5.],d))),1e-12)
        self.assertEqual(nearest(ns,np.full(17,-.05),np.ones(17)),2)

    def test_fit_preserves_exact_endpoints(self):
        q0=np.zeros(17);q1=np.linspace(-.01,.01,17)
        ns=[Node(q0,np.zeros(1),np.zeros(1),np.zeros(1),0.,-1,0.,np.zeros(1),-1),
            Node(q1,np.zeros(1),np.zeros(1),np.zeros(1),27.,0,27.,np.zeros(1),0)]
        robot=SimpleNamespace(lower=np.full(17,-1.),upper=np.ones(17))
        z,r,t,desired,fitted=fit323(ns,robot,27.)
        self.assertTrue(np.array_equal(z[-1],q1));self.assertTrue(np.array_equal(fitted[0],q0))
        self.assertLess(r['maximum_joint_fit_error_rad'],1e-5)

    def test_no_cross_slice_refunds(self):
        with tempfile.TemporaryDirectory() as td:
            b=Budget(Path(td),time.time()+60);b.begin_unit()
            q=SliceBudget(b,1,100,time.time()+60);q.consume(64)
            with self.assertRaises(BudgetExhausted):q.consume(1)
            r=SliceBudget(b,2,100,time.time()+60);r.consume(64)
            self.assertEqual(b.calls,2);self.assertEqual(b.integrations,128)
            b.observe({'coupled':2,'physics_steps':0});saved=b.save()
            self.assertEqual(saved['reserved_coupled'],saved['actual_coupled'])

    def test_total_and_exact_stage_costs(self):
        self.assertLessEqual(6*P['unit_coupled_cap']+60,36000)
        self.assertLessEqual(6*P['unit_integration_cap']+800000,P['integration_cap'])
        self.assertLessEqual(5*(P['seed_coupled_cap']+P['search_coupled_cap']+P['validation_coupled_cap'])
            +P['repair_coupled_cap']+P['validation_coupled_cap'],P['unit_coupled_cap'])
        self.assertLessEqual(5*(P['seed_integration_cap']+P['search_integration_cap']+P['validation_integration_cap'])
            +P['repair_integration_cap']+P['validation_integration_cap'],P['unit_integration_cap'])
        self.assertLessEqual(6*P['unit_wall_s']+360,P['wall_s'])
        self.assertLessEqual(5*P['candidate_wall_s']+P['repair_wall_s']+P['execution_reserve_s'],P['unit_wall_s'])
        self.assertEqual(6*27500,P['native_steps_cap'])

    def test_original_task_and_rng(self):
        from .inputs import identity
        new=identity()['selection'];old=read_json(ROOT/'ff_prior_static_fullchain_v2/evidence/selection.json')
        self.assertEqual(new['entries'],[e for e in old['entries'] if e['sample_id']=='capture_5_obstacles'])
        original=[u for u in old['units'] if u['task_id']=='capture_5_obstacles']
        for a,b in zip(new['units'],original):
            self.assertEqual(a['proposal_seeds'],b['proposal_seeds']);self.assertEqual(a['method'],b['method'])
        self.assertEqual(new['entries'][0]['task']['condition']['duration'],27.)

    def test_unknown_work_no_replay(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td);atomic_json(p/'ledger.json',{'inflight':{'id':'x'}})
            with self.assertRaisesRegex(ValueError,'UNKNOWN_WORK'):assert_resume_safe(p)

    def test_actual_path_envelope(self):
        paths=['units/u05/candidates/04/tree/goal_connections/0399/iteration_05.npz',
            'units/u05/candidates/04/repair/iterations/05/linearization.npz',
            'units/u05/candidates/04/before_repair/grid_16384_geometry_scans.json']
        for p in paths:self.assertLess(len(str(OUT/p)),258)

    def test_full_replay_rejects_missing_nan_or_error(self):
        from ff_prior_static_goal_seed.physical_audit import replay_agrees
        self.assertTrue(replay_agrees({'qpos':0.,'integration':0.},1e-10))
        for r in ({'qpos':0.},{'integration':np.nan},{'integration':1e-5}):
            self.assertFalse(replay_agrees(r,1e-10))

    def test_no_answer_or_training_reads_in_new_planner(self):
        from .protocol import PACKAGE
        for name in ('tree.py','experiment.py','inputs.py'):
            s=(PACKAGE/name).read_text(encoding='utf-8')
            for token in ('capture_trust_001','07_bow','teachers/','optimizer.step','learning/train'):
                self.assertNotIn(token,s)


if __name__=='__main__':unittest.main()
