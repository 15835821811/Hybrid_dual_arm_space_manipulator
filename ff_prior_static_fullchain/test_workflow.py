import ast
import tempfile
import time
import unittest
from pathlib import Path
import numpy as np
import torch
from ff_prior.storage import read_json, digest
from ff_prior_pilot_train.storage import atomic_json
from .budget import Budget, SliceBudget, BudgetExhausted
from .protocol import P, PACKAGE
from .proposals import transverse_frame, load_bundle
from .solver import solve_step, restore_feasible, admissible_difference
from .experiment import assert_resume_safe


class WorkflowTests(unittest.TestCase):
    def test_counter_survives_robot_and_unit_changes(self):
        with tempfile.TemporaryDirectory() as td:
            b=Budget(Path(td),time.time()+60)
            b.begin_unit();b.consume(512);b.observe({'coupled':1,'physics_steps':20})
            b.begin_unit();b.consume(256);b.observe({'coupled':1,'physics_steps':14})
            s=b.save();self.assertEqual((s['actual_coupled'],s['integration_intervals'],s['actual_native_and_replay_steps']),(2,768,34))
            c=Budget(Path(td),time.time()+60,s);c.begin_unit();c.consume(128);c.observe({'coupled':1})
            self.assertEqual(c.save()['actual_coupled'],3)

    def test_slice_quota_and_no_refund(self):
        with tempfile.TemporaryDirectory() as td:
            b=Budget(Path(td),time.time()+60);b.begin_unit()
            s=SliceBudget(b,1,512,time.time()+60);s.consume(512)
            with self.assertRaises(BudgetExhausted):s.consume(1)
            self.assertEqual(b.calls,1)
            s2=SliceBudget(b,1,64,time.time()+60)
            with self.assertRaises(BudgetExhausted):s2.consume(65)
            self.assertEqual(b.integrations,512)

    def test_whole_unit_quota_not_reset_by_slice(self):
        with tempfile.TemporaryDirectory() as td:
            b=Budget(Path(td),time.time()+60);b.begin_unit();b.calls=P['unit_coupled_cap']
            s=SliceBudget(b,100,100000,time.time()+60)
            with self.assertRaises(BudgetExhausted):s.consume(1)

    def test_unknown_work_never_replayed(self):
        with tempfile.TemporaryDirectory() as td:
            d=Path(td);atomic_json(d/'ledger.json',{'inflight':{'id':'x'},'calls':{}})
            with self.assertRaisesRegex(ValueError,'UNKNOWN_WORK'):assert_resume_safe(d)
            atomic_json(d/'ledger.json',{'inflight':None,'calls':{'a.coupled':2}})
            atomic_json(d/'budget.json',{'actual_coupled':1})
            with self.assertRaisesRegex(ValueError,'UNRESOLVED_BUDGET'):assert_resume_safe(d)

    def test_public_transverse_frame(self):
        for d in (np.zeros(3),np.array([1.,2.,3.]),np.array([0.,0.,1.])):
            a,b=transverse_frame(d)
            self.assertAlmostEqual(np.linalg.norm(a),1.);self.assertAlmostEqual(np.linalg.norm(b),1.)
            self.assertAlmostEqual(a@b,0.)
            self.assertAlmostEqual(a@d,0.);self.assertAlmostEqual(b@d,0.)

    def test_no_historical_answer_loading(self):
        for name in ('inputs.py','proposals.py','experiment.py'):
            text=(PACKAGE/name).read_text()
            for forbidden in ('capture_trust_001','capture_refine_001','07_bow','final_path.npz','teachers/'):
                self.assertNotIn(forbidden,text)
        tree=ast.parse((PACKAGE/'proposals.py').read_text())
        for node in ast.walk(tree):
            if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute):
                self.assertNotEqual(node.func.attr,'load')

    def test_full_model_exact_low_transfer(self):
        from ff_prior_sample_alignment.learning import load
        from .protocol import SOURCE
        model,codec=load_bundle();old,low=load(SOURCE/'learning/aligned')
        x=torch.randn(2,323,dtype=torch.float64);c=torch.randn(2,65);t=torch.tensor([333,999])
        with torch.no_grad():v,detail=model.components(x,t,c);expected=old(x[:,:102].float(),t,c)
        self.assertTrue(torch.equal(v,expected));self.assertEqual(detail.count_nonzero(),0)
        self.assertEqual(codec.dim,323)
        self.assertTrue(np.array_equal(codec.basis[:,:6],low.basis))

    def test_small_feasible_solver_uses_actual_control_scale(self):
        z=np.zeros((19,17));directions=np.zeros((2,19,17));directions[0,:,0]=1.;directions[1,:,1]=1.
        e=np.zeros(12);jg=np.zeros((12,2));jd=np.array([[.04,0.]])
        step,record=solve_step(z,directions,e,np.array([-.003]),jg,jd,-np.ones(17),np.ones(17),.0375)
        delta=np.einsum('i,ijk->jk',step,directions)
        self.assertLessEqual(abs(delta).max(),.0375+1e-8)
        self.assertTrue(record['returned_feasible']);self.assertGreater(step[0],0.)

    def test_finite_difference_respects_joint_bounds(self):
        z=np.zeros((19,17));z[0,0]=1.;d=np.zeros_like(z);d[0,0]=1.
        h=admissible_difference(z,d,-np.ones(17),np.ones(17))
        self.assertLess(h,0.);self.assertLessEqual((z+h*d).max(),1.)

    def test_restore_does_not_return_infeasible_solver_point(self):
        f=lambda x:np.isfinite(x).all() and abs(x).max()<=.2
        result,a=restore_feasible(np.array([2.,-1.]),f)
        self.assertTrue(f(result));self.assertLess(a,1.)

    def test_fixed_allocations_fit_global_bounds(self):
        self.assertLessEqual(12*P['unit_coupled_cap'],P['coupled_cap'])
        self.assertLessEqual(12*P['unit_integration_cap'],P['integration_cap'])
        self.assertLess(2*P['repair_coupled_cap']+40,P['unit_coupled_cap'])
        self.assertEqual(2*3*2*P['proposals_per_unit'],60)
        self.assertEqual(12*2*13750,330000)

if __name__=='__main__':unittest.main()
