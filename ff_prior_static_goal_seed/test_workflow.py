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
from ff_prior_static_fullchain_v2.solver import solve_step, restore_feasible, admissible_difference
from .experiment import assert_resume_safe


class WorkflowTests(unittest.TestCase):
    def test_short_ids_and_actual_windows_path_envelope(self):
        from .filesystem import worst_relative_paths
        from .protocol import OUT
        from .inputs import identity
        for path in worst_relative_paths():self.assertLess(len(str(OUT/path)),260)
        selection=identity()['selection']
        self.assertEqual([u['id'] for u in selection['units']],[f'u{k:02d}' for k in range(6)])
        self.assertEqual(len({(u['task_id'],u['seed'],u['method']) for u in selection['units']}),6)

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
        self.assertLessEqual(6*P['unit_coupled_cap'],P['coupled_cap'])
        self.assertLessEqual(6*P['unit_integration_cap'],P['integration_cap'])
        self.assertLessEqual(5*P['goal_coupled_cap']+P['repair_coupled_cap']+10,P['unit_coupled_cap'])
        self.assertLessEqual(5*P['goal_integration_cap']+P['repair_integration_cap']+10*512,P['unit_integration_cap'])
        self.assertLessEqual(P['coupled_cap'],P['P1_remaining_coupled'])
        self.assertLessEqual(P['integration_cap'],P['P1_remaining_integrations'])
        self.assertLessEqual(P['wall_s'],P['P1_remaining_wall_s'])
        self.assertLessEqual(P['output_bytes_cap']+P['P1_prior_reserved_bytes'],20*1024**3)
        self.assertEqual(6*P['proposals_per_unit'],30)
        self.assertEqual(6*2*13750,P['native_steps_cap'])

    def test_task_and_proposal_identity_is_original_public_five(self):
        from .inputs import identity
        from .protocol import ROOT
        new=identity()['selection'];old=read_json(ROOT/'ff_prior_static_fullchain_v2/evidence/selection.json')
        original=[e for e in old['entries'] if e['sample_id']=='capture_5_obstacles']
        self.assertEqual(new['entries'],original)
        self.assertEqual(original[0]['task']['task_hash'],'e66384242aaacf174ab1d7e642f0ba84b55ea580ab1b62951988409c75115496')
        for a,b in zip(new['units'],[u for u in old['units'] if u['task_id']=='capture_5_obstacles']):
            self.assertEqual(a['source_unit'],b['id'])
            self.assertEqual({k:v for k,v in a.items() if k not in ('id','source_unit')},{k:v for k,v in b.items() if k!='id'})

    def test_goal_direction_changes_endpoint_and_preserves_start(self):
        from .goal_seed import directions
        from ff_prior.spline import Reference
        basis=directions();self.assertEqual(basis.shape,(17,19,17))
        self.assertTrue(np.array_equal(basis[:,-1,:],np.eye(17)))
        self.assertEqual(np.linalg.matrix_rank(basis.reshape(17,-1)),17)
        for j in (0,8,16):
            r=Reference(basis[j]*.1,np.zeros(17),27.)
            self.assertLess(np.max(abs(r.evaluate(0))),1e-14)
            self.assertLess(np.max(abs(r.evaluate(27)-.1*np.eye(17)[j])),1e-14)
            self.assertLess(np.max(abs(r.evaluate(0,1))),1e-14)
            self.assertLess(np.max(abs(r.evaluate(27,1))),1e-14)

    def test_goal_step_descends_and_caps_actual_joint_delta(self):
        from .goal_seed import directions,descent_step
        j=np.random.default_rng(713).normal(size=(12,17));e=np.linspace(-3,4,12)
        d=descent_step(j,e)
        self.assertLess(float(e@j@d),0.)
        self.assertLess(float(np.linalg.norm(e+j@d)),np.linalg.norm(e))
        self.assertLessEqual(abs(np.einsum('i,ijk->jk',d,directions())).max(),P['goal_step_rad']+1e-15)
        self.assertTrue(np.array_equal(descent_step(np.zeros_like(j),e),np.zeros(17)))
        with self.assertRaisesRegex(ValueError,'NONFINITE'):descent_step(j,np.full(12,np.nan))

    def test_goal_acceptance_protects_each_previously_valid_group(self):
        from .goal_seed import accepts
        old=np.zeros(12);old[0]=3.;old[3]=.9
        new=old.copy();new[0]=2.;self.assertTrue(accepts(old,new))
        new[3]=1.01;self.assertFalse(accepts(old,new))
        self.assertFalse(accepts(old,old));self.assertFalse(accepts(old,np.full(12,np.nan)))

    def test_full_integration_replay_is_mandatory_and_finite(self):
        from .physical_audit import replay_agrees
        base=dict(qpos=0.,qvel=0.,act=0.,time=0.,qacc_warmstart=0.,integration=0.)
        self.assertTrue(replay_agrees(base,1e-10))
        for delta in (1e-5,np.nan,np.inf):
            self.assertFalse(replay_agrees(dict(base,integration=delta),1e-10))
        base.pop('integration');self.assertFalse(replay_agrees(base,1e-10))

    def test_whole_goal_iteration_budget_reserved(self):
        with tempfile.TemporaryDirectory() as td:
            b=Budget(Path(td),time.time()+100);b.begin_unit()
            q=SliceBudget(b,400,250000,time.time()+100)
            self.assertTrue(q.can_fit(24,24*512))
            for _ in range(377):q.consume(512)
            self.assertFalse(q.can_fit(24,24*512));self.assertEqual(b.calls,377)

if __name__=='__main__':unittest.main()
