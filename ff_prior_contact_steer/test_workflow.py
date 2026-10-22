"""Algebra, scheduling and identity tests only; not robot feasibility evidence."""
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from .protocol import P,ROOT,OUT,PACKAGE
from .tree import Node,blend,minimum_duration,Piecewise,fit323
from .contact import exact_key,bounded_world,solve_direction,tangent,differences
from .budget import Budget,SliceBudget,BudgetExhausted
from .experiment import assert_resume_safe
from ff_prior_pilot_train.storage import atomic_json
from ff_prior.storage import read_json,file_digest


class Tests(unittest.TestCase):
    def test_quintic_derivatives_and_motion_time(self):
        s=np.linspace(.001,.999,1000);h=1e-5
        self.assertLess(np.max(abs((blend(s+h)-blend(s-h))/(2*h)-blend(s,1))),1e-8)
        self.assertLess(np.max(abs((blend(s+h,1)-blend(s-h,1))/(2*h)-blend(s,2))),1e-7)
        dq=np.linspace(-.12,.12,17);v=np.linspace(.1,.5,17);a=np.linspace(.2,.6,17)
        T=minimum_duration(dq,v,a)
        self.assertTrue(np.all(abs(blend(s,1)[:,None]*dq/T)<=v))
        self.assertTrue(np.all(abs(blend(s,2)[:,None]*dq/T**2)<=a))
        self.assertEqual(float(blend(1.,1)),0.)

    def test_dedup_requires_entire_state_time_action_and_context(self):
        n=SimpleNamespace(state=np.arange(836,dtype=float),q=np.zeros(17),qpos=np.zeros(83),qvel=np.zeros(80),elapsed=15.)
        c=dict(policy='a',model='b',grids=[32,64],source='c');target=np.ones(17)
        key=exact_key(n,c,target,1.)
        self.assertEqual(key,exact_key(n,c,target.copy(),1.))
        for attribute in ('state','q','qpos','qvel'):
            a=getattr(n,attribute);before=a[-1].copy();a[-1]+=1e-12
            self.assertNotEqual(key,exact_key(n,c,target,1.));a[-1]=before
        n.elapsed+=1.;self.assertNotEqual(key,exact_key(n,c,target,1.));n.elapsed-=1.
        self.assertNotEqual(key,exact_key(n,dict(c,policy='d'),target,1.))
        self.assertNotEqual(key,exact_key(n,c,target,1.1))
        self.assertNotEqual(key,exact_key(n,c,target+.001,1.))

    def test_joint_direction_solves_collision_instead_of_goal_only(self):
        n=np.zeros(17);nom=n.copy();nom[0]=-.1;nom[1]=.05
        jg=np.zeros((12,17));jg[0,0]=20.;jg[1,1]=20.
        e=np.zeros(12);e[1]=-1.
        jd=np.zeros((2,17));jd[0,0]=1.;jd[1,1]=-1.
        ds=np.array([-.09,.06])
        d,r=solve_direction(n,nom,e,jg,ds,jd,np.full(17,-1.),np.ones(17),0)
        self.assertTrue(r['linear_admissible'])
        self.assertGreaterEqual(float(np.min(ds+jd@(d-nom))),P['contact_margin_m']-1e-8)
        self.assertGreater(d[0],nom[0]);self.assertGreater(d[1],0.)
        self.assertLessEqual(max(abs(d)),P['contact_trust_rad']+1e-9)

    def test_inconsistent_distance_constraints_are_not_qualified(self):
        j=np.zeros((12,17));jd=np.zeros((1,17))
        _,r=solve_direction(np.zeros(17),np.zeros(17),np.zeros(12),j,np.array([-.1]),jd,
            -np.ones(17),np.ones(17),0)
        self.assertFalse(r['linear_admissible'])

    def test_tangent_is_orthogonal_to_obstacle_and_goal(self):
        jd=np.zeros((2,17));jd[0,0]=1.;jd[1,2]=1.;g=np.zeros(17);g[1]=1.
        t=tangent(jd,np.array([0.,.01]),g)
        self.assertAlmostEqual(float(t@jd[0]),0.);self.assertAlmostEqual(float(t@g),0.)
        self.assertGreater(np.linalg.norm(t),0.)

    def test_difference_boundary_and_world_radius(self):
        lo=-np.ones(17);hi=np.ones(17);q=hi.copy();h=differences(q,lo,hi)
        self.assertTrue(np.all(q+h<=hi));self.assertTrue(np.all(q+h>=lo))
        rng=np.random.default_rng(10);j=rng.normal(size=(12,17));e=rng.normal(size=12)
        d=bounded_world(j,e)
        self.assertLessEqual(max(abs(d)),P['contact_trust_rad']+1e-12)
        self.assertLess(np.linalg.norm(e+j@d),np.linalg.norm(e))

    def test_tangent_uses_geometry_sensitive_coordinates(self):
        jd=np.zeros((2,17));jd[0,10:12]=[1.,1.];jd[1,12]=.2
        g=np.zeros(17);g[14]=1.
        t=tangent(jd,np.array([0.,.01]),g)
        self.assertLess(np.max(abs(t[:10])),1e-12)
        self.assertGreater(np.linalg.norm(t[10:]),0.)

    def test_worst_query_and_interval_reservations(self):
        linear=18+18+1;edge=len(P['edge_grids']);variants=len(P['contact_variant_ids']);scales=len(P['contact_line_scales'])
        worst=linear+variants*scales*edge+variants*edge
        self.assertEqual(worst,109);self.assertLessEqual(worst,P['contact_query_reserve'])
        intervals=linear*256+(variants*scales+variants)*sum(P['edge_grids'])
        self.assertLessEqual(intervals,P['contact_interval_reserve'])
        self.assertLessEqual(P['search_coupled_cap']+P['repair_coupled_cap']+80+40,P['unit_coupled_cap'])
        self.assertEqual(3*P['unit_coupled_cap']+100,P['coupled_cap'])
        self.assertLessEqual(P['coupled_cap'],22586)
        self.assertLessEqual(3*P['unit_integration_cap']+400000,P['integration_cap'])
        self.assertLessEqual(P['search_integration_cap']+P['repair_integration_cap']+2*P['validation_integration_cap']+P['assembly_integration_cap'],P['unit_integration_cap'])
        self.assertLessEqual(3*P['unit_wall_s']+600+600,P['wall_s'])
        self.assertLessEqual(P['wall_s']+1733.609,21540)
        self.assertLessEqual(3*27500+3*27000,165000)
        self.assertLess(P['output_bytes_cap']+585993194+512*1024**2,20*1024**3)

    def test_budget_never_transfers_or_overcommits(self):
        with tempfile.TemporaryDirectory() as td:
            b=Budget(Path(td),time.time()+120);b.begin_unit();s=SliceBudget(b,1,256,time.time()+120)
            s.consume(256)
            with self.assertRaises(BudgetExhausted):s.consume(1)
            with self.assertRaises(BudgetExhausted):SliceBudget(b,5,1,time.time()+120).consume(2)
            self.assertEqual(b.calls,1);b.observe({'coupled':1,'physics_steps':0})
            self.assertEqual(b.save()['actual_coupled'],1)

    def test_frozen_prefix_selection_and_no_training(self):
        from .inputs import identity
        ident=identity();s=ident['selection'];stats=read_json(ROOT/'ff_prior_world_steer/evidence/candidate_statistics.json')
        for u in s['units']:
            b=min((r for r in stats if r['unit']==u['id']),key=lambda r:(r['best_Q'],r['candidate'],r['best_node']))
            self.assertEqual((u['source_candidate'],u['source_node']),(b['candidate'],b['best_node']))
            self.assertEqual(file_digest(ROOT/u['prefix_file']),u['prefix_hash'])
            self.assertGreater(u['remaining_time'],0.)
        self.assertEqual(s['entries'][0]['task']['condition']['duration'],27.)
        self.assertEqual(s['entries'][0]['task']['task_hash'],'e66384242aaacf174ab1d7e642f0ba84b55ea580ab1b62951988409c75115496')
        self.assertEqual(s['new_neural_samples'],0);self.assertFalse(P['advantage_comparison'])

    def test_unknown_work_and_replay_gate(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td);atomic_json(p/'ledger.json',{'inflight':None})
            with self.assertRaisesRegex(ValueError,'NOT_REPLAYABLE'):assert_resume_safe(p)
        from ff_prior_static_goal_seed.physical_audit import replay_agrees
        self.assertTrue(replay_agrees({'integration':0.},1e-10))
        for v in ({'qpos':0.},{'integration':float('nan')},{'integration':1e-7}):self.assertFalse(replay_agrees(v,1e-10))

    def test_no_private_answer_reads_or_native_state_injection(self):
        for name in ('contact.py','experiment.py','inputs.py'):
            text=(PACKAGE/name).read_text(encoding='utf-8')
            for token in ('07_bow','capture_trust_001','optimizer.step','teachers/'):
                self.assertNotIn(token,text)
        text=(PACKAGE/'experiment.py').read_text(encoding='utf-8')
        self.assertIn('audit_execution(task,profile,robot,initial,',text)
        self.assertNotIn('mj_setState',text)

    def test_full323_fit_fixed_endpoints(self):
        q0=np.zeros(17);q1=np.linspace(-.01,.01,17)
        ns=[Node(q0,np.zeros(1),np.zeros(1),np.zeros(1),0.,-1,0.,np.zeros(1),-1),
            Node(q1,np.zeros(1),np.zeros(1),np.zeros(1),27.,0,27.,np.zeros(1),0)]
        robot=SimpleNamespace(lower=-np.ones(17),upper=np.ones(17))
        z,r,t,desired,fitted=fit323(ns,robot,27.)
        self.assertTrue(np.array_equal(z[-1],q1));self.assertTrue(np.array_equal(fitted[0],q0))
        self.assertLess(r['maximum_joint_fit_error_rad'],1e-5)


if __name__=='__main__':unittest.main()
