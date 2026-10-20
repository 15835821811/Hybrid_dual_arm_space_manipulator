"""Software checks only: no fabricated physical PASS or training result."""
import copy,tempfile,time,unittest
from pathlib import Path
import numpy as np
import torch
from ff_prior.storage import read_json,digest
from ff_prior.diffusion import weight_hash
from ff_prior_paired_objectives.objectives import get_vector,set_vector,predict,vector_gradient
from ff_prior_origin_learn_bounded.distribution import CoverageSampler
from ff_prior_delivery_train.learning import load as warm_load
from .protocol import ROOT,WARM,CONTROL,PROTOCOL as P
from . import endpoint
from .inputs import identity
from .data import load
from .optimization import admissible,propose
from .learning import base_loss
from .comparison import unguided_sample
from .experiment import Budget

class Software(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.model,cls.codec=warm_load(WARM/'learning/goal_loss')
        cls.arrays,cls.rows,cls.G,cls.data=load(cls.codec)
        cls.selection=read_json(ROOT/'ff_prior_endpoint_precision/evidence/selection.json')
    def test_01_binding_and_selection(self):
        i=identity();s=i['selection'];self.assertEqual(len(s['cases']),24)
        a={e['task_hash'] for e in s['diagnostic']};b={e['task_hash'] for e in s['events']}
        self.assertEqual((len(a),len(b),len(a&b)),(8,32,0))
        self.assertEqual(sum(e['obstacle'] for e in s['diagnostic']),4)
        self.assertEqual(sum(e['obstacle'] for e in s['events']),16)
        for e in s['diagnostic']+s['events']:self.assertEqual(self.arrays[4][self.data['ids'][e['sample_id']]],'TRAIN')
        self.assertEqual(len(i['binding']['files']),1096)
    def test_02_loss_absolute_and_threshold_groups(self):
        ee=np.tile(np.eye(4),(2,1,1));goal=ee.copy();ee[0,0,3]=.01
        value,pos,rot=endpoint.loss_from_ee(ee,goal)
        self.assertAlmostEqual(value,4.05);self.assertAlmostEqual(pos[0],.01)
        self.assertGreater(value,0.)
        clean=torch.tensor([1.,2.],requires_grad=True)
        attached=endpoint.attach_gradient(clean,5.,np.array([3.,4.]));attached.backward()
        self.assertEqual(float(attached),5.);self.assertTrue(torch.equal(clean.grad,torch.tensor([3.,4.])))
    def test_03_reject_passed_group_crossing(self):
        b={'position_m':[.01,.004],'rotation_rad':[0.,0.]}
        self.assertTrue(admissible(b,dict(b,position_m=[.009,.004])))
        self.assertFalse(admissible(b,dict(b,position_m=[.009,.006])))
        self.assertFalse(admissible(b,b))
        self.assertFalse(admissible(b,dict(b,position_m=[float('nan'),0.])))
    def test_04_optimizer_transaction(self):
        model=torch.nn.Linear(3,2);opt=torch.optim.AdamW(model.parameters(),lr=2e-4)
        original=get_vector(model);state=copy.deepcopy(opt.state_dict());g=torch.arange(original.numel(),dtype=torch.float32)+1
        a,sa,_=propose(model,opt,g)
        set_vector(model,original);opt.load_state_dict(copy.deepcopy(state))
        b,sb,_=propose(model,opt,g*1.1)
        set_vector(model,original);opt.load_state_dict(copy.deepcopy(state))
        self.assertTrue(torch.equal(get_vector(model),original));self.assertEqual(opt.state_dict()['state'],{})
        set_vector(model,a);opt.load_state_dict(sa)
        self.assertTrue(all(float(s['step'])==1 for s in opt.state_dict()['state'].values()))
        self.assertTrue(torch.equal(get_vector(model),a));self.assertTrue(torch.isfinite(b).all())
    def test_05_all_6000_A_random_batches(self):
        torch.manual_seed(P['training_seed']);sampler=CoverageSampler(self.data['keys'],P['training_seed']+2)
        expected=read_json(CONTROL/'learning/goal_loss/training.json');proofs=[];segments=[]
        for step in range(6000):
            batch=sampler.take(64);t=torch.randint(0,1000,(64,));noise=torch.randn((64,102))
            proofs.append(digest({'update':step+1,'batch':batch,'t':t.numpy(),'noise':noise.numpy()}))
            if (step+1)%20==0:
                segments.append({'start_update':step-18,'end_update':step+1,'draws':1280,'batch_rng_digest':digest(proofs)});proofs=[]
        self.assertEqual(segments,expected['segments'])
        ids=[self.arrays[3][i] for i in self.data['tr']]
        self.assertEqual(dict(zip(ids,(sampler.row_counts+sampler.origin_row_counts).tolist())),expected['train_draw_counts'])
    def test_06_base_objective_exact_A(self):
        from ff_prior_origin_learn_bounded.learning import objective
        i=self.data['tr'][:3];t=torch.tensor([333,667,999]);noise=torch.randn((3,102),generator=torch.Generator().manual_seed(91))
        loss=base_loss(self.model,self.data,self.G,i,t,noise)
        a=self.data['alpha'][t,None];x=self.data['x'][i];c=self.data['c'][i]
        noisy=a.sqrt()*x+(1-a).sqrt()*noise;target=a.sqrt()*noise-(1-a).sqrt()*x
        old,_,_=objective(self.model(noisy,t,c),target,noisy,a,x,self.G[:3],.05)
        self.assertTrue(torch.equal(loss,old));self.assertTrue(torch.isfinite(vector_gradient(loss,self.model)).all())
    def test_07_unguided_exact_network_budget(self):
        i=self.data['va'][0];before=weight_hash(self.model)
        z,s=unguided_sample((self.model,self.codec),self.arrays[1][i],self.rows[i]['task'],20263111)
        self.assertEqual(len(s),16);self.assertTrue(np.isfinite(z).all());self.assertLessEqual(max(v['residual'] for v in s),1e-10)
        self.assertEqual(weight_hash(self.model),before)
    def test_08_independent_prediction_rng(self):
        i=self.data['tr'][0];before=torch.get_rng_state().clone()
        x=predict(self.model,self.data,i,667,5)
        self.assertTrue(torch.equal(before,torch.get_rng_state()));self.assertEqual(x.shape,(102,))
    def test_09_budget_reject_before_mutation(self):
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory);b=Budget(p,time.time()+100);b.stage('test',2)
            b.reserve(2);before=(p/'query_budget.json').read_bytes()
            with self.assertRaises(RuntimeError):b.reserve(1)
            self.assertEqual(b.total,2);self.assertEqual(before,(p/'query_budget.json').read_bytes())
    def test_10_resume_completed_read_only(self):
        from .job import completed_no_work
        from ff_prior_pilot_train.storage import atomic_json
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)
            atomic_json(p/'run.json',{'identity':identity(),'software_fixture_only':True})
            atomic_json(p/'summary.json',{'status':'FINISHED_WITH_BLOCKED_ITEMS','software_fixture_only':True})
            atomic_json(p/'status.json',{'status':'FINISHED_WITH_BLOCKED_ITEMS','software_fixture_only':True})
            before={f.name:f.read_bytes() for f in p.iterdir()}
            self.assertEqual(completed_no_work(p)['resume'],'NO_WORK_ALREADY_COMPLETE')
            self.assertEqual(before,{f.name:f.read_bytes() for f in p.iterdir()})
    def test_11_missing_endpoint_retains_denominator(self):
        from .statistics import aggregate,paired
        s=aggregate([{'eligible':False},{'eligible':True,'frozen_goal_position_error_m':[.001,.002],'frozen_goal_rotation_error_rad':[0.,0.]}])
        self.assertEqual((s['denominator'],s['missing_endpoint'],s['qualified']),(2,1,1))
        p=paired({'a':{'position_mm':1.,'rotation_deg':0.}}, {})
        self.assertEqual((p['denominator'],p['complete_pairs']),(1,0))
if __name__=='__main__':unittest.main()
