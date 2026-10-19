"""Software-only regression tests; none certify a physical trajectory."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from ff_prior.diffusion import schedule
from ff_prior.storage import read_json, file_digest
from ff_prior_pilot_train.storage import atomic_json, Ledger
from ff_prior_latent.learning import Codec
from ff_prior_delivery.gates import engineering_gate
from .protocol import OLD, ALLOCATIONS, PROTOCOL
from .sampling import ddim_update, LatentProjector, sample
from .inputs import identity_for
from .job import completed_no_work


class SoftwareTests(unittest.TestCase):
    def test_ddim_all_noise_scales(self):
        alpha = schedule().double()
        times = np.linspace(999,0,16).round().astype(int)
        rng = torch.Generator().manual_seed(82)
        for i,t in enumerate(times):
            noisy = torch.randn((3,102),generator=rng,dtype=torch.float64)
            clean = 20*torch.randn((3,102),generator=rng,dtype=torch.float64)
            next_a = alpha[times[i+1]] if i<15 else torch.tensor(1.,dtype=torch.float64)
            value, residual = ddim_update(noisy,clean,alpha[t],next_a)
            self.assertLess(residual,1e-12)
            self.assertTrue(torch.isfinite(value).all())
            if i==15:
                self.assertTrue(torch.equal(value,clean))

    def test_ddim_reject_nonfinite(self):
        with self.assertRaises(ValueError):
            ddim_update(torch.ones(2,dtype=torch.float64),torch.full((2,),float('nan')),
                        torch.tensor(.2,dtype=torch.float64),torch.tensor(.4,dtype=torch.float64))

    def test_codec_feedback_and_projected_directions(self):
        codec = Codec(read_json(OLD/'learning/goal_loss/codec.json'))
        rng = np.random.default_rng(93)
        base = rng.normal(size=(19,17))*.1
        y = rng.normal(size=codec.dim)*.1
        z = codec.decode(y,base)
        np.testing.assert_allclose(codec.decode(codec.encode(z,base),base),z,atol=1e-12,rtol=0)
        center = rng.normal(size=codec.dim)
        x = codec.normalized(codec.encode(z,base),center)
        np.testing.assert_allclose(codec.decode(codec.coefficients(x,center),base),z,atol=1e-12,rtol=0)
        b = codec.basis
        directions = rng.normal(size=(24,19,17))
        projected = np.einsum('tk,sk,nsj->ntj',b,b,directions)
        np.testing.assert_allclose(np.einsum('tk,sk,nsj->ntj',b,b,projected),projected,atol=1e-12,rtol=0)

    def test_budget_caps_do_not_borrow_or_reset(self):
        self.assertEqual(sum(ALLOCATIONS.values()),110)
        guide = LatentProjector.__new__(LatentProjector)
        guide.used = guide.segment_end = 27
        with self.assertRaisesRegex(RuntimeError,'SEGMENT_QUERY'):
            guide.evaluate(np.zeros((19,17)))
        guide.used = guide.segment_end = 110
        with self.assertRaisesRegex(RuntimeError,'SEGMENT_QUERY'):
            guide.evaluate(np.zeros((19,17)))

    def test_sixteen_step_feedback_software_fixture(self):
        # Analytic v=0 fixture exercises sampler plumbing, not robotics evidence.
        _, (_,entries) = identity_for()
        task = entries[0]['task']
        base = np.tile(task['condition']['initial_features'],(19,1))
        codec = Codec(read_json(OLD/'learning/goal_loss/codec.json'))
        class ZeroNetwork:
            calls = 0
            def __call__(self,x,t,c):
                self.calls += 1
                return torch.zeros_like(x)
        network = ZeroNetwork()
        seen = []
        def guide(raw,i):
            seen.append(i)
            y = codec.encode(raw,base)
            y[0] += .001
            z = codec.decode(y,base)
            return z, {'raw_Z':raw,'output_Z':z,'iteration':i}
        z,trace = sample((network,codec),base,task,82,guide)
        self.assertEqual(network.calls,16)
        self.assertEqual(seen,[12,13,14,15])
        self.assertEqual(len(trace['steps']),16)
        self.assertTrue(np.isfinite(z).all())
        np.testing.assert_allclose(z,trace['guidance'][-1]['output_Z'],atol=1e-12,rtol=0)

    def test_frozen_inputs_are_TRAIN_and_balanced(self):
        identity, (_,entries) = identity_for()
        self.assertEqual(len(entries),8)
        self.assertEqual(sum(bool(e['scene']['objects']) for e in entries),4)
        self.assertEqual({e['split'] for e in entries},{'TRAIN'})
        self.assertEqual(file_digest(OLD/'learning/goal_loss/model.pt'),PROTOCOL['model_sha256'])
        self.assertFalse(any('/TEST/' in p for p in identity['source_files']))

    def test_physical_failures_remain_in_frozen_denominator(self):
        m = dict(completed=80,eligible=63,selected_outcomes=16,execution_verified=13,candidate_errors=0)
        self.assertEqual(engineering_gate(m,80,16,True,[])['status'],'BLOCKED')
        m['eligible'] = 64
        self.assertEqual(engineering_gate(m,80,16,True,[])['status'],'PASS')
        m['execution_verified'] = 12
        self.assertEqual(engineering_gate(m,80,16,True,[])['status'],'BLOCKED')

    def test_unknown_interrupted_work_is_consumed(self):
        with tempfile.TemporaryDirectory() as d:
            ledger = Ledger(d,{'test':1})
            self.assertTrue(ledger.begin('once','candidate',10,1))
            recovered = Ledger(d,{'test':1})
            self.assertFalse(recovered.begin('once','candidate',10,1))
            self.assertEqual(recovered.data['completed']['once']['status'],'INTERRUPTED')
            self.assertEqual(recovered.data['unresolved_call_accounting'],['once'])

    def test_complete_resume_no_writes(self):
        # Synthetic I/O fixture only. It is not experimental evidence.
        with tempfile.TemporaryDirectory() as d:
            out = Path(d)
            identity = {'source':'fixture'}
            atomic_json(out/'run.json',{'identity':identity})
            atomic_json(out/'summary.json',{'status':'FINISHED'})
            atomic_json(out/'status.json',{'status':'FINISHED','pid':0})
            before = {p.name:file_digest(p) for p in out.iterdir()}
            with patch('ff_prior_delivery_guidance.inputs.identity_for',return_value=(identity,None)):
                result = completed_no_work(out)
            self.assertEqual(result['resume'],'NO_WORK_ALREADY_COMPLETE')
            self.assertEqual(before,{p.name:file_digest(p) for p in out.iterdir()})


if __name__=='__main__':
    unittest.main()
