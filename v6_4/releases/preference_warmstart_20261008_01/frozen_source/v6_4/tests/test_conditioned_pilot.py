"""Functional B.1 invariants shared by the two finite training runs."""
import unittest
import numpy as np
import torch
from v6_4.conditioned_controlpoint_denoiser import PilotDDPM
from v6_4.diffusion_model import ConditionalDenoiser
from v6_4.differentiable_spline_loss import DecodedReferenceLoss, paired_training_loss
from v6_4.trajectory_codec import CubicBSplineCodec


class PilotTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(73)
        self.c={'global':torch.randn(2,93),'token_features':torch.randn(2,64,40),
                'token_types':torch.ones(2,64,dtype=torch.long),
                'token_mask':torch.ones(2,64,dtype=torch.bool),
                'fixed_controls':torch.zeros(2,2,17),
                'fixed_controls_scaled':torch.zeros(2,2,17),'flat':torch.randn(2,2781)}
        self.x=torch.randn(2,30,17);self.t=torch.tensor([7,86])

    def test_m0_is_exact_existing_body_and_schedule_equal(self):
        m0=PilotDDPM('M0',2781);m1=PilotDDPM('M1',2781)
        existing=ConditionalDenoiser(m0.config)
        existing.load_state_dict(m0.denoiser.body.state_dict());existing.eval();m0.eval()
        torch.testing.assert_close(m0.denoiser(self.x,self.t,self.c),existing(self.x,self.t,self.c['flat']),rtol=0,atol=0)
        torch.testing.assert_close(m0.alpha_bars,m1.alpha_bars,rtol=0,atol=0)
        torch.testing.assert_close(m0.betas,m1.betas,rtol=0,atol=0)
        self.assertEqual(len(m1.denoiser.blocks),4)
        self.assertEqual(m1.denoiser.position.shape,(1,32,128))

    def test_typed_context_permutation_and_arbitrary_padding(self):
        m=PilotDDPM('M1',2781).eval();self.c['token_mask'][:,55:]=False
        a=m.denoiser(self.x,self.t,self.c)
        order=torch.randperm(64);c={k:v.clone() for k,v in self.c.items()}
        for k in ('token_features','token_types','token_mask'):c[k]=c[k][:,order]
        torch.testing.assert_close(a,m.denoiser(self.x,self.t,c),rtol=3e-5,atol=3e-6)
        c={k:v.clone() for k,v in self.c.items()}
        c['token_features'][:,55:]=float('nan');c['token_types'][:,55:]=9999
        torch.testing.assert_close(a,m.denoiser(self.x,self.t,c),rtol=0,atol=0)
        c['token_mask'][:]=False
        with self.assertRaises(ValueError):m.denoiser(self.x,self.t,c)

    def test_condition_changes_reach_every_layer_and_update(self):
        m=PilotDDPM('M1',2781);out=m.denoiser(self.x,self.t,self.c)
        changed={k:v.clone() for k,v in self.c.items()};changed['token_features'][:,0,:3]+=.3
        self.assertGreater(float((out-m.denoiser(self.x,self.t,changed)).abs().max().detach()),1e-7)
        decoder=DecodedReferenceLoss(np.zeros(17),np.ones(17),np.ones(17)*2,np.ones(17))
        loss,parts=paired_training_loss(m,decoder,self.x,self.c,self.t,torch.randn_like(self.x))
        loss.backward()
        for block in m.denoiser.blocks:
            self.assertGreater(float(block.modulation[1].weight.grad.abs().sum()),0)
            self.assertGreater(float(block.cross_attention.in_proj_weight.grad.abs().sum()),0)
        self.assertGreater(float(m.denoiser.token_input.weight.grad.abs().sum()),0)
        self.assertGreater(float(m.denoiser.global_input[0].weight.grad.abs().sum()),0)
        self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in m.parameters()))
        old=m.denoiser.output.weight.detach().clone();torch.optim.AdamW(m.parameters(),lr=1e-4).step()
        self.assertGreater(float((old-m.denoiser.output.weight.detach()).abs().max()),0)
        self.assertEqual(set(parts),{'noise','decoded_q','decoded_dq'})

    def test_torch_decoder_matches_original_basis_and_aux_grad(self):
        mean=np.linspace(-.2,.2,17);scale=np.linspace(.1,.5,17)
        d=DecodedReferenceLoss(mean,scale,np.ones(17)*2,np.ones(17))
        clean=torch.randn(2,30,17,requires_grad=True);fixed=torch.zeros(2,2,17)
        q,dq=d.decode(clean,fixed)
        full=np.concatenate((fixed.numpy(),clean.detach().numpy()*scale+mean),axis=1)
        codec=CubicBSplineCodec(np.zeros(17),np.zeros(17))
        np.testing.assert_allclose(q.detach().numpy(),np.einsum('tc,bcd->btd',codec.basis(np.linspace(0,27,136)),full),atol=1e-7,rtol=2e-6)
        np.testing.assert_allclose(dq.detach().numpy(),np.einsum('tc,bcd->btd',codec.basis(np.linspace(0,27,136),1),full),atol=1e-6,rtol=3e-6)
        lq,lv=d(clean,torch.zeros_like(clean),fixed,torch.tensor([.2,.01]));(lq+lv).backward()
        self.assertGreater(float(clean.grad[0].abs().sum()),0);self.assertEqual(float(clean.grad[1].abs().sum()),0)

    def test_common_ddim20_and_exact_initial_elimination(self):
        for name in ('M0','M1'):
            m=PilotDDPM(name,2781).eval()
            with torch.no_grad():
                a=m.sample_ddim(self.c,initial_noise=self.x);b=m.sample_ddim(self.c,initial_noise=self.x)
            torch.testing.assert_close(a,b,rtol=0,atol=0);self.assertTrue(torch.isfinite(a).all())
            codec=CubicBSplineCodec(np.zeros(17),np.zeros(17));full=codec.decode_free(a[0].numpy())
            self.assertTrue(np.array_equal(full[:2],np.zeros((2,17))))
            with self.assertRaises(ValueError):m.sample_ddim(self.c,steps=4)


if __name__=='__main__':unittest.main()
