"""Finite analytic/process and file-gate checks, with zero optimizer steps."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

import numpy as np
import torch

from v6_4.diffusion_model import DiffusionConfig, build_noise_schedule
from v6_4.residual_dataset import freeze_residual_dataset
from v6_4.residual_diffusion import (ResidualDiffusionConfig,ResidualDDPM,prepare_training,
                                     load_training_bundle,train_model,_validation)
from v6_4.tests.test_residual_dataset import synthetic_bundle


class ResidualDiffusionTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(64201)
        self.model=ResidualDDPM(ResidualDiffusionConfig(7))
        self.mask=torch.tensor([[True,False,False,False,False,False],[True,True,False,False,False,False]])
        self.dim_mask=self.mask.repeat_interleave(2,-1)
        self.x=torch.randn(2,12)*self.dim_mask
        self.eps=torch.randn(2,12)
        self.t=torch.tensor([0,99])
        self.condition=torch.randn(2,7)

    def test_exact_inherited_schedule_and_two_hidden_layers(self):
        betas,alphas=build_noise_schedule(DiffusionConfig(7,objective="v_mse"))
        torch.testing.assert_close(self.model.betas,betas,rtol=0,atol=0)
        torch.testing.assert_close(self.model.alpha_bars,alphas,rtol=0,atol=0)
        linear=[m for m in self.model.denoiser if isinstance(m,torch.nn.Linear)]
        self.assertEqual([(m.in_features,m.out_features) for m in linear],[(159,128),(128,128),(128,12)])
        with self.assertRaises(ValueError): ResidualDiffusionConfig(7,objective="epsilon_mse")

    def test_masked_forward_v_identity_and_per_example_loss(self):
        xt=self.model.q_sample(self.x,self.t,self.eps,self.mask)
        a=self.model.alpha_bars[self.t,None]
        true_v=(a.sqrt()*self.eps-(1-a).sqrt()*self.x)*self.dim_mask
        with patch.object(self.model,"forward",return_value=true_v):
            pred=self.model.prediction(xt,self.t,self.condition,self.mask)
            torch.testing.assert_close(pred["x0"],self.x,atol=3e-7,rtol=1e-6)
            torch.testing.assert_close(pred["epsilon"],self.eps*self.dim_mask,atol=3e-7,rtol=1e-6)
            self.assertEqual(float(self.model.loss(self.x,self.t,self.eps,self.condition,self.mask)),0.)
        with patch.object(self.model,"forward",return_value=true_v+self.dim_mask.to(torch.float32)):
            loss=self.model.loss(self.x,self.t,self.eps,self.condition,self.mask,reduction="none")
            torch.testing.assert_close(loss,torch.ones(2),atol=2e-7,rtol=0)
        torch.testing.assert_close(xt[~self.dim_mask],torch.zeros_like(xt[~self.dim_mask]),atol=0,rtol=0)

    def test_finite_backward_condition_consumption_no_weight_update(self):
        before={n:p.detach().clone() for n,p in self.model.named_parameters()}
        loss=self.model.loss(self.x,self.t,self.eps,self.condition,self.mask)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in self.model.parameters()))
        self.assertGreater(sum(float(p.grad.abs().sum()) for p in self.model.parameters()),0.)
        xt=self.model.q_sample(self.x,self.t,self.eps,self.mask)
        self.assertFalse(torch.equal(self.model(xt,self.t,self.condition,self.mask),self.model(xt,self.t,self.condition+1,self.mask)))
        for n,p in self.model.named_parameters(): torch.testing.assert_close(p,before[n],rtol=0,atol=0)

    def test_ddim_has_twenty_steps_no_clip_and_masks_every_step(self):
        calls=[]
        def prediction(x,t,c,m):
            calls.append((x.clone(),t.clone()))
            return {"x0":torch.full_like(x,123.),"epsilon":torch.zeros_like(x)}
        with patch.object(self.model,"prediction",side_effect=prediction):
            out=self.model.sample_ddim(self.condition,self.mask,initial_noise=self.eps)
        self.assertEqual([int(t[0]) for x,t in calls],torch.linspace(99,0,20).round().to(torch.long).tolist())
        for x,t in calls: self.assertTrue(torch.all(x[~self.dim_mask]==0))
        self.assertTrue(torch.all(out[self.dim_mask]==123.)); self.assertTrue(torch.all(out[~self.dim_mask]==0))

    def test_invalid_inputs_do_not_silently_repair_clean_latent(self):
        bad=self.x.clone(); bad[0,2]=1e-20
        with self.assertRaisesRegex(ValueError,"exactly zero"): self.model.q_sample(bad,self.t,self.eps,self.mask)
        with self.assertRaises(ValueError): self.model.sample_ddim(self.condition,torch.zeros_like(self.mask),initial_noise=self.eps)
        with self.assertRaises(ValueError): self.model(self.x,torch.tensor([0,100]),self.condition,self.mask)
        bad=self.eps.clone(); bad[0,2]=float("nan")
        with self.assertRaises(ValueError): self.model.sample_ddim(self.condition,self.mask,initial_noise=bad)

    def test_validation_averages_draws_then_references_then_tasks(self):
        class FixedLoss:
            def loss(self,x,t,e,c,m,reduction): return x[:,0]
        ds=SimpleNamespace(samples=[{"task_id":"a"},{"task_id":"a"},{"task_id":"b"}])
        clean=torch.zeros(3,12); clean[:,0]=torch.tensor([1.,3.,9.])
        draws={"reference_indices":np.array([0,0,1,1,2,2]),"timesteps":np.zeros(6,dtype=np.int64),
               "epsilon":np.zeros((6,12),dtype=np.float32)}
        metric,details=_validation(FixedLoss(),clean,torch.zeros(3,7),torch.ones(3,6,dtype=torch.bool),draws,ds,"cpu")
        self.assertEqual(metric,5.5)
        self.assertEqual(details["per_task"],{"a":2.,"b":9.})

    def test_no_successful_val_no_training_and_no_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            parent=Path(tmp); teacher,suite,defs=synthetic_bundle(parent,failed_val=True)
            freeze_residual_dataset(teacher,suite,defs,parent/"data")
            with patch.object(torch.optim,"AdamW",side_effect=AssertionError("training forbidden")):
                result=prepare_training(parent/"data/manifest.json",parent/"training")
                self.assertEqual(result["status"],"TRAINING_NOT_RUN")
                self.assertEqual(train_model(parent/"training"),result)
            self.assertFalse((parent/"training/model").exists())
            self.assertEqual(list((parent/"training").glob("*.pt")),[])

    def test_prepare_freezes_train_only_scalers_validation_draws_and_source_guards(self):
        with tempfile.TemporaryDirectory() as tmp:
            parent=Path(tmp); teacher,suite,defs=synthetic_bundle(parent)
            freeze_residual_dataset(teacher,suite,defs,parent/"data")
            with patch.object(torch.optim,"AdamW",side_effect=AssertionError("prepare cannot train")):
                result=prepare_training(parent/"data/manifest.json",parent/"training")
            self.assertEqual(result["status"],"READY")
            b=load_training_bundle(parent/"training")
            self.assertEqual(b["condition_normalizer"].fit_task_ids,["train"])
            self.assertEqual(len(b["validation_draws"]["reference_indices"]),16)
            self.assertTrue(np.all(b["validation_draws"]["reference_indices"]==1))
            self.assertEqual(b["config"]["optimizer_updates"],4000)
            self.assertEqual(b["config"]["data_status"],"DATA_LIMITED")
            self.assertFalse((parent/"training/model").exists())
            with self.assertRaises(FileExistsError): prepare_training(parent/"data/manifest.json",parent/"training")
            p=parent/"training/condition_normalizer.json"; p.write_text("{}")
            with self.assertRaisesRegex(ValueError,"normalizer changed"): load_training_bundle(parent/"training")


if __name__=="__main__": unittest.main()
