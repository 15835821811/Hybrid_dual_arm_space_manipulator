"""Finite synthetic mathematical/identity fixtures; no physical data/run."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from v6_4 import empirical_score_diffusion as c
from v6_4 import prior_diffusion as p
from v6_4.dataset import load_teacher_dataset, sha256
from v6_4.trajectory_codec import CubicBSplineCodec
from v6_4.tests import test_prior_diffusion as prior_fixture
from v6_4.tests.test_learning_dataset import fixture_task


class EmpiricalScoreTests(unittest.TestCase):
    def model(self,means,groups=None,weight=None,bias=None,**kwargs):
        groups=np.arange(len(means)) if groups is None else np.asarray(groups)
        n=int(groups.max())+1
        return c.EmpiricalConditionalScore(means,groups,np.zeros((n,1027)) if weight is None else weight,
                                           np.zeros(n) if bias is None else bias,**kwargs)

    def fixture(self,root):
        f=prior_fixture.PriorDiffusionTests().fixture(root)
        source={"git_head":"fixture_C_head","tracked_dirty":False,
                "sources_sha256":{"v6_4/empirical_score_diffusion.py":sha256(c.REPO/'v6_4/empirical_score_diffusion.py')}}
        return f,source

    def test_group_prior_balances_groups_not_prototype_count(self):
        groups=np.repeat(np.arange(6),[1,1,2,2,3,4]);means=np.random.default_rng(2).normal(size=(13,510));model=self.model(means,groups)
        priors=np.exp(model.component_log_probabilities(np.zeros(1027)))
        np.testing.assert_allclose(np.bincount(groups,weights=priors),np.full(6,1/6),atol=1e-15)
        x=np.random.default_rng(3).normal(size=510);original=model.posterior(x,.43,np.zeros(1027))
        order=np.arange(12,-1,-1);other=self.model(means[order],groups[order]).posterior(x,.43,np.zeros(1027))
        np.testing.assert_allclose(original['mean'],other['mean'],atol=1e-14)
        self.assertAlmostEqual(original['log_density'],other['log_density'],places=12)

    def test_posterior_tanh_and_independent_density_score_difference(self):
        means=np.zeros((2,510));means[:,0]=[-.7,.7];bias=np.array([-.3,.4]);model=self.model(means,bias=bias)
        features=np.zeros(1027);x=np.linspace(-.1,.1,510);alpha=.64;v=1-alpha
        posterior=model.posterior(x,alpha,features)
        oracle=.7*np.tanh(np.sqrt(alpha)*.7*x[0]/v+.5*(bias[1]-bias[0]))
        self.assertAlmostEqual(posterior['mean'][0],oracle,places=14)
        def independent_log_density(z):
            pi=np.exp(bias-np.max(bias));pi/=pi.sum()
            logits=np.log(pi)-((z[None]-np.sqrt(alpha)*means)**2).sum(1)/(2*v)
            mx=logits.max();return mx+np.log(np.exp(logits-mx).sum())-.5*510*np.log(2*np.pi*v)
        for coordinate in [0,3,509]:
            plus=x.copy();minus=x.copy();plus[coordinate]+=1e-5;minus[coordinate]-=1e-5
            gradient=(independent_log_density(plus)-independent_log_density(minus))/(2e-5)
            self.assertAlmostEqual(gradient,posterior['score'][coordinate],places=7)
        np.testing.assert_allclose(posterior['score'],-posterior['epsilon']/np.sqrt(v),atol=1e-14)

    def test_single_gaussian_full20_trace_has_analytic_noise_invariant(self):
        mean=np.linspace(-.2,.2,510);model=self.model(np.stack([mean,mean]));initial=np.random.default_rng(4).normal(size=510)
        final,trace=model.sample_ddim(np.zeros(1027),initial_noise=initial)
        self.assertEqual(trace['state_before'].shape,(20,510));self.assertEqual(trace['next_alpha_bar'][-1],1.)
        self.assertTrue(np.all(np.diff(trace['indices'])<0));self.assertEqual(trace['indices'][0],99);self.assertEqual(trace['indices'][-1],0)
        z=(initial-np.sqrt(trace['alpha_bar'][0])*mean)/np.sqrt(1-trace['alpha_bar'][0])
        for i in range(20):
            np.testing.assert_allclose(trace['state_before'][i],np.sqrt(trace['alpha_bar'][i])*mean+np.sqrt(1-trace['alpha_bar'][i])*z,atol=3e-14)
            expected=np.sqrt(trace['next_alpha_bar'][i])*trace['posterior_mean'][i]+np.sqrt(1-trace['next_alpha_bar'][i])*trace['epsilon'][i]
            np.testing.assert_array_equal(trace['state_after'][i],expected)
        np.testing.assert_allclose(final,mean,atol=1e-15)

    def test_symmetric_full_chain_is_mixture_mean_never_argmax_snap(self):
        means=np.zeros((2,510));means[:,0]=[-.5,.5];model=self.model(means)
        result,trace=model.sample_ddim(np.zeros(1027),initial_noise=np.zeros(510))
        np.testing.assert_array_equal(result,np.zeros(510));np.testing.assert_allclose(trace['posterior_weights'],.5,atol=1e-15)
        np.testing.assert_array_equal(trace['posterior_mean'],np.zeros((20,510)))
        self.assertFalse(any(np.array_equal(result,mean) for mean in means))
        self.assertAlmostEqual(trace['entropy'][-1],np.log(2),places=14)

    def test_real_zero_initialized_adam_and_numeric_condition_response(self):
        features=np.zeros((2,1027));features[:,0]=[-1.,1.];groups=np.array([0,1]);steps=1
        w,b,fit=c.fit_linear_classifier(features,groups,2,steps=steps,seed=71)
        # Independent first Adam update from analytic zero-logit CE gradient.
        probabilities=np.full((2,2),.5)-np.eye(2)
        gradient=probabilities.T@features/2
        expected=-.01*gradient/(abs(gradient)+1e-8)
        np.testing.assert_allclose(w,expected,atol=2e-17);np.testing.assert_array_equal(b,np.zeros(2))
        self.assertEqual(w.dtype,np.float64);self.assertEqual(fit['optimizer_steps'],1)
        means=np.zeros((2,510));means[:,0]=[-.2,.2];model=self.model(means,groups,w,b)
        self.assertLess(model.posterior(np.zeros(510),.2,features[0])['mean'][0],0.)
        self.assertGreater(model.posterior(np.zeros(510),.2,features[1])['mean'][0],0.)
        one=c.fit_linear_classifier(features,groups,2,steps=8,seed=71)
        two=c.fit_linear_classifier(features,groups,2,steps=8,seed=71)
        np.testing.assert_array_equal(one[0],two[0]);np.testing.assert_array_equal(one[1],two[1])

    def test_conditional_affine_fit_restores_train_and_translates_unseen_features(self):
        group_features=np.zeros((6,1027));group_features[:,:5]=np.vstack([np.eye(5),np.zeros(5)])
        groups=np.repeat(np.arange(6),[2,3,2,2,2,2]);features=group_features[groups]
        group_targets=np.zeros((6,510));group_targets[:,:5]=2*group_features[:,:5]+.3
        offsets=np.zeros((13,510));offsets[:,7]=np.r_[-.1,.1,-.1,0.,.1,np.tile([-.1,.1],4)]
        means=group_targets[groups]+offsets
        a,f,t,o,stats=c.fit_conditional_means(features,means,groups,6)
        self.assertEqual(stats['centered_feature_rank'],5);self.assertEqual(stats['closed_form_fits'],1)
        model=self.model(o,groups,affine=a,feature_mean=f,target_mean=t)
        for i in range(13):np.testing.assert_allclose(model.component_means(features[i])[i],means[i],atol=2e-15)
        query=np.zeros(1027);query[:5]=.5
        translated=model.component_means(query)
        np.testing.assert_allclose(translated[:,:5],np.full((13,5),1.3),atol=2e-15)
        np.testing.assert_allclose(translated[:,7],offsets[:,7],atol=1e-15)

    def test_fixture_fit_checkpoint_raw_restore_and_fixed_boundary(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);f,source=self.fixture(root)
            with patch.object(p,'_verify_legacy_bootstrap',return_value=(f[4],f[5])),patch.object(c,'source_identity',return_value=source),patch.object(p,'_model_contract_and_assets',return_value=('a'*64,(f[4],))):
                report=c.fit_empirical_score(f[0],f[1],root/'model',optimizer_steps=2,expected_samples=2,expected_groups=2)
                self.assertEqual(report['noise_training_iterations'],0);self.assertEqual(report['fit']['optimizer_steps'],2)
                centers,_=p.load_centers(f[1],f[2]);sampler=c.load_empirical_sampler(root/'model/checkpoint.json')
                raw,residuals,metadata,traces=sampler.sample(f[2][0],centers['a'],K=1,seed=64)
                np.testing.assert_array_equal(raw,centers['a'].free+residuals)
                full=CubicBSplineCodec(f[2][0].initial_planner_q,f[2][0].initial_planner_dq).decode_free(raw[0])
                np.testing.assert_array_equal(full[:2],centers['a'].full[:2]);self.assertEqual(len(traces[0]['indices']),20)
                self.assertEqual(metadata['postprocessing'],[]);self.assertEqual(metadata['method'],c.METHOD)
                # Different names cannot affect numerical conditioning/inference.
                renamed=fixture_task('train','arbitrary_task_name','arbitrary_group')
                center=SimpleNamespace(task=renamed,free=centers['a'].free)
                np.testing.assert_array_equal(c.condition_features(sampler.normalizer,renamed,center),c.condition_features(sampler.normalizer,f[2][0],centers['a']))
                absolute,_=c.restore_residual(sampler.normalizer,np.full((30,17),1e6),centers['a'])
                self.assertGreater(float(abs(absolute).max()),1e3)

    def test_train_only_foreign_checkpoint_input_source_and_companion_rejection(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);f,source=self.fixture(root)
            with patch.object(p,'_verify_legacy_bootstrap',return_value=(f[4],f[5])),patch.object(c,'source_identity',return_value=source),patch.object(p,'_model_contract_and_assets',return_value=('a'*64,(f[4],))):
                data=load_teacher_dataset(f[0]);data.samples[0]['split']='val'
                with patch.object(c,'load_teacher_dataset',return_value=data):
                    with self.assertRaisesRegex(ValueError,'no VAL/TEST'):c.fit_empirical_score(f[0],f[1],root/'bad',optimizer_steps=1,expected_samples=2,expected_groups=2)
                c.fit_empirical_score(f[0],f[1],root/'model',optimizer_steps=1,expected_samples=2,expected_groups=2)
                with self.assertRaisesRegex(ValueError,'rejects B/legacy'):c.load_empirical_sampler(root/'foreign.pt')
                cp=json.loads((root/'model/checkpoint.json').read_text());bad=deepcopy(cp);bad['schema']=p.CHECKPOINT_SCHEMA
                (root/'foreign.json').write_text(json.dumps(bad))
                with self.assertRaisesRegex(ValueError,'foreign/reinterpreted'):c.load_empirical_sampler(root/'foreign.json')
                sampler=c.load_empirical_sampler(root/'model/checkpoint.json');centers,_=p.load_centers(f[1],f[2])
                changed=deepcopy(source);changed['git_head']='changed'
                with patch.object(c,'source_identity',return_value=changed):
                    with self.assertRaisesRegex(ValueError,'sampling source/HEAD'):sampler.sample(f[2][0],centers['a'])
                normalizer=root/'model/normalizer.json';normalizer.write_text(normalizer.read_text()+' ')
                with self.assertRaisesRegex(ValueError,'companion SHA'):c.load_empirical_sampler(root/'model/checkpoint.json')


if __name__=='__main__':unittest.main()
