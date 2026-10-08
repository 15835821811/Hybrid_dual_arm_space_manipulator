"""Small tensor checks of the declared architecture and diffusion equations."""
import importlib.util
import unittest

HAS_TORCH = importlib.util.find_spec("torch") is not None
if HAS_TORCH:
    import torch
    from torch import nn
    from v6_4.diffusion_model import ConditionalDDPM, DiffusionConfig, build_noise_schedule, noise_schedule_identity


@unittest.skipUnless(HAS_TORCH, "isolated learning dependency not yet installed")
class DiffusionModelTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(64)
        self.model = ConditionalDDPM(DiffusionConfig(condition_dim=7))

    def test_fixed_architecture_and_free_control_interface(self):
        self.assertEqual(len(self.model.denoiser.transformer.layers), 4)
        self.assertEqual(self.model.denoiser.transformer.layers[0].self_attn.num_heads, 4)
        first, second = self.model.denoiser.transformer.layers[:2]
        self.assertFalse(torch.equal(first.linear1.weight, second.linear1.weight))
        result = self.model(torch.zeros(2, 30, 17), torch.tensor([0, 99]), torch.zeros(2, 7))
        self.assertEqual(result.shape, (2, 30, 17))
        with self.assertRaises(ValueError):
            DiffusionConfig(condition_dim=7, layers=6)
        with self.assertRaises(ValueError):
            self.model(torch.zeros(2, 32, 17), torch.tensor([0, 99]), torch.zeros(2, 7))

    def test_cosine100_near_zero_terminal_and_exact_schedule_hash(self):
        config = self.model.config
        self.assertEqual(config.noise_schedule, "cosine")
        betas, alphas = build_noise_schedule(config)
        self.assertEqual(len(betas), 100)
        self.assertTrue(torch.all((betas > 0) & (betas < 1)))
        self.assertTrue(torch.all(alphas[:-1] > alphas[1:]))
        self.assertGreater(float(alphas[-1]), 0.)
        self.assertLess(float(alphas[-1]), 3e-7)
        self.assertAlmostEqual(float(alphas[-1]), 2.4285723e-7, delta=1e-13)
        identity = self.model.schedule_identity()
        self.assertEqual(identity, noise_schedule_identity(config))
        changed = alphas.clone()
        changed[10] *= .999
        self.assertNotEqual(identity["schedule_sha256"], noise_schedule_identity(config, betas, changed)["schedule_sha256"])

    def test_old_missing_schedule_is_explicit_linear_only(self):
        old = self.model.config.to_dict()
        for key in ("noise_schedule", "cosine_s", "beta_cap", "parameterization"):
            old.pop(key)
        with self.assertRaisesRegex(ValueError, "explicitly identify"):
            DiffusionConfig.from_dict(old)
        legacy = DiffusionConfig.from_dict(old, allow_legacy_missing_schedule=True, allow_legacy_missing_parameterization=True)
        self.assertEqual(legacy.noise_schedule, "linear_legacy_unscaled")
        self.assertEqual(legacy.parameterization, "direct_epsilon")
        betas, alphas = build_noise_schedule(legacy)
        self.assertAlmostEqual(float(alphas[-1]), .3635632481, delta=3e-8)
        self.assertNotEqual(noise_schedule_identity(legacy)["schedule_sha256"], self.model.schedule_identity()["schedule_sha256"])

    def test_forward_noise_equation_and_epsilon_mse(self):
        clean, noise = torch.full((2, 30, 17), .4), torch.full((2, 30, 17), -.8)
        times = torch.tensor([0, 99])
        alpha = self.model.alpha_bars[times].reshape(2, 1, 1)
        noisy, epsilon = self.model.q_sample(clean, times, noise)
        torch.testing.assert_close(noisy, alpha.sqrt() * clean + (1-alpha).sqrt() * noise)
        self.assertTrue(torch.equal(epsilon, noise))
        class PerfectEpsilon(nn.Module):
            def forward(self, x, t, c):
                return noise
        direct = ConditionalDDPM(DiffusionConfig(condition_dim=7, parameterization="direct_epsilon"))
        direct.denoiser = PerfectEpsilon()
        loss = direct.epsilon_loss(clean, torch.zeros(2, 7), timesteps=times, noise=noise)
        self.assertEqual(float(loss), 0.)

    def test_real_optimizer_step_updates_weights(self):
        parameter = self.model.denoiser.output.weight
        before = parameter.detach().clone()
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-3, weight_decay=0.)
        loss = self.model.epsilon_loss(torch.randn(2, 30, 17), torch.randn(2, 7))
        self.assertTrue(torch.isfinite(loss))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        self.assertFalse(torch.equal(parameter.detach(), before))

    def test_ddim20_reproducible_condition_dependent_and_unclipped(self):
        condition = torch.zeros(1, 7)
        self.model.train()
        a = self.model.sample_ddim(condition, generator=torch.Generator().manual_seed(8))
        b = self.model.sample_ddim(condition, generator=torch.Generator().manual_seed(8))
        c = self.model.sample_ddim(torch.ones(1, 7), generator=torch.Generator().manual_seed(8))
        self.assertTrue(self.model.training)
        self.assertTrue(torch.isfinite(a).all())
        torch.testing.assert_close(a, b, rtol=0, atol=0)
        self.assertFalse(torch.equal(a, c))
        class ZeroEpsilon(nn.Module):
            def forward(self, x, t, c):
                return torch.zeros_like(x)
        direct = ConditionalDDPM(DiffusionConfig(condition_dim=7, parameterization="direct_epsilon"))
        direct.denoiser = ZeroEpsilon()
        raw = direct.sample_ddim(condition, initial_noise=torch.full((1, 30, 17), 3.))
        expected = torch.full_like(raw, 3.) / direct.alpha_bars[-1].sqrt()
        torch.testing.assert_close(raw, expected)
        self.assertGreater(float(raw.min()), 3.)

    def test_residual_oracle_preserves_epsilon_target_and_reconstructs_x0(self):
        clean, epsilon = torch.randn(2,30,17), torch.randn(2,30,17)
        times = torch.tensor([50,99])
        signal, noise = self.model._coefficients(times)
        true_v = signal*epsilon-noise*clean
        class OracleResidual(nn.Module):
            def forward(self,x,t,c):return true_v
        self.model.denoiser=OracleResidual()
        noisy,_=self.model.q_sample(clean,times,noise=epsilon)
        predicted,x0=self.model.predict_epsilon_and_x0(noisy,times,torch.zeros(2,7))
        torch.testing.assert_close(predicted,epsilon,rtol=1e-6,atol=5e-7)
        torch.testing.assert_close(x0,clean,rtol=1e-6,atol=5e-7)
        self.assertLess(float(self.model.epsilon_loss(clean,torch.zeros(2,7),times,epsilon)),1e-12)

    def test_minimum_alpha_residual_gain_and_gradients_are_finite(self):
        class ScalarResidual(nn.Module):
            def __init__(self):
                super().__init__();self.value=nn.Parameter(torch.tensor(.4))
            def forward(self,x,t,c):return self.value.expand_as(x)
        residual=ScalarResidual();self.model.denoiser=residual
        times=torch.tensor([99]);condition=torch.zeros(1,7)
        clean=torch.ones(1,30,17);noise=torch.zeros_like(clean)
        loss=self.model.epsilon_loss(clean,condition,times,noise)
        loss.backward()
        alpha=self.model.alpha_bars[-1]
        expected=2*alpha*((1-alpha).sqrt()+residual.value.detach())
        torch.testing.assert_close(residual.value.grad,expected,rtol=1e-5,atol=1e-10)
        self.assertGreater(float(residual.value.grad),0.)
        self.assertLess(float(residual.value.grad),1e-6)  # Declared weak terminal supervision.
        extreme=torch.full((1,30,17),1e4)
        _,x0a=self.model.predict_epsilon_and_x0(extreme,times,condition)
        with torch.no_grad():residual.value.add_(1.)
        _,x0b=self.model.predict_epsilon_and_x0(extreme,times,condition)
        self.assertTrue(torch.isfinite(x0a).all() and torch.isfinite(x0b).all())
        torch.testing.assert_close(x0b-x0a,torch.full_like(x0a,-float((1-alpha).sqrt())),rtol=1e-6,atol=1e-6)
        self.assertLessEqual(float((x0b-x0a).abs().max()),1.000001)

    def test_raw_residual_sampling_has_no_control_clipping(self):
        class LargeResidual(nn.Module):
            def forward(self,x,t,c):return torch.full_like(x,100.)
        self.model.denoiser=LargeResidual()
        result=self.model.sample_ddim(torch.zeros(1,7),initial_noise=torch.zeros(1,30,17))
        self.assertTrue(torch.isfinite(result).all())
        self.assertGreater(float(result.abs().min()),10.)

    def test_v_oracle_and_per_example_loss_identities(self):
        model=ConditionalDDPM(DiffusionConfig(condition_dim=7,objective="v_mse"))
        clean,epsilon=torch.randn(3,30,17),torch.randn(3,30,17)
        times=torch.tensor([0,50,99]);signal,noise=model._coefficients(times)
        true_v=signal*epsilon-noise*clean
        error=torch.tensor([.03,.2,1.]).reshape(3,1,1).expand_as(clean)
        class FixedV(nn.Module):
            def forward(self,x,t,c):return true_v+error
        model.denoiser=FixedV();condition=torch.zeros(3,7)
        xt,_=model.q_sample(clean,times,epsilon)
        predicted,x0=model.predict_epsilon_and_x0(xt,times,condition)
        lv=error.square().mean(dim=(1,2))
        le=(predicted-epsilon).square().mean(dim=(1,2))
        lx=(x0-clean).square().mean(dim=(1,2))
        torch.testing.assert_close(le,signal.flatten().square()*lv,rtol=2e-4,atol=1e-9)
        torch.testing.assert_close(lx,noise.flatten().square()*lv,rtol=2e-4,atol=1e-9)
        torch.testing.assert_close(model.training_loss(clean,condition,times,epsilon),lv.mean())
        model.denoiser.forward=lambda x,t,c:true_v
        self.assertEqual(float(model.v_loss(clean,condition,times,epsilon)),0.)
        _,x0=model.predict_epsilon_and_x0(xt,times,condition)
        torch.testing.assert_close(x0,clean,rtol=1e-6,atol=5e-7)

    def test_v_terminal_gradient_and_sampling_math_do_not_change(self):
        model=ConditionalDDPM(DiffusionConfig(condition_dim=7,objective="v_mse"))
        model.load_state_dict(self.model.state_dict())
        noise=torch.randn(1,30,17)
        a=self.model.sample_ddim(torch.zeros(1,7),initial_noise=noise)
        b=model.sample_ddim(torch.zeros(1,7),initial_noise=noise)
        torch.testing.assert_close(a,b,rtol=0,atol=0)
        class ScalarV(nn.Module):
            def __init__(self):super().__init__();self.value=nn.Parameter(torch.tensor(.4))
            def forward(self,x,t,c):return self.value.expand_as(x)
        scalar=ScalarV();model.denoiser=scalar
        loss=model.training_loss(torch.ones(1,30,17),torch.zeros(1,7),torch.tensor([99]),torch.zeros(1,30,17))
        loss.backward()
        expected=2*(scalar.value.detach()+(1-model.alpha_bars[-1]).sqrt())
        torch.testing.assert_close(scalar.value.grad,expected)
        self.assertGreater(float(scalar.value.grad),2.)
        self.assertNotEqual(model.parameterization_identity()["parameterization_sha256"],self.model.parameterization_identity()["parameterization_sha256"])
        with self.assertRaisesRegex(ValueError,"v_mse requires"):
            DiffusionConfig(condition_dim=7,objective="v_mse",parameterization="direct_epsilon")

    def test_missing_objective_requires_explicit_historical_epsilon_mapping(self):
        config=self.model.config.to_dict();config.pop("objective")
        with self.assertRaisesRegex(ValueError,"training objective"):
            DiffusionConfig.from_dict(config)
        old=DiffusionConfig.from_dict(config,allow_legacy_missing_objective=True)
        self.assertEqual(old.objective,"epsilon_mse")

    def test_clean_x0_oracle_and_exact_per_example_v_objective(self):
        from v6_4.diffusion_model import objective_identity
        model=ConditionalDDPM(DiffusionConfig(condition_dim=7,objective="v_mse",parameterization="clean_x0"))
        clean,eps=torch.randn(3,30,17),torch.randn(3,30,17)
        times=torch.tensor([0,50,99]);signal,noise=model._coefficients(times)
        delta=torch.tensor([.0003,.02,.2]).reshape(3,1,1).expand_as(clean)
        class FixedClean(nn.Module):
            def forward(self,x,t,c):return clean+delta
        model.denoiser=FixedClean();condition=torch.zeros(3,7)
        xt,_=model.q_sample(clean,times,eps)
        pred_eps,x0=model.predict_epsilon_and_x0(xt,times,condition)
        torch.testing.assert_close(x0,clean+delta,rtol=0,atol=0)
        pred_v=(signal*xt-x0)/noise;true_v=signal*eps-noise*clean
        lv=(pred_v-true_v).square().mean(dim=(1,2))
        lx=(x0-clean).square().mean(dim=(1,2))
        le=(pred_eps-eps).square().mean(dim=(1,2))
        torch.testing.assert_close(lx,noise.flatten().square()*lv,rtol=2e-3,atol=1e-10)
        torch.testing.assert_close(le,signal.flatten().square()*lv,rtol=2e-3,atol=1e-10)
        torch.testing.assert_close(model.training_loss(clean,condition,times,eps),lv.mean(),rtol=1e-4,atol=1e-8)
        model.denoiser.forward=lambda x,t,c:clean
        self.assertEqual(float(model.training_loss(clean,condition,times,eps)),0.)
        pred_eps,x0=model.predict_epsilon_and_x0(xt,times,condition)
        torch.testing.assert_close(pred_eps,eps,rtol=1e-4,atol=1e-5)
        torch.testing.assert_close(x0,clean,rtol=0,atol=0)
        residual=DiffusionConfig(condition_dim=7,objective="v_mse")
        self.assertEqual(objective_identity(model.config),objective_identity(residual))
        self.assertNotEqual(model.parameterization_identity(),__import__('v6_4.diffusion_model',fromlist=['parameterization_identity']).parameterization_identity(residual))

    def test_clean_sampling_is_stable_unclipped_and_retains_gradient(self):
        model=ConditionalDDPM(DiffusionConfig(condition_dim=7,objective="v_mse",parameterization="clean_x0"))
        class ScalarClean(nn.Module):
            def __init__(self):super().__init__();self.value=nn.Parameter(torch.tensor(100.))
            def forward(self,x,t,c):return self.value.expand_as(x)
        head=ScalarClean();model.denoiser=head
        times=torch.tensor([0,99]);clean=torch.zeros(2,30,17);eps=torch.ones_like(clean)
        loss=model.training_loss(clean,torch.zeros(2,7),times,eps);loss.backward()
        expected=(2*head.value.detach()/(1-model.alpha_bars[times])).mean()
        torch.testing.assert_close(head.value.grad,expected,rtol=1e-5,atol=.01)
        self.assertTrue(torch.isfinite(head.value.grad))
        self.assertLess(float(1/(1-model.alpha_bars[0]).sqrt()),40.)
        sample=model.sample_ddim(torch.zeros(1,7),initial_noise=torch.full((1,30,17),1e4))
        torch.testing.assert_close(sample,torch.full_like(sample,100.),rtol=0,atol=0)
        self.assertTrue(torch.isfinite(sample).all())
        with self.assertRaisesRegex(ValueError,'fixes its objective'):
            DiffusionConfig(condition_dim=7,parameterization='clean_x0')


if __name__ == "__main__":
    unittest.main()
