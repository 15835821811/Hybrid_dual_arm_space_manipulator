"""Temporary synthetic fixtures exercise weights/provenance, never physics."""
import importlib.util
import json
from pathlib import Path
import tempfile
import shutil
import unittest
from unittest.mock import patch

import numpy as np

from v6_4.dataset import write_teacher_manifest
from v6_4.tests.test_learning_dataset import fixture_task, synthetic_teacher

HAS_TORCH = importlib.util.find_spec("torch") is not None
if HAS_TORCH:
    import torch
    from v6_4.train_diffusion import load_sampler, train_from_manifest, optimize_epsilon_from_manifest, train_v_from_manifest, train_clean_x0_from_manifest, _update_ema


@unittest.skipUnless(HAS_TORCH, "isolated learning dependency not yet installed")
class TrainDiffusionTests(unittest.TestCase):
    def make_train_parent(self, root):
        from v6_4.dataset import sha256
        import v6_4.train_diffusion as module
        manifest = root / "train_only.json"
        samples = [synthetic_teacher(root, fixture_task("train", "train0", "train0"), "train0", .05),
                   synthetic_teacher(root, fixture_task("train", "train1", "train1", .1), "train1", .08)]
        write_teacher_manifest(samples, manifest)
        output = root / "parent"
        report = train_from_manifest(manifest, output, epochs=1, seed=4, device="cpu", cpu_threads=2)
        archive = output / "source_snapshot"
        archive.mkdir()
        repo = Path(module.__file__).resolve().parents[1]
        for relative, expected in report["source_before"]["learning_sources_sha256"].items():
            destination = archive / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(repo / relative, destination)
            self.assertEqual(sha256(destination), expected)
        (archive / "source_archive_receipt.json").write_text(json.dumps({
            "source_identity": report["source_before"], "checkpoint_sha256": report["checkpoint_sha256"]}), encoding="utf-8")
        return manifest, output

    def make_manifest(self, root):
        samples = [synthetic_teacher(root, fixture_task("train", "train", "train"), "train", .05),
                   synthetic_teacher(root, fixture_task("val", "val", "val", .1), "val", .1),
                   synthetic_teacher(root, fixture_task("test", "test", "test", .2), "test", 100.)]
        path = root / "teachers.json"
        write_teacher_manifest(samples, path)
        return path

    def test_real_tiny_training_checkpoint_sampling_and_companion_hash(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            manifest = self.make_manifest(root)
            output = root / "training"
            report = train_from_manifest(manifest, output, epochs=2, batch_size=1, seed=9, device="cpu", cpu_threads=2)
            self.assertTrue(report["complete"])
            self.assertEqual(report["optimizer_steps"], 2)
            self.assertEqual(report["test_samples_used_for_optimization"], 0)
            self.assertTrue(report["source_unchanged"] and report["inputs_unchanged"])
            self.assertEqual(report["noise_schedule"]["noise_schedule"], "cosine")
            self.assertEqual(report["parameterization_identity"]["parameterization"], "epsilon_residual")
            self.assertEqual(report["parameterization_identity"]["loss_target"], "true_forward_epsilon")
            self.assertGreater(report["terminal_alpha_bar"], 0.)
            self.assertLess(report["terminal_alpha_bar"], 3e-7)
            normalizer = json.loads((output / "normalizer.json").read_text(encoding="utf-8"))
            np.testing.assert_allclose(normalizer["control_mean"], .05)
            self.assertEqual(normalizer["training_sample_ids"], ["train"])
            sampler = load_sampler(output / "checkpoint.pt", device="cpu")
            task = fixture_task("test", "test", "test", .2)
            free, metadata = sampler.sample(task, K=1, seed=21)
            repeated, _ = sampler.sample(task, K=1, seed=21)
            self.assertEqual(free.shape, (1, 30, 17))
            self.assertTrue(np.isfinite(free).all())
            np.testing.assert_array_equal(free, repeated)
            self.assertFalse(metadata["force_or_torque_output"])
            self.assertFalse(metadata["clipping_projection_or_trajectory_optimization"])
            self.assertEqual(metadata["postprocessing"], [])
            self.assertEqual(metadata["noise_schedule"]["schedule_sha256"], report["schedule_sha256"])
            self.assertEqual(metadata["parameterization_sha256"], report["parameterization_sha256"])
            checkpoint = torch.load(output / "checkpoint.pt", weights_only=True)
            checkpoint["model_config"]["noise_schedule"] = "linear_legacy_unscaled"
            changed = output / "changed_schedule.pt"
            torch.save(checkpoint, changed)
            with self.assertRaisesRegex(ValueError, "schedule buffers differ"):
                load_sampler(changed, device="cpu")
            checkpoint["model_config"]["noise_schedule"]="cosine"
            checkpoint["schema"]="v6_4_conditional_ddpm_checkpoint_v2"
            torch.save(checkpoint,changed)
            with self.assertRaisesRegex(ValueError,"must not reinterpret"):
                load_sampler(changed,device="cpu")
            (output / "normalizer.json").write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "companion identity changed"):
                load_sampler(output / "checkpoint.pt", device="cpu")

    def test_changed_source_cannot_produce_qualified_checkpoint(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            manifest = self.make_manifest(root)
            identities = [{"git_head": "before", "learning_sources_sha256": {"source": "a"}},
                          {"git_head": "after", "learning_sources_sha256": {"source": "a"}}]
            with patch("v6_4.train_diffusion._source_identity", side_effect=identities):
                with self.assertRaisesRegex(ValueError, "changed during training"):
                    train_from_manifest(manifest, root / "changed", epochs=1, batch_size=1, device="cpu", cpu_threads=2)
            self.assertFalse((root / "changed" / "checkpoint.pt").exists())
            failure = json.loads((root / "changed" / "training_failure.json").read_text(encoding="utf-8"))
            self.assertFalse(failure["evidence_valid"])
            self.assertTrue(failure["partial_log_retained"])

    def test_real_noise_repetition_and_last_ema_weights(self):
        from v6_4.diffusion_model import ConditionalDDPM
        import v6_4.train_diffusion as module
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            manifest, parent = self.make_train_parent(root)
            original_loss = ConditionalDDPM.epsilon_loss
            original_update = module._update_ema
            batches, optimizer_states = [], []
            def observe_loss(model, controls, condition, timesteps=None, noise=None, generator=None):
                t = torch.randint(0, 100, (len(controls),), generator=generator)
                noise = torch.randn(controls.shape, generator=generator)
                batches.append((controls.detach().clone(), noise.detach().clone()))
                return original_loss(model, controls, condition, timesteps=t, noise=noise)
            def observe_update(ema, model, decay):
                optimizer_states.append({k:v.detach().clone() for k,v in model.state_dict().items()})
                original_update(ema, model, decay)
            with patch.object(ConditionalDDPM, "epsilon_loss", observe_loss), patch.object(module, "_update_ema", observe_update):
                report = optimize_epsilon_from_manifest(manifest, parent / "checkpoint.pt", root / "optimized",
                    optimizer_steps=2, noise_repeats=4, ema_decay=.5, device="cpu", cpu_threads=2)
            self.assertEqual([len(x[0]) for x in batches], [8,8])
            for controls, noise in batches:
                self.assertEqual(torch.unique(noise.reshape(8,-1),dim=0).shape[0],8)
                torch.testing.assert_close(controls[:4],controls[:1].repeat(4,1,1))
                torch.testing.assert_close(controls[4:],controls[4:5].repeat(4,1,1))
            self.assertEqual(report["independent_noise_draws_total"],16)
            self.assertEqual(report["train_samples"],2)
            self.assertIsNone(report["epochs"])
            self.assertEqual(report["ema_parent_coefficient_at_end"],.25)
            self.assertTrue(report["source_unchanged"] and report["inputs_unchanged"])
            initial=torch.load(parent / "checkpoint.pt",weights_only=True)["model_state_dict"]
            checkpoint=torch.load(root / "optimized/checkpoint.pt",weights_only=True)
            for name,value in checkpoint["model_state_dict"].items():
                if name.startswith("denoiser."):
                    expected=.25*initial[name]+.25*optimizer_states[0][name]+.5*optimizer_states[1][name]
                    torch.testing.assert_close(value,expected,atol=1e-6,rtol=1e-5)
                else:
                    torch.testing.assert_close(value,initial[name],atol=0,rtol=0)
            sampler=load_sampler(root / "optimized/checkpoint.pt",device="cpu")
            self.assertEqual(sampler.weight_selection["weights"],"last_step_ema")
            with self.assertRaisesRegex(ValueError,"not a qualified"):
                load_sampler(root / "optimized/last_optimizer_weights.pt",device="cpu")
            checkpoint["weight_selection"]["weights"]="last_epoch_raw"
            torch.save(checkpoint,root / "optimized/relabelled.pt")
            with self.assertRaisesRegex(ValueError,"weight selection is inconsistent"):
                load_sampler(root / "optimized/relabelled.pt",device="cpu")

    def test_parent_source_archive_and_teacher_binding_fail_closed(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            manifest,parent=self.make_train_parent(root)
            archived=parent / "source_snapshot/v6_4/diffusion_model.py"
            archived.write_text("changed",encoding="utf-8")
            with self.assertRaisesRegex(ValueError,"archive changed"):
                optimize_epsilon_from_manifest(manifest,parent / "checkpoint.pt",root / "rejected",optimizer_steps=1,device="cpu")
            self.assertFalse((root / "rejected").exists())

    def test_continuation_cannot_rename_parent_data_or_publish_on_source_change(self):
        import v6_4.train_diffusion as module
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            manifest,parent=self.make_train_parent(root)
            samples=json.loads(manifest.read_text(encoding="utf-8"))["samples"]
            samples[0]["sample_id"]="renamed_source"
            changed=root / "renamed.json"
            write_teacher_manifest(samples,changed)
            with self.assertRaisesRegex(ValueError,"exactly the parent's"):
                optimize_epsilon_from_manifest(changed,parent / "checkpoint.pt",root / "renamed_output",optimizer_steps=1,device="cpu")
            self.assertFalse((root / "renamed_output").exists())
            identity=module._source_identity()
            after=dict(identity);after["git_head"]="changed"
            with patch.object(module,"_source_identity",side_effect=[identity,identity,after]):
                with self.assertRaisesRegex(ValueError,"source/HEAD or parent"):
                    optimize_epsilon_from_manifest(manifest,parent / "checkpoint.pt",root / "source_changed",
                        optimizer_steps=1,noise_repeats=2,device="cpu",cpu_threads=2)
            self.assertFalse((root / "source_changed/checkpoint.pt").exists())
            failure=json.loads((root / "source_changed/training_failure.json").read_text(encoding="utf-8"))
            self.assertTrue(failure["partial_log_retained"])
            self.assertFalse(failure["evidence_valid"])

    def test_ema_formula_does_not_average_schedule_buffers(self):
        from v6_4.diffusion_model import ConditionalDDPM,DiffusionConfig
        model=ConditionalDDPM(DiffusionConfig(condition_dim=2))
        ema=__import__('copy').deepcopy(model)
        schedule=ema.alpha_bars.clone()
        with torch.no_grad():
            for p in ema.parameters():p.fill_(2.)
            for p in model.parameters():p.fill_(6.)
        _update_ema(ema,model,.9)
        for p in ema.parameters():torch.testing.assert_close(p,torch.full_like(p,2.4))
        torch.testing.assert_close(ema.alpha_bars,schedule,atol=0,rtol=0)

    def test_from_scratch_v_real_noise_and_warmup_ema_exact(self):
        from v6_4.diffusion_model import ConditionalDDPM
        from v6_4.dataset import sha256
        import v6_4.train_diffusion as module
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);manifest,_=self.make_train_parent(root)
            original_loss=ConditionalDDPM.v_loss;original_update=module._update_ema
            batches,initial,states,decays=[],{},[],[]
            def observe_loss(model,controls,condition,timesteps=None,noise=None,generator=None):
                t=torch.randint(0,100,(len(controls),),generator=generator)
                eps=torch.randn(controls.shape,generator=generator)
                batches.append((controls.detach().clone(),eps.clone()))
                if not initial:initial.update({k:v.detach().clone() for k,v in model.state_dict().items()})
                return original_loss(model,controls,condition,t,eps)
            def observe_update(ema,model,decay):
                states.append({k:v.detach().clone() for k,v in model.state_dict().items()});decays.append(decay)
                original_update(ema,model,decay)
            with patch.object(ConditionalDDPM,"v_loss",observe_loss),patch.object(module,"_update_ema",observe_update):
                report=train_v_from_manifest(manifest,root/'v',optimizer_steps=2,noise_repeats=4,device="cpu",cpu_threads=2)
            self.assertEqual([len(x[0]) for x in batches],[8,8])
            self.assertEqual(decays,[2/11,3/12])
            self.assertEqual(report['ema_initialization_coefficient_at_end'],(2/11)*(3/12))
            for controls,eps in batches:
                self.assertEqual(len(torch.unique(eps.reshape(8,-1),dim=0)),8)
                torch.testing.assert_close(controls[:4],controls[:1].repeat(4,1,1))
            cp=torch.load(root/'v/checkpoint.pt',weights_only=True)
            for k,value in cp['model_state_dict'].items():
                if k.startswith('denoiser.'):
                    expected=(3/12)*((2/11)*initial[k]+(1-2/11)*states[0][k])+(1-3/12)*states[1][k]
                    torch.testing.assert_close(value,expected,rtol=1e-5,atol=1e-6)
                else:torch.testing.assert_close(value,initial[k],rtol=0,atol=0)
            self.assertEqual(cp['objective'],'v_mse');self.assertEqual(report['independent_noise_draws_total'],16)
            sampler=load_sampler(root/'v/checkpoint.pt',device="cpu")
            self.assertTrue(sampler.weight_selection['ema_warmup'])
            self.assertEqual(sampler.config.objective,'v_mse')
            cp['objective']='epsilon_mse';torch.save(cp,root/'v/relabelled.pt')
            with self.assertRaisesRegex(ValueError,'objective identity'):
                load_sampler(root/'v/relabelled.pt',device="cpu")

    def test_v_train_only_and_frozen_inputs_fail_closed(self):
        import v6_4.train_diffusion as module
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);mixed=self.make_manifest(root)
            with self.assertRaisesRegex(ValueError,'TRAIN data only'):
                train_v_from_manifest(mixed,root/'mixed',optimizer_steps=1,device="cpu")
            self.assertFalse((root/'mixed').exists())
            manifest,_=self.make_train_parent(root)
            identity=module._source_identity();after=dict(identity);after['git_head']='changed'
            with patch.object(module,'_source_identity',side_effect=[identity,after]):
                with self.assertRaisesRegex(ValueError,'source/HEAD'):
                    train_v_from_manifest(manifest,root/'changed',optimizer_steps=1,noise_repeats=1,device="cpu",cpu_threads=2)
            self.assertFalse((root/'changed/checkpoint.pt').exists())
            self.assertTrue((root/'changed/training_failure.json').exists())
            token=root/'protected.json';token.write_text('before')
            original=module._update_ema
            def change_protected(ema,model,decay):
                original(ema,model,decay);token.write_text('after')
            with patch.object(module,'_update_ema',change_protected):
                with self.assertRaisesRegex(ValueError,'history input changed'):
                    train_v_from_manifest(manifest,root/'input_changed',optimizer_steps=1,noise_repeats=1,
                        device="cpu",cpu_threads=2,protected_inputs=(token,))
            self.assertFalse((root/'input_changed/checkpoint.pt').exists())

    def test_historical_v3_v4_missing_objective_stays_epsilon(self):
        from v6_4.dataset import sha256
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);_,parent=self.make_train_parent(root)
            cp=torch.load(parent/'checkpoint.pt',weights_only=True)
            cp['model_config'].pop('objective')
            config=json.loads((parent/'config.json').read_text(encoding='utf-8'))
            config['model'].pop('objective')
            for key in ('objective','objective_identity','objective_sha256'):
                config.pop(key);cp.pop(key)
            (parent/'config.json').write_text(json.dumps(config),encoding='utf-8')
            cp['config_sha256']=sha256(parent/'config.json')
            for version in (3,4):
                cp['schema']=f'v6_4_conditional_ddpm_checkpoint_v{version}'
                torch.save(cp,parent/'historical.pt')
                sampler=load_sampler(parent/'historical.pt',device='cpu')
                self.assertEqual(sampler.config.objective,'epsilon_mse')
                self.assertTrue(sampler.legacy_missing_objective)
                cp['model_config']['objective']='v_mse';torch.save(cp,parent/'relabelled.pt')
                with self.assertRaisesRegex(ValueError,'must not reinterpret epsilon'):
                    load_sampler(parent/'relabelled.pt',device='cpu')
                cp['model_config'].pop('objective')

    def test_clean_checkpoint_tag_identity_and_source_guard(self):
        import v6_4.train_diffusion as module
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);manifest,_=self.make_train_parent(root)
            report=train_clean_x0_from_manifest(manifest,root/'clean',optimizer_steps=2,noise_repeats=3,device='cpu',cpu_threads=2)
            self.assertEqual(report['effective_noise_batch'],6)
            self.assertIn('finite clean_x0 parameterization mechanism',report['scope'])
            self.assertTrue(report['scope'].startswith(json.loads((root/'clean/config.json').read_text(encoding='utf-8'))['scope']))
            cp=torch.load(root/'clean/checkpoint.pt',weights_only=True)
            self.assertEqual(cp['schema'],'v6_4_conditional_ddpm_checkpoint_v6')
            self.assertEqual(cp['model_config']['parameterization'],'clean_x0')
            self.assertEqual(cp['objective'],'v_mse')
            sampler=load_sampler(root/'clean/checkpoint.pt',device='cpu')
            task=fixture_task('train','train0','train0')
            raw,metadata=sampler.sample(task,K=1,seed=64)
            self.assertTrue(np.isfinite(raw).all())
            self.assertEqual(metadata['parameterization_identity']['network_output'],'unclipped_clean_x0')
            self.assertEqual(metadata['postprocessing'],[])
            cp['schema']='v6_4_conditional_ddpm_checkpoint_v5';torch.save(cp,root/'clean/old_tag.pt')
            with self.assertRaisesRegex(ValueError,'must not reinterpret weights as clean_x0'):
                load_sampler(root/'clean/old_tag.pt',device='cpu')
            cp['schema']='v6_4_conditional_ddpm_checkpoint_v6';cp['model_config']['parameterization']='epsilon_residual'
            torch.save(cp,root/'clean/wrong_parameterization.pt')
            with self.assertRaisesRegex(ValueError,'parameterization identity'):
                load_sampler(root/'clean/wrong_parameterization.pt',device='cpu')
            identity=module._source_identity();identity=dict(identity)
            identity['learning_sources_sha256']=dict(identity['learning_sources_sha256'])
            identity['learning_sources_sha256']['v6_4\\diffusion_model.py']='changed'
            with patch.object(module,'_source_identity',return_value=identity):
                with self.assertRaisesRegex(ValueError,'source'):
                    sampler.sample(task,K=1,seed=64)
            before=module._source_identity();after=dict(before);after['git_head']='changed_during_sampling'
            with patch.object(module,'_source_identity',side_effect=[before,after]):
                with self.assertRaisesRegex(ValueError,'source/HEAD'):
                    sampler.sample(task,K=1,seed=64)

    def test_v5_residual_identity_is_preserved_without_clean_reinterpretation(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);manifest,_=self.make_train_parent(root)
            report=train_v_from_manifest(manifest,root/'v',optimizer_steps=1,noise_repeats=2,device='cpu',cpu_threads=2)
            cp=torch.load(root/'v/checkpoint.pt',weights_only=True);cp['schema']='v6_4_conditional_ddpm_checkpoint_v5'
            torch.save(cp,root/'v/v5.pt')
            sampler=load_sampler(root/'v/v5.pt',device='cpu')
            self.assertEqual(sampler.config.parameterization,'epsilon_residual')
            self.assertEqual(sampler.config.objective,'v_mse')
            self.assertEqual(sampler.parameterization,report['parameterization_identity'])
            self.assertEqual(sampler.objective,report['objective_identity'])
            cp['model_config']['parameterization']='clean_x0';torch.save(cp,root/'v/relabelled.pt')
            with self.assertRaisesRegex(ValueError,'must not reinterpret weights as clean_x0'):
                load_sampler(root/'v/relabelled.pt',device='cpu')

    @unittest.skipUnless(HAS_TORCH and torch.cuda.is_available(), "isolated CUDA runtime unavailable")
    def test_cuda_clean_real_noise_repeats_and_last_ema(self):
        from v6_4.diffusion_model import ConditionalDDPM
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);manifest,_=self.make_train_parent(root)
            sizes=[];original=ConditionalDDPM.v_loss
            def observe(model,controls,condition,*args,**kwargs):
                sizes.append(len(controls));return original(model,controls,condition,*args,**kwargs)
            with patch.object(ConditionalDDPM,'v_loss',observe):
                report=train_clean_x0_from_manifest(manifest,root/'clean_cuda',optimizer_steps=2,noise_repeats=4,device='cuda',cpu_threads=2)
            self.assertEqual(sizes,[8,8]);self.assertEqual(report['independent_noise_draws_total'],16)
            sampler=load_sampler(root/'clean_cuda/checkpoint.pt',device='cuda')
            self.assertEqual(sampler.config.parameterization,'clean_x0')
            self.assertEqual(sampler.weight_selection['weights'],'last_step_ema')

    @unittest.skipUnless(HAS_TORCH and torch.cuda.is_available(), "isolated CUDA runtime unavailable")
    def test_cuda_real_repeated_noise_optimization(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            manifest,parent=self.make_train_parent(root)
            report=optimize_epsilon_from_manifest(manifest,parent / "checkpoint.pt",root / "cuda",
                optimizer_steps=2,noise_repeats=3,device="cuda",cpu_threads=2)
            self.assertEqual(report["effective_noise_batch"],6)
            self.assertEqual(report["independent_noise_draws_total"],12)
            self.assertTrue(report["complete"] and report["inputs_unchanged"])
            self.assertEqual(load_sampler(root / "cuda/checkpoint.pt",device="cuda").device.type,"cuda")


if __name__ == "__main__":
    unittest.main()
