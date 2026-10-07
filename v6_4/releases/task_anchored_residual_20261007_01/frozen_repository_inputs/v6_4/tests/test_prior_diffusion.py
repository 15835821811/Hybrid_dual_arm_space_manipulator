"""Temporary synthetic identity/tensor tests, never new physical teacher data."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

import v6_4.prior_diffusion as p
from v6_4.contracts import TrajectoryProposal
from v6_4.dataset import load_teacher_dataset, sha256, write_teacher_manifest, TrainingNormalizer
from v6_4.diffusion_model import ConditionalDDPM, DiffusionConfig
from v6_4.trajectory_codec import CubicBSplineCodec
from v6_4.tests.test_learning_dataset import fixture_task, synthetic_teacher


class PriorDiffusionTests(unittest.TestCase):
    def fixture(self, root):
        tasks = [fixture_task("train", "a", "ga"), fixture_task("train", "b", "gb", .1)]
        samples = [synthetic_teacher(root, tasks[0], "a", .06), synthetic_teacher(root, tasks[1], "b", .09)]
        teachers = root/"teachers.json"; write_teacher_manifest(samples, teachers)
        bootstrap = root/"fixture_bootstrap.npz"; np.savez(bootstrap, fixture_only=np.array([1]))
        bootstrap_manifest = root/"fixture_bootstrap_manifest.json"; bootstrap_manifest.write_text("{}", encoding="utf-8")
        rows = []
        for index, task in enumerate(tasks):
            folder = root/f"center{index}"; folder.mkdir()
            free = np.full((30, 17), .01+index*.01)
            full = CubicBSplineCodec(task.initial_planner_q, task.initial_planner_dq).decode_free(free)
            np.save(folder/"free.npy", free); np.save(folder/"full.npy", full)
            proposal = TrajectoryProposal.from_controls(task, free, origin="teacher", seed=64,
                metadata={"intrinsic_algorithm": "finite_teacher_arm_anchor_rate_prior_v1", "fixture_only": True})
            (folder/"proposal.json").write_text(json.dumps(proposal.to_dict()), encoding="utf-8")
            # Task-failed center is deliberately allowed as residual coordinates.
            gate = {"task_id": task.task_id, "task_sha256": task.sha256(), "raw_passed": False,
                    "passed": False, "status": "FIXTURE_FINITE_TASK_FAILED"}
            (folder/"gate.json").write_text(json.dumps(gate), encoding="utf-8")
            (folder/"acceleration_guard.json").write_text(json.dumps({"task_id":task.task_id,"task_sha256":task.sha256(),
                "passed":False,"status":"NOT_RUN","limits_rad_s2":[2.5]*10+[4.]*7}),encoding="utf-8")
            row = {"task_id": task.task_id, "task_sha256": task.sha256(), "split": task.split, "group_id": task.group_id,
                   "center_finite": True, "start_boundary_exact": True, "source_unchanged": True,
                   "chosen_attempt_index": 0, "prior_only_gate_raw_passed": False, "prior_only_gate_passed": False,
                   "prior_budget": {"starts": 8, "seed": 64, "maximum_anchor_rate_rounds": 16},
                   "source": {"fixture_only": True}, "costs": {"physics_steps": 0, "total_wall_s": 0.},
                   "bootstrap_identity": {"source": str(bootstrap.resolve()), "sha256": sha256(bootstrap),
                       "source_split": "bootstrap", "source_group_id": "research_acceptance_01:v6_lite_scenario_04"},
                   "proposal_sha256": proposal.sha256()}
            for stem, name in (("center_free", "free.npy"), ("center_full", "full.npy"),
                               ("center_proposal", "proposal.json"), ("prior_only_gate", "gate.json"),
                               ("prior_only_acceleration_guard", "acceleration_guard.json")):
                row.update({stem+"_path": str((folder/name).resolve()), stem+"_sha256": sha256(folder/name)})
            row["center_proposal_file_sha256"] = row["center_proposal_sha256"]
            row["center_proposal_sha256"] = proposal.sha256(); rows.append(row)
        prior_manifest = root/"centers.json"
        prior_manifest.write_text(json.dumps({"schema": p.CENTER_SCHEMA, "complete": True, "source_unchanged": True,
            "all_tasks_attempted": True, "all_centers_constructible": True, "centers": rows}), encoding="utf-8")
        source = {"git_head": "fixture_head", "tracked_dirty": False,
                  "sources_sha256": {"v6_4/prior_diffusion.py": sha256(p.REPO/"v6_4/prior_diffusion.py")}}
        return teachers, prior_manifest, tasks, rows, bootstrap, bootstrap_manifest, source

    def patches(self, fixture):
        return (patch.object(p, "_verify_legacy_bootstrap", return_value=(fixture[4], fixture[5])),
                patch.object(p, "source_identity", return_value=fixture[6]),
                patch.object(p, "_model_contract_and_assets", return_value=("a"*64, (fixture[4],))))

    def train_fixture(self, root, fixture):
        return p.train_prior_residual(fixture[0], fixture[1], root/"model", optimizer_steps=2, noise_repeats=4,
              seed=70, device="cpu", cpu_threads=2, expected_samples=2, expected_groups=2)

    def test_failed_finite_center_retains_every_label_and_original517(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); f=self.fixture(root)
            with self.patches(f)[0]:
                centers,_=p.load_centers(f[1],f[2]);dataset=load_teacher_dataset(f[0])
                normalizer=p.ResidualNormalizer.fit(dataset,centers)
            self.assertEqual(set(centers),{"a","b"})
            self.assertTrue(all(c.row['prior_only_gate_raw_passed'] is False for c in centers.values()))
            self.assertEqual(normalizer.training_sample_ids,('a','b'))
            np.testing.assert_array_equal(normalizer.condition_mean,TrainingNormalizer.fit(dataset).condition_mean)
            residuals=dataset.controls-np.stack([centers[s['task_id']].free for s in dataset.samples])
            np.testing.assert_allclose(normalizer.residual_mean,residuals.mean((0,1)))
            condition=normalizer.condition(f[2][0],centers['a'])
            self.assertEqual(condition.shape,(1027,))
            np.testing.assert_array_equal(condition[517:],centers['a'].free.ravel().astype(np.float32))
            restored=normalizer.restore(normalizer.normalize_residuals(residuals[0]),centers['a'])
            np.testing.assert_allclose(restored,dataset.controls[0],rtol=0,atol=1e-8)
            zero_network=normalizer.restore(np.zeros((30,17)),centers['a'])
            np.testing.assert_array_equal(zero_network,centers['a'].free+normalizer.residual_mean)
            self.assertFalse(np.array_equal(zero_network,centers['a'].free))

    def test_missing_difficult_task_duplicate_and_changed_center_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);f=self.fixture(root); original=json.loads(f[1].read_text())
            with self.patches(f)[0]:
                bad=deepcopy(original);bad['centers']=bad['centers'][:1];f[1].write_text(json.dumps(bad))
                with self.assertRaisesRegex(ValueError,'every declared Task'):p.load_centers(f[1],f[2])
                bad=deepcopy(original);bad['centers'][1]=bad['centers'][0];f[1].write_text(json.dumps(bad))
                with self.assertRaisesRegex(ValueError,'duplicate'):p.load_centers(f[1],f[2])
                f[1].write_text(json.dumps(original));np.save(Path(f[3][0]['center_free_path']),np.full((30,17),9.))
                with self.assertRaisesRegex(ValueError,'SHA mismatch'):p.load_centers(f[1],f[2])

    def test_forged_boundary_task_source_and_future_bootstrap_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);f=self.fixture(root)
            with self.patches(f)[0]:
                row=deepcopy(f[3][0]);row['task_sha256']='f'*64
                with self.assertRaisesRegex(ValueError,'immutable Task'):p.load_center(row,f[2][0])
                row=deepcopy(f[3][0]);row['source_unchanged']=False
                with self.assertRaisesRegex(ValueError,'source changed'):p.load_center(row,f[2][0])
                row=deepcopy(f[3][0]);row['bootstrap_identity']['source_split']='test'
                with self.assertRaisesRegex(ValueError,'no Task future'):p.load_center(row,f[2][0])
                row=deepcopy(f[3][0]);full=np.load(row['center_full_path']);full[0,0]+=.1;np.save(row['center_full_path'],full)
                row['center_full_sha256']=sha256(Path(row['center_full_path']))
                with self.assertRaisesRegex(ValueError,'exact eliminated'):p.load_center(row,f[2][0])

    def test_residual_v_objective_oracle_and_unclipped_center_addition(self):
        config=DiffusionConfig(condition_dim=1027,parameterization='clean_x0',objective='v_mse')
        model=ConditionalDDPM(config);model.eval()
        clean=torch.randn(3,30,17);noise=torch.randn_like(clean);times=torch.tensor([0,50,99]);condition=torch.zeros(3,1027)
        xt,_=model.q_sample(clean,times,noise);a,b=model._coefficients(times)
        predicted_clean=model.denoiser(xt,times,condition);predicted_v=(a*xt-predicted_clean)/b;target_v=a*noise-b*clean
        actual=model.v_loss(clean,condition,timesteps=times,noise=noise)
        self.assertTrue(torch.allclose(actual,(predicted_v-target_v).square().mean(),atol=1e-5,rtol=2e-5))
        actual.backward();self.assertTrue(all(torch.isfinite(v.grad).all() for v in model.parameters() if v.grad is not None))
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);f=self.fixture(root)
            with self.patches(f)[0]:centers,_=p.load_centers(f[1],f[2]);normalizer=p.ResidualNormalizer.fit(load_teacher_dataset(f[0]),centers)
            physical=np.full((30,17),1000.)
            np.testing.assert_allclose(normalizer.restore(normalizer.normalize_residuals(physical),centers['a']),centers['a'].free+physical,rtol=1e-7)
            self.assertGreater(normalizer.restore(normalizer.normalize_residuals(physical),centers['a']).max(),999.)

    def test_true_repeat_ema_checkpoint_sampling_and_no_legacy_reinterpretation(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);f=self.fixture(root);patches=self.patches(f)
            with patches[0],patches[1],patches[2]:
                report=self.train_fixture(root,f)
                self.assertTrue(report['complete'] and report['all_labels_retained'])
                self.assertEqual(report['effective_noise_batch'],8);self.assertEqual(report['independent_noise_draws_total'],16)
                rows=[json.loads(s) for s in (root/'model/training_log.jsonl').read_text().splitlines()]
                self.assertEqual([r['ema_decay'] for r in rows],[2/11,3/12])
                self.assertAlmostEqual(report['ema_initialization_coefficient_at_end'],(2/11)*(3/12))
                centers,_=p.load_centers(f[1],f[2]);sampler=p.load_prior_sampler(root/'model/checkpoint.pt')
                values,residuals,metadata=sampler.sample(f[2][0],centers['a'],K=8,seed=64)
                one,_,_=sampler.sample(f[2][0],centers['a'],K=1,seed=64)
                np.testing.assert_array_equal(values[0],one[0]);np.testing.assert_array_equal(values,centers['a'].free+residuals)
                self.assertEqual(metadata['postprocessing'],[]);self.assertEqual(metadata['method'],p.METHOD)
                self.assertGreater(float(np.max(abs(residuals[1]-residuals[0]))),0.)
                cp=torch.load(root/'model/checkpoint.pt',weights_only=True)
                for version in range(1,7):
                    cp['schema']=f'v6_4_conditional_ddpm_checkpoint_v{version}';torch.save(cp,root/f'legacy{version}.pt')
                    with self.assertRaisesRegex(ValueError,'must not reinterpret v1-v6'):p.load_prior_sampler(root/f'legacy{version}.pt')
                cp['schema']=p.CHECKPOINT_SCHEMA;cp['representation']='absolute_cp';torch.save(cp,root/'wrong.pt')
                with self.assertRaisesRegex(ValueError,'semantic/checkpoint'):p.load_prior_sampler(root/'wrong.pt')
                checkpoint=torch.load(root/'model/checkpoint.pt',weights_only=True);checkpoint['companion_hashes'].pop('normalizer.json')
                torch.save(checkpoint,root/'missing.pt')
                with self.assertRaisesRegex(ValueError,'all checkpoint companions'):p.load_prior_sampler(root/'missing.pt')
                with patch.object(p,'source_identity',return_value={**f[6],'sources_sha256':{'changed':'bad'}}):
                    with self.assertRaisesRegex(ValueError,'sampling source differs'):sampler.sample(f[2][0],centers['a'])

    def test_source_change_retains_failed_training_and_no_checkpoint(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);f=self.fixture(root);changed={**f[6],'git_head':'changed'}
            with self.patches(f)[0],self.patches(f)[2],patch.object(p,'source_identity',side_effect=[f[6],changed]):
                with self.assertRaisesRegex(ValueError,'source/HEAD'):self.train_fixture(root,f)
            failure=json.loads((root/'model/training_failure.json').read_text())
            self.assertFalse(failure['qualified_checkpoint']);self.assertFalse((root/'model/checkpoint.pt').exists())
            self.assertTrue((root/'model/training_log.jsonl').exists())

    def test_acceleration_guard_independent_original_threshold_no_relaxation(self):
        spec=SimpleNamespace(planner_acceleration_limits=np.r_[np.full(10,2.5),np.full(7,4.)])
        state={'time':np.arange(1351)*.02,'ddq':np.zeros((1351,17))}
        self.assertTrue(p.reference_ddq_guard(state,spec)['passed'])
        state['ddq'][77,4]=2.500001;self.assertFalse(p.reference_ddq_guard(state,spec)['passed'])
        state['ddq'][77,4]=0.;state['ddq'][1350,16]=4.000001;self.assertFalse(p.reference_ddq_guard(state,spec)['passed'])
        state['ddq'][1350,16]=0.;state['time'][-1]+=.01;self.assertFalse(p.reference_ddq_guard(state,spec)['passed'])
        state['time']=np.arange(1351)*.02;state['ddq'][0,0]=np.nan;self.assertFalse(p.reference_ddq_guard(state,spec)['passed'])
        state['ddq'][0,0]=0.;spec.planner_acceleration_limits*=2.;self.assertFalse(p.reference_ddq_guard(state,spec)['passed'])


if __name__=='__main__':unittest.main()
