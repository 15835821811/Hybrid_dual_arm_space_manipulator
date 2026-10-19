"""Software and provenance checks; no fixture is a physical success claim."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from ff_prior.storage import read_json,file_digest,digest
from ff_prior_pilot_train.storage import atomic_json
from ff_prior.diffusion import schedule
from ff_prior_origin_learn_bounded.learning import objective
from ff_prior_origin_learn_bounded.distribution import CoverageSampler
from ff_prior_delivery.gates import engineering_gate
from .inputs import identity_for
from .protocol import OLD,CACHE,CACHE_HASHES,PROTOCOL
from .data import same_entry
from .learning import resume_permission
from .job import completed_no_work


class TrainingSoftwareTests(unittest.TestCase):
    def test_real_frozen_input_counts_and_same_development_tasks(self):
        identity,(_,index,entries,new_ids)=identity_for()
        self.assertEqual(len(index),14014)
        self.assertEqual(sum(e['split']=='TRAIN' for e in index),13966)
        self.assertEqual(len(new_ids),128)
        self.assertEqual(entries,read_json(OLD/'frozen_tasks.json')['entries'])
        self.assertEqual(len(entries),12)
        self.assertEqual(identity['comparison_tasks_hash'],digest(entries))
        for name,h in CACHE_HASHES.items():self.assertEqual(file_digest(CACHE/name),h)

    def test_index_scope_flag_does_not_hide_source_changes(self):
        a={'sample_id':'x','arrays_sha256':'a','new_in_this_run':True}
        self.assertTrue(same_entry(a,dict(a,new_in_this_run=False)))
        self.assertFalse(same_entry(a,dict(a,arrays_sha256='b')))

    def test_training_recovery_never_replays_unknown_updates(self):
        self.assertEqual(resume_permission(None,0),'FRESH')
        self.assertEqual(resume_permission({'status':'COMMITTED','committed_updates':20},20),'CONTINUE_COMMITTED')
        self.assertEqual(resume_permission({'status':'PENDING','end_update':40},40),'RECOVER_COMMITTED_CHECKPOINT')
        with self.assertRaisesRegex(RuntimeError,'UNKNOWN_WORK'):
            resume_permission({'status':'PENDING','end_update':40},20)
        with self.assertRaises(ValueError):resume_permission(None,20)
        with self.assertRaises(ValueError):resume_permission({'status':'COMMITTED','committed_updates':40},20)

    def test_local_objective_finite_across_noise_scales(self):
        generator=torch.Generator().manual_seed(47)
        alpha=schedule()
        for t in (0,500,999):
            prediction=torch.randn((4,102),generator=generator,requires_grad=True)
            clean=torch.randn((4,102),generator=generator)
            noise=torch.randn((4,102),generator=generator);a=alpha[t]
            noisy=a.sqrt()*clean+(1-a).sqrt()*noise;target=a.sqrt()*noise-(1-a).sqrt()*clean
            G=torch.randn((4,12,102),generator=generator)*.01
            loss,vmse,pose=objective(prediction,target,noisy,a,clean,G,.05)
            self.assertAlmostEqual(float(loss.detach()),float((vmse+.05*pose).detach()),places=6)
            loss.backward()
            self.assertTrue(torch.isfinite(prediction.grad).all())
            self.assertGreater(float(prediction.grad.norm()),0.)

    def test_uniform_row_stream_consumes_entire_training_population(self):
        keys=[str(i%97) for i in range(13966)]
        sampler=CoverageSampler(keys,82)
        for _ in range(437):sampler.take(64)
        self.assertTrue(np.all(sampler.row_counts>0))
        resumed=CoverageSampler(keys,82,sampler.state())
        self.assertEqual(sampler.take(64),resumed.take(64))

    def test_engineering_gate_frozen_180_36_denominators(self):
        m=dict(completed=180,eligible=144,selected_outcomes=36,execution_verified=29,candidate_errors=0)
        self.assertEqual(engineering_gate(m,180,36,True,[])['status'],'PASS')
        m['execution_verified']=28
        self.assertEqual(engineering_gate(m,180,36,True,[])['status'],'BLOCKED')

    def test_completed_resume_no_writes(self):
        with tempfile.TemporaryDirectory() as d:
            out=Path(d);identity={'source':'software_fixture'}
            atomic_json(out/'run.json',{'identity':identity});atomic_json(out/'summary.json',{'status':'FINISHED'})
            atomic_json(out/'status.json',{'status':'FINISHED','pid':0})
            before={p.name:file_digest(p) for p in out.iterdir()}
            with patch('ff_prior_delivery_train.inputs.identity_for',return_value=(identity,None)):
                result=completed_no_work(out)
            self.assertEqual(result['resume'],'NO_WORK_ALREADY_COMPLETE')
            self.assertEqual(before,{p.name:file_digest(p) for p in out.iterdir()})


if __name__=='__main__':unittest.main()
