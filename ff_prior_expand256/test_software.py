"""Software and lineage checks; fixtures are not physical experiment evidence."""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from ff_prior.storage import read_json,file_digest,digest
from ff_prior_pilot_train.storage import atomic_json,Ledger
from .protocol import proposal_id,proposal_seed,SELECTION
from .inputs import identity_for
from .reporting import gate
from .data import admit,apply_execution_grade,index_for
from .job import completed_no_work


class DatasetSoftwareTests(unittest.TestCase):
    def test_frozen_quota_unique_ids_and_seeds(self):
        ids=[proposal_id(t,s,a) for t in range(64) for s in range(4) for a in range(2)]
        seeds=[proposal_seed(t,s,a) for t in range(64) for s in range(4) for a in range(2)]
        self.assertEqual(len(set(ids)),512)
        self.assertEqual(len(set(seeds)),512)
        for args in ((64,0,0),(0,4,0),(0,0,2),(-1,0,0)):
            with self.assertRaises(ValueError):proposal_id(*args)

    def test_gate_exact_boundaries_and_no_synthetic_pass(self):
        good=[128,32,26,32,256,256,[],'PASS']
        self.assertEqual(gate(*good)['status'],'PASS')
        for position,value in [(0,127),(1,31),(2,25),(3,31),(4,255),(5,257),(6,['unknown']),(7,'FAIL')]:
            bad=good.copy();bad[position]=value
            self.assertEqual(gate(*bad)['status'],'PARTIAL_QUALIFIED')

    def test_real_frozen_lineage_and_strata(self):
        identity,(_,selection,index,refs)=identity_for();entries=selection['entries']
        self.assertEqual(len(entries),64)
        self.assertEqual(sum(bool(e['scene']['objects']) for e in entries),32)
        self.assertEqual(len({e['origin_hash'] for e in entries}),64)
        self.assertEqual(len(set(selection['execution_target_ids'])),32)
        self.assertEqual(sum(e['split']=='TRAIN' for e in index),13966)
        self.assertEqual(selection['entries_digest'],digest(entries))
        self.assertEqual(identity['sampler'],'late4_diffusion');self.assertTrue(refs)
        self.assertEqual(sum(len(e['proposals']) for e in entries),512)

    def test_rejection_and_exact_duplicate_never_write_records(self):
        entry={'target_id':'fixture','task':{'task_hash':'fixture'}}
        with tempfile.TemporaryDirectory() as d:
            out=Path(d)
            r={'status':'DONE','eligible':False,'reference_hash':'rejected','model_calls':{}}
            self.assertFalse(admit(out,entry,'fixture',r,None,set())['admitted'])
            r.update(eligible=True,reference_hash='known')
            duplicate=admit(out,entry,'fixture',r,None,{'known'})
            self.assertFalse(duplicate['admitted'])
            self.assertEqual(duplicate['reason'],'DUPLICATE_REFERENCE')
            self.assertEqual(list(out.iterdir()),[])

    def test_failed_or_unknown_execution_excluded(self):
        for state,expected in [('EXECUTION_FAILED','FAIL'),('INTERRUPTED','UNKNOWN')]:
            with tempfile.TemporaryDirectory() as d:
                out=Path(d)
                atomic_json(out/'executions/t/selection.json',{'selected_id':'x'})
                atomic_json(out/'records/x.json',{'sample_id':'x','validation_level':'MODEL_VALID','execution_status':'NOT_RUN','training_eligible':True})
                ledger=SimpleNamespace(data={'completed':{'execution_t':{'status':state}}})
                apply_execution_grade(out,ledger,'t')
                row=read_json(out/'records/x.json')
                self.assertEqual(row['execution_status'],expected)
                self.assertFalse(row['training_eligible'])
                self.assertEqual(row['validation_level'],'MODEL_VALID')
                with patch('ff_prior_expand256.data.population',return_value=[row]):
                    merged,new=index_for(out,[],ledger)
                self.assertEqual(new,[])

    def test_unknown_generation_is_consumed(self):
        with tempfile.TemporaryDirectory() as d:
            ledger=Ledger(d,{'fixture':True})
            ledger.begin('once','generation',100,256)
            ledger=Ledger(d,{'fixture':True})
            self.assertFalse(ledger.begin('once','generation',100,256))
            self.assertEqual(ledger.data['completed']['once']['status'],'INTERRUPTED')
            self.assertEqual(ledger.data['attempts']['generation'],1)

    def test_completed_resume_is_read_only(self):
        with tempfile.TemporaryDirectory() as d:
            out=Path(d);identity={'source':'software_fixture'}
            atomic_json(out/'run.json',{'identity':identity})
            atomic_json(out/'summary.json',{'status':'FINISHED_WITH_BLOCKED_ITEMS'})
            atomic_json(out/'status.json',{'status':'FINISHED_WITH_BLOCKED_ITEMS','pid':0})
            before={p.name:file_digest(p) for p in out.iterdir()}
            with patch('ff_prior_expand256.inputs.identity_for',return_value=(identity,None)):
                r=completed_no_work(out)
            self.assertEqual(r['resume'],'NO_WORK_ALREADY_COMPLETE')
            self.assertEqual(before,{p.name:file_digest(p) for p in out.iterdir()})


if __name__=='__main__':
    unittest.main()
