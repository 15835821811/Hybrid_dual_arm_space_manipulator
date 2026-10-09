from types import SimpleNamespace
from unittest.mock import patch
import unittest

import numpy as np

from v6_4.contracts import TrajectoryProposal
from v6_4.trajectory_codec import CubicBSplineCodec
from v6_4.proposal_gate import gate_proposal
from v6_4.tests.test_task_protocol import fixture_task


class ProposalGateTests(unittest.TestCase):
    def test_raw_out_of_range_is_rejected_without_clipping_or_geometry(self):
        task=fixture_task();controls=np.zeros((30,17));controls[8,0]=4.
        proposal=TrajectoryProposal.from_controls(task,controls,origin='diffusion')
        spec=SimpleNamespace(planner_lower=np.full(17,-np.pi),planner_upper=np.full(17,np.pi))
        with patch('v6_4.proposal_gate.gate_nominal_prediction') as screen:
            result=gate_proposal(task,proposal,spec,provider=SimpleNamespace())
        self.assertFalse(result['passed']);screen.assert_not_called()
        self.assertEqual(proposal.free_controls[8,0],4.)
        self.assertFalse(result['clipped_or_repaired'])

    def test_other_candidate_prediction_and_repair_credit_cannot_pass_as_raw(self):
        task=fixture_task();controls=np.zeros((30,17));codec=CubicBSplineCodec(task.initial_planner_q,task.initial_planner_dq)
        sample=codec.sample(codec.decode_free(controls),np.arange(1351)*.02)
        provider=SimpleNamespace(prediction={'time':np.arange(1351)*.02,'q':sample['q'].copy(),'dq':sample['dq'].copy()})
        spec=SimpleNamespace(planner_lower=np.full(17,-np.pi),planner_upper=np.full(17,np.pi))
        proposal=TrajectoryProposal.from_controls(task,controls,origin='diffusion',postprocessing=('trajectory_optimization',))
        with patch('v6_4.proposal_gate.gate_nominal_prediction',return_value={'passed':True}):
            result=gate_proposal(task,proposal,spec,provider=provider)
            self.assertTrue(result['passed']);self.assertFalse(result['raw_passed'])
            provider.prediction['q'][20,0]=.1
            different=gate_proposal(task,proposal,spec,provider=provider)
        self.assertFalse(different['passed'])
        self.assertFalse(different['checks']['prediction_bound_to_raw_controls'])


if __name__=='__main__':unittest.main()
