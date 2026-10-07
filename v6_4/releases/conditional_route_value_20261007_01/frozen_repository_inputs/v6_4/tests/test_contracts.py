import unittest
from dataclasses import FrozenInstanceError
import numpy as np

from v6_4.contracts import TrajectoryProposal


class ProposalContractTests(unittest.TestCase):
    def proposal(self):return TrajectoryProposal('task','a'*64,np.zeros((30,17)),'teacher',5,('explicit_fit',),'{}')

    def test_frozen_roundtrip_and_mutation_isolation(self):
        proposal=self.proposal();original=proposal.sha256()
        copied=proposal.free_controls;copied[0,0]=3.
        self.assertEqual(proposal.sha256(),original)
        self.assertEqual(TrajectoryProposal.from_dict(proposal.to_dict()),proposal)
        with self.assertRaises(FrozenInstanceError):proposal.task_id='other'
        self.assertFalse(proposal.to_dict()['closed_loop_success_established'])

    def test_invalid_values_and_unknown_representation_rejected(self):
        for values in [np.zeros((32,17)),np.full((30,17),np.nan)]:
            with self.assertRaises(ValueError):TrajectoryProposal('task','a'*64,values,'diffusion')
        value=self.proposal().to_dict();value['representation']='another_codec'
        with self.assertRaises(ValueError):TrajectoryProposal.from_dict(value)


if __name__=='__main__':unittest.main()
