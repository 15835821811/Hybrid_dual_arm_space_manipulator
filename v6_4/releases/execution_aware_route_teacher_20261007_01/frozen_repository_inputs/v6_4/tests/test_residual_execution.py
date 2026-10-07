"""Prefix evidence must retain consumed data without inventing execution."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

class ResidualExecutionTests(unittest.TestCase):
    def test_missing_last_boundary_uses_saved_actual_state_and_preserves_source(self):
        from v6_4.residual_execution import consumed_prefix_view
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);source=root/'interval_partial_trace.npz';output=root/'evaluation';output.mkdir()
            values={'torque':np.arange(20*67).reshape(20,67), 'task_qpos':np.arange(2*81).reshape(2,81),
                'actual_full_qpos':np.arange(20*81).reshape(20,81), 'task_selected_command':np.zeros((3,17))}
            np.savez_compressed(source,**values);original=source.read_bytes()
            trace,view=consumed_prefix_view(source,output)
            self.assertEqual(source.read_bytes(),original)
            self.assertEqual(trace['task_qpos'].shape,(3,81))
            np.testing.assert_array_equal(trace['task_qpos'][:2],values['task_qpos'])
            np.testing.assert_array_equal(trace['task_qpos'][-1],values['actual_full_qpos'][-1])
            for key in values:
                if key!='task_qpos':np.testing.assert_array_equal(trace[key],values[key])
            receipt=json.loads((output/'trace_view_identity.json').read_text())
            self.assertEqual(receipt['physics_steps_added'],0)
            self.assertFalse(receipt['torques_changed']);self.assertTrue(receipt['source_unchanged'])

    def test_complete_trace_is_not_rewritten(self):
        from v6_4.residual_execution import consumed_prefix_view
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);source=root/'trace.npz';output=root/'evaluation';output.mkdir()
            np.savez_compressed(source,torque=np.zeros((10,67)),task_qpos=np.zeros((2,81)))
            _,view=consumed_prefix_view(source,output)
            self.assertEqual(view,source)
            self.assertFalse((output/'consumed_prefix_evidence_view.npz').exists())

    def test_empty_or_incomplete_ramp_not_accepted_as_evidence(self):
        from v6_4.residual_execution import consumed_prefix_view
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            for n in (0,9,11):
                source=root/f'trace{n}.npz';np.savez_compressed(source,torque=np.zeros((n,67)),task_qpos=np.zeros((2,81)))
                with self.assertRaises(ValueError):consumed_prefix_view(source,root)

if __name__=='__main__':unittest.main()
