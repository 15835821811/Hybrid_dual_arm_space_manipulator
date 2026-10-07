import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from v6_4.evaluate_planning import _terminal_cartesian_reference_binding


class TerminalCartesianBindingTests(unittest.TestCase):
    def expected(self,t):
        return {'rigid_target_position':np.array([t,0.,0.]),
            'rigid_target_velocity':np.array([1.,0.,0.]),
            'rigid_target_rotation':np.eye(3),'rigid_target_angular_velocity':np.zeros(3),
            'continuum_target_position':np.array([0.,t,0.]),
            'continuum_target_velocity':np.array([0.,1.,0.]),
            'continuum_target_rotation':np.eye(3),'continuum_target_angular_velocity':np.zeros(3),
            'posture_reference_q':np.full(17,t),'posture_reference_dq':np.ones(17)}

    def fixture(self):
        state={'time':np.arange(21)*.002,'qpos':np.zeros((21,81)),'qvel':np.zeros((21,79))}
        trace={'torque':np.zeros((20,67))}
        for key in self.expected(0.):
            # A final NaN input was refused before consumption. Preserve it.
            values=[self.expected(t)[key] for t in (0.,.02)]
            values.append(np.full_like(values[0],np.nan))
            trace['task_input_reference_'+key]=np.stack(values)
        for arm in ('rigid','continuum'):
            for suffix in ('position','rotation'):
                trace['generated_'+arm+'_'+suffix]=np.stack([
                    self.expected(t)[arm+'_target_'+suffix] for t in state['time'][1:]])
        return trace,state

    def run_binding(self,trace,state):
        observed=SimpleNamespace(qpos=np.zeros(81),qvel=np.zeros(79),time=0.)
        task=SimpleNamespace(initial_qpos=np.zeros(81),initial_qvel=np.zeros(79))
        provider=SimpleNamespace(sample=self.expected)
        provider.prepare=lambda *args:provider
        with patch('v6_4.run_planning.prepare_provider',return_value=(object(),SimpleNamespace(model=object()))), \
             patch('v6_4.reference_adapter.scenario_from_task',return_value=object()), \
             patch('v6_4.terminal_progress.TerminalProgressReferenceProvider',return_value=provider), \
             patch('mujoco.MjData',return_value=observed), \
             patch('mujoco.mj_forward'), \
             patch('mujoco.mj_step',side_effect=AssertionError('physics forbidden')):
            return _terminal_cartesian_reference_binding(task,trace,state,np.zeros((32,17)),object())

    def test_all_components_bound_and_unconsumed_refusal_preserved(self):
        trace,state=self.fixture()
        result=self.run_binding(trace,state)
        self.assertTrue(result['passed'])
        self.assertEqual(result['planning_inputs_bound'],2)
        self.assertEqual(len(result['component_maximum_absolute_residual']),14)
        self.assertTrue(np.isnan(trace['task_input_reference_rigid_target_position'][-1]).all())

    def test_consumed_velocity_mismatch_and_nonfinite_angular_input_rejected(self):
        trace,state=self.fixture()
        trace['task_input_reference_rigid_target_velocity'][1,0]+=1e-5
        with self.assertRaisesRegex(ValueError,'binding differs'):self.run_binding(trace,state)
        trace,state=self.fixture()
        trace['task_input_reference_continuum_target_angular_velocity'][0,0]=np.nan
        with self.assertRaisesRegex(ValueError,'must be finite'):self.run_binding(trace,state)


if __name__=='__main__':unittest.main()
