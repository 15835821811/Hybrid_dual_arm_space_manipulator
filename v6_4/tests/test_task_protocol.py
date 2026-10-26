from dataclasses import replace
import unittest

import numpy as np

from v6_4.task_protocol import TaskSpec,TaskPoint,canonical_json,validate_task_splits


def fixture_task():
    qpos=np.zeros(81);qpos[3]=1.;qpos[-4]=1.
    rotation=tuple(np.eye(3).flat)
    points=(TaskPoint('via','continuum','world',(0.,0.,0.),rotation,.02,(.01,.03),.01,.1),
            TaskPoint('c_end','continuum','world',(0.,0.,0.),rotation,27.,(26.,27.),.01,.1,'terminal'),
            TaskPoint('r_end','rigid','target',(0.,0.,0.),rotation,27.,(26.,27.),.01,.1,'terminal'))
    return TaskSpec('case_01','group_01','train','end_effector_detour',7,
        canonical_json({'workspace_obstacles':[],'condition':'frozen'}),tuple(qpos),tuple(np.zeros(79)),
        tuple(np.zeros(17)),tuple(np.zeros(17)),(0.,0.,0.,1.,0.,0.,0.),tuple(np.zeros(6)),
        (0.,0.,0.,1.,0.,0.,0.),tuple(np.zeros(6)),points,'a'*64)


class TaskProtocolTests(unittest.TestCase):
    def test_roundtrip_frozen_task_cannot_be_changed_by_returned_dict(self):
        task=fixture_task();sha=task.sha256();value=task.to_dict()
        self.assertEqual(TaskSpec.from_dict(value).sha256(),sha)
        value['scenario']['condition']='changed';value['requirements'][0]['position_m'][0]=8.
        self.assertEqual(task.sha256(),sha)
        self.assertEqual(task.scenario['condition'],'frozen')

    def test_nonfinite_or_missing_terminal_is_rejected(self):
        task=fixture_task()
        with self.assertRaises(ValueError):replace(task,initial_qvel=tuple([float('nan')]*79))
        with self.assertRaises(ValueError):replace(task,requirements=task.requirements[:-1])
        with self.assertRaises(ValueError):replace(task,scenario_json='{"workspace_obstacles":[],"bad":NaN}')

    def test_group_and_relabelled_source_cannot_leak_to_test(self):
        task=fixture_task()
        for group in ('group_01','renamed_group'):
            other=replace(task,task_id='case_02',group_id=group,split='test')
            with self.assertRaises(ValueError):validate_task_splits((task,other))


if __name__=='__main__':unittest.main()
