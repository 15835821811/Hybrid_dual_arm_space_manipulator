import unittest
from unittest.mock import patch
from dataclasses import replace

import mujoco
import numpy as np

from model_test.whole_body_verifier_v5 import WholeBodyCollisionVerifier,WholeBodyVerificationConfig
from v6_lite.run_v6_lite import default_v6_lite_robot_spec,build_scenarios,V6LiteRunConfig
from v6_4.task_protocol import task_from_scenario
from v6_4.reference_adapter import SplineReferenceProvider,scenario_from_task,_RotationReference
from v6_4.trajectory_codec import CubicBSplineCodec


class AdapterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spec=default_v6_lite_robot_spec()
        cls.scene=build_scenarios(cls.spec,V6LiteRunConfig(scenario_count=3))[0]
        cls.task=task_from_scenario(cls.spec,cls.scene,task_id='adapter_unit',group_id='adapter_unit',family='multiple_routes',split='train')
        cls.verifier=WholeBodyCollisionVerifier(cls.spec,cls.scene.obstacles,WholeBodyVerificationConfig(minimum_clearance=.005,query_distance_max=2.5,adaptive_subdivisions=4,include_target_satellite_pairs=True))
        cls.model=cls.verifier.model;cls.model.geom_contype[:]=0;cls.model.geom_conaffinity[:]=0

    def data(self):
        data=mujoco.MjData(self.model);data.qpos[:]=self.task.initial_qpos;data.qvel[:]=self.task.initial_qvel
        data.ctrl[:]=0.;mujoco.mj_forward(self.model,data);return data

    def controls(self, moving=False):
        codec=CubicBSplineCodec(self.task.initial_planner_q,self.task.initial_planner_dq)
        free=np.tile(self.task.initial_planner_q,(30,1))
        if moving:free[:,0]+=.02*np.linspace(0,1,30)
        return codec.decode_free(free)

    def test_prepare_and_sample_never_mutate_real_data_or_step(self):
        data=self.data()
        before={k:np.asarray(getattr(data,k)).tobytes() for k in ['qpos','qvel','ctrl','qacc','qM','xpos','xmat','qacc_warmstart']}
        before_time=data.time
        with patch.object(mujoco,'mj_step',side_effect=AssertionError('physics forbidden')):
            provider=SplineReferenceProvider(self.task,self.controls()).prepare(self.spec,self.model,data,self.scene)
            for t in [0.,13.5,27.,27.-4e-12]:self.assertEqual(len(provider.sample(t)),10)
        for key,value in before.items():self.assertEqual(value,np.asarray(getattr(data,key)).tobytes())
        self.assertEqual(before_time,data.time)
        self.assertEqual(provider.prediction['time'].shape,(1351,))
        self.assertEqual(provider.prediction['full_qpos'].shape,(1351,81))
        np.testing.assert_array_equal(provider.prediction['full_qpos'][0],data.qpos)
        np.testing.assert_array_equal(provider.prediction['full_qvel'][0],data.qvel)
        self.assertLess(provider.prediction['zero_momentum_map_residual'].max(),1e-10)

    def test_nonzero_shape_motion_predicts_base_reaction(self):
        data=self.data();provider=SplineReferenceProvider(self.task,self.controls(True)).prepare(self.spec,self.model,data,self.scene)
        self.assertGreater(np.max(abs(provider.prediction['base_pose'][:,0:3]-provider.prediction['base_pose'][0,0:3])),1e-7)
        self.assertFalse(provider.prediction['full_qpos'].flags.writeable)
        with self.assertRaises(ValueError):provider.prediction['q'][0,0]=1.

    def test_task_scenario_roundtrip_and_identity_rejections(self):
        self.assertEqual(scenario_from_task(self.task).to_dict(),self.scene.to_dict())
        data=self.data();data.qpos[0]+=1e-9
        with self.assertRaises(ValueError):SplineReferenceProvider(self.task,self.controls()).prepare(self.spec,self.model,data,self.scene)
        different=replace(self.scene,scenario_id='another_scene')
        with self.assertRaises(ValueError):SplineReferenceProvider(self.task,self.controls()).prepare(self.spec,self.model,self.data(),different)

    def test_sample_requires_prepare_and_bounded_time(self):
        provider=SplineReferenceProvider(self.task,self.controls())
        with self.assertRaises(RuntimeError):provider.sample(0.)
        provider.prepare(self.spec,self.model,self.data(),self.scene)
        for t in [float('nan'),-.01,27.01]:
            with self.assertRaises(ValueError):provider.sample(t)

    def test_quaternion_reference_world_velocity_and_grid_rotations(self):
        from scipy.spatial.transform import Rotation
        times=np.array([0.,1.,2.]);rotations=Rotation.from_rotvec(np.c_[np.zeros(3),np.zeros(3),.1*times]).as_matrix()
        rotations.setflags(write=False)
        reference=_RotationReference(times,rotations,np.tile([0,0,.1],(3,1)))
        for i,t in enumerate(times):
            rotation,omega=reference.sample(float(t))
            np.testing.assert_allclose(rotation,rotations[i],atol=1e-14,rtol=0.)
            np.testing.assert_allclose(omega,[0,0,.1],atol=1e-14,rtol=0.)


if __name__=='__main__':unittest.main()
