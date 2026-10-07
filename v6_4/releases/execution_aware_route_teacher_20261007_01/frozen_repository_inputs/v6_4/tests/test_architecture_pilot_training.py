"""Focused pairing/scaler/boundary checks; no training or robot physics."""
from copy import deepcopy
import unittest
from unittest.mock import patch

import numpy as np
import torch

from v6_4.architecture_pilot_training import (ConditionNormalizer, _make_model,
    _state_sha, make_conditions, paired_draw)
from v6_4.reference_training_data import ControlNormalizer
from v6_4.tests.test_learning_dataset import fixture_task
from v6_4.typed_condition_encoder import GLOBAL_DIM, TOKEN_COUNT, TOKEN_DIM, CONDITION_SCHEMA


def raw_condition(value=0.):
    mask=np.zeros(TOKEN_COUNT,dtype=bool);mask[:2]=True
    features=np.full((TOKEN_COUNT,TOKEN_DIM),999.)
    features[:2]=value
    types=np.full(TOKEN_COUNT,999,dtype=np.int64);types[:2]=[1,3]
    return {"schema":CONDITION_SCHEMA,"global":np.full(GLOBAL_DIM,value),
        "token_features":features,"token_types":types,"token_mask":mask}


class ArchitecturePilotTrainingTests(unittest.TestCase):
    def test_valid_train_only_condition_scaler_padding_and_types(self):
        encoded=[raw_condition(1.),raw_condition(3.),raw_condition(10000.)]
        scaler=ConditionNormalizer.fit(encoded,[0,1],["train_a","train_b"])
        np.testing.assert_array_equal(scaler.global_mean,np.full(GLOBAL_DIM,2.))
        np.testing.assert_array_equal(scaler.token_mean,np.full(TOKEN_DIM,2.))
        normalized=scaler.normalize(encoded[2])
        np.testing.assert_array_equal(normalized["token_features"][~normalized["token_mask"]],0.)
        np.testing.assert_array_equal(normalized["token_types"][:2],[1,3])
        np.testing.assert_array_equal(normalized["token_types"][2:],0)
        self.assertEqual(scaler.training_sample_ids,("train_a","train_b"))
        restored=ConditionNormalizer.from_dict(scaler.to_dict())
        np.testing.assert_array_equal(restored.global_scale,scaler.global_scale)
        bad=scaler.to_dict();bad["fit_validation_or_test"]=True
        with self.assertRaisesRegex(ValueError,"TRAIN-only"):
            ConditionNormalizer.from_dict(bad)

    def test_std_floor_and_masked_nonfinite_are_not_fit(self):
        item=raw_condition(2.)
        item["token_features"][~item["token_mask"]]=np.nan
        scaler=ConditionNormalizer.fit([item],[0],["train"])
        np.testing.assert_array_equal(scaler.global_scale,np.full(GLOBAL_DIM,.01))
        np.testing.assert_array_equal(scaler.token_scale,np.full(TOKEN_DIM,.01))
        normalized=scaler.normalize(item)
        self.assertTrue(np.all(np.isfinite(normalized["token_features"])))

    def test_m0_flatten_uses_same_normalized_information_and_exact_boundary(self):
        task=fixture_task().to_dict()
        task["initial_planner_q"]=np.linspace(-.1,.1,17).tolist()
        task["initial_planner_dq"]=np.linspace(-.02,.02,17).tolist()
        item=raw_condition(.5)
        scaler=ConditionNormalizer.fit([item],[0],["train"])
        with patch("v6_4.architecture_pilot_training.encode_typed_condition",return_value=item):
            conditions=make_conditions([task],scaler,q_ranges=np.full(17,2.))
        np.testing.assert_allclose(conditions["fixed_controls"][0,0].numpy(),task["initial_planner_q"],atol=4e-9,rtol=0)
        expected=np.asarray(task["initial_planner_q"])+(27./29./3.)*np.asarray(task["initial_planner_dq"])
        np.testing.assert_allclose(conditions["fixed_controls"][0,1].numpy(),expected,atol=4e-9,rtol=0)
        np.testing.assert_array_equal(conditions["fixed_controls_scaled"].numpy(),conditions["fixed_controls"].numpy()/2.)
        self.assertEqual(conditions["token_types"].dtype,torch.long)
        self.assertEqual(conditions["token_mask"].dtype,torch.bool)
        np.testing.assert_array_equal(conditions["flat"][0,:GLOBAL_DIM].numpy(),conditions["global"][0].numpy())

    def test_paired_cpu_draws_ignore_model_initialization_rng(self):
        left=torch.Generator(device="cpu").manual_seed(77)
        right=torch.Generator(device="cpu").manual_seed(77)
        for step in range(4):
            a=paired_draw(left,[2,4,6],32)
            torch.manual_seed(100+step);torch.randn(1000)
            b=paired_draw(right,[2,4,6],32)
            for x,y in zip(a,b):torch.testing.assert_close(x,y,atol=0,rtol=0)
            self.assertTrue(set(a[0].tolist()).issubset({2,4,6}))

    def test_formal_initialization_does_not_depend_on_previous_smoke_weights(self):
        bundle={"config":{"training_seed":64101,"decoded_loss":{"sample_count":136}},
            "control_normalizer":ControlNormalizer(np.zeros(17),np.ones(17),("train",)),
            "q_ranges":np.ones(17),"dq_scales":np.ones(17)}
        first,_=_make_model(bundle,"M0","cpu");initial=_state_sha(first)
        with torch.no_grad():
            for parameter in first.parameters():parameter.add_(1.)
        fresh,_=_make_model(bundle,"M0","cpu")
        self.assertEqual(_state_sha(fresh),initial)
        self.assertNotEqual(_state_sha(first),initial)


if __name__=="__main__":unittest.main()
