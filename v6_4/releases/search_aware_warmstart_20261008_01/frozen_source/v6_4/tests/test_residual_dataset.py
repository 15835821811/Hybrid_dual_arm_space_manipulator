"""Synthetic pure-file fixtures; these tests make no physical success claim."""
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import mujoco
import numpy as np

from v6_4.dataset import object_sha, sha256
from v6_4.task_protocol import canonical_json
from v6_4.tests.test_learning_dataset import fixture_task
from v6_4.task_anchored_reference import build_reference_definition, TaskAnchoredResidualPlan
from v6_4.residual_dataset import (ConditionNormalizer, ResidualNormalizer, encode_residual_condition,
    freeze_residual_dataset, load_residual_dataset, retrieve_train_residual, compatibility_key)


def residual_fixture(split="train", task_id="train", shift=0.):
    t = fixture_task(split,task_id,task_id,shift)
    s = t.scenario
    s["continuum_target"]={"mode":"irregular_waypoints","reference_profile":"minimum_jerk_c2",
        "initial_position_w":[shift,.1,.2],"waypoint_points_m":[[shift+.1,.1,.2],[shift+.4,.2,.3]],
        "segment_durations_s":[18.],"transition_duration_s":3.,"path_duration_s":18.,
        "target_rotation_world":np.eye(3).tolist()}
    t = replace(t,scenario_json=canonical_json(s))
    return t,build_reference_definition(t)


def synthetic_bundle(parent, *, failed_val=False):
    """JSON protocol/evidence fixture, never simulated teacher data."""
    pairs=[residual_fixture(),residual_fixture("val","val",.2),residual_fixture("test","test",.4)]
    records=[]
    for t,d in pairs[:2]:
        z=np.zeros((6,2)); z[np.asarray(d["interval_mask"]),0]=.01
        plan=TaskAnchoredResidualPlan.from_definition(d,z)
        good=not(failed_val and t.split=="val")
        attempt={"task_id":t.task_id,"task_sha256":t.sha256(),"plan_content_sha256":plan.sha256(),
                 "actual_steps":13500 if good else 30,"full_task_success":good,"full_27s_success":good,"fixture_only":True}
        evaluation={"task_id":t.task_id,"task_sha256":t.sha256(),"task_success":good,
            "full_task_success":good,"complete":good,"evidence_valid":True,"fixture_only":True,
            **{k:{"passed":good} for k in ("reference_binding","execution_contract","independent_interval","native_geometry","task_requirements")}}
        evaluation["reference_binding"]["nonzero_reference_consumed"]=good
        evaluation["metrics"]={"physics_steps":13500 if good else 30,"actual_saved_horizon_s":27. if good else .06}
        record={"task_id":t.task_id,"split":t.split,"candidate_index":0,"full_task_success":good,
                "nonzero_reference":True,"consumed_reference_binding_passed":good}
        for name,value in (("plan",plan.to_dict()),("attempt",attempt),("evaluation",evaluation)):
            path=parent/f"{t.task_id}_{name}.json"
            path.write_text(json.dumps(value),encoding="utf-8")
            record[name+"_path"]=str(path); record[name+"_sha256"]=sha256(path)
        records.append(record)
    return {"records":records},{"tasks":[t.to_dict() for t,d in pairs]}, {t.task_id:d for t,d in pairs}


class ResidualDatasetTests(unittest.TestCase):
    def test_normalizer_pools_active_train_axes_and_roundtrips(self):
        mask=np.array([True,False,True,False,False,False])
        z=np.zeros((2,6,2)); z[0,mask]=[.01,0.]; z[1,mask]=[-.01,.002]
        n=ResidualNormalizer.fit(z,np.stack([mask,mask]))
        np.testing.assert_allclose(n.mean_m,[0,.001],atol=1e-18)
        np.testing.assert_allclose(n.std_m,[.01,.001],atol=1e-18)
        for value in z:
            scaled=n.normalize(value,mask)
            np.testing.assert_array_equal(scaled[~mask],0.)
            np.testing.assert_allclose(n.inverse(scaled,mask),value,atol=1e-18)
        self.assertEqual(n.to_dict(),ResidualNormalizer.from_dict(n.to_dict()).to_dict())
        bad=z[0].copy(); bad[1,0]=1e-30
        with self.assertRaisesRegex(ValueError,"exactly zero"): n.normalize(bad,mask)
        bad[1,0]=np.nan
        with self.assertRaises(ValueError): n.normalize(bad,mask)

    def test_condition_is_declared_only_train_unique_and_padding_zero(self):
        t,d=residual_fixture(); v,vd=residual_fixture("val","v",100.)
        with patch.object(mujoco,"MjData",side_effect=AssertionError("no model/data")), patch.object(mujoco,"mj_step",side_effect=AssertionError("no physics")):
            n=ConditionNormalizer.fit([(t,d)])
            original=n.transform(t,d)
            encoded=encode_residual_condition(t,d)
            self.assertLess(len(original),1000)
            np.testing.assert_array_equal(original[~encoded["valid"]],0.)
            np.testing.assert_array_equal(original[encoded["literal"]],encoded["values"][encoded["literal"]])
            self.assertEqual(n.fit_task_ids,["train"])
            self.assertGreater(float(np.max(np.abs(n.transform(v,vd)))),10.)
            changed=deepcopy(t.to_dict()); changed["task_id"]="renamed"; changed["group_id"]="new_group"
            from v6_4.task_protocol import TaskSpec
            renamed=TaskSpec.from_dict(changed); renamed_def=build_reference_definition(renamed)
            np.testing.assert_array_equal(n.transform(renamed,renamed_def),original)
        with self.assertRaisesRegex(ValueError,"unique"): ConditionNormalizer.fit([(t,d),(t,d)])
        with self.assertRaisesRegex(ValueError,"only fit TRAIN"): ConditionNormalizer.fit([(v,vd)])
        self.assertEqual(n.to_dict(),ConditionNormalizer.from_dict(n.to_dict()).to_dict())

    def test_freeze_requires_each_success_gate_preserves_failure_and_hashes(self):
        with tempfile.TemporaryDirectory() as tmp:
            parent=Path(tmp)
            teacher,suite,defs=synthetic_bundle(parent,failed_val=True)
            m=freeze_residual_dataset(teacher,suite,defs,parent/"dataset")
            self.assertEqual(m["training_status"],"TRAINING_NOT_RUN")
            self.assertEqual(m["data_status"],"DATA_LIMITED")
            self.assertEqual(len(m["samples"]),1); self.assertEqual(len(m["excluded"]),1)
            self.assertEqual(m["test_labels_used"],0)
            ds=load_residual_dataset(parent/"dataset/manifest.json")
            self.assertEqual(ds.z_m.shape,(1,6,2)); self.assertEqual(len(ds.indices("val")),0)
            with self.assertRaises(FileExistsError): freeze_residual_dataset(teacher,suite,defs,parent/"dataset")
            path=parent/"dataset"/m["samples"][0]["frozen_plan"]
            path.write_text("{}",encoding="utf-8")
            with self.assertRaisesRegex(ValueError,"changed"): load_residual_dataset(parent/"dataset/manifest.json")

    def test_false_independent_binding_excludes_label_even_if_success_booleans_true(self):
        with tempfile.TemporaryDirectory() as tmp:
            parent=Path(tmp); teacher,suite,defs=synthetic_bundle(parent)
            r=teacher["records"][0]; p=Path(r["evaluation_path"])
            e=json.loads(p.read_text()); e["reference_binding"]["passed"]=False
            p.write_text(json.dumps(e)); r["evaluation_sha256"]=sha256(p)
            m=freeze_residual_dataset(teacher,suite,defs,parent/"data")
            self.assertEqual(m["counts"]["train"]["references"],0)
            self.assertFalse(m["excluded"][0]["eligibility_gates"]["reference_binding"])

    def test_success_booleans_cannot_credit_prefix_and_not_run_is_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            parent=Path(tmp); teacher,suite,defs=synthetic_bundle(parent)
            r=teacher["records"][0]; p=Path(r["attempt_path"])
            a=json.loads(p.read_text()); a["actual_steps"]=13000
            p.write_text(json.dumps(a)); r["attempt_sha256"]=sha256(p)
            r=teacher["records"][1]; p=Path(r["evaluation_path"])
            p.write_text(json.dumps({"task_id":"val","status":"NOT_RUN","metrics":None,"reference_binding":None}))
            r["evaluation_sha256"]=sha256(p)
            m=freeze_residual_dataset(teacher,suite,defs,parent/"data")
            self.assertEqual(len(m["samples"]),0)
            self.assertEqual(len(m["excluded"]),2)
            self.assertEqual(m["training_status"],"TRAINING_NOT_RUN")
            self.assertFalse(m["excluded"][0]["eligibility_gates"]["full_13500_physics_steps"])

    def test_sha_mismatch_duplicate_test_and_group_leaks_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            parent=Path(tmp); teacher,suite,defs=synthetic_bundle(parent)
            bad=deepcopy(teacher); bad["records"][0]["plan_sha256"]="0"*64
            with self.assertRaisesRegex(ValueError,"SHA mismatch"): freeze_residual_dataset(bad,suite,defs,parent/"badsha")
            bad=deepcopy(teacher); bad["records"].append(bad["records"][0])
            with self.assertRaisesRegex(ValueError,"duplicated"): freeze_residual_dataset(bad,suite,defs,parent/"dup")
            bad=deepcopy(teacher); bad["records"][0]["task_id"]="test"; bad["records"][0]["split"]="test"
            with self.assertRaisesRegex(ValueError,"TEST leakage"): freeze_residual_dataset(bad,suite,defs,parent/"test")
            bad=deepcopy(suite); bad["tasks"][1]["group_id"]=bad["tasks"][0]["group_id"]
            with self.assertRaisesRegex(ValueError,"leaks"): freeze_residual_dataset(teacher,bad,defs,parent/"leak")

    def test_e1_matches_version_not_task_sha_and_never_modifies_coefficients(self):
        with tempfile.TemporaryDirectory() as tmp:
            parent=Path(tmp); teacher,suite,defs=synthetic_bundle(parent)
            freeze_residual_dataset(teacher,suite,defs,parent/"data")
            ds=load_residual_dataset(parent/"data/manifest.json")
            n=ConditionNormalizer.fit([(ds.tasks["train"],ds.definitions["train"])])
            query,definition=residual_fixture("test","query",.8)
            self.assertNotEqual(definition["base_reference_sha256"],ds.definitions["train"]["base_reference_sha256"])
            self.assertEqual(compatibility_key(definition),compatibility_key(ds.definitions["train"]))
            z,meta=retrieve_train_residual(ds,query,definition,n)
            np.testing.assert_array_equal(z,ds.z_m[0]); self.assertFalse(meta["z_modified"])
            definition=deepcopy(definition); definition["frame"]="target"
            with self.assertRaisesRegex(ValueError,"NO_COMPATIBLE"): retrieve_train_residual(ds,query,definition,n)


if __name__=="__main__": unittest.main()
