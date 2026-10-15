"""Frozen lineage/goal-neighbourhood splits and separate TRAIN-only scalers."""
from pathlib import Path
import numpy as np
from .contracts import public_condition, SCHEMA
from .coupled_predictor import pose_errors
from .storage import digest, atomic_json, read_json


def split_manifest(records):
    parents = {r["family_id"]:r["family_id"] for r in records}
    def find(k):
        while parents[k] != k:
            parents[k] = parents[parents[k]]
            k = parents[k]
        return k
    def union(a,b):
        a,b = sorted((find(a),find(b)))
        parents[b] = a
    # Pairwise audit is intentionally bounded to smoke/pilot; no scene ID is
    # treated as a lineage for empty scenes. Nonempty scenes join their roots.
    for i,a in enumerate(records):
        for b in records[i+1:]:
            p,r = pose_errors(a["condition"]["goals"],b["condition"]["goals"])
            path_close = "path_descriptor" in a and "path_descriptor" in b and np.max(np.abs(np.asarray(a["path_descriptor"])-b["path_descriptor"])) < .025
            if path_close or np.max(p) < .01 and np.max(r) < np.deg2rad(1) or (
                a.get("scene_family_id") and a["scene_family_id"] == b.get("scene_family_id")):
                union(a["family_id"],b["family_id"])
    members = {}
    for f in parents:
        members.setdefault(find(f),[]).append(f)
    groups = {f:digest(sorted(members[find(f)])) for f in parents}
    dev_groups = {groups[r["family_id"]] for r in records if r["source_partition"] == "dev"}
    result = {}
    for record in records:
        f = record["family_id"]
        bucket = int(groups[f][:8],16)%10
        split = "DEV" if groups[f] in dev_groups else "TRAIN" if bucket < 8 else "VAL" if bucket == 8 else "TEST"
        result[record["sample_id"]] = {"split":split,"family_group":groups[f],"family_id":f}
    return {"schema":"ff_prior_split_1","records":result,"goal_neighborhood_m":.01,
            "goal_neighborhood_rad":np.deg2rad(1),"frozen_before_training":True,
            "empty_scene_is_lineage":False,"test_used_for_development":False}


def export_dataset(index, split):
    rows = [r for r in index if r["validation_level"] == "MODEL_VALID" and r.get("execution_status") != "FAIL"
            and r["data_role"] == "FF_MODEL_PRIOR"]
    result = {s:[r["sample_id"] for r in rows if split["records"][r["sample_id"]]["split"] == s]
              for s in ("TRAIN","VAL","TEST","DEV")}
    return {"eligible":result,"split_hash":digest(split),"schema_hash":digest(SCHEMA),
            "status":"READY" if result["TRAIN"] and result["VAL"] else "BLOCKED_MISSING_TRAIN_OR_VAL",
            "execution_audit_is_training":False}


class Scaler:
    def __init__(self,z_mean,z_scale,c_mean,c_scale,metadata):
        self.z_mean,self.z_scale,self.c_mean,self.c_scale = (np.asarray(x) for x in (z_mean,z_scale,c_mean,c_scale))
        self.metadata = metadata

    @classmethod
    def fit(cls,z,conditions,splits,robot_hash,initial_state_id):
        if not splits or any(s != "TRAIN" for s in splits):
            raise ValueError("scaler fit is TRAIN-only")
        z,c = np.asarray(z),np.asarray([public_condition(v) for v in conditions])
        if z.shape != (len(splits),19,17) or len(c) != len(splits) or not np.isfinite(z).all():
            raise ValueError("scaler shape/data mismatch")
        zs,cs = z.std(axis=(0,1)),c.std(axis=0)
        return cls(z.mean(axis=(0,1)),np.where(zs < 1e-8,1,zs),c.mean(axis=0),np.where(cs < 1e-8,1,cs),
                   {"robot_hash":robot_hash,"initial_state_id":initial_state_id,"schema_hash":digest(SCHEMA),"fit_split":"TRAIN"})

    def z(self,z):
        return (z-self.z_mean)/self.z_scale

    def inverse_z(self,z):
        return z*self.z_scale+self.z_mean

    def condition(self,c):
        return (public_condition(c)-self.c_mean)/self.c_scale

    def save(self,path):
        value = {k:getattr(self,k) for k in ("z_mean","z_scale","c_mean","c_scale","metadata")}
        atomic_json(path,dict(value,hash=digest(value)))

    @classmethod
    def load(cls,path,robot_hash,initial_state_id):
        v = read_json(path)
        h = v.pop("hash")
        if digest(v) != h or v["metadata"] != {"robot_hash":robot_hash,"initial_state_id":initial_state_id,
                "schema_hash":digest(SCHEMA),"fit_split":"TRAIN"}:
            raise ValueError("scaler/robot/initial/schema identity mismatch")
        return cls(**v)
