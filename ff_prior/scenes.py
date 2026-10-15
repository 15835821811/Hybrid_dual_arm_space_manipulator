"""One frozen static sphere/OBB scene contract for prediction and execution."""
from dataclasses import dataclass,asdict
import numpy as np
from .storage import digest


@dataclass(frozen=True)
class Obstacle:
    object_id:str
    shape:str
    position:tuple
    size:tuple
    quaternion_wxyz:tuple=(1.,0.,0.,0.)

    def __post_init__(self):
        if not self.object_id or self.shape not in ("sphere","box"):
            raise ValueError("named sphere or OBB required")
        p,s,q=np.asarray(self.position),np.asarray(self.size),np.asarray(self.quaternion_wxyz)
        if p.shape!=(3,) or s.shape!=(3,) or q.shape!=(4,) or not all(np.isfinite(v).all() for v in (p,s,q)):
            raise ValueError("invalid static geometry dimensions")
        if abs(np.linalg.norm(q)-1)>1e-10 or (s[0]<=0 if self.shape=="sphere" else np.any(s<=0)):
            raise ValueError("invalid static geometry size/quaternion")
        if self.shape=="sphere" and np.any(s[1:]!=0):
            raise ValueError("sphere size must be radius,0,0")


@dataclass(frozen=True)
class SceneSpec:
    objects:tuple=()
    family_id:str=""

    def __post_init__(self):
        if self.objects and not self.family_id:
            raise ValueError("nonempty scenes require an isolated family id")
        if len({o.object_id for o in self.objects})!=len(self.objects):
            raise ValueError("duplicate obstacle identity")

    def to_dict(self):
        return {"schema":"ff_prior_static_scene_1","objects":[asdict(o) for o in self.objects],"family_id":self.family_id,
                "motion":"STATIC","legacy_target":"included_by_robot_model_and_complete_initial_state"}

    @property
    def scene_hash(self): return digest(self.to_dict())

    @classmethod
    def from_dict(cls,value):
        if value.get("motion","STATIC")!="STATIC": raise ValueError("dynamic obstacles NOT_VALIDATED")
        objects=tuple(Obstacle(o["object_id"],o["shape"],tuple(o["position"]),tuple(o["size"]),tuple(o.get("quaternion_wxyz",(1,0,0,0)))) for o in value.get("objects",[]))
        return cls(objects,value.get("family_id",""))
