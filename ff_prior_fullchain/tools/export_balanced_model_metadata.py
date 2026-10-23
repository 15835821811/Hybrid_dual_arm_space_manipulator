"""Read-only metadata for independent algebraic checks; no state evaluation."""
import sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from ff_prior.storage import read_json,file_digest
from ff_prior_pilot_train.storage import atomic_json
from ff_prior.robot_adapter import RobotAdapter
from ff_prior.scenes import SceneSpec

start=time.time()
out=ROOT/'ff_prior_balanced_delivery/model_metadata.json'
assert not out.exists()
run=ROOT/'artifacts/v7_1/balanced_restore_001'
identity=read_json(run/'run.json')['identity']
for path,expected in identity['source_hashes'].items():
    assert file_digest(ROOT/path)==expected,path
scene=read_json(ROOT/'ff_prior_balanced_restore/evidence/selection.json')['entries'][0]['scene']
robot=RobotAdapter(scene=SceneSpec.from_dict(scene));m=robot.model
assert robot.identity()['scene_model_hash']==read_json(run/'units/u00/execution/report.json')['scene_model_hash']
assert all(v==0 for v in robot.calls.values())
atomic_json(out,dict(scene_model_hash=robot.identity()['scene_model_hash'],
    geom_rbound=m.geom_rbound.copy(),geom_bodyid=m.geom_bodyid.copy(),
    jnt_range=m.jnt_range.copy(),jnt_limited=m.jnt_limited.copy(),
    jnt_type=m.jnt_type.copy(),jnt_qposadr=m.jnt_qposadr.copy(),jnt_dofadr=m.jnt_dofadr.copy(),
    index_map=robot.index_map,B=robot.B,Bplus=robot.Bplus,
    planner_lower=robot.lower,planner_upper=robot.upper,qpos_ids=robot.qpos_ids,dof_ids=robot.dof_ids,
    new_native_steps=0,new_coupled_calls=0,new_geometry_queries=0,calls=robot.calls,
    source_hashes_checked=len(identity['source_hashes']),wall_s=time.time()-start,
    provenance='post-run compiled immutable model metadata, not a new state/trajectory evaluation'))
print(out)
