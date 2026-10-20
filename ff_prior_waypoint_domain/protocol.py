from pathlib import Path
from ff_prior.storage import file_digest,read_json,atomic_json
ROOT=Path(__file__).resolve().parents[1]
OLD=ROOT/"artifacts/v7_1/waypoint_scene_001"


def verify_old():
    for name,sha in read_json(OLD/"binding.json").items():
        if file_digest(ROOT/name)!=sha: raise RuntimeError(f"Original dependency changed: {name}")
    return {str(p.relative_to(OLD)):file_digest(p) for p in sorted(OLD.rglob("*")) if p.is_file()}


def freeze(out,hashes):
    atomic_json(out/"frozen_protocol.json",dict(version="waypoint_domain_v2",
        source_run=str(OLD),old_files=hashes,
        source={str(p.relative_to(ROOT)):file_digest(p) for p in
            sorted((ROOT/"ff_prior_waypoint_domain").glob("*.py"))},
        cases=["stationary","drifting"],seeds=[20261020,20261021],
        methods=["task_controller","diffusion_null_prior"],duration_s=27.,
        executions=4,replays=4,physics_steps_each=13500,ddim_calls_per_prior=16,
        wall_budget_s=7200,threads=1,changes=["V7 measured-state velocity bounds",
            "all ten ramp-integrated reference position bounds","native jnt_limited mask",
            "fail closed on actual/reference domain violations"],
        unchanged=["model weights","prior proposals","initial states","references","collision policy",
            "torque controller","velocity/acceleration limits","5mm/1degree","nonpenetration"],
        training=False,expansion=False,TEST=False))
