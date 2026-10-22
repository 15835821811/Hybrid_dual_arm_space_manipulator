import ast
from dataclasses import asdict
import numpy as np
from ff_prior.storage import read_json,file_digest,digest
from ff_prior_pilot_train.storage import atomic_json,atomic_npz
from .protocol import ROOT,PACKAGE,OUT,SOURCE,BASE,DYNAMIC,P
def sources():
    pending = list(PACKAGE.glob('*.py')); found = set()
    def add(module):
        parts=module.split('.')
        for i in range(1,len(parts)+1):
            p=ROOT.joinpath(*parts[:i])
            for path in (p.with_suffix('.py'),p/'__init__.py'):
                if path.is_file() and path not in found: pending.append(path)
    while pending:
        p=pending.pop()
        if p in found: continue
        found.add(p);pkg=p.relative_to(ROOT).with_suffix('').parts[:-1]
        for node in ast.walk(ast.parse(p.read_text(encoding='utf-8-sig'))):
            if isinstance(node,ast.Import):
                for name in node.names: add(name.name)
            elif isinstance(node,ast.ImportFrom):
                prefix=list(pkg[:len(pkg)-node.level+1]) if node.level else []
                module='.'.join(prefix+([node.module] if node.module else []));add(module)
                for name in node.names:
                    if name.name!='*':add(module+'.'+name.name)
    found.add(PACKAGE/'job.ps1')
    found.add(PACKAGE/'docs/EXPERIMENT_PLAN.md')
    return {p.relative_to(ROOT).as_posix():file_digest(p) for p in sorted(found)}

def freeze():
    assert not (PACKAGE/'evidence/selection.json').exists(),'Freeze only once'
    from ff_prior.pipeline import audit,acceptance
    from ff_prior_origin_scene.runtime import at_scene_origin
    from ff_prior.contracts import frozen_task,assert_task_compatible
    from v6_lite.irregular_waypoints import IrregularWaypointTarget
    from .path import BypassPath
    old=read_json(ROOT/'ff_prior_diffusion_demo/evidence/selection.json')
    original=old['entries'][0];w=dict(read_json(DYNAMIC)['waypoints'])
    points=np.asarray(w['waypoint_points_m']);centers=(points[:-1]+points[1:])/2
    objects=list(original['scene']['objects'])
    for i,c in enumerate(centers):
        objects.append(dict(object_id=f'continuum_segment_{i}',position=c.tolist(),shape='sphere',
            size=[P['obstacle_radius_m'],0.,0.],quaternion_wxyz=[1.,0.,0.,0.]))
    scene=dict(original['scene'],objects=objects,family_id='capture3_continuum6_fixed_waypoints')
    config=read_json(BASE/'run.json')['config'];physical,canonical,initial,_=audit(config)
    profile=acceptance(config);condition=original['task']['condition']
    rr,ii,cc,proof=at_scene_origin(canonical,initial,scene,condition['initial_features'])
    task=frozen_task(condition,dict(initial_state_id=ii.state_id,robot_hash=ii.model_hash,
        pair_policy_hash=cc.policy_hash,acceptance=asdict(profile)))
    assert_task_compatible(task,rr,ii,cc,profile)
    entry=dict(sample_id='capture3_continuum6',scene=scene,task=task,split=P['task_split'],initial_status='PASS')
    unit=dict(old['units'][0],task_id=entry['sample_id'],task_hash=task['task_hash'])
    w['initial_position_w']=np.asarray(condition['goals'])[0,:3,3].tolist()
    path=IrregularWaypointTarget(np.asarray(w['initial_position_w']),points,np.asarray(w['target_rotation_world']),
        w['transition_duration_s'],w['path_duration_s'],np.asarray(w['segment_durations_s']),w)
    bypass=BypassPath(path,P['bypass_offset_m'])
    t=np.linspace(0,27,2701);reference=np.array([bypass.sample(x)[0] for x in t])
    distance=np.linalg.norm(reference[:,None,:]-centers[None,:,:],axis=-1)-P['obstacle_radius_m']
    boundaries=bypass.boundaries
    actual_points=np.array([bypass.sample(x)[0] for x in boundaries])
    assert np.allclose(actual_points,points,atol=1e-14,rtol=0)
    assert distance.min()>0,'Point reference itself intersects new obstacles'
    mids=(boundaries[:-1]+boundaries[1:])/2
    old_mid=np.array([path.sample(x)[0] for x in mids])
    assert np.all(np.linalg.norm(old_mid-centers,axis=1)<1e-12)
    contract=dict(schema='dual_obstacle_static_dynamic_v1',scene=scene,scene_model_hash=ii.model_hash,
        static_goals=condition['goals'],waypoints=w,bypass_offset_m=P['bypass_offset_m'],
        original_waypoints_sha256=digest(points),split=P['task_split'],moving_satellite_claim=False,
        static_task_hash=task['task_hash'],target_velocity_changed=False)
    contract['contract_hash']=digest(contract)
    preflight=dict(initial_geometry=proof['initial_geometry'],initial_transfer=proof,
        obstacle_centers=centers,obstacle_radius_m=P['obstacle_radius_m'],
        original_reference_center_distance=np.linalg.norm(old_mid-centers,axis=1),
        bypass_reference_point_clearance_m=float(distance.min()),
        minimum_waypoint_to_new_sphere_m=float((np.linalg.norm(points[:,None]-centers[None],axis=-1)-P['obstacle_radius_m']).min()),
        all_seven_waypoints_preserved=True,all_new_obstacles_used_by_collision=True,
        scope='initial whole-body and reference-point design only; not full-path qualification',new_native_steps=0)
    selection=dict(entries=[entry],units=[unit],candidate_denominator=5,execution_denominator=2,relative_advantage_benchmark=False)
    atomic_json(PACKAGE/'evidence/preflight.json',preflight)
    atomic_npz(PACKAGE/'evidence/reference.npz',time=t,position=reference,old_midpoints=old_mid,centers=centers)
    atomic_json(PACKAGE/'evidence/selection.json',selection)
    atomic_json(PACKAGE/'evidence/contract.json',contract)
    files=dict(read_json(ROOT/'ff_prior_diffusion_demo/evidence/binding.json')['files'])
    for f in (DYNAMIC,ROOT/'ff_prior_diffusion_demo/evidence/selection.json',PACKAGE/'evidence/selection.json',
              PACKAGE/'evidence/contract.json',PACKAGE/'evidence/preflight.json',PACKAGE/'evidence/reference.npz'):
        files[f.relative_to(ROOT).as_posix()]=file_digest(f)
    atomic_json(PACKAGE/'evidence/binding.json',dict(files=files))
    return dict(obstacles=len(objects),initial_clearance=proof['initial_geometry']['minimum_clearance_m'],
        reference_clearance=preflight['bypass_reference_point_clearance_m'],native_steps=0)

def identity():
    from ff_prior.contracts import assert_frozen
    binding=read_json(PACKAGE/'evidence/binding.json')
    for p,h in binding['files'].items():
        assert file_digest(ROOT/p)==h,'FROZEN_INPUT_CHANGED: '+p
    c=read_json(PACKAGE/'evidence/contract.json');raw=dict(c);given=raw.pop('contract_hash');assert digest(raw)==given
    s=read_json(PACKAGE/'evidence/selection.json')
    for e in s['entries']:assert_frozen(e['task'])
    return dict(protocol=P,source_hashes=sources(),binding=binding,contract=c,selection=s)

if __name__=='__main__':
    import sys,json
    print(json.dumps(freeze() if sys.argv[-1]=='freeze' else identity(),ensure_ascii=True))
