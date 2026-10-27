"""Freeze local public-task perturbations, never generate solved trajectories.

The original eight DEV tasks remain byte-identical. New initial states receive
native geometry/bounds checks only; no candidate planning or native integration.
Run from the repository root. Existing frozen manifests are never overwritten.
"""
import copy
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from scipy.spatial.transform import Rotation
from ffgd.problem_io import load_problem
from ffgd.geometry import Geometry


def main():
    root=Path(__file__).resolve().parents[3];out=Path(__file__).resolve().parent
    if (out/'manifest.json').exists():raise RuntimeError('DEV40_ALREADY_FROZEN')
    source=root/'configs/ffgd/dev_v1/manifest.json'
    original=json.loads(source.read_text(encoding='utf-8'));records=[];started=time.perf_counter()
    rng=np.random.default_rng(20261027)
    extra_families=set(np.random.default_rng(20261028).choice(8,4,replace=False).tolist())
    for family,old in enumerate(original['tasks']):
        base=root/old['path']
        if hashlib.sha256(base.read_bytes()).hexdigest()!=old['sha256']:raise RuntimeError('ORIGINAL_DEV_CHANGED')
        public=json.loads(base.read_text(encoding='utf-8'))
        for replicate in range(5):
            if time.perf_counter()-started>240:raise RuntimeError('DEV40_FREEZE_TIME_BUDGET')
            if replicate==0:path=base
            else:
                value=copy.deepcopy(public);value['id']=old['id']+f'__dev40_{replicate:02d}'
                value['description']='DEV public local perturbation; no trajectory solution generated or supplied.'
                value['initial']['q17']=(np.asarray(value['initial']['q17'])+rng.uniform(-.01,.01,17)).tolist()
                duration=value['task']['duration']
                for pose in value['task']['poses']:
                    pose['position']=(np.asarray(pose['position'])+rng.uniform(-.008,.008,3)).tolist()
                    pose['rotation']=(Rotation.from_rotvec(rng.uniform(-.008,.008,3)).as_matrix()@np.asarray(pose['rotation'])).tolist()
                    if pose['time']!=duration:
                        shifted=pose['time']+rng.uniform(-min(.02*duration,.1),min(.02*duration,.1))
                        pose['time']=round(shifted/.002)*.002
                        if not 0<pose['time']<duration:raise RuntimeError('NEW_TASK_TIME_OUTSIDE_DOMAIN')
                for box in value.get('scene',{}).get('boxes',[]):
                    box['center']=(np.asarray(box['center'])+rng.uniform(-.002,.002,3)).tolist()
                path=out/(value['id']+'.json');content=(json.dumps(value,indent=2)+'\n').encode('utf-8')
                if path.exists():
                    if path.read_bytes()!=content:raise RuntimeError('EXISTING_DRAFT_DIFFERS')
                else:
                    with path.open('xb') as stream:stream.write(content)
            robot,snapshot,task,_=load_problem(path);d=robot.restore(snapshot)
            q=robot.Bplus@d.qpos[robot.qpos_ids];limited=robot.model.jnt_limited[robot.joints].astype(bool)
            bounds=robot.model.jnt_range[robot.joints][limited];low=d.qpos[robot.qpos_ids][limited]
            clear=Geometry(robot).native_scan(d)
            valid=bool(np.all(q>=robot.lower) and np.all(q<=robot.upper) and np.all(low>=bounds[:,0]) and np.all(low<=bounds[:,1]) and clear['minimum']>=0)
            records.append(dict(id=json.loads(path.read_text(encoding='utf-8'))['id'],family=old['id'],replicate=replicate,
                path=str(path.relative_to(root)),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                initial_precheck_passed=valid,initial_geometry=clear,
                execution_subset=replicate<2 or (replicate==2 and family in extra_families)))
            print(json.dumps(dict(completed=len(records),initial_precheck_passed=valid)),flush=True)
    manifest=dict(schema='ffgd_frozen_dev40_v1',status='FROZEN' if all(t['initial_precheck_passed'] for t in records) else 'FAILED_INITIAL_PRECHECK',
        task_count=len(records),tasks=records,task_generation_seed=20261027,execution_subset_seed=20261028,
        execution_subset_count=sum(t['execution_subset'] for t in records),planning_seeds=[13001,13002,13003,13004,13005],
        source_manifest_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        profile_path='configs/ffgd/numerical_profile_v1.json',
        profile_sha256=hashlib.sha256((root/'configs/ffgd/numerical_profile_v1.json').read_bytes()).hexdigest(),
        generator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),native_integration_steps=0,
        planning_status='NOT_RUN',execution_status='NOT_RUN',
        scope='DEV local perturbations of eight existing public tasks, which include already observed development cases. Not independent TEST. No hidden solved path.',
        failures_retained=[t['id'] for t in records if not t['initial_precheck_passed']],
        initial_checks_only=True)
    assert len(records)==40 and manifest['execution_subset_count']==20
    with (out/'manifest.json').open('x',encoding='utf-8') as stream:json.dump(manifest,stream,indent=2)
    if manifest['status']!='FROZEN':raise RuntimeError('INITIAL_PRECHECK_FAILED_NO_TASK_REPLACEMENT')


if __name__=='__main__':main()
