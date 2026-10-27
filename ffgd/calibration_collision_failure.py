"""Read-only spatial/temporal diagnosis of an independently rejected candidate."""
import json
from pathlib import Path
import numpy as np
import mujoco
import torch
from .contracts import Candidate
from .problem_io import load_problem
from .api import boundary_codec
from .physics import DifferentiableTree
from .propagation import Propagator
from .geometry import Geometry
from .convex_guide import ConvexSupportGuide
from .trajectory import knots
from .orchestration import check_budget,atomic_json


def run(output,config):
    torch.set_num_threads(1);root=Path(config['planning_run']);folder=root/'candidate_000'
    manifest=json.loads((root/'planning_manifest.json').read_text())
    record=json.loads((folder/'candidate.json').read_text())
    with np.load(folder/'candidate.npz') as data:
        candidate=Candidate(data['control'],float(data['duration']),int(data['seed']))
    if candidate.identity!=record['identity']:raise ValueError('CANDIDATE_IDENTITY_MISMATCH')
    r,snapshot,task,_=load_problem(root/'problem.json');geometry=Geometry(r);guide=ConvexSupportGuide(geometry)
    prop=Propagator(DifferentiableTree(r),snapshot);codec=boundary_codec(r,snapshot,task.duration,len(candidate.control),origin=manifest['budget'].get('boundary_origin','pivot_zero'))
    z=codec.encode(candidate.control);rows=[];arrays={}
    for intervals,label in ((manifest['budget']['intervals'],'guide'),(1024,'dense')):
        check_budget();grid=np.unique(np.r_[np.linspace(0,task.duration,intervals+1),knots(codec.n)*task.duration,[p.time for p in task.poses]])
        xs,_=prop.native_rollout(codec,z,intervals,grid=grid);data=mujoco.MjData(r.model);minima=[];pairs=[];proxies=[]
        for i,x in enumerate(xs):
            check_budget();data.qpos[:]=x;scan=geometry.native_scan(data)
            minima.append(scan['minimum']);pairs.append(scan['pair'])
            if label=='guide':
                gaps=guide.numpy_gaps(data.geom_xpos[geometry.ids],data.geom_xmat[geometry.ids].reshape(-1,3,3))
                proxies.append(float(gaps.min()))
        worst=int(np.argmin(minima));data.qpos[:]=xs[worst];mujoco.mj_kinematics(r.model,data)
        gaps=guide.numpy_gaps(data.geom_xpos[geometry.ids],data.geom_xmat[geometry.ids].reshape(-1,3,3))
        pair=pairs[worst];pair_index=int(np.flatnonzero(np.all(geometry.ids[geometry.pairs]==pair,axis=1))[0])
        row=dict(grid=label,points=len(grid),minimum_native=float(minima[worst]),worst_time=float(grid[worst]),
                 worst_native_pair=pair,worst_pair_names=[r.model.geom(g).name for g in pair],
                 same_pose_proxy_for_worst_pair=float(gaps[pair_index]),same_pose_minimum_proxy=float(gaps.min()),
                 negative_native_samples=int(np.sum(np.asarray(minima)<0)))
        if label=='guide':
            row['minimum_proxy_over_grid']=min(proxies);row['negative_proxy_samples']=int(np.sum(np.asarray(proxies)<0))
            guide_grid,guide_states=grid,xs
        else:
            neighbors=np.unique(np.clip([np.searchsorted(guide_grid,grid[worst])-1,np.searchsorted(guide_grid,grid[worst])],0,len(guide_grid)-1))
            row['bracketing_guide_samples']=[]
            for index in neighbors:
                data.qpos[:]=guide_states[index];mujoco.mj_kinematics(r.model,data)
                value=mujoco.mj_geomDistance(r.model,data,pair[0],pair[1],10.,None)
                row['bracketing_guide_samples'].append(dict(time=float(guide_grid[index]),same_pair_native=float(value)))
        rows.append(row);arrays[label+'_time']=grid;arrays[label+'_native_minimum']=np.asarray(minima)
        atomic_json(output/'geometry_comparison.json',rows)
    np.savez_compressed(output/'distance_profiles.npz',**arrays)
    old=float(record['validation']['minimum_clearance']);error=abs(rows[-1]['minimum_native']-old)
    return dict(passed=error<1e-10,source_run=str(root),candidate_identity=candidate.identity,
                scope='immutable candidate spatial versus temporal collision diagnosis; no repaired plan or execution',
                original_native_clearance=old,reproduction_error=error,rows=rows,main_candidate_count_added=0)
