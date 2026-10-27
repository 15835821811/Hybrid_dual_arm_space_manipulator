"""Independent CLI consumers of saved candidate and native execution inputs."""
import json
from pathlib import Path
import numpy as np
from .problem_io import load_problem
from .contracts import Candidate,Snapshot
from .api import boundary_codec
from .physics import DifferentiableTree
from .propagation import Propagator
from .geometry import Geometry
from .validation import validate
from .execution import replay
from .orchestration import atomic_json,check_budget


def validate_saved(output,planning_directory,candidate_index=None):
    root=Path(planning_directory);r,snapshot,task,problem=load_problem(root/'problem.json')
    manifest=json.loads((root/'planning_manifest.json').read_text(encoding='utf-8'));profile=manifest['profile']
    prop=Propagator(DifferentiableTree(r),snapshot);geometry=Geometry(r);rows=[]
    folders=[root/f'candidate_{int(candidate_index):03d}'] if candidate_index is not None else sorted(root.glob('candidate_*'))
    for folder in folders:
        check_budget()
        if not (folder/'candidate.npz').exists():
            rows.append(dict(candidate=folder.name,status='NOT_RUN_NO_COMPLETE_CANDIDATE'));continue
        with np.load(folder/'candidate.npz') as data:c=Candidate(data['control'],float(data['duration']),int(data['seed']))
        record=json.loads((folder/'candidate.json').read_text(encoding='utf-8'))
        if c.identity!=record['identity']:raise ValueError('CANDIDATE_IDENTITY_MISMATCH')
        codec=boundary_codec(r,snapshot,task.duration,len(c.control),origin=manifest['budget'].get('boundary_origin','pivot_zero'));report=validate(c,snapshot,task,prop,codec,geometry,profile)
        atomic_json(output/(folder.name+'.json'),report);rows.append(dict(candidate=folder.name,**report))
    return dict(passed=bool(rows) and all(x['status']=='PLAN_VALID' for x in rows),rows=rows,source=str(root),read_only=True,source_reports_modified=False)


def replay_saved(output,execution_directory):
    root=Path(execution_directory);provenance=json.loads((root/'provenance.json').read_text(encoding='utf-8'))
    r,snapshot,task,problem=load_problem(Path(provenance['planning_directory'])/'problem.json')
    from .execution import refine_physics
    r,snapshot=refine_physics(r,snapshot,provenance.get('physics_substeps',1))
    if r.model_hash!=provenance.get('execution_model_hash',r.model_hash):raise ValueError('EXECUTION_MODEL_IDENTITY_MISMATCH')
    filename='execution.npz' if (root/'execution.npz').exists() else 'execution_partial.npz'
    with np.load(root/filename) as data:bundle={k:data[k] for k in data.files}
    if len(bundle['states'])!=len(bundle['ctrl'])+1:raise ValueError('INCOMPLETE_INTEGRATION_INPUT_LEDGER')
    np.testing.assert_array_equal(snapshot.state,bundle['initial_state'])
    report=replay(r,bundle,snapshot)
    from .execution_acceptance import inspect_state_bounds
    bounds=inspect_state_bounds(r,bundle,provenance['profile'])
    return dict(passed=report['max_state_abs']<=provenance['profile']['replay_state_atol'],source=str(root),replay=report,
                actual_state_bounds=bounds,actual_state_bounds_passed=all(bounds['gates'].values()),
                model_hash=r.model_hash,source_reports_modified=False,source_file=filename,
                scope='replay of recorded inputs only; does not establish execution completion or task success')


def validate_execution_states(output,execution_directory,tracking_profile=None):
    """Inspect immutable execution records without integrating or editing them."""
    from .execution_acceptance import inspect_state_bounds
    root=Path(execution_directory);provenance=json.loads((root/'provenance.json').read_text(encoding='utf-8'))
    r,snapshot,task,problem=load_problem(Path(provenance['planning_directory'])/'problem.json')
    from .execution import refine_physics
    r,snapshot=refine_physics(r,snapshot,provenance.get('physics_substeps',1))
    if r.model_hash!=provenance.get('execution_model_hash',r.model_hash):raise ValueError('EXECUTION_MODEL_IDENTITY_MISMATCH')
    with np.load(root/'execution.npz') as data:bundle={k:data[k] for k in data.files}
    np.testing.assert_array_equal(snapshot.state,bundle['initial_state'])
    profile=dict(provenance['profile'])
    if tracking_profile is not None:
        thresholds=tracking_profile['thresholds']
        if set(thresholds)!={'tracking_error_rad','manifold_residual_rad'}:raise ValueError('INVALID_TRACKING_PROFILE_FIELDS')
        if any(not np.isfinite(v) or not 0<v<=.02 for v in thresholds.values()):raise ValueError('TRACKING_PROFILE_EXCEEDS_INHERITED_GATES')
        profile.update(thresholds)
    report=inspect_state_bounds(r,bundle,profile)
    return dict(passed=all(report['gates'].values()),source=str(root),state_bounds=report,
                source_reports_modified=False,native_integration_steps=0,tracking_profile_override=tracking_profile,
                scope='saved actual position/velocity/applied-input acceleration bounds only; not a new execution or full M5 acceptance')
