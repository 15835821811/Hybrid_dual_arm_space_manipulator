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
        codec=boundary_codec(r,snapshot,task.duration,len(c.control));report=validate(c,snapshot,task,prop,codec,geometry,profile)
        atomic_json(output/(folder.name+'.json'),report);rows.append(dict(candidate=folder.name,**report))
    return dict(passed=bool(rows) and all(x['status']=='PLAN_VALID' for x in rows),rows=rows,source=str(root),read_only=True,source_reports_modified=False)


def replay_saved(output,execution_directory):
    root=Path(execution_directory);provenance=json.loads((root/'provenance.json').read_text(encoding='utf-8'))
    r,snapshot,task,problem=load_problem(Path(provenance['planning_directory'])/'problem.json')
    with np.load(root/'execution.npz') as data:bundle={k:data[k] for k in data.files}
    if len(bundle['states'])!=len(bundle['ctrl'])+1:raise ValueError('INCOMPLETE_INTEGRATION_INPUT_LEDGER')
    np.testing.assert_array_equal(snapshot.state,bundle['initial_state'])
    report=replay(r,bundle,snapshot)
    return dict(passed=report['max_state_abs']<=provenance['profile']['replay_state_atol'],source=str(root),replay=report,
                model_hash=r.model_hash,source_reports_modified=False)
