"""Read-only contract/semantic checks against real public inputs and weights."""
import ast
import hashlib
import inspect
import json
from pathlib import Path
import numpy as np
import torch
from . import api,execution,contracts
from .trajectory import basis
from .problem_io import load_problem
from .conditioning import encode_condition
from .model_io import normalize_condition
from .physics import DifferentiableTree,tensor
from .propagation import Propagator,FullStatePropagation
from .geometry import Geometry
from .costs import MultiSpaceEnergy


def static_contract(robot,snapshot,task,codec):
    root=Path(__file__).parent
    modules=['api','robot','diffusion','guidance_metric','model','model_io','conditioning','problem_io','costs','physics','propagation','geometry','convex_guide','validation','full_state_validation','execution']
    imports=[];violations=[];hashes={}
    for name in modules:
        source=root/(name+'.py');content=source.read_text(encoding='utf-8');hashes[name]=hashlib.sha256(source.read_bytes()).hexdigest()
        for node in ast.walk(ast.parse(content)):
            names=[item.name for item in node.names] if isinstance(node,ast.Import) else [node.module] if isinstance(node,ast.ImportFrom) and node.module else []
            for imported in names:
                imports.append(dict(module=name,imported=imported))
                if imported.startswith(('ff_prior','v6_lite','tracking_diffusion','ffgd.baselines')):violations.append(imports[-1])
    # Rank17 means all terminal joint coordinates remain adjustable, despite
    # fixed initial state and terminal first/second derivative boundaries.
    endpoint_operator=np.einsum('n,njf->jf',basis(codec.n,task.duration,task.duration),codec.N.reshape(codec.n,codec.dof,-1))
    endpoint_rank=int(np.linalg.matrix_rank(endpoint_operator))
    fields={name:list(getattr(contracts,name).__dataclass_fields__) for name in ('TaskSpec','PoseConstraint','SceneSpec','Snapshot')}
    planner_parameters=list(inspect.signature(api.plan).parameters)
    forbidden={'terminal_q','terminal_q17','solution','expert_path','certificate','bypass'}
    no_solution_fields=not forbidden.intersection(planner_parameters+sum(fields.values(),[]))
    command_parameters=list(inspect.signature(execution.command).parameters)
    controller_reference_only=command_parameters==['robot','data','q','qd','qdd']
    return dict(passed=not violations and endpoint_rank==17 and no_solution_fields,static_import_violations=violations,
                imports=imports,source_hashes=hashes,terminal_position_rank=endpoint_rank,free_dimension=codec.free_dim,
                public_fields=fields,planner_parameters=planner_parameters,no_solution_fields=no_solution_fields,
                controller_reference_only=controller_reference_only,controller_parameters=command_parameters)


def semantic_equivalence(output,source_run,robot,snapshot,task,codec,z,checkpoint,model,manifest=None):
    """Renaming/metadata/object order must preserve both physics and network."""
    source_run=Path(source_run);value=json.loads((source_run/'problem.json').read_text(encoding='utf-8'))
    value['initial']={'snapshot_file':str((source_run/'initial_state.npz').resolve())}
    value['id']='arbitrary_metadata_renaming';value['description']='Semantic equivalence check; no solver input.'
    value['task']['poses'].reverse();value.setdefault('scene',{}).setdefault('boxes',[]).reverse()
    renamed=Path(output)/'renamed_reordered_problem.json';renamed.write_text(json.dumps(value,indent=2),encoding='utf-8')
    other,other_snapshot,other_task,_=load_problem(renamed)
    prop_type=FullStatePropagation if (manifest or {}).get('physics_backend')=='full_state' else Propagator
    prop=prop_type(DifferentiableTree(robot),snapshot);other_prop=prop_type(DifferentiableTree(other),other_snapshot)
    tokens=encode_condition(robot,snapshot,task,prop.H.numpy());other_tokens=encode_condition(other,other_snapshot,other_task,other_prop.H.numpy())
    equal_tokens=np.array_equal(tokens,other_tokens)
    from .model_io import bind_free_coordinates
    checkpoint=bind_free_coordinates(checkpoint,task.duration)
    mean=np.asarray(checkpoint['mean']);scale=np.asarray(checkpoint['scale']);x=torch.tensor((z-mean)/scale,dtype=torch.float32)[None]
    mask=np.zeros((1,len(tokens)),bool)
    with torch.no_grad():
        left=model(x,torch.tensor([100]),normalize_condition(tokens[None],mask,checkpoint,'cpu'))
        right=model(x,torch.tensor([100]),normalize_condition(other_tokens[None],mask,checkpoint,'cpu'))
        manifest={} if manifest is None else manifest
        intervals=manifest.get('budget',{}).get('intervals',32)
        energy=MultiSpaceEnergy(prop,codec,Geometry(robot),task,intervals)
        other_codec=api.boundary_codec(other,other_snapshot,other_task.duration,codec.n,origin=codec.origin)
        alternate=MultiSpaceEnergy(other_prop,other_codec,Geometry(other),other_task,intervals)
        for objective in (energy,alternate):
            objective.weights.update(manifest.get('guide_weights',{}))
            objective.collision_aggregation=manifest.get('collision_aggregation','pair_mean')
            objective.joint_peak_penalty=manifest.get('joint_peak_penalty',False)
            objective.task_objective=manifest.get('task_objective','center_squared')
            objective.set_planning_margins(manifest.get('planning_margins'))
        a={k:float(v) for k,v in energy.evaluate(tensor(z)).items()}
        b={k:float(v) for k,v in alternate.evaluate(tensor(z)).items()}
    energy_error=max(abs(a[k]-b[k]) for k in a);network_error=float(torch.max(abs(left-right)))
    same_model=robot.model_hash==other.model_hash;same_state=np.array_equal(snapshot.state,other_snapshot.state)
    return dict(passed=same_model and same_state and equal_tokens and energy_error==0 and network_error==0,
                same_model=same_model,same_state=same_state,identical_condition_tokens=equal_tokens,
                maximum_energy_error=energy_error,maximum_network_error=network_error,source_run=str(source_run),
                metadata_not_passed_to_solver=True,perturbed_fields=['filename','id','description','pose_order','box_order'])
