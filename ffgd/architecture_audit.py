"""Audit real learned planning artifacts plus controlled physical interventions.

Does not create or repair trajectories. A failed/small-scope audit cannot promote
the milestone or manufacture a successful engineering result.
"""
import json
import hashlib
from pathlib import Path
import numpy as np
import torch
import mujoco
from .problem_io import load_problem
from .model_io import load_model
from .api import boundary_codec
from .physics import DifferentiableTree,tensor
from .propagation import Propagator
from .geometry import Geometry
from .costs import MultiSpaceEnergy
from .contracts import TaskSpec,PoseConstraint
from .orchestration import atomic_json,check_budget


def audit(path,config):
    torch.set_num_threads(1);runs=[Path(p) for p in config['planning_runs']]
    learning=json.loads((Path(config['learning_checks_run'])/'result.json').read_text(encoding='utf-8'))
    rows=[];conditions=set();scene_hashes=set();pose_counts=set();selected_for_counterfactual=None
    for root in runs:
        result=json.loads((root/'planning_result.json').read_text(encoding='utf-8'))
        manifest=json.loads((root/'planning_manifest.json').read_text(encoding='utf-8'))
        sentinel=json.loads((root/'architecture_sentinel.json').read_text(encoding='utf-8'))
        r,snapshot,task,problem=load_problem(root/'problem.json')
        pose_counts.add(len(task.poses));scene_hashes.add(manifest['scene_hash'])
        conditions.add(hashlib.sha256(json.dumps(problem['task'],sort_keys=True).encode()).hexdigest())
        candidates=[];noise_hashes=[]
        for record in result['records']:
            folder=root/f'candidate_{record["index"]:03d}'
            if not record['sampled']:
                candidates.append(dict(index=record['index'],status=record['status'],checked=False));continue
            trace=json.loads((folder/'denoising.json').read_text(encoding='utf-8'))
            with np.load(folder/'denoising.npz') as stored:arrays={k:stored[k] for k in stored.files}
            observed=np.load(folder/'observed_network_inputs.npy')[:,0]
            noise=np.load(folder/'initial_noise.npy');noise_hash=hashlib.sha256(noise.tobytes()).hexdigest();noise_hashes.append(noise_hash)
            consumed=np.array_equal(observed,arrays['input']) and np.array_equal(observed[1:],arrays['output'][:-1])
            active=[i for i,row in enumerate(trace[:-1]) if row['active'] and np.linalg.norm(arrays['guided'][i]-arrays['clean'][i])>1e-12]
            alpha=np.cumprod(1-np.linspace(.0001,.02,1000));counterfactual=[]
            for i in active:
                k=trace[i]['noise_level'];knext=trace[i+1]['noise_level'];a=np.sqrt(alpha[k]);s=np.sqrt(1-alpha[k])
                nominal=np.sqrt(alpha[knext])*arrays['clean'][i]+np.sqrt(1-alpha[knext])*(arrays['input'][i]-a*arrays['clean'][i])/s
                counterfactual.append(float(np.linalg.norm(nominal-arrays['output'][i])))
            candidate=dict(index=record['index'],checked=True,status=record['status'],network_calls=record['observed_network_calls'],
                           requested_calls=manifest['budget']['sampling']['steps'],consumed=consumed,active_nonfinal_steps=len(active),
                           state_intervention_effect=counterfactual,noise_identity=record['initial_noise_hash']==noise_hash,
                           repair_calls=record['post_sampling_trajectory_repair_calls'],readonly=record['validation'].get('input_unchanged',False),
                           energy_measured_every_step=all(row['energy_status']=='MEASURED' for row in trace))
            candidates.append(candidate)
            if selected_for_counterfactual is None and active:selected_for_counterfactual=(root,record['index'],active[0])
        rows.append(dict(run=str(root),candidates=candidates,unique_noise=len(set(noise_hashes))==len(noise_hashes),
                         runtime_sentinel_clear=not sentinel['forbidden_calls'] and sentinel['observed_ffgd_calls']>0,
                         denominator_complete=len(result['records'])==result['requested_candidates'],
                         independent_validation_terms_disabled=manifest['independent_validation_terms_disabled']))
    interventions=None
    if selected_for_counterfactual:
        check_budget();root,index,step=selected_for_counterfactual
        r,snapshot,task,problem=load_problem(root/'problem.json');manifest=json.loads((root/'planning_manifest.json').read_text(encoding='utf-8'))
        model,checkpoint,identity=load_model(config['checkpoint'],'cpu')
        if identity['sha256']!=manifest['checkpoint']['sha256']:raise ValueError('AUDIT_CHECKPOINT_MISMATCH')
        with np.load(root/f'candidate_{index:03d}'/'denoising.npz') as d:z=checkpoint['mean']+checkpoint['scale']*d['clean'][step]
        codec=boundary_codec(r,snapshot,task.duration,manifest['budget']['control_points']);tree=DifferentiableTree(r);g=Geometry(r)
        p=Propagator(tree,snapshot);base=MultiSpaceEnergy(p,codec,g,task,manifest['budget']['intervals'])
        changed=[]
        for item in task.poses:
            changed.append(PoseConstraint(item.time,item.tip,item.position+[0,.03,.02],item.rotation,item.position_tolerance,item.rotation_tolerance,item.position_enabled,item.rotation_enabled))
        goal=MultiSpaceEnergy(p,codec,g,TaskSpec(task.duration,tuple(changed)),manifest['budget']['intervals'])
        data=r.restore(snapshot);mujoco.mj_forward(r.model,data);target=int(r.model.joint(r.spec.target_free_joint_name).qposadr[0])
        data.qpos[target:target+3]=data.xpos[r.tips[0]]+[.15,.01,.01]
        obstacle=MultiSpaceEnergy(Propagator(tree,r.snapshot(data)),codec,g,task,manifest['budget']['intervals'])
        with torch.no_grad():
            before={k:float(v) for k,v in base.evaluate(tensor(z)).items()}
            after_goal={k:float(v) for k,v in goal.evaluate(tensor(z)).items()}
            after_obstacle={k:float(v) for k,v in obstacle.evaluate(tensor(z)).items()}
        interventions=dict(source_run=str(root),candidate=index,step=step,before=before,goal_changed=after_goal,entity_geometry_changed=after_obstacle,
                           goal_effect=abs(before['task']-after_goal['task'])>1e-8,collision_effect=abs(before['collision']-after_obstacle['collision'])>1e-8,
                           changed_inputs_only='public world goal positions; stationary target entity pose')
    checked=[c for row in rows for c in row['candidates'] if c['checked']]
    gates={
        'ARCH_01':bool(rows) and all(x['runtime_sentinel_clear'] for x in rows),
        'ARCH_02':bool(checked) and all(c['readonly'] for c in checked),
        'ARCH_03':bool(checked) and all(c['consumed'] and c['active_nonfinal_steps']>=2 and c['repair_calls']==0 for c in checked),
        'ARCH_04':bool(interventions and interventions['goal_effect'] and interventions['collision_effect']) and all(all(v>1e-10 for v in c['state_intervention_effect']) for c in checked) and all(not r['independent_validation_terms_disabled'] for r in rows),
        'ARCH_05':bool(checked) and all(c['noise_identity'] and c['network_calls']==c['requested_calls'] for c in checked) and all(row['unique_noise'] for row in rows),
        'ARCH_06':learning['passed'] and all(s['condition_shuffle_norm']>1e-8 and s['unlearned_prediction_difference']>1e-8 and s['weight_intervention_norm']>1e-8 for s in learning['samples']),
        'ARCH_07':'NOT_RUN_STATIC_CONTRACT_REVIEW_REQUIRED',
        'ARCH_08':'NOT_RUN_SEMANTIC_ORDER_AND_FILENAME_COMPARISON_REQUIRED',
        'ARCH_09':bool(rows) and all(row['denominator_complete'] for row in rows),
        'ARCH_10':'NOT_RUN_EXECUTION_POLICY_REVIEW_REQUIRED'}
    return dict(passed=all(value is True for value in gates.values()),gates=gates,runs=rows,interventions=interventions,
                tested_task_conditions=len(conditions),tested_pose_counts=sorted(pose_counts),tested_scenes=len(scene_hashes),
                architecture_status='PARTIAL',engineering_status='NOT_RUN',learning_benefit_status='NOT_RUN',
                limitation='Artifact mechanisms only; static contract and execution-policy gates need separate evidence.')
