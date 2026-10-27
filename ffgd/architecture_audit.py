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
from .orchestration import atomic_json,check_budget,source_hash


def audit(path,config):
    torch.set_num_threads(1);runs=[Path(p) for p in config['planning_runs']]
    learning=json.loads((Path(config['learning_checks_run'])/'result.json').read_text(encoding='utf-8'))
    rows=[];conditions=set();scene_hashes=set();pose_counts=set();selected_for_counterfactual=None
    for root in runs:
        result=json.loads((root/'planning_result.json').read_text(encoding='utf-8'))
        manifest=json.loads((root/'planning_manifest.json').read_text(encoding='utf-8'))
        sentinel=json.loads((root/'architecture_sentinel.json').read_text(encoding='utf-8'))
        unit_manifest=json.loads((root/'manifest.json').read_text(encoding='utf-8'))
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
                           repair_calls=record['post_sampling_trajectory_repair_calls'],readonly=record.get('validation',{}).get('input_unchanged',False),
                           energy_measured_every_step=all(row['energy_status']=='MEASURED' for row in trace))
            candidates.append(candidate)
            if selected_for_counterfactual is None and active:selected_for_counterfactual=(root,record['index'],active[0])
        rows.append(dict(run=str(root),sampling_source_hash=unit_manifest['source_hash'],physics_backend=manifest.get('physics_backend'),
                         candidates=candidates,unique_noise=len(set(noise_hashes))==len(noise_hashes),
                         runtime_sentinel_clear=not sentinel['forbidden_calls'] and sentinel['observed_ffgd_calls']>0,
                         file_audit_enforced=sentinel.get('file_audit_enforced',False),actual_methods_recorded=bool(sentinel.get('actual_methods')),
                         execution_called_during_planning=any(module in sentinel['modules'] for module in ('ffgd.execution','ffgd.execution_acceptance')),
                         denominator_complete=len(result['records'])==result['requested_candidates'],
                         independent_validation_terms_disabled=manifest['independent_validation_terms_disabled']))
    interventions=None;contract=None;semantic=None
    if selected_for_counterfactual:
        check_budget();root,index,step=selected_for_counterfactual
        r,snapshot,task,problem=load_problem(root/'problem.json');manifest=json.loads((root/'planning_manifest.json').read_text(encoding='utf-8'))
        model,checkpoint,identity=load_model(config['checkpoint'],'cpu')
        if identity['sha256']!=manifest['checkpoint']['sha256']:raise ValueError('AUDIT_CHECKPOINT_MISMATCH')
        with np.load(root/f'candidate_{index:03d}'/'denoising.npz') as d:z=checkpoint['mean']+checkpoint['scale']*d['clean'][step]
        codec=boundary_codec(r,snapshot,task.duration,manifest['budget']['control_points']);tree=DifferentiableTree(r);g=Geometry(r)
        from .architecture_contracts import static_contract,semantic_equivalence
        contract=static_contract(r,snapshot,task,codec)
        semantic=semantic_equivalence(path,root,r,snapshot,task,codec,z,checkpoint,model,manifest)
        atomic_json(path/'contracts.json',dict(static=contract,semantic=semantic))
        p=Propagator(tree,snapshot);base=MultiSpaceEnergy(p,codec,g,task,manifest['budget']['intervals'])
        changed=[]
        for item in task.poses:
            changed.append(PoseConstraint(item.time,item.tip,item.position+[0,.03,.02],item.rotation,item.position_tolerance,item.rotation_tolerance,item.position_enabled,item.rotation_enabled))
        goal=MultiSpaceEnergy(p,codec,g,TaskSpec(task.duration,tuple(changed)),manifest['budget']['intervals'])
        data=r.restore(snapshot);mujoco.mj_forward(r.model,data);target=int(r.model.joint(r.spec.target_free_joint_name).qposadr[0])
        data.qpos[target:target+3]=data.xpos[r.tips[0]]+[.15,.01,.01]
        obstacle=MultiSpaceEnergy(Propagator(tree,r.snapshot(data)),codec,g,task,manifest['budget']['intervals'])
        for energy in (base,goal,obstacle):energy.weights.update(manifest.get('guide_weights',{}))
        ablations=[]
        with torch.no_grad():
            before={k:float(v) for k,v in base.evaluate(tensor(z)).items()}
            after_goal={k:float(v) for k,v in goal.evaluate(tensor(z)).items()}
            after_obstacle={k:float(v) for k,v in obstacle.evaluate(tensor(z)).items()}
            for name in base.weights:
                weight=base.weights[name];base.weights[name]=0.
                disabled={k:float(v) for k,v in base.evaluate(tensor(z)).items()};base.weights[name]=weight
                ablations.append(dict(disabled_term=name,expected_total=before['total']-weight*before[name],observed_total=disabled['total'],
                                      primitive_terms_unchanged=all(disabled[k]==before[k] for k in base.weights),
                                      removed_only_guide_term=abs(disabled['total']-(before['total']-weight*before[name]))<1e-8))
        interventions=dict(source_run=str(root),candidate=index,step=step,before=before,goal_changed=after_goal,entity_geometry_changed=after_obstacle,
                           goal_effect=abs(before['task']-after_goal['task'])>1e-8,collision_effect=abs(before['collision']-after_obstacle['collision'])>1e-8,
                           changed_inputs_only='public world goal positions; stationary target entity pose',guide_term_ablations=ablations)
    execution_rows=[]
    for source in config.get('execution_runs',[]):
        root=Path(source);result=json.loads((root/'result.json').read_text(encoding='utf-8'))
        acceptance=result['acceptance'];summary=result['execution'];validation=json.loads((root/'validation_before_execution.json').read_text(encoding='utf-8'))
        with np.load(root/'execution.npz') as data:
            # Applied control is recorded independently of raw command. Do not
            # reinterpret a clipped/aborted run as unassisted planning success.
            raw_max=summary['raw_torque_ratio'];native_steps=len(data['ctrl'])
        execution_rows.append(dict(run=str(root),model_plan_valid=validation['status']=='PLAN_VALID',
                                   safety_enabled=acceptance['safety_enabled'],cartesian_assist=acceptance['task_based_cartesian_assist'],
                                   raw_torque_ratio=raw_max,native_steps=native_steps,status=acceptance['status'],
                                   safety_intervention=summary['status']!='FINISHED' or raw_max is None or raw_max>1.,
                                   independent_replay=not acceptance['replay']['per_frame_state_injection'],
                                   distinct_plan_and_execution_reports=True))
    checked=[c for row in rows for c in row['candidates'] if c['checked']]
    gates={
        'ARCH_01':bool(rows and contract and not contract['static_import_violations']) and all(x['runtime_sentinel_clear'] and x['file_audit_enforced'] and x['actual_methods_recorded'] for x in rows),
        'ARCH_02':bool(checked) and all(c['readonly'] for c in checked),
        'ARCH_03':bool(checked) and all(c['consumed'] and c['active_nonfinal_steps']>=2 and c['repair_calls']==0 for c in checked),
        'ARCH_04':bool(interventions and interventions['goal_effect'] and interventions['collision_effect']) and all(a['removed_only_guide_term'] and a['primitive_terms_unchanged'] for a in interventions['guide_term_ablations']) and all(all(v>1e-10 for v in c['state_intervention_effect']) for c in checked) and all(not r['independent_validation_terms_disabled'] for r in rows),
        'ARCH_05':bool(checked) and all(c['noise_identity'] and c['network_calls']==c['requested_calls'] for c in checked) and all(row['unique_noise'] for row in rows),
        'ARCH_06':learning['passed'] and all(s['condition_shuffle_norm']>1e-8 and s['unlearned_prediction_difference']>1e-8 and s['weight_intervention_norm']>1e-8 for s in learning['samples']),
        'ARCH_07':bool(contract and contract['passed']),
        'ARCH_08':bool(semantic and semantic['passed']) and len(pose_counts)>=2,
        'ARCH_09':bool(rows) and all(row['denominator_complete'] for row in rows),
        'ARCH_10':bool(execution_rows and contract and contract['controller_reference_only']) and not any(row['execution_called_during_planning'] for row in rows) and all(e['model_plan_valid'] and e['safety_enabled'] and not e['cartesian_assist'] and e['independent_replay'] for e in execution_rows)}
    return dict(passed=all(value is True for value in gates.values()),gates=gates,runs=rows,interventions=interventions,
                tested_task_conditions=len(conditions),tested_pose_counts=sorted(pose_counts),tested_scenes=len(scene_hashes),
                architecture_status='ARCHITECTURE_PASSED' if all(value is True for value in gates.values()) else 'PARTIAL',
                analysis_source_hash=source_hash(),engineering_status='NOT_RUN',learning_benefit_status='NOT_RUN',
                contracts=contract,semantic_equivalence=semantic,executions=execution_rows,
                limitation='Mechanism evidence only; learning advantage and final engineering completion are separate gates.')
