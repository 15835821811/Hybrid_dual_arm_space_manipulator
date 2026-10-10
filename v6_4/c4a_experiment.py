"""Explicit C4-A DEV preparation and bounded execution, never C3 experiment I/O.

Use prepare, then run. No automatic physical retry after incomplete/error work.
The original candidate evaluator and independent Actual runner own all physics.
"""
import argparse
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from unittest.mock import patch

import mujoco
import numpy as np

from .c4a_frozen import ROOT, RELEASE, load_frozen_sampler
from .c4a_portfolio import optimize_strategy
from .route_optimizer_protocol import (read,write,sha,digest,TaskSpec,PreferenceSpec,active_intervals,
    build_reference_definition,related_pairs,git)

STRATEGIES=('P0','P1','P2')
LIMITS={'candidate_slots':48,'actual_logical_slots':12,'ddim_samples':8}


def now(): return datetime.now(timezone.utc).isoformat()


def reserve(run,category,key,count):
    if category not in LIMITS or count<0: raise ValueError('unknown/invalid budget')
    path=Path(run)/'budget_ledger'/category/(key+'.json')
    if path.exists(): raise RuntimeError('already reserved; no implicit physical retry: '+str(path))
    consumed=sum(read(p)['count'] for p in path.parent.glob('*.json')) if path.parent.exists() else 0
    if consumed+count>LIMITS[category]: raise ValueError('C4-A hard budget exceeded')
    write(path,dict(key=key,count=count,category=category,created_utc=now(),maximum_reservation=True))


def dev_noise(task):
    words=np.frombuffer(hashlib.sha256(task.sha256().encode('ascii')).digest(),dtype='<u4')
    rng=np.random.default_rng(np.random.SeedSequence([64424,*map(int,words)]))
    return {slot:rng.standard_normal(12).astype(np.float32) for slot in (1,3)}


def verify(run):
    from .residual_execution import source_guard
    source_guard(Path(run)/'source_identity.json')
    plan=read(Path(run)/'plan.json')
    if plan['schema']!='v64_c4a_dev_protocol_v1' or plan['limits']!=LIMITS or len(plan['tasks'])!=2:
        raise ValueError('not the frozen C4-A protocol')
    return plan


def prepare(run):
    from .search_aware_warmstart_experiment import _used_seeds
    from .preference_warmstart_protocol import next_unused_seeds
    from v6_lite.run_v6_lite import default_v6_lite_robot_spec,build_scenarios,V6LiteRunConfig
    from .task_protocol import task_from_scenario
    from .task_anchored_reference import _target
    from .route_pair_protocol import _geometry_check
    run=Path(run).resolve()
    if run.exists(): raise FileExistsError('exclusive DEV root required')
    for receipt in ('numerics_01','baseline_regression_02','portfolio_01','dev_contracts_01'):
        if read(ROOT/'docs/audit_receipts/c4a'/receipt/'receipt.json')['exit_code'] != 0:
            raise ValueError('required pre-physics audit has not passed: '+receipt)
    if not read(ROOT/'v6_4/c4a_evidence/audit_01/audit_summary.json')['portable_verified']['portable_bytes_verified']:
        raise ValueError('frozen C3 input audit incomplete')
    # No result-dependent choices, no silent uncommitted producer.
    if git('diff','HEAD','--','v6_4','v6_lite','model_test','dual_arm_space_robot_2026'):
        raise ValueError('commit numerical implementation before freeze')
    used,inventory=_used_seeds();choices=next_unused_seeds(used,count=2)
    if choices!=[(21,2028300010),(22,2028404739)]: raise ValueError('source inventory differs from predeclared seeds; stop before generation')
    run.mkdir(parents=True)
    write(run/'generation_contract.json',dict(created_utc=now(),choices=choices,sides=[1,-1],
        used_seeds=sorted(used),source_inventory=inventory,noise_seed=64424,
        no_result_dependent_redraw=True,protocol_sha256=sha(ROOT/'docs/V6_4_C4A_EXPERIMENT_PROTOCOL.md')))
    spec=default_v6_lite_robot_spec();rows=[]
    with patch.object(mujoco,'mj_step',side_effect=AssertionError('static prepare forbids integration')):
        for index,((source_index,seed),side,label) in enumerate(zip(choices,(1,-1),('plus','minus'))):
            group=f'c4a_dev{index}';tid=group+'_'+label
            mother=build_scenarios(spec,V6LiteRunConfig(scenario_count=1,seed=seed))[0]
            mother_sha=digest(mother.to_dict())
            layout=dict(mother_seed=seed,mother_source_sha256=mother_sha,next_unused_source_index=source_index,
                evaluation_role='C4A_DEV',sphere_offset_m=.055,external_split='dev')
            base=task_from_scenario(spec,mother,task_id=group+'_source',group_id=group,
                family='end_effector_detour',split='test',layout_diagnostics=layout)
            definition=build_reference_definition(base);active=active_intervals(definition)
            point=_target(base).sample(float(np.mean(definition['intervals_s'][2])))[0]
            basis=np.asarray(definition['transverse_bases'][2]);obstacles=list(mother.obstacles)
            if obstacles[1].radius!=.025: raise ValueError('original route sphere contract differs')
            obstacles[1]=replace(obstacles[1],center=point+side*.055*basis[:,0])
            scene=replace(mother,scenario_id=tid,obstacles=tuple(obstacles))
            task=task_from_scenario(spec,scene,task_id=tid,group_id=group,family='end_effector_detour',split='test',
                requirements=base.requirements,layout_diagnostics=layout)
            directory=run/'frozen_tasks'/tid
            write(directory/'mother.json',mother.to_dict());write(directory/'task.json',task.to_dict())
            d=build_reference_definition(task);check=_geometry_check(spec,task,d)
            write(directory/'geometry_precheck.json',check)
            verifier,pairs=related_pairs(spec,task);pair_ids=tuple((p.geom_a_name,p.geom_b_name) for p in pairs)
            windows=[d['intervals_s'][i] for i in active]
            prefs=[PreferenceSpec(name,task.sha256(),(tuple(d['intervals_s'][2]),),tuple(tuple(w) for w in windows),
                task.scenario['workspace_obstacles'][1]['name'],pair_ids).to_dict() for name in ('A','B')]
            write(directory/'preferences.json',prefs)
            write(directory/'noises.json',{str(k):v.tolist() for k,v in dev_noise(task).items()})
            rows.append(dict(task_id=tid,group_id=group,mother_id=group,split='dev',seed=seed,
                task_sha256=task.sha256(),mother_source_sha256=mother_sha,next_unused_source_index=source_index,
                side=side,task_path=f'frozen_tasks/{tid}/task.json',active_intervals=active,active_dim=len(active)*2,
                W_key=[d['intervals_s'][2]],W_support=windows,W_full=[[0.,27.]],
                obstacle_name=task.scenario['workspace_obstacles'][1]['name'],related_pair_ids=pair_ids,
                pair_policy_sha256=verifier._pair_policy_sha256,geometry_precheck_passed=check['passed'],
                initial_geometry_query_count=check['native_distance_queries'],applicable=bool(active),
                initial_state_sha256=digest([task.initial_qpos,task.initial_qvel])))
    old=read(RELEASE/'snapshot/learning_split_manifest.json')['tasks']
    for field in ('seed','mother_source_sha256','task_sha256'):
        if {r[field] for r in rows}&{r[field] for r in old}: raise ValueError('DEV/history identity overlap')
    for name in ('frozen_execution_config.json','frozen_run_config.json'):
        shutil.copyfile(RELEASE/'snapshot'/name,run/name)
    order=[(rows[0]['task_id'],m) for m in STRATEGIES]+[(rows[1]['task_id'],m) for m in ('P1','P2','P0')]
    plan=dict(schema='v64_c4a_dev_protocol_v1',created_utc=now(),tasks=rows,limits=LIMITS,strategies=list(STRATEGIES),
        search_order=order,actual_order=[['P0','P1','P2'],['P1','P2','P0']],noise_seed=64424,
        base_commit='18f6329e3960c76dce8b787749b269695888f9c9',producer_commit=git('rev-parse','HEAD'),
        near_R12='N/A_NOT_RUN_BUDGET',new_training_updates=0,no_C3_TEST_execution=True,
        actual_transport_schema='original C3 executor with independent DEV seal',
        static_passed=all(r['applicable'] and r['geometry_precheck_passed'] for r in rows))
    write(run/'plan.json',plan);write(run/'learning_split_manifest.json',dict(schema='v64_c4a_external_dev_split_v1',tasks=rows,history=old))
    sources=[p for name in ('v6_4','v6_lite','model_test') for p in (ROOT/name).rglob('*.py')
        if not any(x in ('output','releases','visualization','__pycache__','c4a_evidence') for x in p.relative_to(ROOT).parts)]
    sources.extend(spec._source_assets())
    protected={str(p.resolve()):sha(p) for p in run.rglob('*') if p.is_file()}
    freeze=read(RELEASE/'snapshot/model_freeze.json')
    from .visualization.export_search_aware_release import PortableResolver
    resolver=PortableResolver(RELEASE)
    for original,expected in freeze['artifacts'].items():
        p=resolver.resolve(original,expected);protected[str(p)]=sha(p)
    write(run/'source_identity.json',dict(schema='v64_c4a_source_identity_v1',git_head=git('rev-parse','HEAD'),
        source_sha256={p.resolve().relative_to(ROOT).as_posix():sha(p) for p in sorted(set(sources))},
        protected_artifacts=protected,python=sys.version,python_executable=sys.executable,created_utc=now()))
    write(run/'prepare_receipt.json',dict(status='PASS' if plan['static_passed'] else 'BLOCKED_STATIC',
        physical_integrations=0,new_DDIM_samples=0,tasks_frozen=2,geometry_queries=sum(r['initial_geometry_query_count'] for r in rows)))
    if not plan['static_passed']: raise RuntimeError('fixed DEV static contract failed; no task replacement')
    return plan


def load_task(run,tid):
    plan=verify(run);frozen=next(r for r in plan['tasks'] if r['task_id']==tid)
    task=TaskSpec.from_dict(read(Path(run)/frozen['task_path']))
    prefs=[PreferenceSpec(**p) for p in read(Path(run)/'frozen_tasks'/tid/'preferences.json')]
    if task.sha256()!=frozen['task_sha256']: raise ValueError('DEV task differs')
    return task,prefs,frozen


def search(run,tid,strategy):
    from .route_candidate_evaluator import NominalCandidateEvaluator
    from .route_initializers import json_raw
    import torch
    torch.set_num_threads(1)
    run=Path(run).resolve();started=time.perf_counter()
    task,prefs,frozen=load_task(run,tid)
    if strategy not in STRATEGIES: raise ValueError('unknown strategy')
    reserve(run,'candidate_slots',tid+'_'+strategy,8)
    path=run/'search'/tid/strategy;path.mkdir(parents=True,exist_ok=False)
    for name in ('plan.json','source_identity.json','frozen_execution_config.json','frozen_run_config.json'):
        shutil.copyfile(run/name,path/name)
    proposals=None;setup={};t=time.perf_counter()
    if strategy!='P0':
        reserve(run,'ddim_samples',tid+'_'+strategy,2)
        sampler=load_frozen_sampler('D');noises=read(run/'frozen_tasks'/tid/'noises.json');proposals={}
        for slot,p,f in [(1,'A','v1'),(3,'B','v2')]:
            raw,meta=sampler.sample(task,p,f,np.array(noises[str(slot)],dtype=np.float32))
            proposals[slot]={**meta,'raw_z_m':json_raw(raw),'noise_seed':64424}
        setup=dict(total_setup_s=time.perf_counter()-t,model_load_wall_s=sampler.model_load_wall_s,
            ddim_samples=sampler.sample_units,inference_s=sum(p['inference_wall_s'] for p in proposals.values()),
            condition_prepare_s=sum(p['condition_prepare_wall_s'] for p in proposals.values()))
        write(path/'initializer_proposals.json',dict(proposals=json_raw(proposals),timing=setup))
        other=run/'search'/tid/('P1' if strategy=='P2' else 'P2')/'initializer_proposals.json'
        if other.exists():
            saved=read(other)['proposals']
            if any(saved[str(k)]['raw_z_m']!=proposals[k]['raw_z_m'] for k in (1,3)):
                raise ValueError('P1/P2 frozen proposal mismatch; no physics')
    evaluator=NominalCandidateEvaluator(path,task,frozen)
    result=optimize_strategy(strategy,task,prefs,evaluator.execution_identity,evaluator,path/'planning'/tid,proposals=proposals)
    rows=read(path/'planning'/tid/'candidate_registry.json')
    write(path/'planning_cost.json',dict(task_id=tid,strategy=strategy,created_utc=now(),
        end_to_end_cold_service_s=time.perf_counter()-started,setup=setup,selection_sha256=sha(path/'planning'/tid/'selection.json'),
        actual_candidate_slots=len(rows),actual_prediction_calls=sum(r['prediction_rollout_started'] for r in rows),
        shared_A_B_pool=True,outer_process_wall_s='recorded separately in command receipt',
        complete=result['budget']['stop_reason']!='TOOL_ERROR'))
    verify(run)
    if result['budget']['stop_reason']=='TOOL_ERROR': raise RuntimeError('consumed physical tool error; no retry')
    return result['budget']


def seal_dev(run):
    """DEV-specific seal, consumed by unchanged C3 strict Actual executor."""
    run=Path(run).resolve();plan=verify(run);phase=run/'validation'
    if phase.exists(): raise FileExistsError('DEV phase already sealed; no implicit rerun')
    files={};entries=[]
    for frozen in plan['tasks']:
        tid=frozen['task_id']
        for strategy in STRATEGIES:
            stream=run/'search'/tid/strategy;root=stream/'planning'/tid
            source=root/'selection.json';selection=read(source)
            if selection['task_sha256']!=frozen['task_sha256'] or selection['selection_reads_final_actual']:
                raise ValueError('selection identity differs')
            if not read(stream/'planning_cost.json')['complete']: raise ValueError('incomplete search')
            target=phase/'sealed_selections'/tid/(strategy+'.json');write(target,selection)
            entry=dict(task_id=tid,endpoint=strategy,task_sha256=frozen['task_sha256'],
                selection_source_path=str(source),selection_path=str(target),selection_sha256=sha(target),
                candidate_registry_path=str(root/'candidate_registry.json'),proposals_path=str(root/'proposals.json'),
                planning_cost_path=str(stream/'planning_cost.json'))
            for p in (source,target,root/'candidate_registry.json',root/'proposals.json',stream/'planning_cost.json'):
                files[str(p)]=sha(p)
            entries.append(entry)
    write(phase/'sealed_selections/all_selections.json',dict(schema='v64_c4a_all_dev_selections_sealed_v1',phase='DEV',
        sealed_utc=now(),task_ids=[r['task_id'] for r in plan['tasks']],endpoints=list(STRATEGIES),entries=entries,
        files=files,checkpoint_files={},bindings={'protocol_sha256':sha(run/'plan.json'),'scope':'C4A_DEV_ONLY'},
        logical_actual_slots=12,selection_reads_actual=False,actual_started=False))


def actual(run,tid):
    from .closed_loop_warmstart_validation import execute_frozen_task
    run=Path(run).resolve();plan=verify(run);task,prefs,frozen=load_task(run,tid)
    reserve(run,'actual_logical_slots',tid,6)
    order=plan['actual_order'][[r['task_id'] for r in plan['tasks']].index(tid)]
    slots=execute_frozen_task(run/'validation',task,frozen,execution_run=run,endpoints=order)
    verify(run)
    return dict(logical_slots=len(slots),unique_runs=sum(s['unique_run'] for s in slots),
        aliases=sum(bool(s.get('alias_of_slot')) for s in slots),passed=sum(s['full_task_success'] for s in slots))


def child_command(run,key,action,*args):
    directory=Path(run)/'command_logs'/key;directory.mkdir(parents=True,exist_ok=False)
    argv=[sys.executable,'-B','-X','utf8','-m','v6_4.c4a_experiment',action,'--run',str(run),*args]
    write(directory/'started.json',dict(argv=argv,cwd=str(ROOT),created_utc=now()))
    started=time.perf_counter()
    env=os.environ.copy();env.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMBA_NUM_THREADS='1')
    with (directory/'stdout.txt').open('x',encoding='utf8') as out,(directory/'stderr.txt').open('x',encoding='utf8') as err:
        p=subprocess.Popen(argv,cwd=ROOT,env=env,stdout=out,stderr=err)
        try: code=p.wait(timeout=14400)
        except subprocess.TimeoutExpired:
            p.kill();p.wait();code=-9
    write(directory/'receipt.json',dict(argv=argv,cwd=str(ROOT),exit_code=code,elapsed_s=time.perf_counter()-started,
        pid=p.pid,ended_utc=now(),stdout_sha256=sha(directory/'stdout.txt'),stderr_sha256=sha(directory/'stderr.txt')))
    if code: raise RuntimeError('child failed; retained consumed work, no retry: '+key)


def run_all(run):
    run=Path(run).resolve();plan=verify(run)
    if not plan['static_passed']: raise RuntimeError('static gate blocked')
    write(run/'execution_started.json',dict(created_utc=now(),no_implicit_resume=True))
    for tid,strategy in plan['search_order']:
        child_command(run,'search_'+tid+'_'+strategy,'search','--task',tid,'--strategy',strategy)
    seal_dev(run)
    for frozen in plan['tasks']:
        child_command(run,'actual_'+frozen['task_id'],'actual','--task',frozen['task_id'])
    write(run/'execution_complete.json',dict(status='COMPLETED',ended_utc=now(),new_training_updates=0))


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','run','search','actual'])
    p.add_argument('--run',type=Path,required=True);p.add_argument('--task');p.add_argument('--strategy',choices=STRATEGIES)
    a=p.parse_args()
    if a.action=='prepare': result=prepare(a.run)
    elif a.action=='run': result=run_all(a.run)
    elif a.action=='search': result=search(a.run,a.task,a.strategy)
    else: result=actual(a.run,a.task)
    print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__': main()
