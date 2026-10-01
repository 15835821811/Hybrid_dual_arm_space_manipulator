"""Independent frozen-geometry attribution audit; no producer imports/steps.

Recompute all 1024 B1 queries, direct native geom distances, and 18x7 subset
queries. Existing input/producer artifacts are read-only. Never calls the
producer or minimum_mujoco_geom_clearance wrapper. Timing covers saved subset
query samples only and is not a control-cycle or wall qualification result.
"""
from __future__ import annotations
import argparse
from collections import Counter
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import traceback

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from v6_lite.continuum_shape_model import ContinuumShapeModel
from v6_lite.pcc_bounded_clearance import PCCBoundedClearanceEvaluator
from v6_lite.pcc_batched_distance_query import BatchedDistanceDecisionQuery
from v6_lite.pcc_interval_cbf import IntervalPartition
from v6_lite.pcc_clearance import V61A_PCC_TUBE_RADII_M
from v6_lite.run_v6_lite import default_v6_lite_robot_spec
from v6_lite.runtime_command import model_id
from v6_lite.shape_clearance import OrientedBox

SAFE, BELOW, UNKNOWN = 'PROXY_CLEARANCE_AT_LEAST_GATE', 'PROXY_CLEARANCE_BELOW_GATE', 'UNKNOWN_CROSSES_GATE'
GATE, TOL, PAD = .005, .001, 1e-9
VARIANTS = [('B1_distance_bounds', b, None) for b in (255, 510, 1020)] + [
    ('BatchedDistanceDecisionQuery', b, l) for b, l in ((255,128),(255,512),(510,512),(1020,512))]
EXPECTED_COMMIT = '2d76555d8078342bdf28ae56933deb74c80aca96'
EXPECTED_INPUT = 'ec620f3396bc06f98597e76270501a1bd7a88a941e5740239bb10d775d3efe8c'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines()]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    with Path(path).open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=True, allow_nan=False)
        stream.write('\n')


def finite(value):
    return isinstance(value, (int,float)) and not isinstance(value,bool) and math.isfinite(value)


def near(a,b,tol=1e-12):
    return finite(a) and finite(b) and abs(a-b)<=tol


def status(lower,upper):
    if not finite(lower) or not finite(upper) or lower>upper:
        raise ValueError('invalid bounds')
    return SAFE if lower>=GATE else BELOW if upper<GATE else UNKNOWN


def same(a,b):
    if isinstance(b,dict):
        return isinstance(a,dict) and a.keys()==b.keys() and all(same(a[k],v) for k,v in b.items())
    if isinstance(b,list):
        return isinstance(a,list) and len(a)==len(b) and all(same(x,y) for x,y in zip(a,b))
    if isinstance(b,float):
        return near(a,b)
    return a==b


def untimed(value):
    if isinstance(value,dict):
        return {key:untimed(part) for key,part in value.items() if key!='elapsed_ms'}
    if isinstance(value,list):
        return [untimed(x) for x in value]
    return value


def coverage_b1(parts,lengths):
    if not parts:
        return False
    for i,length in enumerate(lengths):
        selected=sorted((p['local_start_m'],p['local_end_m']) for p in parts if p['segment_id']==i)
        if not selected or not near(selected[0][0],0) or not near(selected[-1][1],float(length)):
            return False
        if any(not finite(a) or not finite(b) or not 0<=a<b<=length+1e-12 for a,b in selected):
            return False
        if any(not near(a[1],b[0]) for a,b in zip(selected,selected[1:])):
            return False
    return len(parts)==sum(p['segment_id'] in range(5) for p in parts)


def coverage_online(ids):
    grouped={i:[] for i in range(5)}
    if len(set(ids))!=len(ids):
        return False
    for name in ids:
        prefix,seg,path=name.split(':')
        i=int(seg)
        if prefix!='pcc_interval' or i not in grouped:
            return False
        if path=='root': path=''
        if any(x not in '01' for x in path): return False
        n=int(path,2) if path else 0
        grouped[i].append((Fraction(n,1<<len(path)),Fraction(n+1,1<<len(path))))
    for selected in grouped.values():
        selected.sort()
        if not selected or selected[0][0]!=0 or selected[-1][1]!=1 or any(a[1]!=b[0] for a,b in zip(selected,selected[1:])):
            return False
    return True


def distribution(values):
    x=np.asarray(values,dtype=float)
    if not len(x) or not np.all(np.isfinite(x)):
        raise ValueError('invalid statistic sample')
    return {'count':len(x),'min':float(x.min()),'p50':float(np.percentile(x,50)),
            'p95':float(np.percentile(x,95)),'max':float(x.max())}


def run(trial, output):
    plan, report, provenance = [read(trial/name) for name in ('plan.json','report.json','source_provenance.json')]
    baseline, refined = rows(trial/'baseline.jsonl'), rows(trial/'refinement.jsonl')
    selections=read(trial/'selections.json')
    checks=[]
    def check(name,passed,detail=None):
        checks.append({'name':name,'passed':bool(passed),'detail':detail})
    check('producer_complete_and_valid',report['complete'] is True and report['evidence_valid'] is True
          and all(v is True for v in report['checks'].values()))
    manifest=read(trial/'artifact_manifest.json')
    entries={entry['path']:entry for entry in manifest['artifacts']}
    actual_names={path.name for path in trial.iterdir() if path.is_file() and path.name!='artifact_manifest.json'}
    check('all_artifact_names_hashes_sizes',set(entries)==actual_names and len(entries)==len(manifest['artifacts'])
          and all(sha(trial/name)==e['sha256'] and (trial/name).stat().st_size==e['bytes'] for name,e in entries.items()))
    expected_input=ROOT/'v6_lite/output/v6_2_b1/formal_audit_r02/independent_heldout_inputs.npz'
    old_b1=read(expected_input.parent/'bounded_clearance_audit.json')
    old=read(ROOT/'v6_lite/output/v6_2_b2/heldout_geometry/heldout_geometry_report.json')
    input_manifest=read(expected_input.parent/'audit_manifest.json')
    check('frozen_original_input_manifest',sha(expected_input)==EXPECTED_INPUT==plan['frozen_input']['sha256']
          and expected_input.stat().st_size==173433==plan['frozen_input']['bytes']
          and all(sha(expected_input.parent/e['path'])==e['sha256']
                  and (expected_input.parent/e['path']).stat().st_size==e['bytes'] for e in input_manifest['artifacts']))
    for name in ('frozen_input','historical_report','historical_b1','input_manifest'):
        path=Path(plan[name]['path'])
        check('original_utf8_path_and_hash:'+name,path.is_file() and sha(path)==plan[name]['sha256'],ascii(str(path)))
    check('frozen_b1_numerical_contract',plan['gate_m']==GATE==old_b1['query']['safety_gate_m']
          and plan['tolerance_m']==TOL==old_b1['query']['tolerance_m']
          and plan['numerical_pad_m']==PAD==old_b1['query']['numerical_pad_m']
          and plan['tube_radii_m']==old_b1['query']['tube_radii_m']==V61A_PCC_TUBE_RADII_M.tolist()
          and plan['baseline']=={'count':1024,'B1_max_evaluations':31}
          and old_b1['query']['max_evaluations']==31 and plan['query_distance_max_m']==.5
          and plan['b1_subset_max_evaluations']==[255,510,1020]
          and plan['online_subset_point_leaf_budgets']==[[255,128],[255,512],[510,512],[1020,512]])
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    tracked=subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=ROOT,text=True).strip()
    check('trial_commit_and_source_unchanged',report['git_commit']==plan['source']['git_commit']
          ==provenance['before']['git_commit']==provenance['after']['git_commit']==EXPECTED_COMMIT==commit
          and not tracked and provenance['source_unchanged'] is True and provenance['capture_complete'] is True
          and not provenance['before']['capture_errors'] and not provenance['after']['capture_errors']
          and provenance['before']['files']==provenance['after']['files']==plan['source']['files'])
    source_hashes={name:sha(ROOT/name) for name in plan['source']['files']}
    check('raw_source_files_and_producer_snapshot',all(source_hashes[name]==entry['sha256_raw']
          for name,entry in plan['source']['files'].items())
          and sha(trial/'producer.py')==sha(ROOT/'v6_lite/audit_research_conservatism.py'))
    robot=default_v6_lite_robot_spec()
    shape=ContinuumShapeModel()
    spec=shape.spec
    check('model_contract_current_plan_historical_b1',spec.contract_sha256()==plan['model_contract_sha256']
          ==old_b1['model_contract_sha256']==plan['historical_b1']['model_contract_sha256']
          and same(spec.to_dict(),plan['model_contract']))
    model=robot.compile_dynamic_model()
    data=mujoco.MjData(model)
    target_id=int(mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_GEOM,'target_satellite_collision'))
    target_joint=int(mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,robot.target_free_joint_name))
    target_addr=int(model.jnt_qposadr[target_joint])
    qpos_ids=np.asarray([model.jnt_qposadr[mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,name)]
                        for name in robot.low_level_joint_names],dtype=int)
    check('nominal_compiled_model',model_id(model,spec.contract_sha256())==plan['nominal_compiled_model_sha256'])
    root_body=int(mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,spec.mount_body_name))
    current=int(mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,spec.end_effector_body_name))
    body_ids=set()
    while current!=0:
        body_ids.add(current)
        if current==root_body: break
        current=int(model.body_parentid[current])
    geom_ids=[i for i in range(model.ngeom) if int(model.geom_bodyid[i]) in body_ids
        and int(model.geom_bodyid[i])!=root_body and (int(model.geom_contype[i]) or int(model.geom_conaffinity[i])
        or (mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_GEOM,i) or '').startswith('collision_'))]
    geom_names=[mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_GEOM,i) for i in geom_ids]
    check('independent_reference_geometry_policy',root_body in body_ids and set(geom_names)==set(plan['mujoco_reference_geom_names']))
    with np.load(expected_input,allow_pickle=False) as archive:
        inputs={key:archive[key].copy() for key in archive.files}
    check('frozen_layout_finite_and_original_seed',inputs['configurations'].shape==(1024,10)
          and inputs['centers'].shape==(1024,3) and inputs['rotations'].shape==(1024,3,3)
          and inputs['half_extents'].shape==(1024,3) and int(inputs['seed'])==20260929
          and all(np.all(np.isfinite(value)) for value in inputs.values()))
    bounded=PCCBoundedClearanceEvaluator(shape)
    online=BatchedDistanceDecisionQuery(shape)
    actual_rows=[]; query_rows=[]
    check('all_1024_baseline_indices_once',[row['index'] for row in baseline]==list(range(1024)))
    for index,row in enumerate(baseline):
        q=inputs['configurations'][index]
        box=OrientedBox(inputs['centers'][index],inputs['rotations'][index],inputs['half_extents'][index])
        model.geom_size[target_id]=box.half_extents
        data.qpos[:]=model.qpos0
        planner=robot.planner_zero.copy(); planner[:10]=q
        data.qpos[qpos_ids]=robot.encode_position(planner)
        local_flat=np.empty(9)
        mujoco.mju_quat2Mat(local_flat,model.geom_quat[target_id])
        body_rotation=box.rotation@local_flat.reshape(3,3).T
        quaternion=np.empty(4)
        mujoco.mju_mat2Quat(quaternion,body_rotation.reshape(-1))
        data.qpos[target_addr:target_addr+3]=box.center-body_rotation@model.geom_pos[target_id]
        data.qpos[target_addr+3:target_addr+7]=quaternion
        mujoco.mj_forward(model,data)
        distances=[]
        for geom_id in geom_ids:
            witness=np.empty(6)
            raw=float(mujoco.mj_geomDistance(model,data,geom_id,target_id,.5,witness))
            if not math.isfinite(raw): raise ValueError('nonfinite native geometry distance')
            distances.append((.5 if raw>=.5-1e-12 else raw,geom_id,raw>=.5-1e-12))
        measured,geom_id,truncated=min(distances,key=lambda x:x[0])
        actual_ok=near(measured,row['actual_mujoco_distance_m']) and truncated==row['actual_query_truncated']
        actual_ok=actual_ok and mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_GEOM,geom_id)==row['actual_source_geom']
        actual_ok=actual_ok and model_id(model,spec.contract_sha256())==row['compiled_case_model_sha256']
        difference={'actual_minus_proxy_lower_m':(.5-1e-12 if truncated else measured)-row['upper_m'],
            'actual_minus_proxy_upper_m':None if truncated else measured-row['lower_m'],
            'interpretation':'CENSORED_LOWER_BOUND' if truncated else 'BOUNDED_DIFFERENCE'}
        actual_ok=actual_ok and same(difference,row['discrepancy'])
        actual_rows.append({'index':index,'passed':actual_ok,'actual_direct_m':measured,
            'query_truncated':truncated,'closest_geom':mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_GEOM,geom_id),
            'actual_error_m':measured-row['actual_mujoco_distance_m'],'discrepancy':difference})
        result=bounded.evaluate(q,np.eye(4),box,max_evaluations=31,safety_gate_m=GATE,
                                tolerance_m=TOL,numerical_pad_m=PAD,local_refinement_count=0)
        query_ok=same(untimed(row['query_result']),untimed(result.to_dict()))
        query_ok=query_ok and status(row['lower_m'],row['upper_m'])==row['proxy_status']
        query_ok=query_ok and near(row['gap_m'],row['upper_m']-row['lower_m'])
        query_ok=query_ok and row['bounds_valid'] is True and row['evaluation_count']<=31
        query_ok=query_ok and coverage_b1(row['query_result']['candidate_intervals'],spec.segment_lengths_m)
        query_ok=query_ok and row['tolerance_met']==(row['gap_m']<=TOL)
        query_ok=query_ok and row['budget_exhausted']==(not row['tolerance_met'] and row['evaluation_count']+2>31)
        query_rows.append({'index':index,'family':'baseline_B1','passed':query_ok})
        if (index+1)%128==0: print(f'[independent] native geometry and B1 {index+1}/1024',flush=True)
    check('all_direct_1024_native_distances_witnesses_and_difference_bounds',all(row['passed'] for row in actual_rows),
          {'count':len(actual_rows),'maximum_distance_error_m':max(abs(row['actual_error_m']) for row in actual_rows),
           'censored_count':sum(row['query_truncated'] for row in actual_rows)})
    check('all_baseline_B1_nontiming_partition_and_classification',all(row['passed'] for row in query_rows))
    unknown=[row['index'] for row in baseline if status(row['lower_m'],row['upper_m'])==UNKNOWN]
    false_reject=[row['index'] for row in baseline if status(row['lower_m'],row['upper_m'])!=SAFE and row['actual_mujoco_distance_m']>=GATE]
    geometric=[row['index'] for row in baseline if status(row['lower_m'],row['upper_m'])==BELOW and row['actual_mujoco_distance_m']>=GATE]
    check('all_selection_sets_from_raw_rows',selections=={'unknown_indices':unknown,'false_reject_indices':false_reject,
        'proven_proxy_below_actual_safe_indices':geometric,'high_budget_NOT_RUN_indices':[i for i in range(1024) if i not in unknown]}
        and len(unknown)==18 and all(baseline[i]['high_budget_status']==('PENDING_UNKNOWN_SUBSET' if i in unknown else 'NOT_RUN') for i in range(1024)))
    population=Counter({'checked_count':len(baseline),SAFE:0,BELOW:0,UNKNOWN:0,'bounds_invalid':0,
        'actual_geometry_safe':0,'actual_geometry_below_gate':0,'empirical_proxy_false_safe':0,'empirical_proxy_false_reject':0})
    table={s:{'actual_safe':0,'actual_below_gate':0} for s in (SAFE,BELOW,UNKNOWN)}
    for row in baseline:
        s=status(row['lower_m'],row['upper_m']); good=row['actual_mujoco_distance_m']>=GATE
        population[s]+=1; population['actual_geometry_safe' if good else 'actual_geometry_below_gate']+=1
        population['bounds_invalid']+=int(not row['bounds_valid'])
        population['empirical_proxy_false_safe']+=int(s==SAFE and not good)
        population['empirical_proxy_false_reject']+=int(s!=SAFE and good)
        table[s]['actual_safe' if good else 'actual_below_gate']+=1
    check('population_counts_and_actual_status_cross_table',dict(population)==report['baseline_counts']==old['counts']
        and table==report['status_by_actual_gate'] and len(geometric)==report['actual_safe_proxy_below_count']==172
        and table[UNKNOWN]['actual_safe']==report['actual_safe_proxy_unknown_count']==18)
    keys=[(row['index'],row['family'],row['point_budget'],row['leaf_budget']) for row in refined]
    check('all_18_by_7_combinations_exactly_once',len(keys)==126==len(set(keys))
          and set(keys)=={(index,*variant) for index in unknown for variant in VARIANTS})
    recompute_rows=[]
    for row in refined:
        index=row['index']; q=inputs['configurations'][index]
        box=OrientedBox(inputs['centers'][index],inputs['rotations'][index],inputs['half_extents'][index])
        if row['family']=='B1_distance_bounds':
            result=bounded.evaluate(q,np.eye(4),box,max_evaluations=row['point_budget'],safety_gate_m=GATE,
                                    tolerance_m=TOL,numerical_pad_m=PAD,local_refinement_count=0)
            ok=same(untimed(row['query_result']),untimed(result.to_dict()))
            ok=ok and coverage_b1(row['query_result']['candidate_intervals'],spec.segment_lengths_m)
            ok=ok and row['point_count']==result.evaluation_count and row['leaf_count']==result.interval_count
            ok=ok and row['tolerance_met']==(row['gap_m']<=TOL)
            ok=ok and row['budget_exhausted']==(not row['tolerance_met'] and row['point_count']+2>row['point_budget'])
        else:
            result=online.evaluate(q,np.eye(4),box,IntervalPartition.uniform(),gate_m=GATE,
                max_point_evaluations=row['point_budget'],max_leaves=row['leaf_budget'],numerical_pad_m=PAD)
            ids=[leaf.interval_id for leaf in result.partition.leaves]
            ok=ids==row['partition_ids'] and coverage_online(ids) and row['coverage_complete'] is True
            ok=ok and same(result.lower_by_interval_id,row['lower_by_interval_id'])
            ok=ok and same(result.midpoint_upper_by_interval_id,row['midpoint_upper_by_interval_id'])
            ok=ok and row['point_count']==result.point_evaluation_count and row['leaf_count']==result.interval_count
            ok=ok and row['split_count']==result.split_count and row['budget_exhausted']==result.budget_exhausted
            ok=ok and row['failure_reason']==result.failure_reason and row['leaf_count']<=row['leaf_budget']
            ok=ok and row['point_count']==5+2*row['split_count'] and row['leaf_count']==5+row['split_count']
        ok=ok and near(row['lower_m'],result.distance_lower_bound_m) and near(row['upper_m'],result.distance_upper_bound_m)
        ok=ok and near(row['gap_m'],result.bound_gap_m) and row['bounds_valid'] is True and row['point_count']<=row['point_budget']
        ok=ok and status(row['lower_m'],row['upper_m'])==row['proxy_status']==result.proxy_clearance_status
        recompute_rows.append({'index':index,'family':row['family'],'point_budget':row['point_budget'],
            'leaf_budget':row['leaf_budget'],'passed':ok,'recomputed_status':result.proxy_clearance_status})
    check('all_126_subset_nontiming_queries_partitions_limits_and_statuses',all(row['passed'] for row in recompute_rows))
    summaries=[]
    for family,budget,leaf_cap in VARIANTS:
        selected=[row for row in refined if (row['family'],row['point_budget'],row['leaf_budget'])==(family,budget,leaf_cap)]
        source_summary=next(s for s in report['subset_variants'] if (s['family'],s['point_budget'],s['leaf_budget'])==(family,budget,leaf_cap))
        transition={s:sum(status(row['lower_m'],row['upper_m'])==s for row in selected) for s in (SAFE,BELOW,UNKNOWN)}
        resolved=[row['index'] for row in selected if row['proxy_status']==SAFE]
        below=[row['index'] for row in selected if row['proxy_status']==BELOW]
        unresolved=[row['index'] for row in selected if row['proxy_status']==UNKNOWN]
        timing=distribution([row['elapsed_ms'] for row in selected])
        check(f'subset_summary:{family}:{budget}:{leaf_cap}',dict(Counter(row['proxy_status'] for row in selected))==source_summary['status_counts']
            and source_summary['sample_count']==18 and source_summary['resolved_safe_indices']==resolved
            and source_summary['resolved_below_indices']==below and source_summary['remaining_unknown_indices']==unresolved
            and same(source_summary['point_count'],distribution([row['point_count'] for row in selected]))
            and same(source_summary['leaf_count'],distribution([row['leaf_count'] for row in selected]))
            and same(source_summary['gap_m'],distribution([row['gap_m'] for row in selected]))
            and same(source_summary['query_elapsed_ms'],timing)
            and source_summary['timing_population']=='baseline UNKNOWN subset only, first sample included')
        summaries.append({'family':family,'point_budget':budget,'leaf_budget':leaf_cap,'sample_count':18,
            'UNKNOWN_transitions':transition,'resolved_false_reject_count':sum(baseline[i]['actual_mujoco_distance_m']>=GATE for i in resolved),
            'tolerance_met_count':sum(row.get('tolerance_met',False) for row in selected) if family=='B1_distance_bounds' else None,
            'budget_exhausted_count':sum(row['budget_exhausted'] for row in selected),
            'online_point_limit_hit_count':sum(row['budget_exhausted'] and row['point_count']+2>budget for row in selected) if leaf_cap else None,
            'online_leaf_limit_hit_count':sum(row['budget_exhausted'] and row['leaf_count']+1>leaf_cap for row in selected) if leaf_cap else None,
            'online_failure_reason_counts':dict(Counter(str(row.get('failure_reason')) for row in selected)) if leaf_cap else None,
            'stopping_reason':'requested_1mm_distance_tolerance_reached' if family=='B1_distance_bounds' else 'all_decisions_resolved_before_point_or_leaf_limits',
            'point_count':distribution([row['point_count'] for row in selected]),'leaf_count':distribution([row['leaf_count'] for row in selected]),
            'gap_m':distribution([row['gap_m'] for row in selected]),'original_subset_query_elapsed_ms':timing,
            'resolved_safe_indices':resolved,'resolved_below_indices':below,'remaining_unknown_indices':unresolved})
    check('offline_claim_boundaries_preserved',report['online_controller_changed'] is False and report['closed_loop_claim'] is False
        and report['hardware_claim'] is False and plan['wall_deployment_status']=='NOT_MET')
    for name,data_rows in (('actual_direct_checks.jsonl',actual_rows),('baseline_query_checks.jsonl',query_rows),
                           ('subset_query_checks.jsonl',recompute_rows)):
        with (output/name).open('x',encoding='utf-8',newline='\n') as stream:
            for row in data_rows: stream.write(json.dumps(row,ensure_ascii=True,allow_nan=False)+'\n')
    write(output/'by_item_checks.json',checks)
    write(output/'source_inventory.json',{'source_trial_commit':EXPECTED_COMMIT,'auditor_source_commit':commit,
        'auditor_script_sha256_raw':sha(__file__),'current_source_raw_sha256':source_hashes})
    failed=[row for row in checks if not row['passed']]
    return {'schema':'research_conservatism_independent_audit_v1','evidence_valid':not failed,'checks_passed':len(checks)-len(failed),
        'checks_total':len(checks),'failures':failed,'source_trial_commit':EXPECTED_COMMIT,'auditor_source_commit':commit,
        'auditor_script_sha256_raw':sha(__file__),'producer_report_sha256':sha(trial/'report.json'),
        'producer_manifest_sha256':sha(trial/'artifact_manifest.json'),'frozen_input_sha256':EXPECTED_INPUT,
        'direct_native_geometry_count':1024,'B1_baseline_queries_recomputed':1024,'subset_queries_recomputed':126,
        'baseline_counts':dict(population),'status_by_actual_gate':table,
        'baseline_tolerance_met_count':sum(row['tolerance_met'] for row in baseline),
        'baseline_budget_exhausted_count':sum(row['budget_exhausted'] for row in baseline),
        'baseline_censored_count':sum(row['query_truncated'] for row in actual_rows),
        'subset_variants':summaries,'geometric_false_reject_count':172,'baseline_unknown_actual_safe_count':18,
        'same_proxy_approval_upper_bound_actual_safe':53,'same_proxy_approval_upper_bound_denominator':225,
        'upper_bound_scope':'mathematical attribution upper bound from 172 proven proxy-below cases; not a full-population online run',
        'censored_difference_rule':{'lower':'.5 - 1e-12 - proxy_upper','upper':None},
        'timing_scope':'saved 18-case UNKNOWN subset observations only; no control-cycle or population-online timing claim',
        'geometry_scope':'declared distal continuum collision geoms against frozen target OBBs; no whole-body closed-loop claim',
        'wall_deployment_status':'NOT_MET','hard_realtime_certified':False,'continuous_time_certified':False,
        'online_controller_changed':False,'closed_loop_claim':False,'hardware_claim':False}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trial-dir',type=Path,required=True)
    args=parser.parse_args()
    output=Path(__file__).resolve().parent
    if (output/'report.json').exists(): raise FileExistsError('preserve existing audit')
    trial=args.trial_dir.resolve()
    write(output/'audit_plan.json',{'source_trial_commit':EXPECTED_COMMIT,'auditor_script_sha256_raw':sha(__file__),
        'trial_directory':str(trial),'trial_manifest_sha256':sha(trial/'artifact_manifest.json'),
        'recompute_scope':{'direct_mj_geomDistance':1024,'B1_baseline':1024,'subset_variants':126},
        'minimum_wrapper_used':False,'producer_imported':False,'mj_step_calls':0})
    try: result=run(trial,output)
    except Exception as error:
        result={'schema':'research_conservatism_independent_audit_v1','evidence_valid':False,
            'auditor_script_sha256_raw':sha(__file__),'error':{'type':type(error).__name__,'message':str(error),'traceback':traceback.format_exc()}}
    result['completed_utc']=datetime.now(timezone.utc).isoformat()
    write(output/'report.json',result)
    write(output/'artifact_manifest.json',{'artifacts':[{'path':path.name,'sha256':sha(path),'bytes':path.stat().st_size}
        for path in sorted(output.iterdir()) if path.is_file() and path.name!='artifact_manifest.json']})
    print(json.dumps({k:result.get(k) for k in ('evidence_valid','checks_passed','checks_total','failures','error')},indent=2),flush=True)
    raise SystemExit(0 if result['evidence_valid'] else 1)


if __name__=='__main__': main()
