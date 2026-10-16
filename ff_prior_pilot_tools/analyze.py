"""Read-only saved-array coverage/defect analysis; never run physics or training."""
from pathlib import Path
from collections import Counter
import argparse
import time
import numpy as np
from ff_prior.storage import read_json,atomic_json,file_digest,digest
from ff_prior.initial_state import STATE
from ff_prior.spline import from_arrays
from ff_prior.contracts import MODEL_GATES,all_pass,assert_frozen
from ff_prior.recovery import process_is_alive

ROOT=Path(__file__).resolve().parents[1]


def analyze(source,output):
    source,output=Path(source).resolve(),Path(output).resolve();started=time.time()
    if output==source or source in output.parents:raise ValueError('analysis output must be separate from immutable run')
    state=read_json(source/'status.json');ledger=read_json(source/'ledger.json')
    alive=process_is_alive(state['pid']);terminal=state['status'] in ('FINISHED','FINISHED_WITH_BLOCKED_ITEMS','FAILED') and not alive
    checked=[];pending=[];issues=[];seen=set();references=set();contexts=Counter();shapes=Counter();scales=Counter();durations=Counter();levels=Counter()
    geometries=[];position=[];rotation=[];excursions=[];initials=[];goal_positions=[];families=set();task_ids=set();state_ids=set()
    for p in sorted((source/'records').glob('*.json')):
        if ledger['completed'].get(p.stem,{}).get('status')!='ADMITTED':pending.append(p.stem);continue
        seen.add(p.stem)
        try:
            r=read_json(p);assert_frozen(r['task']);g=read_json(source/r['generation_proof'])
            assert g['admitted'] and g['task_hash']==r['task']['task_hash'] and not r['goal_relabelled']
            assert g['validation_result']['eligible'] and g['validation_result']['validation']==r['validation']
            assert all_pass(r['validation']['gates'],MODEL_GATES)
            assert file_digest(source/r['arrays'])==r['arrays_sha256']
            assert file_digest(source/r['complete_initial_state'])==r['initial_state_sha256']
            with np.load(source/r['arrays']) as a:
                ref=from_arrays(a);assert ref.reference_hash==r['reference_hash']==g['reference_hash']
                assert ref.reference_hash not in references,'duplicate admitted reference'
                references.add(ref.reference_hash)
                for name,order in [('view_q',0),('view_qd',1),('view_qdd',2)]:
                    assert np.allclose(a[name],ref.evaluate(a['view_time'],order),rtol=0,atol=1e-12)
                excursions.append(float(abs(ref.control-ref.q0).max()))
            with np.load(source/r['complete_initial_state']) as a:
                assert digest({'values':a['values'],'model_hash':r['task']['contract']['robot_hash'],'state_mask':int(STATE)})==r['task']['contract']['initial_state_id']
            assert read_json(source/'tasks'/r['sample_id']/'task.json')==r['task']
            checked.append(r['sample_id']);contexts[r['context_id']]+=1;shapes[g['provenance']['proposal_type']]+=1
            scales[str(g['provenance']['motion_scale'])]+=1;durations[str(r['condition']['duration'])]+=1
            families.add(r['family_id']);task_ids.add(r['task']['task_hash']);state_ids.add(r['task']['contract']['initial_state_id'])
            initials.append(r['condition']['initial_features']);goal_positions.append(np.asarray(r['condition']['goals'])[:,:3,3])
            v=r['validation'];geometries.append([v['geometry']['minimum_clearance_m'],v['geometry']['maximum_geometry_step_m']])
            position.append(max(v['frozen_goal_position_error_m']));rotation.append(max(v['frozen_goal_rotation_error_rad']))
            execution=ledger['completed'].get('witness_'+r['sample_id'],{})
            if execution.get('status')=='EXECUTION_VERIFIED' and r['execution_status']=='PASS':levels['EXECUTION_VERIFIED']+=1
            elif r['execution_status']=='FAIL':levels['EXECUTION_FAILED']+=1
            elif r['execution_status']=='NOT_RUN':levels['MODEL_VALID_ONLY']+=1
            else:levels['PENDING_EXECUTION_SNAPSHOT']+=1
        except Exception as error:issues.append({'sample_id':p.stem,'error':repr(error)})
    missing={sid for sid,e in ledger['completed'].items() if e['status']=='ADMITTED'}-seen
    issues.extend({'sample_id':sid,'error':'MISSING_COMMITTED_RECORD'} for sid in sorted(missing))
    rejected=Counter();failed_gates=Counter();reject_contexts=Counter();reject_scales=Counter();pairs=Counter();penetrations=[];generation_errors=[];processed=0;geometry_only_goal_pass=0
    for p in sorted((source/'generation').glob('*.json')):
        if p.stem not in ledger['completed']:continue
        g=read_json(p);processed+=1
        if g['admitted']:continue
        rejected[g.get('reason','UNKNOWN')]+=1;reject_contexts[g.get('context_id','UNKNOWN')]+=1
        reject_scales[str(g.get('provenance',{}).get('motion_scale','UNKNOWN'))]+=1
        v=g.get('validation_result',{}).get('validation',{});failed_gates.update(k for k,val in v.get('gates',{}).items() if not val)
        if {k for k,val in v.get('gates',{}).items() if not val}=={'geometry'} and g.get('validation_result',{}).get('frozen_goal_pass'):
            geometry_only_goal_pass+=1
        geometry=v.get('geometry',{});minimum=geometry.get('minimum_clearance_m')
        if minimum is not None and minimum<0:penetrations.append(-minimum)
        pair=geometry.get('coarse',{}).get('pair',{})
        if pair:pairs[str((pair.get('geom_a'),pair.get('geom_b')))]+=1
        if g['status']=='ERROR':generation_errors.append({'sample_id':g['sample_id'],'reason':g.get('reason')})
    def bounds(values):
        if not len(values):return None
        a=np.asarray(values);return {'minimum':a.min(axis=0).tolist(),'maximum':a.max(axis=0).tolist()}
    result={'status':'PASS_SAVED_ARRAY_AUDIT' if not issues else 'FAILED_SAVED_ARRAY_AUDIT',
        'source':str(source),'started_at':started,'finished_at':time.time(),'run_state_at_start':state,'worker_alive_at_start':alive,
        'terminal_at_start':terminal,'snapshot_not_atomic_across_files':not terminal,'full_run_success_claim':False,
        'committed_saved_rows_checked':len(checked),'pending_uncommitted_records':pending,'issues':issues,
        'validation_levels':dict(levels),'contexts':dict(contexts),'proposal_shapes':dict(shapes),'motion_scales':dict(scales),
        'durations_s':dict(durations),'distinct_family_labels':len(families),'distinct_world_tasks':len(task_ids),'distinct_initial_states':len(state_ids),
        'control_excursion_rad':bounds(excursions),'initial_q17_rad':bounds(initials),'world_goal_positions_m':bounds(goal_positions),
        'saved_geometry_columns':['clearance_m','maximum_step_m'],'saved_geometry':bounds(geometries),
        'max_dual_goal_position_error_m':bounds(position),'max_dual_goal_rotation_error_rad':bounds(rotation),
        'completed_proposals_at_snapshot':processed,'rejections_by_reason':dict(rejected),'failed_gates':dict(failed_gates),
        'geometry_only_rejections_with_frozen_goal_pass':geometry_only_goal_pass,
        'rejections_by_context':dict(reject_contexts),'rejections_by_motion_scale':dict(reject_scales),'rejected_geometry_pairs':dict(pairs),
        'rejected_penetration_depth_m':bounds(penetrations),'generation_errors':generation_errors,
        'extra_physics_calls':0,'extra_training_calls':0,'continuous_time_geometric_certificate_claim':False,
        'new_scene_or_homotopy_claim':False,'note':'Saved MODEL_VALID gates and exact arrays only. Execution levels require individual committed native execution. Family/task/shape counts are coverage descriptors, not statistical independence or useful route diversity.'}
    output.mkdir(parents=True,exist_ok=True);atomic_json(output/'coverage.json',result)
    lines=['# 扩库数据的保存数组与覆盖诊断','',f'数据作业状态：{state["status"]} / {state["stage"]}；开始检查时进程存活：{alive}。',
        f'本次实际核验已提交新数组 {len(checked)} 条；异常 {len(issues)}；新增物理调用0、训练调用0。',
        '若仍运行，本文件只是跨文件实时快照，不是最终同时刻统计或整轮验收。','',
        f'验证等级：{dict(levels)}。不同初态 {len(state_ids)}、不同世界任务 {len(task_ids)}、家族标签 {len(families)}。',
        f'提案形状：{dict(shapes)}；运动倍率：{dict(scales)}；时长秒：{dict(durations)}。',
        f'控制点最大偏移rad范围：{bounds(excursions)}；目标位置误差m范围：{bounds(position)}。',
        f'保存几何列为间距m/最大步距m：{bounds(geometries)}。','',
        f'拒绝原因：{dict(rejected)}；失败门槛：{dict(failed_gates)}。',
        f'仅几何不通过且冻结目标仍通过的拒绝样本：{geometry_only_goal_pass}。origin上下文没有额外添加障碍，不表示没有默认目标卫星等几何。',
        f'拒绝上下文：{dict(reject_contexts)}；拒绝运动倍率：{dict(reject_scales)}。',
        f'拒绝样本穿透深度m范围：{bounds(penetrations)}；几何对：{dict(pairs)}。','',
        '每条核验真实控制点/训练视图一致性、完整初态哈希、冻结世界目标、生成与独立准入证据。未执行者只能称MODEL_VALID；离散几何步距与收敛检查不等同于连续时间严格几何证明。',
        '初态/目标/形状计数不证明新障碍场景、统计独立性或不同绕行同伦，也不证明Diffusion学习收益。',
        f'具体异常：{issues}。']
    (output/'ANALYSIS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',default='artifacts/v7_1/qualified_pilot_001')
    parser.add_argument('--output',default='ff_prior_pilot_tools/evidence/current_snapshot')
    args=parser.parse_args();result=analyze(ROOT/args.source,ROOT/args.output)
    print({k:result[k] for k in ('status','committed_saved_rows_checked','validation_levels','rejections_by_reason','failed_gates','issues')})
    raise SystemExit(0 if not result['issues'] else 1)
