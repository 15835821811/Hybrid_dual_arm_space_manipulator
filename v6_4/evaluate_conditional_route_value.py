"""Collect the declared negative pilot terminal; never run physics or sampling."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'v6_4/output/conditional_route_value_20261007_01'

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def write(path, data):
    with Path(path).open('x', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=2, allow_nan=False)
        file.write('\n')

def collect():
    decision = read(OUT/'pilot/decision.json')
    assert decision['route_value_identifiable'] is False
    assert decision['training_authorized_by_pilot'] is False
    records = read(OUT/'pilot/quality_records.json')
    plan = read(OUT/'plan.json')
    assert len(records) == len(plan['pilot_slots']) == 6
    probe = read(OUT/'old_checkpoint_probe/report.json')
    assert probe['status'] == 'COMPLETED' and probe['ddim_calls'] == 32
    identity = read(OUT/'source_identity.json')
    attempts, ledgers = [], []
    rows = []
    for declared, quality in zip(plan['pilot_slots'], records):
        directory = OUT/'pilot/attempts'/declared['slot_id']
        attempt, ledger = read(directory/'attempt_result.json'), read(directory/'cost_ledger.json')
        assert attempt['slot_id'] == quality['slot_id'] == declared['slot_id']
        assert attempt['source_identity_sha256'] == sha(OUT/'source_identity.json')
        assert ledger['actual_physics_steps'] == attempt['actual_steps']
        for source, digest in quality['sources'].items():
            assert sha(source) == digest, source
        attempts.append(attempt)
        ledgers.append(ledger)
        full = quality['full_metrics'] or {}
        prefix = quality['failed_prefix_metrics'] or {}
        clearance = full.get('continuum_route_obstacle_clearance') or {}
        rows.append({'slot_id': attempt['slot_id'], 'task_id': attempt['task_id'],
                     'method': declared['candidate_name'], 'status': attempt['status'],
                     'actual_steps': attempt['actual_steps'], 'saved_horizon_s': attempt['actual_steps']*.002,
                     'full_task_success': attempt['full_task_success'],
                     'I_route_rad_s': full.get('I_route_rad_s'), 'I_full_rad_s': full.get('I_full_rad_s'),
                     'continuum_path_length_m': full.get('continuum_path_length_m'),
                     'route_clearance_m': clearance.get('route_window_minimum_m'),
                     'full_clearance_m': clearance.get('full_saved_horizon_minimum_m'),
                     'base_translation_peak_m': full.get('base_translation_peak_m'),
                     'base_rotation_peak_rad': full.get('base_rotation_peak_rad'),
                     'failed_prefix_metrics': prefix or None,
                     'metric_unavailable': quality.get('metric_unavailable'),
                     'execution_failure': attempt.get('execution_failure'),
                     'evaluation_errors': (attempt.get('evaluation') or {}).get('errors'),
                     'safety': quality['safety']})
    cost = {name: sum(l[name] for l in ledgers) for name in (
        'actual_physics_steps', 'private_preview_physics_steps',
        'independent_saved_torque_replay_steps', 'native_geometry_query_calls', 'preview_calls')}
    cost['additional_route_quality_geometry_queries'] = sum(r['costs']['additional_route_quality_geometry_queries'] for r in records)
    cost['input_precheck_geometry_queries'] = read(OUT/'route_pair_inputs.json')['geometry_query_count']
    cost.update(actual_slots_consumed=6, actual_runner_entries=sum(a.get('actual_runner_started') is True for a in attempts),
                slots_with_physics=sum(a['actual_steps']>0 for a in attempts),
                old_checkpoint_ddim_calls=32, teacher_actual_slots=0, new_model_ddim_calls=0,
                new_training_runs=0, optimizer_updates=0, new_TEST_actual_slots=0,
                count_scope='actual, private previews and same-torque independent replays are distinct; geometry queries are not independent experiments')
    complete = sum(a['full_task_success'] is True for a in attempts)
    comparisons = []
    for side in decision['sides']:
        local = {r['method']: r for r in rows if r['task_id'] == side['task_id']}
        nonzero = [r for name,r in local.items() if name != 'z0' and r['I_route_rad_s'] is not None]
        best = min(nonzero, key=lambda r:r['I_route_rad_s']) if nonzero else None
        zero = local['z0']
        change = None
        if best and zero['I_route_rad_s'] is not None:
            change = 100*(best['I_route_rad_s']-zero['I_route_rad_s'])/zero['I_route_rad_s'] if zero['I_route_rad_s'] else None
        comparisons.append({'task_id': side['task_id'], 'zero_complete_safe': zero['full_task_success'],
                            'best_complete_safe_nonzero': best['method'] if best else None,
                            'best_nonzero_vs_zero_I_change_percent': change,
                            'preferred_nonzero_direction': side['preferred_nonzero_direction'],
                            'predeclared_side_gate': side})
    pairs = {'schema': 'v64_b3_paired_metrics_v1', 'plan_sha256': sha(OUT/'plan.json'),
             'pilot': {'task_count': 2, 'mother_scene_count': 1, 'slots': rows, 'decision': decision,
                       'complete_safe_nonzero_vs_zero': comparisons},
             'new_independent_TEST': {'status': 'NOT_RUN_PILOT_STOP', 'success_by_method': {m: None for m in plan['TEST']['methods']}},
             'old_checkpoint_condition_probe': {k: v for k, v in probe.items() if k not in (
                 'records', 'same_noise_between_obstacle_conditions', 'different_noise_within_obstacle_condition')},
             'costs': cost}
    write(OUT/'paired_metrics.json', pairs)
    if complete == 0:
        diagnosis = 'NO_COMPLETE_SAFE_NONLEARNING_EXECUTABILITY_WITNESS_IN_FIXED_PILOT'
        answer1 = '六个固定槽均未建立完整安全成功的非学习见证；无法在完整任务上确认零残差已足够，也无法证明非零残差带来收益。'
    else:
        diagnosis = 'PREDECLARED_PAIRED_ROUTE_VALUE_GATE_NOT_MET'
        detail = '；'.join(f"{c['task_id']}较好非零{c['best_complete_safe_nonzero']}相对零残差I_route变化{c['best_nonzero_vs_zero_I_change_percent']:+.3f}%"
                          for c in comparisons if c['best_nonzero_vs_zero_I_change_percent'] is not None)
        answer1 = detail+'。完整成功记录中未达到冻结的双侧A/B路线价值门槛。安全通过和非零运动变化不能代替可辨识的路线收益。'
    preference_text = '；'.join(f"{c['task_id']}: {c['preferred_nonzero_direction'] or '无可分辨的非零偏好'}" for c in comparisons)
    reversed_ordinal = all(c['preferred_nonzero_direction'] for c in comparisons) and comparisons[0]['preferred_nonzero_direction'] != comparisons[1]['preferred_nonzero_direction']
    ordinal_text = '完整合格非零候选的数值排名随障碍换边反转' if reversed_ordinal else '完整合格非零候选未显示相反方向的数值排名'
    questions = [
        {'question': '零残差已经足够时，残差是否有可测收益？', 'answer': answer1},
        {'question': '障碍换边后，正确路线是否随之改变？', 'answer': preference_text+'。'+ordinal_text+'，但未建立满足冻结门槛的双侧相反有利方向；数值排名与达到可辨识价值门槛分别报告。'},
        {'question': '真实条件是否比错条件/无条件经验抽样更好？', 'answer': '未评价新模型的五组TEST。旧权重P0仅观察到条件响应，缺少可信偏好标签，不能据此称正确适应。'},
        {'question': '与简单检索相比是否值得增加Diffusion？', 'answer': '本轮没有建立新增Diffusion的收益证据；维持B.2的非学习检索基线，不能将未运行对照写成检索胜出。'},
        {'question': '负结果下一步指向任务区分度、数据还是方法？', 'answer': '先解决固定任务族中的路线可区分性与非学习可执行性见证，再讨论质量数据或学习方法。本轮停止，不追加障碍搜索、幅值、种子或网络。'}]
    report = {'schema': 'v64_b3_conditional_route_value_report_v1', 'created_utc': datetime.now(timezone.utc).isoformat(),
              'run_id': OUT.name, 'source_producer_commit': identity['algorithm_producer_commit'],
              'B2_algorithm_producer': plan['B2_algorithm_producer'], 'published_B2_base': plan['published_B2_base'],
              'plan_sha256': sha(OUT/'plan.json'), 'source_identity_sha256': sha(OUT/'source_identity.json'),
              'status': 'ROUTE_VALUE_NOT_IDENTIFIABLE_WITHIN_CURRENT_REPRESENTATION',
              'research_execution_complete': True, 'research_delivery_complete': False,
              'route_value_identifiable': False, 'training_executed': False,
              'condition_response_observed': probe['same_noise_pairs_observed']>0,
              'conditional_value_supported_in_pilot': False, 'advantage_over_retrieval': 'NOT_EVALUATED',
              'independent_test_task_success_by_method': {m: 'NOT_RUN_PILOT_STOP' for m in plan['TEST']['methods']},
              'pilot_full_task_success': {'numerator': complete, 'denominator': 6},
              'diagnosis': diagnosis, 'decision': decision, 'five_questions': questions, 'costs': cost,
              'not_run': {'teacher': 'PILOT_NEGATIVE_STOP', 'training': 'PILOT_NEGATIVE_STOP',
                          'new_model_sampling': 'PILOT_NEGATIVE_STOP', 'new_TEST': 'PILOT_NEGATIVE_STOP', 'K4_actual': 'NOT_RUN'},
              'deployment': 'NOT_MET', 'continuous_time_certified': False,
              '20ms_wall_is_research_gate': False, 'timing_scope': 'proposal/precheck generation in frozen inputs; existing instrumented dispatch timelines retained; no new-method timing because P2/P3/P4 NOT_RUN',
              'I_route_scope': plan['route_quality'], 'old_B2_conclusions_rewritten': False,
              'interpretation': 'Negative finite study of one paired mother scene and the existing 20mm residual representation; not a refutation of Diffusion.',
              'sources': {'paired_metrics.json': sha(OUT/'paired_metrics.json'),
                          'pilot/quality_records.json': sha(OUT/'pilot/quality_records.json'),
                          'pilot/decision.json': sha(OUT/'pilot/decision.json'),
                          'old_checkpoint_probe/report.json': sha(OUT/'old_checkpoint_probe/report.json')}}
    write(OUT/'report.json', report)
    fields = ['slot_id','task_id','method','status','actual_steps','saved_horizon_s','full_task_success',
              'I_route_rad_s','I_full_rad_s','continuum_path_length_m','route_clearance_m','full_clearance_m',
              'base_translation_peak_m','base_rotation_peak_rad']
    with (OUT/'method_table.csv').open('x', encoding='utf-8', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows({k:r[k] for k in fields} for r in rows)
    def number(value):
        return '—' if value is None else f'{value:.9g}'
    table = ['| 槽 | 任务 | 固定候选 | 状态 | 保存时长/s | 完整合格 | I_route/(rad/s) | 相关窗口净空/mm |',
             '|---|---|---|---|---:|---|---:|---:|']
    for row in rows:
        table.append(f"| {row['slot_id']} | {row['task_id']} | {row['method']} | {row['status']} | {row['saved_horizon_s']:.3f} | {row['full_task_success']} | {number(row['I_route_rad_s'])} | {number(None if row['route_clearance_m'] is None else row['route_clearance_m']*1000)} |")
    content = '# V6.4-B.3 固定配对路线价值研究\n\n'
    content += '终态：`ROUTE_VALUE_NOT_IDENTIFIABLE_WITHIN_CURRENT_REPRESENTATION`。按预声明停止规则结束P1，P2/P3/P4均未运行。有限负结果不否定Diffusion。\n\n'
    content += f"实际执行producer：`{identity['algorithm_producer_commit']}`；从已发布B.2 `{plan['published_B2_base']}` 派生。旧B.2算法producer为 `{plan['B2_algorithm_producer']}`，旧结果与失败保持。\n\n"
    content += '\n'.join(table)+'\n\n'
    content += '破折号表示没有可用于完整质量比较的值，失败前缀单独保存在paired_metrics.json，未填入完整任务均值。\n\n'
    content += f"P0固定旧update250权重：32 DDIM，16组同噪声对照有{probe['same_noise_pairs_observed']}组观察到条件响应，raw幅值合法{probe['amplitude_legal_raw_outputs']}/32；未裁剪或替换。公开旧TEST仅用于诊断，无新增物理或训练。\n\n"
    content += '## 五个问题的直接回答\n\n'
    for i, entry in enumerate(questions, 1):
        content += f"{i}. **{entry['question']}** {entry['answer']}\n\n"
    content += '## 成本与边界\n\n```json\n'+json.dumps(cost, ensure_ascii=False, indent=2)+'\n```\n\n'
    content += 'I_route由原日志中17维名义速度与选中速度差的范数恢复；名义值经过原速度边界clipping，两个源向量未保存。固定T_route取完整预声明区间的闭区间50Hz样本。完整Task与原五项独立安全门禁通过后才作完整质量比较；相关连续体-球净空不被整机最小值替代。\n\n'
    content += '20ms规划、2ms物理、27s任务及原QP/67路力矩/私有预演/安全标准保持。20ms墙钟不作研究门禁，部署NOT_MET；无硬实时、连续时间或模型失配保证。正式TRAIN/VAL/TEST已冻结但未执行，不能将其计作独立测试成功。\n'
    with (OUT/'REPORT.md').open('x', encoding='utf-8') as file:
        file.write(content)
    print(json.dumps({'status':report['status'],'pilot_full_task_success':report['pilot_full_task_success'],'costs':cost}, ensure_ascii=False))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=OUT)
    args = parser.parse_args()
    OUT = args.source.resolve()
    collect()
