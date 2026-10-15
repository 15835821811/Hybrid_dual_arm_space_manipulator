"""Deterministic report: stable constrained sampling and learning benefit differ."""
from pathlib import Path
from collections import Counter
import numpy as np
from ff_prior.storage import read_json, atomic_json


def write_report(out, ledger, status):
    from .experiment import METHODS
    out = Path(out); groups = read_json(out/'frozen_tasks.json'); summaries = {}
    for split, tasks in groups.items():
        summaries[split] = {}
        for method in METHODS:
            rows = [read_json(p) for p in sorted((out/'candidates'/split/method).glob('*.json'))]
            physical = [read_json(p) for p in sorted((out/'executions'/split/method).glob('*/report.json'))]
            per_task = [sum(r.get('eligible', False) for r in rows if r['task_hash'] == e['task']['task_hash']) for e in tasks]
            p = [max(r['validation']['frozen_goal_position_error_m'])*1000 for r in rows
                 if 'frozen_goal_position_error_m' in r.get('validation', {})]
            rotations = [max(r['validation']['frozen_goal_rotation_error_rad'])*180/np.pi for r in rows
                 if 'frozen_goal_rotation_error_rad' in r.get('validation', {})]
            calls = [sum(g.get('coupled_calls', 0) for g in r.get('sampling', {}).get('guidance', [])) for r in rows]
            summaries[split][method] = {'task_denominator': len(tasks), 'candidate_denominator': 5*len(tasks),
                'candidate_records': len(rows), 'eligible': sum(r.get('eligible', False) for r in rows),
                'eligible_per_task': per_task, 'model_valid': sum(r.get('validation', {}).get('validation_level') == 'MODEL_VALID' for r in rows),
                'goal_valid': sum(r.get('frozen_goal_pass', False) for r in rows),
                'execution_attempted': len(physical), 'execution_verified': sum(r['level'] == 'EXECUTION_VERIFIED' for r in physical),
                'replay_pass': sum(r['replay']['passed'] for r in physical),
                'failed_gates': dict(Counter(k for r in rows for k, v in r.get('validation', {}).get('gates', {}).items() if not v)),
                'candidate_errors': sum(r['status'] != 'DONE' for r in rows),
                'worse_arm_position_mm_range': [min(p), max(p)] if p else None,
                'worse_arm_rotation_deg_range': [min(rotations), max(rotations)] if rotations else None,
                'network_calls': sum(r.get('sampling', {}).get('actual_network_calls', 0) for r in rows),
                'projection_coupled_calls': sum(calls), 'candidate_wall_s': sum(r.get('wall_s', 0) for r in rows)}
    stable = status == 'FINISHED' and all(summaries[s][METHODS[1]]['eligible'] >= .95*5*len(t) and
        min(summaries[s][METHODS[1]]['eligible_per_task']) >= 4 and
        summaries[s][METHODS[1]]['execution_verified'] == len(t) for s, t in groups.items())
    labels = [read_json(p) for p in (out/'labels').glob('*.json')]
    stable = stable and len(labels) == 16 and all(r['status'] == 'PASS' for r in labels)
    summary = {'status': status, 'methods': summaries, 'label_revalidation_pass': sum(r['status'] == 'PASS' for r in labels),
        'label_denominator': 16, 'sampling_stability': 'SUPPORTED_WITHIN_THIS_SMOKE' if stable else 'NOT_ESTABLISHED',
        'diffusion_benefit': 'NOT_ESTABLISHED', 'scope': '6 VAL + original 4 adaptive DEV, five seeds each',
        'generalization': 'NOT_ESTABLISHED', 'hardware_validation': 'NOT_RUN', 'pilot': 'NOT_RUN', 'TEST': 'NOT_RUN',
        'calls': ledger.data['calls'], 'attempts': ledger.data['attempts'], 'wall_s': ledger.data['wall_s'],
        'cpu_s': ledger.data['cpu_s'], 'unresolved_work': ledger.data.get('unresolved_call_accounting', [])}
    atomic_json(out/'summary.json', summary)
    lines = ['# 耦合世界目标投影 smoke 实测报告', '', f"状态：{status}；稳定性：{summary['sampling_stability']}。", '',
        '冻结上一轮166 TRAIN所得模型、scaler与rank6时间先验，无新增训练。所有方法共享曲率平滑；结构基线与Diffusion共享17方向完整耦合目标修正。',
        '每候选两次、每次最多四次阻尼修正；Diffusion仍16次DDIM网络调用。未修正对照仅使用同样的平滑。', '',
        '| 分组 | 方法 | MODEL_VALID | 世界目标合格 | 两者合格 | 力矩执行+重放 |', '|---|---|---:|---:|---:|---:|']
    for split, methods in summaries.items():
        for method, r in methods.items():
            n = r['candidate_denominator']; m = r['task_denominator']
            lines.append(f"| {split} | {method} | {r['model_valid']}/{n} | {r['goal_valid']}/{n} | {r['eligible']}/{n} | {r['execution_verified']}/{m} |")
    lines += ['', f"原16条TRAIN修正标签重新通过模型和世界目标验收：{summary['label_revalidation_pass']}/16。未执行标签不升级。", '',
        '## 误差、失败与成本', '']
    for split, methods in summaries.items():
        for method, r in methods.items():
            lines.append(f"- {split}/{method}：位置范围mm {r['worse_arm_position_mm_range']}；姿态范围度 {r['worse_arm_rotation_deg_range']}；逐任务合格数 {r['eligible_per_task']}；失败门槛 {r['failed_gates']}；投影耦合查询 {r['projection_coupled_calls']}；候选wall {r['candidate_wall_s']:.3f}s。")
    lines += ['', '## 解释边界', '',
        '稳定性要求每个分组至少95%候选同时通过模型和冻结目标、每任务至少4/5、投影Diffusion全部10次预选执行与重放通过。它仅支持本次smoke范围，不支持全工作空间或硬件保证。',
        '普通样条也使用相同修正器。只有额外收益经过独立比较后，才可归因于Diffusion；约束修正成功本身不建立学习收益。',
        '保持非穿透>=0m、5mm/1度、2mm几何细化、原17→67和完整自由基座；不改目标，不输入私有终端构型。',
        '曲率平滑和局部投影均是提案操作，不能取代最终全机几何、真实力矩执行与独立重放。', '',
        f"工作计数：{summary['attempts']}；阶段wall：{summary['wall_s']}。完整调用见verification.json。",
        '全部尝试保留，不补失败候选，不进行pilot、TEST或生产扩规模。终端status/report不调用LLM。']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    return summary
