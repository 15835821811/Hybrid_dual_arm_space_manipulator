"""Counts use frozen denominators, including rejected and blocked candidates."""
from ff_prior.storage import read_json
from ff_prior_pilot_train.storage import atomic_json
from ff_prior_delivery.gates import engineering_gate
from .protocol import METHODS, PROTOCOL


def report(out, state):
    metrics = {}
    failures = []
    for method in METHODS:
        rows = [read_json(p) for p in sorted((out/'candidates').glob(f'*/*/{method}/*.json'))]
        outcomes = [read_json(p) for p in sorted((out/'executions').glob(f'*/*/{method}/outcome.json'))]
        m = {'completed': len(rows), 'eligible': sum(bool(r['eligible']) for r in rows),
             'selected_outcomes': len(outcomes), 'execution_verified': sum(r['status']=='EXECUTION_VERIFIED' for r in outcomes),
             'candidate_errors': sum(r['status']=='ERROR' for r in rows),
             'execution_errors': sum(r['status']=='ERROR' for r in outcomes),
             'representation_blocked': sum(r['status']=='BLOCKED_REPRESENTATION' for r in rows),
             'coupled_queries': sum(r['model_calls'].get('coupled',0) for r in rows),
             'candidate_wall_s': sum(r.get('wall_s',0) for r in rows)}
        metrics[method] = m
        failures.extend({'method': method, 'sample_id': r['sample_id'], 'seed_group': r['seed_group'],
                         'candidate_id': r['candidate_id'], 'status': r['status'], 'reason': r.get('reason'),
                         'validation': r.get('validation')} for r in rows if not r['eligible'])
    audit_path = out/'independent_verification.json'
    audit = read_json(audit_path) if audit_path.exists() else {'status': 'PENDING'}
    gates = {method: engineering_gate(m, 80, 16, audit['status']=='PASS',
             [] if not m['execution_errors'] else ['native execution exception']) for method,m in metrics.items()}
    new, old = metrics['late4_diffusion'], metrics['frozen_diffusion']
    choose_new = (gates['late4_diffusion']['status']=='PASS' and new['eligible']>=old['eligible']
                  and new['execution_verified']>=old['execution_verified'])
    chosen = 'late4_diffusion' if choose_new else 'frozen_diffusion' if gates['frozen_diffusion']['status']=='PASS' else None
    selected = {'method': chosen, 'TRAIN_only': True, 'M2_ready': chosen is not None,
                'new_method_replaces_old': choose_new, 'reason': 'both frozen count criteria' if choose_new else 'retain engineering-qualified old sampler' if chosen else 'no qualified sampler in this block'}
    result = {'status': state, 'stage': 'M1', 'metrics': metrics, 'engineering_gates': gates, 'selection': selected,
              'audit': audit['status'], 'training_updates': 0, 'new_dataset_rows': 0,
              'diffusion_advantage': 'NOT_ESTABLISHED', 'TEST': 'NOT_RUN', 'next_stage': 'M2_READY' if chosen else 'BLOCKED_OR_RUNNING',
              'protocol': PROTOCOL}
    if state.startswith('FINISHED') and not chosen:
        result['status'] = 'FINISHED_WITH_BLOCKED_ITEMS'
    atomic_json(out/'summary.json', result)
    atomic_json(out/'failures.json', failures)
    lines = ['# M1 后段引导与表示闭环报告', '', f'状态：{result["status"]}。审计：{audit["status"]}。', '',
             '8 个冻结 TRAIN 世界目标、2 种子、四方法、每单元 5 候选；原 DEV/VAL/TEST 均未运行。稳定 13774 模型权重未训练或改写。', '',
             '| 方法 | 合格候选/80 | 执行重放/16 | 候选异常 | 工程门 |', '|---|---:|---:|---:|---|']
    for method,m in metrics.items():
        lines.append(f'| {method} | {m["eligible"]}/80 | {m["execution_verified"]}/16 | {m["candidate_errors"]} | {gates[method]["status"]} |')
    rep_path = out/'representation_gate.json'
    if rep_path.exists():
        rep = read_json(rep_path)
        lines.extend(['', f'表示检查：教师 {rep["teacher_eligible"]}/8 合格；编码解码往返后 {rep["roundtrip_eligible"]}/8 合格。',
                      f'后四步方案启用：{rep["late4_enabled"]}。未通过表示门时，预冻结槽记 BLOCKED_REPRESENTATION，没有伪造采样。'])
    lines.extend(['', f'下一阶段选择：{chosen or "尚未满足"}；M2 可进入：{selected["M2_ready"]}。', '',
                  '软件实现、模型层合格、真实力矩执行与 Diffusion 优势分别统计。共同引导的成功不能证明网络优势；本轮不要求查询或耗时胜出。',
                  '物理门保持仅不穿透、世界 5 mm/1°、全机几何与原运动/力矩条件。无失败重标、执行替补或全库升级。',
                  '本报告由原始记录生成，最终独立研究审查与跨阶段交付在 M4 完成。'])
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    return result
