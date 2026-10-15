"""Generate the complete measured report locally, without an LLM service."""
from pathlib import Path
from collections import Counter
import numpy as np
from ff_prior.storage import read_json,atomic_json
from ff_prior_next.experiment import METHODS


def write_report(out,ledger,status):
    out=Path(out); summaries={}
    corrections=[read_json(p) for p in sorted((out/'corrections/records').glob('*.json'))]
    dataset=read_json(out/'training_dataset.json')
    diagnostic=[read_json(p) for p in sorted((out/'consistency').glob('*/diagnostic.json'))]
    for method in METHODS:
        cs=[read_json(p) for p in sorted((out/'candidates'/method).glob('*.json'))]
        executions=[v for v in ledger.data['completed'].values() if v.get('method')==method and v.get('kind')=='execution']
        errors=[max(c['validation']['frozen_goal_position_error_m']) for c in cs if 'frozen_goal_position_error_m' in c.get('validation',{})]
        summaries[method]={'task_denominator':4,'candidate_denominator':20,'candidate_records':len(cs),
            'model_valid':sum(c.get('validation',{}).get('validation_level')=='MODEL_VALID' for c in cs),
            'goal_valid':sum(c.get('frozen_goal_pass',False) for c in cs),
            'execution_verified':sum(v['status']=='EXECUTION_VERIFIED' for v in executions),
            'candidate_errors':sum(c.get('status')!='DONE' for c in cs),
            'failed_gates':dict(Counter(g for c in cs for g,ok in c.get('validation',{}).get('gates',{}).items() if not ok)),
            'worse_arm_position_mm_range':[min(errors)*1000,max(errors)*1000] if errors else None,
            'network_calls':sum(c.get('sampling',{}).get('actual_network_calls',0) for c in cs),
            'full_path_cost_evaluations':sum(c.get('full_path_cost_evaluations',0) for c in cs)}
    physical=[read_json(p) for p in out.glob('consistency/*/execution/report.json')]
    benchmark=[read_json(p) for p in out.glob('executions/*/*/report.json')]
    stronger=summaries[METHODS[1]]['execution_verified']>summaries[METHODS[0]]['execution_verified']
    consistency_flags=[d['comparison']['diagnostic_flags'] for d in diagnostic if 'comparison' in d]
    model_fidelity='CONSISTENT_IN_CHECKED_CASES' if len(consistency_flags)==4 and all(all(f.values()) for f in consistency_flags) else 'UNRESOLVED_OR_MISMATCH'
    summary={'status':status,'scope':'ADAPTIVE_FROZEN_DEV_SMOKE','methods':summaries,
        'correction':{'targets':8,'attempt_denominator':16,'records':len(corrections),
            'accepted_model_and_fixed_goal':sum(r.get('training_eligible',False) for r in corrections),
            'training_after_execution_exclusion':len(dataset['new_eligible_ids']),
            'failed_or_error':sum(not r.get('training_eligible',False) for r in corrections)},
        'training_dataset':dataset,'consistency_diagnostics':diagnostic,'model_fidelity':model_fidelity,
        'physical_execution':{'diagnostic_attempts':len(physical),'diagnostic_verified':sum(r['level']=='EXECUTION_VERIFIED' for r in physical),
            'benchmark_attempts':len(benchmark),'benchmark_verified':sum(r['level']=='EXECUTION_VERIFIED' for r in benchmark),
            'independent_replay_pass':sum(r['replay']['passed'] for r in physical+benchmark)},
        'benefit':'EXPLORATORY_DEV_SIGNAL' if stronger and status=='FINISHED' else 'NOT_ESTABLISHED',
        'generalization_claim':'NOT_ESTABLISHED','hardware_accuracy':'NOT_VALIDATED',
        'pilot':'NOT_RUN','TEST':'NOT_RUN','attempts':ledger.data['attempts'],'calls':ledger.data['calls'],
        'wall_s':ledger.data['wall_s'],'cpu_s':ledger.data['cpu_s'],
        'unknown_interrupted_work':ledger.data.get('unresolved_call_accounting',[]),
        'software_tests':read_json(out/'software_tests.json')}
    if (out/'learning/training.json').exists():
        summary['training']={k:v for k,v in read_json(out/'learning/training.json').items() if k not in ('history','metadata')}
    old_root=Path('artifacts/v7_1/public_seed_residual_001')
    if (old_root/'summary.json').exists():
        summary['historical_comparison']={'scope':'previous run retained; not extra experiments in this run',
            'methods':read_json(old_root/'summary.json')['methods']}
    atomic_json(out/'summary.json',summary)
    lines=['# V7.1 固定 TRAIN 目标修正与预测—执行一致性报告','',
        f"状态：{status}。Diffusion收益：{summary['benefit']}；模型一致性：{model_fidelity}。",'',
        '## 固定目标数据补充','',
        f"冻结8个TRAIN family目标，最多16次修正；模型门槛与原世界目标同时合格 {summary['correction']['accepted_model_and_fixed_goal']}/16。",
        f"诊断中已知执行失败的样本排除后，新增训练样本 {len(dataset['new_eligible_ids'])} 条；TRAIN {len(dataset['TRAIN'])}，VAL {len(dataset['VAL'])}。",
        f"本轮排除的已知执行失败标签：{dataset['excluded_known_execution_failed_ids']}。原始文件不回写。",
        '只改变平滑轨迹参数，不改变冻结目标、时长、初态或物理合同；失败尝试不补样。主先验没有新增每目标五条的配额。','',
        '## 耦合预测与真实力矩执行','',
        '| 样本 | 执行 | 参考预测EE最大偏差 mm | 按实际67关节重建EE最大偏差 mm | 重建基座最大偏差 mm |',
        '|---|---|---:|---:|---:|']
    for d in diagnostic:
        if 'comparison' not in d:
            lines.append(f"| {d['sample_id']} | {d['execution_status']}，比较BLOCKED | — | — | — |"); continue
        c=d['comparison']
        lines.append(f"| {d['sample_id']} | {d['execution_status']} | {c['nominal_reference_vs_execution']['end_effectors']['max_position_m']*1000:.6f} | {c['actual_native_path_reconstruction_vs_execution']['end_effectors']['max_position_m']*1000:.6f} | {c['actual_native_path_reconstruction_vs_execution']['base']['max_position_m']*1000:.6f} |")
    lines += ['', '参考预测使用规定关节路径；实际67关节重建使用记录的全部原生关节位置和速度，从同一初始基座状态积分，绝不逐帧复制实际基座，也不用B+投影代替原生几何。',
        '两者的差别用于区分参考跟踪误差与耦合传播残差。诊断只给一致性证据，不授予未执行样本EXECUTION_VERIFIED，也不能证明真实硬件模型准确。',
        '比较覆盖整个运动和hold；原目标验收仍5 mm/1度，非穿透>=0 m。1 mm/0.2度重建标志仅为预登记诊断，不改变执行门槛。','',
        '## 原冻结DEV smoke','',
        '| 方法 | MODEL_VALID / 20 | 原世界目标合格 / 20 | 执行与重放通过 / 4 |',
        '|---|---:|---:|---:|']
    for method,s in summaries.items():
        lines.append(f"| {method} | {s['model_valid']} | {s['goal_valid']} | {s['execution_verified']} |")
    for method,s in summaries.items():
        lines.append(f"- {method}：较差一臂位置误差范围mm {s['worse_arm_position_mm_range']}；失败模型门槛 {s['failed_gates']}；候选错误 {s['candidate_errors']}。")
    lines += ['', '所有方法仍共享公开目标初始化。三方法每任务五候选；Diffusion每候选16次网络调用，guided与结构基线各两次八方向全路径引导。',
        'TRAIN修正的额外优化和四个选定诊断均单独计费，不作为DEV额外候选；冻结DEV只在模型训练完成后用于本轮比较，不参与选数据或权重。',
        '这是反复查看同四个DEV任务的开发验证，任何收益仅为探索性信号。若无改善，不自动扩数据、训练或调参。','',
        '## 原因分析','']
    if model_fidelity=='CONSISTENT_IN_CHECKED_CASES':
        lines.append('四个检查案例的规定参考预测及实际原生路径重建均满足诊断标志，当前证据支持这些案例内的耦合传播一致性；若DEV目标仍失败，应优先分析条件学习与有限引导求解。不能由四例外推全部机器人状态。')
    else:
        lines.append('存在未完成或不满足诊断标志的案例。应分别读取参考—执行误差、实际原生路径重建残差、网格收敛和跟踪/流形误差，不能直接将全部偏差归因于自由漂浮物理模型。')
    if 'training' in summary:
        t=summary['training']
        lines += [f"训练 {t['status']}，{t['updates']} 更新，重载误差 {t['reload_error']}；VAL零残差RMSE {t['zero_residual_val_rmse_rad']:.6f} rad。",
            f"VAL噪声分段诊断：{t['noise_bands']}。这些控制点去噪指标不能替代目标与真实执行成功。"]
    lines += ['', '## 开销与保存','',
        f"新增真实执行：诊断{len(physical)}次、DEV {len(benchmark)}次；独立重放通过{summary['physical_execution']['independent_replay_pass']}次。",
        f"分阶段wall秒：{summary['wall_s']}；CPU秒：{summary['cpu_s']}。训练耗时另见training.json，进程总时长见status.json。",
        f"实际模型与物理调用：{summary['calls']}。未决中断工作：{summary['unknown_interrupted_work']}。未知调用不作零。",
        '全部16次TRAIN修正、固定目标、诊断预选、逐候选、真实执行/重放、checkpoint与代码快照保存在本运行目录。',
        '旧结果、原未提交修改及旧作业保留；没有pilot或TEST，没有上传权重或推送。']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return summary
