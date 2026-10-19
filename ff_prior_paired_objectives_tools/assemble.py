"""One-time creation of independent wrappers from frozen baseline code."""
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];pkg=ROOT/'ff_prior_paired_objectives'
old=ROOT/'ff_prior_world_endpoint'
job=(old/'job.py').read_text(encoding='utf-8').replace('ff_prior_world_endpoint','ff_prior_paired_objectives').replace('新覆盖实际训练与smoke未完成','配对训练与smoke未完成')
job=job.replace("PYTHONUNBUFFERED='1',", "PYTHONUNBUFFERED='1',PYTHONIOENCODING='utf-8',")
(pkg/'job.py').write_text(job,encoding='utf-8')
(pkg/'job.ps1').write_text((old/'job.ps1').read_text(encoding='utf-8').replace('ff_prior_world_endpoint','ff_prior_paired_objectives').replace('world_endpoint_smoke_001','paired_objectives_smoke_001'),encoding='utf-8')
src=(old/'experiment.py').read_text(encoding='utf-8')
body=src[src.index('        smoke_deadline='):src.index("        atomic_json(out/'runtime_cache_audit.json'")]
body='\n'.join(line[4:] if line.startswith('    ') else line for line in body.splitlines())
header='''from pathlib import Path
import time
import numpy as np
from ff_prior.storage import read_json,digest,file_digest
from ff_prior_coverage_train.storage import atomic_json,atomic_npz
from ff_prior.guidance import public_goal_initializer
from ff_prior_generalization.sampling import candidate
from ff_prior_generalization.experiment import execute
from ff_prior_world_endpoint.experiment import capacity,blocked_candidate
from ff_prior_goal.projection import smooth
from ff_prior_feasible.validation import validate
from .protocol import METHODS,PROTOCOL as P
from .reporting import report

def run(out,entries,cache,profile,ledger,deadline,progress,old_bundle,new_bundle,mean_codec):
'''
(pkg/'smoke.py').write_text(header+body+'\n',encoding='utf-8')
text=(old/'reporting.py').read_text(encoding='utf-8');prefix=text[:text.index("    trpath=")]
tail='''    selection=read_json(out/'TRAIN_selection.json') if (out/'TRAIN_selection.json').exists() else {}
    single=read_json(out/'single_step_summary.json') if (out/'single_step_summary.json').exists() else {}
    training={}
    for arm in P['arms']:
        path=out/'learning'/arm/'training.json'
        if path.exists():
            t=read_json(path);training[arm]={k:t[k] for k in ('status','updates','world_gradient_updates','all_TRAIN_rows_seen','world_alignment','checkpoint_sha256')}
    audit=read_json(out/'independent_verification.json') if (out/'independent_verification.json').exists() else {}
    result={'status':status,'single_step_status':single.get('status','PENDING'),'training':training,
        'TRAIN_selection':selection,'methods':metrics,'task_metrics':tasks,'reused_DEV8_paired_quality':quality,
        'raw_audit':audit,'TEST_executed':False,'TEST_decision':'BLOCKED_BY_CURRENT_USER_SCOPE',
        'broad1024_started':False,'pilot_started':False,'full_goal_achieved':False,'protocol':P}
    atomic_json(out/'summary.json',result)
    lines=['# TRAIN单步诊断与四组配对训练','',f'状态：{status}；单步诊断：{result["single_step_status"]}。',
        '数据复用上一轮13838 TRAIN/48 VAL（包含64条固定目标修正）；本轮新增数据和TRAIN执行次数均为0。',
        '四组共用13774行阶段的冻结初始权重、codec、6000次更新、批次和噪声序列；每50步C/D加入一次真实102维当前预测终点梯度。',
        'A仅v；B增加0.05局部位姿；C增加0.1真实终点；D两者兼用。该实验隔离损失项，仍仅一个训练种子。','',
        f'TRAIN选择：{selection}。选择在接触本轮DEV结果前完成；无合格C/D则DEV依赖BLOCKED。','',
        '|组|更新|真实终点更新|固定TRAIN探针中位损失比|','|---|---:|---:|---:|']
    for arm,t in training.items():lines.append(f'|{arm}|{t["updates"]}|{t["world_gradient_updates"]}|{t["world_alignment"]["median_paired_loss_ratio"]:.6f}|')
    lines+=['','## 冻结DEV复核','',
        '仅TRAIN门通过后运行：原DEV4和已反复使用的开发8任务，3采样种子，4方法，每任务5候选；共720候选、至多144执行重放。8任务不代表未见泛化。',
        '|方法|候选完成|合格|执行重放|修正前位姿通过|裸提案目标通过|','|---|---:|---:|---:|---:|---:|']
    for m,r in metrics.items():lines.append(f'|{m}|{r["completed"]}|{r["eligible"]}|{r["execution_verified"]}|{r["pre_pose_pass"]}|{r["proposal_diagnostics"]["raw"]["eligible_with_frozen_goal"]}|')
    lines+=['',f'原始审计：{audit}。',f'任务配对质量结果：{quality}。',
        '模型层合格、实际力矩执行、独立重放与Diffusion相对强样条/条件均值的收益分别判断。完整引导成功不能替代裸网络的目标命中证据。',
        '本轮禁止1024槽扩库、pilot和TEST；后续扩库及最终TEST须另有范围授权和冻结协议。长期目标尚未完成。',
        '进程结束后的独立结果审查和中文分析报告仍需完成；本文件是后台原始汇总。','',
        '```powershell',r'.\\ff_prior_paired_objectives\\job.ps1 status',r'.\\ff_prior_paired_objectives\\job.ps1 report',r'.\\ff_prior_paired_objectives\\job.ps1 resume','```']
    (out/'REPORT.md').write_text('\\n'.join(lines)+'\\n',encoding='utf-8')
    return result
'''
(pkg/'reporting.py').write_text(prefix+tail,encoding='utf-8')
print('Created independent job, smoke, reporting wrappers; original sources unchanged.')
