"""Consume the qualified pilot, train, and close the unchanged DEV protocol."""
from pathlib import Path
import os
import time
import zipfile
import subprocess
import numpy as np
from ff_prior.storage import read_json,file_digest,digest,RunLock
from .storage import atomic_json,atomic_npz,Ledger
from ff_prior.recovery import process_is_alive
from ff_prior.pipeline import audit,acceptance
from ff_prior.guidance import public_goal_initializer
from ff_prior_coverage.experiment import inputs as coverage_inputs,sources as parent_sources
from ff_prior_pilot_train import learning as frozen_learning
from ff_prior_expand.sampling import candidate
from ff_prior_modes import data as loader
from ff_prior_loop.experiment import execute
from . import learning,sensitivity

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'artifacts/v7_1/smoke_002'
FROZEN=ROOT/'artifacts/v7_1/qualified_pilot_train_002/learning/goal_loss'
METHODS=('structured','frozen_diffusion','learned_mean','expanded_diffusion')
PROTOCOL={'version':'qualified_coverage_training_closure_1','profile':'coverage_training_and_frozen_DEV',
    'training':learning.CONTRACT,'sensitivity':sensitivity.CONTRACT,'TRAIN_range':[12096,12144],'VAL':48,
    'sampling_seeds':[20261219,20261223,20261229],'DEV_tasks':4,'candidates_per_task':5,
    'methods':list(METHODS),'max_candidates':240,'max_selected_executions':48,
    'candidate_wall_s':15,'candidate_coupled_ceiling':113,'candidate_interval_ceiling':15872,
    'wall_budget_s':5400,'max_output_bytes':20*1024**3,'cpu_threads':2,
    'stability_gate':'new model each seed >=18/20 eligible; all tasks reached; 12/12 selected execution/replay; no numerical errors',
    'ingestion_gate':'all qualified coverage TRAIN arrays loaded and sampled; original TRAIN-only codec frozen; exact task sensitivities',
    'selection':'minimum normalized frozen world-goal error among eligible, exactly one execution; no replacement',
    'old_model':'verified 10000-TRAIN model; same frozen codec as warm start',
    'TEST':False,'strict_multimodal':False,'relative_efficiency_required':False,
    'full_goal_auto_complete':False,'automatic_100k_expansion':False}




def sources():
    return dict(parent_sources(),**{p.relative_to(ROOT).as_posix():file_digest(p) for p in sorted((ROOT/'ff_prior_coverage_train').rglob('*.py'))})


def inputs(source):
    source=Path(source).resolve()
    old,ds,dev,_,inherited=coverage_inputs(BASE)
    files={Path(os.path.relpath(BASE/p,source)).as_posix():h for p,h in inherited.items()}
    for p in sorted(source.rglob('*')):
        if p.is_file() and p.suffix!='.tmp' and p.name not in ('launch.guard','run.lock'):
            files[p.relative_to(source).as_posix()]=file_digest(p)
    index=read_json(source/'training_index.json')['records']
    return old,ds,dev,index,files


def require_data(source):
    source=Path(source);state=read_json(source/'status.json');summary=read_json(source/'summary.json')
    if state['status']!='FINISHED' or process_is_alive(state['pid']):raise ValueError('coverage data worker has not completed and exited')
    if not summary['ready_for_training_stage'] or not summary['coverage_data_gate_passed']:raise ValueError('coverage data gate not passed')
    if read_json(source/'independent_verification.json')['status']!='PASS':raise ValueError('coverage raw evidence audit missing')
    index=read_json(source/'training_index.json')['records']
    n=sum(e['split']=='TRAIN' for e in index)
    if not 12096<=n<=12144 or sum(e['split']=='VAL' for e in index)!=48:raise ValueError('coverage population incomplete')
    if any(e['execution_status']=='FAIL' for e in index):raise ValueError('failed execution included in training')
    loader.old_data.assert_disjoint(index)
    return index


def report(out,ledger,status):
    out=Path(out);metrics={}
    for method in METHODS:
        candidates=[read_json(p) for p in (out/'candidates').glob(f'*/*/{method}/*.json')]
        outcomes=[read_json(p) for p in (out/'executions').glob(f'*/*/{method}/outcome.json')]
        by_seed={str(s):{'completed':sum(r['seed_group']==s for r in candidates),
            'eligible':sum(r['seed_group']==s and r['eligible'] for r in candidates),
            'tasks_reached':sum(any(r['seed_group']==s and r['eligible'] and r['sample_id']==f'DEV_{i:02d}' for r in candidates) for i in range(4))} for s in range(3)}
        metrics[method]={'completed':len(candidates),'eligible':sum(r['eligible'] for r in candidates),
            'executions':len(outcomes),'execution_verified':sum(r['status']=='EXECUTION_VERIFIED' for r in outcomes),
            'per_seed':by_seed,'network_calls':sum(r['actual_network_calls'] for r in candidates),
            'coupled_queries':sum(r['model_calls'].get('coupled',0) for r in candidates),'wall_s':sum(r.get('wall_s',0) for r in candidates),
            'stable':len(candidates)==60 and len(outcomes)==12 and all(v['eligible']>=18 and v['tasks_reached']==4 for v in by_seed.values()) and all(r['status']=='DONE' for r in candidates) and all(r['status']=='EXECUTION_VERIFIED' for r in outcomes)}
    p=out/'learning/goal_loss/training.json';training=read_json(p) if p.exists() else {}
    n=len(training.get('train_draw_counts',{}))
    trained=training.get('status')=='PASS' and training.get('all_TRAIN_rows_seen') is True and 12096<=n<=12144
    p=out/'independent_verification.json';audited=p.exists() and read_json(p)['status']=='PASS'
    errors=[k for k,v in ledger.data['completed'].items() if v['status'] in ('ERROR','INTERRUPTED','BLOCKED')]
    passed=bool(status=='FINISHED' and trained and audited and metrics['expanded_diffusion']['stable'] and not errors and not ledger.data.get('unresolved_call_accounting'))
    result={'status':status,'methods':metrics,'actual_training_passed':trained,'independent_audit_passed':audited,
        'coverage_training_closure_passed':passed,'errors':errors,'unresolved_calls':ledger.data.get('unresolved_call_accounting',[]),
        'TEST_executed':False,'full_goal_achieved':False,'diffusion_superiority':'REQUIRES_SEPARATE_SUPPORTED_COMPARISON',
        'relative_efficiency_required':False,'ledger_calls':ledger.data['calls'],'protocol':PROTOCOL}
    result['inherited_failure_accounting']=read_json(FROZEN.parent.parent/'summary.json')['recovery_accounting']
    atomic_json(out/'summary.json',result)
    lines=['# 覆盖扩库的真实训练与冻结DEV闭环','',f'状态：{status}；全体合格TRAIN实际训练消费：{trained}；独立审计：{audited}；本轮闭环：{passed}。','',
        '| 方法 | 合格/60 | 执行及重放/12 |','|---|---:|---:|']
    for m,r in metrics.items():lines.append(f'| {m} | {r["eligible"]} | {r["execution_verified"]} |')
    lines+=['','合格率针对网络/均值/样条提案＋共同目标保持碰撞引导的完整方法，不表示裸网络全部直接合格。',
        '预注册每种子至少18/20候选、各任务可达、12/12执行重放；查询和时间仅辅助记录，不要求效率胜出。',
        '原DEV是已使用的开发任务，不是最终TEST泛化证据。验证等级按实际单条证据保留，不批量升级整个数据集。',
        '失败不重标世界目标、不替补执行；训练或敏感度依赖失败时保留BLOCKED，独立基线仍运行。最终TEST未执行，不自动扩至100000。']
    lines+=['','冻结10k编码器和权重作为起点，新增6000更新；不是从零重训或数据规模单因素对照。',
        '旧敏感度仅在任务、锚点、数组、编码器完全相同时复用，历史计算与本次新计算分别列账。旧001失败和最多39次未知额外调用保留，不记为零。']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return result


def run(source,out,progress,resume=False,previous_elapsed=0.):
    from .audit_results import verify
    source,out=Path(source).resolve(),Path(out).resolve();deadline=time.time()+max(0,5400-previous_elapsed);require_data(source)
    old,_,dev,index,files=inputs(source)
    identity={'protocol':PROTOCOL,'source':str(source),'source_files':files,'tasks_hash':digest(dev),'source_hashes':sources()}
    with RunLock(out):
        tests=read_json(ROOT/'ff_prior_coverage_train/evidence/software_tests.json')
        if tests['status']!='PASS' or tests['source_hashes']!=sources() or tests['junit_sha256']!=file_digest(ROOT/'ff_prior_coverage_train/evidence/tests.xml'):raise ValueError('software proof stale')
        if (out/'run.json').exists() and (not resume or read_json(out/'run.json')['identity']!=identity):raise ValueError('run identity changed')
        physical,canonical,initial,_=audit(old['config']);profile=acceptance(old['config'])
        assert physical['acceptance']==read_json(source/'run.json')['acceptance']
        if not (out/'run.json').exists():
            atomic_json(out/'run.json',{'identity':identity,'acceptance':physical['acceptance'],'producer_head':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()})
            atomic_json(out/'frozen_tasks.json',{'DEV':dev});atomic_json(out/'training_index.json',{'records':index})
            with zipfile.ZipFile(out/'executed_source.zip','w',zipfile.ZIP_DEFLATED) as z:
                for p in sources():z.write(ROOT/p,p)
        assert read_json(out/'frozen_tasks.json')['DEV']==dev
        ledger=Ledger(out,{'run_identity_hash':digest(identity)})
        cache=loader.RuntimeCache(canonical,initial,profile)
        arrays,rows=loader.arrays(index,cache,out,progress);codec=learning.fit_codec(out/'learning',arrays)
        # Keep 1200 seconds for independent baselines and selected execution.
        try:
            matrices,sreport=sensitivity.prepare(out,arrays,rows,codec,cache,ledger,deadline-1200,progress)
        except Exception as error:
            import traceback
            matrices=None;sreport={'status':'BLOCKED','reason':repr(error),'traceback':traceback.format_exc()}
            if ledger.data.get('inflight'):
                ledger.data.setdefault('unresolved_call_accounting',[]).append(ledger.data['inflight']['id'])
                ledger.finish({'status':'BLOCKED','reason':repr(error)},{})
            atomic_json(out/'sensitivity_failure.json',sreport)
        try:
            training=learning.train(out/'learning/goal_loss',arrays,codec,matrices,'goal_loss',sreport,progress,deadline-1200)
        except Exception as error:
            training={'status':'BLOCKED','reason':repr(error)};atomic_json(out/'training_failure.json',training)
        bundle=learning.load(out/'learning/goal_loss') if training['status']=='PASS' else None
        frozen=frozen_learning.load(FROZEN)
        for sj,seed in enumerate(PROTOCOL['sampling_seeds']):
            for ti,e in enumerate(dev):
                task=e['task'];sid=f'DEV_{ti:02d}';rr,ii,cc,transfer=cache.get({'task':task})
                atomic_npz(out/'DEV_tasks'/sid/'initial_state.npz',values=ii.values)
                base=public_goal_initializer(task,rr,ii)
                atomic_json(out/'transfer_audits'/f'{sj}_{sid}.json',dict(transfer,shared_public_base_hash=digest(base)))
                for method in METHODS[ti%4:]+METHODS[:ti%4]:
                    chosen_bundle=None if method=='structured' else frozen if method=='frozen_diffusion' else bundle
                    records=[]
                    for k in range(5):
                        key=f'candidate_{sj}_{sid}_{method}_{k}';path=out/'candidates'/str(sj)/sid/method/f'{k:02d}.json'
                        if key not in ledger.data['completed']:
                            if time.time()>=deadline:raise RuntimeError('TOTAL_BUDGET_EXHAUSTED')
                            if sum(p.stat().st_size for p in out.rglob('*') if p.is_file())>=PROTOCOL['max_output_bytes']:raise RuntimeError('STORAGE_BUDGET_EXHAUSTED')
                            ledger.begin(key,'candidate',5400,240);progress('frozen_DEV_candidates',seed=sj+1,task=ti+1,method=method,candidate=k+1)
                            if method in ('learned_mean','expanded_diffusion') and bundle is None:
                                r={'status':'BLOCKED','eligible':False,'reason':'TRAINING_DEPENDENCY_BLOCKED','model_calls':{},'actual_network_calls':0,'candidate_id':k,'method':method}
                            else:r=candidate(method,k,task,base,seed+ti*101+k*997,chosen_bundle,rr,ii,cc,profile,deadline)
                            r.update(sample_id=sid,split='DEV',seed_group=sj);atomic_json(path,r);ledger.finish({'status':r['status']},r['model_calls'])
                        if path.exists():records.append(read_json(path))
                    key=f'execution_{sj}_{sid}_{method}';directory=out/'executions'/str(sj)/sid/method
                    if key in ledger.data['completed']:continue
                    if time.time()>=deadline:raise RuntimeError('TOTAL_BUDGET_EXHAUSTED')
                    chosen=min((r for r in records if r['eligible']),key=lambda r:(r['score'],r['candidate_id']),default=None)
                    atomic_json(directory/'selection.json',{'selected_id':chosen['candidate_id'] if chosen else None,'task_hash':task['task_hash'],'selection_before_execution':True,'rule':PROTOCOL['selection']})
                    ledger.begin(key,'execution',5400,48);before=rr.calls.copy();progress('DEV_torque_replay',seed=sj+1,task=ti+1,method=method)
                    if chosen:r=execute(directory,task,chosen['Z'],rr,ii,cc,profile,deadline)
                    else:r={'status':'NO_CANDIDATE'};atomic_json(directory/'outcome.json',r)
                    ledger.finish(r,{n:rr.calls[n]-before[n] for n in before})
                report(out,ledger,'RUNNING')
        progress('independent_raw_evidence_audit');proof=verify(out,identity,index,rows,codec)
        atomic_json(out/'independent_verification.json',proof)
        return report(out,ledger,'FINISHED' if proof['status']=='PASS' else 'FINISHED_WITH_BLOCKED_ITEMS')
