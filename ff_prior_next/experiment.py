"""Bounded, resumable residual-prior experiment on four existing frozen DEV tasks."""
from pathlib import Path
from collections import Counter
import subprocess
import time
import zipfile
import numpy as np
from scipy.spatial.transform import Rotation
from ff_prior.pipeline import audit, acceptance, source_manifest
from ff_prior.storage import read_json, atomic_json, atomic_npz, digest, file_digest, Ledger, RunLock
from ff_prior.contracts import assert_task_compatible, assert_frozen, EXECUTION_GATES
from ff_prior.guidance import public_goal_initializer, FullPathCost, SubspaceGuidance
from ff_prior.spline import Reference
from ff_prior.model_gates import evaluate_reference
from ff_prior.coupled_predictor import predict_coupled, pose_errors
from ff_prior.physical_audit import audit_execution
from . import learning

ROOT=Path(__file__).resolve().parents[1]
METHODS=('structured_public_seed','residual_diffusion_guided','residual_diffusion_unguided')
PROTOCOL={'version':'v71_public_seed_residual_smoke_1','profile':'smoke','tasks':4,'candidates_per_task':5,
    'methods':list(METHODS),'training_updates':200,'training_seed':20261015,'ddim_calls':16,
    'guidance_updates':2,'directions':8,'max_selected_executions':12,'wall_budget_s':1800,
    'direction_diagnostics':4,'diagnostic_perturbations_per_task':16,'pilot':False,'TEST':False,
    'benefit_criterion':'more execution-verified tasks than structured baseline; exploratory DEV only',
    'no_parameter_sweep':True,'no_failure_replacement':True}


def sources():
    return dict(source_manifest(),**{p.relative_to(ROOT).as_posix():file_digest(p)
        for p in sorted((ROOT/'ff_prior_next').glob('*.py'))})


def inputs(source):
    source=Path(source).resolve()
    run=read_json(source/'run.json'); data=read_json(source/'dataset.json')
    tasks=read_json(source/'benchmark_tasks.json')
    if run['identity']['profile']!='smoke' or len(tasks)!=4 or len(data['eligible']['TRAIN'])!=150 or len(data['eligible']['VAL'])!=30:
        raise ValueError('requires original smoke with four tasks / 150 TRAIN / 30 VAL')
    files={p:file_digest(source/p) for p in ('run.json','dataset.json','benchmark_tasks.json',
        'publication.json','initial_state.npz','pair_policy.json','baseline_audit.json')}
    publication=read_json(source/'publication.json')['files']; rows=[]
    for split in ('TRAIN','VAL'):
        for sid in data['eligible'][split]:
            record=f'records/{sid}.json'; row=read_json(source/record)
            if row['validation_level']!='MODEL_VALID' or row['data_role']!='FF_MODEL_PRIOR' or row['execution_status'] not in ('NOT_RUN','PASS'):
                raise ValueError('ineligible residual training source')
            for rel in (record,row['arrays']):
                if file_digest(source/rel)!=publication[rel]: raise ValueError('published source mutated: '+rel)
                files[rel]=publication[rel]
            rows.append((row,split))
    return run,data,tasks,rows,files


def direction_diagnostic(task,base,seed,robot,initial,deadline):
    # Diagnostic only: no candidate is selected or modified from these probes.
    guide=SubspaceGuidance(None,seed=seed)
    def errors(z):
        p=predict_coupled(initial,Reference(z,task['condition']['initial_features'],task['condition']['duration']),robot,128,deadline)
        actual=p['ee'][-1]; goal=np.asarray(task['condition']['goals'])
        return np.r_[((actual[:,:3,3]-goal[:,:3,3])/.005).ravel(),
            (Rotation.from_matrix(np.swapaxes(goal[:,:3,:3],-1,-2)@actual[:,:3,:3]).as_rotvec()/np.deg2rad(1)).ravel()]
    e=errors(base); epsilon=guide.epsilon
    J=np.stack([(errors(base+epsilon*d)-errors(base-epsilon*d))/(2*epsilon) for d in guide.directions],axis=1)
    coeff=np.linalg.lstsq(J,-e,rcond=1e-8)[0]; remaining=e+J@coeff
    return {'task_hash':task['task_hash'],'direction_hash':guide.direction_hash,'seed':seed,
        'jacobian_12x8':J,'initial_normalized_pose_error':e,'singular_values':np.linalg.svd(J,compute_uv=False),
        'rank':int(np.linalg.matrix_rank(J,tol=np.linalg.norm(J,2)*1e-8)),
        'optimistic_local_remaining_fraction':float(np.linalg.norm(remaining)/max(np.linalg.norm(e),1e-12)),
        'significant_local_uncovered_error':bool(np.linalg.norm(remaining)>0.1*max(np.linalg.norm(e),1e-12)),
        'unconstrained_linear_step_norm':float(np.linalg.norm(coeff)),
        'scope':'local linear unconstrained diagnostic; ignores domain, collision and step-size limits; not planning success',
        'coupled_calls':17,'candidate_budget_used':0}


def candidate(method,k,task,base,seed,model_bundle,robot,initial,collision,profile,deadline):
    cost=FullPathCost(task,robot,initial,collision,profile,deadline)
    guide=SubspaceGuidance(cost,seed=seed)
    if method==METHODS[0]:
        rng=np.random.default_rng(seed)
        z=base+.015*np.sin(np.linspace(0,np.pi,19))[:,None]*rng.normal(size=(1,17))
        log=[]
        for i in range(2): z,info=guide(z,i); log.append(info)
        sampling={'actual_network_calls':0,'guidance':log}
    else:
        model,scaler,prior=model_bundle
        z,sampling=learning.sample(model,scaler,prior,base,task,seed,guide if method==METHODS[1] else None)
    prediction,validation=evaluate_reference(robot,initial,Reference(z,task['condition']['initial_features'],task['condition']['duration']),collision,profile,deadline)
    goal_valid=False; score=None
    if prediction is not None:
        p,r=pose_errors(prediction['ee'][-1],task['condition']['goals'])
        goal_valid=bool(np.all(p<=profile.position_error_m) and np.all(r<=profile.rotation_error_rad))
        score=float(np.sum((p/profile.position_error_m)**2)+np.sum((r/profile.rotation_error_rad)**2))
        validation.update(frozen_goal_position_error_m=p,frozen_goal_rotation_error_rad=r)
    assert_frozen(task)
    return {'status':'DONE','candidate_id':k,'task_hash':task['task_hash'],'seed':seed,'method':method,
        'public_base_hash':digest(base),'Z':z,'sampling':sampling,'validation':validation,'score':score,
        'frozen_goal_pass':goal_valid,'eligible':validation['validation_level']=='MODEL_VALID' and goal_valid,
        'full_path_cost_evaluations':cost.calls,'private_teacher_fields_accessed':False}


def write_report(out,ledger,status):
    out=Path(out); summaries={}; tests=out/'software_tests.json'
    for method in METHODS:
        cs=[read_json(p) for p in sorted((out/'candidates'/method).glob('*.json'))]
        executions=[v for v in ledger.data['completed'].values() if v.get('method')==method and v.get('kind')=='execution']
        failures=Counter(k for c in cs for k,v in c.get('validation',{}).get('gates',{}).items() if not v)
        errors=[max(c['validation']['frozen_goal_position_error_m']) for c in cs if 'frozen_goal_position_error_m' in c.get('validation',{})]
        summaries[method]={'task_denominator':4,'candidate_denominator':20,'candidate_records':len(cs),
            'model_valid':sum(c.get('validation',{}).get('validation_level')=='MODEL_VALID' for c in cs),
            'goal_valid':sum(c.get('frozen_goal_pass',False) for c in cs),
            'execution_verified':sum(e['status']=='EXECUTION_VERIFIED' for e in executions),
            'execution_attempted':sum(e['status'] not in ('NO_CANDIDATE','BLOCKED') for e in executions),
            'failed_gates':dict(failures),'candidate_errors':sum(c.get('status')!='DONE' for c in cs),
            'worse_arm_error_mm_range':[min(errors)*1000,max(errors)*1000] if errors else None,
            'network_calls':sum(c.get('sampling',{}).get('actual_network_calls',0) for c in cs),
            'full_path_cost_evaluations':sum(c.get('full_path_cost_evaluations',0) for c in cs)}
    stronger=summaries[METHODS[1]]['execution_verified']>summaries[METHODS[0]]['execution_verified']
    summary={'status':status,'scope':'ADAPTIVE_FROZEN_DEV_SMOKE','methods':summaries,
        'benefit':'EXPLORATORY_DEV_SIGNAL' if stronger and status=='FINISHED' else 'NOT_ESTABLISHED',
        'generalization_claim':'NOT_ESTABLISHED','pilot':'NOT_RUN','TEST':'NOT_RUN',
        'calls':ledger.data['calls'],'wall_s':ledger.data['wall_s'],'cpu_s':ledger.data['cpu_s'],
        'attempts':ledger.data['attempts'],'unknown_interrupted_calls':ledger.data.get('unresolved_call_accounting',[]),
        'software_tests':read_json(tests) if tests.exists() else {'status':'NOT_ATTACHED'}}
    if (out/'preparation.json').exists(): summary['preparation']=read_json(out/'preparation.json')
    summary['interrupted_process_accounting']=read_json(out/'interruption.json') if (out/'interruption.json').exists() else None
    training=out/'learning/training.json'
    if training.exists():
        summary['training']={k:v for k,v in read_json(training).items() if k not in ('history','metadata')}
    diagnostics=[read_json(p) for p in sorted((out/'direction_diagnostics').glob('*.json'))]
    summary['direction_diagnostics']=[{k:v for k,v in d.items() if k not in ('jacobian_12x8','initial_normalized_pose_error')} for d in diagnostics]
    atomic_json(out/'summary.json',summary)
    lines=['# V7.1 公开目标初始化与残差 Diffusion 验证报告','',
        f"运行状态：{status}；收益判定：{summary['benefit']}。本轮为原四个 DEV 任务的适应性调试，不能视为独立 TEST。",'',
        '| 方法 | MODEL_VALID / 20 | 冻结目标通过 / 20 | 真实执行通过 / 4 |',
        '|---|---:|---:|---:|']
    for m,s in summaries.items(): lines.append(f"| {m} | {s['model_valid']} | {s['goal_valid']} | {s['execution_verified']} |")
    lines+=['','三个方法使用同一公开目标广义雅可比初始化，基线加平滑随机扰动；残差网络学习 Z 减该初始化，所有统计仅拟合 TRAIN。',
        '每方法每任务 5 候选；两种 Diffusion 每候选 16 网络调用；基线与 guided 均仅 2 次、每次 8 方向全路径引导。',
        '真实执行必须先通过全部模型门槛和冻结双末端目标，再预选一个候选；未执行不代表执行通过。',
        '非穿透 >=0 m、目标 5 mm/1 度、几何细化 2 mm、17→67、自由基座、真实力矩和独立重放保持原合同。','',
        '## 诊断与限制','']
    for m,s in summaries.items():
        lines.append(f"- {m}：较差一臂位置误差范围 mm = {s['worse_arm_error_mm_range']}；失败门槛 = {s['failed_gates']}；候选错误 = {s['candidate_errors']}。")
    if 'training' in summary:
        t=summary['training']; lines += [f"- 训练状态 {t['status']}，更新 {t['updates']} 次；零残差 VAL RMSE = {t['zero_residual_val_rmse_rad']:.6f} rad。",
            f"- 各噪声级别去噪诊断：{t['noise_bands']}。训练误差改善不能替代规划执行成功。"]
    for d in diagnostics:
        lines.append(f"- 引导方向局部秩 {d['rank']}/12；无约束线性化后剩余误差比例 {d['optimistic_local_remaining_fraction']:.4f}。这是乐观局部诊断，不能证明全局可达或不可达。")
    lines += ['', '几何细化失败表示未满足验证分辨率；是否发生穿透要单独读取 geometry 门槛。只有真实执行和独立重放全部通过才计入执行成功。',
        '若残差方案没有超过基线，本轮不支持 Diffusion 贡献；不自动扩数据、扩训练、调参或启动 pilot/TEST。',
        '若出现收益，也仅记为探索性 DEV 信号，仍需以后另行冻结独立验证协议。','',
        '## 资源和复现','',f"分阶段 wall 秒：{summary['wall_s']}；CPU 秒：{summary['cpu_s']}。",
        f"物理与模型调用：{summary['calls']}。训练调用和耗时另见 training.json；进程总耗时见 status.json。",
        f"中断导致未知调用的工作项：{summary['unknown_interrupted_calls']}。未知不计作零。",
        f"准备/公开初始化调用及耗时：{summary.get('preparation')}。进程中断记录：{summary['interrupted_process_accounting']}。",
        '所有逐候选结果、执行与重放报告、冻结源哈希、源码快照、事件日志均保留在本运行目录。']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return summary


def run(source,out,progress,resume=False,previous_elapsed=0.):
    source,out=Path(source).resolve(),Path(out).resolve()
    if source==out or source in out.parents or out in source.parents: raise ValueError('output must be separate from source')
    old,data,tasks,rows,files=inputs(source)
    identity={'protocol':PROTOCOL,'source':str(source),'source_files':files,'tasks_hash':digest(tasks),'source_hashes':sources()}
    with RunLock(out):
        if (out/'run.json').exists():
            if not resume or read_json(out/'run.json')['identity']!=identity: raise ValueError('resume identity conflict')
        proof=read_json(ROOT/'ff_prior_next/evidence/software_tests.json')
        if proof['status']!='PASS' or proof['source_hashes']!=identity['source_hashes']:
            raise ValueError('software regression proof missing or stale; rerun tests before launch')
        atomic_json(out/'software_tests.json',proof)
        report,robot,initial,collision=audit(old['config']); profile=acceptance(old['config'])
        if report['acceptance']!=read_json(source/'baseline_audit.json')['acceptance']: raise ValueError('physical profile changed')
        for entry in tasks: assert_task_compatible(entry['task'],robot,initial,collision,profile)
        manifest={'identity':identity,'physical_acceptance':report['acceptance'],'robot_hash':initial.model_hash,
            'initial_state_id':initial.state_id,'producer_head':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()}
        if not (out/'run.json').exists():
            atomic_json(out/'run.json',manifest); atomic_json(out/'benchmark_tasks.json',tasks)
            with zipfile.ZipFile(out/'executed_source.zip','w',zipfile.ZIP_DEFLATED) as archive:
                for name in identity['source_hashes']: archive.write(ROOT/name,name)
        ledger=Ledger(out,identity)
        if (out/'summary.json').exists() and read_json(out/'summary.json')['status']=='FINISHED':
            return read_json(out/'summary.json')
        deadline=time.time()+max(0,PROTOCOL['wall_budget_s']-previous_elapsed)
        prep_wall,prep_cpu=time.perf_counter(),time.process_time()
        z=[]; bases=[]; conditions=[]; ids=[]; splits=[]
        progress('prepare_public_bases',training_records=180)
        for row,split in rows:
            if time.time()>=deadline: raise RuntimeError('WALL_BUDGET_EXHAUSTED')
            assert_task_compatible(row['task'],robot,initial,collision,profile)
            z.append(np.load(source/row['arrays'])['Z']); bases.append(public_goal_initializer(row['task'],robot,initial))
            conditions.append(row['condition']); ids.append(row['sample_id']); splits.append(split)
        atomic_json(out/'data_transform.json',{'source_ids':ids,'splits':splits,'base_hash':digest(np.array(bases)),
            'definition':'physical Z minus public_goal_initializer(task)','TEST_arrays_read':False,'private_terminal_used':False})
        public_bases=[public_goal_initializer(t['task'],robot,initial) for t in tasks]
        previous_prep=read_json(out/'preparation.json') if (out/'preparation.json').exists() else {}
        atomic_json(out/'preparation.json',{'calls':{k:v+previous_prep.get('calls',{}).get(k,0) for k,v in robot.calls.items()},
            'wall_s':previous_prep.get('wall_s',0)+time.perf_counter()-prep_wall,
            'cpu_s':previous_prep.get('cpu_s',0)+time.process_time()-prep_cpu})
        training_ok=False
        try:
            result=learning.train(out/'learning',z,bases,conditions,ids,splits,initial.model_hash,initial.state_id,progress,deadline)
            training_ok=result['status']=='PASS'
        except Exception as error:
            atomic_json(out/'training_failure.json',{'status':'BLOCKED','reason':repr(error)})
        bundle=learning.load(out/'learning',initial.model_hash,initial.state_id) if training_ok else None
        for ti,entry in enumerate(tasks):
            task=entry['task']; base=public_bases[ti]; key=f'direction_{ti}'
            if key in ledger.data['completed']: continue
            progress('direction_diagnostic',task=ti+1)
            ledger.begin(key,'direction_diagnostics',1800,4); before=robot.calls.copy()
            try:
                diagnostic=direction_diagnostic(task,base,20261017+ti*101,robot,initial,deadline)
                atomic_json(out/'direction_diagnostics'/f'{ti:03d}.json',diagnostic); result={'status':'DONE'}
            except Exception as error: result={'status':'BLOCKED','reason':repr(error)}
            ledger.finish(result,{k:robot.calls[k]-before[k] for k in before})
        for ti,entry in enumerate(tasks):
            task=entry['task']; base=public_bases[ti]
            for method in METHODS:
                candidates=[]
                for k in range(5):
                    key=f'candidate_{ti}_{method}_{k}'; path=out/'candidates'/method/f'{ti:03d}_{k:02d}.json'
                    if key in ledger.data['completed']:
                        if path.exists(): candidates.append(read_json(path))
                        continue
                    progress('candidates',task=ti+1,method=method,candidate=k+1,
                        completed_candidates=ledger.data['attempts'].get('candidates',0),total_candidates=60)
                    ledger.begin(key,'candidates',1800,60); before=robot.calls.copy()
                    try:
                        if time.time()>=deadline: raise RuntimeError('WALL_BUDGET_EXHAUSTED')
                        if method!=METHODS[0] and bundle is None: raise RuntimeError('BLOCKED_TRAINING')
                        result=candidate(method,k,task,base,20261017+ti*101+k*997,bundle,robot,initial,collision,profile,deadline)
                    except Exception as error:
                        result={'status':'ERROR','reason':repr(error),'eligible':False,'candidate_id':k,'task_hash':task['task_hash']}
                    atomic_json(path,result); candidates.append(result)
                    ledger.finish({'status':result['status'],'method':method,'kind':'candidate'},
                        {n:robot.calls[n]-before[n] for n in before})
                ek=f'execution_{ti}_{method}'
                if ek in ledger.data['completed']: continue
                eligible=[c for c in candidates if c.get('eligible')]
                chosen=min(eligible,key=lambda c:(c['score'],c['candidate_id'])) if eligible else None
                directory=out/'executions'/method/f'{ti:03d}'
                selection={'task_hash':task['task_hash'],'selected_id':chosen['candidate_id'] if chosen else None,
                    'selection_before_execution':True,'rule':'minimum normalized dual pose error among fully eligible candidates'}
                atomic_json(directory/'selection.json',selection)
                ledger.begin(ek,'execution',1800,12); before=robot.calls.copy()
                progress('execution',task=ti+1,method=method,selected_id=selection['selected_id'])
                try:
                    if chosen:
                        result=audit_execution(task,profile,robot,initial,Reference(np.array(chosen['Z']),task['condition']['initial_features'],task['condition']['duration']),directory,deadline)
                        status=result['level']
                        if status=='EXECUTION_VERIFIED':
                            assert all(result['gates'][g] for g in EXECUTION_GATES) and result['replay']['passed']
                    else: status='NO_CANDIDATE'
                    result={'status':status,'method':method,'kind':'execution'}
                except Exception as error: result={'status':'EXECUTION_FAILED','reason':repr(error),'method':method,'kind':'execution'}
                ledger.finish(result,{n:robot.calls[n]-before[n] for n in before})
        if not all(file_digest(source/rel)==h for rel,h in files.items()): raise ValueError('source data modified during run')
        if sources()!=identity['source_hashes']: raise ValueError('runtime source changed during run')
        incomplete=not training_ok or (out/'interruption.json').exists() or bool(ledger.data.get('unresolved_call_accounting')) or any(
            v['status'] in ('ERROR','BLOCKED','INTERRUPTED') for v in ledger.data['completed'].values())
        status='FINISHED_WITH_BLOCKED_ITEMS' if incomplete else 'FINISHED'
        atomic_json(out/'verification.json',{'source_inputs_unchanged':True,'runtime_source_unchanged':True,
            'same_tasks':read_json(out/'benchmark_tasks.json')==tasks,'physical_profile_unchanged':True,
            'TEST_arrays_read':False,'pilot_started':False})
        progress('report')
        return write_report(out,ledger,status)
