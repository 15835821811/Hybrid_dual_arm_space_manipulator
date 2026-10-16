"""Bounded post-10k coverage and fixed-goal augmentation with native audit.

Training is a subsequent explicit stage, never inferred from data generation.
"""
from pathlib import Path
import os
import hashlib
import subprocess
import time
import zipfile
import numpy as np
from ff_prior.storage import read_json,file_digest,digest,RunLock
from ff_prior_pilot_train.storage import atomic_json,Ledger
from ff_prior.recovery import process_is_alive
from ff_prior.pipeline import audit,acceptance
from ff_prior_pilot_train.experiment import inputs as trained_inputs,sources as parent_sources
from ff_prior_modes.data import RuntimeCache
from ff_prior_loop.experiment import execute
from ff_prior_complex.tools.audit_completed import audit_physics
from ff_prior.initial_state import STATE
from ff_prior.spline import from_arrays
from . import data

ROOT=Path(__file__).resolve().parents[1]
PARENT=data.PREVIOUS
TRAINED=ROOT/'artifacts/v7_1/qualified_pilot_train_002'
PROTOCOL={'version':'qualified_12k_coverage_data_1','profile':'bounded_coverage_after_10k_closure',
    'data':data.CONTRACT,'wall_budget_s':5400,'generation_reserve_for_audit_s':1200,
    'max_output_bytes':20*1024**3,'cpu_threads':2,'max_witness_executions':48,
    'generation_target_new':2048,'fixed_attempts':96,'VAL_unchanged':48,
    'selection':'24 context minima + 12 hash-ordered fixed modes + 12 context-stratified samples; no replacement',
    'gate':'2048 new forward rows, >=48 fixed modes, >=1 per 24 selected targets, >=350 new empty families, 48/48 native replay and raw audit',
    'relative_efficiency_required':False,'TEST_evaluated':False,'strict_multimodal':False,
    'new_training_in_this_stage':False,'new_neural_benefit_claim':False,
    'next_stage':'actual training ingestion and unchanged DEV closure, separately frozen protocol',
    'scope':'12 new stationary zero-momentum origins; 12 new translated/resized TRAIN obstacle scenes; no homotopy claim'}



def sources():
    return dict(parent_sources(),**{p.relative_to(ROOT).as_posix():file_digest(p) for p in sorted((ROOT/'ff_prior_coverage').rglob('*.py'))})


def require_parent():
    for directory in (PARENT,TRAINED):
        state=read_json(directory/'status.json')
        if state['status']!='FINISHED' or process_is_alive(state['pid']):raise ValueError('parent must finish and exit')
        if read_json(directory/'independent_verification.json')['status']!='PASS':raise ValueError('parent audit missing')
    if not read_json(TRAINED/'summary.json')['pilot_training_closure_passed']:raise ValueError('10k training closure missing')


def inputs(source):
    source=Path(source).resolve()
    old,ds,dev,index,inherited=trained_inputs(PARENT)
    files={Path(os.path.relpath(PARENT/p,source)).as_posix():h for p,h in inherited.items()}
    for p in sorted(TRAINED.rglob('*')):
        if p.is_file() and p.suffix!='.tmp' and p.name not in ('launch.guard','run.lock'):
            files[Path(os.path.relpath(p,source)).as_posix()]=file_digest(p)
    return old,ds,dev,index,files


def report(out,ledger,status):
    out=Path(out);rows=[read_json(p) for p in sorted((out/'records').glob('*.json'))]
    eligible=[r for r in rows if r['execution_status']!='FAIL']
    forward=[r for r in eligible if r['sample_id'].startswith('coverage_train_')]
    modes=[r for r in eligible if r['sample_id'].startswith('coverage_mode_')]
    executions=[read_json(p) for p in (out/'witness_executions').glob('*/outcome.json')]
    families={r['family_id'] for r in forward if r['family_id'].startswith('coverage_family_')}
    audited=(out/'independent_verification.json').exists() and read_json(out/'independent_verification.json')['status']=='PASS'
    errors=[k for k,v in ledger.data['completed'].items() if v['status'] in ('ERROR','INTERRUPTED','BLOCKED')]
    mode_targets={r['task']['task_hash'] for r in modes}
    passed=bool(status=='FINISHED' and len(forward)==2048 and len(modes)>=48 and len(mode_targets)==24 and
        len(families)>=350 and len(executions)==48 and all(r['status']=='EXECUTION_VERIFIED' for r in executions)
        and audited and not errors and not ledger.data.get('unresolved_call_accounting'))
    result={'status':status,'new_admitted':len(rows),'new_eligible':len(eligible),'new_forward':len(forward),
        'new_fixed_modes':len(modes),'fixed_targets_with_admission':len(mode_targets),
        'total_TRAIN':10000+len(eligible),'VAL':48,'new_empty_families':len(families),
        'new_obstacle_rows':sum(bool(r['scene']['objects']) for r in eligible),
        'executions_completed':len(executions),'execution_verified':sum(r['status']=='EXECUTION_VERIFIED' for r in executions),
        'new_MODEL_VALID_only':sum(r['execution_status']=='NOT_RUN' for r in eligible),
        'coverage_data_gate_passed':passed,'ready_for_training_stage':passed,'independent_audit_passed':audited,
        'errors':errors,'unresolved_calls':ledger.data.get('unresolved_call_accounting',[]),
        'training_executed':False,'TEST_executed':False,'diffusion_advantage':'NOT_EVALUATED_IN_DATA_STAGE',
        'inherited_failure_accounting':read_json(TRAINED/'summary.json')['recovery_accounting'],
        'full_goal_achieved':False,'calls':ledger.data['calls'],'protocol':PROTOCOL}
    atomic_json(out/'summary.json',result)
    text=['# 10k闭环后的有界覆盖扩库','',f'状态：{status}；新前向 {len(forward)}/2048，固定目标修正 {len(modes)}/96次尝试，累计 TRAIN {10000+len(eligible)}，VAL 48。',
        f'预选执行及重放 {result["execution_verified"]}/48；其余 {result["new_MODEL_VALID_only"]} 条仅 MODEL_VALID。',
        f'数据审计：{audited}；实际训练接续资格：{passed}。','',
        '12个新静止初态与12个平移/缩放的新TRAIN障碍场景；形状和运动幅度循环；不宣称绕行同伦类别。',
        '主库前向目标只派生一次并冻结；修正集必须到达原冻结世界目标，不沿用单纯关节终点标签。',
        '本阶段不运行新训练或TEST，不把入库和抽查批量升级为全部执行通过，不宣称Diffusion收益。',
        '48例执行前冻结、失败不替补；仅不穿透、自由漂浮、17到67映射、完整初态、全机几何及原门槛保留。',
        '查询和耗时仅辅助核算；不设相对效率胜出门槛。不自动启动10万条。']
    (out/'REPORT.md').write_text('\n'.join(text)+'\n',encoding='utf-8')
    return result


def verify(out,ledger,identity,original):
    out=Path(out);accepted=[];hashes={data.reference_hash(e) for e in original}
    source=Path(identity['source'])
    _,_,dev,_,_=inputs(source)
    heldout=data.heldout_goals(original,dev,source)
    for p in sorted((out/'records').glob('*.json')):
        r=data.check_new_record(out,p,ledger)
        assert not data.goal_neighbours(r['task'],heldout)
        with np.load(out/r['arrays']) as a:
            ref=from_arrays(a);assert ref.reference_hash==r['reference_hash']
            for name,order in [('view_q',0),('view_qd',1),('view_qdd',2)]:
                assert np.allclose(a[name],ref.evaluate(a['view_time'],order),rtol=0,atol=1e-12)
        with np.load(out/r['complete_initial_state']) as a:
            assert digest({'values':a['values'],'model_hash':r['task']['contract']['robot_hash'],'state_mask':int(STATE)})==r['task']['contract']['initial_state_id']
        if r['reference_hash'] in hashes:raise ValueError('duplicate reference')
        hashes.add(r['reference_hash'])
        if r['execution_status']!='FAIL':accepted.append(r)
    index=read_json(out/'training_index.json')['records']
    assert {e['sample_id'] for e in index if e['sample_id'].startswith(('coverage_train_','coverage_mode_'))}=={r['sample_id'] for r in accepted}
    for e in index:
        for k in ('record','arrays'):assert file_digest(ROOT/e[k])==e[k+'_sha256']
    modes=read_json(out/'mode_selection.json');assert len(modes)==24
    forward=[read_json(p) for p in (out/'records').glob('coverage_train_*.json')]
    for entry in modes:
        parent=min((r for r in forward if r['context_id']==entry['context_id']),key=lambda r:digest({'coverage_mode':r['sample_id']}))
        assert entry['parent_sample_id']==parent['sample_id'] and entry['task']==parent['task']
        assert entry['family_id']==parent['family_id'] and entry['frozen_before_mode_generation']
        with np.load(out/parent['arrays']) as a:assert np.array_equal(a['Z'],entry['Z'])
        references=[np.asarray(entry['Z'])]
        for p in sorted((out/'records').glob(entry['sample_id']+'_v*.json')):
            r=read_json(p);assert r['task']==entry['task'] and r['family_id']==entry['family_id']
            with np.load(out/r['arrays']) as a:z=a['Z'].copy()
            assert min(float(np.sqrt(np.mean((z-v)**2))) for v in references)>=.005
            references.append(z)
    selection=read_json(out/'audit_selection.json');steps=0;levels=[]
    assert len(set(selection['risk']+selection['fixed_modes']+selection['random_stratified']))==48
    allrows=[read_json(p) for p in (out/'records').glob('*.json')]
    assert {r['context_id'] for r in allrows}==data.expected_contexts()
    for c in {r['context_id'] for r in allrows}:
        worst=min((r for r in allrows if r['context_id']==c),key=lambda r:(r['validation']['geometry']['minimum_clearance_m'],r['sample_id']))
        assert worst['sample_id'] in selection['risk']
    expected_modes=sorted((r['sample_id'] for r in allrows if r['sample_id'].startswith('coverage_mode_') and r['sample_id'] not in selection['risk']),
                          key=lambda sid:digest({'coverage_mode_audit':sid}))[:12]
    assert selection['fixed_modes']==expected_modes
    by_id={r['sample_id']:r for r in allrows}
    random_contexts=[by_id[sid]['context_id'] for sid in selection['random_stratified']]
    assert len(set(random_contexts))==12 and sum(c.startswith('origin_') for c in random_contexts)==6
    for prefix in ('origin_','obstacle_'):
        chosen=sorted((v for v in data.expected_contexts() if v.startswith(prefix)),key=lambda v:digest({'coverage_context':v}))[:6]
        assert set(chosen)<=set(random_contexts)
    for sid in selection['random_stratified']:
        pool=[r['sample_id'] for r in allrows if r['context_id']==by_id[sid]['context_id'] and r['sample_id'] not in selection['risk']+selection['fixed_modes']]
        assert sid==min(pool,key=lambda v:digest({'coverage_audit':v}))
    for sid in selection['risk']+selection['fixed_modes']+selection['random_stratified']:
        r=read_json(out/'records'/f'{sid}.json');directory=out/'witness_executions'/sid
        p,n=audit_physics(directory,r['task'],r['reference_hash']);steps+=n;levels.append(p['level'])
        assert (p['level']=='EXECUTION_VERIFIED')==(r['execution_status']=='PASS')
    generated=[read_json(p) for p in (out/'generation').glob('*.json')]
    assert len(generated)==ledger.data['attempts']['generation']+ledger.data['attempts']['fixed_generation']
    assert ledger.data['attempts']['fixed_generation']==96 and ledger.data['attempts']['generation']<=3600
    assert sum(r['calls']['coupled'] for r in generated)==ledger.data['calls']['generation.coupled']+ledger.data['calls']['fixed_generation.coupled']
    assert all(not any(r['calls'][k] for k in ('physics_steps','torque_rollout','rrt','sqp')) for r in generated)
    assert all(r['calls']['coupled']<=(113 if r['sample_id'].startswith('coverage_mode_') else 4) for r in generated)
    assert steps==ledger.data['calls']['witness.physics_steps']
    assert sources()==identity['source_hashes']
    with zipfile.ZipFile(out/'executed_source.zip') as z:
        assert all(hashlib.sha256(z.read(p)).hexdigest()==h for p,h in sources().items())
    assert all(file_digest(source/p)==h for p,h in identity['source_files'].items())
    preserved=read_json(ROOT/'ff_prior_expand/evidence/preservation_before.json')
    for group in ('old_files','user_edits'):assert all(file_digest(ROOT/p)==h for p,h in preserved[group].items())
    for branch,h in preserved['branches'].items():assert subprocess.check_output(['git','rev-parse',branch],text=True).strip()==h
    return {'status':'PASS','new_eligible':len(accepted),'actual_arrays_verified':len(index),'native_traces':48,
        'levels':levels,'physics_steps_including_replay':steps,'all_context_minima_audited':True,
        'parent_data_training_outputs_unchanged':True,'old_files_user_edits_branches_unchanged':True,
        'additional_physics_calls':0,'training_or_TEST_executed':False}


def run(source,out,progress,resume=False,previous_elapsed=0.):
    source,out=Path(source).resolve(),Path(out).resolve()
    deadline=time.time()+max(0,5400-previous_elapsed);generation_deadline=deadline-1200
    require_parent()
    old,_,dev,_,files=inputs(source)
    identity={'protocol':PROTOCOL,'source':str(source),'source_files':files,'tasks_hash':digest(dev),'source_hashes':sources()}
    with RunLock(out):
        tests=read_json(ROOT/'ff_prior_coverage/evidence/software_tests.json')
        if tests['status']!='PASS' or tests['source_hashes']!=sources() or tests['junit_sha256']!=file_digest(ROOT/'ff_prior_coverage/evidence/tests.xml'):raise ValueError('software proof stale')
        if (out/'run.json').exists() and (not resume or read_json(out/'run.json')['identity']!=identity):raise ValueError('run identity changed')
        physical,canonical,initial,_=audit(old['config']);profile=acceptance(old['config'])
        assert physical['acceptance']==read_json(PARENT/'run.json')['acceptance']
        original=data.old_index();contexts=data.contexts(canonical,initial,profile,original)
        if not (out/'run.json').exists():
            atomic_json(out/'run.json',{'identity':identity,'acceptance':physical['acceptance'],'producer_head':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()})
            atomic_json(out/'contexts.json',[dict(meta,initial_state_id=ii.state_id) for meta,rr,ii,cc in contexts])
            atomic_json(out/'software_tests.json',tests)
            with zipfile.ZipFile(out/'executed_source.zip','w',zipfile.ZIP_DEFLATED) as z:
                for p in sources():z.write(ROOT/p,p)
        ledger=Ledger(out,{'run_identity_hash':digest(identity)})
        heldout=data.heldout_goals(original,dev,source);hashes={data.reference_hash(e) for e in original}
        admitted=0
        for p in sorted((out/'records').glob('*.json')):
            r=data.check_new_record(out,p,ledger);hashes.add(r['reference_hash'])
            admitted+=int(r['sample_id'].startswith('coverage_train_'))
        for attempt in range(3600):
            sid=f'coverage_train_{attempt:05d}'
            if sid in ledger.data['completed']:continue
            if admitted>=2048 or time.time()>=generation_deadline:break
            if attempt%100==0 and sum(p.stat().st_size for p in out.rglob('*') if p.is_file())>=PROTOCOL['max_output_bytes']:raise RuntimeError('STORAGE_BUDGET_EXHAUSTED')
            ledger.begin(sid,'generation',4200,3600)
            progress('coverage_forward_generation',attempt=attempt+1,admitted=admitted,total_TRAIN=10000+admitted)
            result=data.generate(attempt,contexts[(attempt//2)%24],out,profile,heldout,hashes,generation_deadline)
            atomic_json(out/'generation'/f'{sid}.json',result);ledger.finish({'status':result['status']},result['calls'])
            if result['admitted']:admitted+=1;hashes.add(result['reference_hash'])
            if (attempt+1)%100==0:report(out,ledger,'RUNNING')
        cache=RuntimeCache(canonical,initial,profile)
        entries=data.select_modes(out,ledger)
        for entry in entries:
            for k in range(4):
                key=entry['sample_id']+f'_v{k}'
                if key in ledger.data['completed']:continue
                if time.time()>=generation_deadline:raise RuntimeError('FIXED_MODE_STAGE_BUDGET_EXHAUSTED')
                ledger.begin(key,'fixed_generation',4200,96)
                progress('fixed_TRAIN_world_modes',target=entry['slot']+1,variant=k+1)
                result=data.generate_mode(entry,k,cache,out,profile,generation_deadline)
                atomic_json(out/'generation'/f'{key}.json',result);ledger.finish({'status':result['status']},result['calls'])
        selection=data.select_audits(out,ledger)
        for sid in selection['risk']+selection['fixed_modes']+selection['random_stratified']:
            key='witness_'+sid;directory=out/'witness_executions'/sid
            if key in ledger.data['completed']:continue
            if time.time()>=deadline:raise RuntimeError('TOTAL_BUDGET_EXHAUSTED')
            path=out/'records'/f'{sid}.json';r=data.check_new_record(out,path,ledger)
            rr,ii,cc,_=cache.get(r);before=rr.calls.copy()
            with np.load(out/r['arrays']) as a:z=a['Z'].copy()
            ledger.begin(key,'witness',1200,48);progress('coverage_native_audit',sample=sid)
            result=execute(directory,r['task'],z,rr,ii,cc,profile,deadline)
            r.update(execution_status='PASS' if result['status']=='EXECUTION_VERIFIED' else 'FAIL',training_eligible=result['status']=='EXECUTION_VERIFIED')
            if r['training_eligible']:r['validation_level']='EXECUTION_VERIFIED'
            r['execution_report']=f'witness_executions/{sid}/report.json'
            atomic_json(path,r);ledger.finish(result,{k:rr.calls[k]-before[k] for k in before})
            report(out,ledger,'RUNNING')
        atomic_json(out/'training_index.json',{'records':data.combined(out,ledger),'actual_training_consumed':False,'TEST_arrays_read':False})
        ledger.index();progress('coverage_independent_raw_audit')
        proof=verify(out,ledger,identity,original);atomic_json(out/'independent_verification.json',proof)
        result=report(out,ledger,'FINISHED')
        if not result['coverage_data_gate_passed']:result=report(out,ledger,'FINISHED_WITH_BLOCKED_ITEMS')
        return result
