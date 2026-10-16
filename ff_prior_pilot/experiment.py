"""Stage one of authorized pilot: qualified coverage and individual native audit.

Training is a subsequent explicit stage, never inferred from data generation.
"""
from pathlib import Path
import os
import hashlib
import subprocess
import time
import zipfile
import numpy as np
from ff_prior.storage import read_json,atomic_json,file_digest,digest,Ledger,RunLock
from ff_prior.recovery import process_is_alive
from ff_prior.pipeline import audit,acceptance
from ff_prior_expand.experiment import inputs as parent_inputs
from ff_prior_boundary.experiment import sources as parent_sources
from ff_prior_modes.data import RuntimeCache
from ff_prior_loop.experiment import execute
from ff_prior_complex.tools.audit_completed import audit_physics
from ff_prior.initial_state import STATE
from ff_prior.spline import from_arrays
from . import data

ROOT=Path(__file__).resolve().parents[1]
PARENT=data.PREVIOUS
BOUNDARY=ROOT/'artifacts/v7_1/expansion_boundary_audit_001'
PROTOCOL={'version':'qualified_10k_pilot_data_1','profile':'pilot_data_generation_and_native_audit',
    'data':data.CONTRACT,'wall_budget_s':7200,'generation_reserve_for_audit_s':900,
    'max_output_bytes':40*1024**3,'cpu_threads':2,'max_witness_executions':48,
    'generation_target_new':9000,'cumulative_TRAIN_target':10000,'VAL_unchanged':48,
    'selection':'24 context minima/global lowest clearance + 24 hash-stratified non-overlapping samples; no replacement',
    'gate':'10000 eligible TRAIN, 2500 new empty families, 48/48 native execution/replay, raw audit, no unknown calls/errors',
    'relative_efficiency_required':False,'TEST_evaluated':False,'strict_multimodal':False,
    'new_training_in_this_stage':False,'new_neural_benefit_claim':False,
    'next_stage':'actual training consumption and frozen DEV closure; not automatically marked complete',
    'scope':'12 new stationary zero-momentum origins; 3 existing TRAIN obstacle scenes; no new scene claim'}


def sources():
    return dict(parent_sources(),**{p.relative_to(ROOT).as_posix():file_digest(p) for p in sorted((ROOT/'ff_prior_pilot').rglob('*.py'))})


def require_parent():
    for directory in (PARENT,BOUNDARY):
        s=read_json(directory/'status.json')
        if s['status']!='FINISHED' or process_is_alive(s['pid']):raise ValueError('parent is not completed and exited')
        if read_json(directory/'independent_verification.json')['status']!='PASS':raise ValueError('parent audit missing')
    if not read_json(PARENT/'summary.json')['PILOT_ELIGIBLE']:raise ValueError('parent closed loop not eligible')
    if not read_json(BOUNDARY/'summary.json')['boundary_gate_passed']:raise ValueError('boundary gate failed')
    overlay=read_json(BOUNDARY/'eligibility_overlay.json')
    for sid,e in overlay.items():
        r=read_json(PARENT/'records'/f'{sid}.json')
        if not e['training_eligible'] or e['source_reference_hash']!=r['reference_hash']:raise ValueError('boundary exclusion or wrong reference')
    return overlay


def inputs(source):
    old,ds,dev,rows,files=parent_inputs(source)
    source=Path(source).resolve()
    # Bind all actual parent outputs, including checkpoints and raw traces.
    for root in (PARENT,BOUNDARY):
        for p in sorted(root.rglob('*')):
            if p.is_file() and p.suffix!='.tmp' and p.name not in ('launch.guard','run.lock'):
                files[Path(os.path.relpath(p,source)).as_posix()]=file_digest(p)
    return old,ds,dev,rows,files


def report(out,ledger,status):
    out=Path(out);rows=[read_json(p) for p in sorted((out/'records').glob('*.json'))]
    eligible=[r for r in rows if r['execution_status']!='FAIL']
    executions=[read_json(p) for p in (out/'witness_executions').glob('*/outcome.json')]
    families={r['family_id'] for r in eligible if r['family_id'].startswith('pilot_family_')}
    audit_file=out/'independent_verification.json'
    audited=audit_file.exists() and read_json(audit_file)['status']=='PASS'
    errors=[k for k,v in ledger.data['completed'].items() if v['status'] in ('ERROR','INTERRUPTED','BLOCKED')]
    passed=bool(status=='FINISHED' and len(eligible)+1000>=10000 and len(families)>=2500 and
        len(executions)==48 and all(r['status']=='EXECUTION_VERIFIED' for r in executions) and audited and
        not errors and not ledger.data.get('unresolved_call_accounting'))
    result={'status':status,'generation_attempts':ledger.data['attempts'].get('generation',0),
        'new_admitted':len(rows),'new_eligible':len(eligible),'total_TRAIN':1000+len(eligible),'VAL':48,
        'new_empty_families':len(families),'new_obstacle_rows':sum(bool(r['scene']['objects']) for r in eligible),
        'executions_completed':len(executions),'execution_verified':sum(r['status']=='EXECUTION_VERIFIED' for r in executions),
        'new_MODEL_VALID_only':sum(r['execution_status']=='NOT_RUN' for r in eligible),
        'pilot_data_gate_passed':passed,'ready_for_training_stage':passed,'independent_audit_passed':audited,
        'errors':errors,'unresolved_calls':ledger.data.get('unresolved_call_accounting',[]),
        'training_executed':False,'TEST_executed':False,'diffusion_advantage':'NOT_EVALUATED_IN_DATA_STAGE',
        'full_goal_achieved':False,'calls':ledger.data['calls'],'protocol':PROTOCOL}
    atomic_json(out/'summary.json',result)
    text=['# 10,000条有界扩库 pilot：数据阶段','',f'状态：{status}；本轮合格 {len(eligible)}；累计 TRAIN {1000+len(eligible)}，VAL 48。',
        f'预选执行及独立重放通过 {result["execution_verified"]}/48；其余 {result["new_MODEL_VALID_only"]} 条仅 MODEL_VALID。',
        f'独立审计：{audited}；进入实际训练阶段资格：{passed}。','',
        '本阶段生成数据并逐条准入，不运行新训练、DEV或最终TEST，不能声称新模型或Diffusion收益改善。数据阶段通过后仍须实际加载训练并完成冻结对照。',
        '12组新的静止零动量初态，3个既有TRAIN障碍场景；不声称新增障碍场景或绕行同伦类别。',
        '保留完整初态、17→67映射、自由漂浮、全机几何、真实力矩及重放；失败不重标目标、不替换预选样本。查询/时间为资源账本，不设相对效率优势门槛。']
    (out/'REPORT.md').write_text('\n'.join(text)+'\n',encoding='utf-8')
    return result


def verify(out,ledger,identity,original):
    out=Path(out);accepted=[];hashes={data.reference_hash(e) for e in original}
    source=Path(identity['source'])
    _,_,dev,_,_=parent_inputs(source)
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
    assert {e['sample_id'] for e in index if e['sample_id'].startswith('pilot_train_')}=={r['sample_id'] for r in accepted}
    for e in index:
        for k in ('record','arrays'):assert file_digest(ROOT/e[k])==e[k+'_sha256']
    selection=read_json(out/'audit_selection.json');steps=0;levels=[]
    assert len(set(selection['risk']+selection['random_stratified']))==48
    allrows=[read_json(p) for p in (out/'records').glob('*.json')]
    assert {r['context_id'] for r in allrows}==data.expected_contexts()
    for c in {r['context_id'] for r in allrows}:
        worst=min((r for r in allrows if r['context_id']==c),key=lambda r:(r['validation']['geometry']['minimum_clearance_m'],r['sample_id']))
        assert worst['sample_id'] in selection['risk']
    for sid in selection['risk']+selection['random_stratified']:
        r=read_json(out/'records'/f'{sid}.json');directory=out/'witness_executions'/sid
        p,n=audit_physics(directory,r['task'],r['reference_hash']);steps+=n;levels.append(p['level'])
        assert (p['level']=='EXECUTION_VERIFIED')==(r['execution_status']=='PASS')
    generated=[read_json(p) for p in (out/'generation').glob('*.json')]
    assert len(generated)==ledger.data['attempts']['generation']
    assert sum(r['calls']['coupled'] for r in generated)==ledger.data['calls']['generation.coupled']
    assert all(not any(r['calls'][k] for k in ('physics_steps','torque_rollout','rrt','sqp')) for r in generated)
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
        'parent_and_boundary_outputs_unchanged':True,'old_files_user_edits_branches_unchanged':True,
        'additional_physics_calls':0,'training_or_TEST_executed':False}


def run(source,out,progress,resume=False,previous_elapsed=0.):
    source,out=Path(source).resolve(),Path(out).resolve()
    require_parent()
    old,_,dev,_,files=inputs(source)
    identity={'protocol':PROTOCOL,'source':str(source),'source_files':files,'tasks_hash':digest(dev),'source_hashes':sources()}
    with RunLock(out):
        tests=read_json(ROOT/'ff_prior_pilot/evidence/software_tests.json')
        if tests['status']!='PASS' or tests['source_hashes']!=sources() or tests['junit_sha256']!=file_digest(ROOT/'ff_prior_pilot/evidence/tests.xml'):raise ValueError('software proof stale')
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
        deadline=time.time()+max(0,7200-previous_elapsed);generation_deadline=deadline-900
        heldout=data.heldout_goals(original,dev,source);hashes={data.reference_hash(e) for e in original}
        admitted=0
        for p in sorted((out/'records').glob('*.json')):
            r=data.check_new_record(out,p,ledger);hashes.add(r['reference_hash']);admitted+=1
        for attempt in range(14000):
            sid=f'pilot_train_{attempt:05d}'
            if sid in ledger.data['completed']:continue
            if admitted>=9000 or time.time()>=generation_deadline:break
            if attempt%100==0 and sum(p.stat().st_size for p in out.rglob('*') if p.is_file())>=PROTOCOL['max_output_bytes']:raise RuntimeError('STORAGE_BUDGET_EXHAUSTED')
            ledger.begin(sid,'generation',6300,14000)
            progress('pilot_forward_generation',attempt=attempt+1,admitted=admitted,total_TRAIN=1000+admitted)
            result=data.generate(attempt,contexts[(attempt//2)%15],out,profile,heldout,hashes,generation_deadline)
            atomic_json(out/'generation'/f'{sid}.json',result);ledger.finish({'status':result['status']},result['calls'])
            if result['admitted']:admitted+=1;hashes.add(result['reference_hash'])
            if (attempt+1)%100==0:report(out,ledger,'RUNNING')
        selection=data.select_audits(out,ledger);cache=RuntimeCache(canonical,initial,profile)
        for sid in selection['risk']+selection['random_stratified']:
            key='witness_'+sid;directory=out/'witness_executions'/sid
            if key in ledger.data['completed']:continue
            if time.time()>=deadline:raise RuntimeError('TOTAL_BUDGET_EXHAUSTED')
            path=out/'records'/f'{sid}.json';r=data.check_new_record(out,path,ledger)
            rr,ii,cc,_=cache.get(r);before=rr.calls.copy()
            with np.load(out/r['arrays']) as a:z=a['Z'].copy()
            ledger.begin(key,'witness',900,48);progress('pilot_native_audit',sample=sid)
            result=execute(directory,r['task'],z,rr,ii,cc,profile,deadline)
            r.update(execution_status='PASS' if result['status']=='EXECUTION_VERIFIED' else 'FAIL',training_eligible=result['status']=='EXECUTION_VERIFIED')
            if r['training_eligible']:r['validation_level']='EXECUTION_VERIFIED'
            r['execution_report']=f'witness_executions/{sid}/report.json'
            atomic_json(path,r);ledger.finish(result,{k:rr.calls[k]-before[k] for k in before})
            report(out,ledger,'RUNNING')
        atomic_json(out/'training_index.json',{'records':data.combined(out,ledger),'actual_training_consumed':False,'TEST_arrays_read':False})
        ledger.index();progress('pilot_independent_raw_audit')
        proof=verify(out,ledger,identity,original);atomic_json(out/'independent_verification.json',proof)
        return report(out,ledger,'FINISHED')
