"""A new qualification, never retroactive passage of the previous failed gate."""
import hashlib
import subprocess
import time
import zipfile
from pathlib import Path
import numpy as np
from ff_prior.storage import read_json,file_digest,digest,RunLock
from ff_prior_pilot_train.storage import atomic_json,Ledger
from ff_prior.recovery import process_is_alive
from ff_prior.pipeline import audit,acceptance
from ff_prior_origin_coverage.experiment import identity_for as original_identity,sources as original_sources
from ff_prior_coverage.data import check_new_record
from ff_prior_generalization.experiment import execute
from ff_prior_scene_guard.runtime import at_scene_origin
from ff_prior_complex.tools.audit_completed import audit_physics
from ff_prior.initial_state import STATE
from ff_prior.spline import from_arrays
from .protocol import ROOT,SOURCE,DIAGNOSTIC,SLOTS,PROTOCOL as P
from . import data


def sources():
    result=original_sources()
    for package in ('ff_prior_scene_guard','ff_prior_origin_fill'):
        result.update({p.relative_to(ROOT).as_posix():file_digest(p) for p in sorted((ROOT/package).rglob('*.py'))})
    return result


def identity_for(source):
    source=Path(source).resolve()
    if source!=SOURCE.resolve():raise ValueError('exact partially completed data parent required')
    state=read_json(source/'status.json');summary=read_json(source/'summary.json')
    if state['status']!='FINISHED_WITH_BLOCKED_ITEMS' or process_is_alive(state['pid']):raise ValueError('parent must finish and exit')
    if summary['ready_for_training_stage'] or summary['data_gate_passed']:raise ValueError('parent original failed gate must be preserved')
    if read_json(source/'independent_verification.json')['status']!='PASS' or summary['native_verified']!=32:
        raise ValueError('qualified records and all inherited native audits required')
    if summary['unresolved_calls']:raise ValueError('unknown parent work cannot be silently refunded')
    frozen=read_json(source/'run.json')['identity']
    expected,context=original_identity(frozen['source'])
    if frozen!=expected:raise ValueError('original run identity changed')
    config,dev,base,_,heldout,old_origins=context
    oldledger=read_json(source/'ledger.json');oldrows=[]
    for path in sorted((source/'records').glob('*.json')):
        r=check_new_record(source,path)
        if oldledger['completed'][r['sample_id']]['status']!='ADMITTED' or r['execution_status']=='FAIL':raise ValueError('unqualified inherited record')
        oldrows.append(r);old_origins.add(digest(r['condition']['initial_features']))
    forward=[r for r in oldrows if r['data_role']=='INDEPENDENT_ORIGIN_TRAIN']
    missing=sorted(set(range(512))-{r['slot'] for r in forward})
    if missing!=list(SLOTS) or len(forward)!=503 or sum(bool(r['scene']['objects']) for r in forward)!=119:
        raise ValueError('frozen missing coverage differs')
    if sum(r['split']=='TRAIN' for r in oldrows)!=535 or len(oldrows)!=543:raise ValueError('inherited population differs')
    diagnosis=read_json(DIAGNOSTIC)
    if not (diagnosis['status']=='DIAGNOSED' and diagnosis['inputs_unchanged'] and len(diagnosis['cases'])==39):raise ValueError('complete retained error diagnosis required')
    if set(summary['errors'])!={c['sample_id'] for c in diagnosis['cases']}:raise ValueError('diagnosis not bound to all parent errors')
    if not all(file_digest(p)==h for p,h in diagnosis['inputs_sha256'].items()):raise ValueError('diagnosed inputs changed')
    if not all(c['explained_by_initial_penetration'] and c['different_state_fields']==['mjSTATE_WARMSTART'] and
        c['actual_initial_geometry']['minimum_clearance_m']<0 for c in diagnosis['cases']):raise ValueError('unexplained parent error')
    validation=read_json(source/'frozen_validation.json')
    if len(validation['entries'])!=8:raise ValueError('unchanged independent validation required')
    guard=heldout+[{'sample_id':e['sample_id'],'family_id':e['family_id'],'split':'NEW_VAL','goals':e['task']['condition']['goals']} for e in validation['entries']]
    valq=[np.asarray(e['task']['condition']['initial_features']) for e in validation['entries']]
    oldindex=read_json(source/'training_index.json')['records']
    if sum(e['split']=='TRAIN' for e in oldindex)!=12679 or sum(e['split']=='VAL' for e in oldindex)!=48:raise ValueError('parent index differs')
    files=dict(frozen['source_files'])
    for p in sorted(source.rglob('*')):
        if p.is_file() and p.name not in ('run.lock','launch.guard') and p.suffix!='.tmp':files[p.relative_to(ROOT).as_posix()]=file_digest(p)
    files[DIAGNOSTIC.relative_to(ROOT).as_posix()]=file_digest(DIAGNOSTIC)
    identity={'protocol':P,'source':str(source),'source_files':files,'source_hashes':sources(),
        'guard_hash':digest(guard),'original_DEV_hash':digest(dev),'original_validation_hash':digest(validation),
        'parent_original_gate':False,'parent_elapsed_s':state['elapsed_s'],'historical_errors':summary['errors']}
    return identity,(config,dev,oldindex,oldrows,guard,valq,old_origins,validation)


def report(out,ledger,status):
    out=Path(out);rows=data.rows(out,ledger)
    outcomes=[read_json(p) for p in (out/'witness_executions').glob('*/outcome.json')]
    errors=[k for k,v in ledger.data['completed'].items() if v['status'] in ('ERROR','INTERRUPTED','BLOCKED')]
    proof=read_json(out/'independent_verification.json') if (out/'independent_verification.json').exists() else {}
    n=sum(r['execution_status']!='FAIL' for r in rows)
    passed=bool(status=='FINISHED' and len(rows)==9 and {r['slot'] for r in rows}==set(SLOTS)
        and len(outcomes)==9 and all(r['status']=='EXECUTION_VERIFIED' for r in outcomes)
        and not errors and not ledger.data.get('unresolved_call_accounting') and proof.get('status')=='PASS')
    result={'status':status,'new_forward':len(rows),'new_unique_origins':len({r['origin_hash'] for r in rows}),
        'merged_forward':503+n,'merged_obstacle_forward':119+n,'merged_fixed_modes':32,
        'total_TRAIN':12679+n,'original_VAL':48,'frozen_NEW_VAL':8,
        'new_native_verified':sum(r['status']=='EXECUTION_VERIFIED' for r in outcomes),'inherited_native_verified':32,
        'merged_data_gate_passed':passed,'ready_for_training_stage':passed,'independent_audit_passed':proof.get('status')=='PASS',
        'parent_original_gate_passed':False,'parent_original_status':'FINISHED_WITH_BLOCKED_ITEMS','historical_ERROR_retained':39,
        'errors':errors,'unresolved_calls':ledger.data.get('unresolved_call_accounting',[]),'calls':ledger.data['calls'],
        'training_executed':False,'TEST_executed':False,'diffusion_benefit':'NOT_EVALUATED_IN_DATA_STAGE',
        'relative_efficiency_required':False,'full_goal_achieved':False,'protocol':P}
    atomic_json(out/'summary.json',result)
    lines=['# 缺失初态的独立有界补充与新合并资格','',
        f'状态{status}；补充 {len(rows)}/9 个新球障碍初态，新增执行/重放 {result["new_native_verified"]}/9；新合并资格 {passed}。',
        f'累计TRAIN {12679+n}，原VAL48，新验证8不训练；继承32条执行证据，新9条按各自证据分级。',
        '原qualified_origin_coverage_001仍FINISHED_WITH_BLOCKED_ITEMS、ready_for_training_stage=false，39 ERROR和66拒绝保留，原预算不退款。',
        '新任务新seed和提案ID；旧失败世界目标不重标，既有验证不替换；初始化先拒绝真实穿透，保留完整状态及187转移检查。',
        '新合并门通过也不是实际训练或Diffusion收益；后续必须真实训练及原DEV/新验证同预算对照。',
        '仅不穿透和全部物理门槛保留；查询与时间辅助，不自动最终TEST或十万规模。']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8');return result


def verify(out,ledger,identity,oldindex,oldrows,guard,valq,old_origins,validation):
    out=Path(out);problems=[];physics=[];steps=0
    try:
        assert not ledger.data['inflight'] and sources()==identity['source_hashes']
        assert all(file_digest(ROOT/p)==h for p,h in identity['source_files'].items())
        preserved=read_json(ROOT/'ff_prior_expand/evidence/preservation_before.json')
        for group in ('old_files','user_edits'):assert all(file_digest(ROOT/p)==h for p,h in preserved[group].items())
        for branch,h in preserved['branches'].items():assert subprocess.check_output(['git','rev-parse',branch],cwd=ROOT,text=True).strip()==h
        with zipfile.ZipFile(out/'executed_source.zip') as z:assert all(hashlib.sha256(z.read(p)).hexdigest()==h for p,h in sources().items())
        rows=data.rows(out,ledger);allorig=set(old_origins);refs=set()
        for r in rows:
            assert r['scene']['objects'] and r['scene_parent'] is None and not r['source_failed_task_reused']
            assert r['origin_hash'] not in allorig;allorig.add(r['origin_hash'])
            assert not data.neighbours(r['condition']['goals'],guard)
            assert all(np.max(abs(np.asarray(r['condition']['initial_features'])-v))>=.02 for v in valq)
            with np.load(out/r['arrays']) as a:
                ref=from_arrays(a);assert ref.reference_hash==r['reference_hash'] and ref.reference_hash not in refs;refs.add(ref.reference_hash)
                for name,order in [('view_q',0),('view_qd',1),('view_qdd',2)]:assert np.allclose(a[name],ref.evaluate(a['view_time'],order),rtol=0,atol=1e-12)
            with np.load(out/r['complete_initial_state']) as a:assert digest({'values':a['values'],'model_hash':r['task']['contract']['robot_hash'],'state_mask':int(STATE)})==r['task']['contract']['initial_state_id']
            assert r['validation_level']=='EXECUTION_VERIFIED' and r['execution_status']=='PASS'
        selected=read_json(out/'audit_selection.json')
        assert selected['sample_ids']==[r['sample_id'] for r in rows] and selected['before_execution'] and not selected['replacement_after_failure']
        for r in rows:
            assert ledger.data['completed']['witness_'+r['sample_id']]['status']=='EXECUTION_VERIFIED'
            p,n=audit_physics(out/'witness_executions'/r['sample_id'],r['task'],r['reference_hash']);physics.append(p);steps+=n
        newsteps=steps
        oldselection=read_json(SOURCE/'audit_selection.json')
        for sid in oldselection['risk']+oldselection['fixed_modes']+oldselection['hash_ordered']+oldselection['validation']:
            r=next(r for r in oldrows if r['sample_id']==sid)
            p,n=audit_physics(SOURCE/'witness_executions'/sid,r['task'],r['reference_hash']);physics.append(p);steps+=n
        assert newsteps==ledger.data['calls'].get('witness.physics_steps',0)
        forward=[r for r in oldrows if r['data_role']=='INDEPENDENT_ORIGIN_TRAIN']+rows
        assert {r['slot'] for r in forward}==set(range(512)) and len(forward)==512
        assert len({r['origin_hash'] for r in forward})==512 and sum(bool(r['scene']['objects']) for r in forward)==128
        assert read_json(out/'frozen_validation.json')==validation==read_json(SOURCE/'frozen_validation.json')
        index=read_json(out/'training_index.json')['records'];assert index==data.merged_index(out,oldindex,ledger)
        assert sum(e['split']=='TRAIN' for e in index)==12688 and sum(e['split']=='VAL' for e in index)==48
        assert not {e['sample_id'] for e in validation['entries']}&{e['sample_id'] for e in index}
        trainorig=set()
        for e in index:
            assert all(file_digest(ROOT/e[k])==e[k+'_sha256'] for k in ('record','arrays'))
            if e['split']=='TRAIN':trainorig.add(digest(read_json(ROOT/e['record'])['condition']['initial_features']))
        assert len(trainorig)==555
        generated=[read_json(p) for p in (out/'generation').glob('*.json')]
        assert len(generated)==ledger.data['attempts'].get('generation',0)<=72
        assert sum(r['calls']['coupled'] for r in generated)==ledger.data['calls'].get('generation.coupled',0)
        assert all(r['calls']['coupled']<=4 and not any(r['calls'][k] for k in ('physics_steps','torque_rollout','rrt','sqp')) for r in generated)
        assert read_json(SOURCE/'summary.json')['ready_for_training_stage'] is False
    except Exception as error:
        import traceback
        problems.append({'reason':repr(error),'traceback':traceback.format_exc()})
    if ledger.data.get('unresolved_call_accounting'):problems.append('UNKNOWN_INTERRUPTED_WORK')
    return {'status':'PASS' if not problems else 'BLOCKED','problems':problems,'audited_native_traces':len(physics),
        'physics_steps_including_inherited_replay':steps,'new_physics_steps':ledger.data['calls'].get('witness.physics_steps',0),
        'minimum_execution_clearance_m':min((p['minimum_clearance_m'] for p in physics),default=None),
        'maximum_hold_position_m':max((max(p['hold_max_position_error_m']) for p in physics),default=None),
        'original_gate_errors_and_files_unchanged':not problems,'user_edits_branches_unchanged':not problems,
        'additional_audit_physics_calls':0,'training_or_TEST_executed':False,'full_goal_achieved':False}


def run(source,out,progress,resume=False,previous_elapsed=0.):
    source,out=Path(source).resolve(),Path(out).resolve()
    if source==out or source in out.parents or out in source.parents:raise ValueError('independent output required')
    deadline=time.time()+max(0,1800-previous_elapsed)
    identity,context=identity_for(source);config,dev,oldindex,oldrows,guard,valq,origins,validation=context
    with RunLock(out):
        tests=read_json(ROOT/'ff_prior_origin_fill/evidence/software_tests.json')
        if tests['status']!='PASS' or tests['source_hashes']!=sources() or tests['junit_sha256']!=file_digest(ROOT/'ff_prior_origin_fill/evidence/tests.xml'):raise ValueError('software proof stale')
        if (out/'run.json').exists() and (not resume or read_json(out/'run.json')['identity']!=identity):raise ValueError('frozen identity changed')
        physical,canonical,initial,_=audit(config);profile=acceptance(config)
        if physical['acceptance']!=read_json(SOURCE/'run.json')['acceptance']:raise ValueError('physics gates changed')
        if not (out/'run.json').exists():
            atomic_json(out/'run.json',{'identity':identity,'acceptance':physical['acceptance'],'producer_head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()})
            atomic_json(out/'frozen_validation.json',validation);atomic_json(out/'original_DEV.json',dev)
            atomic_json(out/'inherited_accounting.json',{'status':read_json(SOURCE/'status.json'),'summary':read_json(SOURCE/'summary.json'),
                'ledger_sha256':file_digest(SOURCE/'ledger.json'),'historical_older_failure_accounting':read_json(ROOT/'artifacts/v7_1/qualified_coverage_train_001/summary.json')['inherited_failure_accounting']})
            with zipfile.ZipFile(out/'executed_source.zip','w',zipfile.ZIP_DEFLATED) as z:
                for p in sources():z.write(ROOT/p,p)
        ledger=Ledger(out,{'run_identity_hash':digest(identity)})
        existing=data.rows(out,ledger);done={r['slot'] for r in existing};origins|={r['origin_hash'] for r in existing}
        for slot in SLOTS:
            for attempt in range(8):
                if slot in done:break
                sid=f'origin_fill_train_{slot:03d}_a{attempt}'
                if sid in ledger.data['completed']:continue
                if time.time()>=deadline-600:break
                if sum(p.stat().st_size for p in out.rglob('*') if p.is_file())>=P['max_output_bytes']:raise RuntimeError('STORAGE_BUDGET_EXHAUSTED')
                ledger.begin(sid,'generation',1200,72);progress('new_supplementary_origin',source_slot=slot,attempt=attempt+1)
                r=data.generate(slot,attempt,canonical,initial,profile,guard,origins,valq,out,deadline-600)
                atomic_json(out/'generation'/f'{sid}.json',r);ledger.finish({'status':r['status']},r['calls'])
                if r['admitted']:done.add(slot);origins.add(r['origin_hash'])
            report(out,ledger,'RUNNING')
        path=out/'audit_selection.json'
        if not path.exists():atomic_json(path,{'sample_ids':[r['sample_id'] for r in data.rows(out,ledger)],'before_execution':True,'replacement_after_failure':False})
        for sid in read_json(path)['sample_ids']:
            key='witness_'+sid
            if key in ledger.data['completed']:continue
            if time.time()>=deadline:break
            r=check_new_record(out,out/'records'/f'{sid}.json',ledger)
            rr,ii,cc,_=at_scene_origin(canonical,initial,r['scene'],r['condition']['initial_features']);before=rr.calls.copy()
            with np.load(out/r['arrays']) as a:z=a['Z'].copy()
            ledger.begin(key,'witness',600,9);progress('new_native_torque_replay',sample=sid)
            result=execute(out/'witness_executions'/sid,r['task'],z,rr,ii,cc,profile,deadline)
            r.update(execution_status='PASS' if result['status']=='EXECUTION_VERIFIED' else 'FAIL',training_eligible=result['status']=='EXECUTION_VERIFIED')
            if r['training_eligible']:r['validation_level']='EXECUTION_VERIFIED'
            r['execution_report']=f'witness_executions/{sid}/report.json'
            atomic_json(out/'records'/f'{sid}.json',r);ledger.finish(result,{k:rr.calls[k]-before[k] for k in before});report(out,ledger,'RUNNING')
        atomic_json(out/'training_index.json',{'records':data.merged_index(out,oldindex,ledger),'actual_training_consumed':False,'TEST_arrays_read':False,
            'parent_original_gate_still_failed':True,'qualification_run':out.name})
        ledger.index();progress('merged_independent_raw_audit')
        proof=verify(out,ledger,identity,oldindex,oldrows,guard,valq,origins-set(r['origin_hash'] for r in data.rows(out,ledger)),validation)
        atomic_json(out/'independent_verification.json',proof);result=report(out,ledger,'FINISHED')
        if not result['merged_data_gate_passed']:result=report(out,ledger,'FINISHED_WITH_BLOCKED_ITEMS')
        return result
