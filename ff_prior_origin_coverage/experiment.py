"""Freeze validation before generation, qualify fresh origins, then native audit."""
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
from ff_prior_generalization.experiment import inputs as parent_inputs,sources as parent_sources,execute
from ff_prior_origin_scene.runtime import TaskOriginSceneCache
from ff_prior_complex.tools.audit_completed import audit_physics
from ff_prior.initial_state import STATE
from ff_prior.spline import from_arrays
from .protocol import ROOT,SOURCE,NEW_DEV,PROTOCOL as P
from . import data


def sources():
    result=parent_sources()
    for package in ('ff_prior_origin_scene','ff_prior_origin_coverage'):
        result.update({p.relative_to(ROOT).as_posix():file_digest(p) for p in sorted((ROOT/package).rglob('*.py'))})
    return result


def identity_for(source):
    source=Path(source).resolve()
    if source!=SOURCE.resolve():raise ValueError('frozen 12144 training parent required')
    config,dev,index,guard,files=parent_inputs(source)
    state=read_json(NEW_DEV/'status.json')
    if state['status']!='FINISHED' or process_is_alive(state['pid']):raise ValueError('new validation must complete and exit')
    if read_json(NEW_DEV/'independent_verification.json')['status']!='PASS':raise ValueError('new validation audit required')
    for directory in (NEW_DEV,ROOT/'artifacts/v7_1/independent_new_validation_001'):
        for p in sorted(directory.rglob('*')):
            if p.is_file() and p.name not in ('run.lock','launch.guard') and p.suffix!='.tmp':
                files[p.relative_to(ROOT).as_posix()]=file_digest(p)
    for e in read_json(NEW_DEV/'frozen_tasks.json')['entries']:
        guard.append({'sample_id':'PROTECTED_NEW_DEV_'+e['sample_id'],'split':'DEV',
            'family_id':e['family_id'],'goals':e['task']['condition']['goals']})
    original_origins={digest(read_json(ROOT/e['record'])['condition']['initial_features']) for e in index}
    heldout=[e for e in guard if e['split']!='TRAIN']
    preserved=read_json(ROOT/'ff_prior_expand/evidence/preservation_before.json')
    identity={'protocol':P,'source':str(source),'source_files':files,'source_hashes':sources(),
              'original_DEV_hash':digest(dev),'guard_hash':digest(guard),'preservation':preserved}
    return identity,(config,dev,index,guard,heldout,original_origins)


def report(out,ledger,status):
    out=Path(out);rows=data.rows(out,ledger)
    forward=[r for r in rows if r['data_role']=='INDEPENDENT_ORIGIN_TRAIN']
    modes=[r for r in rows if r['data_role']=='FIXED_NEW_ORIGIN_TRAIN_WORLD_MODES']
    val=[r for r in rows if r['split']=='NEW_VAL']
    executions=[read_json(p) for p in (out/'witness_executions').glob('*/outcome.json')]
    proof=read_json(out/'independent_verification.json') if (out/'independent_verification.json').exists() else {}
    unique={r['origin_hash'] for r in forward}
    errors=[k for k,v in ledger.data['completed'].items() if v['status'] in ('ERROR','INTERRUPTED','BLOCKED')]
    n=sum(r['split']=='TRAIN' and r['execution_status']!='FAIL' for r in rows)
    passed=bool(status=='FINISHED' and len(forward)==512 and len(unique)==512 and
        sum(bool(r['scene']['objects']) for r in forward)==128 and len(modes)>=16 and
        len({r['task']['task_hash'] for r in modes})==8 and len(val)==8 and len(executions)==32 and
        all(r['status']=='EXECUTION_VERIFIED' for r in executions) and proof.get('status')=='PASS' and
        not errors and not ledger.data.get('unresolved_call_accounting'))
    result={'status':status,'new_forward_TRAIN':len(forward),'unique_new_forward_origins':len(unique),
        'new_scratch_obstacle_forward':sum(bool(r['scene']['objects']) for r in forward),
        'new_fixed_modes':len(modes),'fixed_targets_with_admission':len({r['task']['task_hash'] for r in modes}),
        'new_validation_tasks':len(val),'total_TRAIN':12144+n,'original_VAL':48,
        'new_TRAIN_MODEL_VALID_only':sum(r['split']=='TRAIN' and r['execution_status']=='NOT_RUN' for r in rows),
        'new_TRAIN_EXECUTION_VERIFIED':sum(r['split']=='TRAIN' and r['execution_status']=='PASS' for r in rows),
        'native_outcomes':len(executions),'native_verified':sum(r['status']=='EXECUTION_VERIFIED' for r in executions),
        'generation_attempts':ledger.data['attempts'],'calls':ledger.data['calls'],'errors':errors,
        'unresolved_calls':ledger.data.get('unresolved_call_accounting',[]),
        'data_gate_passed':passed,'ready_for_training_stage':passed,'independent_audit_passed':proof.get('status')=='PASS',
        'training_executed':False,'TEST_executed':False,'relative_efficiency_required':False,
        'diffusion_advantage':'NOT_EVALUATED_IN_DATA_STAGE','full_goal_achieved':False,'protocol':P}
    atomic_json(out/'summary.json',result)
    lines=['# 独立初态覆盖与固定世界目标补充','',
        f'状态：{status}；前向TRAIN {len(forward)}/512、不同新初态 {len(unique)}；新球障碍 {result["new_scratch_obstacle_forward"]}/128。',
        f'固定TRAIN目标修正 {len(modes)}/32次尝试，覆盖 {result["fixed_targets_with_admission"]}/8原冻结目标；累计TRAIN {12144+n}，原VAL 48。',
        f'新独立验证 {len(val)}/8，均在TRAIN生成前冻结；真实执行及独立重放 {result["native_verified"]}/32；审计 {proof.get("status","PENDING")}。',
        f'数据接续训练资格：{passed}；实际新训练：False；最终TEST：False。','',
        '每次主提案独立采样q17，单次前向派生世界目标后冻结；失败不重标目标、不搜索多个环境。',
        '新验证和本轮TRAIN按家族、初态邻域及世界目标隔离；原DEV4、已用新DEV12、VAL48保留。',
        '仅32条预选样本具有各自真实力矩及重放证据，其余保持MODEL_VALID，不能批量升级。',
        '仅不穿透、17→67映射、完整初态、自由漂浮、全机几何和全部其他物理门槛保留。',
        '时间和查询只作资源记录，不设相对效率胜出门槛；本数据阶段不宣称Diffusion优势。',
        '下一阶段需实际训练并对原DEV和本轮新验证执行冻结对照；不自动十万规模或最终TEST。']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8');return result


def verify(out,ledger,identity,old,guard,heldout,old_origins):
    out=Path(out);problems=[];rows=data.rows(out,ledger);steps=0;physics=[]
    try:
        assert not ledger.data['inflight']
        assert sources()==identity['source_hashes']
        assert all(file_digest(ROOT/p)==h for p,h in identity['source_files'].items())
        for group in ('old_files','user_edits'):
            assert all(file_digest(ROOT/p)==h for p,h in identity['preservation'][group].items()),group
        for branch,h in identity['preservation']['branches'].items():
            assert subprocess.check_output(['git','rev-parse',branch],cwd=ROOT,text=True).strip()==h
        with zipfile.ZipFile(out/'executed_source.zip') as z:
            assert all(hashlib.sha256(z.read(p)).hexdigest()==h for p,h in sources().items())
        frozen=read_json(out/'frozen_validation.json');val=[r for r in rows if r['split']=='NEW_VAL']
        assert frozen['entries']==data.validation_entries(out,ledger)
        assert frozen['before_TRAIN_generation'] and frozen['before_training']
        assert not {r['family_id'] for r in val}&{r['family_id'] for r in rows if r['split']=='TRAIN'}
        refs=set();origins=set();valq=[np.asarray(r['condition']['initial_features']) for r in val]
        for r in rows:
            assert not data.neighbours(r['condition']['goals'],guard if r['split']=='NEW_VAL' else heldout)
            with np.load(out/r['arrays']) as a:
                ref=from_arrays(a);assert ref.reference_hash==r['reference_hash']
                for name,order in [('view_q',0),('view_qd',1),('view_qdd',2)]:
                    assert np.allclose(a[name],ref.evaluate(a['view_time'],order),rtol=0,atol=1e-12)
            assert ref.reference_hash not in refs;refs.add(ref.reference_hash)
            with np.load(out/r['complete_initial_state']) as a:
                assert digest({'values':a['values'],'model_hash':r['task']['contract']['robot_hash'],'state_mask':int(STATE)})==r['task']['contract']['initial_state_id']
            if r['data_role']=='INDEPENDENT_ORIGIN_TRAIN':
                assert r['origin_hash'] not in old_origins|origins
                assert digest(r['condition']['initial_features'])==r['origin_hash'];origins.add(r['origin_hash'])
                assert all(np.max(abs(np.asarray(r['condition']['initial_features'])-v))>=.02 for v in valq)
            assert (r['validation_level']=='EXECUTION_VERIFIED')==(r['execution_status']=='PASS')
            if r['split']=='NEW_VAL':assert not r['training_eligible']
        modes=read_json(out/'mode_selection.json')
        for entry in modes:
            parent=next(r for r in rows if r['sample_id']==entry['parent_sample_id'])
            assert entry['task']==parent['task'] and entry['family_id']==parent['family_id']
            references=[np.asarray(entry['Z'])]
            for r in rows:
                if r['sample_id'].startswith(entry['sample_id']+'_v'):
                    assert r['task']==entry['task'] and r['family_id']==entry['family_id']
                    with np.load(out/r['arrays']) as a:z=a['Z'].copy()
                    assert min(float(np.sqrt(np.mean((z-v)**2))) for v in references)>=.005
                    references.append(z)
        selected=read_json(out/'audit_selection.json')
        # Selection uses geometry and ids only, so audit grade changes do not affect it.
        assert selected==data.selection(rows)
        ids=selected['risk']+selected['fixed_modes']+selected['hash_ordered']+selected['validation']
        assert len(ids)==len(set(ids))
        assert {r['sample_id'] for r in rows if r['execution_status']=='PASS'}==set(ids)
        for sid in ids:
            r=next(r for r in rows if r['sample_id']==sid)
            assert ledger.data['completed']['witness_'+sid]['status']=='EXECUTION_VERIFIED'
            p,n=audit_physics(out/'witness_executions'/sid,r['task'],r['reference_hash'])
            physics.append(p);steps+=n
            assert p['level']=='EXECUTION_VERIFIED' and r['execution_status']=='PASS'
        assert steps==ledger.data['calls'].get('witness.physics_steps',0)
        generated=[read_json(p) for p in (out/'generation').glob('*.json')]
        assert len(generated)==sum(v for k,v in ledger.data['attempts'].items() if k!='witness')
        for stage in ('validation_generation','forward_generation','fixed_generation'):
            proofs=[r for r in generated if r['ledger_stage']==stage]
            assert sum(r['calls']['coupled'] for r in proofs)==ledger.data['calls'].get(stage+'.coupled',0)
            assert all(r['calls']['coupled']<=(113 if stage=='fixed_generation' else 4) for r in proofs)
            assert all(not any(r['calls'][k] for k in ('physics_steps','torque_rollout','rrt','sqp')) for r in proofs)
        index=read_json(out/'training_index.json')['records']
        assert index==data.index_for(out,old,ledger)
        assert sum(e['split']=='VAL' for e in index)==48
        assert not {r['sample_id'] for r in val}&{e['sample_id'] for e in index}
        for e in index:
            assert all(file_digest(ROOT/e[k])==e[k+'_sha256'] for k in ('record','arrays'))
    except Exception as error:
        import traceback
        problems.append({'reason':repr(error),'traceback':traceback.format_exc()})
    if ledger.data.get('unresolved_call_accounting'):problems.append('UNRESOLVED_INTERRUPTED_WORK')
    return {'status':'PASS' if not problems else 'BLOCKED','problems':problems,
        'actual_new_records':len(rows),'audited_native_traces':len(physics),
        'physics_steps_including_replay':steps,'minimum_execution_clearance_m':min((p['minimum_clearance_m'] for p in physics),default=None),
        'maximum_hold_position_m':max((max(p['hold_max_position_error_m']) for p in physics),default=None),
        'source_files_user_edits_branches_unchanged':not problems,'additional_physics_calls':0,
        'training_executed':False,'TEST_executed':False,'full_goal_achieved':False}


def run(source,out,progress,resume=False,previous_elapsed=0.):
    source,out=Path(source).resolve(),Path(out).resolve()
    if source==out or source in out.parents or out in source.parents:raise ValueError('independent output required')
    started=time.time();deadline=started+max(0,P['wall_budget_s']-previous_elapsed)
    identity,(config,dev,old,guard,heldout,old_origins)=identity_for(source)
    with RunLock(out):
        tests=read_json(ROOT/'ff_prior_origin_coverage/evidence/software_tests.json')
        if tests['status']!='PASS' or tests['source_hashes']!=sources() or tests['junit_sha256']!=file_digest(ROOT/'ff_prior_origin_coverage/evidence/tests.xml'):
            raise ValueError('software proof stale')
        if (out/'run.json').exists() and (not resume or read_json(out/'run.json')['identity']!=identity):raise ValueError('frozen identity changed')
        physical,canonical,initial,_=audit(config);profile=acceptance(config)
        if physical['acceptance']!=read_json(source/'run.json')['acceptance']:raise ValueError('physical constraints changed')
        if not (out/'run.json').exists():
            atomic_json(out/'run.json',{'identity':identity,'acceptance':physical['acceptance'],
                'producer_head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()})
            atomic_json(out/'original_DEV.json',dev);atomic_json(out/'software_tests.json',tests)
            with zipfile.ZipFile(out/'executed_source.zip','w',zipfile.ZIP_DEFLATED) as z:
                for p in sources():z.write(ROOT/p,p)
        ledger=Ledger(out,{'run_identity_hash':digest(identity)})
        if not (out/'frozen_validation.json').exists():
            frozen_slots={r['slot'] for r in data.rows(out,ledger) if r['split']=='NEW_VAL'}
            vd=min(deadline-P['generation_reserve_for_execution_s'],time.time()+max(0,P['validation_generation_budget_s']-ledger.data['wall_s'].get('validation_generation',0)))
            for slot in range(8):
                for attempt in range(8):
                    if slot in frozen_slots:break
                    sid=f'origin_new_val_{slot:03d}_a{attempt}'
                    if sid in ledger.data['completed']:continue
                    if time.time()>=vd:break
                    ledger.begin(sid,'validation_generation',600,64)
                    progress('freeze_fresh_validation',slot=slot+1,attempt=attempt+1)
                    r=data.generate('NEW_VAL',slot,attempt,canonical,initial,profile,guard,old_origins,[],out,vd)
                    r['ledger_stage']='validation_generation'
                    atomic_json(out/'generation'/f'{sid}.json',r);ledger.finish({'status':r['status']},r['calls'])
                    if r['admitted']:frozen_slots.add(slot)
            atomic_json(out/'frozen_validation.json',{'entries':data.validation_entries(out,ledger),
                'before_TRAIN_generation':True,'before_training':True,'replacement_after_freeze':False})
        frozen=read_json(out/'frozen_validation.json')
        valguard=[{'sample_id':e['sample_id'],'family_id':e['family_id'],'split':'NEW_VAL','goals':e['task']['condition']['goals']} for e in frozen['entries']]
        heldout=heldout+valguard
        valq=[np.asarray(e['task']['condition']['initial_features']) for e in frozen['entries']]
        gd=deadline-P['generation_reserve_for_execution_s']
        admitted_slots={r['slot'] for r in data.rows(out,ledger) if r['data_role']=='INDEPENDENT_ORIGIN_TRAIN'}
        for slot in range(512):
            for attempt in range(4):
                if slot in admitted_slots:break
                sid=f'origin_train_{slot:03d}_a{attempt}'
                if sid in ledger.data['completed']:continue
                if time.time()>=gd or ledger.data['attempts'].get('forward_generation',0)>=768:break
                ledger.begin(sid,'forward_generation',4200,768)
                progress('independent_origin_forward',slot=slot+1,attempt=attempt+1)
                r=data.generate('TRAIN',slot,attempt,canonical,initial,profile,heldout,old_origins,valq,out,gd)
                r['ledger_stage']='forward_generation'
                atomic_json(out/'generation'/f'{sid}.json',r);ledger.finish({'status':r['status']},r['calls'])
                if r['admitted']:admitted_slots.add(slot)
            if slot%32==31:report(out,ledger,'RUNNING')
        cache=TaskOriginSceneCache(canonical,initial,profile)
        for entry in data.select_modes(out,ledger):
            for k in range(4):
                sid=entry['sample_id']+f'_v{k}'
                if sid in ledger.data['completed']:continue
                if time.time()>=gd:break
                ledger.begin(sid,'fixed_generation',4200,32);progress('fixed_TRAIN_world_modes',target=entry['slot']+1,variant=k+1)
                r=data.generate_mode(entry,k,cache,out,profile,gd);r['ledger_stage']='fixed_generation'
                atomic_json(out/'generation'/f'{sid}.json',r);ledger.finish({'status':r['status']},r['calls'])
        path=out/'audit_selection.json'
        if not path.exists():atomic_json(path,data.selection(data.rows(out,ledger)))
        selected=read_json(path)
        for sid in selected['risk']+selected['fixed_modes']+selected['hash_ordered']+selected['validation']:
            key='witness_'+sid
            if key in ledger.data['completed']:continue
            if time.time()>=deadline:break
            r=data.check_new_record(out,out/'records'/f'{sid}.json',ledger)
            rr,ii,cc,_=cache.get(r);before=rr.calls.copy()
            with np.load(out/r['arrays']) as a:z=a['Z'].copy()
            ledger.begin(key,'witness',1200,32);progress('native_torque_independent_replay',sample=sid)
            result=execute(out/'witness_executions'/sid,r['task'],z,rr,ii,cc,profile,deadline)
            r.update(execution_status='PASS' if result['status']=='EXECUTION_VERIFIED' else 'FAIL',
                training_eligible=r['split']=='TRAIN' and result['status']=='EXECUTION_VERIFIED')
            if result['status']=='EXECUTION_VERIFIED':r['validation_level']='EXECUTION_VERIFIED'
            r['execution_report']=f'witness_executions/{sid}/report.json'
            atomic_json(out/'records'/f'{sid}.json',r);ledger.finish(result,{k:rr.calls[k]-before[k] for k in before})
            report(out,ledger,'RUNNING')
        # Task, initial state, source witness and split remain exactly identical;
        # individual audit grades in records do not alter frozen validation.
        assert [{k:v for k,v in e.items()} for e in frozen['entries']]==data.validation_entries(out,ledger)
        atomic_json(out/'training_index.json',{'records':data.index_for(out,old,ledger),'actual_training_consumed':False,'TEST_arrays_read':False})
        ledger.index();progress('independent_saved_array_and_native_audit')
        proof=verify(out,ledger,identity,old,guard,heldout,old_origins);atomic_json(out/'independent_verification.json',proof)
        result=report(out,ledger,'FINISHED')
        if not result['data_gate_passed']:result=report(out,ledger,'FINISHED_WITH_BLOCKED_ITEMS')
        return result
