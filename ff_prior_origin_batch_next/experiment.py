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
from ff_prior_generalization.experiment import execute
from ff_prior_origin_learn_bounded.runtime import TaskOriginSceneCache
from ff_prior_complex.tools.audit_completed import audit_physics
from ff_prior.initial_state import STATE
from ff_prior.spline import from_arrays
from .protocol import ROOT,SOURCE,PROTOCOL as P
from .identity import sources,identity_for
from . import data


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
    scenes=sum(bool(r['scene']['objects']) for r in forward)
    targets=len({r['task']['task_hash'] for r in modes})
    passed=bool(status=='FINISHED' and len(forward)>=P['forward_admission_minimum'] and
        len(unique)==len(forward) and scenes>=P['scratch_scene_admission_minimum'] and
        len(modes)>=P['fixed_mode_admission_minimum'] and targets==P['fixed_TRAIN_targets'] and
        len(val)==8 and len(executions)==P['maximum_native_executions'] and
        all(r['status']=='EXECUTION_VERIFIED' for r in executions) and proof.get('status')=='PASS' and
        not errors and not ledger.data.get('unresolved_call_accounting'))
    result={'status':status,'new_forward_TRAIN':len(forward),'unique_new_forward_origins':len(unique),
        'new_scratch_obstacle_forward':scenes,'new_fixed_modes':len(modes),
        'fixed_targets_with_admission':targets,'new_validation_tasks':len(val),'total_TRAIN':P['old_TRAIN']+n,
        'original_VAL':48,'new_TRAIN_MODEL_VALID_only':sum(r['split']=='TRAIN' and r['execution_status']=='NOT_RUN' for r in rows),
        'new_TRAIN_EXECUTION_VERIFIED':sum(r['split']=='TRAIN' and r['execution_status']=='PASS' for r in rows),
        'native_outcomes':len(executions),'native_verified':sum(r['status']=='EXECUTION_VERIFIED' for r in executions),
        'generation_attempts':ledger.data['attempts'],'calls':ledger.data['calls'],'errors':errors,
        'unresolved_calls':ledger.data.get('unresolved_call_accounting',[]),'data_gate_passed':passed,
        'ready_for_training_stage':passed,'independent_audit_passed':proof.get('status')=='PASS',
        'parent_C1':True,'parent_C2':False,'training_executed':False,'TEST_executed':False,
        'relative_efficiency_required':False,'diffusion_advantage':'NOT_EVALUATED_IN_DATA_STAGE',
        'full_goal_achieved':False,'protocol':P}
    atomic_json(out/'summary.json',result)
    lines=['# C1闭环后的有界合格扩库','',
        f'状态：{status}；前向TRAIN {len(forward)}/1024槽位（准入下限1000）、独立新初态 {len(unique)}；新球障碍 {scenes}/256槽位（下限240）。',
        f'固定TRAIN目标修正 {len(modes)}/128次尝试，覆盖 {targets}/32个冻结目标；累计TRAIN {P["old_TRAIN"]+n}，原VAL48。',
        f'新独立验证 {len(val)}/8，在TRAIN生成前冻结；真实执行及独立重放 {result["native_verified"]}/72；原始审计 {proof.get("status","PENDING")}。',
        f'数据接续训练资格：{passed}；本批实际训练：False；最终TEST：False。','',
        '继承上一轮C1完整机制通过、C2强基线联合质量优势不通过。扩库不等待查询或耗时胜出，C2负结果不改写。',
        '每次提案独立采样q17，单次耦合前向派生世界目标后冻结；独立准入拒绝则保留失败，不重标目标。',
        '此前两批已使用NEW_VAL8及更早12任务转为DEV保护；新验证在本轮TRAIN前冻结，按家族、初态邻域及目标隔离。',
        '64条预选TRAIN和全部8条新验证各自真实67力矩执行及独立重放；未执行TRAIN只保留MODEL_VALID。',
        '初态穿透直接拒绝，全部完整状态转移、仅不穿透及其他物理门保留。',
        '新增数量下限在运行前冻结，合法拒绝不补为假通过；原失败门、39 ERROR和历史成本保留。',
        '后续需另冻结抽样策略和全体新增行实际训练、独立验证及强基线同预算对照，不能把本数据阶段写成神经收益。',
        '无自动十万规模、strict_multimodal或最终TEST。']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8');return result


def capacity(out,deadline):
    if time.time()>=deadline:raise RuntimeError('TOTAL_WALL_BUDGET_EXHAUSTED')
    if sum(p.stat().st_size for p in Path(out).rglob('*') if p.is_file())>=P['max_output_bytes']:
        raise RuntimeError('STORAGE_BUDGET_EXHAUSTED')


def verify(out,ledger,identity,old,guard,heldout,old_origins,heldout_initials):
    out=Path(out);problems=[];rows=data.rows(out,ledger);steps=0;physics=[]
    try:
        assert not ledger.data['inflight']
        assert sources()==identity['source_hashes']
        memory=read_json(out/'runtime_cache_audit.json')
        assert memory['peak_entries']<=P['runtime_cache_entries'] and memory['peak_scene_models']<=P['runtime_scene_models']
        assert memory['peak_private_bytes']<=P['max_process_private_bytes'] and not memory['physics_contract_changed']
        memory_samples=read_json(out/'generation_memory_audit.json')['samples']
        assert memory_samples and all(m['private_bytes']<=P['max_process_private_bytes'] and m['available_physical_bytes']>=P['minimum_available_memory_bytes'] for m in memory_samples)
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
        assert not {r['family_id'] for r in rows}&{e['family_id'] for e in old}
        for r in val:
            others=[{'sample_id':v['sample_id'],'goals':v['condition']['goals']} for v in val if v['sample_id']!=r['sample_id']]
            assert not data.neighbours(r['condition']['goals'],others)
            assert r['origin_hash'] not in old_origins
        refs=set();origins=set();valq=list(heldout_initials)+[np.asarray(r['condition']['initial_features']) for r in val]
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
        assert ledger.data['attempts'].get('forward_generation',0)<=P['forward_max_attempts']
        assert ledger.data['attempts'].get('validation_generation',0)<=64
        assert ledger.data['attempts'].get('fixed_generation',0)<=P['fixed_TRAIN_targets']*P['fixed_variants']
        assert ledger.data['attempts'].get('witness',0)<=P['maximum_native_executions']
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
    identity,(config,dev,old,guard,heldout,old_origins,heldout_initials)=identity_for(source)
    with RunLock(out):
        tests=read_json(ROOT/'ff_prior_origin_batch_next/evidence/software_tests.json')
        if tests['status']!='PASS' or tests['source_hashes']!=sources() or tests['junit_sha256']!=file_digest(ROOT/'ff_prior_origin_batch_next/evidence/tests.xml'):
            raise ValueError('software proof stale')
        if (out/'run.json').exists() and (not resume or read_json(out/'run.json')['identity']!=identity):raise ValueError('frozen identity changed')
        physical,canonical,initial,_=audit(config);profile=acceptance(config)
        if physical['acceptance']!=read_json(source/'run.json')['acceptance']:raise ValueError('physical constraints changed')
        if not (out/'run.json').exists():
            atomic_json(out/'run.json',{'identity':identity,'acceptance':physical['acceptance'],
                'producer_head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()})
            atomic_json(out/'original_DEV.json',dev);atomic_json(out/'software_tests.json',tests)
            atomic_json(out/'protected_goals.json',guard)
            atomic_json(out/'inherited_accounting.json',{'parent_status':read_json(source/'status.json'),
                'parent_summary':read_json(source/'summary.json'),'parent_ledger_sha256':file_digest(source/'ledger.json'),
                'prior_history':read_json(source/'inherited_accounting.json')})
            atomic_json(out/'used_validation_role.json',{'source':str(source/'frozen_validation.json'),
                'source_sha256':file_digest(source/'frozen_validation.json'),'role':'DEV_guard_only','source_unchanged':True})
            with zipfile.ZipFile(out/'executed_source.zip','w',zipfile.ZIP_DEFLATED) as z:
                for p in sources():z.write(ROOT/p,p)
        ledger=Ledger(out,{'run_identity_hash':digest(identity)})
        from ff_prior_origin_learn_bounded.runtime import memory_usage
        generation_memory=[]
        def memory_guard():
            m=memory_usage();generation_memory.append(m)
            if m['private_bytes']>P['max_process_private_bytes'] or m['available_physical_bytes']<P['minimum_available_memory_bytes']:
                raise RuntimeError('GENERATION_MEMORY_BUDGET_EXHAUSTED')
        memory_guard()
        if not (out/'frozen_validation.json').exists():
            frozen_slots={r['slot'] for r in data.rows(out,ledger) if r['split']=='NEW_VAL'}
            seen=set(old_origins)|{r['origin_hash'] for r in data.rows(out,ledger)}
            validation_guard=list(guard)
            validation_guard += [{'sample_id':r['sample_id'],'goals':r['condition']['goals']}
                                 for r in data.rows(out,ledger) if r['split']=='NEW_VAL']
            vd=min(deadline-P['generation_reserve_for_execution_s'],time.time()+max(0,P['validation_generation_budget_s']-ledger.data['wall_s'].get('validation_generation',0)))
            for slot in range(8):
                for attempt in range(8):
                    if slot in frozen_slots:break
                    sid=f'batch_v2_new_val_{slot:04d}_a{attempt}'
                    if sid in ledger.data['completed']:continue
                    if time.time()>=vd:break
                    ledger.begin(sid,'validation_generation',600,64)
                    progress('freeze_fresh_validation',slot=slot+1,attempt=attempt+1)
                    r=data.generate('NEW_VAL',slot,attempt,canonical,initial,profile,validation_guard,seen,[],out,vd)
                    r['ledger_stage']='validation_generation'
                    atomic_json(out/'generation'/f'{sid}.json',r);ledger.finish({'status':r['status']},r['calls'])
                    if r['admitted']:
                        frozen_slots.add(slot);seen.add(r['origin_hash'])
                        task=read_json(out/'tasks'/sid/'task.json')
                        validation_guard.append({'sample_id':sid,'goals':task['condition']['goals']})
            atomic_json(out/'frozen_validation.json',{'entries':data.validation_entries(out,ledger),
                'before_TRAIN_generation':True,'before_training':True,'replacement_after_freeze':False})
        frozen=read_json(out/'frozen_validation.json')
        valguard=[{'sample_id':e['sample_id'],'family_id':e['family_id'],'split':'NEW_VAL','goals':e['task']['condition']['goals']} for e in frozen['entries']]
        heldout=heldout+valguard
        valq=list(heldout_initials)+[np.asarray(e['task']['condition']['initial_features']) for e in frozen['entries']]
        gd=deadline-P['generation_reserve_for_execution_s']
        admitted_slots={r['slot'] for r in data.rows(out,ledger) if r['data_role']=='INDEPENDENT_ORIGIN_TRAIN'}
        seen=set(old_origins)|{r['origin_hash'] for r in data.rows(out,ledger)}
        for slot in range(P['forward_TRAIN_slots']):
            for attempt in range(P['attempts_per_TRAIN_slot']):
                if slot in admitted_slots:break
                sid=f'batch_v2_train_{slot:04d}_a{attempt}'
                if sid in ledger.data['completed']:continue
                if time.time()>=gd or ledger.data['attempts'].get('forward_generation',0)>=P['forward_max_attempts']:break
                ledger.begin(sid,'forward_generation',6600,P['forward_max_attempts'])
                progress('independent_origin_forward',slot=slot+1,attempt=attempt+1)
                r=data.generate('TRAIN',slot,attempt,canonical,initial,profile,heldout,seen,valq,out,gd)
                r['ledger_stage']='forward_generation'
                atomic_json(out/'generation'/f'{sid}.json',r);ledger.finish({'status':r['status']},r['calls'])
                if r['admitted']:admitted_slots.add(slot);seen.add(r['origin_hash'])
            if slot%32==31:
                capacity(out,deadline);memory_guard();report(out,ledger,'RUNNING')
        cache=TaskOriginSceneCache(canonical,initial,profile,max_entries=P['runtime_cache_entries'],
            max_scene_models=P['runtime_scene_models'],max_private_bytes=P['max_process_private_bytes'],
            min_available_bytes=P['minimum_available_memory_bytes'])
        for entry in data.select_modes(out,ledger):
            for k in range(P['fixed_variants']):
                sid=entry['sample_id']+f'_v{k}'
                if sid in ledger.data['completed']:continue
                if time.time()>=gd:break
                ledger.begin(sid,'fixed_generation',6600,P['fixed_TRAIN_targets']*P['fixed_variants']);progress('fixed_TRAIN_world_modes',target=entry['slot']+1,variant=k+1)
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
            capacity(out,deadline)
            ledger.begin(key,'witness',2400,P['maximum_native_executions']);progress('native_torque_independent_replay',sample=sid)
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
        ledger.index();memory_guard()
        atomic_json(out/'runtime_cache_audit.json',cache.stats())
        atomic_json(out/'generation_memory_audit.json',{'samples':generation_memory,'scope':'before generation and every32 slots, final audit; not continuous peak'})
        progress('independent_saved_array_and_native_audit')
        proof=verify(out,ledger,identity,old,guard,heldout,old_origins,heldout_initials);atomic_json(out/'independent_verification.json',proof)
        capacity(out,deadline)
        result=report(out,ledger,'FINISHED')
        if not result['data_gate_passed']:result=report(out,ledger,'FINISHED_WITH_BLOCKED_ITEMS')
        return result
