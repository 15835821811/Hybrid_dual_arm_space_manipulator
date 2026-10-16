"""Freeze three boundary cases now, execute only after the parent worker exits."""
from pathlib import Path
import os
import time
import zipfile
import subprocess
import numpy as np
from ff_prior.storage import read_json,atomic_json,file_digest,digest,Ledger,RunLock
from ff_prior.recovery import process_is_alive
from ff_prior.pipeline import audit,acceptance
from ff_prior_modes.data import RuntimeCache
from ff_prior_expand.experiment import sources as parent_sources
from ff_prior_expand.data import check_new_record
from ff_prior_loop.experiment import execute
from ff_prior_complex.tools.audit_completed import audit_physics

ROOT=Path(__file__).resolve().parents[1]
PROTOCOL={'version':'supplemental_obstacle_boundary_1','split':'TRAIN','maximum_executions':3,
    'selection':'lowest model clearance per obstacle_0/1/2 among samples not in original 16 preselection; tie sample id',
    'selection_uses_execution_outcomes':False,'replace_failure':False,'active_wall_budget_s':180,
    'maximum_wait_s':7200,'max_output_bytes':1024**3,'physics_changes':False,
    'wait_for_parent_process_exit':True,'parent_failure_does_not_block_independent_admitted_label_audit':True,
    'original_records_remain_immutable':True,'level_changes_via_individual_overlay_only':True,
    'pilot_requires_all_three_execution_and_replay_pass':True,'TEST':False,'training':False}


def sources():
    return dict(parent_sources(),**{p.relative_to(ROOT).as_posix():file_digest(p) for p in sorted((ROOT/'ff_prior_boundary').rglob('*.py'))})


def targets(source):
    source=Path(source);original=read_json(source/'audit_selection.json')
    excluded=set(original['random_stratified']+original['risk'])
    rows=[read_json(p) for p in (source/'records').glob('*.json')]
    selected=[]
    for context in ('obstacle_0','obstacle_1','obstacle_2'):
        eligible=[r for r in rows if r['context_id']==context and r['sample_id'] not in excluded]
        if not eligible:raise ValueError('missing unselected obstacle stratum')
        r=min(eligible,key=lambda r:(r['validation']['geometry']['minimum_clearance_m'],r['sample_id']))
        assert r['split']=='TRAIN' and r['validation_level']=='MODEL_VALID' and r['execution_status']=='NOT_RUN'
        selected.append({'sample_id':r['sample_id'],'context_id':context,'family_id':r['family_id'],
            'model_clearance_m':r['validation']['geometry']['minimum_clearance_m'],
            'task_hash':r['task']['task_hash'],'reference_hash':r['reference_hash']})
    return selected


def inputs(source):
    source=Path(source).resolve();selected=targets(source)
    paths=[source/'run.json',source/'audit_selection.json']
    for e in selected:
        p=source/'records'/f'{e["sample_id"]}.json';r=check_new_record(source,p)
        paths.extend([p,source/r['arrays'],source/r['complete_initial_state'],source/r['generation_proof'],source/'tasks'/e['sample_id']/'task.json'])
    original=ROOT/'artifacts/v7_1/smoke_002/run.json';paths.append(original)
    files={Path(os.path.relpath(p,source)).as_posix():file_digest(p) for p in paths}
    return read_json(original),None,selected,None,files


def report(out,ledger,status):
    out=Path(out);selection=read_json(out/'selection.json')['selected'];rows=[];overlay={}
    for e in selection:
        directory=out/'executions'/e['sample_id'];p=directory/'outcome.json'
        if not p.exists():continue
        result=read_json(p);row=dict(e,execution_status=result['status'])
        if (directory/'report.json').exists():
            r=read_json(directory/'report.json')
            row.update(actual_clearance_m=r['minimum_clearance_m'],hold_max_position_error_m=r['hold_max_position_error_m'],replay_passed=r['replay']['passed'])
        rows.append(row)
        overlay[e['sample_id']]={'source_reference_hash':e['reference_hash'],'execution_report':str(directory/'report.json'),
            'validation_level':'EXECUTION_VERIFIED' if result['status']=='EXECUTION_VERIFIED' else 'EXECUTION_FAILED',
            'training_eligible':result['status']=='EXECUTION_VERIFIED','applies_to_this_sample_only':True}
    passed=len(rows)==3 and all(r['execution_status']=='EXECUTION_VERIFIED' for r in rows)
    audited=(out/'independent_verification.json').exists() and read_json(out/'independent_verification.json')['status']=='PASS'
    result={'status':status,'rows':rows,'completed':len(rows),'all_three_execution_verified':passed,
        'boundary_gate_passed':bool(passed and audited and not ledger.data.get('unresolved_call_accounting')),
        'scope':'three preselected TRAIN obstacle-boundary labels only; no network or generalization claim',
        'original_records_changed':False,'additional_training_or_TEST':False,'calls':ledger.data['calls']}
    atomic_json(out/'eligibility_overlay.json',overlay);atomic_json(out/'summary.json',result)
    lines=['# 扩库障碍边界补充物理审计','',f'状态：{status}；完成 {len(rows)}/3；全部执行及重放通过：{passed}；独立审计：{audited}。','',
        '| 样本 | 预测间距m | 执行状态 | 实际间距m |','|---|---:|---|---:|']
    for r in rows:lines.append(f'| {r["sample_id"]} | {r["model_clearance_m"]:.9f} | {r["execution_status"]} | {r.get("actual_clearance_m","—")} |')
    lines+=['','原16条抽查每家族一条，随机组覆盖障碍家族后，风险组不能再选这些家族的临界轨迹。本补充在查看新执行结果前冻结三条最低间距样本，不修改原协议或历史判定。',
        '只对这些单条提供等级/隔离覆盖表；原记录不改动。若有执行失败，后续训练和pilot必须应用隔离记录。当前模型此前已读该样本时，不隐瞒该事实；补充未运行新的训练或TEST。']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8');return result


def run(source,out,progress,resume=False,previous_elapsed=0.):
    source,out=Path(source).resolve(),Path(out).resolve();old,_,selected,_,files=inputs(source)
    identity={'protocol':PROTOCOL,'source':str(source),'source_files':files,'tasks_hash':digest(selected),'source_hashes':sources()}
    with RunLock(out):
        proof=read_json(ROOT/'ff_prior_boundary/evidence/software_tests.json')
        if proof['status']!='PASS' or proof['source_hashes']!=sources() or proof['junit_sha256']!=file_digest(ROOT/'ff_prior_boundary/evidence/tests.xml'):
            raise ValueError('boundary software proof stale')
        if (out/'run.json').exists() and (not resume or read_json(out/'run.json')['identity']!=identity):raise ValueError('identity changed')
        if not (out/'run.json').exists():
            atomic_json(out/'run.json',{'identity':identity,'producer_head':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()})
            atomic_json(out/'selection.json',{'selected':selected,'frozen_before_execution':True,'parent_status_at_selection':read_json(source/'status.json'),'no_replacement':True})
            with zipfile.ZipFile(out/'executed_source.zip','w',zipfile.ZIP_DEFLATED) as z:
                for p in sources():z.write(ROOT/p,p)
        ledger=Ledger(out,identity);report(out,ledger,'WAITING_FOR_PARENT')
        wait_start=time.monotonic();progress('waiting_for_parent_exit',parent_pid=read_json(source/'status.json')['pid'])
        while True:
            parent=read_json(source/'status.json');alive=process_is_alive(parent['pid'])
            if not alive:
                if parent['status'] not in ('FINISHED','FINISHED_WITH_BLOCKED_ITEMS','FAILED'):raise RuntimeError('parent interrupted without terminal accounting')
                break
            if time.monotonic()-wait_start+previous_elapsed>=7200:raise RuntimeError('WAIT_BUDGET_EXHAUSTED')
            time.sleep(5)
        prior_active=sum(ledger.data['wall_s'].values());deadline=time.time()+max(0,180-prior_active)
        atomic_json(out/'parent_exit.json',{'parent_status':parent,'parent_process_alive':False,'wait_wall_s':time.monotonic()-wait_start})
        physical,canonical,initial,_=audit(old['config']);profile=acceptance(old['config'])
        assert physical['acceptance']==read_json(source/'run.json')['acceptance']
        cache=RuntimeCache(canonical,initial,profile)
        for e in selected:
            sid=e['sample_id'];key='execution_'+sid
            if key in ledger.data['completed']:continue
            if time.time()>=deadline or sum(p.stat().st_size for p in out.rglob('*') if p.is_file())>=1024**3:raise RuntimeError('ACTIVE_BUDGET_EXHAUSTED')
            r=check_new_record(source,source/'records'/f'{sid}.json');rr,ii,cc,_=cache.get(r)
            with np.load(source/r['arrays']) as a:z=a['Z'].copy()
            before=rr.calls.copy();ledger.begin(key,'execution',180,3);progress('boundary_torque_replay',sample=sid,model_clearance_m=e['model_clearance_m'])
            result=execute(out/'executions'/sid,r['task'],z,rr,ii,cc,profile,deadline)
            ledger.finish(result,{k:rr.calls[k]-before[k] for k in before});report(out,ledger,'RUNNING')
        steps=0;audited=[]
        for e in selected:
            r=read_json(source/'records'/f'{e["sample_id"]}.json');directory=out/'executions'/e['sample_id']
            p,n=audit_physics(directory,r['task'],r['reference_hash']);steps+=n;audited.append(p['level'])
        assert steps==ledger.data['calls']['execution.physics_steps']
        assert sources()==identity['source_hashes'] and all(file_digest(source/p)==h for p,h in files.items())
        atomic_json(out/'independent_verification.json',{'status':'PASS','audited_traces':3,'levels':audited,'physics_steps_including_replay':steps,
            'original_source_records_unchanged':True,'additional_audit_physics_calls':0,'source_code_unchanged':True})
        return report(out,ledger,'FINISHED')
