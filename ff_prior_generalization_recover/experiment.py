"""Import completed evidence, reuse the untouched reservation, finish diagnostics."""
from pathlib import Path
import inspect
import shutil
import subprocess
import time
import zipfile
import numpy as np
from ff_prior.storage import read_json, file_digest, digest, RunLock
from ff_prior_pilot_train.storage import atomic_json, Ledger
from ff_prior.recovery import process_is_alive
from ff_prior.pipeline import audit, acceptance
from ff_prior_modes.data import RuntimeCache
from ff_prior_goal.projection import smooth
from ff_prior_feasible.validation import validate
from ff_prior_coverage_train.job import Monitor
from ff_prior_generalization.protocol import ROOT, BASE, METHODS, PROTOCOL
from ff_prior_generalization.audit_results import verify
from ff_prior_generalization.reporting import report as original_report

SOURCE=ROOT/'artifacts/v7_1/independent_new_validation_001'
SKIP={'status.json','launcher.json','worker.stdout.log','worker.stderr.log','failure.json',
      'summary.json','REPORT.md','ANALYSIS.md','run.lock','launch.guard'}


def sources():
    return {p.relative_to(ROOT).as_posix():file_digest(p)
            for p in sorted((ROOT/'ff_prior_generalization_recover').rglob('*.py'))}


def known_zero_failure(run,state,failure,ledger,code):
    """Fail closed unless the exact exception preceded the first physical call."""
    expected="TypeError(\"Monitor.update() got multiple values for argument 'stage'\")"
    if state['status']!='FAILED' or failure['reason']!=expected:raise ValueError('not the known callback failure')
    active=ledger.get('inflight')
    if active is None or active['id']!='diagnostic_newval_00_structured_raw' or active['stage']!='diagnostic':
        raise ValueError('unexpected outstanding reservation')
    if ledger['attempts'].get('diagnostic')!=1 or any(k.startswith('diagnostic.') for k in ledger['calls']):
        raise ValueError('diagnostic work may already have run')
    if ledger.get('unresolved_call_accounting'):raise ValueError('unknown work cannot be refunded')
    for stage,n in (('candidate',720),('execution',144),('witness',12)):
        values=[v for v in ledger['completed'].values() if v['stage']==stage]
        if len(values)!=n:raise ValueError('completed trial population incomplete')
    needle="progress('independent_pre_guidance_validation',task=sid,method=method,stage=stage)"
    if needle not in code or needle not in failure['traceback']:raise ValueError('call site not proven')
    # The frozen source places this callback before the try block containing
    # validate. Python argument binding fails before Monitor.update enters.
    tail=code.split(needle,1)[1]
    if not tail.lstrip().startswith('try:') or 'check=validate(z,task,rr,ii,cc,profile,diagnostic_deadline)' not in tail:
        raise ValueError('physical validation boundary changed')
    if inspect.signature(Monitor.update).parameters['stage'].kind!=inspect.Parameter.POSITIONAL_OR_KEYWORD:
        raise ValueError('monitor signature changed')
    return {'status':'PROVEN_ZERO_DIAGNOSTIC_PHYSICS','original_reservation':active,
            'inherited_elapsed_s':state['elapsed_s'],'failed_attempt_retained':True,
            'failed_diagnostic_wall_s_upper_bound':max(0,state['heartbeat']-active['started']),
            'failed_diagnostic_coupled_calls':0,'failed_diagnostic_network_calls':0,
            'policy':'reuse same pre-physics reservation once; no candidate/native replay or unknown-call refund'}


def inspect_source(source):
    source=Path(source).resolve();state=read_json(source/'status.json')
    if process_is_alive(state['pid']):raise ValueError('original worker is still alive')
    run=read_json(source/'run.json');failure=read_json(source/'failure.json');ledger=read_json(source/'ledger.json')
    if any((source/'proposal_diagnostics').rglob('*.json')):raise ValueError('unexpected diagnostic artifacts')
    if run['identity']['protocol']!=PROTOCOL:raise ValueError('original protocol differs')
    for p,h in run['identity']['source_hashes'].items():
        if file_digest(ROOT/p)!=h:raise ValueError('original frozen source changed')
    code=(ROOT/'ff_prior_generalization/experiment.py').read_text(encoding='utf-8-sig')
    proof=known_zero_failure(run,state,failure,ledger,code)
    manifest={p.relative_to(source).as_posix():file_digest(p) for p in sorted(source.rglob('*'))
              if p.is_file() and p.name not in ('run.lock','launch.guard')}
    return run,manifest,proof


def diagnostic_progress(progress,sid,method,proposal_stage):
    progress('independent_pre_guidance_validation',task=sid,method=method,proposal_stage=proposal_stage)


class DiagnosticLedger(Ledger):
    def __init__(self,out,identity,active):
        out=Path(out);value=read_json(out/'ledger.json')
        marker=out/'diagnostics_started.json'
        if value['inflight']==active and not marker.exists():
            self.directory,self.path,self.identity,self.data=out,out/'ledger.json',identity,value
            if value['identity']!=identity:raise ValueError('imported ledger identity differs')
            self.wall,self.cpu=time.perf_counter(),time.process_time()
            self.reused_reservation=active['id']
        else:
            super().__init__(out,identity);self.reused_reservation=None
            if self.data.get('unresolved_call_accounting'):
                raise RuntimeError('uncertain recovery work preserved; reserved physics is not repeated')


def run(source,out,progress,resume=False,previous_elapsed=0.):
    source,out=Path(source).resolve(),Path(out).resolve();started=time.time()
    if source==out or source in out.parents or out in source.parents:raise ValueError('independent recovery output required')
    old_run,manifest,zero=inspect_source(source)
    identity={'source':str(source),'source_run_identity':digest(old_run['identity']),
              'source_manifest':manifest,'source_hashes':sources(),'repair':'progress field stage -> proposal_stage',
              'original_protocol':PROTOCOL,'no_sampling_or_torque_reexecution':True}
    deadline=started+max(0,7200-previous_elapsed)
    with RunLock(out):
        tests=read_json(ROOT/'ff_prior_generalization_recover/evidence/software_tests.json')
        if tests['status']!='PASS' or tests['source_hashes']!=sources() or tests['junit_sha256']!=file_digest(ROOT/'ff_prior_generalization_recover/evidence/tests.xml'):
            raise ValueError('recovery software proof stale')
        if (out/'recovery_run.json').exists():
            if not resume or read_json(out/'recovery_run.json')['identity']!=identity:raise ValueError('recovery identity changed')
        else:
            # No old file is changed. Keep original data/trace bytes and their
            # recorded times; imported events retain their original timestamps.
            if sum((source/rel).stat().st_size for rel in manifest if rel not in SKIP)>=PROTOCOL['max_output_bytes']:
                raise RuntimeError('RECOVERY_STORAGE_BUDGET_EXHAUSTED')
            for rel,h in manifest.items():
                if rel in SKIP:continue
                target=out/rel;target.parent.mkdir(parents=True,exist_ok=True)
                if rel=='events.jsonl':
                    with target.open('ab') as f:f.write((source/rel).read_bytes())
                elif target.exists():
                    if file_digest(target)!=h:raise ValueError('partial import differs from source')
                else:shutil.copy2(source/rel,target)
            atomic_json(out/'recovery_run.json',{'identity':identity,'zero_call_proof':zero,
                        'producer_head':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()})
            with zipfile.ZipFile(out/'recovery_executed_source.zip','w',zipfile.ZIP_DEFLATED) as z:
                for p in sources():z.write(ROOT/p,p)
        ledger=DiagnosticLedger(out,{'identity_hash':digest(old_run['identity'])},zero['original_reservation'])
        config=read_json(BASE/'run.json')['config'];physical,canonical,initial,_=audit(config);profile=acceptance(config)
        if physical['acceptance']!=old_run['acceptance']:raise ValueError('physical acceptance changed')
        entries=read_json(out/'frozen_tasks.json')['entries'];cache=RuntimeCache(canonical,initial,profile)
        remaining=max(0,600-zero['failed_diagnostic_wall_s_upper_bound']-ledger.data['wall_s'].get('diagnostic',0.))
        diagnostic_deadline=min(deadline,time.time()+remaining)
        for e in entries:
            if e['status']!='FROZEN':raise ValueError('incomplete frozen population')
            sid=e['sample_id'];task=e['task'];rr,ii,cc,_=cache.get(e)
            for method in METHODS:
                path=out/'candidates'/'0'/sid/method/'00.json';candidate=read_json(path)
                for stage in ('raw','smoothed_clipped'):
                    key=f'diagnostic_{sid}_{method}_{stage}';p=out/'proposal_diagnostics'/sid/method/f'{stage}.json'
                    if key in ledger.data['completed']:continue
                    if time.time()>=diagnostic_deadline:raise RuntimeError('ORIGINAL_DIAGNOSTIC_BUDGET_EXHAUSTED')
                    if sum(q.stat().st_size for q in out.rglob('*') if q.is_file())>=PROTOCOL['max_output_bytes']:
                        raise RuntimeError('RECOVERY_STORAGE_BUDGET_EXHAUSTED')
                    if key==ledger.reused_reservation:
                        ledger.wall,ledger.cpu=time.perf_counter(),time.process_time()
                        atomic_json(out/'diagnostics_started.json',{'original_reservation_reused':key,
                                    'known_failed_prior_physical_calls':0,'start':time.time()})
                    else:ledger.begin(key,'diagnostic',600,96)
                    before=rr.calls.copy();diagnostic_progress(progress,sid,method,stage)
                    try:
                        z=np.asarray(candidate['sampling']['guidance'][-1]['raw_Z'])
                        if stage=='smoothed_clipped':z=np.clip(smooth(z,task['condition']['initial_features']),rr.lower,rr.upper)
                        check=validate(z,task,rr,ii,cc,profile,diagnostic_deadline)
                        r=dict(check,status='DONE',stage=stage,task_hash=task['task_hash'],source_candidate_sha256=file_digest(path),
                               proposal_Z_hash=digest(z),affects_selection=False)
                    except Exception as error:r={'status':'ERROR','reason':repr(error),'eligible':False,'stage':stage}
                    r['calls']={k:rr.calls[k]-before[k] for k in before}
                    atomic_json(p,r);ledger.finish({'status':r['status']},r['calls'])
        progress('independent_new_validation_audit')
        proof=verify(out,old_run['identity'],read_json(out/'protected_goal_metadata.json'))
        # Verify all imported experiment bytes and all recovery code separately.
        immutable={p:h for p,h in manifest.items() if p not in SKIP|{'ledger.json','events.jsonl'}}
        assert all(file_digest(out/p)==h for p,h in immutable.items())
        assert all(file_digest(source/p)==h for p,h in manifest.items())
        assert identity['source_hashes']==sources()
        with zipfile.ZipFile(out/'recovery_executed_source.zip') as z:
            import hashlib
            assert all(hashlib.sha256(z.read(p)).hexdigest()==h for p,h in sources().items())
        proof.update(original_failed_run_unchanged=True,inherited_candidates=720,inherited_selected_executions=144,
                     inherited_teacher_executions=12,new_network_calls=0,new_native_execution_steps=0,
                     recovery_source_verified=True,known_pre_diagnostic_failure_physical_calls=0)
        atomic_json(out/'independent_verification.json',proof)
        result=original_report(out,'FINISHED' if proof['status']=='PASS' else 'FINISHED_WITH_BLOCKED_ITEMS')
        result['recovery']={'source':str(source),'original_status':'FAILED','original_elapsed_s':zero['inherited_elapsed_s'],
                            'sampling_rerun':False,'torque_execution_rerun':False,'diagnostic_actual_attempts':ledger.data['attempts']['diagnostic'],
                            'known_zero_failure_proof':zero,'source_manifest_sha256':digest(manifest)}
        atomic_json(out/'summary.json',result)
        footer='\n## 独立恢复说明\n\n原作业在首次诊断前因进度字段stage冲突失败，原全部文件保留。此作业导入720候选、144选定及12教师执行/重放的原始字节，只首次执行96项诊断及最终审计。没有重新采样、重新执行力矩或改变物理协议。已证明首次失败未进入诊断物理调用，沿用同一预留项，实际诊断尝试仍为96；原耗时计入7200秒总预算。额外运行时初始化产生前向/质量矩阵准备开销，不属于新网络或native执行，不能称全部计算零成本。\n'
        for name in ('REPORT.md','ANALYSIS.md'):
            with (out/name).open('a',encoding='utf-8') as f:f.write(footer)
        return result
