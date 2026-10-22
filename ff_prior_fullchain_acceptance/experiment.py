"""Six final static executions, exact dynamic reuse, twelve explicit dependencies."""
import subprocess
import time
import zipfile
import numpy as np
from ff_prior.storage import read_json,file_digest,digest,RunLock
from ff_prior_pilot_train.storage import atomic_json
from ff_prior.pipeline import audit,acceptance
from ff_prior.spline import Reference
from ff_prior_origin_learn_bounded.runtime import TaskOriginSceneCache
from ff_prior_static_goal_seed.physical_audit import audit_execution
from .protocol import ROOT,PACKAGE,OUT,STATIC,DYNAMIC,BASE,P
from .inputs import identity


class Budget:
    def __init__(self,out):
        self.out=out;self.started=time.monotonic();self.reserved=0;self.actual=0;self.inflight=None
    def check(self):
        if time.monotonic()-self.started>P['wall_s']:raise RuntimeError('WALL_BUDGET')
        size=sum(f.stat().st_size for f in self.out.rglob('*') if f.is_file())
        if size>P['output_bytes_cap']:raise RuntimeError('OUTPUT_BUDGET')
        return size
    def save(self):
        size=self.check()
        atomic_json(self.out/'budget.json',dict(reserved_native_replay_steps=self.reserved,
            actual_native_replay_steps=self.actual,elapsed_s=time.monotonic()-self.started,
            output_bytes=size,inflight=self.inflight,limits=P))
    def reserve(self,unit,n):
        if self.inflight is not None:raise RuntimeError('UNRESOLVED_WORK')
        if self.reserved+n>P['new_native_replay_steps']:raise RuntimeError('STEP_BUDGET')
        self.reserved+=n;self.inflight=dict(unit=unit,reserved=n);self.save()
    def finish(self,n):
        if n>self.inflight['reserved']:raise RuntimeError('ACTUAL_EXCEEDS_RESERVED_STEPS')
        self.actual+=n;self.inflight=None;self.save()


def exact_reuse(spec):
    d=ROOT/spec['source'];origin=ROOT/spec['execution_source']
    report=read_json(origin/'result.json')
    if report['status']!='EXECUTION_VERIFIED' or not all(report['gates'].values()):
        raise ValueError('HISTORICAL_EXECUTION_NOT_VERIFIED')
    if file_digest(origin/'native_trace.npz')!=spec['native_trace_sha256']:
        raise ValueError('CONTROL_TRACE_IDENTITY_CHANGED')
    if digest(read_json(d/'task.json'))!=spec['task_hash']:raise ValueError('TASK_CHANGED')
    full=report['replay_full_integration_error']
    if not np.isfinite(full) or full>P['full_integration_atol']:raise ValueError('FULL_STATE_REPLAY_NOT_SUPPORTED')
    # All archive, source, model and input hashes are already bound by identity.
    # Never run an old simulator, optimizer, audit or resume for this reuse.
    return dict(level='EXECUTION_VERIFIED',scope='EXACT_HISTORICAL_EXECUTION_REUSE',
        new_execution=False,report=report,execution_source=spec['execution_source'],
        source_record=spec['source'],original_new_execution=spec['original_new_execution'],
        shared_not_independent=not spec['original_new_execution'],
        new_physics_steps=0,full_integration_gate_explicitly_required=True)


def summarize(rows):
    scene_counts={}
    for scene in P['scenes']:
        scene_counts[scene]={}
        for method in P['methods']:
            r=[x for x in rows if x['spec']['scene']==scene and x['spec']['method']==method]
            scene_counts[scene][method]=dict(denominator=3,completed=len(r),
                verified=sum(x['result']['level']=='EXECUTION_VERIFIED' for x in r),
                new_executions=sum(x['result'].get('new_execution',False) for x in r),
                reused=sum(x['result'].get('scope')=='EXACT_HISTORICAL_EXECUTION_REUSE' for x in r),
                blocked=sum(x['result']['level']=='BLOCKED' for x in r),
                failed=sum(x['result']['level']=='EXECUTION_FAILED' for x in r),missing=3-len(r))
    coverage={scene:any(v['verified']>0 for v in methods.values()) for scene,methods in scene_counts.items()}
    return dict(status='FINISHED_WITH_BLOCKED_ITEMS',scene_counts=scene_counts,coverage=coverage,
        engineering_complete=all(coverage.values()),scientific_advantage='NOT_ESTABLISHED_BY_SUITE_COMPLETION',
        logical_denominator=30,results=rows,TRAIN_added=0,new_training_updates=0,new_proposals=0,TEST=False,
        continuous_five_obstacle_combo_executed=False,
        comparison_scope='system configurations; static and temporal networks have different representations')


def run(out,notify):
    ident=identity();budget=Budget(out)
    def progress(stage,**kwargs):budget.save();notify(stage,**kwargs)
    with RunLock(out):
        atomic_json(out/'run.json',dict(identity=ident,producer_head=subprocess.check_output(
            ['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()))
        with zipfile.ZipFile(out/'executed_source.zip','w',zipfile.ZIP_DEFLATED) as z:
            for p in ident['source_hashes']:z.write(ROOT/p,p)
        config=read_json(BASE/'run.json')['config'];physical,canonical,initial,_=audit(config)
        profile=acceptance(config);old=read_json(STATIC/'physical_contract.json')
        assert physical['acceptance']==old['acceptance'] and digest(physical['identity'])==digest(old['identity'])
        assert physical['initial_state_id']==old['initial_state_id']
        atomic_json(out/'physical_contract.json',physical)
        entries={e['sample_id']:e for e in ident['selection']['entries']}
        cache=TaskOriginSceneCache(canonical,initial,profile,max_entries=1,max_scene_models=1)
        rows=[];seen=set()
        # Dependencies and historic evidence are explicit before spending native budget.
        order=sorted(ident['selection']['units'],key=lambda u:u['action']=='NEW_STATIC_EXECUTION')
        for spec in order:
            directory=out/spec['id'];directory.mkdir();atomic_json(directory/'spec.json',spec)
            if spec['action']=='BLOCKED':
                result=dict(level='BLOCKED',reason=spec['reason'],new_execution=False,new_physics_steps=0,
                    task_status=spec.get('task_status','ORIGINAL_TASK_FROZEN_NO_QUALIFIED_PATH'))
            elif spec['action']=='EXACT_HISTORICAL_REUSE':
                progress('exact_identity_reuse',unit=spec['id']);result=exact_reuse(spec)
            else:
                entry=entries[spec['scene']];robot,init,collision,proof=cache.get(entry)
                v=read_json(ROOT/spec['reference']);assert v['eligible']
                ref=Reference(v['Z'],entry['task']['condition']['initial_features'],entry['task']['condition']['duration'])
                assert ref.reference_hash==v['reference_hash']
                steps=int(round((ref.duration+profile.hold_s)/robot.model.opt.timestep))
                budget.reserve(spec['id'],steps*2);progress('native67_and_full_state_replay',unit=spec['id'])
                atomic_json(directory/'initial_audit.json',proof)
                before=robot.calls.copy()
                report=audit_execution(entry['task'],profile,robot,init,ref,directory/'execution',
                    min(time.time()+P['per_unit_wall_s'],time.time()+P['wall_s']-(time.monotonic()-budget.started)))
                delta={k:robot.calls[k]-before[k] for k in before}
                atomic_json(directory/'calls.json',delta)
                result=dict(level=report['level'],scope='NEW_FINAL_STATIC_EXECUTION_OF_FROZEN_PUBLIC_REFERENCE',
                    new_execution=True,new_planning=False,new_physics_steps=delta['physics_steps'],report=report,
                    reference_source=spec['reference'],historical_planner_source=spec['source'])
                atomic_json(directory/'result.json',result);budget.finish(delta['physics_steps'])
            atomic_json(directory/'result.json',result);rows.append(dict(spec=spec,result=result));seen.add(spec['id'])
            atomic_json(out/'progress_results.json',rows)
        progress('terminal_identity_and_accounting')
        assert identity()==ident and len(seen)==30 and budget.inflight is None
        summary=summarize(rows);summary.update(elapsed_s=time.monotonic()-budget.started,
            actual_native_replay_steps=budget.actual,source_identity_unchanged=True,fresh_review='PENDING')
        atomic_json(out/'summary.json',summary)
        atomic_json(out/'runtime_cache.json',cache.stats());budget.save()
        (out/'REPORT.md').write_text('# 固定开发套件已运行\n\n'
            '原五障碍及五障碍连续组合仍BLOCKED；不能声称完整工程通过。\n'
            '新执行和精确历史复用分别记录，独立审查及中文分析报告待收尾。\n',encoding='utf-8')
        return summary
