import time
import subprocess
import zipfile
import numpy as np
import torch
from ff_prior.storage import RunLock,read_json,digest,file_digest
from ff_prior_pilot_train.storage import atomic_json,atomic_npz
from ff_prior_dynamic_reference.execution import evaluate
from .protocol import ROOT,PACKAGE,P
from .inputs import identity
from .data import load_data,source_environment,development_environment
from .features import public_features,encode_conditions,decode_reference
from .models import sample
from .representation import prepare
from .learning import train
from .execution import guided_velocity

class Budget:
    def __init__(self,out):
        self.out=out;self.start=time.monotonic();self.native=0;self.replay=0;self.inflight=None
    def check(self):
        if time.monotonic()-self.start>P['wall_s']:raise TimeoutError('WALL_BUDGET')
    def step(self,kind):
        self.check()
        if self.native+self.replay>=P['max_native_replay_steps']:raise TimeoutError('STEP_BUDGET')
        setattr(self,kind,getattr(self,kind)+1)
    def reserve_native(self,n):
        self.check()
        if self.native+self.replay+n>P['max_native_replay_steps']:raise TimeoutError('STEP_BUDGET')
        self.native+=n
    def save(self):
        self.check();size=sum(p.stat().st_size for p in self.out.rglob('*') if p.is_file())
        if size>P['max_bytes']:raise TimeoutError('OUTPUT_BUDGET')
        atomic_json(self.out/'budget.json',dict(native_steps_reserved=self.native,replay_steps_reserved=self.replay,
            output_bytes=size,elapsed_s=time.monotonic()-self.start,inflight=self.inflight,limits=P))

def proposal(model,method,env,scene,codec,seed,directory):
    features=public_features(env,scene);global_condition,local_condition=encode_conditions(features,codec)
    with torch.no_grad():x,records=sample(model,torch.tensor(global_condition[None],dtype=torch.float32),
        torch.tensor(local_condition[None],dtype=torch.float32),[seed],method)
    raw=decode_reference(x[0].numpy(),features,codec)
    atomic_npz(directory/'proposal.npz',raw_coefficients=raw,normalized_output=x[0].numpy(),
        global_condition=global_condition,local_condition=local_condition,initial=env.initial.values)
    atomic_json(directory/'sampling.json',dict(seed=seed,method=method,steps=records,world_guidance_in_sampler=False,
        raw_grade='NOT_PHYSICALLY_VALIDATED',condition_schema=features['schema']))
    r=env.robot
    spline,report=prepare(raw,features['q0'],features['v0'],r.lower,r.upper,r.spec.planner_velocity_limits,r.spec.planner_acceleration_limits)
    atomic_npz(directory/'prepared.npz',knots=spline.t,coefficients=spline.c)
    atomic_json(directory/'prepare.json',report)
    return spline,report

def run(out,notify):
    torch.set_num_threads(P['threads']);ident=identity();budget=Budget(out)
    def progress(stage,**kw):budget.save();notify(stage,**kw)
    with RunLock(out):
        atomic_json(out/'run.json',dict(identity=ident,producer_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()))
        with zipfile.ZipFile(out/'executed_source.zip','w',zipfile.ZIP_DEFLATED) as z:
            for path in ident['source_hashes']:z.write(ROOT/path,path)
        codec,arrays,rows,features=load_data(progress);atomic_json(out/'codec.json',codec)
        for split,values in arrays.items():
            atomic_npz(out/(split+'_encoded.npz'),**{k:v for k,v in values.items() if isinstance(v,np.ndarray)})
        atomic_json(out/'data_consumption_contract.json',dict(train=arrays['train']['units'],validation=arrays['validation']['units'],
            TRAIN=8,VAL=4,DEV_in_training=False,master_library_modified=False,source_rows=rows))
        learning=out/'learning';learning.mkdir();budget.inflight={'stage':'training'}
        trained,models=train(learning,arrays['train'],progress,budget);atomic_json(out/'training.json',trained)
        if trained['status']!='TRAINED':
            budget.inflight=None;budget.save()
            result=dict(status='FINISHED_WITH_BLOCKED_ITEMS',training=trained,VAL='BLOCKED',DEV='BLOCKED',TEST=False)
            atomic_json(out/'summary.json',result)
            (out/'REPORT.md').write_text('# TRAIN诊断未通过\n\n正式训练、VAL与DEV均BLOCKED；保留全部诊断与原模型。\n',encoding='utf-8')
            return result
        results=[]
        # VAL diagnoses full sampling with joint torque tracking but no world/obstacle correction.
        for i,row in enumerate([r for r in rows if r['split']=='validation']):
            for m,method in enumerate(P['methods']):
                unit=f'v{i}{m}';directory=out/unit;directory.mkdir();budget.inflight={'unit':unit}
                progress('VAL_initialize',unit=unit);env,scene=source_environment(row)
                atomic_npz(directory/'initial.npz',values=env.initial.values)
                atomic_json(directory/'task.json',read_json(ROOT/row['identity']))
                spline,prepared=proposal(models[method],method,env,scene,codec,20266000+i,directory)
                if all(prepared['gates'].values()):
                    reason=None
                    try:
                        env.record()
                        for k in range(13500):
                            budget.step('native');env.step_reference(spline,(k+1)*.002)
                            if k%500==0:progress('VAL_unguided_native',unit=unit,steps=k+1)
                    except RuntimeError as error:reason=str(error)
                    result=evaluate(env,directory,reason,progress,budget)
                    result.update(new_execution=True)
                else:result=dict(status='PREPARE_REJECTED',new_execution=False,steps=0)
                entry=dict(unit=unit,split='VAL',method=method,source_unit=row['unit'],result=result)
                atomic_json(directory/'result.json',result);results.append(entry);budget.inflight=None
                atomic_json(out/'progress_results.json',results);del env
        baseline={}
        for si,scene_name in enumerate(['stationary','drifting']):
            for group in range(3):
                for m,method in enumerate(P['methods']):
                    unit=f'd{si}{group}{m}';directory=out/unit;directory.mkdir();budget.inflight={'unit':unit}
                    progress('DEV_initialize',unit=unit);env,scene=development_environment(scene_name)
                    feature=public_features(env,scene)
                    condition_id=digest(dict(g=feature['global_condition'],local=feature['local_condition']))
                    atomic_npz(directory/'initial.npz',values=env.initial.values)
                    atomic_json(directory/'task.json',dict(scene=scene_name,initial_id=env.initial.state_id,
                        condition_hash=condition_id,waypoints=env.path.to_dict(),grasp=env.grasp_local,target_velocity=env.data.qvel[env.tv]))
                    if method=='deterministic_sequence' and scene_name in baseline:
                        previous,old,old_condition=baseline[scene_name]
                        assert condition_id==old_condition
                        if 'native_trace_sha256' in old:assert file_digest(previous/'native_trace.npz')==old['native_trace_sha256']
                        result=dict(old,new_execution=False,reused_from=previous.relative_to(ROOT).as_posix())
                        atomic_json(directory/'reuse.json',dict(source=result['reused_from'],identical_condition=True,
                            identical_control_hash=old.get('native_trace_sha256'),not_independent_replication=True))
                    else:
                        seed=20261020+si+group*1000
                        spline,prepared=proposal(models[method],method,env,scene,codec,seed,directory)
                        if all(prepared['gates'].values()):
                            reason=None;feedback=[]
                            try:
                                for tick in range(1350):
                                    velocity,record=guided_velocity(env,spline);feedback.append(record)
                                    budget.reserve_native(10);env.advance_native(velocity)
                                    if tick%50==0:progress('DEV_common_guided_native',unit=unit,tick=tick+1)
                            except RuntimeError as error:reason=str(error)
                            atomic_json(directory/'feedback.json',feedback);atomic_json(directory/'filter.json',env.filter_history)
                            result=evaluate(env,directory,reason,progress,budget);result.update(new_execution=True)
                        else:result=dict(status='PREPARE_REJECTED',new_execution=False,steps=0)
                        if method=='deterministic_sequence':baseline[scene_name]=(directory,result,condition_id)
                    atomic_json(directory/'result.json',result)
                    results.append(dict(unit=unit,split='DEV',method=method,scene=scene_name,seed_group=group,result=result))
                    budget.inflight=None;atomic_json(out/'progress_results.json',results);del env
        assert identity()==ident,'FROZEN_SOURCE_OR_INPUT_CHANGED'
        counts={split:{m:dict(logical=sum(r['split']==split and r['method']==m for r in results),
            verified=sum(r['split']==split and r['method']==m and r['result']['status']=='EXECUTION_VERIFIED' for r in results),
            unique_executions=sum(r['split']==split and r['method']==m and r['result'].get('new_execution',False) for r in results))
            for m in P['methods']} for split in ['VAL','DEV']}
        summary=dict(status='FINISHED' if all(r['result']['status']=='EXECUTION_VERIFIED' for r in results) else 'FINISHED_WITH_BLOCKED_ITEMS',
            training=trained,counts=counts,results=results,TEST=False,TRAIN_added=0,source_identity_unchanged=True,
            elapsed_s=time.monotonic()-budget.start,independent_review='PENDING',
            loss_scope='qualified temporal coefficient supervision, not current-world nonlinear loss',
            scientific_advantage='UNDECIDED_REQUIRES_RAW_PAIRED_REVIEW')
        atomic_json(out/'summary.json',summary);budget.save()
        (out/'REPORT.md').write_text('# 完整时序学习与固定开发评价\n\n'+str(counts)+'\n独立审查待完成；世界反馈贡献须分开。\n',encoding='utf-8')
        return summary
