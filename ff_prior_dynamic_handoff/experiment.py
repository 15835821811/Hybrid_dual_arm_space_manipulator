"""Two exact dynamic tasks; deterministic baseline sharing is not replication."""
import time
import zipfile
import subprocess
import numpy as np
import torch
from ff_prior.robot_adapter import RobotAdapter
from ff_prior.storage import RunLock,read_json,digest,file_digest
from ff_prior_pilot_train.storage import atomic_json,atomic_npz
from ff_prior.contracts import frozen_task
from ff_prior.guidance import public_goal_initializer
from ff_prior.spline import Reference
from ff_prior_endpoint_jacobian.learning import load
from ff_prior_endpoint_precision.comparison import unguided_sample
from .protocol import ROOT,P,OLD,MODEL,GATE
from .inputs import identity
from .execution import Environment,evaluate,task_identity

class Budget:
    def __init__(self,out):
        self.out=out;self.start=time.monotonic();self.native=0;self.replay=0;self.inflight=None
    def check(self):
        if time.monotonic()-self.start>P['wall_s']:raise TimeoutError('WALL_BUDGET')
    def reserve_native(self,n):
        self.check()
        if self.native+self.replay+n>P['max_native_replay_steps']:raise TimeoutError('STEP_BUDGET')
        self.native+=n
    def step(self,kind):
        assert kind=='replay';self.check()
        if self.native+self.replay>=P['max_native_replay_steps']:raise TimeoutError('STEP_BUDGET')
        self.replay+=1
    def save(self):
        self.check();size=sum(p.stat().st_size for p in self.out.rglob('*') if p.is_file())
        if size>P['max_bytes']:raise TimeoutError('OUTPUT_BUDGET')
        atomic_json(self.out/'budget.json',dict(native_steps_reserved=self.native,replay_steps_reserved=self.replay,
            elapsed_s=time.monotonic()-self.start,output_bytes=size,inflight=self.inflight,limits=P))

def run(out,notify):
    torch.set_num_threads(P['threads']);budget=Budget(out);ident=identity()
    def progress(stage,**kw):budget.save();notify(stage,**kw)
    assert read_json(GATE/'summary.json')['next_dynamic_DEV_gate']
    review=read_json(ROOT/'paper/review-traces/experiment-result-to-claim/20261022_dynamic_margin_run01/verdict.json')
    if review['claim_supported'] not in ('yes','partial'):raise ValueError('P2A_REVIEW_BLOCKED')
    with RunLock(out):
        atomic_json(out/'run.json',dict(identity=ident,producer_head=subprocess.check_output(
            ['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),TEST=False))
        with zipfile.ZipFile(out/'executed_source.zip','w',zipfile.ZIP_DEFLATED) as z:
            for path in ident['source_hashes']:z.write(ROOT/path,path)
        bundle=load(MODEL)
        atomic_json(out/'model_loaded.json',dict(model_sha256=file_digest(MODEL/'model.pt'),
            codec_sha256=file_digest(MODEL/'codec.json'),training_updates=0,model_class=type(bundle[0]).__name__))
        results=[];baselines={}
        for row in ident['selection']['units']:
            key=row['id'];directory=out/key;directory.mkdir()
            budget.inflight=dict(unit=key);progress('initialize',unit=key)
            rr=RobotAdapter();env=Environment(rr,row['scene']=='drifting',row['task_seed'])
            task=task_identity(env,row['scene'],row['task_seed'])
            assert digest(task)==row['task_hash'],'frozen original dynamic task changed'
            atomic_json(directory/'task.json',task)
            atomic_json(directory/'identity.json',dict(robot=rr.identity(),initial_state_id=env.initial.state_id,
                task_hash=digest(task),method=row['method'],proposal_seed=row['proposal_seed']))
            atomic_npz(directory/'initial.npz',values=env.initial.values)
            if row['method']=='task_controller' and row['scene'] in baselines:
                previous,old=baselines[row['scene']]
                same=bool(digest(read_json(previous/'task.json'))==digest(task)
                          and digest(read_json(previous/'identity.json')['robot'])==digest(rr.identity())
                          and file_digest(previous/'native_trace.npz')==old['native_trace_sha256'])
                if not same:raise ValueError('COMPLETED_BASELINE_IDENTITY_CHANGED')
                result=dict(old,new_execution=False,reused_from=previous.relative_to(ROOT).as_posix(),
                            paired_reference_only=True,actual_new_steps=0)
                atomic_json(directory/'reuse.json',dict(identity_exact=True,source_control_sha256=old['native_trace_sha256'],
                    reason='Sampler seed is not an input to deterministic controller; same executed unit reused',
                    new_independent_replication=False,source=result['reused_from']))
                atomic_json(directory/'result.json',result)
            else:
                prior=None
                if row['method']=='diffusion_null_prior':
                    goal=frozen_task(dict(initial_features=rr.spec.planner_zero,goals=env.desired_poses(27.),
                        duration=27.,pose_mask=np.ones((2,6))),
                        {'scope':'dynamic_transfer_terminal_only_NOT_native_V7_eligible'})
                    base=public_goal_initializer(goal,rr,env.initial)
                    z,steps=unguided_sample(bundle,base,goal,row['proposal_seed'])
                    if not np.isfinite(z).all():raise RuntimeError('NONFINITE_PRIOR')
                    prior=Reference(z,rr.spec.planner_zero,27.)
                    atomic_json(directory/'prior.json',dict(task=goal,base=base,Z=z,steps=steps,
                        seed=row['proposal_seed'],new_network_calls=16,
                        scope='terminal-only prior; dynamic task followed by common feedback controller'))
                reason=None
                try:
                    for tick in range(1350):
                        velocity=env.velocity(prior)
                        budget.reserve_native(10);env.advance_native(velocity)
                        if tick%50==0:progress('native_torque',unit=key,tick=tick+1)
                except RuntimeError as error:reason=str(error)
                atomic_json(directory/'control_filter.json',env.filter_history)
                result=evaluate(env,directory,reason,progress,budget)
                result.update(new_execution=True,actual_new_steps=result['steps'])
                atomic_json(directory/'result.json',result)
                if row['method']=='task_controller':baselines[row['scene']]=(directory,result)
            results.append(dict(unit=row,result=result));budget.inflight=None
            progress('unit_complete',unit=key,status=result['status'])
            atomic_json(out/'progress_results.json',results)
            del env,rr
        if identity()!=ident:raise ValueError('FROZEN_IDENTITY_CHANGED')
        counts={method:dict(logical_units=6,verified=sum(x['result']['status']=='EXECUTION_VERIFIED'
            for x in results if x['unit']['method']==method)) for method in ('task_controller','diffusion_null_prior')}
        passed=all(x['result']['status']=='EXECUTION_VERIFIED' for x in results)
        pairs=[]
        for scene in P['task_seeds']:
            for group in range(3):
                matched={x['unit']['method']:x for x in results if x['unit']['scene']==scene and x['unit']['seed_group']==group}
                a,b=[matched[m]['result'] for m in ('task_controller','diffusion_null_prior')]
                pairs.append(dict(scene=scene,seed_group=group,baseline=matched['task_controller']['unit']['id'],
                    prior=matched['diffusion_null_prior']['unit']['id'],baseline_status=a['status'],prior_status=b['status'],
                    both_complete=bool(a['gates']['complete_native_time'] and b['gates']['complete_native_time']),
                    position_delta_mm={k:(np.asarray(b['position_mm'][k])-np.asarray(a['position_mm'][k])).tolist()
                        for k in ('mean','p95','maximum','final')},
                    rotation_delta_deg={k:(np.asarray(b['rotation_deg'][k])-np.asarray(a['rotation_deg'][k])).tolist()
                        for k in ('mean','p95','maximum','final')}))
        atomic_json(out/'paired_statistics.json',dict(pairs=pairs,delta='prior minus baseline',
            independent_tasks=2,logical_pairs=6,baseline_replications_are_shared=True,
            failures_retained=True,unseen_generalization_claim=False))
        summary=dict(status='FINISHED' if passed else 'FINISHED_WITH_BLOCKED_ITEMS',results=results,by_method=counts,
            logical_units=12,unique_new_executions=sum(x['result']['new_execution'] for x in results),
            deterministic_shared_units=4,network_calls=bundle[0].calls,training_updates=0,TRAIN_added=0,TEST=False,
            elapsed_s=time.monotonic()-budget.start,independent_review='PENDING',
            caveat='Two fixed tasks; baseline sharing is not six independent repetitions; terminal network has no full temporal condition')
        atomic_json(out/'summary.json',summary);budget.save()
        (out/'REPORT.md').write_text('# P2b动态交接运行结束\n\n'+str(counts)+'\n独立审查待完成。\n',encoding='utf-8')
        return summary
