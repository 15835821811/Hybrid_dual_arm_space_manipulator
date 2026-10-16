"""New independent origins/families; immutable goals and retained rejections."""
import time
import numpy as np
from ff_prior.storage import read_json, file_digest, digest
from ff_prior_pilot_train.storage import atomic_json, atomic_npz
from ff_prior.proposal_generator import propose, quick_gates, motion_valid
from ff_prior.coupled_predictor import pose_errors
from ff_prior.model_gates import evaluate_reference
from ff_prior.spline import Reference
from ff_prior.scenes import SceneSpec
from ff_prior_complex.tasks import task_for, choose_obstacle, scene_runtime, transfer_audit
from ff_prior_loop.runtime import at_origin, origin_audit
from ff_prior_feasible.validation import validate
from .protocol import ROOT, BASE, GENERATION


def guard_population(index, dev):
    """All previous TRAIN/VAL and original DEV/TEST goals, never TEST arrays."""
    entries=[]
    for e in index:
        if file_digest(ROOT/e['record']) != e['record_sha256']:
            raise ValueError('old record identity changed')
        row=read_json(ROOT/e['record'])
        entries.append({'sample_id':e['sample_id'], 'split':e['split'],
                        'family_id':e['family_id'], 'goals':row['condition']['goals']})
    for j,e in enumerate(dev):
        entries.append({'sample_id':f'DEV_{j}', 'split':'DEV', 'family_id':'DEV',
                        'goals':e['task']['condition']['goals']})
    for sid,e in read_json(BASE/'splits.json')['records'].items():
        if e['split']=='TEST':
            row=read_json(BASE/'records'/f'{sid}.json')
            entries.append({'sample_id':'RESERVED_'+sid, 'split':'TEST_METADATA_ONLY',
                            'family_id':row['family_id'], 'goals':row['condition']['goals']})
    return entries


def neighbours(goals, guard):
    if not guard: return []
    pos,rot=pose_errors(np.asarray(goals),np.stack([e['goals'] for e in guard]))
    mask=(pos.max(axis=1)<.01)&(rot.max(axis=1)<np.deg2rad(1.))
    return [guard[i]['sample_id'] for i in np.flatnonzero(mask)]


def assert_isolated(entry, guard):
    if entry['family_id'] in {e['family_id'] for e in guard}:
        raise ValueError('family reused across validation boundary')
    if neighbours(entry['task']['condition']['goals'],guard):
        raise ValueError('world goal is a protected neighbour')
    if entry['training_eligible'] or entry['parent_sample_id'] is not None:
        raise ValueError('validation task derived from an existing teacher')
    if entry['method_results_used'] or entry['goal_relabelled']:
        raise ValueError('task selection or relabelling used planner results')
    return True


def generate(slot, attempt, canonical, canonical_initial, profile, guard, out, deadline):
    """Only teacher model validity selects a task; a frozen task is never replaced."""
    if not 0<=slot<12 or not 0<=attempt<8: raise ValueError('outside generation budget')
    sid=f'newval_{slot:02d}'; family=f'independent_newval_family_{slot:02d}'
    result={'sample_id':sid,'family_id':family,'slot':slot,'attempt':attempt,
            'split':'NEW_VAL','parent_sample_id':None,'training_eligible':False,
            'method_results_used':False,'goal_relabelled':False,'status':'REJECTED'}
    deadline=min(deadline,time.time()+GENERATION['attempt_wall_s'])
    start=time.perf_counter(); tracked=[]; screen_calls={};screen_started=False;screen_closed=False
    try:
        rng=np.random.default_rng(GENERATION['seed']+slot*173)
        jitter=GENERATION['origin_jitter_rad'][slot%6]
        q0=canonical.spec.planner_zero+rng.uniform(-jitter,jitter,17)
        rr,ii,cc=at_origin(canonical,q0); tracked.append((rr,rr.calls.copy()))
        result['origin_audit']=origin_audit(canonical,rr,ii)
        ref,provenance=propose(rr,600000+slot*8+attempt,(slot+attempt)%5,GENERATION['seed'])
        result['provenance']=provenance
        if ref is None: result['reason']='NO_MOTION_DURATION'; return result
        scale=GENERATION['motion_scales'][slot%3]; z=q0+scale*(ref.z-q0)
        if slot>=6:
            direction=np.random.default_rng(GENERATION['seed']+slot*101+attempt*997).normal(size=17)
            direction[:10]*=.5;direction/=max(abs(direction))
            phase=np.r_[np.linspace(0,1,20)[1:-1],1.]
            z+=.12*np.sin(np.pi*phase)[:,None]*direction
        ref=next((v for duration in (10.,15.,20.,25.,29.5)
                  if motion_valid(rr,v:=Reference(z,q0,duration))),None)
        if ref is None or not all(quick_gates(rr,ref).values()):
            result['reason']='QUICK_GATES'; return result
        prediction,gates=evaluate_reference(rr,ii,ref,cc,profile,deadline)
        result['empty_scene_teacher_validation']=gates
        if gates['validation_level']!='MODEL_VALID':
            result['reason']='EMPTY_TEACHER_INVALID'; return result
        scene=SceneSpec()
        if slot>=6:
            screen_started=True
            scene,screen=choose_obstacle(rr,ii,cc,profile,ref,prediction,family,deadline)
            result['obstacle_screen']=screen; screen_calls=screen['scratch_calls'];screen_closed=True
            if scene is None: result['reason']='NO_VALID_SCRATCH_SCENE'; return result
            child,ci,_=scene_runtime(scene)
            result['scene_transfer']=transfer_audit(canonical,canonical_initial,child,ci)
            rr,ii,cc=at_origin(child,q0);tracked.append((rr,rr.calls.copy()))
        # Static geometry does not change the robot dynamics. Derive once from
        # the original teacher; independent final-scene validation cannot relabel it.
        task=task_for(rr,ii,cc,profile,prediction['ee'][-1],ref.duration)
        result.update(task=task,scene=scene.to_dict(),scene_hash=scene.scene_hash,
                      kind='static_detour' if slot>=6 else 'new_origin_large_motion',
                      reference_hash=ref.reference_hash,goal_source='new_teacher_forward_before_final_validation',
                      motion_scale=scale,maximum_motion_rad=float(abs(z-q0).max()))
        result['protected_goal_neighbours']=neighbours(task['condition']['goals'],guard)
        if result['protected_goal_neighbours']:
            result['reason']='PROTECTED_WORLD_GOAL_NEIGHBOUR';return result
        assert_isolated(result,guard)
        checks=validate(ref.z,task,rr,ii,cc,profile,deadline)
        result['witness_model']=checks
        if not checks['eligible']: result['reason']='FINAL_SCENE_WITNESS_INVALID'; return result
        if time.time()>deadline: raise RuntimeError('GENERATION_ATTEMPT_WALL_EXCEEDED')
        atomic_json(out/'tasks'/sid/'task.json',task)
        atomic_npz(out/'tasks'/sid/'initial_state.npz',values=ii.values)
        atomic_npz(out/'tasks'/sid/'witness.npz',**ref.arrays())
        result.update(status='FROZEN',initial_state_sha256=file_digest(out/'tasks'/sid/'initial_state.npz'),
                      witness_sha256=file_digest(out/'tasks'/sid/'witness.npz'),validation_level='MODEL_VALID',
                      execution_status='NOT_RUN',frozen_before_comparison=True)
    except Exception as error:
        import traceback
        result.update(status='ERROR',reason=repr(error),traceback=traceback.format_exc())
    finally:
        result['calls']={}
        for robot,before in tracked:
            for key,value in robot.calls.items():
                result['calls'][key]=result['calls'].get(key,0)+value-before[key]
        for key,value in screen_calls.items():result['calls'][key]=result['calls'].get(key,0)+value
        result['unknown_screen_call_accounting']=screen_started and not screen_closed
        result['wall_s']=time.perf_counter()-start
    return result
