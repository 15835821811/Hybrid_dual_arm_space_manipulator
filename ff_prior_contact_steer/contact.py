"""Whole-short-edge distance feedback; planning only, never physical state writes."""
import hashlib
import json
import mujoco
import numpy as np
from scipy.optimize import minimize
from ff_prior.storage import digest
from ff_prior_fixed.correction import pose_vector
from ff_prior_feasible.guidance import group_norms
from ff_prior_pilot_train.storage import atomic_json, atomic_npz
from .protocol import P
from .tree import propagate, Connector, collect, minimum_duration
from .budget import BudgetExhausted


def quality(node, goals):
    return float(max(group_norms(pose_vector(node.ee, goals))))


def exact_key(node, context, target=None, duration=None):
    h=hashlib.sha256()
    h.update(json.dumps(context, sort_keys=True, separators=(',',':')).encode())
    for a in (node.state, node.q, node.qpos, node.qvel,
              np.asarray([node.elapsed]), np.asarray([] if target is None else target),
              np.asarray([] if duration is None else [duration])):
        a=np.ascontiguousarray(a, dtype=np.float64)
        h.update(str(a.shape).encode());h.update(a.tobytes())
    return h.hexdigest()


def bounded_world(j, e):
    x=-np.linalg.solve(j.T@j+.001*np.eye(17),j.T@e)
    return x*min(1.,P['contact_trust_rad']/max(float(abs(x).max()),1e-12))


def tangent(jd, distances, goal_delta):
    """A deterministic tangent in planner coordinates, with both signs tested."""
    if not len(distances):return np.zeros(17)
    normal=jd[int(np.argmin(distances))].copy()
    normal/=max(float(np.linalg.norm(normal)),1e-12)
    g=goal_delta/max(float(np.linalg.norm(goal_delta)),1e-12)
    # Select a tangent orthogonal to the nearest obstacle normal and goal ray.
    matrix=np.stack([normal,g]);projector=np.eye(17)-np.linalg.pinv(matrix)@matrix
    # Prefer coordinates that actually influence active robot geometry; choosing
    # a large nullspace column alone could move only an unrelated arm.
    sensitivity=np.linalg.norm(jd,axis=0)
    if float(sensitivity.max())<1e-12:return np.zeros(17)
    k=int(np.argmax(np.linalg.norm(projector,axis=0)*sensitivity))
    t=projector[:,k];t/=max(float(abs(t).max()),1e-12)
    return P['contact_tangent_rad']*t


def solve_direction(parent_q, nominal, e, jg, distances, jd, lower, upper, variant):
    """Linearized distance constraints and bounded physical joint increments.

    SLSQP status is not a physical gate. Only finite, constrained solutions
    proceed to original full-edge checks. Unsafe nominal guesses are never nodes.
    """
    radius=P['contact_trust_rad'];scale=P['contact_distance_scale_m']
    lo=np.maximum(lower-parent_q,-radius);hi=np.minimum(upper-parent_q,radius)
    bias=variant*tangent(jd,distances,nominal)
    normalizer=max(1.,float(e@e))
    def fun(x):
        r=e+jg@(x-nominal)
        return .5*float(r@r)/normalizer+.015*float(np.sum(((x-nominal-bias)/radius)**2))
    def jac(x):
        return jg.T@(e+jg@(x-nominal))/normalizer+.03*(x-nominal-bias)/radius**2
    def distance(x):return (distances+jd@(x-nominal)-P['contact_margin_m'])/scale
    constraints=[] if not len(distances) else [{'type':'ineq','fun':distance,'jac':lambda x:jd/scale}]
    solved=minimize(fun,np.clip(nominal,lo,hi),jac=jac,bounds=list(zip(lo,hi)),
        constraints=constraints,method='SLSQP',options={'maxiter':P['contact_solver_iterations'],'ftol':1e-9})
    x=np.asarray(solved.x);margin=float(np.min(distance(x),initial=np.inf))
    admissible=bool(np.isfinite(x).all() and np.all(x>=lo-1e-9) and np.all(x<=hi+1e-9)
        and margin>=-1e-6 and abs(x).max()>1e-9)
    return x,dict(success=bool(solved.success),message=str(solved.message),
        iterations=int(solved.nit),linear_admissible=admissible,
        minimum_normalized_linear_distance=margin if np.isfinite(margin) else None,
        objective=float(fun(x)),variant=variant,tangent_bias=bias,
        lower=lo,upper=hi,delta=x,qualification='DIRECTION_ONLY')


def predict(parent, target, duration, connector):
    child,r,a=propagate(parent,target,duration,connector.robot,connector.collision,
        connector.budget,P['contact_grid'],False)
    if child is None or not r['numerical_ok']:raise ValueError('CONTACT_DERIVATIVE_NUMERICAL_FAILURE')
    return child,r,a


def scan_active(arrays, robot, collision):
    scratch=mujoco.MjData(robot.model);values={};scans=[]
    for i in range(P['contact_phase_stride'],P['contact_grid']+1,P['contact_phase_stride']):
        scratch.qpos[:]=arrays['qpos'][i];robot.kinematics(scratch)
        result=collision.scan(scratch,P['contact_buffer_m'])
        scans.append(dict(index=i,minimum=result['minimum_clearance_m'],pair=result['minimum_pair']))
        for v in result['violations']:
            ga=robot.id(mujoco.mjtObj.mjOBJ_GEOM,v['geom_a']);gb=robot.id(mujoco.mjtObj.mjOBJ_GEOM,v['geom_b'])
            values[(i,ga,gb)]=float(v['distance_m'])
    # First preserve each phase's worst pair, then each pair's worst phase.
    selected=set()
    for i in sorted({k[0] for k in values}):
        selected.add(min((k for k in values if k[0]==i),key=lambda k:(values[k],k)))
    for pair in sorted({k[1:] for k in values}):
        if len(selected)>=P['contact_active_cap']:break
        selected.add(min((k for k in values if k[1:]==pair),key=lambda k:(values[k],k)))
    for k in sorted(values,key=lambda k:(values[k],k)):
        if len(selected)>=P['contact_active_cap']:break
        selected.add(k)
    active=sorted(selected)
    return active,np.asarray([values[k] for k in active]),scans,len(values)


def distances_at(arrays, active, robot):
    scratch=mujoco.MjData(robot.model);result=[];last=None
    for i,ga,gb in active:
        if i!=last:scratch.qpos[:]=arrays['qpos'][i];robot.kinematics(scratch);last=i
        result.append(mujoco.mj_geomDistance(robot.model,scratch,int(ga),int(gb),2.5,None))
    robot.calls['distance']+=len(active)
    return np.asarray(result)


def differences(target, lower, upper):
    h=np.full(17,P['contact_difference_rad'])
    h[target+h>upper]*=-1
    if np.any(target+h<lower):raise ValueError('NO_DIFFERENCE_DOMAIN')
    return h


def linearize(path,parent,goals,connector):
    r=connector.robot;duration=P['contact_duration_s']
    center,cr,ca=predict(parent,parent.q,duration,connector)
    e0=pose_vector(center.ee,goals);h=differences(parent.q,r.lower,r.upper)
    hold_ee=[];hold_states=[]
    for k in range(17):
        q=parent.q.copy();q[k]+=h[k];n,_,_=predict(parent,q,duration,connector)
        hold_ee.append(n.ee);hold_states.append(n.state)
    hold_j=np.stack([(pose_vector(ee,goals)-e0)/hh for ee,hh in zip(hold_ee,h)],axis=1)
    nominal=bounded_world(hold_j,e0)
    # Constrain a proposal, never clip an actual or accepted state.
    nominal=np.clip(nominal,r.lower-parent.q,r.upper-parent.q)
    target=parent.q+nominal
    duration=max(duration,minimum_duration(nominal,r.spec.planner_velocity_limits,r.spec.planner_acceleration_limits))
    child,cr,arrays=predict(parent,target,duration,connector)
    active,ds,scans,total_active=scan_active(arrays,r,connector.collision)
    e=pose_vector(child.ee,goals);h2=differences(target,r.lower,r.upper)
    ees=[];states=[];qposes=[];dds=[]
    for k in range(17):
        q=target.copy();q[k]+=h2[k];n,_,a=predict(parent,q,duration,connector)
        ees.append(n.ee);states.append(n.state);qposes.append(a['qpos']);dds.append(distances_at(a,active,r))
    jg=np.stack([(pose_vector(ee,goals)-e)/hh for ee,hh in zip(ees,h2)],axis=1)
    jd=np.stack([(d-ds)/hh for d,hh in zip(dds,h2)],axis=1)
    if not np.isfinite(jg).all() or not np.isfinite(jd).all():raise ValueError('NONFINITE_CONTACT_JACOBIAN')
    # Independent half step for the strongest distance coordinate; observation
    # only, never an alternative safety certificate or a replacement acceptance.
    k=int(np.argmax(np.linalg.norm(jd,axis=0))) if len(active) else int(np.argmax(np.linalg.norm(jg,axis=0)))
    q=target.copy();q[k]+=.5*h2[k];half,_,ha=predict(parent,q,duration,connector)
    hd=distances_at(ha,active,r)
    atomic_npz(path/'jacobian.npz',parent_state=parent.state,parent_q=parent.q,center_state=center.state,
        center_ee=center.ee,hold_e=e0,hold_j=hold_j,hold_epsilon=h,hold_ee=np.asarray(hold_ee),
        hold_states=np.asarray(hold_states),nominal=nominal,duration=np.asarray(duration),
        nominal_ee=child.ee,nominal_state=child.state,nominal_qpos=arrays['qpos'],goal=e,
        distances=ds,jg=jg,jd=jd,epsilon=h2,perturb_ee=np.asarray(ees),
        perturb_states=np.asarray(states),perturb_qpos=np.asarray(qposes),perturb_distances=np.asarray(dds),
        half_coordinate=np.asarray(k),half_ee=half.ee,half_distances=hd,half_qpos=ha['qpos'])
    atomic_json(path/'jacobian.json',dict(active=active,scans=scans,all_near_pairs=total_active,
        active_cap=P['contact_active_cap'],queries=37,parent_hash=digest(parent.state),
        geometry_policy_hash=connector.collision.policy_hash,qualification='DERIVATIVE_ONLY',
        intervals=P['contact_grid'],duration=duration,whole_edge_sample_indices=list(range(8,257,8))))
    return nominal,e,jg,ds,jd


def search(directory,prefix,task,robot,collision,budget,context,progress):
    nodes=list(prefix);first=len(nodes)-1;expanded=set();attempts=[];keys=set()
    connector=Connector(directory,robot,collision,budget);goals=task['condition']['goals'];total=27.
    nodekeys={exact_key(n,context) for n in nodes};solution=None;stop='PARENT_LIMIT'
    try:
        for iteration in range(P['search_attempt_cap']):
            possible=[i for i in range(first,len(nodes)) if i not in expanded
                and nodes[i].elapsed+P['contact_duration_s']+P['tree_min_duration_s']<=total]
            if not possible:stop='NO_UNEXPANDED_TIME_FEASIBLE_PARENT';break
            if not budget.can_fit(110,P['contact_interval_reserve']):stop='RESERVED_BUDGET_LIMIT';break
            index=min(possible,key=lambda i:(np.log1p(quality(nodes[i],goals))+
                P['contact_parent_time_weight']*nodes[i].elapsed/total,i))
            parent=nodes[index];expanded.add(index);path=directory/'linear'/f'{iteration:03d}'
            progress('contact_jacobian_short_edge',parent=index,iteration=iteration,nodes=len(nodes),best_Q=min(quality(n,goals) for n in nodes))
            nominal,e,jg,ds,jd=linearize(path,parent,goals,connector)
            row=dict(iteration=iteration,parent=index,before_Q=quality(parent,goals),trials=[],solvers=[])
            for variant in P['contact_variant_ids']:
                step,solved=solve_direction(parent.q,nominal,e,jg,ds,jd,robot.lower,robot.upper,variant)
                row['solvers'].append(solved)
                if not solved['linear_admissible']:continue
                for scale in P['contact_line_scales']:
                    target=parent.q+scale*step
                    duration=max(P['contact_duration_s'],minimum_duration(target-parent.q,robot.spec.planner_velocity_limits,robot.spec.planner_acceleration_limits))
                    trial=dict(variant=variant,scale=scale,target=target,duration=duration,accepted=False)
                    if parent.elapsed+duration+P['tree_min_duration_s']>total:
                        trial['reason']='REMAINING_TIME';row['trials'].append(trial);continue
                    key=exact_key(parent,context,target,duration);trial['exact_action_key']=key
                    if key in keys:trial['reason']='EXACT_ACTION_DUPLICATE';row['trials'].append(trial);continue
                    keys.add(key);before=connector.calls;child=connector.edge(parent,target,duration)
                    trial['edge_ids']=list(range(before,connector.calls))
                    if child is None:trial['reason']='ORIGINAL_EDGE_GATE_REJECTED'
                    else:
                        cq=quality(child,goals);trial['after_Q']=cq
                        nk=exact_key(child,context)
                        if nk in nodekeys:trial['reason']='EXACT_FULL_STATE_DUPLICATE'
                        elif cq>P['contact_world_growth']*quality(parent,goals):trial['reason']='BOUNDED_WORLD_DETOUR_EXCEEDED'
                        else:
                            child.parent=index;nodes.append(child);nodekeys.add(nk);idx=len(nodes)-1
                            trial.update(accepted=True,node=idx,reason='EDGE_SCREEN_ONLY',edge=child.edge_id)
                            if cq<=P['steer_terminal_margin']:
                                final=connector.edge(child,child.q,total-child.elapsed)
                                if final is not None and quality(final,goals)<=P['steer_terminal_margin']:
                                    final.parent=idx;nodes.append(final);solution=len(nodes)-1;stop='COMPLETE_SCREEN_ROUTE'
                    row['trials'].append(trial)
                    if trial['accepted']:break
                if solution is not None:break
            attempts.append(row);atomic_json(directory/'attempts.json',attempts)
            if solution is not None:break
            if len(nodes)-len(prefix)>=P['accepted_node_cap']:stop='NODE_CAP';break
    except BudgetExhausted as error:stop=str(error)
    atomic_json(directory/'attempts.json',attempts)
    atomic_npz(directory/'nodes.npz',**{k:np.asarray([getattr(n,k) for n in nodes])
        for k in ('q','state','qpos','qvel','elapsed','parent','duration','ee','edge_id')})
    result=dict(stop=stop,nodes=len(nodes),historical_prefix_nodes=len(prefix),new_nodes=len(nodes)-len(prefix),
        expanded_parents=len(expanded),attempts=len(attempts),propagations=connector.calls,solution_node=solution,
        best_Q=min(quality(n,goals) for n in nodes),qualification='TREE_SCREEN_ONLY',
        exact_unique_action_keys=len(keys),context=context,full_initial_state_execution_required=True)
    atomic_json(directory/'result.json',result)
    return collect(nodes,solution) if solution is not None else None,result
