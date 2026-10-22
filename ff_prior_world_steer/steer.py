"""Short world-goal extensions and saved-state parent selection.

Every accepted short edge remains in the tree even when the final target is
not reached. A complete path ends at the original 27 s deadline; the final hold
starts from the actual propagated node, never a reset to a nominal base pose.
"""
import numpy as np
from ff_prior_fixed.correction import pose_vector
from ff_prior_feasible.guidance import group_norms
from ff_prior.storage import digest
from ff_prior_pilot_train.storage import atomic_json, atomic_npz
from .protocol import P
from .budget import BudgetExhausted


def residual(node,goals):
    return pose_vector(node.ee,goals)


def quality(node,goals):
    return float(max(group_norms(residual(node,goals))))


def world_parent(nodes,visits,goals,total,rng):
    eligible=[i for i,n in enumerate(nodes)
              if n.elapsed+P['steer_duration_s']+P['tree_min_duration_s']<=total
              and len(_ancestors(nodes,i))<P['steer_max_depth']]
    if not eligible:return None
    ranked=sorted(eligible,key=lambda i:(
        np.log1p(quality(nodes[i],goals))
        +P['steer_parent_visit_weight']*np.log1p(visits[i])
        +P['steer_parent_time_weight']*nodes[i].elapsed/total,i))
    # Deterministic best-first most of the time, with a frozen seeded top-k mix.
    return ranked[0] if rng.random()<.75 else int(rng.choice(ranked[:P['steer_parent_topk']]))


def _ancestors(nodes,index):
    path=[]
    while index>=0:
        path.append(index);index=nodes[index].parent
    return path


def world_nearest(nodes,target,scale,goals,total,visits):
    eligible=[i for i,n in enumerate(nodes)
              if n.elapsed+P['steer_duration_s']+P['tree_min_duration_s']<=total
              and len(_ancestors(nodes,i))<P['steer_max_depth']]
    if not eligible:return None
    return min(eligible,key=lambda i:(
        np.linalg.norm((nodes[i].q-target)/scale)
        +P['world_nearest_weight']*np.log1p(quality(nodes[i],goals))
        +P['steer_parent_time_weight']*nodes[i].elapsed/total
        +.05*np.log1p(visits[i]),i))


def delta_from_jacobian(j,e):
    delta=-np.linalg.solve(j.T@j+P['steer_damping']*np.eye(17),j.T@e)
    delta*=min(1.,P['steer_max_step_rad']/max(float(abs(delta).max()),1e-12))
    if not np.isfinite(delta).all():raise ValueError('NONFINITE_WORLD_STEER')
    return delta


def linearize(parent,connector,goals,path):
    from .tree import propagate
    robot=connector.robot
    args=(robot,connector.collision,connector.budget,P['steer_difference_intervals'],False)
    center,report,_=propagate(parent,parent.q,P['steer_duration_s'],*args)
    if not report['numerical_ok']:raise ValueError('WORLD_STEER_CENTER_NUMERICAL')
    e=residual(center,goals);poses=[];states=[];columns=[];eps=[];targets=[]
    for k in range(17):
        h=P['tree_endpoint_difference_rad']
        if parent.q[k]+h>robot.upper[k]:h=-h
        target=parent.q.copy();target[k]+=h
        if target[k]<robot.lower[k]:raise ValueError('NO_DIFFERENCE_DOMAIN')
        other,r,_=propagate(parent,target,P['steer_duration_s'],*args)
        if not r['numerical_ok']:raise ValueError('WORLD_STEER_DIFFERENCE_NUMERICAL')
        eps.append(h);poses.append(other.ee);states.append(other.state);targets.append(target)
        columns.append((residual(other,goals)-e)/h)
    j=np.asarray(columns).T;delta=delta_from_jacobian(j,e)
    atomic_npz(path/'linearization.npz',parent_state=parent.state,parent_q=parent.q,
        center_state=center.state,center_ee=center.ee,goal=e,jacobian=j,
        perturb_ee=np.asarray(poses),perturb_state=np.asarray(states),
        perturb_target=np.asarray(targets),epsilon=np.asarray(eps),delta=delta)
    atomic_json(path/'linearization.json',{'parent_hash':digest(parent.state),
        'duration':P['steer_duration_s'],'intervals':P['steer_difference_intervals'],
        'queries':18,'qualified':False,'purpose':'model derivative, never accepted as path',
        'all_numerical_checks':True})
    return delta


def short_target(parent,delta,scale,robot,total):
    from .tree import minimum_duration
    target=parent.q+scale*delta
    duration=max(P['steer_duration_s'],
        minimum_duration(target-parent.q,robot.spec.planner_velocity_limits,robot.spec.planner_acceleration_limits))
    if parent.elapsed+duration+P['tree_min_duration_s']>total:return None
    # No clipping of a propagated state or an accepted edge endpoint.
    if np.any(target<robot.lower) or np.any(target>robot.upper):return None
    return target,duration


def search(directory,proposal,seed,task,robot,initial,collision,budget,progress):
    from .tree import root_node,Connector,Reference,collect
    root=root_node(robot,initial);nodes=[root];visits=[0];derivatives={}
    rng=np.random.default_rng(seed+440001);total=float(task['condition']['duration'])
    goals=task['condition']['goals'];connector=Connector(directory,robot,collision,budget)
    scale=np.r_[np.full(10,.10),np.full(7,.65)]
    reference=Reference(proposal,root.q,total);qgoal=proposal[-1].copy()
    attempts=[];solution=None;stop='ATTEMPT_LIMIT'
    try:
        for attempt in range(P['search_attempt_cap']):
            progress('short_world_goal_expansion',candidate=directory.parent.name,
                     attempt=attempt,nodes=len(nodes),best_Q=min(quality(n,goals) for n in nodes))
            if not budget.can_fit(P['steer_query_reserve'],P['steer_interval_reserve']):
                stop='SEARCH_BUDGET_RESERVED';break
            r=rng.random()
            if r<P['steer_world_probability']:
                mode='WORLD_DESCENT'
                index=world_parent(nodes,visits,goals,total,rng)
                if index is None:stop='NO_TIME_FEASIBLE_PARENT';break
                parent=nodes[index]
                if index not in derivatives:
                    derivatives[index]=linearize(parent,connector,goals,directory/'steers'/f'{index:04d}')
                delta=derivatives[index].copy()
            else:
                mode='PROPOSAL' if r<P['steer_world_probability']+P['steer_proposal_probability'] else 'EXPLORATION'
                if mode=='PROPOSAL':
                    target=reference.evaluate(rng.uniform(0,total))+rng.normal(0,.20,17)*scale
                elif rng.random()<.25:
                    target=rng.uniform(robot.lower,robot.upper)
                else:
                    s=rng.random();target=(1-s)*root.q+s*qgoal+rng.normal(0,1.,17)*scale
                # Bounds constrain a newly drawn planning proposal only.
                target=np.minimum(np.maximum(target,robot.lower),robot.upper)
                index=world_nearest(nodes,target,scale,goals,total,visits)
                if index is None:stop='NO_TIME_FEASIBLE_PARENT';break
                parent=nodes[index];delta=target-parent.q
                delta*=min(1.,P['edge_max_joint_step_rad']/max(float(abs(delta).max()),1e-12))
            visits[index]+=1
            row={'attempt':attempt,'parent':index,'sampling':mode,'accepted':False,
                 'before_Q':quality(parent,goals),'parent_elapsed':parent.elapsed,'trials':[]}
            for line_scale in P['steer_line_scales']:
                proposed=short_target(parent,delta,line_scale,robot,total)
                trial={'scale':line_scale,'accepted':False,'edge_ids':[]}
                if proposed is None:
                    trial['reason']='DOMAIN_OR_REMAINING_TIME';row['trials'].append(trial);continue
                target,duration=proposed;before=connector.calls
                child=connector.edge(parent,target,duration)
                trial.update(target=target,duration=duration,edge_ids=list(range(before,connector.calls)))
                if child is None:trial['reason']='MOTION_OR_GEOMETRY_REJECTED'
                else:
                    e0,e1=residual(parent,goals),residual(child,goals)
                    trial.update(after_Q=quality(child,goals),ee=child.ee)
                    if mode=='WORLD_DESCENT' and not e1@e1<e0@e0-1e-10:
                        trial['reason']='NO_WORLD_DESCENT'
                    else:
                        child.parent=index;nodes.append(child);visits.append(0)
                        idx=len(nodes)-1
                        trial.update(accepted=True,node=idx,edge=child.edge_id)
                        row.update(accepted=True,node=idx,after_Q=quality(child,goals),
                                   child_elapsed=child.elapsed,remaining_time=total-child.elapsed)
                        # A short improving node is retained regardless of final goal success.
                        if quality(child,goals)<=P['steer_terminal_margin']:
                            remaining=total-child.elapsed;first_edge=connector.calls
                            final=connector.edge(child,child.q,remaining)
                            row['terminal_hold_edges']=list(range(first_edge,connector.calls))
                            if final is not None and quality(final,goals)<=P['steer_terminal_margin']:
                                final.parent=idx;nodes.append(final);visits.append(0)
                                solution=len(nodes)-1;stop='GOAL_CONNECTED_SCREEN_ONLY'
                row['trials'].append(trial)
                if trial['accepted']:break
            attempts.append(row)
            # Persist attempts alongside immutable edge records; unknown interrupted work is not resumed.
            atomic_json(directory/'attempts.json',attempts)
            if solution is not None:break
            if len(nodes)>=P['accepted_node_cap']:stop='NODE_LIMIT';break
    except BudgetExhausted as error:stop=str(error)
    best=min(range(len(nodes)),key=lambda i:quality(nodes[i],goals))
    atomic_json(directory/'attempts.json',attempts)
    atomic_npz(directory/'nodes.npz',q=np.asarray([n.q for n in nodes]),
        qpos=np.asarray([n.qpos for n in nodes]),qvel=np.asarray([n.qvel for n in nodes]),
        state=np.asarray([n.state for n in nodes]),ee=np.asarray([n.ee for n in nodes]),
        elapsed=np.asarray([n.elapsed for n in nodes]),duration=np.asarray([n.duration for n in nodes]),
        parent=np.asarray([n.parent for n in nodes]),edge_id=np.asarray([n.edge_id for n in nodes]),
        visits=np.asarray(visits),Q=np.asarray([quality(n,goals) for n in nodes]))
    result={'stop':stop,'nodes':len(nodes),'attempts':len(attempts),'propagations':connector.calls,
        'solution_node':solution,'best_node':best,'best_Q':quality(nodes[best],goals),
        'best_elapsed':nodes[best].elapsed,'seed':seed+440001,'qualification':'TREE_SCREEN_ONLY',
        'complete_parent_state_carried':True,'same_q_nodes_not_merged':True,
        'private_solution_used':False,'short_improving_states_retained':True,
        'world_linearizations':len(derivatives),'final_original_deadline':total}
    atomic_json(directory/'result.json',result)
    return collect(nodes,solution) if solution is not None else None,result
