"""Forward tree whose nodes retain the complete floating model state.

This is model planning, never a native trajectory or a reset of actual execution.
Edges are quintic rest-to-rest motions; zero node velocity follows from the
analytic endpoint derivative, rather than being overwritten after propagation.
"""
from dataclasses import dataclass
import time
import mujoco
import numpy as np
from scipy.optimize import lsq_linear
from ff_prior.initial_state import STATE
from ff_prior.spline import Reference
from ff_prior.storage import digest
from ff_prior_fixed.correction import pose_vector
from ff_prior_feasible.guidance import group_norms
from ff_prior_pilot_train.storage import atomic_npz, atomic_json
from .protocol import P
from .budget import BudgetExhausted


def blend(s, derivative=0):
    s=np.asarray(s)
    if derivative==0:return s**3*(10+s*(-15+6*s))
    if derivative==1:return 30*s**2*(1-s)**2
    if derivative==2:return 60*s*(1-s)*(1-2*s)
    raise ValueError('derivative')


@dataclass
class Node:
    q: np.ndarray
    state: np.ndarray
    qpos: np.ndarray
    qvel: np.ndarray
    elapsed: float
    parent: int
    duration: float
    ee: np.ndarray
    edge_id: int


def state_array(model,data):
    value=np.empty(mujoco.mj_stateSize(model,STATE))
    mujoco.mj_getState(model,data,value,STATE)
    return value


def root_node(robot,initial):
    d=initial.reset(robot);robot.kinematics(d,mass=True)
    if np.max(abs(d.qvel))>1e-12 or np.any(d.xfrc_applied) or np.any(d.qfrc_applied):
        raise ValueError('ONLY_ORIGINAL_STATIC_ZERO_MOMENTUM_SUPPORTED')
    return Node(robot.Bplus@d.qpos[robot.qpos_ids],state_array(robot.model,d),
        d.qpos.copy(),d.qvel.copy(),0.,-1,0.,robot.ee(d),-1)


def minimum_duration(delta,velocity,acceleration):
    # Exact maxima of the scalar quintic velocity and acceleration.
    return max(P['tree_min_duration_s'],P['tree_time_margin']*max(
        float(np.max(1.875*abs(delta)/velocity)),
        float(np.max(np.sqrt((10/np.sqrt(3))*abs(delta)/acceleration)))))


def edge_motion_valid(robot,a,b,duration):
    q=np.stack([a,b]);low=q@robot.B.T
    limited=robot.model.jnt_limited[robot.joints].astype(bool)
    limits=robot.model.jnt_range[robot.joints]
    return bool(np.isfinite(q).all() and duration>=minimum_duration(b-a,
        robot.spec.planner_velocity_limits,robot.spec.planner_acceleration_limits)
        and np.all(q>=robot.lower) and np.all(q<=robot.upper)
        and np.all(low[:,limited]>=limits[limited,0]) and np.all(low[:,limited]<=limits[limited,1]))


def propagate(parent,target,duration,robot,collision,budget,intervals,check_collision=True):
    """Same midpoint zero-momentum equations as predict_coupled, from parent.

    No model InitialState.reset is called here. All INTEGRATION fields are
    restored from the parent node. Only prescribed arm coordinates and derived
    model velocities are written; free-base pose is integrated continuously.
    """
    budget.consume(intervals);robot.calls['coupled']+=1
    m=robot.model;d=mujoco.MjData(m)
    mujoco.mj_setState(m,d,parent.state,STATE)
    assert np.array_equal(d.qpos,parent.qpos) and np.array_equal(d.qvel,parent.qvel)
    assert np.max(abs(d.qpos[robot.qpos_ids]-robot.B@parent.q))<1e-10
    assert np.max(abs(parent.qvel))<1e-10, 'rest-to-rest node invariant'
    assert np.max(abs(d.qfrc_applied),initial=0)==0 and not np.any(d.xfrc_applied)
    start_time=float(d.time);a,b=robot.dof_ids,robot.base_dofs
    bb,ba=np.ix_(b,b),np.ix_(b,a);delta=np.asarray(target)-parent.q
    residual=0.;momentum=0.;qerror=0.;previous=None;extent=0.;minimum=2.5;pair=None
    records={k:[] for k in ('qpos','qvel','ee','time','minimum','geometry_step')}
    def velocity(s):
        nonlocal residual,momentum,qerror
        d.qpos[robot.qpos_ids]=robot.B@(parent.q+blend(s)*delta)
        robot.kinematics(d,mass=True);M=robot.mass(d)
        v=np.zeros(m.nv);v[a]=robot.B@(blend(s,1)*delta/duration)
        rhs=M[ba]@v[a];v[b]=-np.linalg.solve(M[bb],rhs)
        residual=max(residual,float(np.max(abs(M[bb]@v[b]+rhs))))
        momentum=max(momentum,float(np.max(abs((M@v)[b]))))
        qerror=max(qerror,abs(float(np.linalg.norm(d.qpos[robot.base_qpos[3:]]))-1))
        return v
    complete=True
    for i in range(intervals+1):
        budget.check();s=i/intervals;v=velocity(s);d.qvel[:]=v;d.time=start_time+s*duration
        current=(d.geom_xpos.copy(),d.geom_xmat.copy().reshape(-1,3,3))
        step=0. if previous is None else float(np.max(np.linalg.norm(current[0]-previous[0],axis=-1)
            +np.linalg.norm(current[1]-previous[1],axis=(-1,-2))*m.geom_rbound))
        previous=current;extent=max(extent,step)
        scan=collision.scan(d,0.) if check_collision else None
        clearance=float(scan['minimum_clearance_m']) if scan else np.nan
        if scan and clearance<minimum:minimum=clearance;pair=scan['minimum_pair']
        for key,value in (('qpos',d.qpos),('qvel',v),('ee',robot.ee(d)),('time',d.time),
                          ('minimum',clearance),('geometry_step',step)):
            records[key].append(np.array(value).copy())
        if scan and clearance<0:
            complete=False;break
        if i<intervals:
            origin=d.qpos.copy();mujoco.mj_integratePos(m,d.qpos,v,duration/(2*intervals))
            vm=velocity((i+.5)/intervals);d.qpos[:]=origin
            mujoco.mj_integratePos(m,d.qpos,vm,duration/intervals)
    arrays={k:np.asarray(v) for k,v in records.items()}
    child=Node(np.asarray(target).copy(),state_array(m,d),d.qpos.copy(),d.qvel.copy(),
        parent.elapsed+duration,-1,duration,robot.ee(d),-1) if complete else None
    report={'complete':complete,'intervals':intervals,'actual_intervals':len(records['time'])-1,
        'minimum_clearance_m':minimum if check_collision else None,'minimum_pair':pair,
        'max_geometry_step_m':extent,'momentum_residual':momentum,'solve_residual':residual,
        'quaternion_error':qerror,'parent_state_hash':digest(parent.state),
        'start_time':start_time,'end_time':float(d.time),'velocity_derived_not_clipped':True,
        'geometry_checked':check_collision,'qualification':'TREE_SCREEN_ONLY',
        'numerical_ok':residual<1e-10 and momentum<1e-10 and qerror<1e-12}
    return child,report,arrays


class Connector:
    def __init__(self,directory,robot,collision,budget):
        self.directory,self.robot,self.collision,self.budget=directory,robot,collision,budget
        self.calls=0
    def edge(self,parent,target,duration):
        if not edge_motion_valid(self.robot,parent.q,target,duration):return None
        for n in P['edge_grids']:
            path=self.directory/'edges'/f'{self.calls:04d}';self.calls+=1
            child,report,arrays=propagate(parent,target,duration,self.robot,self.collision,self.budget,n)
            accepted=bool(child is not None and report['numerical_ok']
                and report['max_geometry_step_m']<=P['edge_max_step_m'])
            report.update(accepted=accepted,target=target,duration=duration,edge_id=self.calls-1)
            atomic_npz(path/'trace.npz',**arrays,parent_state=parent.state,target=target,
                child_state=child.state if child else np.empty(0))
            atomic_json(path/'report.json',report)
            if child is None or not report['numerical_ok']:return None
            if accepted:
                child.edge_id=self.calls-1;return child
        return None


def nearest(nodes,target,scale):
    # State is never merged by q. Each potential parent keeps its own base/time.
    return int(np.argmin([np.linalg.norm((n.q-target)/scale)+.02*n.elapsed for n in nodes]))


def collect(nodes,index):
    result=[]
    while index>=0:
        result.append(nodes[index]);index=nodes[index].parent
    return list(reversed(result))


class Piecewise:
    def __init__(self,nodes,duration):
        self.nodes=nodes;self.duration=float(duration)
        assert nodes[0].elapsed==0. and nodes[-1].elapsed<=duration+1e-9
    def evaluate(self,t,derivative=0):
        t=np.asarray(t);flat=t.ravel();out=[]
        for value in flat:
            found=False
            for a,b in zip(self.nodes[:-1],self.nodes[1:]):
                if a.elapsed<=value<=b.elapsed:
                    s=(value-a.elapsed)/b.duration
                    x=blend(s,derivative)*(b.q-a.q)/b.duration**derivative
                    out.append(a.q+x if derivative==0 else x);found=True;break
            if not found:out.append(self.nodes[-1].q if derivative==0 else np.zeros(17))
        return np.asarray(out).reshape(t.shape+(17,))


def fit323(nodes,robot,duration):
    """Bounded least squares in the existing decoder, fixed endpoint exactly.

    This fit is not qualified and may lose clearance. The original validator
    determines admissibility after conversion; no endpoint relabel or clipping.
    """
    q0=nodes[0].q;endpoint=nodes[-1].q;times=np.linspace(0,duration,P['fit_samples'])
    base=Reference(np.zeros((19,17)),q0,duration).evaluate(times)
    matrix=[]
    for j in range(18):
        z=np.zeros((19,17));z[j,:]=1.
        matrix.append((Reference(z,q0,duration).evaluate(times)-base)[:,0])
    A=np.asarray(matrix).T
    z=np.zeros((19,17));z[-1]=endpoint
    constant=Reference(z,q0,duration).evaluate(times)
    desired=Piecewise(nodes,duration).evaluate(times)
    x=np.empty((19,17));x[-1]=endpoint
    for j in range(17):
        sol=lsq_linear(A,desired[:,j]-constant[:,j],bounds=(robot.lower[j],robot.upper[j]),
            tol=1e-10,max_iter=200)
        if not sol.success:raise ValueError('FIT_SOLVER_FAILED')
        x[:18,j]=sol.x
    fitted=Reference(x,q0,duration).evaluate(times)
    return x,{'maximum_joint_fit_error_rad':float(np.max(abs(fitted-desired))),
        'rms_joint_fit_error_rad':float(np.sqrt(np.mean((fitted-desired)**2))),
        'endpoint_fixed_exactly':np.array_equal(x[-1],endpoint),'qualified':False},times,desired,fitted


def goal_connect(parent,qguess,connector,task,directory):
    """Endpoint correction from this particular floating parent, not root.

    Finite differences do not enter the tree. An accepted final edge must pass
    whole-body geometry after correction. No fixed-base IK answer is accepted.
    """
    robot=connector.robot;budget=connector.budget
    duration=task['condition']['duration']-parent.elapsed;q=np.asarray(qguess).copy()
    history=[];goals=task['condition']['goals']
    if not edge_motion_valid(robot,parent.q,q,duration):return None
    child=connector.edge(parent,q,duration)
    if child is None:return None
    for iteration in range(P['tree_endpoint_iterations']+1):
        e=pose_vector(child.ee,goals);Q=float(max(group_norms(e)))
        if Q<=.7:return child
        if iteration==P['tree_endpoint_iterations'] or not budget.can_fit(26,26*512):break
        center,_,_=propagate(parent,q,duration,robot,connector.collision,budget,256,False)
        e=pose_vector(center.ee,goals)
        columns=[];poses=[];steps=[]
        for j in range(17):
            h=P['tree_endpoint_difference_rad']
            if q[j]+h>robot.upper[j]:h=-h
            trial=q.copy();trial[j]+=h
            node,_,_=propagate(parent,trial,duration,robot,connector.collision,budget,256,False)
            columns.append((pose_vector(node.ee,goals)-e)/h);poses.append(node.ee);steps.append(h)
        J=np.asarray(columns).T
        delta=-np.linalg.solve(J.T@J+1e-3*np.eye(17),J.T@e)
        delta*=min(1.,P['tree_endpoint_step_rad']/max(abs(delta).max(),1e-12))
        trials=[];accepted=False
        for scale in (1.,.5,.25):
            proposed=q+scale*delta
            other=connector.edge(parent,proposed,duration)
            row={'scale':scale,'edge_pass':other is not None}
            if other is not None:
                other_e=pose_vector(other.ee,goals);row['ee']=other.ee
                row['accepted']=bool(other_e@other_e<e@e)
                if row['accepted']:q=proposed;child=other;accepted=True
            trials.append(row)
            if accepted:break
        atomic_npz(directory/f'iteration_{iteration:02d}.npz',goal=e,jacobian=J,
            perturb_ee=np.asarray(poses),epsilon=np.asarray(steps),delta=delta)
        history.append({'Q':Q,'accepted':accepted,'trials':trials})
        atomic_json(directory/'iterations.json',history)
        if not accepted:break
    return None


def search(directory,proposal,seed,task,robot,initial,collision,budget,progress):
    root=root_node(robot,initial);nodes=[root];rng=np.random.default_rng(seed+440001)
    qgoal=proposal[-1].copy();connector=Connector(directory,robot,collision,budget)
    weights=np.r_[np.full(10,.10),np.full(7,.65)]
    reference=Reference(proposal,root.q,task['condition']['duration'])
    attempts=[];solution=None;best=0;stop='ATTEMPT_LIMIT'
    try:
        for attempt in range(P['search_attempt_cap']):
            progress('forward_floating_tree',candidate=directory.parent.name,attempt=attempt,nodes=len(nodes))
            if not budget.can_fit(6,4096):stop='SEARCH_BUDGET_RESERVED';break
            r=rng.random()
            if r<P['tree_goal_probability']:target=qgoal.copy();sampling='GOAL'
            elif r<P['tree_goal_probability']+P['tree_proposal_probability']:
                target=reference.evaluate(rng.uniform(0,reference.duration));sampling='PROPOSAL'
                target+=rng.normal(0,.15,17)*weights
            elif r<1-P['tree_uniform_probability']:
                s=rng.random();target=(1-s)*root.q+s*qgoal+rng.normal(0,1,17)*weights;sampling='LOCAL_EXPLORATION'
            else:target=rng.uniform(robot.lower,robot.upper);sampling='GLOBAL_DOMAIN'
            # Projection applies only to a newly drawn planning proposal.
            target=np.minimum(np.maximum(target,robot.lower),robot.upper)
            index=nearest(nodes,target,weights);parent=nodes[index];delta=target-parent.q
            delta*=min(1.,P['edge_max_joint_step_rad']/max(abs(delta).max(),1e-12))
            target=parent.q+delta
            duration=minimum_duration(delta,robot.spec.planner_velocity_limits,robot.spec.planner_acceleration_limits)
            row={'attempt':attempt,'parent':index,'sampling':sampling,'target':target,'duration':duration,'accepted':False}
            if parent.elapsed+duration>task['condition']['duration']-P['tree_goal_reserve_s']:
                row['reason']='TIME_RESERVE';attempts.append(row);continue
            child=connector.edge(parent,target,duration)
            if child is not None:
                child.parent=index;nodes.append(child);idx=len(nodes)-1;row.update(accepted=True,node=idx,edge=child.edge_id)
                if np.linalg.norm((child.q-qgoal)/weights)<np.linalg.norm((nodes[best].q-qgoal)/weights):best=idx
                if (len(nodes)%P['tree_goal_connection_period']==0 or np.max(abs(child.q-qgoal))<.12) and budget.can_fit(130,130*512):
                    # Start endpoint IK at the collision-free explored node,
                    # not at a potentially colliding nominal terminal posture.
                    final=goal_connect(child,child.q,connector,task,directory/'goal_connections'/f'{idx:04d}')
                    if final is not None:
                        final.parent=idx;nodes.append(final);solution=len(nodes)-1;stop='GOAL_CONNECTED_SCREEN_ONLY'
            attempts.append(row)
            if solution is not None:break
            if len(nodes)>=P['accepted_node_cap']:stop='NODE_LIMIT';break
    except BudgetExhausted as error:stop=str(error)
    atomic_json(directory/'attempts.json',attempts)
    atomic_npz(directory/'nodes.npz',q=np.asarray([n.q for n in nodes]),
        qpos=np.asarray([n.qpos for n in nodes]),qvel=np.asarray([n.qvel for n in nodes]),
        state=np.asarray([n.state for n in nodes]),ee=np.asarray([n.ee for n in nodes]),
        elapsed=np.asarray([n.elapsed for n in nodes]),duration=np.asarray([n.duration for n in nodes]),
        parent=np.asarray([n.parent for n in nodes]),edge_id=np.asarray([n.edge_id for n in nodes]))
    result={'stop':stop,'nodes':len(nodes),'attempts':len(attempts),'propagations':connector.calls,
        'solution_node':solution,'best_node':best,'seed':seed+440001,'qualification':'TREE_SCREEN_ONLY',
        'complete_parent_state_carried':True,'same_q_nodes_not_merged':True,'private_solution_used':False}
    atomic_json(directory/'result.json',result)
    return collect(nodes,solution) if solution is not None else None,result
