"""Same feedback inequalities with complete saved selection and command evidence."""
import numpy as np
import mujoco
from scipy.optimize import minimize
from .braking import braking_velocity_bounds
from ff_prior_pilot_train.storage import atomic_json,atomic_npz


class ObservedWholeBodyFilter:
    def clearance(self,g=None,drift=None,activation=.08):
        model,data=self.model,self.data
        a,b=self.pair_a,self.pair_b
        bounds=np.linalg.norm(data.geom_xpos[a]-data.geom_xpos[b],axis=1)
        bounds-=model.geom_rbound[a]+model.geom_rbound[b]
        selected=np.flatnonzero(bounds<=activation+.01)
        distances=bounds.copy();exact=np.zeros(len(a),dtype=bool)
        # 0=sphere-screened lower bound;1=queried outside activation;2=selected row;
        # 3=invalid witness caused stop. Every pair retains identity in pair catalog.
        disposition=np.zeros(len(a),dtype=np.uint8)
        rows,lower,drift_rows,margins,row_pairs,witnesses=[],[],[],[],[],[]
        observation=dict(time=float(data.time),bounds=bounds,distances=distances,exact=exact,
            disposition=disposition,activation=float(activation),row_pairs=row_pairs,
            rows=rows,lower=lower,drift=drift_rows,margins=margins,witnesses=witnesses)
        self.clearance_observations.append(observation)
        self.latest_constraint_rows=[]
        witness=np.zeros(6)
        for k in selected:
            pair=self.pairs[k]
            distance=float(mujoco.mj_geomDistance(model,data,pair.geom_a,pair.geom_b,activation+.01,witness))
            distances[k]=distance;exact[k]=True;disposition[k]=1
            if g is None or distance>activation:continue
            pa,pb=witness[:3].copy(),witness[3:].copy()
            if model.geom_type[pair.geom_a]>model.geom_type[pair.geom_b]:pa,pb=pb,pa
            normal=pa-pb;norm=np.linalg.norm(normal)
            if norm<1e-10 or distance<=0:
                disposition[k]=3
                observation['error']='INVALID_OR_PENETRATING_WITNESS'
                raise RuntimeError(f'Invalid/penetrating clearance witness: {pair}')
            normal/=norm;jac=[]
            for geom,point in ((pair.geom_a,pa),(pair.geom_b,pb)):
                jp=np.zeros((3,model.nv))
                mujoco.mj_jac(model,data,jp,None,point,int(model.geom_bodyid[geom]))
                jac.append(jp)
            j=normal@(jac[0]-jac[1]);row=j@g;offset=float(j@drift)
            rhs=-8.*(distance-self.safe_distances[k])-offset
            rows.append(row);lower.append(rhs);drift_rows.append(offset)
            margins.append(float(distance-self.safe_distances[k]));row_pairs.append(int(k))
            witnesses.append(np.r_[pa,pb]);disposition[k]=2
            self.latest_constraint_rows.append(dict(pair_index=int(k),geom_a=int(pair.geom_a),geom_b=int(pair.geom_b),
                geom_a_name=pair.geom_a_name,geom_b_name=pair.geom_b_name,
                distance_m=distance,margin_m=margins[-1],jacobian17=row.copy(),lower_rhs=float(rhs+.008),
                witness_a=pa,witness_b=pb,selected=True,reason='EXACT_DISTANCE_LE_ACTIVATION_80MM'))
        k=int(np.argmin(distances-self.safe_distances));margin=float(distances[k]-self.safe_distances[k])
        self.last_clearance=float(np.min(distances));self.last_margin=margin
        self.minimum_clearance=min(self.minimum_clearance,self.last_clearance)
        if margin<self.minimum_margin:
            self.minimum_margin=margin
            self.minimum_pair={'class':self.pairs[k].pair_class,'a':self.pairs[k].geom_a_name,
                               'b':self.pairs[k].geom_b_name,'time':float(data.time)}
        if g is not None:
            self.collision_linearization=dict(jacobian=np.asarray(rows).reshape(-1,17),
                drift=np.asarray(drift_rows),margin=np.asarray(margins))
        return np.asarray(rows).reshape(-1,17),np.asarray(lower),margin

    def velocity_filter(self,desired,g,drift,hessian=None):
        desired=np.asarray(desired).copy();filtered=None
        self.latest_constraint_rows=[]
        event=dict(time=float(self.data.time),desired=desired.copy(),filtered=None,delta=None,
            reason='FILTER_NOT_COMPLETED',slsqp_used=False,solver_success=None)
        try:
            q,_=self.state()
            lo,hi,braking=braking_velocity_bounds(q,self.reference_q,self.command,self.robot.lower,self.robot.upper,
                .70*self.spec.planner_velocity_limits,.60*self.spec.planner_acceleration_limits)
            event['braking_bounds']=braking
            event.update(lower_bounds=lo.copy(),upper_bounds=hi.copy(),previous_command=self.command.copy())
            a,b,margin=self.clearance(g,drift);b=b+.008
            if np.any(lo>hi) or margin<-.001:raise RuntimeError('NO_FEASIBLE_V7_DOMAIN_VELOCITY')
            x=np.clip(desired,lo,hi);boxed=x.copy()
            active=bool(len(b) and np.min(a@x-b)<-1e-8)
            changed_box=bool(np.max(abs(x-desired))>1e-10)
            event.update(boxed=boxed,geometry_projection_required=active,box_projection_required=changed_box)
            if active or (hessian is not None and changed_box):
                h=np.eye(17) if hessian is None else hessian/max(np.max(np.diag(hessian)),1e-8)
                result=minimize(lambda z:.5*(z-desired)@h@(z-desired),x,
                    jac=lambda z:h@(z-desired),bounds=list(zip(lo,hi)),method='SLSQP',
                    constraints=({'type':'ineq','fun':lambda z:a@z-b,'jac':lambda z:a},) if len(b) else (),
                    options={'ftol':1e-12,'maxiter':200})
                x=result.x
                event.update(slsqp_used=True,solver_success=bool(result.success),solver_message=str(result.message),
                    solver_iterations=int(result.nit),solver_candidate=x.copy())
                if not result.success or (len(b) and np.min(a@x-b)<-1e-6):
                    self.qp_failures+=1
                    raise RuntimeError(f'No feasible velocity: {result.message}')
            if not np.isfinite(x).all() or np.any(x<lo-1e-12) or np.any(x>hi+1e-12):
                raise RuntimeError('QP_DOMAIN_BOUND_VIOLATION')
            self.last_filter_delta=float(np.linalg.norm(x-desired));filtered=x.copy()
            event.update(filtered=filtered,delta=self.last_filter_delta,
                reason='GEOMETRY_AND_BOX_QP' if active else ('BOX_TASK_QP' if event['slsqp_used'] else 'DIRECT_OR_BOX_FEASIBLE'))
            for row in self.latest_constraint_rows:
                row['unfiltered_slack']=float(row['jacobian17']@desired-row['lower_rhs'])
                row['filtered_slack']=float(row['jacobian17']@filtered-row['lower_rhs'])
                row['binding_after_filter']=abs(row['filtered_slack'])<=1e-6
            return filtered
        except Exception as error:
            event.update(reason='FILTER_FAILED',error=repr(error));raise
        finally:
            event['selected_constraint_count']=len(self.latest_constraint_rows)
            self.filter_history.append(event)
            # Pure observation at this pre-native control tick, including failed filters.
            try:
                self.body_recorder.sample(self.data,command_before=desired,command_after=filtered,
                    active_constraints=self.latest_constraint_rows,filter_reason=event['reason'],
                    native_index=len(self.controls))
            except Exception as error:
                # A diagnostic failure cannot replace the controller exception
                # or retroactively change a saved physical outcome.
                self.body_diagnostic_failures.append(dict(time=float(self.data.time),
                    stage='pre_control',reason=repr(error),physical_outcome_unchanged=True))
                event['body_diagnostic_error']=repr(error)

    def save_constraint_observations(self,directory):
        observations=self.clearance_observations;count=len(self.pairs)
        def stacked(key,dtype=float):
            return np.asarray([row[key] for row in observations],dtype=dtype).reshape(len(observations),count)
        offsets=np.r_[0,np.cumsum([len(row['row_pairs']) for row in observations])]
        def flat(key,width=None,dtype=float):
            values=[value for row in observations for value in row[key]]
            result=np.asarray(values,dtype=dtype)
            return result if width is None else result.reshape(-1,width)
        atomic_npz(directory/'controller_clearance.npz',time=[row['time'] for row in observations],
            sphere_lower_bound_m=stacked('bounds'),distance_or_lower_bound_m=stacked('distances'),
            exact_query_mask=stacked('exact',bool),disposition=stacked('disposition',np.uint8),
            row_offsets=offsets,row_pair_index=flat('row_pairs',dtype=int),jacobian17=flat('rows',17),
            lower_without_ramp_buffer=flat('lower'),drift=flat('drift'),margin_m=flat('margins'),
            witnesses=flat('witnesses',6))
        atomic_json(directory/'controller_clearance_metadata.json',dict(
            pairs=[dict(index=i,geom_a=p.geom_a,geom_b=p.geom_b,geom_a_name=p.geom_a_name,
                        geom_b_name=p.geom_b_name) for i,p in enumerate(self.pairs)],
            disposition={'0':'SPHERE_LOWER_BOUND_GT_90MM','1':'EXACT_QUERY_OUTSIDE_80MM_ACTIVATION',
                         '2':'EXACT_GEOMETRY_JACOBIAN_ROW','3':'INVALID_WITNESS_STOP'},
            queried_distance_cap_m=.09,controller_activation_m=.08,planner_distance_buffer_m=.001,
            lower_rhs_ramp_buffer=.008,acceptance_clearance_m=0.,new_coupled_calls=0,new_native_steps=0,
            note='Far sphere distances are lower bounds; queried 90mm values may be capped. No exact-distance claim for omitted far pairs.'))
