"""Independent closed-form saved-data audit, executed within audit_saved.py.

No project import, solver, robot integration or distance query. For future j,
let m=ceil(|u|/A), k=min(j,m), A=.02*a. The signed cumulative reference
increment is sign(u)*([.009*k+.011*(k-1)]*|u|-.02*A*k*(k-1)/2
                        +.011*max(|u|-k*A,0)).
This finite arithmetic sum is evaluated directly, unlike production cumsum.
"""
brake_stats={'ticks':0,'coordinates':0,'tightened_ticks':0,'tightened_lower_coordinates':0,
    'tightened_upper_coordinates':0,'actual_command_on_tightened_bound_coordinates':0,
    'maximum_bound_reconstruction_error':0.,'endpoint_records':0,
    'future_coordinates_checked_per_candidate':0,'first_tightened_time':None,'last_tightened_time':None}
brake_future_min=float('inf')
brake_effect_rows=[]
brake_effect_lower=np.zeros(17,dtype=int);brake_effect_upper=np.zeros(17,dtype=int)
brake_effect_command_margin=np.full(17,np.inf);brake_effect_future_margin=np.full(17,np.inf)
brake_effect_original_margin=np.full(17,np.inf)
def envelope_formula(ref,prev,u,acc,N):
    speed=np.abs(u); A=.02*acc; m=np.ceil(speed/A)
    j=np.arange(N+1)[:,None]; k=np.minimum(j,m)
    offset=np.sign(u)*((.009*k+.011*(k-1))*speed-.02*A*k*(k-1)/2+.011*np.maximum(speed-k*A,0))
    pos=ref+.009*prev+.011*u+offset
    nextv=np.sign(u)*np.maximum(speed-(j+1)*A,0)
    barrier=pos+nextv/4
    return pos,barrier

def check_brake(ti,f,q,ref,prev,original_lo,original_hi):
    global brake_future_min
    d=f['braking_bounds'];acc=.6*alim;vmax=.7*vlim;N=int(np.ceil(np.max(vmax/(acc*.02))))+1
    label='brake:'+str(ti)+':'
    def eq(name,x,y,tol=2e-12):close(label+name,x,y,tol)
    for name,x in [('q',q),('reference',ref),('previous',prev),('lower',L),('upper',U),('vmax',vmax),('acceleration',acc),('tracking_error',q-ref),('original_lower',original_lo),('original_upper',original_hi),('robust_future_barrier_lower',L+.06),('robust_future_barrier_upper',U-.06)]:eq(name,d[name],x)
    ck(label+'constants',d['tick_s']==.02 and d['substep_s']==.002 and d['substeps']==10 and abs(d['alpha']-.009)<1e-16 and d['beta']==.011 and d['barrier_gain']==4 and d['domain_inset_rad']==.04 and d['tracking_envelope_rad']==.02 and d['bisection_steps']==64 and d['future_ticks_checked']==N+1)
    ck(label+'scope',not d['geometry_torque_task_feasibility_guaranteed'] and not d['original_constraints_relaxed'] and not d['physical_state_modified'] and d['new_physics_steps']==d['new_geometry_queries']==0)
    k=np.arange(1,11)[:,None];weights=.002*k*(k+1)/20;offsets=ref+.002*(k-k*(k+1)/20)*prev
    comps={'velocity_lower':-vmax,'velocity_upper':vmax,'measured_barrier_lower':4*(L+.04-q),'measured_barrier_upper':4*(U-.04-q),'acceleration_lower':prev-.02*acc,'acceleration_upper':prev+.02*acc,'reference_substep_lower':(L+.04-offsets)/weights,'reference_substep_upper':(U-.04-offsets)/weights}
    for name,x in comps.items():eq('original_component_'+name,d['original_components'][name],x)
    # Compute boundaries afresh over the full [-vmax,vmax] domain.
    _,bl=endlo=envelope_formula(ref,prev,-vmax,acc,N);_,bh=endhi=envelope_formula(ref,prev,vmax,acc,N)
    left=-vmax.copy();right=vmax.copy()
    for _ in range(64):
        mid=(left+right)/2;_,b=envelope_formula(ref,prev,mid,acc,N)
        ok=b.min(0)>=L+.06;right=np.where(ok,mid,right);left=np.where(ok,left,mid)
    blo=np.where(bl.min(0)>=L+.06,-vmax,np.where(bh.min(0)<L+.06,vmax+1,right))
    left=-vmax.copy();right=vmax.copy()
    for _ in range(64):
        mid=(left+right)/2;_,b=envelope_formula(ref,prev,mid,acc,N)
        ok=b.max(0)<=U-.06;left=np.where(ok,mid,left);right=np.where(ok,right,mid)
    bhi=np.where(bh.max(0)<=U-.06,vmax,np.where(bl.max(0)>U-.06,-vmax-1,left))
    eq('independent_braking_lower',d['braking_lower'],blo);eq('independent_braking_upper',d['braking_upper'],bhi)
    brake_stats['maximum_bound_reconstruction_error']=max(brake_stats['maximum_bound_reconstruction_error'],float(np.max(abs(blo-d['braking_lower']))),float(np.max(abs(bhi-d['braking_upper']))))
    tracking=np.abs(q-ref)<=.02;lo=np.maximum(original_lo,blo);hi=np.minimum(original_hi,bhi);lo=np.where(tracking,lo,np.maximum(lo,hi+1))
    eq('final_lower',d['final_lower'],lo);eq('final_upper',d['final_upper'],hi)
    eq('command_lower',f['lower_bounds'],lo);eq('command_upper',f['upper_bounds'],hi)
    ck(label+'tracking_prerequisite',np.array_equal(d['current_tracking_envelope_satisfied'],tracking) and tracking.all())
    ck(label+'empty_indices',np.array_equal(d['empty_intersection_indices'],np.flatnonzero(lo>hi)) and np.all(lo<=hi))
    tl=np.asarray(d['braking_lower'])>original_lo;th=np.asarray(d['braking_upper'])<original_hi
    ck(label+'tightened_flags',np.array_equal(d['tightened_lower'],tl) and np.array_equal(d['tightened_upper'],th))
    for name,u in [('braking_lower',blo),('braking_upper',bhi)]:
        e=d['endpoint_details'][name];u=np.clip(u,-vmax,vmax);pos,b=envelope_formula(ref,prev,u,acc,N)
        eq(name+'_candidate',e['candidate'],blo if name=='braking_lower' else bhi)
        ck(label+name+'_search_range',np.array_equal(e['within_search_range'],np.abs(e['candidate'])<=vmax))
        eq(name+'_minimum_barrier',e['minimum_barrier_coordinate'],b.min(0));eq(name+'_maximum_barrier',e['maximum_barrier_coordinate'],b.max(0))
        lower_j=np.array(e['lower_limiting_future_tick']);upper_j=np.array(e['upper_limiting_future_tick'])
        eq(name+'_lower_limiting_tick_witness',b[lower_j,np.arange(17)],b.min(0));eq(name+'_upper_limiting_tick_witness',b[upper_j,np.arange(17)],b.max(0))
        eq(name+'_stop_reference',e['stop_reference'],pos[-1]);eq(name+'_stopping_ticks',e['stopping_ticks'],np.ceil(abs(u)/(.02*acc)),0)
        eq(name+'_braking_displacement',e['braking_displacement'],pos[-1]-(ref+.009*prev+.011*u))
        brake_stats['endpoint_records']+=1
    u=np.array(f['filtered']);pos,b=envelope_formula(ref,prev,u,acc,N)
    slack=min(float(np.min(b-(L+.06))),float(np.min((U-.06)-b)));brake_future_min=min(brake_future_min,slack)
    ck(label+'chosen_command_full_future_braking',slack>=-2e-12 and np.all(pos>=L+.04-2e-12) and np.all(pos<=U-.04+2e-12),{'minimum_barrier_slack':slack})
    ck(label+'chosen_command_all_original_and_new_bounds',np.all(u>=lo-2e-12) and np.all(u<=hi+2e-12) and np.all(u>=original_lo-2e-12) and np.all(u<=original_hi+2e-12))
    brake_effect_lower[:]+=tl;brake_effect_upper[:]+=th
    command_margin=np.minimum(u-np.array(f['lower_bounds']),np.array(f['upper_bounds'])-u)
    original_margin=np.minimum(u-original_lo,original_hi-u)
    future_margin=np.minimum(np.min(b-(L+.06),axis=0),np.min((U-.06)-b,axis=0))
    brake_effect_command_margin[:]=np.minimum(brake_effect_command_margin,command_margin)
    brake_effect_original_margin[:]=np.minimum(brake_effect_original_margin,original_margin)
    brake_effect_future_margin[:]=np.minimum(brake_effect_future_margin,future_margin)
    row={'tick':ti,'absolute_time':f['time'],'relative_time':f['time']-n['time'][0],'reason':f['reason'],'tightened_dimensions':int(np.sum(tl|th)),'minimum_command_margin':float(command_margin.min()),'minimum_future_braking_margin':float(future_margin.min())}
    for j in range(17):
        for key,value in [('q',q[j]),('reference',ref[j]),('previous_command',prev[j]),('command',u[j]),('original_lower',original_lo[j]),('original_upper',original_hi[j]),('braking_lower',blo[j]),('braking_upper',bhi[j]),('final_lower',f['lower_bounds'][j]),('final_upper',f['upper_bounds'][j]),('tightened_lower',int(tl[j])),('tightened_upper',int(th[j])),('command_margin',command_margin[j]),('future_braking_margin',future_margin[j])]:row[str(j)+'_'+key]=value
    brake_effect_rows.append(row)
    brake_stats['ticks']+=1;brake_stats['coordinates']+=17;brake_stats['future_coordinates_checked_per_candidate']+=(N+1)*17
    brake_stats['tightened_lower_coordinates']+=int(tl.sum());brake_stats['tightened_upper_coordinates']+=int(th.sum())
    brake_stats['actual_command_on_tightened_bound_coordinates']+=int(np.sum((tl & (abs(u-np.array(f['lower_bounds']))<=1e-9)) | (th & (abs(u-np.array(f['upper_bounds']))<=1e-9))))
    if np.any(tl|th):
        brake_stats['tightened_ticks']+=1
        if brake_stats['first_tightened_time'] is None:brake_stats['first_tightened_time']=f['time']
        brake_stats['last_tightened_time']=f['time']
    # Use exact recorded final bounds downstream, after the independent check.
    return np.array(f['lower_bounds']),np.array(f['upper_bounds'])
