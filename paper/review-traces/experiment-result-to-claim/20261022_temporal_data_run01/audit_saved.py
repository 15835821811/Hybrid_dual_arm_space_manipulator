"""Independent saved-evidence audit. No project imports or simulator calls.

Only pathlib/hash/json/zip/AST/XML, numpy and scipy interpolation/rotation algebra.
Old executions are hashed only, never loaded as physical arrays or re-evaluated.
"""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import ast
import hashlib
import json
from pathlib import Path
import time
import zipfile
import xml.etree.ElementTree as ET
import numpy as np
from scipy.interpolate import BSpline, PPoly
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[4]
TRACE = Path(__file__).resolve().parent
RUN = ROOT/'artifacts/v7_1/temporal_data_001'
INPUTS = {}
CHECKS = []
RAW = {'review_role': 'independent paper architect/reviewer', 'units': {}}

def normpath(p):
    p = Path(p)
    return p if p.is_absolute() else ROOT/p

def sha(p):
    p = normpath(p)
    h = hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    key = p.relative_to(ROOT).as_posix()
    INPUTS[key] = {'sha256': h.hexdigest(), 'size_bytes': p.stat().st_size,
                   'mtime_ns': p.stat().st_mtime_ns}
    return h.hexdigest()

def read(p):
    p = normpath(p); sha(p)
    return json.loads(p.read_text(encoding='utf-8-sig'))

def load(p):
    p = normpath(p); sha(p)
    with np.load(p, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}

def digest(x):
    return hashlib.sha256(json.dumps(x, sort_keys=True, separators=(',', ':'),
        default=lambda a: a.tolist(), allow_nan=False).encode()).hexdigest()

def check(name, passed, detail=None):
    CHECKS.append(dict(name=name, passed=bool(passed), detail=detail))
    if not passed: print('FAIL', name, detail, flush=True)

def close(a, b, tol=1e-10):
    return np.shape(a) == np.shape(b) and bool(np.max(np.abs(np.asarray(a)-np.asarray(b)), initial=0) <= tol)

def err(a,b):
    return float(np.max(np.abs(np.asarray(a)-np.asarray(b)),initial=0))

def closure():
    todo = list((ROOT/'ff_prior_temporal_data').glob('*.py')); seen = set()
    def include(name):
        parts = name.split('.')
        for i in range(1,len(parts)+1):
            path = ROOT.joinpath(*parts[:i])
            for candidate in (path.with_suffix('.py'),path/'__init__.py'):
                if candidate.is_file() and candidate not in seen:todo.append(candidate)
    while todo:
        path=todo.pop()
        if path in seen:continue
        seen.add(path); pkg=path.relative_to(ROOT).with_suffix('').parts[:-1]
        for node in ast.walk(ast.parse(path.read_text(encoding='utf-8-sig'))):
            if isinstance(node,ast.Import):
                for alias in node.names:include(alias.name)
            elif isinstance(node,ast.ImportFrom):
                prefix=list(pkg[:len(pkg)-node.level+1]) if node.level else []
                module='.'.join(prefix+([node.module] if node.module else []));include(module)
                for alias in node.names:
                    if alias.name!='*':include(module+'.'+alias.name)
    seen.update([ROOT/'ff_prior_temporal_data/job.ps1',ROOT/'ff_prior_temporal_data/docs/EXPERIMENT_PLAN.md'])
    return {p.relative_to(ROOT).as_posix() for p in seen}

def polynomial_extrema(s):
    result=[]
    for order in range(3):
        low=[];high=[]
        for j in range(17):
            p=PPoly.from_spline((s.t,s.c[:,j],3)).derivative(order)
            vals=[]
            for k,width in enumerate(np.diff(p.x)):
                if width<=0:continue
                roots=np.roots(np.polyder(p.c[:,k]))
                points=[0.,width]+[z.real for z in roots if abs(z.imag)<1e-10 and 0<z.real<width]
                vals.extend(np.polyval(p.c[:,k],points))
            low.append(min(vals));high.append(max(vals))
        result.append([low,high])
    return np.asarray(result)

def quat_rotation(q):
    return Rotation.from_quat(q[:,[1,2,3,0]]).as_matrix()

URDF=ROOT/'dual_arm_space_robot_2026/urdf/dual_arm_space_robot_2026.urdf'
sha(URDF)
xml=ET.parse(URDF).getroot()
JOINTS=xml.findall('joint')
LINKS={l.attrib['name']:l for l in xml.findall('link')}

def vec(e, key, default):
    return np.fromstring(e.attrib.get(key,default) if e is not None else default,sep=' ')

def transforms(qpos, identity):
    """URDF forward transforms from saved actual qpos, with no MuJoCo import."""
    n=len(qpos); pos={'world':np.zeros((n,3))}; rot={'world':np.broadcast_to(np.eye(3),(n,3,3))}
    for j in JOINTS:
        name=j.attrib['name']; typ=j.attrib['type']; parent=j.find('parent').attrib['link']; child=j.find('child').attrib['link']
        if typ=='floating':
            start=0 if name=='world_joint' else 74
            pos[child]=qpos[:,start:start+3];rot[child]=quat_rotation(qpos[:,start+3:start+7]);continue
        origin=j.find('origin'); off=vec(origin,'xyz','0 0 0'); ori=Rotation.from_euler('xyz',vec(origin,'rpy','0 0 0')).as_matrix()
        pos[child]=pos[parent]+np.einsum('nij,j->ni',rot[parent],off)
        r=rot[parent]@ori
        if typ in ('revolute','continuous'):
            axis=vec(j.find('axis'),'xyz','1 0 0'); index=identity['robot']['joint_index_map'][name]['qpos']
            r=r@Rotation.from_rotvec(qpos[:,index,None]*axis).as_matrix()
        elif typ!='fixed':raise ValueError(typ)
        rot[child]=r
    return pos,rot

def mesh_radii():
    """Conservative collision support radii around URDF link origins.

    STL vertex norm bounds include every vertex and hence its convex hull.
    This is algebra over frozen mesh bytes, not a new distance query.
    """
    result={}
    dtype=np.dtype([('normal','<f4',(3,)),('vertex','<f4',(3,3)),('attr','<u2')])
    cache={}
    for name,l in LINKS.items():
        radius=0.
        for c in l.findall('collision'):
            origin=c.find('origin'); xyz=vec(origin,'xyz','0 0 0')
            geom=c.find('geometry');mesh=geom.find('mesh')
            if mesh is not None:
                file=ROOT/'dual_arm_space_robot_2026/meshes'/Path(mesh.attrib['filename']).name
                scale=vec(mesh,'scale','1 1 1')
                if str(file) not in cache:
                    sha(file)
                    with file.open('rb') as f:
                        f.seek(80);count=int.from_bytes(f.read(4),'little')
                    data=np.memmap(file,dtype=dtype,mode='r',offset=84,shape=(count,))
                    cache[str(file)]=float(np.sqrt(np.sum(np.asarray(data['vertex'],dtype=float)**2,axis=2)).max())
                    del data
                bound=cache[str(file)]*max(abs(scale))
            elif geom.find('sphere') is not None:bound=float(geom.find('sphere').attrib['radius'])
            elif geom.find('box') is not None:bound=float(np.linalg.norm(vec(geom.find('box'),'size','0 0 0'))/2)
            elif geom.find('cylinder') is not None:
                e=geom.find('cylinder');bound=float(np.hypot(float(e.attrib['radius']),float(e.attrib['length'])/2))
            else:raise ValueError(ET.tostring(geom))
            radius=max(radius,float(np.linalg.norm(xyz))+bound)
        if radius:result[name]=radius
    return result

def desired_algebra(h,ident,source,profile,pos,rot):
    t=h['time'];w=ident['waypoints'];p=np.asarray(w['waypoint_points_m']);dur=np.asarray(w['segment_durations_s'])
    times=np.r_[0.,4.5,4.5+np.cumsum(dur)];points=np.vstack([w['initial_position_w'],p])
    index=np.minimum(np.searchsorted(times,t,side='right')-1,len(times)-2)
    tau=np.clip((t-times[index])/(times[index+1]-times[index]),0.,1.)
    alpha=tau**3*(10-15*tau+6*tau**2)
    pc=points[index]+alpha[:,None]*(points[index+1]-points[index])
    target_r=rot['target_satellite'];target_p=pos['target_satellite']
    rr0=rot['end_effector_r'][0];pr0=pos['end_effector_r'][0]
    local_r=target_r[0].T@rr0
    target=target_p+np.einsum('nij,j->ni',target_r,[0.,-.205,0.])
    s=np.clip(t/4.5,0.,1.);a=s**3*(10-15*s+6*s*s);bend=64*s**3*(1-s)**3
    pr=pr0+a[:,None]*(target-pr0)+bend[:,None]*np.asarray(profile['approach_offset'])
    rv=Rotation.from_matrix((target_r@local_r)@rr0.T).as_rotvec()
    rr=Rotation.from_rotvec(a[:,None]*rv).as_matrix()@rr0
    result=np.broadcast_to(np.eye(4),(len(t),2,4,4)).copy()
    result[:,0,:3,3]=pc;result[:,0,:3,:3]=w['target_rotation_world']
    result[:,1,:3,3]=pr;result[:,1,:3,:3]=rr
    return result

def main():
    run=read(RUN/'run.json');ident=run['identity'];summary=read(RUN/'summary.json');budget=read(RUN/'budget.json')
    protocol=ident['protocol'];selection=read('ff_prior_temporal_data/evidence/selection.json');binding=read('ff_prior_temporal_data/evidence/binding.json')
    check('producer_head',run['producer_head']=='292021c273c2db591677f30e64a6fa6c99bcb0e8')
    check('frozen_selection_exact',selection==ident['selection']);check('frozen_binding_exact',binding==ident['binding'])
    check('source_closure_exact',closure()==set(ident['source_hashes']),{'count':len(ident['source_hashes'])})
    check('source_count_170',len(ident['source_hashes'])==170)
    for path,h in ident['source_hashes'].items():check('source_current:'+path,sha(path)==h)
    sha(RUN/'executed_source.zip')
    with zipfile.ZipFile(RUN/'executed_source.zip') as z:
        check('zip_inventory_exact',len(z.namelist())==170 and set(z.namelist())==set(ident['source_hashes']))
        for p,h in ident['source_hashes'].items():check('source_zip:'+p,hashlib.sha256(z.read(p)).hexdigest()==h)
    check('binding_count_214',len(binding['files'])==214)
    for p,h in binding['files'].items():check('binding:'+p,sha(p)==h)
    oldfreeze=read('artifacts/v7_1/waypoint_moving_gate_001/freeze.json')
    check('original_inventory_preserved',selection['inventory']==oldfreeze['selected'])
    expected=[]
    for split,n in [('train',8),('validation',4)]:
        expected.extend(selection['inventory'][split][:n])
    check('prefix_selection',expected==[x['source'] for x in selection['rows']])
    check('denominator_selection',len(selection['rows'])==12 and selection['denominator']==12)
    allrows=sum(selection['inventory'].values(),[])
    check('inventory_40_unique',len(allrows)==40 and len({r['seed'] for r in allrows})==40 and len({r['trace_sha256'] for r in allrows})==40)
    check('no_dev_test_nominal_seed',all(r['seed']!=20261012 and r['split'] in ('train','validation') and r['target_motion']=='moving' for r in allrows))
    check('frozen_train_seeds',[r['source']['seed'] for r in selection['rows'][:8]]==[31000000,31104729,31209458,31314187,31418916,31523645,31628374,31733103])
    check('frozen_val_seeds',[r['source']['seed'] for r in selection['rows'][8:]]==[32570935,32675664,99378320,99483049])
    inventory=read(RUN/'inventory.json')
    check('unselected_28_not_upgraded',inventory['unselected_count']==28 and inventory['unselected_current_grade']=='LEGACY_NOT_REQUALIFIED' and inventory['source']==selection['inventory'])
    profile=read('tracking_diffusion/config/local10mm_v1.json');radii=mesh_radii()
    for item in selection['rows']:
        unit=item['id'];d=RUN/unit;start=len(CHECKS);result=read(d/'result.json');i=read(d/'identity.json');rep=read(d/'representation.json');lim={k:np.array(v) for k,v in read(d/'limits.json').items()}
        ref=load(d/'reference.npz');initial=load(d/'initial.npz')['values'];source=read(Path(item['source']['path'])/'result.json');original=load(Path(item['source']['path'])/'trace.npz')
        check(unit+':task_identity',i['task']==source['task'] and i['task_hash']==digest(source['task']) and i['source']==item['source'])
        check(unit+':initial_full_shape',initial.shape==(836,) and np.isfinite(initial).all())
        check(unit+':initial_hash',i['initial_state_id']==digest({'values':initial,'model_hash':i['robot']['scene_model_hash'],'state_mask':8191}))
        check(unit+':original_qpos_qvel',np.array_equal(initial[1:82],original['qpos'][0]) and np.array_equal(initial[82:161],original['qvel'][0]))
        check(unit+':source_samples_exact',all(np.array_equal(ref['source_'+k],original[k]) for k in ['time','q','dq']))
        check(unit+':reference_shape',ref['coefficients'].shape==(273,17) and ref['knots'].shape==(277,))
        check(unit+':knots',np.array_equal(ref['knots'],np.r_[np.zeros(4),np.arange(.1,27-1e-9,.1),np.full(4,27.)]))
        check(unit+':original_time',original['time'].shape==(1351,) and abs(original['time'][-1]-27)<1e-9 and close(np.diff(original['time']),np.full(1350,.02)))
        check(unit+':waypoints_exact',i['waypoints']==source['path'])
        check(unit+':obstacles_exact',all(close(o['position'],p['center']) and close(o['size'],[p['radius'],0.,0.]) for o,p in zip(i['scene']['objects'],profile['obstacles'])))
        par=source['task_parameters'];rng=np.random.default_rng(np.random.SeedSequence([i['task']['seed'],37]));vel=rng.uniform([-.0025,.001,-.0015],[.0025,.004,.0015]);omega=rng.uniform(-.0015,.0015,3)
        check(unit+':target_initial_motion',np.array_equal(vel,par['target_velocity']) and np.array_equal(omega,par['target_omega_local']) and close(initial[155:161],np.r_[vel,omega]))
        check(unit+':target_initial_position',close(initial[75:78],np.array([1.2,0,.05])+np.asarray(par['target_shift'])))
        check(unit+':original_gate_values',np.array_equal(lim['velocity'],np.r_[np.full(10,.8),np.full(7,1.2)]) and np.array_equal(lim['acceleration'],np.r_[np.full(10,2.5),np.full(7,4.)]) and np.array_equal(lim['native_torque'],np.r_[np.full(60,4.),np.full(7,80.)]) and np.array_equal(lim['lower'],np.r_[np.full(10,-1.),np.full(7,-np.pi)]) and np.array_equal(lim['upper'],np.r_[np.full(10,1.),np.full(7,np.pi)]))
        if item['reuse']:
            prev=ROOT/item['reuse'];receipt=read(d/'reuse.json');oldid=read(prev/'identity.json');oldresult=read(prev/'result.json');oldrun=read(prev.parent/'run.json')
            check(unit+':reuse_receipt',receipt['passed'] and all(receipt['checks'].values()) and not receipt['new_execution'] and receipt['reused_from']==item['reuse'])
            check(unit+':reuse_identity',i==oldid)
            for name in ['reference.npz','initial.npz','limits.json']:check(unit+':reuse_file:'+name,sha(d/name)==sha(prev/name))
            for path,h in oldrun['identity']['source_hashes'].items():check(unit+':old_source_identity:'+path,sha(path)==h)
            check(unit+':old_native_hash',sha(prev/'native_trace.npz')==oldresult['native_trace_sha256'])
            check(unit+':old_replay_control_hash',sha(prev/'replay_trace.npz')==binding['files'][(prev/'replay_trace.npz').relative_to(ROOT).as_posix()])
            check(unit+':reuse_qualified',oldresult['status']=='EXECUTION_VERIFIED' and all(oldresult['gates'].values()) and result['status']=='EXECUTION_VERIFIED')
            expected_result=dict(oldresult,new_execution=False,actual_new_steps=0,reused_from=item['reuse'],unit=unit,split=item['split'],seed=item['source']['seed'])
            check(unit+':reuse_result_unchanged',result==expected_result)
            RAW['units'][unit]={'mode':'EXACT_HISTORICAL_REUSE','physical_reaudit':False,'failures':[c for c in CHECKS[start:] if not c['passed']]}
            print(unit,'reuse checked',flush=True);continue
        s=BSpline(ref['knots'],ref['coefficients'],3,extrapolate=False);bounds=polynomial_extrema(s)
        check(unit+':polynomial_bounds',close(bounds,rep['polynomial_bounds']))
        check(unit+':polynomial_q',np.all(bounds[0,0]>=lim['lower']) and np.all(bounds[0,1]<=lim['upper']))
        check(unit+':polynomial_dq',np.all(abs(bounds[1])<=lim['velocity']))
        check(unit+':polynomial_ddq',np.all(abs(bounds[2])<=lim['acceleration']))
        check(unit+':initial_q_dq',close(s(0),original['q'][0]) and close(s(0,1),original['dq'][0]))
        check(unit+':C2_simple_interior_knots',len(np.unique(s.t[4:-4]))==269 and s.k==3)
        check(unit+':solver',len(rep['solver'])==17 and all(r['success'] for r in rep['solver']) and all(rep['gates'].values()))
        h=load(d/'native_trace.npz');r=load(d/'replay_trace.npz');g=load(d/'geometry.npz');pairs=read(d/'geometry_pairs.json')
        shapes={'time':(13501,),'qpos':(13501,81),'qvel':(13501,79),'integration':(13501,836),'ctrl':(13500,67),'raw_torque':(13500,67),'ee':(13501,2,4,4),'desired':(13501,2,4,4)}
        check(unit+':full_shapes',all(h[k].shape==shape for k,shape in shapes.items()))
        check(unit+':all_finite',all(np.isfinite(a).all() for a in h.values()) and all(np.isfinite(a).all() for a in r.values()))
        check(unit+':native_hash',sha(d/'native_trace.npz')==result['native_trace_sha256'])
        check(unit+':all_13500_steps',close(h['time'],np.arange(13501)*.002,1e-8) and result['steps']==13500 and result['stop_reason'] is None)
        check(unit+':full_replay_exact',all(np.array_equal(h[k],r[k]) for k in ['integration','qpos','qvel','ctrl']))
        check(unit+':full_initial_equal',np.array_equal(h['integration'][0],initial))
        check(unit+':integration_layout',np.array_equal(h['integration'][:,0],h['time']) and np.array_equal(h['integration'][:,1:82],h['qpos']) and np.array_equal(h['integration'][:,82:161],h['qvel']) and np.array_equal(h['integration'][1:,240:307],h['ctrl']))
        check(unit+':no_external_applied_wrench',np.count_nonzero(h['integration'][:,307:])==0)
        ordered=sorted(i['robot']['joint_index_map'].values(),key=lambda m:m['actuator']);qids=[m['qpos'] for m in ordered];vids=[m['dof'] for m in ordered]
        lowq=h['qpos'][:,qids];lowv=h['qvel'][:,vids];q=lowq@lim['Bplus'].T;v=lowv@lim['Bplus'].T;a=np.diff(v,axis=0)/.002;la=np.diff(lowv,axis=0)/.002
        check(unit+':q17_recomputed',close(q,h['q17']))
        check(unit+':q_domain',np.all(q>=lim['lower']) and np.all(q<=lim['upper']) and np.all(h['qref']>=lim['lower']) and np.all(h['qref']<=lim['upper']))
        joint_pass=True
        for j in JOINTS:
            if j.attrib['type']=='revolute':
                limits=j.find('limit');address=i['robot']['joint_index_map'][j.attrib['name']]['qpos']
                joint_pass &= bool(np.all(h['qpos'][:,address]>=float(limits.attrib['lower'])) and np.all(h['qpos'][:,address]<=float(limits.attrib['upper'])))
        check(unit+':native_joint_domain',joint_pass)
        actual_ratios=[float(np.max(abs(v)/lim['velocity'])),float(np.max(abs(a)/lim['acceleration'])),float(np.max(abs(lowv)/(abs(lim['B'])@lim['velocity']))),float(np.max(abs(la)/(abs(lim['B'])@lim['acceleration'])))]
        check(unit+':actual_dq_ddq',max(actual_ratios)<=1,actual_ratios)
        cmdt=np.arange(1,13501)*.002
        check(unit+':command_spline',close(h['qref'][1:],s(cmdt)) and close(h['reference_qd'],s(cmdt,1)) and close(h['reference_qdd'],s(cmdt,2)))
        check(unit+':command_derivatives',np.all(abs(h['reference_qd'])<=lim['velocity']) and np.all(abs(h['reference_qdd'])<=lim['acceleration']))
        torque_ratio=float(np.max(abs(h['raw_torque'])/lim['native_torque']))
        check(unit+':raw_torque',torque_ratio<=1 and np.array_equal(h['ctrl'],h['raw_torque']),torque_ratio)
        tracking=err(q,h['qref']);manifold=err(lowq,q@lim['B'].T)
        check(unit+':tracking_manifold',tracking<=.02 and manifold<=.02,[tracking,manifold])
        mp=np.array([np.max(np.linalg.norm(h['momentum'][:,x],axis=1)) for x in [slice(0,3),slice(3,6)]])
        me=np.array([np.max(np.linalg.norm(h['exchange'][:,x],axis=1)) for x in [slice(0,3),slice(3,6)]])
        check(unit+':momentum_saved_arrays',np.all(mp<=1e-5+.002*me) and close(mp,result['momentum_peak']) and close(1e-5+.002*me,result['momentum_threshold']))
        check(unit+':geometry_shape_pairs',g['qpos'].shape==(27001,81) and g['minima'].shape==(27001,) and len(pairs)==27001)
        check(unit+':geometry_actual_frames',np.array_equal(g['qpos'][0],h['qpos'][0]) and np.array_equal(g['qpos'][1::2],h['qpos'][1:]))
        mids=(h['qpos'][:-1]+h['qpos'][1:])/2
        for quat in [slice(3,7),slice(77,81)]:
            qa=h['qpos'][:-1,quat];qb=h['qpos'][1:,quat];qb=qb*np.where(np.sum(qa*qb,axis=1)<0,-1,1)[:,None]
            mids[:,quat]=(qa+qb)/np.linalg.norm(qa+qb,axis=1)[:,None]
        check(unit+':geometry_midpoints',close(g['qpos'][2::2],mids))
        expected_clear=np.r_[g['minima'][0],np.minimum(g['minima'][1::2],g['minima'][2::2])]
        check(unit+':all_geometry_nonpenetrating',np.all(g['minima']>=0) and close(expected_clear,h['clearance']))
        check(unit+':minimum_pair_records',all(isinstance(p,dict) and all(k in p for k in ['geom_a','geom_b','body_a','body_b']) for p in pairs))
        pos,rot=transforms(h['qpos'],i)
        ee=np.broadcast_to(np.eye(4),h['ee'].shape).copy()
        for arm,body in enumerate(['link_30','end_effector_r']):
            ee[:,arm,:3,:3]=rot[body];ee[:,arm,:3,3]=pos[body]
        ee[:,0,:3,3]+=np.einsum('nij,j->ni',ee[:,0,:3,:3],[.0475,0.,0.])
        pose_delta=err(ee,h['ee']);check(unit+':independent_URDF_world_pose',pose_delta<1e-10,pose_delta)
        desired=desired_algebra(h,i,source,profile,pos,rot);desired_delta=err(desired,h['desired']);check(unit+':independent_world_target',desired_delta<1e-10,desired_delta)
        position=np.linalg.norm(ee[:,:,:3,3]-desired[:,:,:3,3],axis=2);rotation=Rotation.from_matrix((ee[:,:,:3,:3]@desired[:,:,:3,:3].transpose(0,1,3,2)).reshape(-1,3,3)).magnitude().reshape(-1,2)
        hold=h['time']>=25.5-1e-9
        check(unit+':all_world_errors',np.all(position<=.005) and np.all(rotation<=np.pi/180))
        check(unit+':hold_complete',hold.sum()==751 and np.all(position[hold]<=.005) and np.all(rotation[hold]<=np.pi/180))
        for key,values in [('position_mm',position*1000),('rotation_deg',np.rad2deg(rotation))]:
            dist=dict(mean=values.mean(axis=0),p95=np.percentile(values,95,axis=0),maximum=values.max(axis=0),final=values[-1])
            check(unit+':summary_'+key,all(close(dist[k],result[key][k],1e-7) for k in dist))
        # Direct conservative motion bound of all collision supports per half native step.
        midpos,midrot=transforms(mids,i)
        motion=0.
        for body,radius in radii.items():
            for p0,p1,r0,r1 in [(pos[body][:-1],midpos[body],rot[body][:-1],midrot[body]),(midpos[body],pos[body][1:],midrot[body],rot[body][1:])]:
                translation=np.linalg.norm(p1-p0,axis=1)
                angle=Rotation.from_matrix(r1@r0.transpose(0,2,1)).magnitude()
                motion=max(motion,float(np.max(translation+radius*angle)))
        check(unit+':conservative_geometry_step_bound',motion<=.002,{'m':motion,'method':'URDF_collision_link_origin_translation_plus_full_mesh_radius_times_angle_for_each_actual_halfstep'})
        for boundary in [45,255]:
            z=load(d/f'transition_{boundary}.npz');index=round(boundary/10/.002)
            check(unit+f':transition_{boundary}_unchanged',all(np.array_equal(z[k],h[k][index-1:index+2]) for k in z))
        check(unit+':all_report_gates',result['status']=='EXECUTION_VERIFIED' and all(result['gates'].values()) and result['new_execution'] and result['actual_new_steps']==13500)
        RAW['units'][unit]={'mode':'NEW_EXECUTION_AND_REPLAY','steps_each':13500,'all_saved_states':13501,'hold_states':int(hold.sum()),'actual_velocity_acceleration_ratios':actual_ratios,'raw_torque_ratio':torque_ratio,'tracking_max_rad':tracking,'manifold_max_rad':manifold,'position_max_mm':(position.max(axis=0)*1000).tolist(),'rotation_max_deg':np.rad2deg(rotation.max(axis=0)).tolist(),'hold_position_max_mm':(position[hold].max(axis=0)*1000).tolist(),'minimum_clearance_mm':float(g['minima'].min()*1000),'URDF_pose_error':pose_delta,'world_target_error':desired_delta,'mesh_motion_halfstep_upper_bound_m':motion,'recorded_geometry_step_m':result['maximum_geometry_step_m'],'momentum_peak':mp.tolist(),'momentum_threshold':(1e-5+.002*me).tolist(),'failures':[c for c in CHECKS[start:] if not c['passed']]}
        print(unit,RAW['units'][unit],flush=True)
        del h,r,g,original,midpos,midrot,pos,rot
    rows=selection['rows'];results=[read(RUN/x['id']/'result.json') for x in rows];qualified=read(RUN/'qualified_view.json')
    check('result_denominator',len(results)==12 and summary['denominator']==12 and results==summary['results'])
    check('qualified_exact_12',[r['unit'] for r in qualified['rows']]==[r['id'] for r in rows] and all(r['grade']=='EXECUTION_VERIFIED' for r in qualified['rows']))
    for q,item in zip(qualified['rows'],rows):
        check('qualified_pointer:'+q['unit'],q['split']==item['split'] and q['seed']==item['source']['seed'] and q['reference_sha256']==sha(q['reference']) and q['evidence']==(item['reuse'] or (RUN/item['id']).relative_to(ROOT).as_posix()))
    check('train_val_gate',qualified['gate']=={'counts':{'train':8,'validation':4},'passed':True} and summary['training_gate']==qualified['gate'])
    check('no_training_expansion',summary['training_updates']==0 and summary['TRAIN_added']==0 and not summary['TEST'] and not qualified['master_library_modified'] and not qualified['validation_used_in_training'] and qualified['new_tasks']==0)
    check('new_8_reuse_4',summary['actual_new_executions']==8 and summary['historical_reused']==4)
    check('budget_216000',budget['native_steps_reserved']==108000 and budget['replay_steps_reserved']==108000 and budget['inflight'] is None and budget['output_bytes']<=5368709120 and budget['elapsed_s']<=3000)
    status=read(RUN/'status.json');check('terminal_finished',status['status']=='FINISHED' and status['stage']=='complete')
    pid=read(TRACE/'pid_check.json');check('pid_dead',pid['pid']==status['pid'] and not pid['running'])
    check('exact_unit_directory_inventory',{p.name for p in RUN.iterdir() if p.is_dir()}=={r['id'] for r in rows})
    sha(RUN/'events.jsonl');events=[json.loads(line) for line in (RUN/'events.jsonl').read_text().splitlines()]
    check('event_initialize_denominator',[e['unit'] for e in events if e['stage']=='initialize']==[r['id'] for r in rows])
    check('event_complete_denominator',[e['unit'] for e in events if e['stage']=='case_complete']==[r['id'] for r in rows])
    check('event_single_fit_each_new_coordinate',sum(e['stage']=='constrained_fit' for e in events)==8*17)
    check('no_failure_files',not (RUN/'failure.json').exists() and not (RUN/'run.lock').exists())
    manifest=read('ff_prior_temporal_data/evidence/core_manifest.json')
    check('core_manifest_146',len(manifest)==146)
    check('core_manifest_inventory',{p.relative_to(RUN).as_posix() for p in RUN.rglob('*') if p.is_file()}==set(manifest))
    for path,h in manifest.items():check('core_manifest:'+path,sha(RUN/path)==h)
    resume=ROOT/'ff_prior_temporal_data/evidence/completed_resume.json'
    if resume.exists():
        receipt=read(resume);RAW['executor_completed_resume']=receipt
        check('executor_resume_no_work_once',receipt['calls']==1 and receipt['returncode']==0 and receipt['core_files']==146 and receipt['core_files_unchanged'] and receipt['verification']=='COMPLETE' and 'NO_WORK_ALREADY_COMPLETE' in receipt['stdout'])
    audit=ROOT/'paper/PAPER_CLAIM_AUDIT.json';RAW['paper_claim_audit']=read(audit) if audit.exists() else 'unavailable'
    RAW.update(checks=CHECKS,checks_total=len(CHECKS),checks_failed=sum(not c['passed'] for c in CHECKS),source_count=len(ident['source_hashes']),binding_count=len(binding['files']),protocol=protocol,budget=budget,status=status,prohibited_actions={'simulation':0,'geometry_queries':0,'optimizer_step':0,'training':0,'launcher':0,'resume':0,'old_execution_physical_reaudit':0})

if __name__=='__main__':
    started=time.time()
    try:main()
    finally:
        RAW['audit_elapsed_s']=time.time()-started
        (TRACE/'raw_audit.json').write_text(json.dumps(RAW,ensure_ascii=False,indent=2),encoding='utf-8')
        (TRACE/'inputs.json').write_text(json.dumps({'inputs':INPUTS,'generated_at_epoch':time.time()},ensure_ascii=False,indent=2),encoding='utf-8')
    print('CHECKS',len(CHECKS),'FAILED',sum(not c['passed'] for c in CHECKS),flush=True)
