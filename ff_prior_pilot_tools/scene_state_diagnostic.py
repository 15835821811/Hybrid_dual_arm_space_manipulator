"""Explain retained scene-state errors without modifying runs or advancing physics."""
import argparse
from pathlib import Path
import time
import mujoco
import numpy as np
from ff_prior.storage import read_json, file_digest, digest
from ff_prior_pilot_train.storage import atomic_json
from ff_prior.pipeline import audit, acceptance
from ff_prior.initial_state import STATE
from ff_prior.scenes import SceneSpec, Obstacle
from ff_prior.geometry import CollisionBackend
from ff_prior_complex.tasks import scene_runtime, transfer_audit
from ff_prior_loop.runtime import at_origin, origin_audit

ROOT=Path(__file__).resolve().parents[1]


def components(robot,initial):
    # Preserve the exact captured vector, including warmstart. No mj_step or
    # forward is called here, because forward may change solver state.
    d=mujoco.MjData(robot.model)
    mujoco.mj_resetData(robot.model,d)
    mujoco.mj_setState(robot.model,d,initial.values.copy(),STATE)
    result={}
    for name,flag in mujoco.mjtState.__members__.items():
        value=int(flag)
        if value<=0 or value&(value-1) or not (value&int(STATE)):continue
        a=np.empty(mujoco.mj_stateSize(robot.model,flag))
        mujoco.mj_getState(robot.model,d,a,flag);result[name]=a
    return result


def compare(parent,pi,child,ci):
    a,b=components(parent,pi),components(child,ci)
    return {k:{'elements':len(a[k]),'equal':bool(np.array_equal(a[k],b[k])),
               'maximum_abs_difference':float(np.max(abs(a[k]-b[k]))) if len(a[k]) else 0.,
               'different_elements':int(np.sum(a[k]!=b[k]))} for k in a}


def diagnose(source,output):
    source,output=Path(source).resolve(),Path(output).resolve()
    if output.exists():raise ValueError('new diagnostic output required; refusing overwrite')
    run=read_json(source/'run.json')
    if any(p.startswith('ff_prior_pilot_tools/') for p in run['identity']['source_hashes']):
        raise ValueError('diagnostic source belongs to the frozen run')
    files=[p for p in sorted((source/'generation').glob('*.json'))
           if read_json(p).get('reason')=="ValueError(\"scene transfer changed robot: ['integration_state']\")"]
    if not files:raise ValueError('no retained matching errors')
    output.mkdir(parents=True);started=time.perf_counter()
    inputs={str(p):file_digest(p) for p in files}
    inputs[str(source/'frozen_tasks.json')]=file_digest(source/'frozen_tasks.json')
    config=read_json(ROOT/'artifacts/v7_1/smoke_002/run.json')['config']
    _,canonical,initial,collision=audit(config)
    cases=[];calls={}
    for path in files:
        r=read_json(path);screen=r['obstacle_screen']['attempts'][-1]
        assert screen['passed'] and not r['unknown_screen_call_accounting']
        # Reconstruct the exact scratch sphere emitted by the frozen generator.
        scene=SceneSpec((Obstacle('detour','sphere',tuple(screen['position']),(.012,0.,0.)),),r['family_id'])
        child,ci,cc=scene_runtime(scene)
        home_difference=compare(canonical,initial,child,ci)
        try:transfer_audit(canonical,initial,child,ci);home_transfer='PASS'
        except ValueError as error:home_transfer=str(error)
        home_geometry=cc.scan(ci.reset(child),0.)
        q0=np.asarray(r['origin_audit']['planning_origin'])
        parent_origin,pi,pc=at_origin(canonical,q0)
        child_origin,cii,ccc=at_origin(child,q0)
        assert pi.state_id==r['origin_audit']['initial_state_id']
        origin_difference=compare(parent_origin,pi,child_origin,cii)
        try:origin_transfer=transfer_audit(parent_origin,pi,child_origin,cii)
        except ValueError as error:origin_transfer={'status':'FAIL','reason':str(error)}
        actual_geometry=ccc.scan(cii.reset(child_origin),0.)
        origin_proof=origin_audit(child,child_origin,cii)
        case={'source':str(path),'slot':r['slot'],'attempt':r['attempt'],'scene':scene.to_dict(),
              'home_state_components':home_difference,'home_transfer_result':home_transfer,
              'home_scene_clearance_m':home_geometry['minimum_clearance_m'],
              'actual_origin_state_components':origin_difference,'actual_origin_transfer':origin_transfer,
              'actual_origin_clearance_m':actual_geometry['minimum_clearance_m'],'actual_origin_audit':origin_proof,
              'actual_origin_state_complete':True,'state_fields_dropped':False}
        case['explained']=bool([k for k,v in home_difference.items() if not v['equal']]==['mjSTATE_WARMSTART']
                               and home_geometry['minimum_clearance_m']<0
                               and all(v['equal'] for v in origin_difference.values())
                               and origin_transfer['status']=='PASS' and actual_geometry['minimum_clearance_m']>=0)
        cases.append(case)
        for robot in (child,parent_origin,child_origin):
            for k,v in robot.calls.items():calls[k]=calls.get(k,0)+v
    for k,v in canonical.calls.items():calls[k]=calls.get(k,0)+v
    assert all(file_digest(p)==h for p,h in inputs.items())
    assert calls['physics_steps']==calls['coupled']==calls['torque_rollout']==0
    result={'status':'DIAGNOSED' if all(c['explained'] for c in cases) else 'INCONCLUSIVE',
            'cases':cases,'inputs_sha256':inputs,'inputs_unchanged':True,'calls':calls,'network_calls':0,
            'wall_s':time.perf_counter()-started,'old_run_modified':False,
            'state_or_physics_gates_relaxed':False,'new_execution_pass_claimed':False,
            'scope':'initial-state/geometry diagnosis only; no trajectory validation, inference, integration or execution'}
    atomic_json(output/'diagnostic.json',result)
    lines=['# 场景转移完整初态错误诊断','',f'结论：{result["status"]}；实际复现{len(cases)}例，0耦合传播、0物理步、0网络调用。','',
           '保留原作业、任务、拒绝和错误，未改动正在运行的源码、协议或物理门槛。以下为初始化诊断，不是新增轨迹或执行通过。','']
    for c in cases:
        different=[k for k,v in c['home_state_components'].items() if not v['equal']]
        lines.append(f'- slot={c["slot"]}, attempt={c["attempt"]}：默认home完整初态差异字段{different}；home间距{c["home_scene_clearance_m"]:.9g}m；实际任务初态间距{c["actual_origin_clearance_m"]:.9g}m；实际初态完整状态转移审计{c["actual_origin_transfer"]["status"]}。')
    lines+=['','若DIAGNOSED：新障碍在默认home构型下接触，初始化forward产生不同的求解器warmstart，导致旧home转移检查拒绝；任务本身采用另一个完整初态。实际任务初态逐字段完全一致、机器人参数及静态几何转移检查通过，说明这两例错误不等于机器人动力学参数被改动。',
            '可供后续版本验证的修复是在实际任务初态比较完整integration state，并保留warmstart字段、全部机器人转移检查及全机不穿透验证；不能简单删除warmstart检查。当前冻结运行不采用此修复，原两次失败继续保留。',
            '本诊断不验证完整教师轨迹，不将此前拒绝变为已准入，也不升级任何数据等级。']
    (output/'ANALYSIS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return {'status':result['status'],'cases':len(cases),'calls':calls,'output':str(output)}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',default='artifacts/v7_1/independent_new_validation_001')
    p.add_argument('--output',required=True);a=p.parse_args()
    print(diagnose(a.source,a.output))


if __name__=='__main__':main()
