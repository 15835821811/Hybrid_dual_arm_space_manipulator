"""Read-only explanation of retained task-origin scene integration-state errors."""
import argparse
from pathlib import Path
import time
import numpy as np
from ff_prior.storage import read_json,file_digest
from ff_prior_pilot_train.storage import atomic_json
from ff_prior.pipeline import audit
from ff_prior.scenes import SceneSpec
from ff_prior_complex.tasks import scene_runtime,transfer_audit
from ff_prior_loop.runtime import at_origin
from .scene_state_diagnostic import compare

ROOT=Path(__file__).resolve().parents[1]
REASON="ValueError(\"scene transfer changed robot: ['integration_state']\")"


def diagnose(source,output):
    source,output=Path(source).resolve(),Path(output).resolve()
    if output.exists():raise ValueError('new diagnostic output required')
    run=read_json(source/'run.json')
    if 'ff_prior_pilot_tools/origin_contact_diagnostic.py' in run['identity']['source_hashes']:
        raise ValueError('diagnostic must be independent of frozen source')
    paths=[p for p in sorted((source/'generation').glob('*.json')) if read_json(p).get('reason')==REASON]
    if not paths:raise ValueError('no matching retained errors')
    inputs={str(p):file_digest(p) for p in paths};inputs[str(source/'run.json')]=file_digest(source/'run.json')
    output.mkdir(parents=True);started=time.perf_counter()
    _,canonical,initial,_=audit(read_json(ROOT/'artifacts/v7_1/smoke_002/run.json')['config'])
    cases=[];calls={}
    for p in paths:
        r=read_json(p);scene=SceneSpec.from_dict(r['scene']);q0=np.asarray(r['q17_origin'])
        parent,pi,_=at_origin(canonical,q0)
        base,_,_=scene_runtime(scene);child,ci,cc=at_origin(base,q0)
        assert pi.state_id==r['empty_origin_audit']['scene_task_origin_state_id']
        components=compare(parent,pi,child,ci)
        try:transfer=transfer_audit(parent,pi,child,ci)
        except ValueError as error:transfer={'status':'FAIL','reason':str(error)}
        geometry=cc.scan(ci.reset(child),0.)
        different=[k for k,v in components.items() if not v['equal']]
        cases.append({'source':str(p),'sample_id':r['sample_id'],'split':r['split'],
            'state_components':components,'different_state_fields':different,'transfer':transfer,
            'actual_initial_geometry':geometry,'source_status_retained':r['status'],
            'explained_by_initial_penetration':bool(different==['mjSTATE_WARMSTART'] and
                transfer.get('reason')=="scene transfer changed robot: ['integration_state']" and not geometry['passed']),
            'initial_state_fields_dropped':False,'admission_or_execution_upgraded':False})
        for robot in (parent,base,child):
            for key,value in robot.calls.items():calls[key]=calls.get(key,0)+value
    for key,value in canonical.calls.items():calls[key]=calls.get(key,0)+value
    assert calls['coupled']==calls['physics_steps']==calls['torque_rollout']==0
    assert all(file_digest(p)==h for p,h in inputs.items())
    result={'status':'DIAGNOSED' if all(c['explained_by_initial_penetration'] for c in cases) else 'INCONCLUSIVE',
        'cases':cases,'inputs_sha256':inputs,'inputs_unchanged':True,'calls':calls,'network_calls':0,
        'wall_s':time.perf_counter()-started,'original_errors_or_data_gate_changed':False,
        'scope':'complete initial-state and geometry diagnosis only; no propagation, sampling, native step, training or admission'}
    atomic_json(output/'diagnostic.json',result)
    lines=['# 实际任务初态的场景积分状态错误诊断','',f'状态{result["status"]}；{len(cases)}例；0耦合/物理步/网络。','']
    for c in cases:lines.append(f'- {c["sample_id"]}：不同字段{c["different_state_fields"]}，初始最小间隙{c["actual_initial_geometry"]["minimum_clearance_m"]*1000:.6f}mm；原ERROR保留，未准入或升级。')
    lines+=['','若DIAGNOSED，当前障碍在实际初态穿透，forward产生不同warmstart，转移检查先于几何拒绝触发。后续版本可先检查实际初态几何，使预期物理拒绝有明确原因；仍须保持完整状态和全部转移检查。',
        '本次不修订正在运行的源码、冻结门槛或原始错误；若原数据门要求无ERROR，最终仍按原协议判定。']
    (output/'ANALYSIS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return {'status':result['status'],'cases':len(cases),'calls':calls,'output':str(output)}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();print(diagnose(a.source,a.output))


if __name__=='__main__':main()
