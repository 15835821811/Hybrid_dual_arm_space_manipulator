"""Saved public-condition coverage; no inference, physics, or checkpoint selection."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from scipy.spatial import cKDTree
from ff_prior.storage import read_json, file_digest, digest
from ff_prior_pilot_train.storage import atomic_json
from ff_prior.contracts import public_condition

ROOT=Path(__file__).resolve().parents[1]
GROUPS={'initial_q17':(0,17),'world_goal_position':(17,23),'world_goal_rotation6d':(23,35),
        'duration':(35,36),'pose_mask':(36,48)}


def summary(a):
    a=np.asarray(a)
    return {'minimum':float(a.min()),'median':float(np.median(a)),'maximum':float(a.max())}


def analyze(source,validation,output):
    source,validation,output=map(lambda p:Path(p).resolve(),(source,validation,output))
    if output.exists():raise ValueError('choose a new output; existing evidence is preserved')
    if output==source or source in output.parents or output==validation or validation in output.parents:
        raise ValueError('output must be outside immutable source runs')
    if read_json(source/'independent_verification.json')['status']!='PASS':raise ValueError('training raw audit required')
    started=time.perf_counter();output.mkdir(parents=True)
    paths=[source/'training_index.json',source/'loaded_training_arrays.npz',source/'learning/goal_loss/training.json',
           source/'learning/goal_loss/codec.json',source/'independent_verification.json',validation/'frozen_tasks.json']
    inputs={str(p):file_digest(p) for p in paths}
    index=read_json(source/'training_index.json')['records'];rows=[]
    for e in index:
        raw=(ROOT/e['record']).read_bytes()
        if hashlib.sha256(raw).hexdigest()!=e['record_sha256']:raise ValueError('record hash mismatch')
        rows.append(json.loads(raw))
    with np.load(source/'loaded_training_arrays.npz') as a:z,bases=a['Z'].copy(),a['bases'].copy()
    assert z.shape==bases.shape==(len(index),19,17)
    conditions=[r['condition'] for r in rows];ids=[e['sample_id'] for e in index]
    splits=[e['split'] for e in index];states=[r['task']['contract']['initial_state_id'] for r in rows]
    training=read_json(source/'learning/goal_loss/training.json');codec=read_json(source/'learning/goal_loss/codec.json')
    actual_hash=digest({'z':z,'bases':bases,'conditions':conditions,'ids':ids,'splits':splits,'states':states})
    assert actual_hash==training['metadata']['data_hash'] and training['status']=='PASS'
    fit_ids=set(codec['fit_ids']);fit_mask=np.array([e['sample_id'] in fit_ids for e in index])
    assert digest({'z':z[fit_mask],'bases':bases[fit_mask],
                   'conditions':[c for c,m in zip(conditions,fit_mask) if m],
                   'ids':[sid for sid,m in zip(ids,fit_mask) if m]})==codec['fit_hash']
    assert file_digest(source/'learning/goal_loss/codec.json')==training['metadata']['codec_sha256']
    mask=np.array([s=='TRAIN' for s in splits]);assert sum(mask)==12144
    features=np.stack([public_condition(c) for c in conditions])[mask]
    assert features.shape==(12144,48)
    mean=np.asarray(codec['feature_mean'])[:48];scale=np.asarray(codec['feature_scale'])[:48]
    normalized=(features-mean)/scale
    lo,hi=features.min(axis=0),features.max(axis=0)
    unique=np.unique(normalized,axis=0);tree=cKDTree(unique)
    # Public conditions only: deliberately exclude the 17 initializer-derived
    # features. This metric diagnoses coverage; it is not a physical distance.
    train_nn=tree.query(unique,k=2,workers=1)[0][:,1]
    q0=np.asarray([c['initial_features'] for c,m in zip(conditions,mask) if m])
    origins,origin_counts=np.unique(q0,axis=0,return_counts=True)
    train_motion=np.max(abs(z[mask]-q0[:,None,:]),axis=(1,2))
    train_curvature=np.sqrt(np.mean(np.diff(z[mask],n=2,axis=1)**2,axis=(1,2)))
    frozen=read_json(validation/'frozen_tasks.json');tasks=[]
    for e in frozen['entries']:
        if e['status']!='FROZEN':continue
        c=e['task']['condition'];v=public_condition(c);standard=(v-mean)/scale
        outside=(v<lo-1e-12)|(v>hi+1e-12)
        path=validation/'tasks'/e['sample_id']/'witness.npz';inputs[str(path)]=file_digest(path)
        with np.load(path) as a:witness=a['Z'].copy()
        assert file_digest(path)==e['witness_sha256']
        q=np.asarray(c['initial_features']);motion=float(abs(witness-q).max())
        origin_l2=float(np.linalg.norm(origins-q,axis=1).min())
        origin_linf=float(np.max(abs(origins-q),axis=1).min())
        curvature=float(np.sqrt(np.mean(np.diff(witness,n=2,axis=0)**2)))
        distance,nearest=tree.query(standard,k=1,workers=1)
        tasks.append({'sample_id':e['sample_id'],'kind':e['kind'],
                      'outside_TRAIN_features':int(outside.sum()),
                      'outside_by_group':{name:int(outside[a:b].sum()) for name,(a,b) in GROUPS.items()},
                      'maximum_abs_codec_standardized_public_feature':float(abs(standard).max()),
                      'nearest_TRAIN_public_feature_distance':float(distance),
                      'nearest_TRAIN_q17_origin_L2_rad':origin_l2,
                      'nearest_TRAIN_q17_origin_Linf_rad':origin_linf,
                      'maximum_motion_rad':motion,'motion_above_TRAIN_maximum':bool(motion>train_motion.max()),
                      'control_second_difference_RMS_rad':curvature,
                      'curvature_above_TRAIN_maximum':bool(curvature>train_curvature.max())})
    assert len(tasks)==12
    assert all(file_digest(p)==h for p,h in inputs.items())
    result={'status':'DIAGNOSTIC_COMPLETE','TRAIN_rows':int(sum(mask)),
            'unique_TRAIN_public_conditions':len(unique),'public_features_checked':48,
            'unique_TRAIN_q17_origins':len(origins),'TRAIN_rows_per_q17_origin':summary(origin_counts),
            'initializer_features_checked':0,'all_saved_training_data_hash_verified':actual_hash,
            'old_codec_fit_hash_verified':codec['fit_hash'],'inputs_sha256':inputs,
            'TRAIN_max_abs_codec_standardized_public_feature':float(abs(normalized).max()),
            'TRAIN_nearest_other_unique_public_condition_distance':summary(train_nn),
            'TRAIN_maximum_motion_rad':summary(train_motion),'TRAIN_control_second_difference_RMS_rad':summary(train_curvature),
            'new_tasks':tasks,'new_tasks_outside_TRAIN_envelope':sum(t['outside_TRAIN_features']>0 for t in tasks),
            'outside_tasks_by_group':{name:sum(t['outside_by_group'][name]>0 for t in tasks) for name in GROUPS},
            'new_task_nearest_TRAIN_public_condition_distance':summary([t['nearest_TRAIN_public_feature_distance'] for t in tasks]),
            'network_calls':0,'physics_calls':0,'training_updates':0,'inputs_unchanged':True,
            'not_a_physical_validator':True,'not_a_failure_causality_test':True,'not_used_for_current_selection':True,
            'wall_s':time.perf_counter()-started}
    atomic_json(output/'diagnostic.json',result)
    lines=['# 新验证任务的公开条件与运动覆盖诊断','',
           f'实际核验12144 TRAIN，{len(unique)}个不同公开条件；12个新任务中{result["new_tasks_outside_TRAIN_envelope"]}个至少一维超出TRAIN逐维范围。',
           f'TRAIN只有{len(origins)}个不同q17初态；每初态样本数量范围{result["TRAIN_rows_per_q17_origin"]}。轨迹数量增加不等同于初态空间已覆盖。',
           '读取实际保存数组和逐条记录，重算完整训练data_hash及冻结10k codec fit_hash一致。无网络、物理、训练或检查点选择调用。','',
           '| 任务 | 类型 | 初态越界维数 | 目标位置越界维数 | 目标旋转越界维数 | 到最近TRAIN公开条件距离 | 最大运动(rad) |','|---|---|---:|---:|---:|---:|---:|']
    for t in tasks:
        g=t['outside_by_group'];lines.append(f'| {t["sample_id"]} | {t["kind"]} | {g["initial_q17"]} | {g["world_goal_position"]} | {g["world_goal_rotation6d"]} | {t["nearest_TRAIN_public_feature_distance"]:.6f} | {t["maximum_motion_rad"]:.6f} |')
    lines+=['','这里的“越界”指超出训练数据的经验范围，不是违反机器人关节或其他物理门槛。逐维范围内也不保证联合分布覆盖；条件距离经过旧codec标准化，不等同世界位姿物理距离。',
            '仅检查公开48维条件，没有检查另外17维公开初始化器派生特征；本诊断不运行网络，不能确定外推造成了哪些具体失败，也不能证明扩大范围后必然改善。未使用当前对照输赢筛选任务或改动本轮模型。',
            f'TRAIN最大运动范围：{result["TRAIN_maximum_motion_rad"]}；控制点二阶差分RMS范围：{result["TRAIN_control_second_difference_RMS_rad"]}。',
            '上述轨迹形状统计来自实际教师控制点，不是完整动力学或表示重建误差。完整物理合格与神经网络收益仍以独立验证、真实力矩执行及重放结果判断。']
    (output/'ANALYSIS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return {k:result[k] for k in ('status','TRAIN_rows','unique_TRAIN_public_conditions','new_tasks_outside_TRAIN_envelope','outside_tasks_by_group','wall_s')}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',default='artifacts/v7_1/qualified_coverage_train_001')
    p.add_argument('--validation',default='artifacts/v7_1/independent_new_validation_001')
    p.add_argument('--output',required=True);a=p.parse_args()
    print(analyze(a.source,a.validation,a.output))


if __name__=='__main__':main()
