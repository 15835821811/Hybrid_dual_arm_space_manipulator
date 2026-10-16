"""Compare saved public-condition coverage without inference or simulation."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
from ff_prior.storage import read_json,file_digest,digest
from ff_prior.contracts import public_condition
from ff_prior_coverage_train.storage import atomic_json

ROOT=Path(__file__).resolve().parents[1]
GROUPS={'initial_q17':(0,17),'goal_position':(17,23),'goal_rotation6d':(23,35),
        'duration':(35,36),'pose_mask':(36,48)}


def population(source,protected):
    path=source/'training_index.json';protected[str(path)]=file_digest(path)
    index=read_json(path)['records'];features=[];ids=[]
    for e in index:
        if e['split']!='TRAIN':continue
        path=ROOT/e['record'];protected[str(path)]=e['record_sha256']
        if file_digest(path)!=e['record_sha256']:raise ValueError('source record changed')
        if e['execution_status']=='FAIL':raise ValueError('failed row in TRAIN')
        row=read_json(path);features.append(public_condition(row['condition']));ids.append(e['sample_id'])
    return np.stack(features),ids


def stats(x):
    return {'minimum':float(np.min(x)),'median':float(np.median(x)),
            'p95':float(np.quantile(x,.95)),'maximum':float(np.max(x))}


def analyze(source,old,output):
    source,old,output=map(lambda p:Path(p).resolve(),(source,old,output))
    if output.exists() or any(output==p or p in output.parents for p in (source,old)):
        raise ValueError('new independent diagnostic output required')
    start=time.perf_counter();protected={}
    for path in (source/'summary.json',source/'independent_verification.json',
                 source/'frozen_validation.json',old/'learning/goal_loss/codec.json'):
        protected[str(path)]=file_digest(path)
    if not read_json(source/'summary.json')['ready_for_training_stage'] or read_json(source/'independent_verification.json')['status']!='PASS':
        raise ValueError('qualified merged input required')
    new,nids=population(source,protected);prior,pids=population(old,protected)
    codec=read_json(old/'learning/goal_loss/codec.json')
    center=np.asarray(codec['feature_mean'])[:48];scale=np.asarray(codec['feature_scale'])[:48]
    if new.shape!=(12688,48) or prior.shape!=(12144,48):raise ValueError('unexpected populations')
    if not set(pids)<set(nids):raise ValueError('old TRAIN lineage missing')
    validation=read_json(source/'frozen_validation.json');tasks=[]
    for entry in validation['entries']:
        sid=entry['sample_id'];v=public_condition(entry['task']['condition'])
        if sid in set(nids):raise ValueError('NEW_VAL in TRAIN')
        row={'sample_id':sid,'groups':{}}
        for name,features in (('old12144',prior),('merged12688',new)):
            outside=(v<features.min(0)-1e-12)|(v>features.max(0)+1e-12)
            standardized=(features-v)/scale
            row['groups'][name]={'outside_dimensions':int(sum(outside)),
                'outside_by_group':{k:int(sum(outside[a:b])) for k,(a,b) in GROUPS.items()},
                'nearest_public48_standardized_L2':float(np.linalg.norm(standardized,axis=1).min()),
                'nearest_q17_Linf_rad':float(np.max(abs(features[:,:17]-v[:17]),axis=1).min())}
        row['maximum_abs_frozen_codec_standardized_public_feature']=float(abs((v-center)/scale).max())
        tasks.append(row)
    assert len(tasks)==8 and all(file_digest(p)==h for p,h in protected.items())
    result={'status':'DIAGNOSTIC_COMPLETE','inputs_unchanged':True,'inputs_sha256':protected,
        'old_TRAIN':len(prior),'merged_TRAIN':len(new),'old_unique_q17':len(np.unique(prior[:,:17],axis=0)),
        'merged_unique_q17':len(np.unique(new[:,:17],axis=0)),
        'old_max_abs_standardized_public_per_row':stats(abs((prior-center)/scale).max(axis=1)),
        'merged_max_abs_standardized_public_per_row':stats(abs((new-center)/scale).max(axis=1)),
        'frozen_codec_q17_feature_scales':scale[:17],
        'old_public48_hash':digest(prior),'merged_public48_hash':digest(new),'NEW_VAL_tasks':tasks,
        'network_calls':0,'physics_calls':0,'training_updates':0,'initializer_features_checked':0,
        'not_a_physical_validator':True,'not_a_causal_test':True,'not_used_for_current_tuning':True,
        'wall_s':time.perf_counter()-start}
    atomic_json(output/'diagnostic.json',result)
    lines=['# 新覆盖库的公开条件与冻结归一化诊断','',
        f'实际读取旧{len(prior)}和合并{len(new)} TRAIN记录，q17初态{result["old_unique_q17"]}→{result["merged_unique_q17"]}。所有来源哈希不变，无网络、物理或训练调用。',
        '仅公开48维条件；不读取教师答案，不检查另17维公共初始化器派生特征；没有改动正在运行的任务、模型或协议。','',
        '| NEW_VAL任务 | 旧/新初态越界维数 | 旧/新目标位置越界维数 | 旧/新最近条件距离 |',
        '|---|---:|---:|---:|']
    for t in tasks:
        a,b=t['groups']['old12144'],t['groups']['merged12688']
        lines.append(f'| {t["sample_id"]} | {a["outside_by_group"]["initial_q17"]}/{b["outside_by_group"]["initial_q17"]} | {a["outside_by_group"]["goal_position"]}/{b["outside_by_group"]["goal_position"]} | {a["nearest_public48_standardized_L2"]:.6f}/{b["nearest_public48_standardized_L2"]:.6f} |')
    lines += ['',f'旧/新每行最大冻结codec标准化绝对特征统计：{result["old_max_abs_standardized_public_per_row"]} / {result["merged_max_abs_standardized_public_per_row"]}。',
        '“越界”只指数据逐维经验范围，不是物理越界。范围内不代表联合分布密集覆盖，条件距离也不是末端物理距离。',
        '本诊断不能证明初态覆盖、冻结缩放或任何单一因素造成网络误差。正式训练、raw提案、共同修正和真实执行的结果仍须分别核验。若后续据此修改模型，另立协议；本轮不调参。']
    (output/'ANALYSIS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return {k:result[k] for k in ('status','old_unique_q17','merged_unique_q17','wall_s')}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',default='artifacts/v7_1/qualified_origin_fill_001')
    p.add_argument('--old',default='artifacts/v7_1/qualified_coverage_train_001')
    p.add_argument('--output',required=True);a=p.parse_args()
    print(json.dumps(analyze(a.source,a.old,a.output)))


if __name__=='__main__':main()
