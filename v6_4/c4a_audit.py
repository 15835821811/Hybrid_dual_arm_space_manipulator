"""Read-only C3 architecture/data/forensic audit plus TRAIN/VAL-only sampling.

No training, TEST sampling, physics, or checkpoint selection. Outputs are
exclusive and all consumed frozen paths carry SHA-256 identities.
"""
import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
from unittest.mock import patch

import mujoco
import numpy as np
import torch

from .c4a_frozen import RELEASE, load_frozen_sampler
from .route_optimizer_protocol import read, write, sha, digest, initial_candidates, parameter_plan
from .route_initializers import raw_seed_plan
from .preference_teacher_dataset import encode_condition
from .search_effect_teacher import load_search_aware_dataset
from .visualization.export_search_aware_release import PortableResolver


def csv_write(path, rows):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with path.open('x',newline='',encoding='utf8') as f:
        w=csv.DictWriter(f,keys); w.writeheader()
        for row in rows:
            w.writerow({k:json.dumps(v,ensure_ascii=False,allow_nan=False) if isinstance(v,(list,dict)) else v for k,v in row.items()})


def data_audit(ds, out):
    keys=sorted({(s['task_id'],s['preference'],s['family']) for s in ds.samples})
    enc=[encode_condition(ds.tasks[t],ds.definitions[t,f],p,f) for t,p,f in keys]
    raw=np.stack([e['values'] for e in enc]); valid=np.stack([e['valid'] for e in enc])
    norm=np.stack([ds.condition_scaler.transform_condition(e) for e in enc]).astype(float)
    varying=np.ptp(norm,axis=0)>1e-10
    centered=norm-norm.mean(0); rank=int(np.linalg.matrix_rank(centered,tol=1e-6))
    correlations=np.corrcoef(norm[:,varying],rowvar=False)
    variable_ids=np.flatnonzero(varying)
    pairs=[]
    for i,j in zip(*np.where(np.triu(np.abs(correlations)>=.99,1))):
        pairs.append(dict(feature_a=enc[0]['names'][variable_ids[i]],feature_b=enc[0]['names'][variable_ids[j]],correlation=float(correlations[i,j])))
    csv_write(out/'feature_correlations.csv',pairs)
    features=[]
    for i,name in enumerate(enc[0]['names']):
        features.append(dict(index=i,name=name,literal=bool(enc[0]['literal'][i]),valid_rows=int(valid[:,i].sum()),
            always_missing=bool(not valid[:,i].any()),constant_normalized=bool(not varying[i]),
            normalized_min=float(norm[:,i].min()),normalized_max=float(norm[:,i].max()),
            raw_finite=bool(np.isfinite(raw[:,i]).all()),group=name.split('.')[0]))
    csv_write(out/'condition_features.csv',features)
    labels=ds.samples
    semantic=[digest(dict(task=s['task_sha256'],preference=s['preference'],family=s['family'],z=s['z_m'])) for s in labels]
    source_counts=Counter(src for s in labels for src in s['source_types'])
    split=read(RELEASE/'snapshot/learning_split_manifest.json')
    identities={k:{s:sorted({str(r[k]) for r in split['tasks'] if r['split']==s}) for s in ('train','val','test')}
                for k in ('mother_id','mother_source_sha256','seed','task_sha256')}
    overlap={k:{f'{a}/{b}':sorted(set(v[a])&set(v[b])) for a,b in [('train','val'),('train','test'),('val','test')]} for k,v in identities.items()}
    label_buckets=[]
    for key in keys:
        t,p,f=key; z=np.array([s['z_m'] for s in labels if (s['task_id'],s['preference'],s['family'])==key]).reshape(-1,12)
        distances=np.linalg.norm(z[:,None]-z[None,:],axis=-1)
        label_buckets.append(dict(task_id=t,preference=p,family=f,n=len(z),unique_z=len(np.unique(z,axis=0)),
            rank=int(np.linalg.matrix_rank(z-z.mean(0),tol=1e-10)),max_distance_m=float(distances.max()),
            separated_modes_established=False))
    csv_write(out/'label_buckets.csv',label_buckets)
    result=dict(train_condition_rows=len(keys),condition_dimension=norm.shape[1],varying_dimensions=int(varying.sum()),
        constant_dimensions=int((~varying).sum()),always_missing_dimensions=int((~valid.any(0)).sum()),
        partial_missing_dimensions=int(((valid.sum(0)>0)&(valid.sum(0)<len(valid))).sum()),
        nonfinite_values=int((~np.isfinite(raw)).sum()),centered_rank_at_1e_6=rank,
        near_collinear_pairs_abs_r_ge_099=len(pairs),feature_groups=dict(Counter(f['group'] for f in features)),
        label_count=len(labels),unique_task_preference_family_z=len(set(semantic)),
        unique_task_family_z=len({digest(dict(task=s['task_sha256'],family=s['family'],z=s['z_m'])) for s in labels}),
        unique_z_only=len({digest(s['z_m']) for s in labels}),source_counts=dict(source_counts),
        both_source_types=sum(len(s['source_types'])==2 for s in labels),
        label_family_preference=dict(Counter(s['preference']+'/'+s['family'] for s in labels)),
        split_identities=identities,split_overlaps=overlap,
        scaler_fit_task_sha256=ds.condition_scaler.fit_task_sha256,
        multimodality='NOT_ESTABLISHED: sparse near-optimal selected labels do not establish separated feasible basins')
    write(out/'data_audit.json',result)
    return result


def sample_diagnostics(ds,out):
    # Predeclared, all retained; no task-result filtering and no TEST calls.
    torch.set_num_threads(1)
    rng=np.random.default_rng(6441001)
    noises=rng.standard_normal((8,12)).astype(np.float32)
    np.save(out/'diagnostic_noises.npy',noises)
    d=load_frozen_sampler('D'); s=load_frozen_sampler('S')
    rows=[]; summaries=[]; outputs={}; repeats=0
    for tid in sorted(t for t in ds.tasks if ds.task_splits[t] in ('train','val')):
        task=ds.tasks[tid]
        for slot,p,f in [(1,'A','v1'),(3,'B','v2')]:
            values=[]
            labels=np.array([r['z_m'] for r in ds.samples if r['preference']==p and r['family']==f]).reshape(-1,12)
            for i,noise in enumerate(noises):
                raw,meta=d.sample(task,p,f,noise)
                if i==0:
                    again,_=d.sample(task,p,f,noise); repeats+=1
                    if not np.array_equal(raw,again): raise AssertionError('same-runtime deterministic sampling differs')
                plan,check=raw_seed_plan(task,dict(source='diffusion',family=f,preference=p,raw_z_m=raw),slot)
                values.append(raw.reshape(12)); norms=np.linalg.norm(raw,axis=1)
                rows.append(dict(task_id=tid,split=ds.task_splits[tid],preference=p,family=f,draw=i,
                    noise_sha256=meta['noise_sha256'],raw_z_m=raw.tolist(),raw_legal=check['raw_legal'],
                    rejection_reason=check.get('rejection_reason'),max_interval_norm_m=float(norms.max()),
                    active_total_norm_m=float(np.linalg.norm(raw)),nearest_matching_train_label_l2_m=float(np.linalg.norm(labels-raw.reshape(12),axis=1).min()),
                    inference_wall_s=meta['inference_wall_s']))
            values=np.array(values); outputs[tid,p,f]=values
            distances=np.linalg.norm(values[:,None]-values[None,:],axis=-1)[np.triu_indices(8,1)]
            reg,reg_meta=s.sample(task,p,f)
            _,reg_check=raw_seed_plan(task,dict(source='regression',family=f,preference=p,raw_z_m=reg),slot)
            summaries.append(dict(task_id=tid,split=ds.task_splits[tid],preference=p,family=f,
                legal=sum(r['raw_legal'] for r in rows[-8:]),draws=8,mean_pairwise_distance_m=float(distances.mean()),
                max_pairwise_distance_m=float(distances.max()),mean_output_m=values.mean(0).tolist(),
                regression_raw_z_m=reg.tolist(),regression_legal=reg_check['raw_legal'],
                regression_inference_s=reg_meta['inference_wall_s']))
    sensitivity=[]
    for p,f in [('A','v1'),('B','v2')]:
        tids=sorted(t for t,pp,ff in outputs if (pp,ff)==(p,f))
        for i,a in enumerate(tids):
            for b in tids[i+1:]:
                sensitivity.append(dict(task_a=a,task_b=b,preference=p,family=f,
                    same_noise_mean_l2_m=float(np.linalg.norm(outputs[a,p,f]-outputs[b,p,f],axis=1).mean())))
    csv_write(out/'offline_samples.csv',rows);csv_write(out/'offline_sampling_summary.csv',summaries)
    csv_write(out/'condition_sensitivity.csv',sensitivity)
    result=dict(seed=6441001,draws_per_condition=8,task_count=8,condition_count=16,diagnostic_outputs=128,
        repeated_determinism_checks=repeats,total_DDIM_calls=d.sample_units,total_S_forwards=s.sample_units,
        raw_legal=sum(r['raw_legal'] for r in rows),rejections=dict(Counter(r['rejection_reason'] for r in rows if not r['raw_legal'])),
        test_calls=0,new_training_updates=0,new_physics_steps=0,model_identity=d.identity,
        inference_source_checks=d.c4a_source_checks,torch_threads=torch.get_num_threads(),
        D_inference_median_s=float(np.median([r['inference_wall_s'] for r in rows])),
        S_inference_median_s=float(np.median([r['regression_inference_s'] for r in summaries])),
        no_selection_or_repair=True)
    write(out/'sampling_audit.json',result);return result


def forensics(out):
    snap=RELEASE/'snapshot';matrix=[];streams=[];cases={}
    for registry in sorted(snap.rglob('candidate_registry.json')):
        parts=registry.relative_to(snap).parts
        stage='test' if parts[0]=='test_search' else 'teacher' if parts[0]=='teacher_search' else 'val'
        root=registry.parent; rows=read(registry); proposals=read(root/'proposals.json'); selection=read(root/'selection.json')
        by_prop={r['proposal_index']:r for r in rows};tid=selection['task_id'];method=root.parent.parent.name
        if digest(rows)!=selection['registry_content_sha256']: raise AssertionError('registry seal changed')
        endpoints={}
        for n in (4,8,12):
            if (root/f'prefix_{n:02d}.json').exists():
                prefix=read(root/f'prefix_{n:02d}.json'); seal=prefix.pop('snapshot_content_sha256')
                assert digest(prefix)==seal and digest(rows[:prefix['budget']['slots_consumed']])==prefix['registry_content_sha256']
                endpoints[n]=prefix
        for i,p in enumerate(proposals):
            r=by_prop.get(i); evidence=r or next(x for x in rows if x['candidate_id']==p['cache_hit_candidate_id'])
            metrics=evidence.get('prediction_metrics') or {};diag=p.get('raw_seed_diagnostics') or {}
            raw=p.get('raw_z_m')
            if raw is None and evidence.get('plan'):
                raw=evidence['plan'].get('z_m')
            record=dict(stage=stage,task_id=tid,stream=method,proposal_index=i,slot_1based=rows.index(r)+1 if r else None,
                candidate_id=r['candidate_id'] if r else None,source=p['source'],family=p['family'],
                initial_position=next((x['initial_position'] for x in p.get('proposal_lineage',[]) if 'initial_position' in x),None),
                origin_source=p.get('origin_source'),parent_candidate_id=p.get('parent_candidate_id'),preference_center=p.get('preference_center'),
                raw_z_m=raw,x_m=p.get('x_m'),raw_legal=diag.get('raw_legal'),raw_rejection=diag.get('rejection_reason'),
                cache_hit_candidate_id=p.get('cache_hit_candidate_id'),status=evidence['status'] if r else 'LEGAL_EXACT_CACHE_HIT',
                evidence_candidate_id=evidence['candidate_id'],prediction_admissible=evidence.get('prediction_admissible'),
                prediction_steps=evidence.get('prediction_steps') if r else 0,
                I_support=metrics.get('I_support'),L_full=metrics.get('L_full'),d_support=metrics.get('d_support'),
                B30=bool(evidence.get('prediction_admissible') and metrics.get('d_support',-1)>=.03),
                base_translation_peak_m=metrics.get('base_translation_peak_m'),base_rotation_peak_rad=metrics.get('base_rotation_peak_rad'),
                costs=evidence.get('costs') if r else {},plan_sha256=evidence.get('plan_sha256'),
                source_path=str(registry.relative_to(RELEASE)),source_sha256=sha(registry),
                actual_status='NOT_EXECUTED_AS_CANDIDATE; selected endpoint evidence separate')
            for n,endpoint in endpoints.items():
                for pref in ('A','B'):
                    record[f'selected_at_{n}_{pref}']=bool(r and endpoint['preferences'][pref]['source_candidate_id']==r['candidate_id'])
                    record[f'endpoint_{n}_{pref}_status']=endpoint['preferences'][pref]['status']
            matrix.append(record)
        stream=dict(stage=stage,task_id=tid,method=method,slots=len(rows),proposals=len(proposals),
            qualified=sum(r['prediction_admissible'] for r in rows),B30=sum(r['prediction_admissible'] and (r.get('prediction_metrics') or {}).get('d_support',-1)>=.03 for r in rows),
            raw_rejected=sum(r['status']=='INITIALIZER_RAW_REJECTED' for r in rows),cache_hits=len(proposals)-len(rows),
            selection=selection['preferences'],source_sha256=sha(registry))
        streams.append(stream)
        if stage=='test' and method in ('R','D'):
            cases.setdefault(tid,{})[method]=dict(rows=rows,proposals=proposals,selection=selection,endpoints=endpoints)
    csv_write(out/'seed_comparison.csv',matrix);write(out/'stream_diagnostics.json',streams)
    write(out/'B_failure_evidence_chains.json',cases)
    for name in ('first_hits.json','first_hits.csv','per_task_endpoints.csv','stream_accounting.csv','cost_accounting.json','endpoint_summary.csv'):
        shutil.copyfile(snap/'result_tables'/name,out/('c3_'+name))
    result=dict(stream_count=len(streams),candidate_slots=sum(s['slots'] for s in streams),
        proposal_rows=len(matrix),cache_hits=sum(s['cache_hits'] for s in streams),
        stage_slots={stage:sum(s['slots'] for s in streams if s['stage']==stage) for stage in ('teacher','val','test')},
        new_TEST_runs=0,new_physics_steps=0)
    write(out/'forensic_summary.json',result);return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False)
    with patch.object(mujoco,'mj_step',side_effect=AssertionError('C4-A audit forbids physics')),patch.object(torch.optim.AdamW,'step',side_effect=AssertionError('C4-A forbids training')):
        verified=PortableResolver(RELEASE).verify_all()
        ds=load_search_aware_dataset(RELEASE/'snapshot/dataset/manifest.json')
        data=data_audit(ds,a.output);sample=sample_diagnostics(ds,a.output);forensic=forensics(a.output)
    inputs={str(p.relative_to(RELEASE)):sha(p) for p in RELEASE.rglob('*') if p.is_file()}
    write(a.output/'input_sha256.json',inputs)
    write(a.output/'audit_summary.json',dict(completed_utc=datetime.now(timezone.utc).isoformat(),
        portable_verified=verified,data=data,sampling=sample,forensics=forensic,
        verification_status='ANALYZED; numerical tests and same-runtime sampling VERIFIED separately',
        physics_steps=0,training_updates=0,test_samples=0))
    print(json.dumps(dict(status='PASS',data=data,sampling=sample,forensics=forensic)))


if __name__=='__main__': main()
