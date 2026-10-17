"""Fresh reviewer: read saved evidence only; no training, inference or physics calls."""
from pathlib import Path
import json, hashlib, sys, collections, zipfile
import numpy as np
from scipy.spatial.transform import Rotation
import torch

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
OUT = ROOT/'artifacts/v7_1/world_endpoint_smoke_001'
PARENT = ROOT/'artifacts/v7_1/qualified_origin_learn_002'
def read(p): return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def digest(v):
    return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False,
        default=lambda x:x.tolist() if isinstance(x,np.ndarray) else x.item()).encode()).hexdigest()
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for chunk in iter(lambda:f.read(1048576),b''):h.update(chunk)
    return h.hexdigest()
def pose(a,b):
    a,b=np.asarray(a),np.asarray(b)
    pos=np.linalg.norm(a[...,:3,3]-b[...,:3,3],axis=-1)
    rel=np.swapaxes(b[...,:3,:3],-1,-2)@a[...,:3,:3]
    rot=Rotation.from_matrix(rel.reshape(-1,3,3)).magnitude().reshape(pos.shape)
    return pos,rot
def loss(a,b):
    p,r=pose(a,b)
    return ((p/.005)**2+(r/np.deg2rad(1))**2).sum(axis=-1)/4
def qstats(v):
    v=np.asarray(v);return dict(min=float(v.min()),median=float(np.median(v)),max=float(v.max()))
E={'scope':'read-only saved evidence recomputation; no training/inference/physics/replay rerun', 'checks':{}}
inputs=read(HERE/'inputs.json')['inputs']
E['checks']['input_hashes']={str(Path(x['path']).relative_to(ROOT)):sha(x['path'])==x['sha256'] for x in inputs}
assert all(E['checks']['input_hashes'].values())
run=read(OUT/'run.json');ledger=read(OUT/'ledger.json');summary=read(OUT/'summary.json')
E['producer_head']=run['producer_head']
with zipfile.ZipFile(OUT/'executed_source.zip') as z:
    source_hashes=run['identity']['source_hashes']
    assert all(sha(ROOT/p)==h==hashlib.sha256(z.read(p)).hexdigest() for p,h in source_hashes.items())
E['checks']['frozen_source_hashes_and_zip']=len(source_hashes)
assert ledger['inflight'] is None and not ledger.get('unresolved_call_accounting')
index=read(OUT/'training_index.json')['records'];old_index=read(PARENT/'training_index.json')['records']
assert index[:len(old_index)]==old_index
assert collections.Counter(x['split'] for x in index)=={'TRAIN':13838,'VAL':48}
training=read(OUT/'learning/goal_loss/training.json');meta=training['metadata']
ids=[x['sample_id'] for x in index if x['split']=='TRAIN']
assert ids==meta['training_ids'] and [x['sample_id'] for x in index if x['split']=='VAL']==meta['validation_ids']
assert not set(meta['training_ids'])&set(meta['validation_ids'])
assert not {x['family_id'] for x in index if x['split']=='TRAIN'}&{x['family_id'] for x in index if x['split']=='VAL'}
assert set(training['train_draw_counts'])==set(ids) and min(training['train_draw_counts'].values())>0
assert sum(training['train_draw_counts'].values())==384000
assert all(training['train_draw_counts'][sid]==training['uniform_row_draw_counts'][sid]+training['origin_stream_row_draw_counts'][sid] for sid in ids)
progress=torch.load(OUT/'learning/goal_loss/progress.pt',map_location='cpu',weights_only=False)
checkpoint=torch.load(OUT/'learning/goal_loss/model.pt',map_location='cpu',weights_only=False)
assert progress['updates']==6000 and progress['metadata']==meta==checkpoint['metadata']
assert all(int(s['step'].item())==6000 for s in progress['optimizer']['state'].values())
assert all(torch.equal(v,checkpoint['weights'][k]) for k,v in progress['weights'].items())
assert np.array_equal(progress['counts'],[training['train_draw_counts'][x] for x in ids])
assert sha(OUT/'learning/goal_loss/model.pt')==training['checkpoint_sha256']==read(OUT/'checkpoint_binding.json')['new']
assert sha(PARENT/'learning/goal_loss/model.pt')==meta['warm_start_checkpoint_sha256']
assert read(PARENT/'learning/goal_loss/codec.json')==read(OUT/'learning/goal_loss/codec.json')
E['training']={'updates':6000,'draws':384000,'rows':len(ids),'new_rows':len(meta['new_ids']),
    'all_row_draws':qstats(list(training['train_draw_counts'].values())),
    'new_row_draws':qstats([training['train_draw_counts'][s] for s in meta['new_ids']]),
    'new_row_total_draws':sum(training['train_draw_counts'][s] for s in meta['new_ids']),
    'optimizer_steps_verified':len(progress['optimizer']['state']), 'VAL_gradient_updates':training['VAL_gradient_updates'],
    'index_pretraining_consumed_flag':read(OUT/'training_index.json')['actual_training_consumed']}
assert training['VAL_gradient_updates']==0 and not training['VAL_checkpoint_selection']
targets=read(OUT/'fixed_TRAIN_targets.json')['entries'];newrows={p.stem:read(p) for p in (OUT/'records').glob('*.json')}
assert len(newrows)==64 and set(newrows)==set(meta['new_ids'])
bytarget={x['parent_sample_id']:x for x in targets}
assert len(bytarget)==16
newchecks=[]
for t in targets:
    inherited=next(x for x in old_index if x['sample_id']==t['parent_sample_id'])
    parentrow=read(ROOT/inherited['record']);assert parentrow['task']==t['task'] and inherited['split']=='TRAIN'
    refs=[np.asarray(t['Z'])]
    for sid,r in sorted(newrows.items()):
        if r['parent_sample_id']!=t['parent_sample_id']:continue
        assert r['task']==t['task'] and r['condition']==t['task']['condition'] and not r['goal_relabelled']
        assert r['training_eligible'] and r['split']=='TRAIN' and all(r['validation']['gates'].values())
        v=r['validation'];assert max(v['frozen_goal_position_error_m'])<=.005 and max(v['frozen_goal_rotation_error_rad'])<=np.deg2rad(1)
        assert sha(OUT/r['arrays'])==r['arrays_sha256'] and ledger['completed'][sid]['status']=='ADMITTED'
        with np.load(OUT/r['arrays']) as a:z=a['Z'].copy()
        gap=min(np.sqrt(np.mean((z-q)**2)) for q in refs);assert gap>=.005;refs.append(z);newchecks.append(gap)
selection=read(OUT/'data_audit_selection.json')
assert selection['before_execution'] and not selection['replacement_after_failure']
assert selection['ids']==[min((r for r in newrows.values() if r['parent_sample_id']==t['parent_sample_id']),key=lambda r:digest({'endpoint_witness':r['sample_id']}))['sample_id'] for t in targets]
E['new_data']={'rows':64,'targets':16,'validation_levels':dict(collections.Counter(r['validation_level'] for r in newrows.values())),
    'minimum_pairwise_control_rms_rad':min(newchecks),'targets_unchanged_from_parent':True,'selection_recomputed':True}
# Recompute all 120 full102 gradients directly from saved endpoint matrices.
halferr=[];direrr=[];dpall=[];drall=[];eventloss=[];gradnorm=[];pnorm=[];times=collections.Counter();targetcounts=collections.Counter();variant_times=collections.defaultdict(set)
for ev in training['world_records']:
    f=OUT/'world_gradients'/f"{ev['event']:03d}.json";p=read(f);r=newrows[p['sample_id']]
    assert sha(f)==ev['proof_sha256'] and sha(f.with_suffix('.npz'))==p['arrays_sha256']
    assert p['status']=='PASS' and p['task_hash']==r['task']['task_hash'] and p['training_only']
    assert p['input_weight_hash']==ev['input_weight_hash']!=ev['output_weight_hash']
    assert p['update']==ev['update']==50*(ev['event']+1)
    assert ledger['completed'][f"world_gradient_{ev['event']:03d}"]['status']=='PASS'
    with np.load(f.with_suffix('.npz')) as a:
        x=a['x'];g=a['gradient'];ls=a['perturbation_losses'];ee=a['perturbation_ee']
        assert x.shape==g.shape==(102,) and ee.shape==(102,2,2,4,4) and digest(x)==p['x_hash']
        assert np.isfinite(x).all() and np.isfinite(g).all()
        computed=loss(ee,r['condition']['goals'])
        assert np.allclose(computed,ls,rtol=1e-12,atol=1e-12)
        assert np.array_equal(g,(ls[:,1]-ls[:,0])/.02)
        assert np.isclose(loss(p['central']['ee'],r['condition']['goals']),p['true_loss'],rtol=1e-12,atol=1e-12)
        for h in p['half_checks']:
            slope=(loss(h['samples'][1]['ee'],r['condition']['goals'])-loss(h['samples'][0]['ee'],r['condition']['goals']))/.01
            error=abs(slope-g[h['dimension']])/max(abs(slope),abs(g[h['dimension']]),1e-6);halferr.append(error);assert error<=.02
        slope=(loss(p['direction_samples'][1]['ee'],r['condition']['goals'])-loss(p['direction_samples'][0]['ee'],r['condition']['goals']))/.02
        expected=g@np.asarray(p['direction']);error=abs(slope-expected)/max(abs(slope),abs(expected),1e-6);direrr.append(error);assert error<=.05
    dp,dr=pose(p['central']['ee'],p['convergence']['fine']['ee']);dpall.extend(dp);drall.extend(dr)
    assert max(dp)<=.001 and max(dr)<=np.deg2rad(.2)
    assert p['calls']['coupled']==212 and p['calls']['physics_steps']==0
    times[p['noise_t']]+=1;targetcounts[r['parent_sample_id']]+=1;variant_times[p['sample_id']].add(p['noise_t'])
    eventloss.append(p['true_loss']);gradnorm.append(ev['gradient_l2']);pnorm.append(ev['parameter_gradient_norm'])
assert len(eventloss)==120 and ledger['calls']['world_gradient.coupled']==25440
E['gradients']={'events':120,'dimensions':102,'saved_perturbation_losses_recomputed':24480,'coupled_calls':25440,
    'max_half_relative_error':float(max(halferr)),'max_direction_relative_error':float(max(direrr)),
    'max_256_512_position_delta_m':float(max(dpall)), 'max_256_512_rotation_delta_rad':float(max(drall)),
    'noise_times':dict(times),'per_target_visits':dict(targetcounts),'unique_rows':len(variant_times),
    'rows_with_both_noise_times':sum(len(v)==2 for v in variant_times.values()),
    'loss':qstats(eventloss),'latent_gradient_l2':qstats(gradnorm),'total_parameter_gradient_norm_before_clipping':qstats(pnorm),
    'events_gradient_norm_gt1':sum(x>1 for x in pnorm)}
# Frozen independent-noise TRAIN diagnostic, all 32 pairs.
before=read(OUT/'learning/goal_loss/world_diagnostic_before.json');after=read(OUT/'learning/goal_loss/world_diagnostic_after.json')
ratios=[];bytime=collections.defaultdict(list)
for a,b in zip(before['cases'],after['cases']):
    assert (a['sample_id'],a['t'],a['seed'])==(b['sample_id'],b['t'],b['seed'])
    goals=newrows[a['sample_id']]['condition']['goals']
    for v in (a,b):assert np.isclose(loss(v['ee'],goals),v['loss'],rtol=1e-12,atol=1e-12)
    ratio=b['loss']/a['loss'];ratios.append(ratio);bytime[a['t']].append(ratio)
assert len(ratios)==32 and float(np.median(ratios))==summary['TRAIN_endpoint_alignment']['median_paired_loss_ratio']
E['TRAIN_alignment']={'median_paired_loss_ratio':float(np.median(ratios)),'improved':sum(x<1 for x in ratios),'n':32,
    'by_t':{t:{'median_ratio':float(np.median(v)),'improved':sum(x<1 for x in v),'n':len(v)} for t,v in bytime.items()}}
entries=read(OUT/'frozen_tasks.json')['entries'];assert entries==read(PARENT/'frozen_tasks.json')['entries']
assert run['acceptance']==read(PARENT/'run.json')['acceptance']
methods=['structured','frozen_diffusion','learned_mean','expanded_diffusion'];seeds=[20262119,20262123,20262129]
stats={m:{'candidates':0,'eligible':0,'pre_pose_pass':0,'raw_goal':0,'raw_physics':0,'native':0,'max_queries':0,'max_intervals':0,'max_wall_s':0} for m in methods}
taskvals={};traces=[]
for e in entries:
    sid=e['sample_id'];taskvals[sid]={m:[] for m in methods}
    for sj,seed in enumerate(seeds):
        bases=set()
        for m in methods:
            records=[]
            for k in range(5):
                r=read(OUT/'candidates'/str(sj)/sid/m/f'{k:02d}.json');records.append(r)
                assert r['status']=='DONE' and r['task_hash']==e['task']['task_hash'] and r['seed']==seed+e['comparison_slot']*101+k*997
                assert r['method']==m and r['candidate_id']==k and r['eligible'] and not r['private_witness_used']
                assert r['model_calls']['coupled']<=113 and r['integration_intervals']<=15872 and r['wall_s']<=15
                assert all(r['validation']['gates'].values()) and r['frozen_goal_pass']
                assert r['actual_network_calls']==(16 if 'diffusion' in m else 0)
                if 'diffusion' in m:assert [s['timestep'] for s in r['sampling']['steps']]==np.linspace(999,0,16).round().astype(int).tolist()
                norm=float(np.linalg.norm(np.asarray(r['sampling']['guidance'][-1]['initial_goal_error']).reshape(4,3),axis=1).max())
                taskvals[sid][m].append(norm);s=stats[m];s['candidates']+=1;s['eligible']+=int(r['eligible']);s['pre_pose_pass']+=int(norm<=1)
                s['max_queries']=max(s['max_queries'],r['model_calls']['coupled']);s['max_intervals']=max(s['max_intervals'],r['integration_intervals']);s['max_wall_s']=max(s['max_wall_s'],r['wall_s']);bases.add(r['public_base_hash'])
            chosen=min(records,key=lambda r:(r['score'],r['candidate_id']));d=OUT/'executions'/str(sj)/sid/m
            selection=read(d/'selection.json');assert selection['selected_id']==chosen['candidate_id'] and selection['selection_before_execution']
            assert read(d/'outcome.json')['status']=='EXECUTION_VERIFIED';stats[m]['native']+=1;traces.append((d,e['task'],chosen['reference_hash']))
        assert len(bases)==1
    for m in methods:
        for stage in ('raw','smoothed_clipped'):
            d=read(OUT/'proposal_diagnostics'/sid/m/f'{stage}.json');assert d['status']=='DONE' and not d['affects_selection']
            assert d['task_hash']==e['task']['task_hash'] and sha(OUT/'candidates'/'0'/sid/m/'00.json')==d['source_candidate_sha256']
            v=d['validation'];goal=max(v['frozen_goal_position_error_m'])<=.005 and max(v['frozen_goal_rotation_error_rad'])<=np.deg2rad(1)
            physics=all(v['gates'].values());assert d['eligible']==(goal and physics)
            if stage=='raw':stats[m]['raw_goal']+=int(goal and physics);stats[m]['raw_physics']+=int(physics)
E['methods']=stats;E['paired_reused_DEV8']={}
reuse=[e['sample_id'] for e in entries if e['split']=='NEW_VAL']
for m in methods[:-1]:
    ratios=np.array([np.median(taskvals[s]['expanded_diffusion'])/np.median(taskvals[s][m]) for s in reuse])
    rng=np.random.default_rng(20262171);boot=np.exp(np.log(ratios)[rng.integers(0,8,(10000,8))].mean(axis=1));ci=np.quantile(boot,[.025,.975]).tolist()
    assert np.isclose(np.median(ratios),summary['reused_DEV8_paired_quality'][m]['median_task_ratio'])
    assert np.allclose(ci,summary['reused_DEV8_paired_quality'][m]['bootstrap95_geometric_ratio'])
    E['paired_reused_DEV8'][m]={'median_ratio':float(np.median(ratios)),'geometric_ratio':float(np.exp(np.log(ratios).mean())),
        'bootstrap95':ci,'new_lower':int(sum(ratios<1)),'new_higher':int(sum(ratios>1)),'task_ratios':dict(zip(reuse,ratios.tolist()))}
for sid in read(OUT/'data_audit_selection.json')['ids']:
    r=newrows[sid];traces.append((OUT/'TRAIN_executions'/sid,r['task'],r['reference_hash']))
steps=0;maxhold=0.;minclear=float('inf');maxreplay=0.
for d,task,ref in traces:
    p=read(d/'report.json');assert p['metadata']['reference_hash']==ref and read(d/'frozen_task.json')==task
    assert p['level']=='EXECUTION_VERIFIED' and all(p['gates'].values()) and not p['hardware_validated']
    assert p['metadata']['robot_state_writes_during_step_loop']==0 and not p['metadata']['external_wrench_applied']
    assert p['replay']['passed'] and not p['replay']['copies_actual_state_each_step']
    maxreplay=max(maxreplay,max(p['replay']['maximum_errors'].values()));assert maxreplay<=1e-10
    with np.load(d/'native_trace.npz') as a:
        ctrl=a['ctrl'];time=a['time'];ee=a['ee'];qpos=a['qpos'];clearance=a['clearance_m']
        assert ctrl.shape==(5250,67) and qpos.shape==(5251,81) and np.isfinite(ctrl).all() and np.isfinite(qpos).all()
        assert digest(ctrl)==p['ctrl_hash'];pos,rot=pose(ee,task['condition']['goals']);hold=time-time[0]>=task['condition']['duration']-1e-9
        assert np.allclose(pos[hold].max(axis=0),p['hold_max_position_error_m'],rtol=0,atol=1e-12)
        assert np.allclose(rot[hold].max(axis=0),p['hold_max_rotation_error_rad'],rtol=0,atol=1e-12)
        assert pos[hold].max()<=.005 and rot[hold].max()<=np.deg2rad(1) and clearance.min()>=0
        assert p['replay']['physics_steps']==len(ctrl);steps+=2*len(ctrl);maxhold=max(maxhold,float(pos[hold].max()));minclear=min(minclear,float(clearance.min()))
assert len(traces)==160 and steps==ledger['calls']['execution.physics_steps']+ledger['calls']['TRAIN_execution.physics_steps']
E['native_saved_trace_audit']={'traces':len(traces),'comparison_traces':144,'TRAIN_traces':16,'physics_steps_including_reported_independent_replay':steps,
    'max_hold_position_m':maxhold,'minimum_saved_state_clearance_m':minclear,'max_reported_replay_error':maxreplay,
    'scope':'all160 ctrl hash/shape/finite/EE hold/clearance arrays; reported independent replay checked, not rerun; no fresh dynamics or all-geometric-pairs reconstruction'}
E['limitations']=['Intermediate model weights are not separately retained for every event; x at current prediction supported by frozen source and per-event input hash, not full per-event inference replay.',
    'Existing inherited source-array/initializer audit accepted; only parent index and new records/arrays independently checked here, not all176k sourcefiles.',
    'Saved reports establish independent replay result; this review does not rerun replay physics.',
    'No separate parameter-gradient contribution or cosine logged for v and endpoint terms; cannot establish dominance, conflict, or causal cause.',
    'Eight comparison tasks labelled NEW_VAL in raw paths are reused development tasks; all statistical claims are conditional on these tasks and one training seed.']
E['TEST_gates']=summary['conditional_TEST_gates'];E['status']='PASS'
(HERE/'verification_evidence.json').write_text(json.dumps(E,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(E,ensure_ascii=False,indent=2))
