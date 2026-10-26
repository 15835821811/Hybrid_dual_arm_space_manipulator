"""Read checkpoint tensors/metadata only; no forward, sampler, training or physics."""
from pathlib import Path
import hashlib, json, datetime
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'ff_prior_guided_chain_plan/evidence/model_reuse.json'
def read(rel): return json.loads((ROOT/rel).read_text(encoding='utf-8-sig'))
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for part in iter(lambda:f.read(1048576),b''): h.update(part)
    return h.hexdigest()
def weights_hash(weights):
    h=hashlib.sha256()
    for name,value in weights.items():
        h.update(name.encode());h.update(value.detach().cpu().numpy().tobytes())
    return h.hexdigest()
static='artifacts/v7_1/sample_alignment_001/learning/aligned'
temporal='artifacts/v7_1/temporal_learning_001'
paths=[static+'/model.pt',static+'/codec.json',static+'/training.json',
       'artifacts/v7_1/control_transfer_bounded_001/codec.json',
       temporal+'/learning/temporal_diffusion/model.pt',temporal+'/codec.json',temporal+'/training.json',
       'ff_prior/coupled_predictor.py','ff_prior/contracts.py','ff_prior_latent/learning.py',
       'ff_prior_sample_alignment/learning.py','ff_prior_control_transfer/model.py',
       'ff_prior_static_fullchain_v2/proposals.py','ff_prior_temporal_learning/models.py',
       'ff_prior_temporal_learning/features.py','ff_prior_temporal_learning/representation.py',
       'ff_prior_control_transfer/docs/ANALYSIS.md','ff_prior_temporal_learning/docs/ANALYSIS.md',
       'ff_prior_momentum_diagnosis/evidence/budget_derivation.json',
       'ff_prior_momentum_diagnosis/evidence/repair_spec.json',
       'ff_prior_momentum_diagnosis/evidence/momentum_contract.json',
       'ff_prior_balanced_scene/evidence/layout.json']
files={p:dict(sha256=sha(ROOT/p),bytes=(ROOT/p).stat().st_size) for p in paths}
s=torch.load(ROOT/static/'model.pt',map_location='cpu',weights_only=False)
t=torch.load(ROOT/temporal/'learning/temporal_diffusion/model.pt',map_location='cpu',weights_only=False)
sr=read(static+'/training.json');tr=read(temporal+'/training.json')['methods']['temporal_diffusion']
sc=read(static+'/codec.json');tc=read(temporal+'/codec.json')
fc=read('artifacts/v7_1/control_transfer_bounded_001/codec.json')
checks=dict(static_checkpoint_hash=files[static+'/model.pt']['sha256']==sr['checkpoint_sha256'],
    static_metadata=s['metadata']==sr['metadata'],
    static_weights=weights_hash(s['weights'])==sr['trained_weight_hash'],
    temporal_weights=weights_hash(t['weights'])==tr['weight_hash'],
    temporal_condition_dim=t['condition_dim']==len(tc['global_mean'])==1014,
    temporal_output_shape=list(t['weights']['position'].shape)==[1,9,271] and list(tc['free_shape'])==[271,17],
    both_finite=all(torch.isfinite(v).all().item() for model in (s,t) for v in model['weights'].values()),
    static_low_basis_shape=np.asarray(sc['basis']).shape==(19,6),
    full_basis_shape=np.asarray(fc['basis']).shape==(19,19),
    temporal_codec_train_only=tc['fit_count']==8 and tc['fit_split']=='TRAIN',
    full_codec_train_only=fc['detail_fit_split']=='TRAIN')
assert all(checks.values()),checks
out=dict(status='READ_ONLY_STRUCTURE_AND_IDENTITY_VERIFIED_PHYSICAL_REUSE_NOT_VALIDATED',
    created_at=datetime.datetime.now().astimezone().isoformat(),checks=checks,files=files,
    static=dict(condition_dim=65,learned_output_dim=102,decoded_shape=[19,17],
        full_codec_dimension=323,learned_detail_dimension=0,unlearned_detail_dimension=221,
        weight_hash=weights_hash(s['weights']),parameters=sum(v.numel() for v in s['weights'].values()),
        training_rows=sr['TRAIN_rows_seen'],training_updates=sr['updates'],
        obstacle_conditioned=False,complete_initial_conditioned=False,
        role='frozen low102 prior with public initializer; full323 physics guidance',
        trained_full323_candidate_available=False,
        exclusion='control_transfer TRAIN diagnostic is not a qualified production checkpoint'),
    temporal=dict(condition_dim=1014,local_condition_shape=[271,36],free_output_shape=[271,17],
        decoded_shape=[273,17],duration_s=27,max_network_obstacles=5,
        weight_hash=weights_hash(t['weights']),parameters=tr['parameters'],
        training_rows=tr['TRAIN_rows'],training_updates=t['updates'],
        schema=tc['schema'],role='frozen trajectory prior; new adapters and physical qualification required',
        unsupported_existing_interfaces=['7 obstacle features','nonzero initial velocity prepare','29 second time basis',
          'zero-momentum static predictor cannot propagate actual dynamic initial state']),
    new_model_forward_calls=0,new_samples=0,optimizer_updates=0,new_data=0,
    native_steps=0,coupled_calls=0,geometry_queries=0,old_results_recomputed=False)
OUT.parent.mkdir(parents=True,exist_ok=True)
OUT.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(dict(status=out['status'],checks=checks,static_weight_hash=out['static']['weight_hash'],
 temporal_weight_hash=out['temporal']['weight_hash'],outputs=str(OUT)),ensure_ascii=False))

