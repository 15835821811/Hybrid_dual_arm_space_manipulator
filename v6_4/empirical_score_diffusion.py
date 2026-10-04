"""Finite empirical conditional score diffusion, separate from Transformer B.

The support is thirteen TRAIN route offsets around a learned conditional affine
mean, not learned success covariance or new route topology. A linear softmax
and one least-norm conditional mean are fitted. Twenty unclipped DDIM steps never inspect
Task IDs to choose components and never snap, repair, project or emit torques.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import sys
import time

import numpy as np
import torch

from . import prior_diffusion as prior
from .contracts import TrajectoryProposal
from .dataset import encode_condition, load_teacher_dataset, object_sha, sha256
from .task_protocol import TaskSpec

METHOD = "teacher_prior_empirical_score_diffusion"
REPRESENTATION = "teacher_prior_empirical_residual_32x17_v1"
CONDITION_SCHEMA = "initial517_plus_physical_prior510_float64_v1"
CHECKPOINT_SCHEMA = "v6_4_empirical_conditional_score_checkpoint_v1"
NORMALIZER_SCHEMA = "v6_4_empirical_train_only_residual_normalizer_v1"
REPO = prior.REPO
DIM = 510


def source_identity():
    value = prior.source_identity()
    value["sources_sha256"]["v6_4/empirical_score_diffusion.py"] = sha256(Path(__file__))
    value["scope"] = "empirical conditional score C, frozen B utilities/prior/gates; actual physics separate"
    return value


def semantic_identity():
    value = {"method": METHOD, "representation": REPRESENTATION,
             "condition_schema": CONDITION_SCHEMA, "condition_dim": 1027,
             "condition_features": "TRAIN-normalized initial517 plus physical prior510; no Task ID/hash lookup",
             "training": "zero-initialized six-group softmax CE plus one mean-centered least-norm affine mean fit on six unique TRAIN groups",
             "component_prior": "softmax(group|features)/TRAIN label count in group",
             "support": "conditional affine mean plus thirteen within-group TRAIN residual offsets; no estimated success covariance",
             "conditional_mean": "Tmean+(features-Fmean)@pinv(F_center,rcond1e-12)@T_center; unique TRAIN groups equally weighted",
             "forward": "sum pi_m N(sqrt(alpha_bar)*mu_m,(1-alpha_bar)*I)",
             "denoising": "analytic posterior mean; epsilon=(x-sqrt(alpha_bar)*posterior_mean)/sqrt(1-alpha_bar)",
             "sampling": "float64 cosine100 DDIM20 eta0; final next-alpha1 uses same posterior mean formula",
             "restoration": "prior_free + TRAIN residual_mean17 + scale17*unclipped_normalized_residual",
             "postprocessing": [], "snap_argmax_nearest_component": False,
             "legacy_checkpoint_reinterpreted": False,
             "scope": "condition-translated finite TRAIN route offsets, no new topology; DDIM20 is not exact empirical-distribution sampling"}
    return {**value, "semantic_sha256": object_sha(value)}


def cosine_schedule():
    times = np.arange(101, dtype=np.float64)/100.
    curve = np.cos((times+.008)/1.008*np.pi/2.)**2
    betas = np.minimum(1.-curve[1:]/curve[:-1], .999)
    alpha_bars = np.cumprod(1.-betas)
    return betas, alpha_bars


def schedule_identity():
    betas, alphas = cosine_schedule()
    return {"steps": 100, "name": "cosine", "s": .008, "beta_cap": .999,
            "dtype": "float64", "betas_sha256": _array_sha(betas),
            "alpha_bars_sha256": _array_sha(alphas), "terminal_alpha_bar": float(alphas[-1]),
            "ddim_steps": 20, "eta": 0., "indices": np.rint(np.linspace(99, 0, 20)).astype(int).tolist()}


def _array_sha(value):
    return hashlib.sha256(np.asarray(value, dtype="<f8").tobytes(order="C")).hexdigest()


def _logsumexp(value, axis=-1):
    maximum = np.max(value, axis=axis, keepdims=True)
    return np.squeeze(maximum+np.log(np.sum(np.exp(value-maximum), axis=axis, keepdims=True)), axis=axis)


def _vector(value, size, name):
    value = np.asarray(value, dtype=np.float64)
    if value.shape != (size,) or not np.all(np.isfinite(value)):
        raise ValueError(f"finite {name} vector of length {size} required")
    return value


def condition_features(normalizer, task, center):
    if task.sha256() != center.task.sha256():
        raise ValueError("immutable condition/center mismatch")
    original = _vector(encode_condition(task), 517, "initial condition")
    # Reuse exactly B's TRAIN statistics, avoiding B's float32 conversion.
    result = np.r_[(original-normalizer.condition_mean)/normalizer.condition_scale,
                   np.asarray(center.free, dtype=np.float64).reshape(DIM)]
    return _vector(result, 1027, "empirical condition")


def normalize_residual(normalizer, values):
    values = np.asarray(values, dtype=np.float64)
    if values.shape[-2:] != (30, 17) or not np.all(np.isfinite(values)):
        raise ValueError("finite 30x17 residual required")
    return (values-normalizer.residual_mean)/normalizer.residual_scale


def restore_residual(normalizer, normalized, center):
    normalized = np.asarray(normalized, dtype=np.float64)
    if normalized.shape[-2:] != (30, 17) or not np.all(np.isfinite(normalized)):
        raise ValueError("finite unclipped 30x17 normalized residual required")
    residual = normalized*normalizer.residual_scale+normalizer.residual_mean
    return center.free+residual, residual


def fit_conditional_means(features, means, component_groups, group_count):
    features,means=np.asarray(features,dtype=np.float64),np.asarray(means,dtype=np.float64)
    component_groups=np.asarray(component_groups)
    if (features.shape!=(len(means),1027) or means.shape!=(len(component_groups),DIM)
            or set(component_groups.tolist())!=set(range(group_count))
            or not np.all(np.isfinite(features)) or not np.all(np.isfinite(means))):
        raise ValueError("finite paired TRAIN features/means/groups required")
    group_features=[];group_targets=[]
    for group in range(group_count):
        rows=features[component_groups==group]
        if not np.all(rows==rows[0]): raise ValueError("one immutable condition per TRAIN source group required")
        group_features.append(rows[0]);group_targets.append(means[component_groups==group].mean(0))
    group_features,group_targets=np.stack(group_features),np.stack(group_targets)
    feature_mean,target_mean=group_features.mean(0),group_targets.mean(0)
    fc,tc=group_features-feature_mean,group_targets-target_mean
    affine=np.linalg.pinv(fc,rcond=1e-12)@tc
    offsets=means-group_targets[component_groups]
    fitted=target_mean+fc@affine
    singular=np.linalg.svd(fc,compute_uv=False)
    rank=int(np.sum(singular>singular[0]*1e-12)) if singular[0]>0 else 0
    stats={"unique_TRAIN_groups":group_count,"rcond":1e-12,"ridge":None,"closed_form_fits":1,
           "centered_feature_rank":rank,"centered_feature_singular_values":singular.tolist(),
           "TRAIN_group_mean_fit_max_abs_normalized":float(np.max(abs(fitted-group_targets))),
           "TRAIN_group_mean_fit_RMSE_normalized":float(np.sqrt(np.mean((fitted-group_targets)**2))),
           "scope":"conditional mean translation plus finite within-group route offsets; no new route topology"}
    return affine,feature_mean,target_mean,offsets,stats


class EmpiricalConditionalScore:
    """Analytic Gaussian mixture; prediction consumes numerical features only."""
    def __init__(self, offsets, component_groups, weight, bias, affine=None, feature_mean=None, target_mean=None):
        self.offsets = np.asarray(offsets, dtype=np.float64)
        self.affine=np.zeros((1027,DIM),dtype=np.float64) if affine is None else np.asarray(affine,dtype=np.float64)
        self.feature_mean=np.zeros(1027,dtype=np.float64) if feature_mean is None else _vector(feature_mean,1027,"affine feature mean")
        self.target_mean=np.zeros(DIM,dtype=np.float64) if target_mean is None else _vector(target_mean,DIM,"affine target mean")
        self.component_groups = np.asarray(component_groups)
        self.weight, self.bias = np.asarray(weight, dtype=np.float64), np.asarray(bias, dtype=np.float64)
        if (self.offsets.ndim != 2 or self.offsets.shape[1] != DIM or self.offsets.shape[0] < 1
                or self.component_groups.shape != (self.offsets.shape[0],) or self.affine.shape!=(1027,DIM)
                or self.component_groups.dtype.kind not in "iu"
                or self.weight.ndim != 2 or self.weight.shape[1] != 1027
                or self.bias.shape != (self.weight.shape[0],)
                or set(self.component_groups.tolist()) != set(range(self.weight.shape[0]))
                or not all(np.all(np.isfinite(a)) for a in (self.offsets,self.affine,self.weight,self.bias))):
            raise ValueError("invalid empirical prototype/group/classifier arrays")
        self.group_counts = np.bincount(self.component_groups, minlength=len(self.bias))

    def component_means(self,features):
        features=_vector(features,1027,"condition")
        return self.target_mean+(features-self.feature_mean)@self.affine+self.offsets

    def group_log_probabilities(self, features):
        features = _vector(features, 1027, "condition")
        logits = self.weight@features+self.bias
        if not np.all(np.isfinite(logits)): raise ValueError("nonfinite conditional logits")
        return logits-_logsumexp(logits)

    def component_log_probabilities(self, features):
        return self.group_log_probabilities(features)[self.component_groups]-np.log(self.group_counts[self.component_groups])

    def posterior(self, state, alpha_bar, features):
        state = _vector(state, DIM, "diffusion state")
        if not np.isfinite(alpha_bar) or not 0 < alpha_bar < 1:
            raise ValueError("posterior requires 0 < alpha_bar < 1")
        signal, variance = np.sqrt(alpha_bar), 1.-alpha_bar
        means=self.component_means(features)
        delta = state[None]-signal*means
        logits = self.component_log_probabilities(features)-np.sum(delta*delta, axis=1)/(2.*variance)
        if not np.all(np.isfinite(logits)): raise ValueError("nonfinite posterior logits")
        normalization = _logsumexp(logits)
        weights = np.exp(logits-normalization)
        clean = weights@means
        epsilon = (state-signal*clean)/np.sqrt(variance)
        entropy = -float(np.sum(weights[weights > 0]*np.log(weights[weights > 0])))
        return {"weights": weights, "mean": clean, "epsilon": epsilon,
                "score": (signal*clean-state)/variance,
                "log_density": float(normalization-.5*DIM*np.log(2.*np.pi*variance)),
                "entropy": entropy}

    def sample_ddim(self, features, *, seed=64, initial_noise=None):
        if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
            raise ValueError("nonnegative integer seed required")
        _vector(features, 1027, "condition")
        state = np.random.default_rng(seed).standard_normal(DIM) if initial_noise is None else _vector(initial_noise, DIM, "initial noise").copy()
        initial = state.copy(); _, alphas = cosine_schedule()
        indices = np.asarray(schedule_identity()["indices"])
        trace = {key: [] for key in ("state_before", "state_after", "posterior_weights", "posterior_mean", "epsilon", "entropy", "alpha_bar", "next_alpha_bar")}
        for i, index in enumerate(indices):
            alpha = float(alphas[index]); next_alpha = float(alphas[indices[i+1]]) if i+1 < len(indices) else 1.
            denoised = self.posterior(state, alpha, features)
            # Including the last step: no hard component selection or snap.
            next_state = np.sqrt(next_alpha)*denoised["mean"]+np.sqrt(1.-next_alpha)*denoised["epsilon"]
            if not np.all(np.isfinite(next_state)): raise ValueError("nonfinite DDIM state")
            for key, value in (("state_before", state.copy()), ("state_after", next_state.copy()),
                               ("posterior_weights", denoised["weights"]), ("posterior_mean", denoised["mean"]),
                               ("epsilon", denoised["epsilon"]), ("entropy", denoised["entropy"]),
                               ("alpha_bar", alpha), ("next_alpha_bar", next_alpha)):
                trace[key].append(value)
            state = next_state
        return state, {**{k: np.asarray(v, dtype=np.float64) for k,v in trace.items()},
                       "initial_noise": initial, "indices": indices, "component_prior": np.exp(self.component_log_probabilities(features))}


def fit_linear_classifier(features, group_indices, group_count, *, steps=2000, seed=71, log_stream=None):
    """Real full-batch CE optimization; no diffusion-noise training or EMA."""
    features = np.asarray(features, dtype=np.float64); groups = np.asarray(group_indices)
    if (isinstance(group_count,bool) or not isinstance(group_count,int) or group_count<1
            or isinstance(seed,bool) or not isinstance(seed,int) or seed<0
            or features.ndim != 2 or features.shape[1] != 1027 or groups.shape != (features.shape[0],)
            or groups.dtype.kind not in "iu" or set(groups.tolist()) != set(range(group_count))
            or not np.all(np.isfinite(features)) or isinstance(steps, bool) or not isinstance(steps, int) or steps <= 0):
        raise ValueError("finite TRAIN feature/group full batch and positive optimizer steps required")
    torch.set_num_threads(4); torch.manual_seed(seed)
    model = torch.nn.Linear(1027, group_count, dtype=torch.float64, device="cpu")
    with torch.no_grad(): model.weight.zero_(); model.bias.zero_()
    optimizer = torch.optim.Adam(model.parameters(), lr=.01, betas=(.9,.999), weight_decay=0.)
    x, targets = torch.from_numpy(features), torch.from_numpy(groups.astype(np.int64))
    first = last = None; start = time.perf_counter()
    for step in range(1, steps+1):
        optimizer.zero_grad(set_to_none=True); loss = torch.nn.functional.cross_entropy(model(x), targets)
        if not torch.isfinite(loss): raise ValueError("nonfinite conditional CE loss")
        loss.backward(); optimizer.step(); last = float(loss.detach()); first = last if first is None else first
        if log_stream is not None:
            log_stream.write(json.dumps({"optimizer_step":step,"cross_entropy":last,"full_batch_labels":len(groups),"elapsed_s":time.perf_counter()-start},allow_nan=False)+"\n");log_stream.flush()
    with torch.no_grad():
        final_loss = float(torch.nn.functional.cross_entropy(model(x), targets))
        fitted = int((model(x).argmax(1)==targets).sum())
    return model.weight.detach().numpy().copy(), model.bias.detach().numpy().copy(), {"optimizer_steps":steps,"first_cross_entropy":first,"last_logged_preupdate_cross_entropy":last,"final_cross_entropy":final_loss,"TRAIN_group_accuracy_count":fitted,"TRAIN_labels":len(groups),"elapsed_s":time.perf_counter()-start}


def fit_empirical_score(teacher_manifest, prior_manifest, output, *, optimizer_steps=2000,
                        seed=71, expected_samples=13, expected_groups=6, protected_inputs=()):
    before = source_identity(); dataset = load_teacher_dataset(Path(teacher_manifest))
    if len(dataset.samples) != expected_samples or any(s["split"] != "train" for s in dataset.samples):
        raise ValueError("retain every fixed successful TRAIN label; no VAL/TEST fitting")
    tasks = {}
    for sample in dataset.samples:
        task = TaskSpec.from_dict(sample["task"])
        if task.task_id in tasks and tasks[task.task_id].sha256() != task.sha256(): raise ValueError("aliased immutable Task")
        tasks[task.task_id] = task
    groups = list(dict.fromkeys(tasks[s["task_id"]].group_id for s in dataset.samples))
    if len(tasks) != expected_groups or len(groups) != expected_groups: raise ValueError("all original TRAIN groups required")
    centers, prior_files = prior.load_centers(Path(prior_manifest), list(tasks.values()))
    contract, assets = prior._model_contract_and_assets()
    if any(t.model_contract_sha256 != contract for t in tasks.values()): raise ValueError("nominal model contract changed")
    inputs = set(dataset.source_files)|prior_files|set(assets)|{Path(p).resolve() for p in protected_inputs}
    input_before = prior._files(inputs); output = Path(output).resolve()
    if any(p.is_relative_to(output) for p in inputs) or Path(__file__).resolve().is_relative_to(output): raise ValueError("output contains frozen input/producer")
    output.mkdir(parents=True, exist_ok=False)
    normalizer = prior.ResidualNormalizer.fit(dataset, centers)
    center_array = np.stack([centers[s["task_id"]].free for s in dataset.samples])
    residuals = dataset.controls-center_array
    means = normalize_residual(normalizer, residuals).reshape(expected_samples,DIM)
    features = np.stack([condition_features(normalizer,tasks[s["task_id"]],centers[s["task_id"]]) for s in dataset.samples])
    component_groups = np.asarray([groups.index(tasks[s["task_id"]].group_id) for s in dataset.samples],dtype=np.int64)
    affine,feature_mean,target_mean,offsets,affine_fit=fit_conditional_means(features,means,component_groups,expected_groups)
    prototypes = [{"sample_id":s["sample_id"],"task_id":s["task_id"],"task_sha256":s["task_sha256"],"group_id":tasks[s["task_id"]].group_id,"component_group_index":int(g),"inference_role":"source metadata only, not an inference lookup key"} for s,g in zip(dataset.samples,component_groups)]
    plan = {"schema":CHECKPOINT_SCHEMA+"_fit_plan","semantics":semantic_identity(),"method":METHOD,
            "optimizer_steps":optimizer_steps,"seed":seed,"optimizer":"zero-init float64 CPU Adam","learning_rate":.01,"betas":[.9,.999],"weight_decay":0.,
            "batch_labels":expected_samples,"train_groups":expected_groups,"noise_training_iterations":0,"EMA":False,
            "weight_selection":"last2000CEiterate_only" if optimizer_steps==2000 else "last_declared_CE_iterate_only",
            "training_objective":"six-group conditional CE plus one mean-centered least-norm affine fit; empirical route offsets from all TRAIN labels",
            "conditional_mean_fit":affine_fit,
            "schedule":schedule_identity(),"source_before":before,"inputs_before":input_before,
            "teacher_manifest":str(Path(teacher_manifest).resolve()),"teacher_manifest_sha256":dataset.manifest_sha256,
            "prior_manifest":str(Path(prior_manifest).resolve()),"prior_manifest_sha256":sha256(Path(prior_manifest)),
            "all_labels_retained":True,"failed_center_excluded":False,"center_costs_separate":{k:c.identity for k,c in centers.items()},
            "prototypes":prototypes,"model_contract_sha256":contract,"VAL_TEST_data_used":False,"postprocessing":[]}
    normalizer_value={"schema":NORMALIZER_SCHEMA,"statistics":normalizer.to_dict(),"calculation_dtype":"float64","fit_VAL_TEST":False}
    prior._write(output/"config.json",plan);prior._write(output/"normalizer.json",normalizer_value);prior._write(output/"split.json",dataset.split)
    np.savez_compressed(output/"paired_training_data.npz",absolute_labels=dataset.controls,centers=center_array,residuals=residuals,means=means,features=features,component_groups=component_groups,offsets=offsets)
    archive=output/"source_snapshot"
    for relative,expected in before["sources_sha256"].items():
        target=archive/relative;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(REPO/relative,target)
        if sha256(target)!=expected: raise ValueError("source changed during archive")
    prior._write(archive/"receipt.json",{"source_identity":before,"copied_before_optimization":True})
    prior._write(output/"environment.json",{"python":sys.version,"numpy":np.__version__,"torch":str(torch.__version__),"device":"cpu","dtype":"float64"})
    start=time.perf_counter(); fit=None
    try:
        with (output/"training_log.jsonl").open("x",encoding="utf-8") as stream:
            weight,bias,fit=fit_linear_classifier(features,component_groups,expected_groups,steps=optimizer_steps,seed=seed,log_stream=stream)
        after=source_identity()
        if not prior._same_source(before,after) or prior._files(inputs)!=input_before or prior._model_contract_and_assets()[0]!=contract:
            raise ValueError("source/HEAD/assets/prior/teacher/history changed during fit")
        betas,alphas=cosine_schedule()
        np.savez_compressed(output/"checkpoint.npz",classifier_weight=weight,classifier_bias=bias,affine=affine,feature_mean=feature_mean,target_mean=target_mean,offsets=offsets,component_groups=component_groups,betas=betas,alpha_bars=alphas)
        companions={name:sha256(output/name) for name in ("config.json","normalizer.json","split.json","paired_training_data.npz")}
        checkpoint={"schema":CHECKPOINT_SCHEMA,"semantics":semantic_identity(),"method":METHOD,
                    "weights_file":"checkpoint.npz","weights_sha256":sha256(output/"checkpoint.npz"),
                    "normalizer":normalizer_value,"schedule":schedule_identity(),"split":dataset.split,"prototypes":prototypes,
                    "source_identity":before,"model_contract_sha256":contract,"fit":fit,"conditional_mean_fit":affine_fit,"seed":seed,"companion_hashes":companions,
                    "VAL_TEST_data_used":False,"weights_selection":plan["weight_selection"]}
        prior._write(output/"checkpoint.json",checkpoint)
        report={"schema":"v6_4_empirical_conditional_fit_report_v1","complete":True,"evidence_valid":True,
                "method":METHOD,"semantics":semantic_identity(),"fit":fit,"conditional_mean_fit":affine_fit,"source_before":before,"source_after":after,
                "source_unchanged":True,"inputs_unchanged":True,"all_labels_retained":True,"train_samples":expected_samples,"train_groups":expected_groups,
                "noise_training_iterations":0,"empirical_components":expected_samples,"checkpoint_sha256":sha256(output/"checkpoint.json"),
                "weights_sha256":sha256(output/"checkpoint.npz"),"elapsed_s":time.perf_counter()-start,"physics_steps":0,"test_used":False,
                "scope":"conditional affine mean plus empirical TRAIN route offsets; classifier CE and one least-norm fit are not DDPM noise training, success covariance, new topology, or raw/actual qualification"}
        prior._write(output/"fit_report.json",report)
        paths=sorted(p for p in output.rglob("*") if p.is_file())
        prior._write(output/"artifact_manifest.json",{"files":[{"path":p.relative_to(output).as_posix(),"sha256":sha256(p),"bytes":p.stat().st_size} for p in paths]})
        return report
    except BaseException as exc:
        prior._write(output/"fit_failure.json",{"complete":False,"evidence_valid":False,"error":f"{type(exc).__name__}: {exc}","partial_retained":True,"physics_steps":0})
        raise


class EmpiricalScoreSampler:
    def __init__(self,path,checkpoint,model,normalizer):
        self.path,self.checkpoint,self.model,self.normalizer=path,checkpoint,model,normalizer
        self.checkpoint_sha256=sha256(path)
        self.model_files={path,path.parent/"checkpoint.npz",*(path.parent/name for name in checkpoint["companion_hashes"])}
        self.model_files_identity=prior._files(self.model_files)

    def sample(self,task,center,*,K=1,seed=64):
        if K not in (1,8) or isinstance(K,bool): raise ValueError("fixed K1/K8 required")
        before=source_identity()
        if prior._files(self.model_files)!=self.model_files_identity: raise ValueError("frozen C model inputs changed before sampling")
        if not prior._same_source(before,self.checkpoint["source_identity"]): raise ValueError("sampling source/HEAD differs from frozen C")
        if task.model_contract_sha256!=self.checkpoint["model_contract_sha256"] or prior._model_contract_and_assets()[0]!=task.model_contract_sha256:
            raise ValueError("sampling nominal model contract differs")
        current=prior.load_center(center.row,task)
        if not np.array_equal(current.free,center.free): raise ValueError("center changed")
        features=condition_features(self.normalizer,task,center)
        raws,residuals,traces,rows=[],[],[],[];start=time.perf_counter()
        for i in range(K):
            normalized,trace=self.model.sample_ddim(features,seed=seed+i)
            absolute,residual=restore_residual(self.normalizer,normalized.reshape(30,17),center)
            distances=np.sqrt(np.mean((self.model.component_means(features)-normalized[None])**2,axis=1))
            final_weights=trace["posterior_weights"][-1]
            rows.append({"candidate_index":i,"seed":seed+i,"final_posterior_entropy":float(trace["entropy"][-1]),
                         "final_max_component_posterior":float(np.max(final_weights)),"final_min_normalized_prototype_RMSE":float(np.min(distances)),
                         "near_single_condition_translated_TRAIN_offset_within_1e_10_normalized_RMSE":bool(np.min(distances)<=1e-10),
                         "center_equivalent_exact":bool(np.array_equal(absolute,center.free)),"trace_sha256":{k:_array_sha(v) for k,v in trace.items()}})
            raws.append(absolute);residuals.append(residual);traces.append(trace)
        after=source_identity()
        if not prior._same_source(before,after) or prior._files(self.model_files)!=self.model_files_identity or prior._model_contract_and_assets()[0]!=task.model_contract_sha256:
            raise ValueError("source/HEAD/weights/assets changed during C generation")
        prior.load_center(center.row,task)
        metadata={"method":METHOD,"semantics":semantic_identity(),"checkpoint_sha256":self.checkpoint_sha256,
                  "task_sha256":task.sha256(),"prior":center.identity,"postprocessing":[],"repair":False,
                  "source_before":before,"source_after":after,"sampling_elapsed_s":time.perf_counter()-start,"candidates":rows,
                  "condition_group_probabilities":np.exp(self.model.group_log_probabilities(features)).tolist(),
                  "component_prior_probabilities":np.exp(self.model.component_log_probabilities(features)).tolist(),
                  "prior_construction_costs_separate":center.row["costs"],
                  "scope":"finite DDIM20 posterior mean; condition-translated TRAIN route offsets, no new topology; full raw gate/ddq/actual executor still required"}
        return np.stack(raws),np.stack(residuals),metadata,traces


def load_empirical_sampler(path):
    path=Path(path).resolve()
    if path.suffix!=".json": raise ValueError("C loader rejects B/legacy tensor checkpoints")
    checkpoint=prior._read(path)
    if checkpoint.get("schema")!=CHECKPOINT_SCHEMA or checkpoint.get("semantics")!=semantic_identity() or checkpoint.get("method")!=METHOD:
        raise ValueError("C loader rejects foreign/reinterpreted checkpoint identity")
    if checkpoint.get("VAL_TEST_data_used") is not False or checkpoint["schedule"]!=schedule_identity(): raise ValueError("C training/schedule identity mismatch")
    if checkpoint["weights_file"]!="checkpoint.npz" or sha256(path.parent/"checkpoint.npz")!=checkpoint["weights_sha256"]: raise ValueError("C weights SHA mismatch")
    names={"config.json","normalizer.json","split.json","paired_training_data.npz"}
    if set(checkpoint["companion_hashes"])!=names: raise ValueError("C checkpoint companions incomplete")
    for name,expected in checkpoint["companion_hashes"].items():
        if sha256(path.parent/name)!=expected: raise ValueError("C companion SHA mismatch")
    normalizer_value=prior._read(path.parent/"normalizer.json");plan=prior._read(path.parent/"config.json")
    if (normalizer_value!=checkpoint["normalizer"] or normalizer_value.get("schema")!=NORMALIZER_SCHEMA
            or normalizer_value.get("calculation_dtype")!="float64" or normalizer_value.get("fit_VAL_TEST") is not False
            or plan["semantics"]!=semantic_identity() or plan["schedule"]!=schedule_identity()
            or plan["source_before"]!=checkpoint["source_identity"] or plan["prototypes"]!=checkpoint["prototypes"]
            or plan["weight_selection"]!=checkpoint["weights_selection"] or plan["optimizer_steps"]!=checkpoint["fit"]["optimizer_steps"]
            or prior._read(path.parent/"split.json")!=checkpoint["split"]): raise ValueError("C config/normalizer/split identity mismatch")
    normalizer=prior.ResidualNormalizer.from_dict(normalizer_value["statistics"])
    with np.load(path.parent/"checkpoint.npz",allow_pickle=False) as z:
        if set(z.files)!={"classifier_weight","classifier_bias","affine","feature_mean","target_mean","offsets","component_groups","betas","alpha_bars"}: raise ValueError("C NPZ identity mismatch")
        betas,alphas=cosine_schedule()
        if not np.array_equal(z["betas"],betas) or not np.array_equal(z["alpha_bars"],alphas): raise ValueError("C float64 schedule differs")
        model=EmpiricalConditionalScore(z["offsets"].copy(),z["component_groups"].copy(),z["classifier_weight"].copy(),z["classifier_bias"].copy(),z["affine"].copy(),z["feature_mean"].copy(),z["target_mean"].copy())
    with np.load(path.parent/"paired_training_data.npz",allow_pickle=False) as z:
        if not np.array_equal(z["offsets"],model.offsets) or not np.array_equal(z["component_groups"],model.component_groups): raise ValueError("C offsets differ from paired TRAIN data")
        if not np.array_equal(normalize_residual(normalizer,z["residuals"]).reshape(z["means"].shape),z["means"]): raise ValueError("C residual normalization differs")
        if not np.array_equal(z["absolute_labels"]-z["centers"],z["residuals"]): raise ValueError("C TRAIN center/label residual pairing differs")
        # Bind stored affine coefficients without performing another pinv fit.
        group_features=np.stack([z["features"][model.component_groups==g][0] for g in range(len(model.bias))])
        group_targets=np.stack([z["means"][model.component_groups==g].mean(0) for g in range(len(model.bias))])
        if (not np.array_equal(group_features.mean(0),model.feature_mean)
                or not np.array_equal(group_targets.mean(0),model.target_mean)
                or not np.array_equal(z["means"]-group_targets[model.component_groups],model.offsets)):
            raise ValueError("C conditional mean/offset TRAIN pairing differs")
        stats=checkpoint["conditional_mean_fit"]
        if stats!=plan["conditional_mean_fit"] or stats["closed_form_fits"]!=1 or stats["rcond"]!=1e-12 or stats["ridge"] is not None:
            raise ValueError("C affine fit metadata mismatch")
    if len(checkpoint["prototypes"])!=len(model.offsets) or len(normalizer.training_sample_ids)!=len(model.offsets): raise ValueError("C retained label count differs")
    return EmpiricalScoreSampler(path,checkpoint,model,normalizer)


def generate_empirical_proposals(task,center,checkpoint,output,*,K=8,seed=64,device="cpu"):
    if device!="cpu": raise ValueError("C fixes CPU float64 generation")
    output=Path(output).resolve()
    if any(p.is_relative_to(output) for p in (*center.source_files,Path(checkpoint).resolve())): raise ValueError("output contains input")
    output.mkdir(parents=True,exist_ok=False);start=time.perf_counter()
    sampler=load_empirical_sampler(checkpoint);raw,residuals,metadata,traces=sampler.sample(task,center,K=K,seed=seed)
    proposals=[]
    for i,controls in enumerate(raw):
        folder=output/f"c{i}";folder.mkdir();np.save(folder/"controls_free.npy",controls,allow_pickle=False);np.save(folder/"residual_free.npy",residuals[i],allow_pickle=False)
        np.savez_compressed(folder/"ddim_trace.npz",**traces[i])
        proposal=TrajectoryProposal.from_controls(task,controls,origin="diffusion",seed=seed+i,postprocessing=(),
                    metadata={**metadata,"candidate_index":i,"candidate_budget":K,"ddim_trace_sha256":sha256(folder/"ddim_trace.npz")})
        prior._write(folder/"proposal.json",proposal.to_dict());proposals.append(proposal)
    np.save(output/"center_free.npy",center.free,allow_pickle=False)
    metadata["total_generation_wall_s_including_load_and_artifacts"]=time.perf_counter()-start
    prior._write(output/"generation.json",metadata)
    return proposals,metadata
