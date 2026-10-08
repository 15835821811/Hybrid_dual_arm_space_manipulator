"""C.2 high-level seed providers and raw qualification, without repair.

Only initial positions 1 (A/v1) and 3 (B/v2) are replaceable. Providers
construct those two proposals once; the optimizer owns slot accounting and
its original exact reference cache. No provider reads candidate quality.
"""
from __future__ import annotations

import copy

import numpy as np

from .route_optimizer_protocol import VERSIONS, active_intervals, build_reference_definition, digest, parameter_plan


OVERRIDE_CONDITIONS = {1: ("A", "v1"), 3: ("B", "v2")}


def json_raw(value):
    """Lossless finite JSON values, with explicit tokens for nonfinite raw data.

    JSON NaN/Infinity literals are forbidden in retained evidence. Tokens keep
    their locations and kinds instead of replacing illegal coordinates by 0.
    """
    if isinstance(value, np.ndarray):
        return json_raw(value.tolist())
    if isinstance(value, np.generic):
        return json_raw(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return {"nonfinite_float": "NaN" if np.isnan(value) else "Infinity" if value > 0 else "-Infinity"}
    if isinstance(value, dict):
        return {str(k): json_raw(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_raw(v) for v in value]
    return value


def search_mask(task):
    mask = [False] * 6
    for index in active_intervals(build_reference_definition(task)):
        mask[index] = True
    return mask


def raw_seed_plan(task, proposal, slot):
    """Qualify raw output as is; return (plan or None, diagnostics).

    Full 12D raw residuals are mandatory so inactive coordinates cannot be
    silently discarded. Original reference precheck semantics remain frozen.
    """
    preference, family = OVERRIDE_CONDITIONS[slot]
    raw = proposal.get("raw_z_m")
    diagnostics = {"raw_z_m": json_raw(raw), "requested_preference": preference,
        "requested_family": family, "raw_legal": False, "raw_repaired": False,
        "resampled": False, "search_interval_mask": search_mask(task)}
    if proposal.get("initializer_rejection"):
        diagnostics["rejection_reason"] = proposal["initializer_rejection"]
        return None, diagnostics
    try:
        if proposal.get("source") not in ("diffusion", "retrieval"):
            raise ValueError("initializer source must be diffusion or retrieval")
        if proposal.get("family") != family or proposal.get("preference") != preference:
            raise ValueError("initializer preference/family differs from fixed slot")
        z = np.asarray(raw, dtype=float)
        if z.shape not in ((12,), (6, 2)):
            raise ValueError("raw seed must contain all 12 residual coordinates")
        z = z.reshape(6, 2)
        if not np.isfinite(z).all():
            raise ValueError("raw seed contains nonfinite coordinates")
        mask = np.asarray(diagnostics["search_interval_mask"], dtype=bool)
        if np.any(z[~mask] != 0.):
            raise ValueError("raw seed has nonzero inactive search coordinates")
        if np.any(np.linalg.norm(z, axis=1) > .020):
            raise ValueError("raw seed exceeds the original 20mm interval disk")
        plan = parameter_plan(task, family, z[mask].reshape(-1))
        from .task_anchored_reference import reference_precheck
        check = reference_precheck(task, plan, cartesian_speed_limit_m_s=.24)
        diagnostics["reference_precheck"] = check
        if not check["passed"]:
            raise ValueError("raw seed fails original analytic reference precheck")
        diagnostics.update(raw_legal=True, rejection_reason=None,
            canonical_family="v1" if not np.any(plan.z_m) else family)
        return plan, diagnostics
    except (ValueError, TypeError, KeyError, IndexError, OverflowError) as error:
        diagnostics.update(rejection_reason=str(error), error_type=type(error).__name__)
        return None, diagnostics


class FrozenSeedInitializer:
    """Retain exactly two already-produced seeds; calls never regenerate them."""

    def __init__(self, overrides, identity=None):
        if set(overrides) != set(OVERRIDE_CONDITIONS):
            raise ValueError("exactly slots 1 and 3 must be overridden")
        self._overrides = copy.deepcopy(overrides)
        self.identity = {**(identity or {}), "schema": "v64_c2_frozen_seed_initializer_v1",
            "overrides_sha256": digest(json_raw(overrides))}

    def __call__(self, task):
        return copy.deepcopy(self._overrides)


class RetrievalInitializer:
    """Two deterministic nearest references from D's identical TRAIN label pool.

    Distance is the mean squared difference across the complete C.2
    TRAIN-normalized condition. Type/presence/mask dimensions retain the
    dataset scaler's discrete semantics. Preference, family and search mask
    must match exactly; equal distances use frozen input/reference IDs.
    """

    def __init__(self, labels, condition_scaler, identity=None):
        self.labels = copy.deepcopy([row for row in labels if row.get("split", "").lower() == "train"])
        self.condition_scaler = copy.deepcopy(condition_scaler)
        self.identity = {**(identity or {}), "schema": "v64_c2_train_retrieval_v1",
            "distance": "mean_squared_all_train_normalized_condition_dimensions_v1",
            "tie_order": ["task_sha256", "plan_sha256", "sample_id"],
            "train_labels_sha256": digest(json_raw(self.labels))}
        if hasattr(self.condition_scaler, "to_dict"):
            self.identity["condition_scaler_sha256"] = digest(json_raw(self.condition_scaler.to_dict()))

    def __call__(self, task):
        from .preference_teacher_dataset import encode_condition
        mask = search_mask(task)
        result = {}
        for slot, (preference, family) in OVERRIDE_CONDITIONS.items():
            definition = build_reference_definition(task, version=VERSIONS[family])
            encoded = encode_condition(task, definition, preference, family, search_mask=mask)
            query = np.asarray(self.condition_scaler.transform_condition(encoded), dtype=float)
            if query.ndim != 1 or not np.isfinite(query).all():
                raise ValueError("retrieval query condition must be a finite encoded vector")
            ranked = []
            for row in self.labels:
                if (row.get("preference") != preference or row.get("reference_family", row.get("family")) != family
                        or list(row.get("search_interval_mask", [])) != mask):
                    continue
                condition = row.get("condition_normalized")
                if condition is None:
                    raise ValueError("TRAIN retrieval label missing normalized condition")
                values = np.asarray(condition, dtype=float)
                if values.shape != query.shape or not np.isfinite(values).all():
                    raise ValueError("TRAIN retrieval condition schema/finiteness differs")
                distance = float(np.mean(np.square(values - query)))
                key = tuple(str(row.get(k, "")) for k in ("task_sha256", "plan_sha256", "sample_id"))
                ranked.append((distance, key, row))
            proposal = {"source": "retrieval", "family": family, "preference": preference,
                "raw_z_m": None, "retrieval_rules": self.identity}
            if not ranked:
                proposal["initializer_rejection"] = "UNSUPPORTED_TRAINING_CONDITION"
            else:
                distance, _, selected = min(ranked, key=lambda item: (item[0], item[1]))
                proposal.update(raw_z_m=copy.deepcopy(selected["z_m"]),
                    retrieval_source={k: selected.get(k) for k in
                        ("sample_id", "task_id", "task_sha256", "mother_id", "plan_sha256", "reference_family")},
                    retrieval_distance=distance, matched_search_interval_mask=mask)
            result[slot] = proposal
        return result
