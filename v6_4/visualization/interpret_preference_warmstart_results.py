"""Bounded Chinese interpretation of completed C.2 evidence, without evaluation.

Run only after final actual, validation and the frozen evaluator's base report.
Initial report/summary bytes are preserved before enriched versions are written.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import math
from pathlib import Path

from v6_4.route_optimizer_protocol import read, sha, write

SCHEMA = "v64_c2_reviewable_results_interpretation_v1"
ENDPOINTS = ("R8", "R12", "N8", "D8")
COMPARISONS = (("D8", "N8"), ("D8", "R8"), ("D8", "R12"), ("N8", "R8"), ("N8", "R12"))
METRICS = ("I_support", "L_full", "d_support", "base_translation_peak_m", "base_rotation_peak_rad")
REVIEW_SCHEMA = "v64_c2_pilot_scientific_review_v1"


def load_scientific_review(run, model_freeze_sha256):
    """Optional root review is bound to unchanged raw evidence before reporting.

    Required keys: schema, reviewed=true, model_freeze_sha256, source_sha256
    (run-relative raw table/actual paths), learning_benefit_established_in_pilot
    (boolean or NOT_ESTABLISHED), default_initializer_decision and a short list
    scientific_interpretation_zh. The actual completion receipt and at least one
    raw tables/ source must be included; the completion receipt binds all slots.
    """
    run = Path(run).resolve()
    path = run / "tables/pilot_scientific_review.json"
    if not path.exists():
        return None
    review = read(path)
    if review.get("schema") != REVIEW_SCHEMA or review.get("reviewed") is not True:
        raise ValueError("scientific review requires its explicit reviewed schema")
    if review.get("model_freeze_sha256") != model_freeze_sha256:
        raise ValueError("scientific review binds another model freeze")
    sources = review.get("source_sha256") or {}
    if not isinstance(sources, dict):
        raise ValueError("scientific review must bind actual completion and its raw table evidence")
    raw_tables = False
    interpretation = run / "tables/report_interpretation"
    for relative, expected in sources.items():
        if not isinstance(relative, str):
            raise ValueError("scientific review source paths must be canonical run-relative strings")
        target = (run / relative).resolve()
        if (run not in target.parents or target in (run / "summary.json", run / "REPORT.md") or
                target == interpretation or interpretation in target.parents):
            raise ValueError("scientific review sources must be immutable raw run evidence")
        if relative != target.relative_to(run).as_posix():
            raise ValueError("scientific review source paths must be canonical run-relative strings")
        if sha(target) != expected:
            raise ValueError("scientific review evidence hash changed: " + relative)
        raw_tables |= (run / "tables" in target.parents and target != path)
    if "actual_complete.json" not in sources or not raw_tables:
        raise ValueError("scientific review must bind actual completion and its raw table evidence")
    benefit = review.get("learning_benefit_established_in_pilot")
    if type(benefit) is not bool and benefit != "NOT_ESTABLISHED":
        raise ValueError("reviewed benefit must be boolean or NOT_ESTABLISHED")
    decision = review.get("default_initializer_decision")
    paragraphs = review.get("scientific_interpretation_zh")
    if not isinstance(decision, str) or not decision.strip() or len(decision) > 160:
        raise ValueError("scientific review requires an explicit concise initializer decision")
    if (not isinstance(paragraphs, list) or not 1 <= len(paragraphs) <= 6 or
            any(not isinstance(p, str) or not p.strip() or len(p) > 1200 for p in paragraphs) or
            sum(len(p) for p in paragraphs) > 4000):
        raise ValueError("scientific interpretation must be one to six concise Chinese paragraphs")
    return {**review, "review_file_sha256": sha(path)}


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def complete_quality(slot):
    quality = slot.get("quality") or {}
    if not (slot.get("full_task_success") is True and slot.get("original_independent_gates_passed") is True
            and slot.get("actual_steps") == 13500 and all(finite(quality.get(k)) for k in METRICS)):
        return None
    return {k: quality[k] for k in METRICS}


def near_quality(method, baseline, preference):
    a, b = complete_quality(method), complete_quality(baseline)
    if a is None or b is None:
        return None
    return bool(a["L_full"] <= b["L_full"] + .005 and
        (a["I_support"] <= b["I_support"] + .001 if preference == "A" else a["d_support"] >= .030))


def cold_cost_bracket(task_id, endpoint, planner_internal_cold_s, outer_worker_elapsed_s,
        stream_internal_cold_s, final_selection_elapsed_s, prefix08_elapsed_s=None):
    """Bound missing startup/common-check timing with already recorded intervals."""
    values = (planner_internal_cold_s, outer_worker_elapsed_s, stream_internal_cold_s,
        final_selection_elapsed_s)
    if endpoint not in ENDPOINTS or not all(finite(v) and v >= 0. for v in values):
        raise ValueError("cold timing bounds require finite nonnegative measured intervals")
    if not final_selection_elapsed_s <= stream_internal_cold_s <= outer_worker_elapsed_s:
        raise ValueError("final selection, internal planner and outer worker intervals are not ordered")
    if endpoint == "R8":
        if not finite(prefix08_elapsed_s) or not 0. <= prefix08_elapsed_s <= final_selection_elapsed_s:
            raise ValueError("R8 prefix timing must lie within its R12 selection interval")
        expected_lower = stream_internal_cold_s - final_selection_elapsed_s + prefix08_elapsed_s
        upper = outer_worker_elapsed_s - final_selection_elapsed_s + prefix08_elapsed_s
        formula = "outer_R_worker_elapsed - R_final_selection_elapsed + R_prefix08_elapsed"
    else:
        expected_lower, upper = stream_internal_cold_s, outer_worker_elapsed_s
        formula = "outer_worker_elapsed"
    if not math.isclose(planner_internal_cold_s, expected_lower, rel_tol=1e-12, abs_tol=1e-9):
        raise ValueError("endpoint planner-internal cost disagrees with its recorded stream/prefix intervals")
    if planner_internal_cold_s > upper:
        raise ValueError("cold timing lower bound exceeds its outer worker upper bracket")
    return {"task_id": task_id, "endpoint": endpoint,
        "planner_internal_cold_s": planner_internal_cold_s,
        "cold_lower_bound_s": planner_internal_cold_s, "cold_upper_bound_s": upper,
        "outer_worker_elapsed_s": outer_worker_elapsed_s, "stream_internal_cold_s": stream_internal_cold_s,
        "final_selection_elapsed_s": final_selection_elapsed_s, "prefix08_elapsed_s": prefix08_elapsed_s,
        "startup_and_tail_residual_s": outer_worker_elapsed_s - stream_internal_cold_s,
        "bound_width_s": upper - planner_internal_cold_s, "upper_bound_formula": formula,
        "not_exact_process_start_to_seal_measurement": True}


def paired_evidence(slots, costs, task_ids, brackets=None):
    result = []
    for method, baseline in COMPARISONS:
        rows = []
        for tid in task_ids:
            for preference in ("A", "B"):
                first, second = slots[(tid, method, preference)], slots[(tid, baseline, preference)]
                qa, qb = complete_quality(first), complete_quality(second)
                ca, cb = costs[(tid, method)], costs[(tid, baseline)]
                ma = (brackets or {}).get((tid, method)); ba = (brackets or {}).get((tid, baseline))
                saving_bound = ba["cold_lower_bound_s"] - ma["cold_upper_bound_s"] if ma and ba else None
                rows.append({"task_id": tid, "preference": preference,
                    "method_status": first["status"], "baseline_status": second["status"],
                    "method_full_and_gates": qa is not None, "baseline_full_and_gates": qb is not None,
                    "method_quality": qa, "baseline_quality": qb,
                    "quality_delta_method_minus_baseline": {k: qa[k] - qb[k] for k in METRICS} if qa and qb else None,
                    "within_near_quality_band": near_quality(first, second, preference),
                    "method_cold_planning_s": ca["cold_s"], "baseline_cold_planning_s": cb["cold_s"],
                    "cold_planning_delta_s": ca["cold_s"] - cb["cold_s"],
                    "cold_planning_scope": "planner_internal_timer_excludes_worker_startup_and_common_input_checks",
                    "conservative_cold_saving_lower_bound_s": saving_bound,
                    "guaranteed_cheaper_within_recorded_bounds": saving_bound > 0. if saving_bound is not None else None,
                    "method_warm_path_decomposition_estimate_s": ca.get("warm_s"),
                    "baseline_warm_path_decomposition_estimate_s": cb.get("warm_s"),
                    "warm_is_observed_repeat": False})
        result.append({"method": method, "baseline": baseline, "task_denominator": 4,
            "logical_preference_denominator": 8,
            "near_quality_rule_scope": "predeclared_R12_near_quality_band" if baseline == "R12" else
                "diagnostic_reuse_of_thresholds_not_formal_noninferiority",
            "complete_paired_preferences": sum(r["quality_delta_method_minus_baseline"] is not None for r in rows),
            "near_quality_preferences": sum(r["within_near_quality_band"] is True for r in rows),
            "rows": rows})
    return result


def amortization(slots, costs, task_ids, teacher_elapsed_s, teacher_service_s, training_s, brackets=None):
    """No amortization from configured quotas or survivor-only averages."""
    result = []
    for method in ("D8", "N8"):
        rows = []
        for tid in task_ids:
            prefs = {p: near_quality(slots[(tid, method, p)], slots[(tid, "R12", p)], p) for p in ("A", "B")}
            baseline_B = complete_quality(slots[(tid, "R12", "B")])
            if baseline_B is None or baseline_B["d_support"] < .030:
                prefs["B"] = None if baseline_B is None else False
            saving = costs[(tid, "R12")]["cold_s"] - costs[(tid, method)]["cold_s"]
            ma = (brackets or {}).get((tid, method)); ba = (brackets or {}).get((tid, "R12"))
            saving_bound = ba["cold_lower_bound_s"] - ma["cold_upper_bound_s"] if ma and ba else None
            rows.append({"task_id": tid, "A_full_gates_and_R12_band": prefs["A"],
                "B_full_gates_30mm_and_R12_band": prefs["B"], "measured_cold_saving_s": saving,
                "measured_cold_saving_scope": "planner_internal_timer_partial_scope_only",
                "conservative_cold_saving_lower_bound_s": saving_bound,
                "guaranteed_cheaper_within_recorded_bounds": finite(saving_bound) and saving_bound > 0.,
                "qualifies": all(v is True for v in prefs.values()) and finite(saving_bound) and saving_bound > 0.})
        all_four = len(rows) == 4 and all(r["qualifies"] for r in rows)
        mean_saving = sum(r["conservative_cold_saving_lower_bound_s"] for r in rows) / 4 if all_four else None
        offline_known = all(finite(v) and v >= 0. for v in (teacher_elapsed_s, teacher_service_s, training_s))
        eligible = all_four and offline_known
        result.append({"method": method, "baseline": "R12", "all_four_tasks_jointly_preserved_and_cheaper": all_four,
            "eligibility_convention": "extra_conservative_all_four_both_preferences_and_each_Task_positive_not_predeclared_protocol_gate",
            "tasks": rows, "mean_cold_saving_per_task_s": mean_saving,
            "mean_cold_saving_scope": "mean_of_four_Task_conservative_cold_saving_lower_bounds_shared_A_B_cost_once",
            "new_teacher_elapsed_makespan_s": teacher_elapsed_s,
            "new_teacher_cumulative_service_s": teacher_service_s,
            "training_elapsed_service_s": training_s,
            "break_even_tasks": (teacher_elapsed_s + training_s) / mean_saving if eligible else "NOT_ESTABLISHED",
            "break_even_tasks_upper_bound": (teacher_elapsed_s + training_s) / mean_saving if eligible else "NOT_ESTABLISHED",
            "break_even_elapsed_basis": "(new_teacher_phase_makespan + measured_training_elapsed) / mean_Task_conservative_cold_saving_lower_bound",
            "break_even_scope": "conditional_upper_bound_in_recorded_shared_resource_timing_context_not_observed_repeated_Task_amortization",
            "cumulative_service_cost_per_observed_wall_saving_ratio":
                (teacher_service_s + training_s) / mean_saving if eligible else None,
            "service_ratio_is_not_elapsed_makespan": True,
            "reason": "四个Task均保留A/B实际资格、原门禁和R12近质量带，且逐任务cold节省下界为正；摊销次数仅为所记录共享资源计时条件下的条件上界。"
                if eligible else "至少一个Task未同时保留A/B完整actual、原门禁、R12近质量带及正的cold节省下界，或计时间隔/离线成本证据缺失。",
            "benefit_verdict_requires_review": True})
    return result


def raw_initializer_evidence(task_id, endpoint, proposals):
    source = "diffusion" if endpoint == "D8" else "retrieval"
    result = []
    for proposal in proposals:
        if proposal.get("source") != source:
            continue
        diagnostics = proposal.get("raw_seed_diagnostics") or {}
        result.append({"task_id": task_id, "endpoint": endpoint, "source": source,
            "preference": proposal.get("preference"), "family": proposal.get("family"),
            "initializer_slot": proposal.get("initializer_slot"), "proposal_index": proposal.get("proposal_index"),
            "raw_legal": diagnostics.get("raw_legal"),
            "rejection_reason": diagnostics.get("rejection_reason") or proposal.get("initializer_rejection"),
            "raw_repaired": diagnostics.get("raw_repaired"), "resampled": diagnostics.get("resampled"),
            "ddim_sample_units": proposal.get("ddim_sample_units", 0),
            "raw_seed_acceptance_is_actual_success": False})
    return result


def observed_mixture_amortization(slots, costs, task_ids, teacher_elapsed_s, teacher_service_s,
        training_s, brackets=None, raw_initializers=()):
    """Descriptive sensitivity over all Tasks and R12's observed abilities only."""
    result = []
    for method in ("D8", "N8"):
        rows = []
        for tid in task_ids:
            preferences = []
            for pref in ("A", "B"):
                first, baseline = slots[(tid, method, pref)], slots[(tid, "R12", pref)]
                qa, qb = complete_quality(first), complete_quality(baseline)
                required = qb is not None
                band = near_quality(first, baseline, pref) if required else None
                baseline_no_plan = baseline["status"] == "NO_PLAN"
                preferences.append({"preference": pref, "method_status": first["status"],
                    "baseline_status": baseline["status"], "baseline_observed_complete_gated": required,
                    "method_complete_gated": qa is not None, "required_observed_baseline_ability": required,
                    "within_predeclared_R12_band": band,
                    "lost_observed_baseline_ability": required and qa is None,
                    "lost_predeclared_quality": required and qa is not None and band is not True,
                    "preserved_observed_ability_and_quality": band is True if required else None,
                    "comparison_status": ("PRESERVED" if band is True else "LOST_ACTUAL_ABILITY" if qa is None else "LOST_NEAR_QUALITY")
                        if required else "N/A_R12_NO_PLAN" if baseline_no_plan else "N/A_R12_NOT_COMPLETE_GATED",
                    "unchanged_missing_capability": baseline_no_plan and first["status"] == "NO_PLAN",
                    "added_full_gated_method_capability": baseline_no_plan and qa is not None,
                    "added_B30mm_method_capability": baseline_no_plan and pref == "B" and qa is not None and qa["d_support"] >= .030,
                    "N_A_is_quality_success": False,
                    "capability_attribution": "whole_initializer_plus_C1_search_plus_original_control_pipeline_not_raw_neural_seed"})
            ma = (brackets or {}).get((tid, method)); ba = (brackets or {}).get((tid, "R12"))
            saving = ba["cold_lower_bound_s"] - ma["cold_upper_bound_s"] if ma and ba else None
            rows.append({"task_id": tid, "preferences": preferences,
                "planner_internal_cold_saving_s": costs[(tid, "R12")]["cold_s"] - costs[(tid, method)]["cold_s"],
                "conservative_cold_saving_lower_bound_s": saving,
                "negative_Task_saving_retained": finite(saving) and saving < 0.,
                "raw_initializer_diagnostics": [r for r in raw_initializers if (r["task_id"], r["endpoint"]) == (tid, method)]})
        prefs = [p for r in rows for p in r["preferences"]]
        coverage = {"Task_denominator": 4, "logical_preference_denominator": 8,
            "R12_observed_complete_gated_preferences": sum(p["baseline_observed_complete_gated"] for p in prefs),
            "method_complete_gated_preferences": sum(p["method_complete_gated"] for p in prefs),
            "R12_NO_PLAN_N_A_preferences": sum(p["comparison_status"] == "N/A_R12_NO_PLAN" for p in prefs),
            "R12_incomplete_gated_N_A_preferences": sum(p["comparison_status"] == "N/A_R12_NOT_COMPLETE_GATED" for p in prefs),
            "preserved_observed_ability_and_quality_preferences": sum(p["preserved_observed_ability_and_quality"] is True for p in prefs),
            "lost_observed_baseline_abilities": sum(p["lost_observed_baseline_ability"] for p in prefs),
            "lost_predeclared_quality_preferences": sum(p["lost_predeclared_quality"] for p in prefs),
            "unchanged_missing_capability_preferences": sum(p["unchanged_missing_capability"] for p in prefs),
            "added_full_gated_method_capability_preferences": sum(p["added_full_gated_method_capability"] for p in prefs),
            "added_B30mm_method_capability_preferences": sum(p["added_B30mm_method_capability"] for p in prefs)}
        timing_complete = len(rows) == 4 and all(finite(r["conservative_cold_saving_lower_bound_s"]) for r in rows)
        mean_saving = sum(r["conservative_cold_saving_lower_bound_s"] for r in rows) / 4 if timing_complete else None
        no_loss = coverage["lost_observed_baseline_abilities"] == 0 and coverage["lost_predeclared_quality_preferences"] == 0
        offline_known = all(finite(v) and v >= 0. for v in (teacher_elapsed_s, teacher_service_s, training_s))
        eligible = (timing_complete and coverage["R12_observed_complete_gated_preferences"] > 0 and no_loss and
            mean_saving > 0. and offline_known)
        reasons = []
        if not no_loss: reasons.append("LOST_OBSERVED_R12_ABILITY_OR_PREDECLARED_QUALITY")
        if not timing_complete: reasons.append("INCOMPLETE_FOUR_TASK_TIMING_BOUNDS")
        if timing_complete and mean_saving <= 0.: reasons.append("NONPOSITIVE_FOUR_TASK_MEAN_SAVING_LOWER_BOUND")
        if not offline_known: reasons.append("MISSING_OFFLINE_COST_EVIDENCE")
        if coverage["R12_observed_complete_gated_preferences"] == 0: reasons.append("NO_OBSERVED_R12_ABILITY_TO_COMPARE")
        result.append({"method": method, "baseline": "R12", "descriptive_sensitivity_only": True,
            "scope": "observed_four_Task_empirical_mixture_not_predeclared_protocol_gate_or_population_claim",
            "coverage": coverage, "tasks": rows,
            "no_observed_R12_ability_or_quality_lost": no_loss,
            "mean_conservative_cold_saving_per_task_s": mean_saving,
            "negative_Task_savings_included": True, "shared_A_B_search_cost_counted_once_per_Task": True,
            "conditional_empirical_mixture_break_even_tasks_upper_bound":
                (teacher_elapsed_s + training_s) / mean_saving if eligible else "NOT_ESTABLISHED",
            "new_teacher_elapsed_makespan_s": teacher_elapsed_s, "training_elapsed_service_s": training_s,
            "new_teacher_cumulative_service_s": teacher_service_s,
            "cumulative_service_cost_per_observed_wall_saving_ratio":
                (teacher_service_s + training_s) / mean_saving if eligible else None,
            "service_ratio_is_not_elapsed_makespan": True,
            "break_even_basis": "(teacher_phase_makespan + training_elapsed) / mean_of_all_four_Task_(R12_lower-method_upper)",
            "ineligibility_reasons": reasons,
            "reason": "描述性观察混合中保留全部已观察R12能力和近质量带，四Task含负值的平均节省下界为正；仅给共享资源条件下的经验混合摊销上界。"
                if eligible else "观察混合未保留全部已观察R12能力/近质量，或四Task平均节省下界非正，或计时/成本证据不足。",
            "invalid_initializer_slot_credited_as_neural_benefit": False,
            "benefit_verdict_requires_fresh_review": True})
    return result


def first_hits(rows, oracle, budget):
    admissible = lambda r: r.get("prediction_admissible") is True and all(
        finite((r.get("prediction_metrics") or {}).get(k)) for k in ("I_support", "L_full"))
    first = next((i for i, r in enumerate(rows, 1) if admissible(r)), None)
    first_B = next((i for i, r in enumerate(rows, 1) if admissible(r) and
        finite(r["prediction_metrics"].get("d_support")) and r["prediction_metrics"]["d_support"] >= .030), None)
    matches = {}
    for pref in ("A", "B"):
        target = oracle["preferences"][pref].get("prediction_metrics")
        hit = None
        if target:
            for i, r in enumerate(rows, 1):
                if not admissible(r):
                    continue
                m = r["prediction_metrics"]
                if m["L_full"] <= target["L_full"] + .005 and (m["I_support"] <= target["I_support"] + .001 if pref == "A"
                        else finite(m.get("d_support")) and m["d_support"] >= .030):
                    hit = i
                    break
        matches[pref] = {"slot": hit, "status": "N/A_R12_NO_PLAN" if not target else "HIT" if hit else "RIGHT_CENSORED",
            "right_censored_budget": len(rows) if target and hit is None else None,
            "posthoc_only_not_online_or_checkpoint_input": True}
    return {"declared_budget": budget, "slots_consumed": len(rows),
        "N_first_admissible": {"slot": first, "status": "HIT" if first else "RIGHT_CENSORED", "right_censored_budget": len(rows) if first is None else None},
        "N_first_B": {"slot": first_B, "status": "HIT" if first_B else "RIGHT_CENSORED", "right_censored_budget": len(rows) if first_B is None else None},
        "N_match_R12": matches, "qualification_scope": "prediction_only"}


def _fmt(value, digits=6):
    if value is None:
        return "—"
    if type(value) is bool:
        return "是" if value else "否"
    return f"{value:.{digits}g}" if finite(value) else str(value)


def _report(evidence, summary):
    data, training = evidence["data"], evidence["training"]
    lines = ["# V6.4-C.2：偏好条件化Diffusion初始化先导结果", "",
        f"研究交付完成：`{summary['research_delivery_complete']}`；学习初始化实际运行：`{summary['learned_initializer_operational']}`；"
        f"学习效益：`{summary['learning_benefit_established_in_pilot']}`；默认决策：`{summary['default_initializer_decision']}`。结论状态与初始化是否运行分开记录。",
        "", "范围固定为4个TEST Task、2个新母场景、1个训练seed。历史C.1结论保持原范围，不作为本轮泛化证据。",
        "", "## 数据与训练", "",
        f"物理候选共{data['physical_candidates']}个（历史48＋新教师{data['new_candidates']}）；偏好标签{data['label_views']}行，"
        f"其中零残差{data['zero_label_views']}行。证据等级：" + "、".join(f"{k}={v}" for k, v in data["evidence_tiers_unique"].items()) + "。",
        f"实际训练{training['optimizer_updates_total']}次更新；选中checkpoint为第{training['selected_checkpoint_update']}次，"
        f"该权重只经历{training['selected_checkpoint_updates_experienced']}次更新。16次固定VAL检查共{training['validation_ddim_sample_units']}个DDIM样本；"
        f"训练曝光{training['sample_exposures_total']}次，不能当作独立轨迹数。TRAIN-only scaler与外部TaskSHA划分保持冻结。",
        f"TEST学习初值生成{summary['learned_seeds_generated']}个，raw合法{summary['learned_seeds_raw_legal']}个。D8的actual归属于Diffusion初始化＋C.1搜索＋原控制层，不是raw Diffusion策略成功率。",
        "", "| 条件 | TRAIN标签 | VAL标签 | TRAIN母场景数 |", "|---|---:|---:|---:|"]
    review = evidence.get("scientific_review")
    if review:
        insertion = lines.index("## 数据与训练") - 1
        paragraphs = [line for paragraph in review["scientific_interpretation_zh"] for line in (paragraph, "")]
        lines[insertion:insertion] = ["", "## 科学判读", "", *paragraphs]
    for c in data["supported_conditions"]:
        lines.append(f"|{c['preference']}-{c['reference_family']}|{c['train_labels']}|{c['val_labels']}|{len(c['train_mothers'])}|")
    if data["empty_teacher_buckets"]:
        lines += ["", "空教师桶：" + "；".join(f"{r['task_id']} {r['preference']}-{r['reference_family']}" for r in data["empty_teacher_buckets"]) +
            "。只表示本次教师预算内无标签，不表示物理不可行。"]
    lines += ["", "## 独立actual（每行分母均为4个Task）", "",
        "| 端点 | 偏好 | 完整Task | 原五门禁 | 完整＋30mm | NO_PLAN | actual失败 | actual前拒绝 | actual别名 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for s in evidence["endpoint_status"]:
        lines.append(f"|{s['endpoint']}|{s['preference']}|{s['full_task']}/4|{s['original_five_gates']}/4|{s['full_and_B30mm']}/4|{s['NO_PLAN']}|{s['actual_failed']}|{s['rejected_before_actual']}|{s['aliases']}|")
    lines += ["", "NO_PLAN、预检拒绝和失败都保留在分母；四槽前缀仅作预测，未额外执行actual。别名共享严格相同的执行证据，不增加独立样本数。",
        "", "## 首次预测达标与删失", "",
        "首次合格只是完整名义预测资格。R12匹配次数是全部搜索封存后的后处理，未输入网络、检索或停止规则。`>n（预算内未命中）`表示首次命中时间在预算n处右删失；`N/A`表示R12无对应计划。",
        "", "| Task | 端点 | 首次合格 | 首次B30mm | 匹配R12-A | 匹配R12-B |", "|---|---|---:|---:|---:|---:|"]
    def hit(h):
        return "N/A" if h["status"] == "N/A_R12_NO_PLAN" else str(h["slot"]) if h["slot"] is not None else f">{h['right_censored_budget']}（预算内未命中）"
    for h in evidence["first_hits"]:
        lines.append(f"|{h['task_id']}|{h['endpoint']}|{hit(h['N_first_admissible'])}|{hit(h['N_first_B'])}|{hit(h['N_match_R12']['A'])}|{hit(h['N_match_R12']['B'])}|")
    lines += ["", "## 配对actual质量与planner内部cold成本", "",
        "原差值均为前者减后者，原数值保持不变。只在双方完整Task且原门禁通过时计算质量差；失败行保持空值。ΔI单位rad/s，ΔL、Δd、Δ基座平移单位m，Δ基座转角单位rad，Δcold单位s。"
        "Δcold来自planner内部计时，缺少worker启动和公共输入检查。另列节省下界=基线下界−方法上界，正值才表示所记录区间内保证更低成本。",
        "", "相对R12的预声明近质量带：A同时满足I≤R12＋0.001、L≤R12＋0.005；B保持d≥0.030，再满足L≤R12＋0.005。"
        "同一阈值用于N8/R8比较时仅作诊断，不构成预声明统计非劣或Pareto支配结论。同一Task的A/B共享一次搜索，表中cold重复展示，不相加。"]
    for comparison in evidence["comparisons"]:
        lines += ["", f"### {comparison['method']} 对 {comparison['baseline']}", "",
            "| Task | 偏好 | 双方完整＋门禁 | ΔI | ΔL | Δd | Δ基座平移 | Δ基座转角 | Δcold内部 | 节省下界 | 近质量带 |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|"]
        for r in comparison["rows"]:
            delta = r["quality_delta_method_minus_baseline"] or {}
            status = "是" if delta else f"{r['method_status']} / {r['baseline_status']}"
            lines.append("|" + "|".join([r["task_id"], r["preference"], status,
                *[_fmt(delta.get(k)) for k in METRICS], _fmt(r["cold_planning_delta_s"]),
                _fmt(r["conservative_cold_saving_lower_bound_s"]), _fmt(r["within_near_quality_band"])]) + "|")
    offline = evidence["offline_costs"]
    lines += ["", "## 成本与有限结论", "",
        f"新教师阶段makespan {_fmt(offline['teacher_phase_makespan_s'])} s；累计教师候选服务工作量 {_fmt(offline['teacher_candidate_service_s'])} s；"
        f"训练 {_fmt(offline['training_elapsed_service_s'])} s。并行服务时间不等于端到端makespan，嵌套计时不重复相加。",
        "摊销分子仅包含已记录的新教师阶段makespan与正式训练时长，未包含未单独计时的数据导入、整理、审阅和媒体生成；"
        "因此下述条件上界只适用于这个成本口径，不能称完整生命周期的摊销上界。N8本身不需要Diffusion训练，"
        "这里为便于核算仍向N8计入本次研究的训练时长，属于保守的整轮研究成本口径。",
        "planner内部cold计时包含初始化准备、模型首次加载与推理，但开始于公共输入校验之后。以下仅给已有计时收据支持的区间："
        "下界为原内部cold，上界为外层worker elapsed；R8上界为R-worker elapsed−R最终selection elapsed＋第8槽prefix elapsed。"
        "外层收据还包括进程启动、公共检查和退出尾部，不能当作精确的process-start到selection封存时延。"
        "启动与尾部残差=worker elapsed−完整stream内部cold，不能再拆成已分别测量的启动或退出成本。",
        f"搜索使用{evidence['search_timing_context']['workers']}个并行worker共享资源；这些非隔离时延区间不证明单独运行或重复Task的时延。"
        "warm resident路径仅为分解估计，并未测量一次重复warm规划。候选物理、十步预演、最终actual、独立重放和几何查询分账；"
        "8比12少33.3%配额是协议设置，不能自动称为加速。",
        "", "| Task | 端点 | cold下界s | cold上界s | 启动/公共检查/尾部残差s |",
        "|---|---|---:|---:|---:|"]
    for row in evidence["endpoint_cold_cost_brackets"]:
        lines.append("|" + "|".join([row["task_id"], row["endpoint"], _fmt(row["cold_lower_bound_s"]),
            _fmt(row["cold_upper_bound_s"]), _fmt(row["startup_and_tail_residual_s"])]) + "|")
    lines += ["", "### 额外保守约定：四Task均具备A/B且逐Task成本更低", "",
        "这一严格摊销表额外要求四Task的R12与方法均具备完整A/B、B30mm和近质量带，且每个Task节省下界都为正。"
        "它是附加保守约定，并非预声明协议通过门槛；R12自身NO_PLAN会使该约定不成立，不能据此称协议失败。"]
    for a in evidence["amortization"]:
        lines += ["", f"{a['method']}相对R12：四任务同时保留A/B实际任务、门禁、近质量带且逐任务cold节省下界为正：{_fmt(a['all_four_tasks_jointly_preserved_and_cheaper'])}；"
            f"break_even_tasks条件上界=`{_fmt(a['break_even_tasks'])}`。以四Task节省下界均值为分母，A/B共享成本每Task只计一次；"
            f"这不是观察到的重复任务摊销结果。{a['reason']}"]
    lines += ["", "### 附加描述性分析：四Task观察能力混合", "",
        "只要求保留R12已经观察到的完整actual＋原门禁能力及同一预声明近质量带。R12的NO_PLAN保持N/A，"
        "方法也NO_PLAN是缺失能力未变，方法完整则单列为整个方法流程新增的实际能力；N/A不计质量成功。"
        "R12有计划但actual/门禁不完整也单列N/A，不虚构基线质量。四Task全部进入成本均值，包括负节省，A/B共享搜索只计一次。"
        "该敏感性是描述性的经验混合核算，不替代原表、严格约定或最终科学判读，不是总体结论或神经种子因果收益。"]
    for a in evidence["observed_mixture_amortization_sensitivity"]:
        c = a["coverage"]
        lines += ["", f"{a['method']}：R12已观察完整门禁能力{c['R12_observed_complete_gated_preferences']}/8；"
            f"方法完整门禁能力{c['method_complete_gated_preferences']}/8；保留原能力且近质量{c['preserved_observed_ability_and_quality_preferences']}；"
            f"丢失原实际能力{c['lost_observed_baseline_abilities']}、丢失近质量{c['lost_predeclared_quality_preferences']}；"
            f"R12 NO_PLAN N/A={c['R12_NO_PLAN_N_A_preferences']}，其它不完整基线N/A={c['R12_incomplete_gated_N_A_preferences']}；"
            f"缺失能力未变{c['unchanged_missing_capability_preferences']}，新增完整方法能力{c['added_full_gated_method_capability_preferences']}"
            f"（其中新增B30mm={c['added_B30mm_method_capability_preferences']}）。",
            "", "| Task（分母4） | A观察能力比较 | B观察能力比较 | cold节省下界s（含负值） |",
            "|---|---|---|---:|"]
        for row in a["tasks"]:
            prefs = {p["preference"]: p for p in row["preferences"]}
            lines.append("|" + "|".join([row["task_id"], prefs["A"]["comparison_status"], prefs["B"]["comparison_status"],
                _fmt(row["conservative_cold_saving_lower_bound_s"])]) + "|")
        lines += ["", f"四Task平均cold节省下界={_fmt(a['mean_conservative_cold_saving_per_task_s'])} s；"
            f"经验混合break-even条件上界=`{_fmt(a['conditional_empirical_mixture_break_even_tasks_upper_bound'])}` Task；"
            f"累计服务工作量/观察墙钟节省比={_fmt(a['cumulative_service_cost_per_observed_wall_saving_ratio'])}，该服务比不等于makespan摊销次数。"
            f"{a['reason']}"]
    lines += ["", "### 原始初始化合法性与拒绝原因", "",
        "非法raw初始化仍消耗声明槽位，后续C.1搜索找到的计划不能记为非法神经种子成功；没有修复或重采样。",
        "", "| Task | 端点 | 偏好/参考 | 初值槽 | raw合法 | 来源拒绝原因 |", "|---|---|---|---:|---|---|"]
    for row in evidence["raw_initializer_diagnostics"]:
        reason = row["rejection_reason"]
        text = str(reason).replace("|", "\\|").replace("\n", " ") if reason is not None else "—"
        lines.append("|" + "|".join([row["task_id"], row["endpoint"], f"{row['preference']}/{row['family']}",
            str(row["initializer_slot"]), _fmt(row["raw_legal"]), text]) + "|")
    lines += ["", f"学习效益状态为`{summary['learning_benefit_established_in_pilot']}`，默认`{summary['default_initializer_decision']}`。预测接受、学习来源占比或较少物理积分不能代替独立actual质量和包含推理的成本对照。",
        "", "机器证据见 `tables/report_interpretation/evidence.json`、`tables/` 与 `teacher_update/`；初始生成报告原字节保存在 `tables/report_interpretation/initial_REPORT.md` 和 `initial_summary.json`。",
        "", "`deployment=NOT_MET`；`continuous_time_safety=NOT_ESTABLISHED`；`hardware_safety=NOT_ESTABLISHED`。"]
    return "\n".join(lines) + "\n"


def interpret_results(run):
    run = Path(run).resolve(); output = run / "tables/report_interpretation"
    if (output / "manifest.json").exists():
        manifest = read(output / "manifest.json")
        for relative, expected in {**manifest["source_sha256"], **manifest["output_sha256"]}.items():
            if sha(run / relative) != expected:
                raise ValueError("immutable interpretation evidence changed: " + relative)
        return read(output / "evidence.json")
    if output.exists():
        raise FileExistsError("unfinished interpretation retained; inspect before recovery")
    from v6_4.evaluate_preference_warmstart import verify_selections
    verify_selections(run)
    validation = read(run / "validation/validation.json")
    if validation.get("all_terminal") is not True or validation.get("logical_actual_slots") != 32:
        raise ValueError("interpretation requires completed actual and final terminal validation")
    original_summary = read(run / "summary.json")
    if original_summary.get("research_delivery_complete") is not True:
        raise ValueError("base report has not completed the declared protocol")
    source_paths = set()
    def load(relative):
        source_paths.add(relative)
        return read(run / relative)
    load("validation/validation.json"); load("actual_complete.json"); load("sealed_selections/all_selections.json")
    frozen_sha = sha(run / "model_freeze.json"); source_paths.add("model_freeze.json")
    tasks = [t for t in load("plan.json")["tasks"] if t["split"] == "test"]
    tids = [t["task_id"] for t in tasks]
    if len(tids) != 4:
        raise ValueError("four TEST Task denominator required")
    slots = {}
    complete = load("actual_complete.json")
    for relative, expected in complete["slot_hashes"].items():
        if sha(run / relative) != expected:
            raise ValueError("actual completion receipt changed")
        slot = load(relative); key = (slot["task_id"], slot["endpoint"], slot["preference"])
        if key in slots:
            raise ValueError("duplicate logical actual slot")
        slots[key] = slot
    if set(slots) != {(t, e, p) for t in tids for e in ENDPOINTS for p in ("A", "B")}:
        raise ValueError("all 32 logical methods, including failures, must remain visible")
    costs = {(r["task_id"], r["endpoint"]): r for r in load("tables/D_endpoint_planning_costs.json")}
    if any(not finite(costs[(t, e)].get("cold_s")) for t in tids for e in ENDPOINTS):
        raise ValueError("measured endpoint cold costs are missing")
    search_phase = load("search_phase.json")
    if search_phase.get("phase") != "search" or search_phase.get("passed") is not True:
        raise ValueError("cold timing brackets require the successful outer search phase receipt")
    jobs = {}
    for job in search_phase["jobs"]:
        key = (job["task_id"], job["method"])
        if key in jobs or job.get("exit_code") != 0:
            raise ValueError("search timing receipt has duplicate or failed worker intervals")
        jobs[key] = job
    if set(jobs) != {(t, m) for t in tids for m in ("R", "N", "D")}:
        raise ValueError("all twelve outer search worker intervals are required")
    brackets = {}
    for tid in tids:
        for method in ("R", "N", "D"):
            stream = f"benchmark_search/{tid}/{method}"
            planning = f"{stream}/planning/{tid}"
            cost_path, selection_path = stream + "/planning_cost.json", planning + "/selection.json"
            stream_cost, final_selection = load(cost_path), load(selection_path)
            if (stream_cost.get("task_id") != tid or stream_cost.get("method") != method or
                    stream_cost.get("selection_sha256") != sha(run / selection_path)):
                raise ValueError("search internal timing receipt does not bind its final selection")
            prefix_path = planning + "/prefix_08.json"
            prefix = load(prefix_path) if method == "R" else None
            for endpoint in (("R8", "R12") if method == "R" else (method + "8",)):
                row = cold_cost_bracket(tid, endpoint, costs[(tid, endpoint)]["cold_s"],
                    jobs[(tid, method)]["elapsed_wall_s"], stream_cost["end_to_end_cold_planning_s"],
                    final_selection["elapsed_wall_s"], prefix["elapsed_wall_s"] if endpoint == "R8" else None)
                row["source_receipts"] = ["tables/D_endpoint_planning_costs.json", "search_phase.json", cost_path,
                    selection_path, *([prefix_path] if endpoint == "R8" else [])]
                brackets[(tid, endpoint)] = row
    bracket_rows = [brackets[(t, e)] for t in tids for e in ENDPOINTS]
    timing_context = {"workers": search_phase["workers"], "phase_makespan_s": search_phase["makespan_s"],
        "cumulative_worker_service_s": search_phase["cumulative_worker_service_s"],
        "lower_scope": "planner_internal_timer_excludes_worker_startup_and_common_input_checks",
        "upper_scope": "outer_worker_bracket_includes_startup_common_checks_and_tail_not_exact_process_start_to_selection_seal",
        "shared_resources_nonisolated": True,
        "limitation": "concurrent_workers_share_resources; recorded_bounds_do_not_establish_isolated_or_repeated_Task_latency",
        "warm_path_scope": "decomposition estimate; no observed repeated warm planning"}
    dataset = load("dataset/dataset_manifest.json"); facts = load("dataset/candidate_facts.json")
    labels = load("dataset/labels.json"); buckets = load("dataset/bucket_coverage.json")
    training = load("model/training_summary.json")
    curve_path = "model/model/curves.jsonl"; source_paths.add(curve_path)
    curves = [json.loads(line) for line in (run / curve_path).read_text(encoding="utf8").splitlines() if line]
    train_updates = [r["update"] for r in curves if r["kind"] == "train"]
    val_updates = [r["update"] for r in curves if r["kind"] == "validation"]
    if train_updates != list(range(1, 4001)) or val_updates != list(range(250, 4001, 250)):
        raise ValueError("4000 updates and the 16 predeclared VAL checks must be evidenced")
    hits = []; raw_initializers = []
    for task in tasks:
        tid = task["task_id"]; oracle = load(f"sealed_selections/{tid}/R12.json")
        for endpoint in ENDPOINTS:
            planning = f"benchmark_search/{tid}/{endpoint[0]}/planning/{tid}"
            rows = load(planning + "/candidate_registry.json")
            budget = 12 if endpoint == "R12" else 8
            prefix = load(planning + f"/prefix_{budget:02d}.json")
            hits.append({"task_id": tid, "endpoint": endpoint, **first_hits(rows[:prefix["budget"]["slots_consumed"]], oracle, budget)})
            if endpoint in ("N8", "D8"):
                proposal_path = planning + "/proposals.json"
                raw_initializers.extend({**r, "source_receipt": proposal_path}
                    for r in raw_initializer_evidence(tid, endpoint, load(proposal_path)))
    statuses = []
    for endpoint in ENDPOINTS:
        for pref in ("A", "B"):
            values = [slots[(tid, endpoint, pref)] for tid in tids]
            statuses.append({"endpoint": endpoint, "preference": pref, "denominator": 4,
                "full_task": sum(s["full_task_success"] for s in values),
                "original_five_gates": sum(s["original_independent_gates_passed"] for s in values),
                "full_and_B30mm": sum(complete_quality(s) is not None and s.get("clearance_30mm_met") is True for s in values),
                "NO_PLAN": sum(s["status"] == "NO_PLAN" for s in values),
                "actual_failed": sum(s["entered_actual"] and not s["full_task_success"] for s in values),
                "rejected_before_actual": sum(not s["entered_actual"] and s["status"] != "NO_PLAN" for s in values),
                "aliases": sum("alias_of_method" in s for s in values), "unique_actual": sum(s["unique_run"] for s in values)})
    teacher_phase = load("teacher_phase.json")
    cost_summary = original_summary["measured_cost_and_amortization"]
    offline = {"teacher_phase_makespan_s": teacher_phase["makespan_s"],
        "teacher_worker_cumulative_service_s": teacher_phase["cumulative_worker_service_s"],
        "teacher_candidate_service_s": cost_summary["teacher"]["cumulative_service_elapsed_s"],
        "training_elapsed_service_s": training["elapsed_s"],
        "teacher_work": cost_summary["teacher"], "test_prediction_work": cost_summary["test_prediction"],
        "actual_replay_and_geometry_work": cost_summary["actual_including_independent_replay_and_geometry"],
        "overlapping_nested_times_not_added": True, "historical_reuse_additional_physics": 0}
    evidence = {"schema": SCHEMA, "created_utc": datetime.now(timezone.utc).isoformat(),
        "model_freeze_sha256": frozen_sha, "review_status": "PROVISIONAL_REQUIRES_FINAL_INTERPRETATION",
        "scope": {"TEST_tasks": 4, "new_TEST_mothers": 2, "training_seeds": 1},
        "data": {"physical_candidates": len(facts), "new_candidates": len(facts) - 48,
            "label_views": len(labels), "zero_label_views": sum(not any(any(row) for row in s["z_m"]) for s in labels),
            "evidence_tiers_unique": dict(Counter(f["evidence_tier"] for f in facts)),
            "supported_conditions": dataset["supported_preference_family_conditions"],
            "empty_teacher_buckets": [b for b in buckets if b["split"] in ("train", "val") and b["label_count_unique"] == 0]},
        "training": {k: training[k] for k in ("optimizer_updates_total", "selected_checkpoint_update", "selected_checkpoint_updates_experienced",
            "sample_exposures_total", "validation_checkpoint_count", "validation_ddim_sample_units", "selected_validation", "test_used_for_selection")},
        "endpoint_status": statuses, "first_hits": hits, "comparisons": paired_evidence(slots, costs, tids, brackets),
        "endpoint_cold_cost_brackets": bracket_rows, "search_timing_context": timing_context,
        "raw_initializer_diagnostics": raw_initializers,
        "offline_costs": offline, "amortization": amortization(slots, costs, tids, teacher_phase["makespan_s"],
            teacher_phase["cumulative_worker_service_s"], training["elapsed_s"], brackets),
        "observed_mixture_amortization_sensitivity": observed_mixture_amortization(slots, costs, tids,
            teacher_phase["makespan_s"], teacher_phase["cumulative_worker_service_s"], training["elapsed_s"], brackets, raw_initializers),
        "warm_path_scope": "decomposition estimate; no observed repeated warm planning",
        "physics_steps_added": 0, "training_updates_added": 0, "inference_samples_added": 0}
    reviewed = load_scientific_review(run, frozen_sha)
    if reviewed:
        source_paths.add("tables/pilot_scientific_review.json")
        source_paths.update(reviewed["source_sha256"])
    evidence["scientific_review"] = reviewed
    summary = {**original_summary, "results_interpretation": evidence,
        "learning_benefit_established_in_pilot": reviewed["learning_benefit_established_in_pilot"] if reviewed else "NOT_ESTABLISHED",
        "default_initializer_decision": reviewed["default_initializer_decision"] if reviewed else "retain_C1_rule_pending_review"}
    if reviewed:
        summary["pilot_interpretation_review"] = reviewed
    evidence["review_status"] = "REVIEWED_PILOT_INTERPRETATION" if reviewed else "PROVISIONAL_REQUIRES_FINAL_INTERPRETATION"
    summary["measured_cost_and_amortization"] = {**cost_summary, "offline_separation": offline,
        "break_even_by_method": evidence["amortization"], "warm_path_is_decomposition_estimate": True,
        "strict_amortization_convention": "extra_conservative_all_four_both_preferences_and_each_Task_positive_not_predeclared_protocol_gate",
        "observed_mixture_amortization_sensitivity": evidence["observed_mixture_amortization_sensitivity"],
        "endpoint_cold_cost_brackets": bracket_rows, "search_timing_context": timing_context,
        "break_even_tasks": evidence["amortization"][0]["break_even_tasks"],
        "break_even_tasks_scope": evidence["amortization"][0]["break_even_scope"],
        "reason": evidence["amortization"][0]["reason"]}
    rendered = _report(evidence, summary)
    source_hashes = {p: sha(run / p) for p in sorted(source_paths)}
    initial_hashes = {"summary.json": sha(run / "summary.json"), "REPORT.md": sha(run / "REPORT.md")}
    output.mkdir(parents=True)
    for source, target in ((run / "summary.json", output / "initial_summary.json"), (run / "REPORT.md", output / "initial_REPORT.md")):
        with target.open("xb") as stream:
            stream.write(source.read_bytes())
    write(output / "evidence.json", evidence)
    write(output / "endpoint_cold_cost_brackets.json", bracket_rows)
    write(output / "observed_mixture_amortization_sensitivity.json", evidence["observed_mixture_amortization_sensitivity"])
    (run / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf8")
    (run / "REPORT.md").write_text(rendered, encoding="utf8")
    if sha(run / "model_freeze.json") != frozen_sha:
        raise ValueError("model freeze changed during interpretation")
    files = (output / "initial_summary.json", output / "initial_REPORT.md", output / "evidence.json",
        output / "endpoint_cold_cost_brackets.json", output / "observed_mixture_amortization_sensitivity.json",
        run / "summary.json", run / "REPORT.md")
    write(output / "manifest.json", {"schema": SCHEMA, "source_sha256": source_hashes,
        "output_sha256": {p.relative_to(run).as_posix(): sha(p) for p in files},
        "initial_root_sha256": initial_hashes,
        "final_root_sha256": {"summary.json": sha(run / "summary.json"), "REPORT.md": sha(run / "REPORT.md")},
        "model_freeze_unchanged": True, "initial_bytes_preserved": True})
    return evidence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    args = parser.parse_args()
    evidence = interpret_results(args.run)
    print(json.dumps({"schema": evidence["schema"], "review_status": evidence["review_status"],
        "amortization": evidence["amortization"], "physics_steps_added": 0}, ensure_ascii=False))


if __name__ == "__main__":
    main()
