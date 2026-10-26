"""Aggregate recorded results without filtering failed tasks or sampling seeds."""
import os
for _k in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ[_k] = "1"
import csv
from datetime import datetime
from pathlib import Path
import numpy as np
from .round2_data import BASE, read, teacher_path
from .pipeline import write_json


def intervals(rows, field):
    """Resample tasks, keeping their repeated sampling runs together."""
    ids = sorted({r["id"] for r in rows})
    values = np.array([np.mean([r[field] for r in rows if r["id"] == i]) for i in ids])
    rng = np.random.default_rng(20261013)
    samples = values[rng.integers(0, len(ids), (5000, len(ids)))].mean(axis=1)
    return (100*np.quantile(samples, [.025, .975])).tolist()


def reasons(result):
    if not result:
        return ["运行异常，未保存 result.json"]
    out = []
    if result["failure"]:
        out.append(result["failure"])
    if not result["full_task_executed"]:
        out.append("未完成 27 s")
    for arm in ("continuum", "rigid"):
        m = result["metrics"][arm]
        for key, limit in (("position_max_mm", 3.), ("orientation_max_deg", .5)):
            if m[key] > limit:
                out.append(f"{arm}.{key}={m[key]:.6g}>{limit}")
    ex = result["execution"]
    for key in ("torque_saturation_count", "qp_failures", "joint_limit_violation_steps"):
        if ex[key]:
            out.append(f"{key}={ex[key]}")
    if ex["minimum_pair_margin_m"] < -1e-6:
        out.append(f"碰撞余量={ex['minimum_pair_margin_m']:.6g} m")
    if result["meets_stage_tracking"] and not result["meets_original_tracking_precision"]:
        c, r = result["metrics"]["continuum"], result["metrics"]["rigid"]
        for label, value, limit in (("柔性路径 RMSE/mm", c["path_rmse_mm"], .18),
            ("刚性最终误差/mm", r["final_position_mm"], .10),
            ("刚性稳态 RMSE/mm", r["steady_rmse_mm"], .15),
            ("最大姿态误差/deg", max(c["orientation_max_deg"], r["orientation_max_deg"]), .25)):
            if value > limit:
                out.append(f"{label}={value:.6g}>{limit}")
    return out


def aggregate(name, expected):
    summary = read(BASE/name/"summary.json")
    rows = summary["episodes"]
    durations, first, warm, table = [], [], [], []
    for row in rows:
        path = Path(row["result_path"])
        result = read(path) if path.exists() else None
        flat = {"group": name, "id": row["id"], "sampling_seed": row["sampling_seed"],
            "completed": row["completed"], "basic": row["passed"], "strict": row["strict"],
            "runtime_failure":row.get("failure"),
            "reasons": "; ".join(reasons(result)), "result_path": str(path)}
        if result:
            for arm, m in result["metrics"].items():
                flat.update({f"{arm}_{k}": v for k, v in m.items()})
            ex = result["execution"]
            for key in ("minimum_pair_margin_m", "maximum_actual_velocity_ratio", "maximum_actual_acceleration_ratio",
                "maximum_requested_torque_ratio", "physics_computation_p95_ms", "physics_computation_p99_ms",
                "physics_computation_max_ms", "physics_2ms_deadline_misses", "joint_limit_violation_steps"):
                flat[key] = ex[key]
            for key in ("sampling_and_selection_p95_ms", "sampling_and_selection_p99_ms",
                "sampling_and_selection_max_ms", "planning_100ms_deadline_misses", "wall_seconds"):
                flat[key] = result[key]
        trace = path.parent/"trace.npz"
        if trace.exists():
            with np.load(trace) as f:
                p = f["planning_seconds"]
                durations.extend(p)
                if len(p):
                    first.append(p[0]); warm.extend(p[1:])
                    flat["planning_first_call_ms"] = float(p[0]*1000)
                    flat["planning_p50_ms"] = float(np.median(p)*1000)
                e, t = f["error"], f["time"]
                for arm, start in (("continuum", 0), ("rigid", 6)):
                    position = np.linalg.norm(e[:, start:start+3], axis=1)*1000
                    k = int(np.argmax(position))
                    flat[arm+"_peak_time_s"] = float(t[k])
        table.append(flat)
    seeds = sorted({r["sampling_seed"] for r in rows if r["sampling_seed"] is not None})
    grouped = {str(s): {"total": sum(r["sampling_seed"] == s for r in rows),
        "basic": sum(r["passed"] and r["sampling_seed"] == s for r in rows),
        "strict": sum(r["strict"] and r["sampling_seed"] == s for r in rows)} for s in seeds}
    by_task = {i: {"total": sum(r["id"] == i for r in rows),
        "basic": sum(r["passed"] and r["id"] == i for r in rows),
        "strict": sum(r["strict"] and r["id"] == i for r in rows)} for i in sorted({r["id"] for r in rows})}
    def timing(values):
        return {"calls": len(values), **({"p50_ms": float(np.percentile(values, 50)*1000),
            "p95_ms": float(np.percentile(values, 95)*1000), "p99_ms": float(np.percentile(values, 99)*1000),
            "max_ms": float(max(values)*1000), "over_100ms": int(np.sum(np.asarray(values) > .1)),
            "over_100ms_fraction": float(np.mean(np.asarray(values) > .1))} if values else {})}
    return {"name": name, "expected": expected, "finished": len(rows), "complete_batch": len(rows) == expected,
        "completed": sum(r["completed"] for r in rows), "basic": sum(r["passed"] for r in rows),
        "strict": sum(r["strict"] for r in rows), "basic_task_cluster_ci95_percent": intervals(rows, "passed"),
        "strict_task_cluster_ci95_percent": intervals(rows, "strict"), "by_sampling_seed": grouped,
        "by_task": by_task, "timing_isolated": summary["timing_isolated"], "planning": timing(durations),
        "first_planning_call": timing(first), "warm_planning_calls": timing(warm)}, table, durations


def main():
    required = {"test_teacher": 20, "test_old_model": 100, "test_final": 100}
    groups, table, times = {}, [], {}
    for name, count in required.items():
        result, rows, durations = aggregate(name, count)
        groups[name] = result; table.extend(rows); times[name] = durations
    latest = groups["test_final"]
    targets = {"all_100_completed": latest["complete_batch"] and latest["completed"] == 100,
        "basic_at_least_95": latest["basic"] >= 95, "strict_at_least_90": latest["strict"] >= 90,
        "every_seed_at_least_18_of_20": len(latest["by_sampling_seed"]) == 5 and all(v["total"] == 20 and v["basic"] >= 18 for v in latest["by_sampling_seed"].values()),
        "planning_p95_at_most_100ms": bool(latest["timing_isolated"] and latest["planning"].get("p95_ms", float("inf")) <= 100)}
    production = read(BASE/"teachers/summary.json")
    manifest = read(BASE/"manifest.json")
    teacher_failures = []
    entries = {e["id"]: e for e in manifest["tasks"]}
    for row in production["episodes"]:
        if not row["passed"]:
            result = read(teacher_path(entries[row["id"]])/"result.json")
            teacher_failures.append({**row, "reasons": reasons(result)})
    report = {"generated_at": datetime.now().isoformat(), "groups": groups, "targets": targets,
        "all_performance_targets_met": all(targets.values()), "dataset": read(BASE/"data_500/summary.json"),
        "teacher_production": {k: v for k, v in production.items() if k != "episodes"},
        "teacher_failures": teacher_failures, "deployment": read(BASE/"test_config_frozen.json"),
        "test_tasks_independent": 20, "test_repeated_runs": 100,
        "limitations": ["Cluster intervals resample the 20 task clusters, not the 100 repeated runs",
            "Planning latency includes state conditioning, network, guidance, projection and selection; model loading and graph capture are outside per-call timing",
            "Old-model timings are throughput measurements with four workers; do not infer a paired speedup from the final table",
            "Error traces are logged at 50 Hz, actual joint limits and sampled collision checks at 500 Hz; no continuous-time certificate",
            "No wall-clock scheduled closed-loop or hard real-time guarantee"]}
    write_json(BASE/"results.json", report)
    keys = sorted({k for r in table for k in r})
    with (BASE/"all_test_runs.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, keys); writer.writeheader(); writer.writerows(table)
    with (BASE/"failures.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, keys); writer.writeheader(); writer.writerows(r for r in table if not r["strict"])
    lines = ["# 第二轮执行结果", "", f"生成时间：{report['generated_at']}。", "",
        "主测试固定 20 个全新任务，每任务 5 个预登记采样种子。训练 seed=42；没有用测试选择模型、删失败任务或换噪声种子。", "",
        "| 方法 | 完整执行 | 基础通过 | 严格通过 | 合并规划 P95/ms | 计时环境 |",
        "|---|---:|---:|---:|---:|---|"]
    labels = {"test_teacher": "教师", "test_old_model": "首轮扩散", "test_final": "第二轮扩散"}
    for key, value in groups.items():
        n = value["expected"]; p = value["planning"].get("p95_ms")
        time_label=f"{p:.2f}" if p is not None else "—"
        scope="教师无扩散采样" if p is None else "单 GPU 推理作业" if value["timing_isolated"] else "并行吞吐测量"
        lines.append(f"| {labels[key]} | {value['completed']}/{n} | {value['basic']}/{n} | {value['strict']}/{n} | {time_label} | {scope} |")
    lines.extend(["", "性能目标：" + ("全部达到。" if all(targets.values()) else "存在未达到的指标，不能认定已实现稳定追踪。"), ""])
    for key, value in targets.items():
        lines.append(f"- {key}：{'达到' if value else '未达到'}")
    lines.extend(["", "## 数据与训练", "",
        f"教师生产 {production['finished']} 次、合格 {production['passed']} 次；合格训练候选 {production['train_admitted']} 条。实际训练选择 {report['dataset']['train_tasks']} 个任务、{report['dataset']['train_windows']} 个窗口；验证 {report['dataset']['validation_tasks']} 个任务、{report['dataset']['validation_windows']} 个窗口。", "",
        "500 条完整轨迹不是 135,000 个独立任务。数据先按任务划分再切片，测试与旧诊断任务排除在标签构建之外。所有生产尝试、备用补样、拟合排除和原始失败见 teachers/summary.json、data_500/coverage.json。", "",
        "| 训练 | 步数 | 全时段最佳验证 v-MSE | 末次启动验证 v-MSE |",
        "|---|---:|---:|---:|"])
    for name in ("train_mid", "train_t1", "train_t2", "train_t3", "train_selected_seed43", "train_selected_seed44"):
        path = BASE/name/"training_report.json"
        if path.exists():
            tr = read(path)
            lines.append(f"| {name} | {tr['steps']} | {tr['best_validation_v_mse']:.6g} | {tr['history'][-1]['startup_validation_v_mse']:.6g} |")
    lines.extend(["", "训练损失衡量离线去噪，闭环性能以实际完整执行为准。训练 seed=43、44 仅作为补充稳定性检查，没有替换主模型。", "",
        "## 验证选型与消融", "", "验证选型记录：selected_validation_config.json；T1/T2 评分比较：screen_results.json；T3 触发与结果：refine_stage.json、refine_results.json（若触发）；完整验证：validation_full/summary.json。", "",
        "| 配置 | 完整次数 | 基础 | 严格 |", "|---|---:|---:|---:|"])
    for pattern in ("screen_*", "seedcheck_*", "ablation_*", "speed_*", "validation_full"):
        for p in sorted(BASE.glob(pattern+"/summary.json")):
            s = read(p); lines.append(f"| {p.parent.name} | {s['finished']}/{s['total']} | {s['passed']} | {s['strict']} |")
    lines.extend(["", "## 新测试种子稳定性", "", "| 采样种子 | 基础 | 严格 |", "|---|---:|---:|"])
    for seed, value in latest["by_sampling_seed"].items():
        lines.append(f"| {seed} | {value['basic']}/{value['total']} | {value['strict']}/{value['total']} |")
    ci = latest["basic_task_cluster_ci95_percent"]
    sci = latest["strict_task_cluster_ci95_percent"]
    lines.extend(["", f"按 20 个任务聚类自举（5,000 次）的 95% 区间：基础通过率 [{ci[0]:.1f}%, {ci[1]:.1f}%]；严格通过率 [{sci[0]:.1f}%, {sci[1]:.1f}%]。这不是 100 个独立任务的置信区间。", "",
        "## 延迟与执行限制", "", f"第二轮合并规划统计：{latest['planning']}。首调用统计：{latest['first_planning_call']}；其余调用：{latest['warm_planning_calls']}。首次调用不包含模型加载和 CUDA Graph 捕获。", "",
        "正式规划统计来自串行单 GPU 推理作业，包含条件读取、网络、引导、投影与筛选。旧模型最终表的计时来自 4 个并行进程，不能据此计算速度提升。另有同任务完整 27 s eager/graph 对照：cuda_graph_full_comparison.json，物理轨迹逐元素一致。", "",
        "每条任务的规划 P50/P95/P99/最大值、超期次数和 2 ms 物理计算统计见 all_test_runs.csv。没有进行实际墙钟调度验收；P95 达标也不等于硬实时。", "",
        "误差保存频率为 50 Hz；关节/力矩与采样碰撞约束按 500 Hz 检查。两者都不能推出连续时间安全证明。", "",
        "## 失败与界面核对", "", "failures.csv 保留所有基础或严格指标未通过的运行，含失败原因、两臂峰值时刻和结果路径。运行异常同样进入分母。教师生产失败：", ""])
    for failure in teacher_failures:
        lines.append(f"- {failure['id']}：{'; '.join(failure['reasons'])}")
    ui = BASE/"ui_verification.json"
    lines.extend(["", f"界面核对：{read(ui) if ui.exists() else '待最终界面完整运行核对，当前不得标记全部交付完成。'}", "",
        "冻结模型、推理配置与数据哈希：test_config_frozen.json。原始轨迹、代码快照及运行协议保存在各实验目录。", ""])
    (BASE/"RESULTS_ZH.md").write_text("\n".join(lines), encoding="utf-8")
    figures(groups, times)
    print({"groups": {k: {f: v[f] for f in ("finished", "completed", "basic", "strict")} for k, v in groups.items()}, "targets": targets}, flush=True)


def figures(groups, times):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    labels = {"test_old_model": "First-round diffusion", "test_final": "Round-2 diffusion"}
    for name, label in labels.items():
        rows = groups[name]["by_sampling_seed"]
        axes[0].plot(list(rows), [100*r["basic"]/r["total"] for r in rows.values()], "o-", label=label)
    axes[0].axhline(90, color="gray", ls="--", label="18/20 per seed")
    axes[0].set(xlabel="Sampling seed", ylabel="Basic pass rate (%)", ylim=(-3, 103))
    axes[0].legend(fontsize=8); axes[0].grid(alpha=.2)
    t = np.sort(np.asarray(times["test_final"])*1000)
    if len(t): axes[1].plot(t, np.arange(1, len(t)+1)/len(t), label="Round-2 single worker")
    axes[1].axvline(100, color="gray", ls="--", label="100 ms target")
    axes[1].set(xlabel="Planning latency (ms)", ylabel="Empirical cumulative fraction", ylim=(0, 1.01))
    axes[1].legend(fontsize=8); axes[1].grid(alpha=.2)
    for ext in ("png", "svg"):
        fig.savefig(BASE/f"final_success_latency.{ext}", dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    main()
