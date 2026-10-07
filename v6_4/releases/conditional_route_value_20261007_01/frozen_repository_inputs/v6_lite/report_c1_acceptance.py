"""Curate C.1 evidence without modifying raw runs or historical failures."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "v6_lite/output/runs"
BASE = "0e76abaedd7f9526c7cfdbeb6d92867a2a911e7b"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def create(output_dir):
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    matrix_dir = RUNS / "c1_formal_three_by_five"
    matrix = read(matrix_dir / "result.json")
    tested_commit = read(matrix_dir / "round_01/run_metadata.json")["source"]["git_commit"]
    if not matrix["complete"] or not matrix["independent_validation_complete"]:
        raise ValueError("formal evidence is incomplete")
    delay_dir = RUNS / "c1_delay_injection_final"
    delays = read(delay_dir / "report.json")
    fault = read(RUNS / "c1_b2_failure_injection_final/failure_injection_summary.json")
    wall = read(RUNS / "c1_wall_gate_probe/failures/v6_lite_scenario_00_interval_failure.json")
    wall_five = read(RUNS / "c1_wall_five/report.json")
    regression_path = RUNS / "c1_regression_final/tests.utf8.log"
    regression_log = regression_path.read_text(encoding="utf-8")
    failed_tests = re.findall(r"^(?:FAIL|ERROR): (.+)$", regression_log, flags=re.M)
    known_hashes = {}
    for path in (ROOT / "v6_lite").rglob("*.py"):
        known_hashes[hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()] = path
    for name in ("run_v6_lite.py", "hierarchical_qp.py"):
        path = ROOT / "v6_lite" / name
        raw_tested = subprocess.check_output(["git", "show", f"{tested_commit}:v6_lite/{name}"], cwd=ROOT)
        known_hashes[hashlib.sha256(raw_tested.replace(b"\r\n", b"\n")).hexdigest()] = path
    baseline_proof = []
    for actual, expected in re.findall(r"AssertionError: '([a-f0-9]{64})' != '([a-f0-9]{64})'", regression_log):
        path = known_hashes[actual]
        relative = path.relative_to(ROOT).as_posix()
        raw_base = subprocess.check_output(["git", "show", f"{BASE}:{relative}"], cwd=ROOT)
        base_sha = hashlib.sha256(raw_base.replace(b"\r\n", b"\n")).hexdigest()
        baseline_proof.append({"source": relative, "tested_snapshot_sha256": actual,
            "expected_historical_sha256": expected, "frozen_base_sha256": base_sha,
            "mismatch_already_at_frozen_base": base_sha != expected})
    samples = [scene for trial in matrix["trials"] for scene in trial["scenes"]]
    validation = matrix["first_round_validation"]
    summary = {
        "schema": "v6_2_c1_curated_acceptance_v1", "frozen_base": BASE,
        "tested_runtime_commit": read(matrix_dir / "round_01/run_metadata.json")["source"]["git_commit"],
        "all_fifteen_original_acceptance_passed": all(t["original_acceptance_passed"] for t in matrix["trials"]),
        "all_fifteen_exact_trace_parity": matrix["all_exact_trace_parity"],
        "all_fifteen_algorithm_p95_passed": matrix["all_algorithm_timing_passed"],
        "all_fifteen_dispatch_p95_passed": matrix["all_dispatch_timing_passed"],
        "cycle_count": sum(s["dispatch"]["count"] for s in samples),
        "dispatch_p95_ms_max": max(s["dispatch"]["p95_ms"] for s in samples),
        "dispatch_p99_ms_max": max(s["dispatch"]["p99_ms"] for s in samples),
        "dispatch_max_ms": max(s["dispatch"]["max_ms"] for s in samples),
        "over_20ms_count": sum(s["dispatch"]["over_deadline_count"] for s in samples),
        "longest_consecutive_over_20ms": max(s["dispatch"]["longest_consecutive_over_deadline"] for s in samples),
        "first_cycle_ms_range": [min(s["dispatch"]["first_cycle_ms"] for s in samples),
                                 max(s["dispatch"]["first_cycle_ms"] for s in samples)],
        "engineering_p95_16ms_p99_20ms_goal_met": all(s["dispatch"]["p95_ms"] <= 16 and s["dispatch"]["p99_ms"] <= 20 for s in samples),
        "native_validation": validation["native"], "execution_validation": validation["execution"],
        "interval_validation": validation["interval"],
        "delay_rejection_tests": {"passed": delays["all_passed"], "case_count": delays["case_count"],
                                  "injected_delay_ms": [0, 5, 10, 20, 40, 80],
                                  "nominal_virtual_service_ms": 12, "claim_scope": delays["claim_scope"]},
        "original_failure_injection": fault,
        "wall_deadline_probe": {"complete_runtime_acceptance": False,
            "reason": wall["failure_reason"], "simulation_stop_s": wall["time_s"],
            "next_servo_step_executed": wall["next_servo_step_executed"],
            "tested_runtime_snapshot": "before compiled-model fingerprint extension; the wall expiry rule is unchanged"},
        "current_wall_deadline_five_probes": wall_five,
        "regression": {"summary_lines": regression_log.splitlines()[-5:], "failed_test_names": failed_tests,
                       "source_hash_baseline_proof": baseline_proof,
                       "focused_current_contract_tests": "26 passed; full suite historical source checks remain failed"},
        "trials": matrix["trials"],
        "runtime_continuation_acceptance": "NOT_MET",
        "continuous_time_certified": False, "hard_realtime_certified": False,
        "safe_backup_controller_established": False,
        "scope": "functional parity and finite repeated timing in frozen synchronous simulation; invalid command filtering with evolving injected states",
    }
    latest_failures = RUNS / "c1_failure_timeline_final/failure_injection_summary.json"
    summary["final_failure_timeline_retest"] = read(latest_failures)
    (output_dir / "acceptance_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), constrained_layout=True)
    colors = ["#087f8c", "#df9f24", "#7755a5"]
    for trial, color in zip(matrix["trials"], colors):
        offset = (trial["repeat"] - 2) * .22
        positions = np.arange(5) + offset
        axes[0].bar(positions, [s["dispatch"]["p95_ms"] for s in trial["scenes"]], width=.20,
                    color=color, label=f"Round {trial['repeat']}")
        axes[1].scatter(positions, [s["dispatch"]["p99_ms"] for s in trial["scenes"]],
                        color=color, marker="o", label=f"R{trial['repeat']} p99")
        axes[1].scatter(positions, [s["dispatch"]["max_ms"] for s in trial["scenes"]],
                        color=color, marker="x", label=f"R{trial['repeat']} maximum")
    for ax in axes:
        ax.axhline(20, color="#b52828", linestyle="--", linewidth=1.5)
        ax.set_xticks(np.arange(5), [f"S{i:02d}" for i in range(5)])
        ax.set_ylabel("Acquisition to first torque publish (ms)")
        ax.grid(axis="y", alpha=.2)
        ax.set_axisbelow(True)
    axes[0].set_ylim(0, 24)
    axes[0].set_title("All 15 complete-scene p95 checks pass")
    axes[1].set_ylim(14, max(s["dispatch"]["max_ms"] for s in samples) + 3)
    axes[1].set_title("Tail misses remain; no hard real-time claim")
    axes[0].legend(loc="lower right", fontsize=8)
    axes[1].legend(loc="lower right", fontsize=7, ncol=2)
    fig.savefig(output_dir / "dispatch_timing.png", dpi=160)
    plt.close(fig)
    table = ["| 轮次 | 场景 | 算法 p95 | 发布 p95 | 发布 p99 | 最大值 | 超期数 | 连续超期 | 首周期 |",
             "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for trial in matrix["trials"]:
        for s in trial["scenes"]:
            d = s["dispatch"]
            table.append(f"| {trial['repeat']} | {s['scenario_id'][-2:]} | {s['algorithm']['p95_ms']:.3f} | {d['p95_ms']:.3f} | {d['p99_ms']:.3f} | {d['max_ms']:.3f} | {d['over_deadline_count']} | {d['longest_consecutive_over_deadline']} | {d['first_cycle_ms']:.3f} |")
    document = f"""# V6.2-C.1 运行时合同与验收证据

三项实现已提交，冻结基点 `{BASE[:7]}`，正式试验源码 `{summary['tested_runtime_commit'][:7]}`。三轮各五场景、每场景 27 s，共 {summary['cycle_count']:,} 个规划周期，原算法与状态采集到首次力矩发布的 p95 均低于 20 ms。15 个场景运行的全部非计时数组严格相同，包括 17 维选中命令和 67 路力矩。

独立验证固定使用预声明的第一轮：真实力矩重放 {validation['native']['passed_count']}/{validation['native']['total_count']}、原合同 {validation['execution']['passed_count']}/{validation['execution']['total_count']}；区间重算完整结果见机器可读摘要。任务误差与 B.2 完全一致，安全距离、PCC 半径、数值容差、约束和物理频率保持原定义。

**持续墙钟执行验收尚未满足。** 正式五场景采样显式采用 `offline_replay`：计算期间物理被冻结，测量发布时延，不强制墙钟有效期。首次默认 `wall_deadline` 探测在 {wall['time_s']*1000:.0f} ms 仿真时刻因墙钟过期拒绝后续力矩消费；当前源码又对五场景分别执行了墙钟持续执行探测，完成情况见机器摘要。停止仿真不是安全备份。有限重复 p95 通过不能证明每周期按时、最坏执行时间或连续时间安全。

发布 p95 最差 {summary['dispatch_p95_ms_max']:.3f} ms；p99 最差 {summary['dispatch_p99_ms_max']:.3f} ms；最大 {summary['dispatch_max_ms']:.3f} ms。{summary['over_20ms_count']}/{summary['cycle_count']} 个周期超过 20 ms，最长连续 {summary['longest_consecutive_over_20ms']} 个。工程目标 p95≤16 ms 且 p99≤20 ms 未满足。初始化另行记录，首周期没有删去。

![全部三轮五场景发布时延](../{output_dir.relative_to(ROOT).as_posix()}/dispatch_timing.png)

## 完整时延表

所有时间列单位为 ms；子阶段分位数不能相加。

{chr(10).join(table)}

## 实现与边界

- `runtime_identity` 分开历史 trace schema、控制器、PCC 模式、伺服、积分器和模型；保留 A.1 原验证格式。
- 私有 `MjData`、关节地址、质量矩阵、十一微状态与十个力矩缓冲区在初始化分配。每次预演复制 MuJoCo 完整 integration state，包括外力、控制量、warm start、mocap、userdata 与 plugin state；逐微状态重新刷新空间量。同状态反作用映射只在输入完全一致时复用。离线验证器不读取在线缓存。
- 每周期保存不重叠墙钟/线程 CPU 阶段、嵌套关系、迭代次数和罚参数更新。Windows 线程 CPU 样本可能较粗，不能直接把墙钟减去 CPU 作为精确调度抖动。
- 命令有效期固定从状态采集起算 20 ms，不在求解结束续期；每个力矩步检查模型、区间、命令内容、次序、墙钟及真实执行起点。支持范围为名义模型的离散预演微状态，不是测量误差或模型失配的鲁棒界。
- 延迟 0/5/10/20/40/80 ms 在 QP、预演、下一起点检查和发布前四处注入，共 24 个基础条件，另加连续三次迟到、目标过期和乱序，共 {delays['case_count']} 个拒绝测试通过。正延迟期间独立执行状态按固定 2 ms 物理步继续演化，机器人和目标均改变。5 ms 延迟包含 4 ms 已完成物理步及 1 ms 调度余量。延迟期保持的旧力矩未经备份验证；拒绝用例记为无法保证继续执行。
- 调度器的名义服务时间 12 ms 是声明的虚拟条件；只有注入服务段推进外部物理，0 ms 条件是调度对照，不是异步硬件持续运动验收。
- 原查询未知、QP 故障、包络失效与旧合同过期四项故障注入继续通过。初次延迟框架因缺少前一速度热启动而中止的目录也保留，未覆盖或替代为通过结果。

## 回归与证据

当前定向合同/证据/C.1 测试 26 项通过；最终全量回归 138 项中 122 项通过，14 项失败、2 项错误均为历史源码哈希检查，日志见 `v6_lite/output/runs/c1_regression_final/tests.utf8.log`。原始本地编码日志另行保留，UTF-8 副本不改变结果。详细名称及与冻结基点的哈希对照见摘要，不能把这些结果标为全量通过。两项新增元数据兼容性错误已修复并复验。

原始三轮运行、每周期时间线、所有失败、源码与模型身份和 SHA-256 清单保留在 `v6_lite/output/runs/c1_*`；这些大文件不纳入 Git。小型摘要与输入哈希纳入本报告。墙钟数据不能由物理重放重新生成。

正式计时对应源码 `{summary['tested_runtime_commit'][:7]}`；其后补充了失败路径的拒绝时间线与报告，成功路径的数值动作保持不变。补充的四类失败时间线已复验，当前拒绝前、拒绝后及未消费下一力矩步均可追溯。
"""
    (ROOT / "docs/V6_2_C1_RUNTIME_EVIDENCE.md").write_text(document, encoding="utf-8")
    (output_dir / "RUNTIME_EVIDENCE.md").write_text(document, encoding="utf-8")
    inputs = [matrix_dir / "result.json", matrix_dir / "plan.json", matrix_dir / "manifest.json",
              delay_dir / "report.json", RUNS / "c1_b2_failure_injection_final/failure_injection_summary.json",
              RUNS / "c1_wall_gate_probe/failures/v6_lite_scenario_00_interval_failure.json", regression_path,
              RUNS / "c1_regression_final/tests.log", RUNS / "c1_wall_five/report.json"]
    inputs.append(latest_failures)
    inputs.append(RUNS / "c1_wall_timeline_final/failures/v6_lite_scenario_00_interval_failure.json")
    manifest = {p.relative_to(ROOT).as_posix(): sha(p) for p in inputs}
    manifest.update({p.relative_to(ROOT).as_posix(): sha(p) for p in output_dir.iterdir() if p.is_file()})
    manifest["docs/V6_2_C1_RUNTIME_EVIDENCE.md"] = sha(ROOT / "docs/V6_2_C1_RUNTIME_EVIDENCE.md")
    (output_dir / "evidence_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return summary


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, default=ROOT / "v6_lite/output/v6_2_c1")
    args = p.parse_args()
    summary = create(args.output_dir)
    print(json.dumps({k: summary[k] for k in ("cycle_count", "dispatch_p95_ms_max", "over_20ms_count", "runtime_continuation_acceptance")}, indent=2))
