"""Aggregate every pre-registered evaluation, including failed runs."""
import csv
import json
from pathlib import Path
import numpy as np
from .pipeline import write_json


def main():
    folder=Path("tracking_output/paper100_v1")
    read=lambda p:json.loads(p.read_text(encoding="utf-8"))
    teacher=read(folder/"teachers/summary.json")
    feedback=read(folder/"feedback_a01/summary.json")
    diffusion=read(folder/"test_a01/summary.json")
    training=read(folder/"train_a01/training_report.json")
    coverage=read(folder/"data/coverage.json")
    if diffusion["finished"]!=30 or feedback["finished"]!=10:
        raise ValueError("Evaluation still running; do not report a partial success rate as final")
    rows=[]
    for method,episodes in (("teacher",[e for e in teacher["episodes"] if e["split"]=="test"]),
                            ("feedback-only",feedback["episodes"]),("guided-diffusion",diffusion["episodes"])):
        for e in episodes:
            result=read(Path(e["result_path"])) if Path(e["result_path"]).is_file() else {}
            m,x=result.get("metrics",{}),result.get("execution",{})
            row={"method":method,"task_id":e["id"],"sampling_seed":e.get("sampling_seed"),
                 "completed":e["completed"],"basic_pass":e["passed"],"strict_pass":e["strict"],
                 "failure":e["failure"],"c_path_mm":m.get("continuum",{}).get("path_rmse_mm"),
                 "c_max_mm":m.get("continuum",{}).get("position_max_mm"),
                 "r_path_mm":m.get("rigid",{}).get("path_rmse_mm"),
                 "r_final_mm":m.get("rigid",{}).get("final_position_mm"),
                 "r_steady_mm":m.get("rigid",{}).get("steady_rmse_mm"),
                 "margin_mm":None if x.get("minimum_pair_margin_m") is None else 1000*x["minimum_pair_margin_m"],
                 "joint_violation_steps":x.get("joint_limit_violation_steps"),
                 "torque_saturations":x.get("torque_saturation_count"),
                 "result_path":e["result_path"]}
            trace=Path(e["result_path"]).parent/"trace.npz"
            row["c_peak_time_s"] = None
            if trace.is_file():
                with np.load(trace) as data:
                    index=int(np.argmax(np.linalg.norm(data["error"][:,:3],axis=1)))
                    row["c_peak_time_s"]=float(data["time"][index])
            rows.append(row)
    counts={}
    for name in ("teacher","feedback-only","guided-diffusion"):
        selected=[r for r in rows if r["method"]==name]
        counts[name]={"attempts":len(selected),"completed":sum(r["completed"] for r in selected),
                      "basic_pass":sum(r["basic_pass"] for r in selected),
                      "strict_pass":sum(r["strict_pass"] for r in selected)}
    by_seed=[]
    fb={r["task_id"]:r for r in rows if r["method"]=="feedback-only"}
    for seed in (73,137,211):
        selected=[r for r in rows if r["method"]=="guided-diffusion" and r["sampling_seed"]==seed]
        paired=[r for r in selected if r["completed"] and fb[r["task_id"]]["completed"]]
        before=np.mean([(fb[r["task_id"]]["c_path_mm"]+fb[r["task_id"]]["r_path_mm"])/2 for r in paired]) if paired else None
        after=np.mean([(r["c_path_mm"]+r["r_path_mm"])/2 for r in paired]) if paired else None
        by_seed.append({"sampling_seed":seed,"attempts":len(selected),
                        "completed":sum(r["completed"] for r in selected),
                        "basic_pass":sum(r["basic_pass"] for r in selected),
                        "strict_pass":sum(r["strict_pass"] for r in selected),
                        "paired_complete_tasks":len(paired),"feedback_mean_path_mm":before,
                        "diffusion_mean_path_mm":after,
                        "relative_reduction_percent":100*(before-after)/before if before else None})
    summary={"counts":counts,"by_sampling_seed":by_seed,
             "teacher_production":{"attempted":100,"passed":teacher["passed"],"strict":teacher["strict"]},
             "training":{k:v for k,v in training.items() if k!="history"},
             "excluded_supervision":[e for e in coverage["episodes"] if not e["admitted"]],
             "rows":rows,"all_test_failures_retained":True,
             "limitations":["Only 10 held-out tasks, 3 sampling seeds each; repeats are not 30 independent tasks",
                "Guidance uses a measured-state local kinematic linearization, not an exact dynamic rollout",
                "Final joint projection is additional to gradient guidance",
                "Feedback-only excludes analytic reference feedforward; teacher is the stronger non-learning baseline",
                "Batch timings are concurrent throughput timings, not isolated real-time latency"]}
    write_json(folder/"comparison.json",summary)
    with (folder/"test_results.csv").open("w",newline="",encoding="utf-8-sig") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)
    lines=["# 论文流程执行结果", "", "数据、模型、引导和真实力矩执行均已落地。以下为本次固定测试集的全部结果。", "",
           f"100 个新任务均完成教师执行，{teacher['passed']} 条通过基础验收。训练划分原为 80/10/10；最终 {training['train_tasks']} 个合格训练任务、{training['validation_tasks']} 个验证任务参与学习。",
           f"训练窗口 {training['train_windows']}，验证窗口 {training['validation_windows']}；优化 {training['steps']} 步。模型为三层 AdaLN Transformer，60 步 DDIM、后 18 步约束引导。", "",
           "| 方法 | 完整执行 | 基础验收 | 严格精度 |", "|---|---:|---:|---:|"]
    for method,entry in counts.items():
        n=entry["attempts"]
        lines.append(f"| {method} | {entry['completed']}/{n} | {entry['basic_pass']}/{n} | {entry['strict_pass']}/{n} |")
    lines += ["", "| 扩散采样种子 | 基础验收 | 严格精度 |", "|---|---:|---:|"]
    for entry in by_seed:
        lines.append(f"| {entry['sampling_seed']} | {entry['basic_pass']}/10 | {entry['strict_pass']}/10 |")
    lines += ["", "基础验收包括完整 27 秒、双臂全程位置最大≤3 mm、姿态最大≤0.5°、实际关节与力矩限值、碰撞余量。严格指标额外要求柔性臂路径 RMSE≤0.18 mm、刚性臂最终≤0.10 mm、末段稳态≤0.15 mm、姿态最大≤0.25°。",
              "", "失败运行未从分母中删除；30 次采样对应 10 个独立任务。逐运行误差、约束和路径保存在 test_results.csv / comparison.json。",
              "", "教师是使用解析参考速度的无学习控制器。仅反馈没有这一前馈项，因此优于仅反馈不能据此宣称扩散优于教师或所有传统控制器。关闭引导的首轮消融同时关闭了最终关节投影，不能把全部差异单独归因于梯度。",
              "", "原文的常量绳长映射没有套用于当前力矩模型；这里以自由基座广义雅可比、关节边界和碰撞距离引导，实际力矩在执行中检查。引导是局部近似，仍须看全程执行结果。",
              "", "界面 http://127.0.0.1:8765 显示真实采样与动力学结果；可关闭引导并回放。当前未证明规划和控制计算满足硬实时期限。"]
    failed=[r for r in rows if r["method"]=="guided-diffusion" and not r["basic_pass"]]
    if failed:
        peaks=[r["c_peak_time_s"] for r in failed if r["c_peak_time_s"] is not None]
        lines += ["", f"本次 {len(failed)} 次扩散基础验收失败全部保留。其柔性臂峰值时刻范围为 {min(peaks):.3f}–{max(peaks):.3f} s；逐项见 CSV。这支持继续检查启动窗口的生成与衔接，不能直接断言任务不可达，也不能单凭该相关性断定唯一根因。"]
    (folder/"RESULTS_ZH.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    print(json.dumps({"counts":counts,"by_seed":by_seed},ensure_ascii=False,indent=2))


if __name__=="__main__":
    main()
