from ff_prior.storage import read_json
from ff_prior_pilot_train.storage import atomic_json
from ff_prior_endpoint_jacobian.statistics import aggregate, pose, paired
from .protocol import P, CONTROL, METHODS


def statistics(out, entries):
    answer = {"scope": "12 already-used DEV tasks, three sampling seeds, one training seed; not TEST",
              "methods": {}, "unguided": {}, "paired": {}}
    mapped = {}
    for kind in (*METHODS, *P["reuse_methods"]):
        root = out if kind in METHODS else CONTROL
        cs, es, outcomes, cm, em = [], [], [], {}, {}
        for sj in range(3):
            for e in entries:
                sid = e["sample_id"]
                for k in range(5):
                    p = root / "candidates" / str(sj) / sid / kind / f"{k:02d}.json"
                    r = read_json(p) if p.exists() else {}
                    cs.append(r)
                    cm[f"{sj}/{sid}/{k}"] = pose(r)
                directory = root / "executions" / str(sj) / sid / kind
                r = read_json(directory / "report.json") if (directory / "report.json").exists() else {}
                o = read_json(directory / "outcome.json") if (directory / "outcome.json").exists() else {}
                es.append(r)
                outcomes.append(o)
                em[f"{sj}/{sid}"] = pose(r, True)
        success = sum(o.get("status") == "EXECUTION_VERIFIED" for o in outcomes)
        qualified = sum(r.get("eligible", False) for r in cs)
        answer["methods"][kind] = {
            "candidate": aggregate(cs), "execution": aggregate(es, True),
            "execution_units": 36, "execution_verified": success,
            "execution_statuses": {s: sum(o.get("status", "MISSING") == s for o in outcomes)
                                   for s in sorted({o.get("status", "MISSING") for o in outcomes})},
            "candidate_errors_or_missing": sum(r.get("status") != "DONE" for r in cs),
            "engineering_counts_pass": qualified >= 144 and success >= 29,
            "engineering_no_unknown_errors": all(r.get("status") == "DONE" for r in cs) and
                all(o.get("status") not in ("ERROR", None) for o in outcomes),
            "new_physics": kind in METHODS}
        mapped[kind] = {"candidate": cm, "execution": em}
    for other in ("mlp", "structured", "learned_mean"):
        for stage in ("candidate", "execution"):
            answer["paired"][f"transformer_vs_{other}_{stage}"] = paired(mapped[other][stage], mapped["transformer"][stage])
    for sj in range(2):
        m = {}
        for kind in METHODS:
            rs = [read_json(out / "unguided" / str(sj) / kind / f"{j:02d}.json") for j in range(48)]
            answer["unguided"][f"{sj}_{kind}"] = aggregate(rs)
            m[kind] = {str(j): pose(r) for j, r in enumerate(rs)}
        answer["paired"][f"unguided_seed{sj}"] = paired(m["mlp"], m["transformer"])
    atomic_json(out / "paired_statistics.json", answer)
    return answer


def report(out, entries, audit):
    stats = statistics(out, entries)
    training = {k: read_json(out / "learning" / k / "training.json") for k in METHODS}
    status = "FINISHED" if audit["status"] == "PASS" else "FINISHED_WITH_BLOCKED_ITEMS"
    summary = {"status": status, "raw_audit": audit["status"],
               "engineering": {k: audit["status"] == "PASS" and stats["methods"][k]["engineering_counts_pass"]
                               and stats["methods"][k]["engineering_no_unknown_errors"] for k in METHODS},
               "new_candidates": 360, "new_execution_units": 72,
               "reused_candidates": 360, "reused_execution_units": 72,
               "TRAIN": 14222, "VAL": 48, "updates_per_model": {k: t["updates"] for k, t in training.items()},
               "claim_verdict": "PENDING_FRESH_INDEPENDENT_REVIEW", "TEST": False, "expand": False}
    atomic_json(out / "summary.json", summary)
    lines = ["# MLP／Transformer 同表示有界比较：运行结果", "",
             f"后台阶段状态：{status}。原始审计：{audit['status']}。独立科学审查待完成。",
             "两个去噪网络从头训练，共用原 TRAIN 拟合 codec；每组6000更新，单训练种子，不作普遍架构结论。",
             "仅新增两组360候选／72执行单元；旧简单基线360候选／72执行单元为身份匹配复用。", "",
             "|方法|候选合格/180|执行重放/36|执行位置均值/mm|P95/mm|来源|",
             "|---|---:|---:|---:|---:|---|"]
    for k, m in stats["methods"].items():
        p = m["execution"]["position_mm"]
        lines.append(f"|{k}|{m['candidate']['qualified']}|{m['execution_verified']}|{p.get('mean','缺失')}|{p.get('p95','缺失')}|{'新增' if m['new_physics'] else '历史复用'}|")
    lines += ["", "|无物理引导|种子1合格/48|种子2合格/48|", "|---|---:|---:|"]
    for k in METHODS:
        lines.append(f"|{k}|{stats['unguided']['0_'+k]['qualified']}|{stats['unguided']['1_'+k]['qualified']}|")
    lines += ["", "完整分母、缺失端点、逐任务/种子配对、姿态和尾部见 paired_statistics.json。",
              "模型层合格不等于执行资格。只有实际力矩仿真和独立重放通过的样本才计 EXECUTION_VERIFIED。",
              "工程通过不要求相对胜出；负结果不追加训练。未扩库、未启动pilot/study/strict_multimodal/TEST。"]
    (out / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary
