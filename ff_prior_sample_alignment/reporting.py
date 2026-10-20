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
    for other in ("plain", "structured", "learned_mean"):
        for stage in ("candidate", "execution"):
            answer["paired"][f"aligned_vs_{other}_{stage}"] = paired(mapped[other][stage], mapped["aligned"][stage])
    for sj in range(2):
        m = {}
        for kind in METHODS:
            rs = [read_json(out / "unguided" / str(sj) / kind / f"{j:02d}.json") for j in range(48)]
            answer["unguided"][f"{sj}_{kind}"] = aggregate(rs)
            m[kind] = {str(j): pose(r) for j, r in enumerate(rs)}
        answer["paired"][f"unguided_seed{sj}"] = paired(m["plain"], m["aligned"])
    atomic_json(out / "paired_statistics.json", answer)
    return answer


def report(out, entries, audit):
    stats = statistics(out, entries)
    from .application import APP_METHODS
    app = {}
    for e in read_json(out / "frozen_tasks.json")["application"]["entries"]:
        sid = e["sample_id"]
        app[sid] = {"initial_status": e["initial_status"], "methods": {}}
        for kind in APP_METHODS:
            root = out / "application" / sid / kind
            cs = [read_json(p) if p.exists() else {} for p in [root / "candidates" / f"{k:02d}.json" for k in range(5)]]
            ep = root / "execution/report.json"
            op = root / "execution/outcome.json"
            er = read_json(ep) if ep.exists() else {}
            outcome = read_json(op) if op.exists() else {"status": "INITIAL_BLOCKED"}
            app[sid]["methods"][kind] = {"candidate": aggregate(cs), "execution": aggregate([er], True),
                                       "outcome": outcome, "new_physics": True}
    atomic_json(out / "application_statistics.json", app)
    status = "FINISHED" if audit["status"] == "PASS" else "FINISHED_WITH_BLOCKED_ITEMS"
    summary = {"status": status, "raw_audit": audit["status"],
               "engineering": {k: audit["status"] == "PASS" and stats["methods"][k]["engineering_counts_pass"]
                               and stats["methods"][k]["engineering_no_unknown_errors"] for k in METHODS},
               "updates_per_model": 1200, "TRAIN": 14222, "VAL": 48,
               "new_DEV_candidates": 360, "new_DEV_execution_units": 72,
               "reused_candidates": 360, "reused_execution_units": 72,
               "application": app, "TEST": False, "expand": False, "physics_relaxed": False,
               "claim_verdict": "PENDING_FRESH_INDEPENDENT_REVIEW"}
    atomic_json(out / "summary.json", summary)
    lines = ["# 完整采样世界目标对齐：后台结果", "",
             f"运行状态{status}，原始审计{audit['status']}；独立主张审查待收尾。",
             "同一个14222旧模型温启动、同普通训练序列，两组各1200更新；aligned增加完整16步教师及真实世界终点监督。",
             "原冻结DEV新360/72；旧简单基线360/72仅复用。静态抓捕三/五障碍独立列出，非原DEV或TEST。", "",
             "|方法|DEV候选合格/180|执行重放/36|执行位置均值mm|",
             "|---|---:|---:|---:|"]
    for k, m in stats["methods"].items():
        lines.append(f"|{k}|{m['candidate']['qualified']}|{m['execution_verified']}|{m['execution']['position_mm'].get('mean','缺失')}|")
    lines += ["", "|完整无引导|种子1合格/48|种子2合格/48|", "|---|---:|---:|"]
    for k in METHODS:
        lines.append(f"|{k}|{stats['unguided']['0_'+k]['qualified']}|{stats['unguided']['1_'+k]['qualified']}|")
    lines += ["", "|抓捕场景|方法|候选合格/5|执行状态|", "|---|---|---:|---|"]
    for sid, s in app.items():
        for k, r in s["methods"].items():
            lines.append(f"|{sid}|{k}|{r['candidate']['qualified']}|{r['outcome']['status']}|")
    lines += ["", "仅MODEL_VALID候选不升级为执行通过。所有失败/缺失保留分母。",
              "本轮没有动态航点训练或接触抓取；静态抓捕是无接触预抓捕接近与保持。",
              "尚未放宽物理门、扩库或启动最终TEST。相关决策在新独立审查及中文ANALYSIS后作出，不依据单个指标自动追加实验。"]
    (out / "REPORT.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    return summary
