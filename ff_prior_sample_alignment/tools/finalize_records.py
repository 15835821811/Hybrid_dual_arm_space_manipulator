"""Version the completed experiment's goal/tracker and independent review trace."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PKG = ROOT / "ff_prior_sample_alignment"
OUT = ROOT / "artifacts/v7_1/sample_alignment_001"
TRACE = ROOT / "paper/review-traces/experiment-result-to-claim/20261020_sample_alignment_run01"
EV = PKG / "evidence/completed"
def read(p):
    return json.loads(p.read_text(encoding="utf-8"))
def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p, text):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")
def dump(p, value):
    write(p, json.dumps(value, ensure_ascii=False, indent=2))

verdict = read(TRACE / "verdict.json")
assert verdict["engineering"]["bounded_completion"] == "PASS"
assert not verdict["TEST"]["ready"]
assert read(EV / "resume_verification.json")["actual_resume_calls"] == 1
assert all(sha(ROOT / n) == h for n, h in read(PKG / "evidence/preflight.json")["dirty_file_hashes"].items())
stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
goal = """# 当前执行目标收尾：完整采样对齐与静态避障预抓捕

2026-10-20；用户确认先静态避障抓捕、后动态不规则航点。
本轮 sample_alignment_001 已完成；源码b242dbeb，启动登记7699382f。
有效冻结协议ff_prior_sample_alignment/docs/EXPERIMENT_PLAN.md；结果ff_prior_sample_alignment/docs/ANALYSIS.md。
本轮完成指有界实现、实验、审查和交付，不代表长期可靠规划或科学优势全部完成。

|阶段|状态|实测结果|
|---|---|---|
|实现/冻结/M0|PASS|9软件测试；173源码/29027输入绑定；2固定TRAIN完整采样世界梯度数值检查|
|M1配对训练|DONE|同14222起点，两组各1200实际更新；全部TRAIN参与；候选1200完整16步教师批次＋60世界事件|
|M2无引导|DONE；有界C1支持|plain9/96→aligned24/96；位置均值16.635→13.769mm，尾部改善，仍72/96失败|
|M3原冻结DEV|工程PASS；C2优势未建立|两新模型各180/180合格、36/36执行重放；执行位置均值0.7125→0.7177mm，收益混合|
|M4三障碍|有限场景PASS|两Diffusion各5/5及1次真实MuJoCo执行重放；不是接触抓取|
|M4五障碍|实验DONE；应用BLOCKED|所有方法0/5；aligned终点5/5过但全部碰撞；不改目标/障碍|
|M5审计/恢复/报告|PASS|全新独立审查、64保存更新/60VJP重建；一次resume，1403核心文件清单/hash不变|
|最终TEST|BLOCKED，未启动|五障碍应用未满足冻结门；相对优势不是TEST前提|

新410规划候选：374 MODEL_VALID，但同时满足指定目标的全门合格373；
新增75次实际力矩执行重放通过（DEV72＋应用3），另7应用单元NO_CANDIDATE。
旧基线360候选/72执行仅复用。无扩库、无物理门放宽、无TEST或动态航点运行。
全部旧模型、结果、用户修改和其他作业保留。额外训练rollout不算新增合格数据。

独立review裁决C1=yes（只限冻结单训练种子/既有VAL的组合方法比较），C2=no，总体partial。
原环境OMP/MKL/OPENBLAS=2及torch线程2下只读重建通过；未改变数值断言。
论文claim audit缺失，下游论文写作裁决provisional，不妨碍本轮工程完成。

下一阶段建议（未启动）：固定五障碍的可行性与引导诊断，先区分是否存在合法路径、
当前102维表示是否能够表达、目标保持碰撞求解为何退回goal-only。仅在找到可行见证后
判断是否补少量独立TRAIN场景教师，避免把当前开发场景加入训练后仍称泛化验证。
条件均值的2mm几何检查密度不足应通过明确预算下加密验收处理，不追认旧失败。
静态闭环通过后再冻结未触碰TEST协议，之后才接动态不规则航点；不自动新开实验。

终端入口 .\\ff_prior_sample_alignment\\tools\\job.ps1 status|report|resume。
完成态不重复工作；本轮恢复核验已经做过一次，不需要再次核验。
"""
write(PKG / "docs/GOAL_CLOSEOUT.md", goal)
write(ROOT / f"paper/EXPERIMENT_TRACKER_{stamp}.md", goal)
write(ROOT / "paper/EXPERIMENT_TRACKER.md", goal)
write(ROOT / "DIFFUSION_GOAL_CURRENT.md", goal)
finding = """
## sample_alignment_001：完整采样对齐与静态应用（2026-10-20）

两组同14222权重及普通序列各1200更新；aligned增加完整16步教师与60真实世界端点事件。
无引导9/96→24/96，位置均值16.635→13.769mm，P95/最坏改善；C1仅有界比较yes。
原DEV两模型各180/180和36/36，执行位置均值0.7125→0.7177mm；C2=no，总体partial。
三障碍两Diffusion5/5及选定执行通过，五障碍全方法0/5；aligned穿透39.46–52.37mm，
20条局部求解记录16条goal-only回退。路径不可达尚未证实，覆盖缺口只是待检验假设。
410新候选374 MODEL_VALID、373同时命中目标；75次新执行重放、7无候选单元；
旧360/72仅复用，不升级未执行样本。保持所有原物理门，无扩库/TEST。
新独立审查核173源码/29027输入、64保存AdamW、60世界VJP、原始候选执行；
精确BLAS环境2解决1.49e-12重建微差，无容差更改。重放第二完整轨迹未保存。
一次完成resume，1403核心文件清单/hash不变。建议下一项是固定五障碍可行路径与
目标保持碰撞引导诊断，不盲目扩库/换主干/反复加端点训练。当前TEST门未满足。
独立review: paper/review-traces/experiment-result-to-claim/20261020_sample_alignment_run01。
paper claim audit不可用，论文写作provisional；本轮完成不等于长期目标完成。
"""
write(PKG / "docs/findings.md", finding)
root_finding = ROOT / "findings.md"
if "## sample_alignment_001：" not in root_finding.read_text(encoding="utf-8"):
    with root_finding.open("a", encoding="utf-8") as f:
        f.write("\n" + finding)
entry = f"\n| {stamp} | experiment-result-to-claim / closeout | ff_prior_sample_alignment/docs/ANALYSIS.md; paper/EXPERIMENT_TRACKER_{stamp}.md; paper/review-traces/experiment-result-to-claim/20261020_sample_alignment_run01 | 工程完成PASS；C1有界yes/C2 no；三障碍通过、五障碍BLOCKED；无扩库/TEST |\n"
for p in (ROOT / "MANIFEST.md", PKG / "docs/MANIFEST.md"):
    with p.open("a", encoding="utf-8") as f:
        f.write(entry)
write(TRACE / "handoff.md", """# Executor handoff

Implemented and actually ran the complete frozen bounded workflow, then collected raw evidence and delegated a fresh reviewer. Accepted C1 only in its frozen single-training-seed scope, C2=no, overallpartial, engineering completionPASS. TEST remains blocked by the five-obstacle application, not lack of relative advantage.

Applied review corrections: distinguish374 MODEL_VALID from373 all-gate/goal-eligible candidates; do not infer isolated world-loss necessity; mark path coverage as a hypothesis; record exact BLAS environment required for reproducible normalized reconstruction. Reviewer corrected worst collision body names against raw records. No source/physical tolerance/experimental output was changed.

Complete-resume was called exactly once by the executor before review finalization;1403 files and hashes unchanged. Reviewer made zero resume/optimizer/physics/experiment calls. Source and all bound inputs independently checked. Final report covers known second-replay-array evidence limits and absent paper claim audit. Report helpers live outside frozen source closure.

Next action is a proposed separately frozen static feasibility/collision-guidance diagnosis; it was not started. No data expansion, new training attempt, finalTEST, dynamic waypoint or contact-grasp experiment was initiated. Goal/tracker versioned before fixed-entry updates. Original user modifications and other tasks preserved. Heartbeat will be paused and only task-owned evidence/docs/helpers committed.
""")
core = read(EV / "core_manifest_after_resume.json")
inputs = [{"path": (OUT / n).as_posix(), "sha256": h, "mtime_ns": (OUT / n).stat().st_mtime_ns}
          for n, h in core.items()]
ident = read(OUT / "run.json")["identity"]
for n, h in ident["source_hashes"].items():
    p = ROOT / n
    assert sha(p) == h
    inputs.append({"path": p.as_posix(), "sha256": h, "mtime_ns": p.stat().st_mtime_ns})
for p in [PKG / "docs/EXPERIMENT_PLAN.md", PKG / "docs/REFERENCE_SOURCE_ADDENDUM.md",
          PKG / "docs/ANALYSIS.md", EV / "evidence_summary.json", EV / "additional_diagnostics.json",
          EV / "resume_verification.json", TRACE / "prompt.md", TRACE / "response.md", TRACE / "verdict.json"]:
    inputs.append({"path": p.as_posix(), "sha256": sha(p), "mtime_ns": p.stat().st_mtime_ns})
dump(TRACE / "inputs.json", {"created_utc": datetime.now(timezone.utc).isoformat(),
    "scope": "all1403 frozen run files +173 executed sources + protocol/claim artifacts; all29027 bindings separately in run.json identity",
    "artifacts": inputs})
dump(EV / "delivery_records.json", {"timestamp": stamp, "tracker": f"paper/EXPERIMENT_TRACKER_{stamp}.md",
    "C1": verdict["claims"]["C1"]["claim_supported"], "C2": verdict["claims"]["C2"]["claim_supported"],
    "overall": verdict["claim_supported"], "engineering": "PASS", "TEST_ready": False,
    "expanded": False, "physics_relaxed": False, "user_modified_files_unchanged": True,
    "worker_elapsed_s": 3482.6559999994934, "source_commit": "b242dbeb",
    "fresh_review": TRACE.relative_to(ROOT).as_posix(), "source_files_rechecked": len(ident["source_hashes"])})
print(stamp)
