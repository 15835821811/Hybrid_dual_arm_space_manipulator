"""Closeout documents and saved-evidence checks only; no experiment execution."""
from pathlib import Path
import datetime
import hashlib
import json

ROOT = Path(__file__).resolve().parents[2]
PKG = ROOT / "ff_prior_backbone_v2"
EV = PKG / "evidence/completed"
TRACE = ROOT / "paper/review-traces/experiment-result-to-claim/20261020_backbone_run01"


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    verdict = json.loads((TRACE / "verdict.json").read_text(encoding="utf-8"))
    assert verdict["claims"]["C1"]["claim_supported"] == "no"
    assert verdict["claims"]["C2"]["claim_supported"] == "partial"
    assert verdict["engineering_verdict"]["status"] == "PASS"
    run = ROOT / "artifacts/v7_1/backbone_pair_smoke_002"
    old = json.loads((EV / "core_manifest_after_resume.json").read_text(encoding="utf-8"))
    now = {p.relative_to(run).as_posix(): sha(p) for p in run.rglob("*") if p.is_file()}
    assert now == old and len(now) == 956
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    findings = """# 本轮结果与主张

backbone_pair_smoke_002完成，冻结源码4ccba221，耗时2741.157秒。
两个主干各6000正式更新、14222TRAIN全部参与、300随机序列摘要相同。
无引导各0/96，Transformer位置均值19.0167→21.0883mm、尾部恶化；C1=no。
原冻结DEV两组各180/180与36/36；Transformer相对新MLP执行位置均值
0.7872→0.6593mm，下降16.24%；成功率持平，条件均值平均精度更优；C2=partial。
新增360候选/72执行；旧简单基线360/72仅复用。工程PASS。
单训练种子、共同物理引导、已用开发任务，不证明普遍优势。
完整replay第二轨迹未保存；审查核残差/源码/native_trace，未追加物理仿真。
一次完成resume返回NO_WORK_ALREADY_COMPLETE，956文件清单/hash不变。
保留旧默认MLP与Transformer实验候选；无扩库或TEST。长期科学目标未完成。
新review见paper/review-traces/experiment-result-to-claim/20261020_backbone_run01。
paper claim audit不可用，下游论文写作为provisional。
"""
    (PKG / "docs/findings.md").write_text(findings, encoding="utf-8")
    with (ROOT / "findings.md").open("a", encoding="utf-8") as f:
        f.write("\n\n" + findings)
    goal = """# 当前执行目标：有界MLP与Transformer对照已完成

2026-10-20。用户授权的本轮实现、训练、完整无引导诊断、原冻结DEV同预算
候选/力矩执行/重放、原始审计、新独立审查及报告均完成。
有效运行artifacts/v7_1/backbone_pair_smoke_002；冻结源码4ccba221。
入口 .\\ff_prior_backbone_v2\\tools\\job.ps1 status|report|resume。

|阶段|实际结果|状态|
|---|---|---|
|M0|14/14软件测试；原001存储故障独立v2修复，失败证据保留|PASS|
|M1|14222TRAIN/48VAL；两个近参数规模主干各6000正式更新；同300随机摘要|PASS|
|M2|完整16步无引导各96条，均0/96指定目标全门合格|已完成，C1=no|
|M3|原DEV4+已用8，3种子，每任务5候选；两组各180/180与36/36|工程PASS|
|M4|原始审计及全新独立审查、一次resume956文件不变、中文报告|已完成|

Transformer对新MLP执行位置均值下降16.24%，但成功率持平，条件均值平均精度更优。
C2=partial，仅支持固定开发任务上共同late4后的有限残差收益；单训练种子。
两组无引导均0/96，Transformer位置/Q均值和尾部更差；不支持普遍替代MLP。
软件实现、MODEL_VALID、具体执行重放通过和科学收益分别判定。

保留旧默认模型、全部数据/结果/用户修改/其他作业。原001失败证据保留。
保持仅不穿透、5mm/1度、2mm几何采样、17→67、完整初态、自由漂浮、
全机几何及全部运动/力矩/重放门。不扩库、不启动pilot/study/strict_multimodal/TEST。
本轮接续收尾后暂停。没有自动追加实验或替换默认模型。

下一研究方向仅建议：固定现有TRAIN与预算，验证完整采样输出的目标对齐，
监控无引导命中和尾部，同时保留简单基线；若研究架构稳定性需独立训练重复。
新实验须另行确定方案；不以本轮完成宣称长期Diffusion优势目标完成。
完整报告ff_prior_backbone_v2/docs/ANALYSIS.md；paper claim audit缺失，
下游论文写作仍provisional。重放审查边界见报告。
"""
    (ROOT / "DIFFUSION_GOAL_CURRENT.md").write_text(goal, encoding="utf-8")
    tracker = goal.replace("# 当前执行目标：", "# Experiment Tracker：", 1)
    version = ROOT / f"paper/EXPERIMENT_TRACKER_{stamp}.md"
    version.write_text(tracker, encoding="utf-8")
    (ROOT / "paper/EXPERIMENT_TRACKER.md").write_text(tracker, encoding="utf-8")
    handoff = """# 新独立审查交接

本轮新review已结束；原始response.md/verdict.json完整保留，不改审查结论。
C1=no；C2=partial；工程PASS。所有结论已纳入中文ANALYSIS.md及目标/tracker。
未复用旧review。审查未运行训练、物理仿真、optimizer.step或resume。
主代理一次resume已保存前后956文件清单及hash；收尾再次只读比对仍一致。
独立重放未保存完整第二轨迹；只读审查范围及未重算几何/质量矩阵明确写入报告。
没有paper claim audit，下游写作provisional。无未决外部审查。
行动：完成有界交付，保留默认模型，暂停本接续；不扩库或TEST，不追加实验。
"""
    (TRACE / "handoff.md").write_text(handoff, encoding="utf-8")
    matrix = {
        "status": "PASS", "scope": "bounded backbone experiment only",
        "run_file_count": len(now), "core_unchanged_after_closeout": True,
        "requirements": {
            "implementation": "14/14 software tests; independent storage fix",
            "actual_training": "2x6000; all 14222 TRAIN; matched 300 random segments",
            "complete_unguided": "192; full 16 DDIM; zero physics guidance",
            "frozen_DEV_budget": "new 360/72; reused 360/72; logical 720/144",
            "physical_execution": "72 new native67 torque rollouts and original independent replays",
            "raw_audit": "PASS; saved evidence and algebra only",
            "fresh_claim_review": "C1=no; C2=partial; engineering=PASS",
            "completed_resume": "once; 956 exact filenames and hashes unchanged",
            "physics_and_scope": "unchanged; no expansion/pilot/study/strict_multimodal/TEST",
            "report": "ff_prior_backbone_v2/docs/ANALYSIS.md",
            "versioned_tracker": str(version.relative_to(ROOT)),
        },
        "scientific_goal_complete": False,
        "remaining_delivery_actions": ["pause heartbeat", "task-only git commit"],
    }
    (EV / "completion_requirements.json").write_text(json.dumps(matrix, ensure_ascii=False, indent=2), encoding="utf-8")
    refs = [PKG / "docs/ANALYSIS.md", TRACE / "response.md", TRACE / "verdict.json", TRACE / "handoff.md"]
    (TRACE / "delivery_inputs.json").write_text(json.dumps(
        {str(p.relative_to(ROOT)): sha(p) for p in refs}, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest = f"\n| {stamp} | experiment-result-to-claim / closeout | ff_prior_backbone_v2/docs/ANALYSIS.md; ff_prior_backbone_v2/evidence/completed; {version.relative_to(ROOT).as_posix()}; paper/review-traces/experiment-result-to-claim/20261020_backbone_run01 | 工程PASS，C1=no，C2=partial；956文件不变；本轮完成，长期目标未完成 |\n"
    for p in [PKG / "docs/MANIFEST.md", ROOT / "MANIFEST.md"]:
        with p.open("a", encoding="utf-8") as f:
            f.write(manifest)
    print(json.dumps({"status": "PASS", "tracker": str(version.relative_to(ROOT)), "core_files_unchanged": len(now)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
