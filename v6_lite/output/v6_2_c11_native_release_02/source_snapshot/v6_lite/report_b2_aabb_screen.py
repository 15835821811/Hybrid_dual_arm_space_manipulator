"""Render the immutable private MuJoCo AABB screen comparison."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(output_dir: Path, audit_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=False)
    input_path = audit_dir / "aabb_pair_screen_summary.json"
    audit = json.loads(input_path.read_text(encoding="utf-8"))
    if (audit["state_count"] != 1351
            or audit["missed_original_active_pair_count"] != 0
            or audit["all_pair_count"] != 2927):
        raise ValueError("AABB audit is incomplete or lost an active pair")
    retained = [audit["sphere_retained"]["p95"],
                audit["aabb_retained"]["p95"]]
    duration = [audit["sphere_screen_and_exact_ms"]["p95"],
                audit["aabb_screen_and_exact_ms"]["p95"]]
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.1))
    for ax, values, title, label, ylim in (
            (axes[0], retained, "Retained exact pairs, p95",
             "pairs / state", (0, 260)),
            (axes[1], duration, "Screen plus exact query, p95",
             "milliseconds", (0, 1.0))):
        bars = ax.bar([0, 1], values, color=["#0f766e", "#c2410c"],
                      width=.55)
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    value + (.015 if ax is axes[1] else 3),
                    f"{value:.2f}" if ax is axes[1] else f"{value:.0f}",
                    ha="center", fontsize=10)
        ax.set_xticks([0, 1], ["Sphere", "Sphere + AABB"])
        ax.set_ylim(*ylim)
        ax.set_ylabel(label)
        ax.set_title(title, loc="left")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=.15)
        ax.set_axisbelow(True)
    fig.text(.07, .015,
             "1,351 private states; all 2,927 original pairs remain eligible; "
             "zero originally active pairs omitted.",
             fontsize=9, color="#334155")
    fig.tight_layout(rect=(0, .06, 1, 1), w_pad=3.0)
    image_path = output_dir / "aabb-screen-comparison.png"
    fig.savefig(image_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    report = {
        "schema": "v6_2_b2_aabb_screen_visual_v1",
        "state_count": audit["state_count"],
        "original_pair_count": audit["all_pair_count"],
        "missed_original_active_pair_count": 0,
        "retained_p95": retained,
        "screen_plus_exact_p95_ms": duration,
        "adopted_for_private_controller": False,
        "production_online_controller_changed": False,
        "source_sha256": {
            "audit_summary": _sha(input_path),
            "audit_rows": _sha(audit_dir / "aabb_pair_screen_rows.jsonl"),
            "report_b2_aabb_screen.py": _sha(Path(__file__)),
            "image": _sha(image_path),
        },
    }
    (output_dir / "aabb_screen_visual_summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8", newline="\n")
    lines = ["# B.2 MuJoCo 碰撞对 AABB 筛选试验", "",
             "在场景 01 的 1351 个保存状态逐一重算原 2927 对精确距离，"
             "检查包围球筛选与包围球＋AABB 筛选是否漏掉原激活对。"
             "AABB 对 MuJoCo 原始几何只作额外保守下界；所有保留对继续使用"
             "原 `mj_geomDistance`。", "",
             "| 方案 | 精确调用对数 p95 | 筛选＋精确查询 p95 | 漏掉原激活对 |",
             "| --- | ---: | ---: | ---: |",
             f"| 包围球 | {retained[0]:.0f} | {duration[0]:.3f} ms | 0 |",
             f"| 包围球＋AABB | {retained[1]:.0f} | {duration[1]:.3f} ms | 0 |", "",
             "AABB 只略微减少保留对数，但增加计算时间；当前私有控制路径"
             "不采用该变体。此处只测筛选与 MuJoCo 精确对查询，不是完整周期"
             "或连续时间安全证明。", "",
             "![AABB 筛选计时](aabb-screen-comparison.png)", "",
             f"- `audit summary SHA-256`: `{_sha(input_path)}`",
             f"- `audit rows SHA-256`: "
             f"`{_sha(audit_dir / 'aabb_pair_screen_rows.jsonl')}`", ""]
    (output_dir / "AABB_SCREEN.md").write_text(
        "\n".join(lines), encoding="utf-8", newline="\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--audit-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.output_dir, args.audit_dir), indent=2))


if __name__ == "__main__":
    main()
