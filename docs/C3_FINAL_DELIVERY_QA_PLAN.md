# C.3 最终交付 QA 执行计划

编制日期：2026-10-08。**本文件是静态执行计划，以下命令均未在编制时执行；实际视觉 QA 尚未完成。** 编制时正式 VAL 仍在顺序运行。须待根任务确认正式研究、原独立验收及最终报告完成，并安排无正式计时作业重叠的交付时段，才执行本计划。不得为交付 QA 增加候选、actual、训练更新、回归前向或 DDIM 样本。

已完整核对 [媒体构建器](../v6_4/visualization/build_search_aware_warmstart_media.py)、[portable 导出器](../v6_4/visualization/export_search_aware_release.py) 与 [completion matrix](C3_COMPLETION_EVIDENCE_MATRIX.md)。媒体复用的实际实现来自 [C.2 saved-state renderer](../v6_4/visualization/build_preference_warmstart_media.py)。本计划细化矩阵 14.6–14.11、U.1–U.7 的交付操作，不替代矩阵中的科学证据、预算、五门禁、最终独立审计或已有 helper/mock 审计。

## 1. 固定路径、运行前提与回执

后续在 PowerShell 使用正式记录的解释器；其他机器可调整解释器和仓库根路径，但必须记录实际值、版本与源码 SHA。下列目录是本轮预定位置，不能把路径存在当作交付已存在。

```powershell
$RepoPath = 'E:\v64c3'
$PythonPath = 'E:\v64c2\.venv-c2\Scripts\python.exe'
$RunPath = Join-Path $RepoPath 'v6_4\output\search_aware_warmstart_20261008_01'
$MediaPath = Join-Path $RepoPath 'v6_4\visualization\search_aware_warmstart_20261008_01'
$ReleasePath = Join-Path $RepoPath 'v6_4\releases\search_aware_warmstart_20261008_01'
$QaPath = 'E:\v64c3_delivery_qa\search_aware_warmstart_20261008_01'
Set-Location -LiteralPath $RepoPath
```

执行者先建立独立 QA 回执目录 `$QaPath`，记录每条命令、UTC 起止、退出码、解释器、工具版本和检查人。stdout/stderr、浏览器截图、逐视频视觉清单均写入该目录，不写回冻结 RUN，也不在已封存 MEDIA/REL 中增加未登记文件。任何原始证据缺失或身份不符应停止相应交付项并保留诊断；不得重跑科学样本、修补原始 seal 或换成功计划。

前置核查引用 completion matrix，不重复实施实验：正式阶段已终态；验证账本为 VAL 20 / TEST 40 逻辑 actual；两模型真实 4000 更新；四个候选 checkpoint 与最终 model freeze 完整；根任务的最终报告和独立审计可读。研究完成不要求 D 获胜；失败、NO_PLAN、N/A、alias 及 NOT_MET 均原样交付。

最终报告必须有 `REPORT.md`、`summary.json`、`result_tables/report_artifact_identity.json`。导出器实际检查 `summary.status` 的 `research_execution_completed`、`independent_test_completed`、`closed_loop_val_completed`、`training_completed_D`、`training_completed_S` 均为 true，且 `summary.evidence_verification.complete=true`；它同时核对 reporter SHA、当前 root reports、全部 derived artifacts 和 input files 的 SHA。不能只检查几个 completion 布尔值。

## 2. 实际 CLI 与构建边界

实际媒体 CLI 为：

```text
PY -B -X utf8 -m v6_4.visualization.build_search_aware_warmstart_media
  --run RUN --output MEDIA [--check-inputs | --render]
  [--workers 1|2] [--fps FPS]
  [--width W --height H --focus-width FW --focus-height FH]
```

默认 `workers=2, fps=15, width=640, height=480, focus_width=960, focus_height=720`。FPS 要求有限且在 `(0,60]`；尺寸为偶数且至少 640×480。`--check-inputs` 与 `--render` 互斥；没有 `--validate` / `--portable` 选项。不加两个选项也会创建图、dashboard 和 manifest，并非只读验证。

MEDIA 必须与 RUN 分离且互不包含。首次输出要求空目录；恢复时 `build_request.json` 必须与当前 run/config 一致。已完成 replay 可核对 SHA 后复用；留下未完成 replay 子目录会明确失败，不能自动删除后重来。若已有 partial 输出或不同配置，保留原目录并由根任务指定新的唯一输出位置。

先执行输入检查，再以相同配置渲染，均只使用正式保存的 actual：

```powershell
& $PythonPath -B -X utf8 -m v6_4.visualization.build_search_aware_warmstart_media --run $RunPath --output $MediaPath --check-inputs --workers 2 --fps 15 --width 640 --height 480 --focus-width 960 --focus-height 720
if ($LASTEXITCODE -ne 0) { throw 'C.3 media input check failed' }
& $PythonPath -B -X utf8 -m v6_4.visualization.build_search_aware_warmstart_media --run $RunPath --output $MediaPath --render --workers 2 --fps 15 --width 640 --height 480 --focus-width 960 --focus-height 720
if ($LASTEXITCODE -ne 0) { throw 'C.3 saved-state media build failed' }
```

`--check-inputs` 会写 `build_request.json` 和 `input_checks.json`，不会生成视频/图/dashboard；这一步不是新物理验收。`--render` 需要 PATH 中有 ffmpeg、ffprobe，并使用 saved qpos/qvel + `mj_forward`。渲染守卫禁止物理步、几何距离查询和 QP；render 成本另记，不加入正式规划计时。

输入源是 `RUN/frozen_test/actual/<task>/<endpoint>_<A|B>/`，不是另造的 `RUN/actual/`。加载器核对 TEST selection seal、Task/plan/config/初始历史、trace、已有 fresh replay、evaluation 与编译模型绑定；qpos/qvel parity 上限分别为 `1e-9/1e-8`，保存 generated-reference parity 为 `1e-12 m`。禁止通过重积分或另做独立 replay 来弥补缺失。

## 3. 产物清单与机器可核查项目

| 位置 | 最终要求 |
|---|---|
| MEDIA 根目录 | `build_request.json`、`input_checks.json`、`dashboard_data.json`、`index.html`、`visualization_manifest.json` |
| `MEDIA/charts/` | `budget_coverage_near_quality.png`、`actual_quality_planning_cost.png`，逐字节等于当前 RUN/result_tables 同名图 |
| `MEDIA/figures/` | 每个有保存步的 unique actual 各有 `<sid>_trajectory.png`、`<sid>_tracking.png`、`<sid>_tracking_native.csv` |
| `MEDIA/replays/<sid>/` | 七个 MP4、`replay_metadata.json`、`<sid>_five_view_preview.png`、`<sid>_continuum_focus_preview.png`、`<sid>_last_saved_state.png` |

`sid=<task_id>_<endpoint>_<A|B>`。七个 MP4 后缀为 `overview, front, side, top, iso, five_view_grid, continuum_focus`。默认普通单视角 640×480，focus 960×720，grid 1920×960；grid 是 3×2 排布的五视角加空白格。`continuum_focus` 是现有连续体近景相机，实际侧向可见性仍须视觉检查。

dashboard 应有四个 TEST Task × R8/R12/N8/S8/D8 × A/B 共 40 行。预期视频数为 `7 × unique 且 actual_steps>0 的记录数`，不预设为 280。严格 alias 共用同 Task 的原实际媒体与图；NO_PLAN 及其他零步拒绝无伪造视频/图。失败且有保存步的前缀应有描述性媒体，停在真实末状态。

后续可执行以下 stdlib 检查，不导入实验核心；它只读生成物。输出收进外部 QA 回执：

```powershell
@'
import hashlib, json, sys
from pathlib import Path
run, media = map(lambda p: Path(p).resolve(), sys.argv[1:3])
def read(p): return json.loads(p.read_text(encoding="utf-8-sig"))
def sha(p):
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(4 * 1024 * 1024), b""): h.update(b)
    return h.hexdigest()
data = read(media / "dashboard_data.json")
manifest = read(media / "visualization_manifest.json")
checks = read(media / "input_checks.json")
views = {"overview", "front", "side", "top", "iso", "five_view_grid", "continuum_focus"}
assert len(data["tasks"]) == 4 and set(data["endpoints"]) == {"R8", "R12", "N8", "S8", "D8"}
assert set(data["views"]) == views and len(data["slots"]) == 40
assert len({(s["task_id"], s["endpoint"], s["preference"]) for s in data["slots"]}) == 40
unique = [s for s in data["slots"] if not s.get("alias_of_slot") and s["actual_steps"] > 0]
assert checks["status"] == "PASS" and checks["unique_saved_actuals"] == len(unique)
assert manifest["logical_actual_slots"] == 40
assert manifest["unique_actual_media"] == len(unique) and manifest["video_count"] == 7 * len(unique)
assert manifest["visualization_is_new_acceptance"] is False
assert manifest["historical_outputs_overwritten"] is False
for k in ("physics_steps_executed", "geometry_queries", "QP_solves", "optimizer_updates", "new_model_samples"):
    assert manifest[k] == 0, k
current = {p.relative_to(media).as_posix() for p in media.rglob("*") if p.is_file() and p.name != "visualization_manifest.json"}
assert set(manifest["files"]) == current
for name, row in manifest["files"].items():
    p = (media / name).resolve()
    assert media in p.parents and p.stat().st_size == row["bytes"] and sha(p) == row["sha256"], name
for name in ("budget_coverage_near_quality.png", "actual_quality_planning_cost.png"):
    assert data["charts"][name] == "charts/" + name
    assert sha(media / "charts" / name) == sha(run / "result_tables" / name), name
by_key = {(s["task_id"], s["method"]): s for s in data["slots"]}
for s in data["slots"]:
    if s["actual_steps"] == 0:
        assert s.get("media") is None and s.get("figures") is None
        continue
    meta = s["media"]
    assert meta and s["figures"] and set(meta["videos"]) == views and set(meta["video_records"]) == views
    assert meta["includes_exact_saved_endpoint"] is True
    assert meta["selected_saved_indices"][-1] == s["actual_steps"]
    assert abs(meta["displayed_saved_end_s"] - meta["source_end_s"]) <= 1e-9
    assert abs(meta["source_end_s"] - s["actual_steps"] * .002) <= 1e-9
    assert meta["frame_count"] == len(meta["selected_saved_indices"]) == len(meta["selected_saved_times_s"])
    assert abs(meta["encoded_duration_s"] - meta["frame_count"] / meta["fps"]) <= 1e-9
    for k in ("physics_steps", "geometry_queries", "QP_solves"): assert meta["render_cost"][k] == 0
    assert not meta["render_cost"]["forbidden_call_attempts"]
    for view, row in meta["video_records"].items():
        assert meta["videos"][view] == row["relative_path"]
        assert sha(media / row["relative_path"]) == row["sha256"]
    if s.get("full_task_success"): assert s["actual_steps"] == 13500
    if s.get("alias_of_slot"):
        original = by_key[(s["task_id"], s["media_source_method"])]
        assert not original.get("alias_of_slot")
        assert s["media"] == original["media"] and s["figures"] == original["figures"]
assert data["summary"] == read(run / "summary.json")
print(json.dumps({"saved_media_machine_checks": "PASS", "logical_slots": 40,
                  "unique_saved_actuals": len(unique), "video_count": 7 * len(unique)}, ensure_ascii=False))
'@ | & $PythonPath -B -X utf8 - $RunPath $MediaPath
if ($LASTEXITCODE -ne 0) { throw 'C.3 media inventory/endpoint checks failed' }
```

媒体代码允许缺少核心图时继续构建；上面的最终检查特意要求两图齐全。manifest 的自身 SHA、两个 generator 源码 SHA、输入 run identity 与报告 provenance SHA 另外记录于 QA 回执。

逐个 unique 视频用 ffprobe 独立读取；下面的 `$VideoPath` 替换为 metadata 中精确的 `relative_path`：

```powershell
ffprobe -v error -count_frames -select_streams v:0 -show_entries stream=width,height,r_frame_rate,nb_read_frames:format=duration -of json $VideoPath
if ($LASTEXITCODE -ne 0) { throw 'Video probe failed' }
```

核对七视角 frame count、尺寸、FPS 与 metadata 的 `video_records.*.probe` 一致。视频长度按 `frame_count/fps` 检查；包含时间 0 和精确终点，不能机械要求所有文件恰为 27.000 秒。完整 actual 的物理末时间为 27 秒，失败前缀按自身保存末时间。

## 4. 轨迹/误差数值核查与实际视觉 QA

后续补充工具（2026-10-08 编制，当前只做 AST 静态语法检查，尚未运行）：

```powershell
& 'E:\v64c2\.venv-c2\Scripts\python.exe' -B -X utf8 'E:\v64c3\docs\audit_scripts\c3_video_review_pack.py' --media 'E:\v64c3\v6_4\visualization\search_aware_warmstart_20261008_01' --output 'E:\v64c3_delivery_qa\search_aware_warmstart_20261008_01\video_review_01'
```

此工具只能在正式流水线结束、七视角媒体全部渲染完成后运行。它独立用 ffprobe 解码计帧、核对尺寸/FPS/时长与原 metadata，核对所有视频字节 SHA，并提取每个 unique 视频首/中/末完整帧和每槽七视角联系表。输出必须是全新外部目录，保留实际命令与退出码，失败不覆盖旧回执。机器通过状态为 `MACHINE_PASS_VISUAL_REVIEW_PENDING`；每个图片的 `visual_review` 仍为 `NOT_RUN`。后续须实际查看联系表、必要时展开原尺寸帧，并另记录人工视觉判定，不能把已生成缩略图当作已查看。该工具不检查 CSV 数值、浏览器交互或轨迹图版式，以下工作仍需完成。

`tracking_native.csv` 应有 `actual_steps+1` 行，时间为 0、0.002、…、保存终点；六列依次为：

```text
physical_time_s
continuum_generated_position_error_m
continuum_Task_base_position_error_m
rigid_current_Task_position_error_m
continuum_orientation_error_rad
rigid_orientation_error_rad
```

后续数值 QA 只读已有数组和冻结参考，可调用 `load_actual(RUN, slot)`，不调用 renderer、executor 或独立 replay。对每个 unique 正步 actual，使用返回的 `replay, reference, base` 逐行重算并与 CSV 比较（建议绝对容差 `1e-12`，不修改任何实验容差）：连续体位置分别与 reference、base 求范数；刚性目标为 `target_position + target_rotation @ grasp_point_target_frame_m`；姿态误差为 `arccos(clip((tr(R_actual.T @ R_target)-1)/2,-1,1))`。连续体目标姿态为冻结 Task 的 `continuum_target_rotation_world`；刚性目标姿态为当前 `target_rotation @ grasp_rotation_target_frame`。记录最大差、行数、终点与有限性。此 API 会导入核心，**只能在当前正式实验全部结束、根任务允许交付 QA 后调用**；本次静态编制没有调用。

图像显示位置单位为 mm、姿态为 degree，CSV 保留 m/rad。轨迹图应有 actual、冻结生成参考、Task/base 或当前刚性抓取目标及精确保存终点；不能把生成参考误差替换为 Task 目标误差。两个核心结果图须另与当前表格核对系列、单位、失败/缺测和 actual/predicted 区分，科学数字核对仍归最终独立审计。

以下是待执行的**实际视觉 QA**，helper 测试、JSON PASS 和 ffprobe 均不能替代：

1. 在真实浏览器打开 `MEDIA/index.html`，核查结论读自 `summary.status.research_execution_completed/default_initializer_decision`，没有 `NOT_REPORTED`、旧 C.2 标签或空白图；保留首屏及两图截图。
2. 遍历全部 40 个 Task/endpoint/preference 组合，逐一切换七个 view，确认表格、状态、下载链接、视频和两张轨迹图对应同一槽。记录 alias 的明确源方法与共用路径；NO_PLAN/零步拒绝切换后应清除上一个视频与图，不保留旧画面。
3. 对每个 unique 的七个 MP4，实际查看首帧、中间帧、最后帧。检查机器人/障碍/目标可见、五视角排布、focus 连续体及局部交互可读、注释未裁切、标签为 C.3/正确 Task 与方法、没有黑屏或明显帧损坏。另查看两个 preview 和 last_saved_state PNG；失败前缀最后状态和文字一致，没有续播、插值或外推。
4. 对每个 unique trajectory/tracking PNG 实际查看坐标轴、图例、窗口阴影、单位与终点；alias 能到达同一组图/CSV；下载链接可打开。对完整、失败前缀及 NO_PLAN 各选实际存在的代表记录截图，若某类不存在则记 N/A。
5. 用清单记录 `slot/view、首/中/末检查、图/CSV检查、问题、处理后复查、检查人/时间`。任一未看项标为 NOT_RUN，不能因媒体数量正确而写“全部视觉 QA 通过”。

## 5. final portable release 与 relocation API

先完成当前报告、媒体 QA 和需要写入 RUN 的所有正式回执，再固定 RUN 文件清单。导出器对整个 RUN 逐文件 inventory；导出后在 RUN 新增日志、改报告或派生图，会使既有 release 的“当前来源”验证失败。导出日志须写到 `$QaPath`。

实际 CLI 只有 `export` 与 `verify`：

```powershell
& $PythonPath -B -X utf8 -m v6_4.visualization.export_search_aware_release export --run $RunPath --release $ReleasePath
if ($LASTEXITCODE -ne 0) { throw 'Final C.3 export failed' }
& $PythonPath -B -X utf8 -m v6_4.visualization.export_search_aware_release verify --release $ReleasePath
if ($LASTEXITCODE -ne 0) { throw 'Portable byte verification failed' }
```

final 使用默认严格模式，不传 `--allow-partial`。partial release 不能原地升级 final，需保留原件并换新的唯一目的目录；已有 final 即使带 `--allow-partial` 也仍按 final 验证。既有 release 身份、source-run inventory 或报告 provenance 不符时，同样使用新目录，不覆写原文件。

REL 应有 `release_manifest.json`、`portable_paths.json`、`frozen_source_manifest.json`、`snapshot/`、`frozen_source/`；若存在历史导入，另有 `historical_evidence_inventory.json`。检查 stage=final、producer/base 身份、source/report/exporter SHA、copied/omitted 数量及零新增物理/训练/DDIM 字段。`snapshot/` 中应包含当前报告和表、dataset/条件与残差 schema、真实 D/S 权重/训练记录、freeze、原始初值/selection、VAL/TEST unique actual trace+fresh replay 和必要验收记录。

导出采用每文件 `90,000,000` bytes 上限。必需 actual/exposure 数组、权重或 JSON 超限会失败；不能用缩小证据、重跑或静默删除来通过。private nominal 大原始数据、raw timing/geometry/certificate 和部分诊断按角色留原本地归档，inventory 记 SHA/size/reason，portable map 标 `available=false`。必须明确这些项不随 release 携带，不能把原 E: 路径冒充可用 portable 文件。**MEDIA 不会被该导出器自动复制，须独立随分支交付。**

portable API 的职责如下：

| API | 实际验证范围 |
|---|---|
| `PortableResolver(REL).verify_all()` | copied inventory、map 路径/大小/SHA、全部 available bytes、frozen source manifest 与 snapshot 身份及唯一映射；不访问原 E: 文件兜底 |
| `.resolve(original, expected_sha)` | 仅从 map 的 available copied bytes 解析；local-only 或无映射必须明确失败 |
| `.snapshot(run_relative_path)` | 精确且唯一的 inventory 项解析 |
| `.verify_sources()` | release frozen source bytes，及当前导入代码的仓库源码与 frozen producer SHA 相符 |
| `.load_dataset()` | 用原 dataset loader 读取 mapped TRAIN 数据；不产生新参考 |
| `.load_selected_sampler('D'/'S', device='cpu')` | 原数值实现加载 frozen 选中权重/条件与残差 scaler/schema、校验 tensor SHA；不前向、不采样、不训练 |
| `verify_inputs(RUN, require_complete=True)` | 原 RUN 与当前 producer、训练/freeze/phase/完整报告的当前来源核验；export 调用它，单独 `verify --release` 不替代它 |

后续在一个**全新**迁移目录复制 REL 和 MEDIA，保留原件；不要移动/删除正式归档。用迁移目录运行 `verify --release`，然后从与 frozen source 相符的仓库执行下面只加载、不采样的 smoke。两个 loader 没有独立 CLI：

```powershell
$RelocatedReleasePath = Join-Path $QaPath 'relocated_release'
$RelocatedMediaPath = Join-Path $QaPath 'relocated_media'
if ((Test-Path -LiteralPath $RelocatedReleasePath) -or (Test-Path -LiteralPath $RelocatedMediaPath)) { throw 'Use fresh relocation destinations' }
Copy-Item -LiteralPath $ReleasePath -Destination $RelocatedReleasePath -Recurse
Copy-Item -LiteralPath $MediaPath -Destination $RelocatedMediaPath -Recurse
& $PythonPath -B -X utf8 -m v6_4.visualization.export_search_aware_release verify --release $RelocatedReleasePath
if ($LASTEXITCODE -ne 0) { throw 'Relocated release byte verification failed' }
@'
import json, sys
from v6_4.visualization.export_search_aware_release import PortableResolver
r = PortableResolver(sys.argv[1])
result = r.verify_all()
r.verify_sources()
dataset = r.load_dataset()
models = {}
for name in ("D", "S"):
    sampler = r.load_selected_sampler(name, device="cpu")
    assert sampler.sample_units == 0 and not sampler.model.training
    assert all(not p.requires_grad for p in sampler.model.parameters())
    models[name] = sampler.identity
print(json.dumps({"portable": result, "dataset_type": type(dataset).__name__,
                  "selected_models_loaded_without_sampling": models}, ensure_ascii=False))
'@ | & $PythonPath -B -X utf8 - $RelocatedReleasePath
if ($LASTEXITCODE -ne 0) { throw 'Relocated dataset/selected-weight loading failed' }
```

loader smoke 会导入冻结核心并加载模型，当前正式运行期间禁止执行；将来执行也不得调用 `.sample()`、`.initializer_proposals()`、模型 forward 或其他推理入口。记录 checkpoint/scaler/schema 身份与零 sample_units。`verify_all()` 是 portable 字节核验，不声称独立重验物理、科学结论或原 live RUN 的新鲜度。

迁移后的 MEDIA 需重做 manifest 文件 SHA 和相对 asset 路径检查，并实际打开 `relocated_media/index.html` 查看图片/视频/下载，无原位置依赖。媒体 `_portable_path` 只支持完整 RUN 的原根相对路径迁移；精简 release 故意未携带部分原 timing/certificate，**不能保证从 `REL/snapshot` 独立重建媒体**。正式媒体先对原完整 RUN 完成验证、渲染，再独立搬运成品。

## 6. 导航、分支发布与验收回执

发布前按矩阵 U.6 更新当前 C.3 README 入口及新的可视化说明，链接 REPORT、REL、MEDIA/index.html、两核心图和 QA/最终审计；保留旧 C.1/C.2/B viewer、报告、权重和历史结论。不要覆盖 `docs/V6_4_C2_VISUALIZATION.md`。检查相对链接在仓库树和下载后的本地目录都可达；GitHub 的仓库 HTML 文件查看页不等同于可直接播放 dashboard。

待最终产物存在后，由根任务审核完整 git diff/status 和待发布文件清单。发布清单至少包括：实现/文档；final REL 的所有 copied bytes 与遗漏 inventory；独立 MEDIA 的所有 manifest 列出文件；当前导航；最终报告及独立审计/QA 回执的可发布副本。保留 command/runtime 身份，并明确实验 producer 与最终交付 commit 是两个身份。检查待发布单文件大小，尤其独立 MEDIA 的 MP4 也需检查，导出器的 90MB 限制未覆盖它们。

用户已明确要求单独分支上传 GitHub；由根任务在完成具体产物审阅后提交并推送 `v6.4-c3-search-aware-closed-loop-val`，不合并 main、不强推或改写历史。最终只读核对：

```powershell
git branch --show-current
git status --short
git rev-parse HEAD
git ls-remote origin refs/heads/v6.4-c3-search-aware-closed-loop-val
```

remote 精确 branch head 应等于交付 HEAD。另实际查看远端分支文件树、报告与必要 raw/weight/media 文件可访问，核对链接；远端网页身份检查不重新运行科学程序。将 producer、交付 HEAD、REL/MEDIA manifest SHA、源 RUN/报告 provenance SHA、每项 QA verdict 和 unresolved 项写入最终 delivery receipt。

## 7. 尚未覆盖的实际工作

本计划没有执行命令、helper 测试、核心导入、科学计算、媒体渲染、视频解码/视觉检查、导出/迁移、导航编辑或提交推送。此刻不能据此声称媒体、portable release 或 GitHub 交付完成。

待后续实际完成：正式程序与最终报告/独立审计；saved-actual 输入绑定与全部 CSV 数值核对；七视角逐视频首/中/末视觉 QA；40 槽菜单/alias/NO_PLAN 清屏检查；两核心图和轨迹图视觉 QA；最终字节导出与迁移加载；当前导航和远端 HEAD/文件访问。已有 implementation/interim/helper 审计保留原适用范围，不能改名为上述真实视觉或最终交付验收。
