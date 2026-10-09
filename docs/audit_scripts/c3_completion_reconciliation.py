"""Reconcile explicit completion obligations with retained audits, without science calls.

This is a review index, not a replacement for the linked independent computations.
The deliberately explicit row map must cover every numbered preparation obligation.
"""
import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[2]
DOC = ROOT / 'docs'
DELIVERY = 'docs/audit_receipts/c3_delivery/'
RELEASE = 'v6_4/releases/search_aware_warmstart_20261008_01/'
MEDIA = 'v6_4/visualization/search_aware_warmstart_20261008_01/'


def read(path):
    return json.loads((ROOT / path).read_text(encoding='utf-8-sig'))


def sha(path):
    return hashlib.sha256((ROOT / path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--publication', help='Verified publication receipt relative to repo root')
    args = parser.parse_args()
    evidence = {
        'implementation': 'docs/C3_INDEPENDENT_IMPLEMENTATION_AUDIT.json',
        'state': 'docs/C3_STATE_ISOLATION_AUDIT.json',
        'teacher': 'docs/C3_FINAL_TEACHER_DATASET_AUDIT.json',
        'val_search': 'docs/C3_FINAL_VAL_SEARCH_AUDIT.json',
        'val_selection': 'docs/C3_FINAL_VAL_SELECTION_AUDIT.json',
        'test': 'docs/C3_FINAL_TEST_METADATA_AUDIT.json',
        'binary': 'docs/C3_FINAL_ACTUAL_BINARY_AUDIT.json',
        'required_tests': 'docs/C3_REQUIRED_TEST_COVERAGE_AUDIT.json',
        'mock': 'docs/audit_receipts/c3_final_mock_suite_06/receipt.json',
        'maintenance': DELIVERY + 'seed_binding_maintenance_01/receipt.json',
        'parity': DELIVERY + 'teacher_maintenance_parity.json',
        'source_delta': DELIVERY + 'publication_source_difference.json',
        'report': RELEASE + 'snapshot/REPORT.md',
        'status': RELEASE + 'snapshot/result_tables/status.json',
        'release': RELEASE + 'release_manifest.json',
        'portable': DELIVERY + 'portable_load_01.json',
        'frozen_load': DELIVERY + 'frozen_producer_load_01/load_receipt.json',
        'media': MEDIA + 'visualization_manifest.json',
        'numeric': DELIVERY + 'media_numeric_01.json',
        'video': DELIVERY + 'video_review_01/receipt.json',
        'visual': DELIVERY + 'media_visual_review_01.json',
        'plots': DELIVERY + 'figures_review_01/visual_review.json',
        'core_plots': DELIVERY + 'core_figures_visual_review.json',
        'browser': DELIVERY + 'browser_ui_01.json',
        'relocation': DELIVERY + 'media_relocation_01.json',
        'browser_relocation': DELIVERY + 'browser_relocation_01.json',
        'preflight': 'docs/audit_receipts/c3_publication_preflight.json',
        'git_index': 'docs/audit_receipts/c3_git_index_01.json',
    }
    verified_counts = {}
    for key, count in [('teacher', 1182), ('val_search', 1604), ('val_selection', 1975),
                       ('test', 11552), ('binary', 16342)]:
        obj = read(evidence[key])
        assert obj.get('checks_total', obj.get('check_total')) == count, key
        assert not obj.get('errors', []) and not obj.get('discrepancies', []), key
        verified_counts[key] = count
    assert read(evidence['binary'])['status'] == 'PASS'
    assert not read(evidence['test'])['missing_or_pending']
    mock = read(evidence['mock'])
    assert mock['status'] == 'PYTEST_EXIT_ZERO'
    assert mock['junit']['counts'] == dict(reported_cases=212, failures=0, errors=0, skipped=0, passed_cases=212)
    assert read(evidence['maintenance'])['exit_code'] == 0
    for key in ['parity', 'source_delta', 'portable', 'frozen_load', 'numeric', 'visual',
                'plots', 'core_plots', 'browser', 'relocation', 'browser_relocation', 'preflight', 'git_index']:
        assert read(evidence[key])['status'] == 'PASS', key
    release, media = read(evidence['release']), read(evidence['media'])
    assert release['stage'] == 'final' and release['copied_count'] == 4283 and release['omitted_count'] == 3090
    assert (media['logical_actual_slots'], media['unique_actual_media'], media['video_count'],
            media['alias_slots'], media['no_plan_slots']) == (40, 28, 196, 6, 6)
    assert read(evidence['browser'])['summary']['combinations'] == 280
    status = read(evidence['status'])
    for k in ['research_execution_completed', 'implementation_completed', 'closed_loop_val_completed',
              'independent_test_completed', 'training_executed_D', 'training_executed_S']:
        assert status[k] is True, k
    for k in ['learning_benefit_over_rules', 'learning_benefit_over_retrieval', 'learning_benefit_over_simple_regression']:
        assert status[k] is False, k
    assert status['default_initializer_decision'] == 'KEEP_C1_RULE'
    assert status['selected_checkpoint_D'] == 'D4000' and status['selected_checkpoint_S'] == 'S4000'
    published = False
    if args.publication:
        publication = read(args.publication)
        assert publication['status'] == 'PASS'
        assert publication['remote_head'] == publication['artifact_commit']
        assert publication['branch'] == 'v6.4-c3-search-aware-closed-loop-val'
        assert publication['remote_tree_verified'] is True
        assert publication['release_blob_count'] == 4717 and publication['media_blob_count'] == 399
        assert len(publication['raw_files']) == 10
        assert all(row['status'] == 'PASS' for row in publication['raw_files'])
        evidence['publication'] = args.publication
        published = True

    # Each entry is a reviewed obligation group, not a prefix-based automatic pass.
    groups = []
    def group(ids, refs, finding, verdict='SATISFIED'):
        groups.append((ids.split(), refs.split(), finding, verdict))

    group('0.1 1.1 1.2', 'implementation binary source_delta preflight release',
          'Correct fixed base, independent requested branch and frozen producer. 430 producer files and 37 protected artifacts verified; historical artifacts preserved. Publication changes one teacher null-identity edge, with original bytes archived and all 12 original summaries invariant.')
    group('1.3 U.7', 'publication' if published else 'preflight',
          'Normal independent-branch publication and exact remote artifact head/file-tree verification recorded; no main merge or history rewrite.' if published else 'Local deliverables verified; GitHub push and exact remote head verification still required.',
          'SATISFIED' if published else 'PUBLICATION_PENDING')
    group('0.2 3.1 14.1', 'teacher val_selection test binary report preflight',
          'Real 12 teacher streams, fresh D/S4000 training, 10 VAL streams and 16 TEST streams completed. Negative outcome is retained. Producer and later evidence commits are separate; staged commits are not claimed as new experiments.')
    group('1.4 14.8', 'report release media core_plots plots',
          'Exactly two scientific core charts. User-requested current saved-actual media are a separate supplement. Original history remains unchanged.')
    group('2.1a 2.1b 2.1c 2.1d 12.1 12.2', 'implementation binary state mock preflight',
          'Frozen low-level model/controller/geometry/search numerical source, runtime-normalized identities and original five-gate evidence checked. 27s/13500 main steps for every unique actual; 20ms planning/2ms physics and original reference bounds retained. Current C.3 wrapper/S-source tests preserve C.2 defaults.')
    group('2.2 2.3a 2.3b 2.3c 10.1 10.2', 'implementation teacher val_search test report mock',
          'R/N/S/D and A/B semantics explicit. Only slots1/3 supplied, common0/2 unchanged, 12-D mask and interval bounds preserved. TEST S rejects5/8 raws, D1/8; rejected raws occupy slots with zero rollout and no repair or replacement. Direct/common/descendant attribution retained, without extra control authority.')
    group('4.1a 4.1b 4.1c 4.1d', 'teacher binary release portable',
          '72 deduplicated historical TRAIN physical facts, 52 route labels, 24 old VAL/TEST facts excluded; evidence levels and B qualification distinct. Historical tasks unchanged. Bound archive availability is checked, and omitted portable files remain explicitly unavailable.')
    group('4.2a 4.2b 4.2c', 'implementation teacher val_search test report',
          'External split frozen before teachers: old3 TRAIN mothers/6 tasks, new1 VAL mother/2 tasks and new2 TEST mothers/4 tasks. Original source-only seed policy, generator and 55mm offsets retained without adaptive redraw. Single training seed and DATA_LIMITED scope remain explicit.')
    group('5.1 5.2a 5.2b 5.3b 5.3c 5.4a 5.4b 5.4c', 'teacher implementation parity',
          'All12 prefrozen local/leave-one-mother transfer pairs, original seed+partner and lineage independently reconciled. Nominal-only teacher output: 19 qualified endpoints,14 near; 12 effect-evidenced,7 rule-only,5 without qualified output. Original seeds, not final solutions, form effect supervision. No teacher actual and no late historical-prefix backfill.')
    group('5.3a 8.3b 13.13', 'state binary teacher test mock',
          'Source constructs fresh run-local physical/controller/reference objects; saved initial qpos/qvel and runtime identities checked for all39 actuals. Stream-local directories/caches and small-array QP warm-history/reset test pass. No direct live-object/dual-history telemetry was recorded; evidence is frozen constructor paths, saved state/identity and focused tests, not a new instrumentation experiment.', 'SATISFIED_WITH_STATED_EVIDENCE_SCOPE')
    group('5.5a 5.5b 5.5c 6.1 6.2 6.3a 6.3b', 'teacher mock release frozen_load',
          '55 unique labels:52 route,8 effect,5 overlap,3 zero;10 effect rows deduplicate to8. D/S/N share pool/schema and TRAIN scalers. D/S fresh two128-SiLU models each4000×32 AdamW; paired draw bytes and all4 checkpoints/tensors verified. D cosine100/v/DDIM20; S direct original-reference MSE. Only250/4000 checkpoints; selected weights load without forward or sampling.')
    group('7.1a 7.1b 7.2 7.3a 7.3b 7.3c 7.4 8.1', 'val_search val_selection binary teacher test',
          'VAL88 slots,20 logical actuals=11 unique+3 alias+6 NO_PLAN. All searches sealed before actual; same prefrozen D noise and 8 D samples. Original lexicographic scorer independently recomputed: D4000 wins criterion3, S4000 criterion1. One checkpoint/model plus pool/scalers/protocol frozen before TEST; no TEST selection or adaptive changes.')
    group('8.2a 8.3a', 'test val_search mock',
          'TEST4×(R12+N8+S8+D8)=144 consumed slots. Two initials fixed before each request; R8 sealed before ninth slot; N/S/D stop at8. A/B share one pool and cost. Prefix seal/order and raw/source bytes reconciled.')
    group('8.4a 8.4b 8.4c', 'test binary',
          'All TEST choices sealed before actual;40 logical=28 unique+6 strict aliases+6 NO_PLAN. All28 actuals use original reset feedback execution and pass full27s/five gates. No actual failures/tool errors, no replacement, no cross-task alias. Binary initial state, replay and original manifest evidence verified.')
    group('9.1a 9.1b 9.2a 9.2b 9.3 9.4a 9.4b', 'test report status core_plots',
          'All denominators, A/B endpoints, paired I/L/d/base metrics, prediction-to-actual differences, first hits and cost tables reconciled. R12 A/B4/4; R8 A/B4/4; N8 A4/B3; S8 A4/B2; D8 A4/B1. Near A/B: R12 4/4,R8 3/2,N8 3/1,S8 3/1,D8 2/1. Prefix4 remains predicted only. D fails capability/quality preservation and adoption criteria; no overall benefit or amortization claim.')
    group('11.1a 11.1b 11.1c 11.2a 11.2b 11.2c', 'teacher val_search val_selection test report',
          '328 candidate and60 actual logical slots;4,058,970 main prediction steps and526,500 main actual steps within4,428,000/810,000 caps.39 unique actuals,9 aliases,12 NO_PLAN overall. Preview/replay/geometry/QP costs separate. D formal VAL8+TEST8=16 DDIM samples, no additional scientific inference. Sequential rotated cold requests include outer timing; warm latency and effective speedup not inferred.')
    group('13.01 13.02 13.03 13.04 13.05 13.06 13.07 13.08 13.09 13.10 13.11 13.12 13.14 13.15 13.16 13.17 13.18 13.H',
          'required_tests mock maintenance parity binary',
          'All18 required test bullets mapped in the retained coverage audit; six additional assertion cases and current residual-binding fixture executed in final17-module suite:162 tests+50 subtests,212 JUnit cases,0 fail/error/skip. Historical fixture unchanged; attempts03–06 retained with failure history. Post-freeze teacher edge repair adds6 cases;19 focused teacher cases pass,12 saved pair summaries identical. Mock checks supplement original physical evidence, not replace it.')
    group('14.2 14.3', 'teacher release',
          'prepare/import-history/teacher-search/build-dataset/train original command receipts, frozen splits/pairs/teacher results, data and four checkpoint bytes retained. All formal stages terminal, no teacher actual.')
    group('14.4', 'val_search val_selection binary release',
          'Original closed-loop-val and freeze-models terminal receipts,20 actual slots, scoring references and model freeze verified and archived.')
    group('14.5 14.6 14.7 14.9', 'test binary report release',
          'Original test-search/execute-test/validate/report terminal receipts and report provenance verified. Every search/selection/actual failure and cost remains represented. Completion uses authoritative frozen_test/actual and existing seals; no cosmetic replacement run manifest. Recovery did not replenish consumed budgets.')
    group('14.10', 'release portable frozen_load preflight git_index',
          'Final release4717 files:4283 copied scientific files,430 frozen producer files and4 release metadata.3090 omitted nominal/timing/geometry artifacts explicitly inventoried with SHA/size and unavailable portable mappings. Necessary weights, paired draws and all39 actual/replay arrays included. Independent relocation verifies bytes and loads55-label dataset plus selected D/S under archived producer with model forward forbidden.')
    group('14.11', 'teacher val_search val_selection test binary mock preflight',
          'This final row-by-row reconciliation closes the earlier timestamped preparation using separate independent teacher/VAL/TEST audits plus full actual/source-byte and delivery QA. Their original limited verdicts remain intact. Publication status is separately explicit; this index does not pretend to be a new blind reviewer or physical re-execution.')
    group('U.1', 'browser browser_relocation media',
          'All40 logical rows×7 views=280 browser selector checks pass;6 aliases reuse bound media and6 NO_PLAN clear video/figures/downloads. Representative relocated playback reaches ended state and downloaded CSV hash equals source. Rapid binding checks alone are not claimed as playback of every video.')
    group('U.2', 'media video visual',
          '28 unique actuals×7=196 videos; all frame counts/dimensions/fps/durations decoded and first/middle/last frames visually inspected via28 contact sheets. All84 previews viewed via7 sheets. Exact27s saved endpoint included; encoded406/15=27.066667s. No actual failed-prefix case exists; that visual example is N/A, not fabricated.')
    group('U.3', 'numeric plots browser',
          'All28 native13501-row tracking CSVs independently recalculated from saved arrays and frozen reference. Position1e-12m/time1e-9s, orientation arccos arithmetic1e-7rad comparison tolerance; no experiment tolerance changed. All56 trajectory/tracking PNGs visually inspected. Frozen-reference versus Task/base and grasp targets remain distinct; spikes are preserved.')
    group('U.4', 'media numeric visual',
          'Saved qpos/qvel + mj_forward display only; guards prohibit mj_step/QP/distance queries/training/sampling. Manifest binds source/task/plan/config/replay. Media adds zero scientific steps/samples/updates and does not confer safety acceptance.')
    group('U.5', 'media core_plots report',
          'Both core chart bytes equal frozen report charts and were visually inspected. A/B cost sharing, missing R8 cold cost and predicted-prefix scope are explicit.')
    group('U.6', 'preflight media report',
          'Current README and both navigation entries point to C.3 report/viewer/docs. Historical C.2/B/C.1 entries and assets retained. Local HTTP and relocated relative assets verified; GitHub source HTML is correctly described as requiring local viewing.')

    matrix = (DOC / 'C3_COMPLETION_EVIDENCE_MATRIX.md').read_text(encoding='utf-8-sig')
    requirements = {}
    for line in matrix.splitlines():
        if re.match(r'^\| (?:[0-9]|U\.)', line):
            columns = line.split('|')
            identifier = columns[1].strip().split()[0]
            requirements[identifier] = columns[2].strip()
    rows = {}
    for ids, refs, finding, verdict in groups:
        assert all(ref in evidence for ref in refs), refs
        for identifier in ids:
            assert identifier in requirements and identifier not in rows, identifier
            rows[identifier] = dict(id=identifier, requirement=requirements[identifier],
                verdict=verdict, evidence=refs, finding=finding)
    assert set(rows) == set(requirements), sorted(set(requirements)-set(rows))
    audit = dict(schema='c3_final_completion_reconciliation_v1',
        utc=dt.datetime.now(dt.timezone.utc).isoformat(),
        status='COMPLETE' if published else 'LOCAL_COMPLETE_PUBLICATION_PENDING',
        reviewer='Codex root, reconciliation of retained independent audits and current delivery checks',
        objective_sha256=sha('docs/V6_4_C3_OBJECTIVE.md'),
        preparation_matrix_sha256=sha('docs/C3_COMPLETION_EVIDENCE_MATRIX.md'),
        script_sha256=sha('docs/audit_scripts/c3_completion_reconciliation.py'),
        algorithm_producer='9f39b42775283432eb933f63a9047c488ba22070',
        independent_audit_check_counts=verified_counts,
        evidence={key:dict(path=path, sha256=sha(path)) for key,path in evidence.items()},
        numbered_obligations=[rows[key] for key in requirements],
        section15_final_fields=status,
        section16_verdict='Finite initializer-only study complete; learning benefit not established; keep C.1.',
        new_physics_steps=0, new_training_updates=0, new_DDIM_samples=0,
        limitations=[
            'Three TRAIN, one VAL and two TEST mothers; one seed/model; no statistical noninferiority or independent causal component claim.',
            'No measured live Python object/dual-history telemetry; isolation supported by frozen construction code, complete saved initial state/runtime identities and focused reset/warm-history tests.',
            'Teacher effects nominal shared-search evidence, not actual teacher trials or causal necessity.',
            'Large private prediction/timing/geometry/certificate arrays remain in original local archive; portable absence is explicit.',
            'Every video decoded, visually sampled at first/middle/last frames; not human continuous playback of all frames. NO_PLAN has no artificial trajectory.',
            'deployment NOT_MET; continuous-time and hardware safety NOT_ESTABLISHED.',
        ])
    (DOC/'C3_FINAL_COMPLETION_AUDIT.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    lines = ['# C.3 最终逐项完成核对', '', f"状态：**{audit['status']}**。更新时间：{audit['utc']}。", '',
        '本文件把独立教师、VAL、TEST 审查与后续实际数组/源码字节、软件测试、媒体和迁移检查按任务书逐项对齐。原准备清单和各次中间审查保持原文；本核对不是新的物理验收，也不将旧的有限范围审查改名为最终全覆盖审查。', '',
        '真实研究完成，Diffusion 总体收益未建立，默认保留 C.1。D/S 均由闭环 VAL 选中 update4000。TEST40 个逻辑槽包括28次独立 actual、6个严格别名和6个 NO_PLAN；34槽完整通过。VAL+TEST39次 unique actual 共526500主物理步；主预测4058970步。正式D共16个DDIM样本。', '',
        '软件最终套件162 tests +50 subtests（JUnit212案例）通过；维护修复另19项通过。28个实际执行的196视频、56轨迹/误差图、84预览、28 CSV、两个核心图均完成检查；280个浏览器组合及迁移后播放/下载检查通过。', '',
        '## 证据索引', '', '|键|文件（SHA-256 见 JSON）|', '|---|---|']
    for key,path in evidence.items():
        link = path[5:] if path.startswith('docs/') else '../'+path
        lines.append(f'|{key}|[{path}]({link})|')
    lines += ['', '## 编号义务逐项结论', '', '|ID|结论|依据|最终观察|', '|---|---|---|---|']
    for row in audit['numbered_obligations']:
        lines.append('|'+ '|'.join([row['id'],row['verdict'],', '.join(row['evidence']),row['finding'].replace('|','/')])+'|')
    lines += ['', '## §15 最终状态与 §16 边界', '', '所有原始状态字段逐项保存在配套 JSON 的 `section15_final_fields`，与封存 `status.json` 完全相等。实现、研究、教师证据、训练、VAL、TEST 为 true；对 R/N/S 的总体学习收益分别为 false。D8 A/B完成4/1（各分母4），近质量2/1；R12完成与近质量均4/4。', '',
        '本轮只给两个高层初值，搜索/控制/安全权限未扩展。候选/actual预算、失败和来源分账完整。当前维护修复的12个教师摘要与原 producer 完全一致，原权重、数据、报告和 frozen_source 不改写。', '', '## 明确保留的限制', '']
    lines += ['- '+limit for limit in audit['limitations']]
    (DOC/'C3_FINAL_COMPLETION_AUDIT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(dict(status=audit['status'], numbered_obligations=len(rows), evidence_files=len(evidence)),ensure_ascii=False))


if __name__ == '__main__':
    main()
