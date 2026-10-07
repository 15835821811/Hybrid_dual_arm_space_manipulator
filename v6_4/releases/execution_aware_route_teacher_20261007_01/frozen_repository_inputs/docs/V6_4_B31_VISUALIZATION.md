# V6.4-B.3.1 当前可视化与证据

当前分支 `v6.4-b3-1-execution-aware-route-teacher`；固定开发研究为一个母场景、四个任务、七种参考，共 28 槽（六条旧原件复用、22 条新增）。

**有限teacher较最佳常量低3.27%、较冻结几何规则低1.03%；同幅值v2没有一致优势，Diffusion收益未建立。**

完整 Task 与原安全门禁通过 **28/28**，质量可比较 **28/28**。安全 actual 后的质量后处理缺测单列，不能当作原 Task/安全失败。无训练、模型采样或独立泛化 TEST，deployment 为 `NOT_MET`。

沿用 2ms 物理步长、20ms 控制 tick 与原门禁；20ms wall-clock 不是本轮研究淘汰门槛。本补充不建立硬实时或连续时间安全证明。

[静态总览](../v6_4/visualization/execution_aware_route_teacher_20261007_01/index.html) · [全部 28 行 CSV](../v6_4/visualization/execution_aware_route_teacher_20261007_01/all_candidates.csv) · [有限教师与对照](../v6_4/visualization/execution_aware_route_teacher_20261007_01/teacher_records.json) · [结构化质量与成本](../v6_4/visualization/execution_aware_route_teacher_20261007_01/dashboard_data.json) · [portable 研究报告](../v6_4/releases/execution_aware_route_teacher_20261007_01/report.md) · [run03 response](../paper/review-traces/experiment-result-to-claim/2026-10-07_run03/response.md) · [run03 verdict](../paper/review-traces/experiment-result-to-claim/2026-10-07_run03/verdict.json)

![四任务、七参考的参考和实际横向投影](../v6_4/visualization/execution_aware_route_teacher_20261007_01/fig_reference_actual.png)

第一张图展示完整 route 窗内 reference-base、actual-base 与同任务非零 actual-zero actual。参考来自解析定义，actual 来自保存数组；失败前缀和缺测保留标记。

![实际路线质量与参考到实际响应对照](../v6_4/visualization/execution_aware_route_teacher_20261007_01/fig_route_quality.png)

第二张图对照 17D I_route、相关球路线净空、路径及参考到实际响应。两图覆盖全部候选；旧六条未保存的 QP 分量/约束活动不反算。球约束活动和全约束 I_route 不能当作该球的因果贡献；route minimum 与全程 witness 分开记录。

新成本分账来自完整报告（旧六条复用不计作新 actual）：

```json
{
  "actual_physics_steps": 297000,
  "private_preview_physics_steps": 297000,
  "independent_saved_torque_replay_steps": 297000,
  "native_geometry_query_calls": 737681587,
  "additional_route_quality_geometry_queries": 18415364,
  "qp_solve_calls": 29700,
  "preview_calls": 29700,
  "initial_and_anchor_geometry_queries": 5978,
  "phase_counts": {
    "actual": {
      "mj_step": {
        "started": 297000,
        "returned": 297000,
        "raised": 0
      },
      "mj_step2": {
        "started": 0,
        "returned": 0,
        "raised": 0
      },
      "mj_geomDistance": {
        "started": 361052677,
        "returned": 361052677,
        "raised": 0
      },
      "qp_solve": {
        "started": 29700,
        "returned": 29700,
        "raised": 0
      }
    },
    "independent_torque_replay": {
      "mj_step": {
        "started": 297000,
        "returned": 297000,
        "raised": 0
      },
      "mj_step2": {
        "started": 0,
        "returned": 0,
        "raised": 0
      },
      "mj_geomDistance": {
        "started": 376628910,
        "returned": 376628910,
        "raised": 0
      },
      "qp_solve": {
        "started": 0,
        "returned": 0,
        "raised": 0
      }
    },
    "private_preview": {
      "mj_step": {
        "started": 0,
        "returned": 0,
        "raised": 0
      },
      "mj_step2": {
        "started": 297000,
        "returned": 297000,
        "raised": 0
      },
      "mj_geomDistance": {
        "started": 0,
        "returned": 0,
        "raised": 0
      },
      "qp_solve": {
        "started": 0,
        "returned": 0,
        "raised": 0
      }
    }
  },
  "route_quality_phase_counts": {
    "actual": {
      "mj_step": {
        "started": 0,
        "returned": 0,
        "raised": 0
      },
      "mj_step2": {
        "started": 0,
        "returned": 0,
        "raised": 0
      },
      "mj_geomDistance": {
        "started": 0,
        "returned": 0,
        "raised": 0
      },
      "qp_solve": {
        "started": 0,
        "returned": 0,
        "raised": 0
      }
    },
    "independent_torque_replay": {
      "mj_step": {
        "started": 0,
        "returned": 0,
        "raised": 0
      },
      "mj_step2": {
        "started": 0,
        "returned": 0,
        "raised": 0
      },
      "mj_geomDistance": {
        "started": 0,
        "returned": 0,
        "raised": 0
      },
      "qp_solve": {
        "started": 0,
        "returned": 0,
        "raised": 0
      }
    },
    "private_preview": {
      "mj_step": {
        "started": 0,
        "returned": 0,
        "raised": 0
      },
      "mj_step2": {
        "started": 0,
        "returned": 0,
        "raised": 0
      },
      "mj_geomDistance": {
        "started": 0,
        "returned": 0,
        "raised": 0
      },
      "qp_solve": {
        "started": 0,
        "returned": 0,
        "raised": 0
      }
    },
    "route_quality": {
      "mj_step": {
        "started": 0,
        "returned": 0,
        "raised": 0
      },
      "mj_step2": {
        "started": 0,
        "returned": 0,
        "raised": 0
      },
      "mj_geomDistance": {
        "started": 18415364,
        "returned": 18415364,
        "raised": 0
      },
      "qp_solve": {
        "started": 0,
        "returned": 0,
        "raised": 0
      }
    }
  },
  "total_new_native_geometry_query_calls_including_initial_and_route_quality": 756102929,
  "route_quality_phase_counts_are_separate_from_execution_phase_counts": true,
  "training_runs": 0,
  "model_sampling": 0,
  "seed_search": 0,
  "reducer_physics_steps": 0,
  "reducer_geometry_queries": 0,
  "reducer_qp_solves": 0,
  "reducer_DDIM_calls": 0,
  "reducer_optimizer_updates": 0
}
```

actual producer：`d4464c8ae2aa7913a730ebbd775917e7a3b1af71`。后处理、绘图与发布 producer/hash 独立绑定；`plot_manifest.json`、visualization/release manifests 及 run03 inputs 提供来源。两张 PNG 是保存数组与解析参考的文件绘图，不包含新增物理或几何查询；没有新增 PDF/视频。

独立审查：run03为partial / high，blocking_issues=[]；支持有限开发集结论，下游论文写作因缺少paper claim audit保持provisional。 发布状态：研究结论已独立审阅；本地封存、便携导出和远端提交验证由publication_receipts中的收据记录。 下一步：结束本轮，保留有限质量向量和非学习基线，暂不训练新网络。

旧 B.3 六槽完整安全通过，但 c+/c− 的最佳方向相对零参考改善仅 2.79%/3.81%，低于原 10% 门槛，旧停止结论保持。旧 P2/P3 与五组 TEST 仍为 `NOT_RUN_PILOT_STOP`；本补充为开发研究，独立 TEST 为 `NOT_RUN_DEVELOPMENT_STUDY`。

历史：[B.3 图表](../v6_4/visualization/conditional_route_value_20261007_01/index.html) · [B.3 原报告](../v6_4/releases/conditional_route_value_20261007_01/report.md) · [B.2 图表](../v6_4/visualization/task_anchored_residual_20261007_01/index.html) · [V6.2 历史](V6_2_LATEST_VISUALIZATION.md)。历史媒体、结果与 manifest 保持原来源。
