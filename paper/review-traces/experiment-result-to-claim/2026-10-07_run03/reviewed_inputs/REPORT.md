# V6.4-B.3.1 控制器感知参考与有限质量教师

固定开发槽位 28；完整 Task 与原五门禁通过 28，质量指标可比 28；旧原件复用 6；新固定槽位 22。
本轮只评价一个母场景上的四个开发任务、七种固定参考。没有训练、模型采样或独立泛化 TEST；部署仍为 `NOT_MET`。

v2 为 r=0.25、f=0.75 的 C² 提前平台参考。解析位置、速度和加速度经过原参考门禁；实际消费版本与 z 以独立参考绑定为准。旧 B.3 封存结论保持不变。

| 槽位 | 任务 | 模式 | 来源 | actual状态 | 质量阶段 | 完整安全 | 质量可比 | I_route rad/s | 全程路径 m | 路线球净空 mm |
|---|---|---|---|---|---|---|---|---:|---:|---:|
| EA_00 | b3_mother_00_c_plus | z0 | REUSE_ORIGINAL_EVIDENCE | TASK_COMPLETED | QUALITY_METRICS_AVAILABLE | True | True | 0.108601632 | 1.11027634 | 25.8842929 |
| EA_01 | b3_mother_00_c_plus | v1_plus12 | REUSE_ORIGINAL_EVIDENCE | TASK_COMPLETED | QUALITY_METRICS_AVAILABLE | True | True | 0.111963142 | 1.1177148 | 25.7778153 |
| EA_02 | b3_mother_00_c_plus | v1_minus12 | REUSE_ORIGINAL_EVIDENCE | TASK_COMPLETED | QUALITY_METRICS_AVAILABLE | True | True | 0.105570475 | 1.10553 | 25.8612794 |
| EA_03 | b3_mother_00_c_plus | v2_plus12 | NEW_FIXED_ACTUAL_SLOT | TASK_COMPLETED | QUALITY_METRICS_AVAILABLE | True | True | 0.112543343 | 1.12962589 | 25.7707794 |
| EA_04 | b3_mother_00_c_plus | v2_minus12 | NEW_FIXED_ACTUAL_SLOT | TASK_COMPLETED | QUALITY_METRICS_AVAILABLE | True | True | 0.10579419 | 1.10915752 | 25.8714508 |
| EA_05 | b3_mother_00_c_plus | v2_plus20 | NEW_FIXED_ACTUAL_SLOT | TASK_COMPLETED | QUALITY_METRICS_AVAILABLE | True | True | 0.125254979 | 1.14855294 | 25.7214791 |
| EA_06 | b3_mother_00_c_plus | v2_minus20 | NEW_FIXED_ACTUAL_SLOT | TASK_COMPLETED | QUALITY_METRICS_AVAILABLE | True | True | 0.106562306 | 1.11625243 | 25.9183526 |
| EA_07 | b3_mother_00_c_minus | z0 | REUSE_ORIGINAL_EVIDENCE | TASK_COMPLETED | QUALITY_METRICS_AVAILABLE | True | True | 0.112069755 | 1.11029111 | 25.8777431 |
| EA_08 | b3_mother_00_c_minus | v1_plus12 | REUSE_ORIGINAL_EVIDENCE | TASK_COMPLETED | QUALITY_METRICS_AVAILABLE | True | True | 0.10779655 | 1.10554135 | 25.9001972 |
| EA_09 | b3_mother_00_c_minus | v1_minus12 | REUSE_ORIGINAL_EVIDENCE | TASK_COMPLETED | QUALITY_METRICS_AVAILABLE | True | True | 0.114338061 | 1.11773974 | 25.7664573 |
| EA_10 | b3_mother_00_c_minus | v2_plus12 | NEW_FIXED_ACTUAL_SLOT | TASK_COMPLETED | QUALITY_METRICS_AVAILABLE | True | True | 0.106783203 | 1.10915794 | 25.8776933 |
| EA_11 | b3_mother_00_c_minus | v2_minus12 | NEW_FIXED_ACTUAL_SLOT | TASK_COMPLETED | QUALITY_METRICS_AVAILABLE | True | True | 0.115185538 | 1.12965793 | 25.7735153 |
| EA_12 | b3_mother_00_c_minus | v2_plus20 | NEW_FIXED_ACTUAL_SLOT | TASK_COMPLETED | QUALITY_METRICS_AVAILABLE | True | True | 0.10318377 | 1.11626087 | 25.9269321 |
| EA_13 | b3_mother_00_c_minus | v2_minus20 | NEW_FIXED_ACTUAL_SLOT | TASK_COMPLETED | QUALITY_METRICS_AVAILABLE | True | True | 0.119822969 | 1.1485902 | 25.7315476 |
| EA_14 | b31_mother_00_d055_c_plus | z0 | NEW_FIXED_ACTUAL_SLOT | TASK_COMPLETED | QUALITY_METRICS_AVAILABLE | True | True | 0.105289264 | 1.10180087 | 25.8724307 |
| EA_15 | b31_mother_00_d055_c_plus | v1_plus12 | NEW_FIXED_ACTUAL_SLOT | TASK_COMPLETED | QUALITY_METRICS_AVAILABLE | True | True | 0.108692853 | 1.10831409 | 25.8982963 |
| EA_16 | b31_mother_00_d055_c_plus | v1_minus12 | NEW_FIXED_ACTUAL_SLOT | TASK_COMPLETED | QUALITY_METRICS_AVAILABLE | True | True | 0.105368773 | 1.09996164 | 26.319935 |
| EA_17 | b31_mother_00_d055_c_plus | v2_plus12 | NEW_FIXED_ACTUAL_SLOT | TASK_COMPLETED | QUALITY_METRICS_AVAILABLE | True | True | 0.108712061 | 1.11763808 | 25.8839181 |
| EA_18 | b31_mother_00_d055_c_plus | v2_minus12 | NEW_FIXED_ACTUAL_SLOT | TASK_COMPLETED | QUALITY_METRICS_AVAILABLE | True | True | 0.105699691 | 1.1053662 | 26.3479169 |
| EA_19 | b31_mother_00_d055_c_plus | v2_plus20 | NEW_FIXED_ACTUAL_SLOT | TASK_COMPLETED | QUALITY_METRICS_AVAILABLE | True | True | 0.111474703 | 1.13559983 | 25.8030219 |
| EA_20 | b31_mother_00_d055_c_plus | v2_minus20 | NEW_FIXED_ACTUAL_SLOT | TASK_COMPLETED | QUALITY_METRICS_AVAILABLE | True | True | 0.106668739 | 1.11566906 | 34.2270395 |
| EA_21 | b31_mother_00_d055_c_minus | z0 | NEW_FIXED_ACTUAL_SLOT | TASK_COMPLETED | QUALITY_METRICS_AVAILABLE | True | True | 0.107260164 | 1.10179752 | 25.8794386 |
| EA_22 | b31_mother_00_d055_c_minus | v1_plus12 | NEW_FIXED_ACTUAL_SLOT | TASK_COMPLETED | QUALITY_METRICS_AVAILABLE | True | True | 0.103902872 | 1.0999621 | 26.3106094 |
| EA_23 | b31_mother_00_d055_c_minus | v1_minus12 | NEW_FIXED_ACTUAL_SLOT | TASK_COMPLETED | QUALITY_METRICS_AVAILABLE | True | True | 0.112167077 | 1.10831466 | 25.9008549 |
| EA_24 | b31_mother_00_d055_c_minus | v2_plus12 | NEW_FIXED_ACTUAL_SLOT | TASK_COMPLETED | QUALITY_METRICS_AVAILABLE | True | True | 0.103430713 | 1.10536742 | 26.3666171 |
| EA_25 | b31_mother_00_d055_c_minus | v2_minus12 | NEW_FIXED_ACTUAL_SLOT | TASK_COMPLETED | QUALITY_METRICS_AVAILABLE | True | True | 0.111869172 | 1.11765278 | 25.8774495 |
| EA_26 | b31_mother_00_d055_c_minus | v2_plus20 | NEW_FIXED_ACTUAL_SLOT | TASK_COMPLETED | QUALITY_METRICS_AVAILABLE | True | True | 0.103335623 | 1.11567488 | 34.2031867 |
| EA_27 | b31_mother_00_d055_c_minus | v2_minus20 | NEW_FIXED_ACTUAL_SLOT | TASK_COMPLETED | QUALITY_METRICS_AVAILABLE | True | True | 0.113969963 | 1.13566592 | 25.7953663 |

失败、预检拒绝和执行拒绝保留在全部槽位和成功率分母内；其前缀指标单列，不进入成功轨迹质量均值。

## 固定有限教师

按原 I_route 排序，使用冻结的 0.001 rad/s tie-band。每组锚定其最低代价，不通过链式近邻扩大并列组；组内按全程路径、系数范数和候选 ID 排序。并列表示固定工程分辨率，不表示统计等价。

| 任务 | 合格/7 | 严格最低槽位 | 最低 I_route | tie-selected | selected I_route | 几何规则槽位 |
|---|---:|---|---:|---|---:|---|
| b3_mother_00_c_plus | 7/7 | EA_02 | 0.105570475 | EA_02 | 0.105570475 | EA_04 |
| b3_mother_00_c_minus | 7/7 | EA_12 | 0.10318377 | EA_12 | 0.10318377 | EA_10 |
| b31_mother_00_d055_c_plus | 7/7 | EA_14 | 0.105289264 | EA_16 | 0.105368773 | EA_18 |
| b31_mother_00_d055_c_minus | 7/7 | EA_26 | 0.103335623 | EA_22 | 0.103902872 | EA_24 |

严格最低代价与 tie-selected 分开保存；所有同 z 的 v1/v2 差异、原 10%/0.001 标准、路径/净空/基座代价冲突见 `teacher_records.json`。

## 零残差、几何规则与最佳常量

| 规则 | 完整安全/4 | 质量可比/4 | 四任务均值 I_route | 成功子集均值（分母可能不同） | 七模式共同完整任务子集均值 |
|---|---:|---:|---:|---:|---:|
| z0 | 4/4 | 4/4 | 0.108305204 | 0.108305204 | 0.108305204 |
| v1_plus12 | 4/4 | 4/4 | 0.108088854 | 0.108088854 | 0.108088854 |
| v1_minus12 | 4/4 | 4/4 | 0.109361096 | 0.109361096 | 0.109361096 |
| v2_plus12 | 4/4 | 4/4 | 0.10786733 | 0.10786733 | 0.10786733 |
| v2_minus12 | 4/4 | 4/4 | 0.109637148 | 0.109637148 | 0.109637148 |
| v2_plus20 | 4/4 | 4/4 | 0.110812269 | 0.110812269 | 0.110812269 |
| v2_minus20 | 4/4 | 4/4 | 0.111755994 | 0.111755994 | 0.111755994 |
| geometric_rule_v2_away12 | 4/4 | 4/4 | 0.10542695 | 0.10542695 | 0.10542695 |
| finite_teacher_minimum | 4/4 | 4/4 | 0.104344783 | 0.104344783 | 0.104344783 |
| finite_teacher_tie_selected | 4/4 | 4/4 | 0.104506473 | 0.104506473 | 0.104506473 |

`V_cond = min_z mean_c J(z,c) - mean_c min_z J(z,c)`。它只反映固定开发任务和有限候选上的离线条件选择潜力。

- full_development_matrix: `AVAILABLE`；V_cond = 0.00352254719 rad/s；任务数 4。
- common_complete_task_subset: `AVAILABLE`；V_cond = 0.00352254719 rad/s；任务数 4。

缺失或失败没有被填入惩罚代价。全部七种模式共同完成的任务子集与完整四任务矩阵分别报告；最佳常量包含零残差，不只比较最差固定方向。

## 原 B.3 门槛

原函数对每方向与 zero、opposite 的比较采用 OR：任一比较满足完整成功对失败的 A 条件，或 10% 且 0.001 rad/s 的 B 条件，可进入 favorable；再要求两侧实际代价优选方向合格且反转。两种比较都报告，不把原规则改成 AND。

| 距离 mm | 参考族 | 可解释状态 | 原 OR 与两侧反转判定 |
|---:|---|---|---|
| 43 | v1_12mm | AVAILABLE | False |
| 43 | v2_12mm | AVAILABLE | False |
| 43 | v2_20mm | AVAILABLE | True |
| 55 | v1_12mm | AVAILABLE | False |
| 55 | v2_12mm | AVAILABLE | False |
| 55 | v2_20mm | AVAILABLE | False |

43mm/v1 六条原件仍不满足原门槛；任何新组结果只属于本补充，不改变旧 B.3 停止结论。安全 actual 后质量后处理缺测时相关判定 unavailable，不能把缺测当作碰撞失败。另报的 AND 字段明确属于更严格的 supplementary 检查。

## 向量与路线响应

新运行的 10D/7D 干预分量、连续体参考/反馈期望/实际速度与误差、已有相关球行活动和绑定时长、力矩饱和来自保存数组。selected task velocity 是命令诊断，不能替代实际物理速度；I_route 是全部约束共同作用的 17D 指标，不能称作单球贡献。旧六槽的缺失向量和球行活动为 `NOT_MEASURED_OLD_EVIDENCE`。

完整合格结果按保存的 2ms 时刻报告 reference-base、actual-base、actual-reference 和同任务非零 actual-zero actual 的全窗口响应；不插值重建轨迹，不推断全域传递增益。见 `quality_matrix.json`。

## 成本与结论边界

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

actual、private preview、独立保存力矩 replay、原生几何、附加路线净空查询分别计账。旧六条复用不算新 actual；本 reducer 新物理、几何、QP、DDIM、optimizer 均为 0。

后续判断：`REFERENCE_VALUE_REQUIRES_COST_AND_GEOMETRIC_RULE_REVIEW_DIFFUSION_NECESSITY_UNESTABLISHED`。即使参考或有限教师改善，也不能由本轮宣称 Diffusion 优势、生成模型必要性、独立泛化、硬实时或连续时间安全。独立审阅与 GitHub 发布由后续交付步骤完成。

## 最终七问（研究结论；发布状态另由收据记录）

### 1. 补充研究是否按协议完成

是。4×7固定槽位全部终态，6条严格复用、22条新增；28/28完整Task与原五门禁通过，28/28质量可比。没有重试、补候选、训练、模型采样或seed搜索。研究执行完成，GitHub封存与远端核验另由发布收据记录。

### 2. v2是否实际消费并保持原任务锚点

是。16条v2全部完整消费独立v2版本、definition、plan和z，原Task锚点、姿态及终态门禁全部通过；23.98s保护与解析p/v/a由冻结参考测试验证。v1与零残差数值路径保留。

### 3. 非零参考是否产生完整且可比的任务结果

是。24条非零参考全部完整安全且质量可比。20mm参考能改变整个路线窗口中的实际运动；这不等于改善质量或证明控制器传递增益。55mm正侧严格最低I_route仍是零残差，不能强制非零标签。

### 4. 是否降低原I_route，是否牺牲其他代价

存在任务相关的小幅收益，但提前平台不一致更优。同任务同z的8组12mm对照仅3组v2降低I_route，5组升高，8组全程路径均变长；最大降低约0.00101335rad/s（0.94%）。严格teacher相对各自零残差改善依次2.79%、7.93%、0%、3.66%，无任务达到相对零残差10%门槛。43mm负侧与55mm负侧严格最小teacher的全程路径分别增加约5.97mm和13.88mm，伴随基漂移/净空冲突；所有原门禁仍通过。

### 5. 有限条件选择相对最佳固定模式和几何规则的潜力

四任务全部七模式完整可比，V_cond=0.00352254719rad/s。严格有限teacher均值0.104344783rad/s，最佳常量v2_plus12为0.107867330，零残差0.108305204，冻结几何规则0.105426950；teacher分别比最佳常量、零残差和几何规则低3.27%、3.66%和1.03%。tie-selected均值0.104506473，与严格最低分别报告。仅为有限开发集的回顾性选择潜力，未实现条件policy，也无统计或泛化保证。

### 6. 原B.3标准是否仍未满足

旧43mm/v1六槽仍未满足，封存停止结论不变。本补充必须另报：43mm/v2_20按原zero OR opposite及两侧反转谓词通过，收益来自相对反方向14.92%/13.89%，而相对零残差仅1.88%/7.93%。其余五个同幅值距离/版本组均不通过。不能说本补充所有组都未通过，也不能写成相对零残差改善超过10%；更严格AND均未通过。

### 7. 是否值得另立新学习试验

本轮结束，暂不启动新的Diffusion训练试验。保留有限质量向量、零残差/幅值近优组和简单几何或检索基线：相对几何规则的严格oracle平均优势仅约1.03%，四任务只有一个母场景，且存在路径与漂移冲突，尚无生成模型必要性或同预算搜索成本收益证据。若未来有独立工程需求，再另立按母任务分组的独立TEST协议，包含几何、检索、幅值匹配随机、正确/错配条件，K1固定、K4另报完整成本；本轮不执行这些未来试验。


严格最低与冻结tie-break的区别：55mm正侧最低为EA_14/z0，tie-selected为EA_16/v1_minus12（全程路径更短）；55mm负侧最低为EA_26/v2_plus20，tie-selected为EA_22/v1_plus12。0.001rad/s并列组是工程分辨率，不能当作统计等价或强监督偏好。


|任务|预声明远离侧20mm模式|route参考−base世界3D RMS mm|route actual−base世界3D RMS mm|route actual−z0世界3D RMS mm|全27s actual−z0世界3D RMS mm|
|---|---|---:|---:|---:|---:|
|b3_mother_00_c_plus|v2_minus20|16.684441|17.253843|13.706981|4.946163|
|b3_mother_00_c_minus|v2_plus20|16.684441|17.260658|13.698688|4.943171|
|b31_mother_00_d055_c_plus|v2_minus20|16.684441|16.711087|14.808427|5.343620|
|b31_mother_00_d055_c_minus|v2_plus20|16.684441|16.716829|14.806674|5.342987|

以上为保存原生2ms全窗口的描述性响应；图1为固定世界第一横向投影，图2响应柱为同轴RMS，不能与此世界3D范数表混用。中点切片不是全域传递增益或最小必要避让量。


相关连续体—球路线最低净空的时刻与保存状态已记录，但该局部minimum对应的几何对witness没有保存，保留`NOT_SAVED_FOR_ROUTE_WINDOW_MINIMUM`；全程global witness不替代路线局部witness。这是诊断证据缺口，不把它写成已测量。


执行账本合计actual、private preview、独立保存力矩replay各297000物理步，QP29700次，执行几何737681587、附加路线几何18415364、初始/锚点5978，总原生几何756102929。另列必要日志检查的4次原约束QP调用和4次既有无约束名义计算；不算新actual，在线未增加第二个求解。绘图仅两张PNG，物理/几何/QP/模型调用为0，无新PDF/video。


研究范围保持research_simulation，部署`NOT_MET`。不宣称Diffusion优势、全域鲁棒性、生成模型必要性、硬实时或连续时间安全。完整四任务机器结论见`final_conclusions.json`；独立审阅在科学内容稳定后另行绑定。
