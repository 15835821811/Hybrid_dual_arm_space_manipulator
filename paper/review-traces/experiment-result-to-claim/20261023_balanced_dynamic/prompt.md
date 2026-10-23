# 独立 experiment-result-to-claim 审查任务

审查角色：全新独立 paper architect/reviewer。根委托本审查员，只读审查 `artifacts/v7_1/balanced_tracking_001`。冻结计划为 `ff_prior_balanced_tracking/docs/EXPERIMENT_PLAN.md` 与同一单元总计划 `ff_prior_balanced_restore/docs/EXPERIMENT_PLAN.md`。使用 experiment-result-to-claim、review-tracing 与 assurance-contract 合同。

本单元原授权为一个 2R+5C APP_DEV 静态→真实末态动态工程单元，不训练、扩库、对照或 TEST，不增加候选、种子、调参或第二修复。旧静态永久收尾，不重审旧 raw。仅以 `ff_prior_fullchain/evidence/balanced_restore_closeout.json`、静态独立 verdict 和新动态绑定 `terminal.npz` 核实静态前驱；历史绑定文件只核 hash。审查过程中根转达用户新请求修复后两点/保持失败；该请求由根在独立身份处理，本次审查范围仍仅原失败 run，绝不追认或改写其失败。

禁止运行算法、物理、FK、新距离查询、优化器、重放或 resume。禁止修改冻结源和 run；唯一写目录为本 trace。审查所有新动态 native/replay 数组元素、全部输入/source zip 身份、geometry/constraints/body 事件、控制前后命令、筛选及遗漏、七点到达分母、停止原因与预算；只重建保存数组足够支持的门，对未保存中间量明确限制。不得以采样核对代替完整数组审查。

需分别裁决 C1 完整静态+动态组合、Cprefix 真实连续交接后的已执行动态部分与其完整重放、Caux 臂身描述与约束参与。静态通过不能覆盖动态失败。不能声称因果优势、神经自主绕障、实体卫星漂移、全梯度通过、连续时间证明或硬件/泛化成功。

生产待核声明：终态 FINISHED_WITH_BLOCKED_ITEMS、EXECUTION_FAILED；9830/13500 步实际执行及 9830 步重放；836 差零；交接 exact；停止 NO_FEASIBLE_V7_DOMAIN_VELOCITY，absolute 47.16 / local 19.66 s；前五点 recorded_through_arrival=true，W6/W7=false，后两点终态距离不能改称到达误差；body 984/1351 PARTIAL；最低距离 0.268222 mm；已执行时刻位置<5mm；完整时长和保持为 false。187 源/256 输入、生产 Git c9e8980e、原算法基点5a21e088、0新neural均需独立核实。

根负责且只做一次完成 resume，本审查员不能调用。后补收据 `ff_prior_balanced_tracking/evidence/completed_001/receipt.json` 报告 80 文件426455409bytes前后hash/size全同、NO_WORK_ALREADY_COMPLETE、stdout500bytes/stderr0，须检查原始清单和输出。原媒体 PARTIAL，820 帧固定15fps使47.16s物理记录播放54.6667s；原run保持不改，独立delivery修正由根处理。本审查不扩为 paper claim audit；缺失该文件则任何写作结论 provisional。

输入明细、hash、size、时间戳见 `inputs.json`；所有数组字段/形状/元素统计与检查见 `numerical_audit.json`。执行方式为 `audit_saved.py` 和补齐索引/日志来源的 `audit_finish.py`。两者均只导入标准库与 NumPy，不导入项目、MuJoCo 或优化器。另委托一个只读源码审查员核对门语义、停止分支和未保存中间量；科学裁决由本独立审查员完成。
