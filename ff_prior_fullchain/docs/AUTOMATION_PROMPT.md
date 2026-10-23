继续自由漂浮路径—避障—连续目标追踪长期工程。工作区E:\桌面\mujoco_irregular_waypoints_diffusion_adapter_bundle_20260902-main\Hybrid_dual_arm_space_manipulator。当前以DIFFUSION_GOAL_CURRENT.md、ff_prior_fullchain/control.json及最新明确用户指示为准，不伪造complete/paused来替换旧goal卡。

最新用户2026-10-23指出刚性侧障碍过密、连续体侧过少，要求原七不规则航点及臂身影响，已选择刚性侧2个疏散障碍、连续体侧5个（3个末端航点段+2个中段/中远段臂身）。完整记录ff_prior_balanced_scene/evidence/user_layout_selection.json。已交付具体场景布局、初态检查和新完整初态工程预算草案ff_prior_balanced_scene/docs/LAYOUT_AND_BUDGET.md，预览http://127.0.0.1:8770/index.html。当前新增实际运行范围待确认：选择问题明确是先给布局与预算；不能把已有原五障碍“仅一个单元”授权自动扩为第二静态单元。没有新明确范围时，只读统一 .\ff_prior_fullchain\job.ps1 status，没有变化静默结束，不重复提问、报告、审查或旧raw分析。generic action IMPLEMENT_AND_FREEZE_NEXT_PHASE不覆盖next_phase=BALANCED_2R5C_LAYOUT_READY_NEW_FULLCHAIN_RUN_SCOPE_PENDING。不要自动启动旧五障碍动态或新七障碍worker。已选障碍数量不需再问。

新布局R1=(.82,-.91,.22),r=.035；R2=(1.24,-.43,-.045),r=.030，两球面相距.625670m。C1/C2/C3为原W1-W2/W3-W4/W5-W6中点，r=.025；C4=(1.12,.75,.025),r=.028；C5=(1.50,.54,-.04),r=.028。准确浮点/原七航点见evidence/layout.json。只做一次固定初态几何检查，原836数值逐项保持、新模型身份独立，48精确距离查询，3406合法对，最小8.540646mm无穿透；0耦合/0原生。这不是新轨迹资格或动态结果，不能放大为整条路径通过。全机几何门已有，后续还须保存每障碍/臂段距离、几何ID、约束活跃及过滤前后命令，展示臂形作用。

新工程单元草案须在同一个2+5场景从原完整数值初态重新开始，不能在旧真实末态中替换障碍后冒称连续：27s接近+.5s保持，模型/执行/完整836重放及必要审查通过，才从该新场景真实完整末态接27s动态，保持原七航点、姿态、各段时长。草案最多28轮323恢复，1候选；静态11000耦合/15000秒/12GiB/27500原生含重放，动态2000耦合/1200秒/2GiB/27000原生，另1GiB收尾；合计13000耦合/16200秒/54500原生。代码/参考/预算/源和输入身份须实现并冻结后才能运行。无额外候选、种子、调参、第二修复、扩库/训练/优势对照/TEST。没有新增物理许可不能启动，剩余额不是许可。

horizon_restore_001原五障碍静态已经永久收尾。报告ff_prior_horizon_restore/docs/ANALYSIS.md；收据ff_prior_fullchain/evidence/horizon_restore_closeout.json；报告提交09a4535ce6616dd146a097be9cbf7d6898f9608f，原始byte归档修正后提交68e981f3eda1225cd1f2c9835e652dc2930865d9。Git基点bfc4ea9c436a32bb5fc9aa49db4e66857e11283f加142实际源码hash/zip和233绑定输入为身份；用户已有hierarchical_qp工作树差异在运行前已冻结归档且本轮不调用，不能称纯HEAD全部一致。MODEL_VALID1/1、EXECUTION_VERIFIED1/1、13750执行+13750独立836重放，13751×836差0。468.375秒、2681耦合、1987216预留模型区间（非原生步）、8恢复轮。实际保持段最大位置误差连续体/刚性.035850/.026383mm，最小间隙.233857mm，最大几何步.214364mm，原15执行门通过。0新神经采样、训练或数据。历史Diffusion20264627只作为未合格初猜，系统成功不能归给裸网络。

本新独立审查paper/review-traces/experiment-result-to-claim/20261023_horizon_restore已完成，C1/Caux yes/high；2249检查=2248直接一致+1项Git身份差异已解释，0未决。无paper claim audit，写作provisional。32半步/正弦观察无冻结pass门；前6轮SLSQP达180上限，后2轮23/19，不能称全部收敛。未保存质量矩阵、逐扰动残差、逐对距离/剪枝、geom_rbound、momentum exchange、完整callback等核验边界已列，不能声称全部物理量独立重建或连续时间碰撞证明。唯一完成resume已PASS/NO_WORK_ALREADY_COMPLETE；127核心文件458379716bytes完整清单/hash/size不变，前清单先持久化，stdout468/stderr0原始bytes保留且提交hash校验一致。严禁再resume、审查、重建神经、旧raw重复分析或追加本run物理。

D3已用合计46454耦合/2458.484秒/1224791218核心bytes/27500原生。新草案上限后D3最多59454耦合/18658.484秒/82000原生及旧bytes+15GiB<20GiB；全周期保守79454耦合/40258.484秒/229500原生及36.141GiB<12h/40GiB/80000/312500原额。此账本不授权更多轮次。

ff_prior_horizon_tracking只有未冻结未运行代码草稿，inputs.freeze有最新next_phase守卫；本次用户场景修改后不自动继续旧五障碍动态。原五障碍媒体http://127.0.0.1:8769/index.html真实静态27.5秒，有双臂EE/Goal RGB XYZ轴与累计末端轨迹、7组曲线、两视频和交互三维；未包含动态七航点。用户要求最终全部视频/三维再加W1–W7标注、臂段高亮/距离连线、距离热图与形状/曲率曲线。动态目标轴随保存目标序列同步；新run未有原生状态不得造成功动画。已打开8770只是明确标示的布局草案。

所有旧contact_steer、world_steer、D1/D2/D2b/P0–P5均已永久收尾，禁止再次审查/resume/物理。旧九障碍D2b七点通过只属预定义C2参考+反馈，不是网络自主避障、实体卫星漂移或原五障碍替代。8766/8767/8768/8769/8770是只读媒体服务，保留不杀。保留旧数据模型结果、用户修改、main/分支/旧作业，不强推或上传大数据。

保持不穿透、5mm/1度、2mm全机几何步长、完整初态、17→67、自由漂浮及全部运动/力矩/动量/836重放门；无实际状态裁剪/重置、失败目标重标、mock、抽查升级。DEV/07不入TRAIN，不偷用07答案。不训练/扩库/1024/30000/pilot/study/strict_multimodal/最终TEST/硬件/接触捕获。实体移动卫星必须有物理一致起源，不能给零速前驱暗注速度或把虚拟目标运动写成实体漂移。新组合仅APP_DEV，软件/模型/执行/网络贡献分开，优势不是工程推进门。长期工程仍缺新布局真实连续追踪及物理一致移动卫星，不标complete。阶段结束不暂停/删除唯一heartbeat；同一真实范围连续三次无独立进展按goal规则blocked，新明确授权重置计数。仅新阶段切换、完成、失败或需用户处理时通知，无变化静默；不重复询问已明确的同一动作。
