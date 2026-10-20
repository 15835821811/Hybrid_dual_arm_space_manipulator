import json
from pathlib import Path
D=Path(__file__).resolve().parent
a=json.loads((D/"reviewer_raw_audit.json").read_text(encoding="utf-8"))
s=json.loads((D/"reviewer_supplemental.json").read_text(encoding="utf-8"))
t=json.loads((D/"reviewer_trajectory_algebra.json").read_text(encoding="utf-8"))
C1={
 "claim_supported":"no",
 "what_results_support":["冻结单训练种子、同表示/数据/损失/更新预算下，Transformer无引导姿态均值与中位改善，位置及Q中位改善；存在有限辅助指标收益。","完整16步无物理引导、共同prepare后96配对样本，MLP与Transformer全部门合格率都为0/96，缺失端点0。"],
 "what_results_dont_support":["预定主要指标没有改善；位置均值19.016722→21.088268mm、P95 31.097930→43.740550mm，Q均值4.417744→4.571178、P95 6.722665→10.317055，不能称整体世界终点质量改善。","不能据本轮认定MLP是主要瓶颈、Transformer普遍更差或无效；不能把架构包差异单独归因注意力。"],
 "missing_evidence":["跨独立训练种子的可重复收益。","达到5mm/1度及全部几何门的无引导改善。","保持最终TEST隔离的更广任务分布证据；当前48VAL×2不是96个独立目标，也不是训练重复。"],
 "suggested_claim_revision":"在本冻结预算的探索性配对比较中，小型Transformer改善了部分中心误差统计，但未改善无引导全部门合格率，且位置及综合Q的均值与尾部退化；尚无证据证明其整体改善无引导世界终点质量。",
 "next_experiments_needed":["本轮不执行任何追加实验。若未来另获授权，应先冻结新的假设和判定指标，再进行独立训练种子的等预算重复；优先解释尾部退化，而非按本轮结果择优追加种子或训练步数。"],
 "confidence":"high"
}
C2={
 "claim_supported":"partial",
 "what_results_support":["在固定12个已用DEV任务×3采样种子、每任务5候选及共同late4下，两新主干各180/180候选合格、36/36执行重放通过。","相对本轮从头训练MLP，Transformer的执行位置均值0.787201→0.659327mm，降低0.127875mm（约16.24%），位置P95 1.225679→0.950734mm，姿态均值0.093666→0.080320度；位置26/36配对更好、10/36更差。","三采样种子均值、原DEV4及已用8任务分层的执行位置均值均改善；该结果支持有限开发任务上的末端精度收益。"],
 "what_results_dont_support":["四方法成功率相同，不能宣称成功率优势；完整流程通过不证明无引导生成合格。","不能宣称普遍规划优势、跨训练种子稳定性或样本效率优势；同一训练模型下的36执行不是36个独立训练重复。","条件均值执行位置均值0.366634mm、姿态0.043928度，比Transformer更低；Transformer相对条件均值位置仅8/36更好。","Transformer候选位置/姿态最大误差比MLP更大；执行收益依赖固定合格候选选择和引导流程，不能反推整体无引导分布改善。"],
 "missing_evidence":["独立训练重复及预先规定的任务层统计分析。","更广且未用于开发的任务分布；当前任务已使用，非最终TEST。","没有同任务关闭引导的DEV消融；无引导在VAL、执行在DEV，不能跨表相减作为引导的独立因果贡献。","若要归因注意力，需将归一化、残差、激活等因素分开控制；本轮不具备此设计。"],
 "suggested_claim_revision":"在单训练种子、固定12个已用开发任务和共同late4流程中，小型Transformer相对本轮MLP显示执行末端误差的探索性改善；四方法成功率持平，且尚未超过条件均值基线的平均精度，不能据此认定普遍规划优势。",
 "next_experiments_needed":["仅在另行批准的新计划中考虑多训练种子等预算重复和任务层分析；保留简单基线与原物理门。本轮无需为完成工程流程而追加实验，不启动扩库、pilot、study、strict_multimodal或TEST。"],
 "confidence":"medium"
}
eng={
 "status":"PASS",
 "claim_supported":"yes",
 "scope":"仅本轮360新增模型候选、72实际MuJoCo原生67力矩执行及原运行独立重放证据；非硬件、非全库执行证明。",
 "what_results_support":["源码/输入身份、训练过程证据、无引导采样来源、固定DEV候选/预执行选择、已保存轨迹与执行门形成闭环；达到每模型144/180、29/36工程计数门，且无未知工作。","逐条新轨迹代数复核通过，工程有效不依赖C1或相对优势。"],
 "what_results_dont_support":["未执行候选或训练库全部具有力矩可执行性。","连续时间无碰撞证书、硬件验证、动态环境或任意初态适用性。"],
 "missing_evidence":["独立重放没有单独保存第二条原始轨迹；本次核对原始重放残差、计数与冻结重放实现，没有重新进行物理仿真。","完整积分初态由任务身份、输入绑定和冻结初始化代码支撑；native_trace单独不包含所有初态applied-force/mocap/userdata字段。"],
 "confidence":"high"
}
verdict={
 "review_role":"fresh independent paper architect/reviewer",
 "experiment":"artifacts/v7_1/backbone_pair_smoke_002",
 "frozen_source":"4ccba22122c5ad241fe2d701ff907d82d99d41e4",
 "claims":{"C1":C1,"C2":C2},
 "engineering_verdict":eng,
 "paper_claim_audit":"unavailable",
 "downstream_writing_status":"provisional — no paper claim audit available",
 "pending_external_review":False,
 "recommend_replace_default_model":False,
 "recommend_continue_dataset_expansion":False,
 "permitted_next_stage":"完成本轮中文结果分析、版本化实验记录与有界结果报告；可暂定写作上述限定结论。科学路线收窄，不自动进入优势确认、扩库或TEST。新实验须单独计划和授权。",
 "routing_action":"C1收窄/否定当前整体改善主张；C2仅记录有限探索性精度收益；工程完成；保留模型和全部结果。",
 "verification_scope":{"raw_helper":a["raw_helper"],"source_files":168,"bound_inputs":28861,
   "selected_updates_per_model":[1,3000,6000],"training_updates_each":6000,"matched_segments":300,
   "unguided":192,"new_candidates":360,"new_executions":72,"reused_candidates":360,"reused_executions":72,
   "trajectory_algebra":"PASS","actual_coupled":29527,"reserved_coupled":41256,"coupled_limit":42000,
   "wall_s":a["budget"]["elapsed_s"],"output_bytes":a["budget"]["output_bytes"],"threads":2,
   "resume_handoff":{"source":"主代理执行一次；审查者仅核验其原始前后manifest与当前956文件hash","status":"PASS","resume":"NO_WORK_ALREADY_COMPLETE","files":956}},
 "evidence_limitations":[
   "仅两个固定主干各一个训练种子；采样种子非训练重复。",
   "无引导指标取共同prepare后的轨迹，非raw轨迹的另一次世界终点测量；raw神经采样精确重建，未调用传播。",
   "只对6个预选更新进行梯度/AdamW代数重建；其余更新由300段序列摘要、覆盖计数、最终状态和运行日志共同支撑，不声称逐步重建全部12000次优化。",
   "未重新调用mj_step、coupled propagation、optimizer.step、训练或resume；原执行重放是否正确依赖绑定源码及保存的执行证据。",
   "对全部72条native_trace重算可由已存数组推导的门；未重算几何距离、碰撞中点和质量矩阵。",
   "旧基线仅读取结果及身份哈希，不重跑旧物理审计；没有把旧360/72算成新增。",
   "几何是不穿透门和既定离散细化，continuous_time_certified=false；不是安全裕量或连续时间证书。",
   "14软件测试只作为软件证据，不作为物理证据。",
   "没有PAPER_CLAIM_AUDIT，因此下游论文表述为provisional。"],
 "artifacts":["response.md","reviewer_raw_audit.json","reviewer_trajectory_algebra.json","reviewer_supplemental.json"]
}
(D/"verdict.json").write_text(json.dumps(verdict,indent=2,ensure_ascii=False),encoding="utf-8")

def tbl(header,rows):
    return "\n".join(["|"+"|".join(header)+"|","|"+"|".join(["---"]*len(header))+"|"]+["|"+"|".join(map(str,r))+"|" for r in rows])
def distribution_table(source,kinds):
    rows=[]
    for kind in kinds:
        for metric in ("position_mm","rotation_deg","Q"):
            m=source[kind][metric]
            rows.append([kind,metric]+[f'{m[k]:.6f}' for k in ("mean","median","p95","max")])
    return tbl(["模型","指标","均值","中位","P95","最大"],rows)
ugtable=distribution_table(a["unguided"],["mlp","transformer"])
exectable=distribution_table({k:v["execution"] for k,v in a["methods"].items()},["mlp","transformer","structured","learned_mean"])
candtable=distribution_table({k:v["candidate"] for k,v in a["methods"].items()},["mlp","transformer"])
stratatable=tbl(["执行分层","单模型分母","MLP位置均值/mm","T位置均值/mm","T位置更好"],
 [[k,v["n"],f'{v["means"]["mlp"]["p"]:.6f}',f'{v["means"]["transformer"]["p"]:.6f}',v["T_vs_MLP_position_better"]] for k,v in s["strata"].items()])
text=f"""本轮独立 RESULT-TO-CLAIM 审查

日期：2026-10-20。审查对象：`artifacts/v7_1/backbone_pair_smoke_002`。执行角色为未参与实现的独立 paper architect/reviewer。依据本目录 `prompt.md` 和 `experiment-result-to-claim` 技能，直接读取原始文件，并先于结论重算分母和指标；没有复用旧review的判断。

**最终判断：C1 = no，C2 = partial，工程门 = PASS。** 不建议凭此轮替换默认模型，也不建议把扩库作为自动下一阶段。允许完成有界实验报告及限定结论的暂定写作；不启动任何新试验、训练、扩库、pilot、study、strict_multimodal或TEST。

**证据来源与冻结设计**

`run.json` 的 producer_head 为 `4ccba22122c5ad241fe2d701ff907d82d99d41e4`。重新运行只读身份核对：168项源码与当前文件、executed_source.zip一致，28861项输入绑定一致；12个DEV任务与旧冻结任务逐项相同。后续工作区HEAD的其他提交不是本轮运行源码。旧强样条/条件均值的360候选、72执行为身份绑定复用，本轮新增只有两神经主干合计360候选、72执行；逻辑四方法合计720/144不能写成新增。

两个神经主干共享TRAIN拟合codec、中心、102维表示和65维条件，各自从头初始化；相同初始化seed不能解释为相同权重。MLP 208998参数，Transformer 211969参数（约+1.42%）。Transformer的6×17 token是时间基底系数，并非6个真实时间点。主干同时改变注意力、归一化、残差、激活等结构，不能作注意力单因素归因；不能用本轮fresh模型与历史warm模型的差异证明主干效果。

重新加载14222 TRAIN和48 VAL的实际数组、所有行、敏感度G；数据内容hash通过。张量为14270×102、条件14270×65，G为14222×12×102。两个最终progress.pt均6000更新、384000抽样、全部14222行被使用，逐行13–136次；非均匀覆盖在两模型间相同。300段batch/t/noise摘要精确相等并与旧A一致，最终optimizer参数状态步数全为6000，矩非空且有限，journal均COMMITTED。初始与最终权重hash不同、checkpoint重载身份通过。固定6000终点；VAL梯度更新0、不按VAL选checkpoint。损失为v MSE + 0.05教师局部G，新增非线性世界监督事件0。

分别对1/3000/6000的before/after.pt重新求神经梯度并手写重构AdamW，六项全通过；没有optimizer.step。第1步前状态为空、后MLP8/Transformer39个参数状态；3000和6000前后状态、矩和计数真实非空。两模型三组预选实际indices/t/noise精确相同。以上是六次预选更新的直接代数证明，其余更新以最终状态、300段序列与覆盖记录支撑；不声称逐步重放全部12000次训练更新。

14项已有软件测试及M0临时更新只支持软件实现；不是物理证明。001因首次保存before.pt目录不存在、正式更新尚未执行而失败，属于启动故障，不纳入科学比较或新增结果。

**C1：完整无引导比较不支持整体改善**

逐条重建192条神经样本，与保存raw_Z精确相同；每条完整16步DDIM，physics_guidance_calls=0，并非late4末步raw。世界端点统计对应共同prepare后的轨迹；两模型所有coefficient_shrink均1，但仍存在共同压缩处理，不能将下列数值冒充未经prepare的raw世界终点。没有新增传播测量。

每模型48目标×2采样种子=96，端点缺失0、全部有限。两组主要指标全部门合格均0/96、双臂联合位置姿态通过也均0/96。MLP非目标模型门93/96通过、3条geometry拒绝；Transformer95/96通过、1条geometry拒绝。单位置门MLP0/96、T3/96；单姿态门2/96、6/96。因此既不能隐去几何拒绝，也不能将某个单门改善写成总体合格改善。

每条取两臂最坏位置误差和最坏姿态误差；Q=max(位置毫米/5, 姿态度/1)，Q≤1表示两臂联合端点达标，仍需其他模型门才可eligible。P95按全部96条有端点记录计算，未剔除失败：

{ugtable}

Transformer相对MLP：位置配对51更好/45更差、均值差+2.071546mm；姿态72更好/24更差、均值差−0.459769度；Q 59更好/37更差、均值差+0.153433。中心改善与尾部退化并存，尤其Q最大9.051073→17.496519。两个采样种子分别都是0/48合格，Transformer每个种子的位置均值都更差。由此C1原表述判no，confidence=high；这表示本轮不足以支持整体改善，并不等于证明Transformer普遍无效。

建议表述：{C1["suggested_claim_revision"]}

**C2：完整流程有局部精度收益，不能升级为普遍优势**

每模型原DEV4+已用8任务×3采样种子×5候选=180候选，36个预执行选择单元；两新主干均180/180候选eligible、36/36原生执行及原运行独立重放通过。四方法均相同成功率。候选状态全部DONE、执行全部EXECUTION_VERIFIED，没有NO_CANDIDATE/ERROR/缺失端点，未按结果更换或补抽候选。

逐候选核对冻结task_hash、seed公式、candidate_id、预算、16网络调用、late4同27/27/27/29配额、潜变量反馈及同一x_t重算epsilon的分解残差；完整候选池digest和score/candidate_id选择一致。冻结源码在execute前写selection.json，原始selection的selected_id与重算一致，因此选择规则没有使用执行结果。此证据由源码顺序和保存池记录支撑，不将文件时间单独当作证明。

候选完整分布：

{candtable}

Transformer候选均值/中位/P95略好，但最大位置2.562680→3.426416mm、最大姿态0.278217→0.667317度，不能只报告均值。实际执行分布如下，各36条、缺失0；前两模型新增，后两模型历史复用：

{exectable}

对本轮MLP，Transformer执行位置均值减少0.127875mm（约16.24%），P95减少0.274945mm；姿态均值减少0.013346度。36配对位置26好/10差、姿态25好/11差、Q26好/10差；这是有限任务下精度改善的描述性证据。保持段最坏位置均值MLP0.787365/T0.659435mm，最坏值1.301480/1.228094mm；保持段最坏姿态均值0.093676/0.080326度，也满足原门。终点统计与保持门分别核对，未混用。

{stratatable}

三个采样种子以及原DEV4/已用8分层均有执行位置均值下降，说明全体均值收益不是只来自一个采样种子或单一分层。但任务内多候选、重复采样共享同一训练模型和目标，不能作为独立训练重复计算普遍性；本审查没有使用未预注册的显著性检验或伪独立置信区间。无引导测在VAL、引导执行测在DEV，本轮没有同任务关闭引导的DEV消融，不能将两个表的差值归因为引导的独立因果贡献。

相对强样条，Transformer执行平均位置较低但位置仅13/36配对更好，中位数0.654990mm劣于强样条0.238577mm，存在明显任务分布效应。相对条件均值，Transformer平均位置0.659327mm劣于0.366634mm、姿态0.080320度劣于0.043928度，位置仅8/36更好、28/36更差；其P95更好并不能改写为全面胜出。简单基线必须保留。

C2判partial，confidence=medium。原始结果完整，但只有单训练seed和已用DEV；缺少跨训练重复、未用任务、单组件归因证据。建议表述：{C2["suggested_claim_revision"]}

**工程证据、物理门与预算**

本次重新执行只读raw helper通过，360候选、72执行、756000原生+重放步数与ledger相符。随后用独立脚本逐条检查全部72个native_trace.npz（没有创建MjModel或调用仿真）：每条5250控制步、5251状态，qpos81、qvel79、ctrl67，dt=0.002s、10s运动+0.5s保持。保存端点矩阵重新计算位置/姿态，与report吻合；原17→67矩阵及左逆、任务初始17维位置与零初速、参考曲线、限位、低层与规划层速度/加速度、原始力矩及ctrl映射、tracking/manifold、保存clearance、动量和基座运动均通过代数核查。

全72条原始力矩最大额定比例约0.001654744，无饱和；保存原生时刻最小clearance为0.002660063m，均不穿透。实际验收门仍为0m，没有把此样本最小值改为新安全裕量。最大基座平移0.005254934m、每条存在基座运动；源代码保留free joint、零重力、67直接力矩和动态耦合，没有锁基座，执行步循环没有覆盖实际qpos/qvel，亦无外部wrench。完整初始化使用mjSTATE_INTEGRATION，任务/模型/初态ID与保存合同一致。全机几何按冻结显式拓扑排除策略，包括刚性连接/直接相邻关节/明确连续体相邻模块排除；不是所有几何无条件两两相碰测试，也没有基于少量接触点代替距离审计。每执行原报告检查10501个原生/中点状态。

工程PASS仅用于这些实际执行样本和已有离散门。候选MODEL_VALID的reference_torque_feasibility仍是NOT_EVALUATED，不能把180候选或14222训练样本整体升级为EXECUTION_VERIFIED；只有被选中并实际执行的36/模型具有该证据。continuous_time_certified=false、hardware_validated=false，不能写连续时间证书或硬件验证。

预算复核：192无引导+360候选+72执行共624已完成ledger单元，inflight=null、无unresolved；训练journal均COMMITTED。逐原始文件合计耦合29527=384无引导+29143候选，预留上界41256≤42000；候选最大109≤113、积分间隔和单条15秒均在门内。总elapsed={a["budget"]["elapsed_s"]:.3f}s（约45.69分钟）<6h；输出{a["budget"]["output_bytes"]}字节（约{a["budget"]["output_bytes"]/1024**3:.3f}GiB）<20GiB。冻结launcher代码设置OMP/MKL/OPENBLAS=2且torch线程2，本审查也设置2线程；原运行没有CPU亲和性/系统级峰值线程采样记录，2线程判断基于冻结配置与运行代码，不能扩大为操作系统级的硬隔离证明。

主代理完成一次resume返回NO_WORK_ALREADY_COMPLETE。本审查没有调用resume，只读取其前后manifest并重算当前956文件SHA256，三者完全相同；可将其作为已完成handoff证据。审查所有输出仅写本review目录，未修改冻结源码/原始结果、未提交git。

**审查边界与后续路线**

独立重放的原始记录包括逐字段最大误差（全≤1e−10）、控制hash、5250步计数和禁止逐步覆盖实际状态的实现。本次没有再跑重放，目录也没有单独第二条replay原始轨迹。几何距离/中点与质量矩阵没有重新计算；本次核对保存原生数组、report残差、输入绑定及冻结实现。native_trace自身不包含全部初态外力/mocap/userdata字段，完整初态保障来自绑定身份与初始化源码；不能声称仅由trace重新构造了完整积分状态。以上是只读审查的证据范围，不是本轮出现未知工作或工程失败。

本审查先独立重算后对照paired_statistics，全部比较均值/中位/P95/最大及分母吻合；independent_verification仅作为待核对摘要，已由本次自己的只读运行复核。原始只读细节保存在reviewer_raw_audit.json、reviewer_trajectory_algebra.json、reviewer_supplemental.json，脚本也保存于此目录。

最终结论形成后，按主代理要求读取本轮docs/ANALYSIS.md草稿。其MODEL_VALID与目标合格区分、单训练seed、VAL/DEV分布不同与缺少关闭引导DEV消融的表述正确，主要数值与原始证据一致，未发现重大冲突。建议在最终主报告审计范围中补入“未保存单独完整replay轨迹，本轮审查只核原始重放残差、源码和native_trace，没有重跑物理仿真”的边界；这不是撤销原运行已有独立重放结果。

PAPER_CLAIM_AUDIT.json不存在，记录unavailable，下游写作provisional；这不阻塞本轮实验报告，但不能将暂定陈述视为论文到证据的最终审计通过。

不建议替换默认模型：C1主要指标未改善且尾部恶化，C2局部收益尚未超过简单条件均值平均精度，稳定性未证实。不建议因工程通过就继续扩库；当前结果并未检验扩库能解决哪个瓶颈，也没有“为得到优势追加预算”的正当性。当前允许的下一阶段是收窄主张、保存负结果及局部正结果、完成有界分析/版本化记录与暂定稿件表述。

若未来另获授权，可先制定新的冻结假设，再考虑多独立训练种子的等预算重复、任务层统计和尾部来源诊断；若专门研究注意力，另需分离归一化/残差等因素。此处仅列建议，未执行任何补实验或扩大任务范围。工程有效无需无引导合格或相对成功率优势才成立；本轮不因C1=no撤销已经通过的DEV工程结论。
"""
(D/"response.md").write_text(text,encoding="utf-8")
print("Wrote response.md and verdict.json")
