"""Write the bounded reviewer response and trace; no project runtime imports."""
from pathlib import Path
import json,datetime,subprocess,zipfile,hashlib
H=Path(__file__).resolve().parent;R=H.parents[3]
A=json.loads((H/'raw_checks.json').read_text(encoding='utf-8'))
run=json.loads((R/'artifacts/v7_1/fullchain_acceptance_001/run.json').read_text())
def put(name,x):
    (H/name).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n' if isinstance(x,(dict,list)) else x,encoding='utf-8')
provenance=[]
with zipfile.ZipFile(R/'artifacts/v7_1/fullchain_acceptance_001/executed_source.zip') as z:
    for p in A['producer_git_differences']:
        b=subprocess.run(['git','show',run['producer_head']+':'+p],cwd=R,capture_output=True,check=True).stdout;a=z.read(p)
        provenance.append(dict(path=p,line_endings_only=a.replace(b'\r\n',b'\n')==b.replace(b'\r\n',b'\n'),archive_sha256=hashlib.sha256(a).hexdigest(),git_blob_sha256=hashlib.sha256(b).hexdigest()))
put('source_provenance_note.json',dict(producer_commit=run['producer_head'],archive_manifest_all_match=True,raw_git_byte_differences=len(provenance),line_ending_only=sum(x['line_endings_only'] for x in provenance),non_line_ending_differences=[x['path'] for x in provenance if not x['line_endings_only']],differences=provenance,conclusion='executed_source.zip + run.json identity.source_hashes are the actual reproducibility source; producer Git commit alone is insufficient. Existing user dependency is preserved.'))
prompt='''# Exact bounded reviewer assignment

你是本次全新独立 paper architect/reviewer，按 C:/Users/admin/.codex/skills/experiment-result-to-claim/SKILL.md 做 RESULT-TO-CLAIM。工作区 E:/桌面/mujoco_irregular_waypoints_diffusion_adapter_bundle_20260902-main/Hybrid_dual_arm_space_manipulator。仅审新run artifacts/v7_1/fullchain_acceptance_001，冻结源提交370a2041，协议ff_prior_fullchain_acceptance/docs/EXPERIMENT_PLAN.md。当前终态FINISHED_WITH_BLOCKED_ITEMS，worker已退出。目标C1：公开三/五障碍到真实67力矩执行+完整状态重放+五障碍连续动态组合完整工程；C2正向Diffusion收益分开。不要相信summary，读原新6条static native/replay/geometry/metadata，重建门/误差/步数/身份；不运行物理、optimizer或其他新实验。动态12逻辑/8独特历史复用仅核准确源码/模型/task/initial/ref/ctrl绑定、旧report/review既有完整状态证据，不重新做旧物理审计。五障碍6+连续组合6明确BLOCKED分母；组合尚未具体冻结执行任务，不能伪造continuous success。接口软件测试不是物理证明。新125源码/589绑定，Archive+manifest是实际source，git producer可含旧用户uncommitted依赖。核止境/预算(165000新原生含重放步、2线程/6h/20GiB)、无新训练提案数据TEST。保留scope，不把18/30逻辑或3/5场景缩成整体成功。阅读全链route DIFFUSION_GOAL_CURRENT.md/ff_prior_fullchain/docs/ROADMAP.md 了解长期scope，仅报告缺口/可推进工作建议，不自行启动新run或修改目标。只写paper/review-traces/experiment-result-to-claim/20261022_fullchain_acceptance_run01，包括prompt、inputs、raw_checks.py/json、raw_response、verdict、routing和简短findings；其它文件不写，不commit。不要执行job resume，该once由主执行者负责。保存fulltrace；PAPER_CLAIM_AUDIT若无则provisional。向主线程报告真实counts/主张判断/缺口，未来实验建议不构成授权。

# Reviewed inputs

The complete hashed, timestamped list is inputs.json (942 evidence/source files). Main groups: frozen protocol/run identity/selection/binding/source archive; all six new static native_trace/replay_trace/geometry_scans/initial/frozen_task/report/calls files; all 30 unit spec/result records; exact historic dynamic model, codec, task, initial, proposal, prepared, controller, native/replay and historical review bindings; status/budget/software test evidence; long-term route documents. The review does not use run summary counts as its numerical authority. Historical dynamic evidence is checked for identity and existing independent full-state evidence only.

# Method and trace

Independent reviewer role; trace full; pending_external_review=false. raw_checks.py imports no project runtime or MuJoCo and performs only hash, schema and saved-array algebra. Geometry engine distances and extent maxima are source-inspected and reconciled from stored records, not newly queried. PID snapshot and producer Git byte-versus-line-ending differences are saved separately. Only this trace directory is written. No simulator, optimizer, training, experiment, resume, goal change or commit.
'''
put('prompt.md',prompt)
table='\n'.join('| {id} | {method} | {seed} | {p:.6f} | {r:.6f} | {c:.6f} | {g:.6f} |'.format(id=x['id'],method=x['method'],seed=x['seed'],p=max(x['hold_position_mm']),r=max(x['hold_rotation_deg']),c=x['minimum_clearance_mm'],g=x['recorded_max_geometry_step_mm']) for x in A['static'])
response='''# 独立 RESULT-TO-CLAIM 结论

整体 **partial / high**。C1完整主张 **no / high**；其三障碍与原动态组件的局部覆盖得到支持；C2正向稳定Diffusion收益 **no / high（未建立）**。缺少 paper/PAPER_CLAIM_AUDIT.json，任何下游论文主张均为 **provisional — no paper claim audit available**。这是新独立审查，pending_external_review=false。

## 真实分母与身份

从冻结selection和30个单位的spec/result重新建立计数，未拿summary计数作依据。五场景×三组×两配置共30机会，18有合格证据，12 BLOCKED，覆盖3/5场景。每配置9/15逻辑合格、6依赖阻塞；这里的60%是包含阻塞的有限套件证据覆盖比例，不是独立规划成功率估计。

| 场景 | 确定性配置 | Diffusion配置 | 证据性质 |
|---|---:|---:|---|
| 三障碍静态 | 3/3 | 3/3 | 6条新最终接口执行；既定P1公开路径，不是6次新自主规划 |
| 五障碍静态 | 0/3，3 BLOCKED | 0/3，3 BLOCKED | 无合格公开输入路径，未补选/再执行 |
| 原stationary七航点 | 3/3逻辑 | 3/3逻辑 | 确定性1条独特+2共享，Diffusion3条独特，均为历史证据 |
| 原drifting七航点 | 3/3逻辑 | 3/3逻辑 | 确定性1条独特+2共享，Diffusion3条独特，均为历史证据 |
| 五障碍接近→drifting连续组合 | 0/3，3 BLOCKED | 0/3，3 BLOCKED | 缺合法五障碍终态，实际执行任务尚未冻结 |

动态为12逻辑、8独特历史执行、4确定性共享；本轮新增动态物理步0。已核源码/模型/codec/task/initial/proposal/prepared/控制记录及执行来源hash；共享记录的task和完整初态与对应源一致。原报告fullerr均为0、保存native/replay均具836维完整状态schema；旧独立review确已审全部16条历史执行的完整状态，其中本次复用8条DEV。未重做旧物理审计，也不把历史VAL失败改写为成功。

125新冻结源码与589绑定输入全部hash匹配，executed_source.zip中的125项逐一匹配run manifest；静态和动态旧源码归档也与各自manifest及selection一致。Git producer为370a2041a4af0e4615fae259b2e863d4b676e490。相对Git blob的37个字节差异中36仅CRLF/LF，1个为既有v6_lite/hierarchical_qp.py用户依赖内容；实际复现必须使用归档+manifest，不能只checkout producer commit。没有覆盖用户修改。

## 六条新静态原始证据

2,119个保存证据谓词全部通过，0不一致。每条有13,751原生状态、13,750个67维ctrl及raw torque，27.5秒/2ms，836维INTEGRATION；保存独立重放在qpos/qvel/act/time/qacc_warmstart/完整integration上逐元素差均为0。完整初态hash、源参考hash、冻结世界目标、控制hash、17→67映射及每条27,500执行+重放步一致。

世界位姿误差从每条保存ee数组对原冻结双臂目标重算；保持段为27.0–27.5秒，原5mm/1°门没有改变。全运动过程初始远离目标的误差另外保存在raw_checks.json，不把静态保持门错误解释为全时域世界位置≤5mm。

| 单位 | 配置 | 种子 | 双臂保持位置最坏 mm | 姿态最坏 ° | 最小扫描间隙 mm | 记录最大几何步 mm |
|---|---|---:|---:|---:|---:|---:|
'''+table+'''

最坏保持位置3.804815mm、姿态0.246484°；最小扫描间隙0.414746mm；记录最大几何步0.208174mm≤2mm。每条27,501几何扫描对应13,751原生状态和13,750中点，扫描状态/标号/全部minima与原生clearance对齐，标量及平移中点代数一致。原始力矩、关节、参考与实际导数、跟踪、流形、保存动量峰值/阈值公式、完整时间及重放接受门均复核。参考依然是完整323表示，没有改选候选；候选依次2/4/4/2/1/3。

最大实际低层加速度比约0.075922，最大原力矩比0.015339，没有饱和；自由基座有非零平移。tracker源中实际状态由初始reset与mj_step推进，未在步循环裁剪/覆盖qpos/qvel；replay从独立完整初态只应用保存ctrl，不逐步复制实际状态。新验收源码明确将full integration和geometry_step作为门，避免把仅qpos/qvel差为零当完整重放。

审查的边界：未重新运行模拟器或距离引擎。几何距离和最大几何extent依据源代码、保存的全扫描及记录报告；原始记录未保存每个geom的位姿/rbound，因此本审查未独立重算该extent标量。动量交换尺度使用已绑定报告，而保存动量峰值与阈值公式独立重算。这是对原run证据的审查，不是又一次物理试验或连续时间安全证明。

## C1和C2判断

C1可写成：“在固定开发套件中，最终静态执行接口对六条既定公开来源三障碍路径全部通过，原静止/漂移动态组件具备可准确复用的完整合格执行证据；五障碍路径与其连续动态组合仍阻塞。”不能写“完成全链路工程”“30/30成功”“18次新执行”或“泛化到五障碍连续组合”。

连续组合六个task_hash均为null，明确DEPENDENCY_BLOCKED_NOT_FROZEN_FOR_EXECUTION。PhaseSession提供同一live model/data的完整状态与目标连续性检查，37项软件/身份检查有价值，但本run没有一次物理PhaseSession组合轨迹，不能把旧动态内部4.5/25.5秒阶段切换当作五障碍接近→drifting组合。

C2未建立稳定正向优势。静态最终三条Diffusion来源参考的保持误差小于三条结构来源参考，复现了局部精度描述；这是已选择/共同修正后的历史参考经新接口执行，成功率双方3/3，本轮新规划/采样为0，不能把控制/共同修正收益归为裸网络能力。动态既有结果仍是stationary位置退化、drifting部分尾部改善、无引导VAL两法0/4、共同反馈DEV两法6/6逻辑。静态323网络与动态时序网络为两组件，且两任务×三个噪声并非六个独立任务；本轮也没有新因果消融或独立训练重复。C2不作为工程推进前置条件。

## 终态、预算与行动边界

worker终态FINISHED_WITH_BLOCKED_ITEMS，PID119636和launcher PID161364在只读快照中均已不存在。新原生82,500步+重放82,500步=165,000，恰好用完冻结新步预算，inflight=null。worker219.797秒，预算计时216.922秒；末次budget记录953,760,889字节、终态文件合计953,761,239字节（写终态元数据的350字节差），均远低于6h/20GiB。源绑定启动环境为2线程；退出后没有运行时线程计数器证据，不能将配置值当峰值测量。

本run新训练0、新提案0、新TRAIN数据0、TEST未启动。审查新增物理步/几何查询/optimizer更新/resume均0。审查脚本第一次因URDF连续关节无limit而停止，修正了审查脚本处理；一次Git前缀路径诊断也只修正审查侧读取。两者都没有改变实验结果或生产源码，最终纯证据审查通过。

当前应收尾本P4固定套件：主执行者已完成唯一一次终态resume安全核验，保存receipt为PASS/NO_WORK_ALREADY_COMPLETE、133文件前后hash不变；reviewer只读该receipt，没有调用或重复resume。报告/曲线/交付命令和本地提交继续由主执行者完成。长期工程目标继续未完成，不能因这个阶段完成而标complete。

缺口先后关系：公开原五障碍可执行合格路径 → 合格实际终态 → 准确组合任务/模型/初态/目标时序冻结 → 同live状态连续执行及完整重放。原同瓶颈机制机会已经用尽、当前新步预算已耗满；不建议原地改源码、再resume、局部扫参或自动追加训练来追求好看的结论。后续仍可整理失败约束和提出有区别的全局可行性/绕行机制设计，或实现不依赖合格终态的交接接口文档与静态合同检查，但不能把它们升级为成功证据。若未来确需新物理实验，须先作新的具体方案、身份和有界预算决策，并另行独立审查；本建议不构成启动授权，也不放开TEST/扩库。若要增强C2，未来公平任务级对照、共同修正/反馈贡献消融和独立重复必须另立证据，而不是当前工程门。
'''
put('raw_response.md',response);put('response.md',response)
verdict=dict(audit_skill='experiment-result-to-claim',review_role='fresh independent paper architect/reviewer',review_trace=H.name,generated_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),run='artifacts/v7_1/fullchain_acceptance_001',frozen_source_commit=run['producer_head'],claim_supported='partial',verdict='partial',confidence='high',provisional=True,paper_claim_audit='unavailable',provisional_reason='provisional — no paper claim audit available',pending_external_review=False,C1=dict(claim='公开三/五障碍、原静止/漂移及五障碍接近到动态连续组合完整工程',claim_supported='no',component_coverage_supported='partial',complete_claim_accepted=False,confidence='high',local_static_final_interface='yes',dynamic_exact_reuse='yes',five_obstacle='BLOCKED',continuous_combo='DEPENDENCY_BLOCKED_NOT_FROZEN_FOR_EXECUTION'),C2=dict(claim='稳定正向Diffusion贡献',claim_supported='no',confidence='high',local_static_precision='descriptive improvement in fixed previously selected references, not new sampling or causal isolation',dynamic='mixed existing outcomes; no new scientific comparison'),what_results_support=['六条既定公开来源三障碍参考在新最终接口6/6原生67执行及836完整状态保存重放通过。','12动态逻辑/8独特历史执行身份准确，4确定性共享不计独立重复。','30分母中18逻辑有合格证据、12 BLOCKED，覆盖3/5场景。','125归档源码及589输入绑定准确；实际source是archive+manifest。'],what_results_dont_support=['完整五障碍与连续组合工程成功。','18次新执行、30/30通过、六独立动态基线或六次新自主规划。','稳定跨场景Diffusion优势、裸网络成功、未见泛化或连续时间/硬件安全。'],missing_evidence=['原公开五障碍MODEL_VALID且完整执行/重放合格路径。','可合法冻结的真实五障碍终态与组合执行任务。','同live model/data实际连续组合原生控制及完整状态重放。','C2的公平独立任务比较和共同修正/反馈贡献分离。','paper/PAPER_CLAIM_AUDIT.json。'],suggested_claim_revision='固定开发套件中三障碍最终接口和原静止/漂移动态证据有效；五障碍及其连续组合仍阻塞，完整工程目标尚未完成；稳定Diffusion收益未建立。',next_experiments_needed=['本固定run已达新步上限，完成原授权报告/交付；主执行者已完成唯一一次resume核验，不能重复。','未来五障碍可行路径和真实连续组合证据需先另冻具体方案、身份、预算；禁止直接复用本review作为启动授权。','未来如要求C2正向主张，另规划控制/修正归因与独立任务重复；不是当前工程完成的前置条件。'],routing_action='close_current_fixed_suite_preserve_partial_then_explicit_gap_planning_no_automatic_new_run',long_term_goal_complete=False,counts=A['counts'],resources=A['resources'],validation=dict(saved_evidence_checks=2119,saved_evidence_failures=0,source_files=125,bound_inputs=589,reviewed_hashed_files=942,new_physics_steps=0,new_geometry_queries=0,optimizer_steps=0,resume_calls=0),limitations=A['limitations'])
put('verdict.json',verdict)
routing=dict(route='partial',confirm=['six final-interface static executions','exact historical dynamic identity reuse'],supplement_required=['public five-obstacle valid path and actual terminal state','concretely frozen five-obstacle to drifting task','same-state continuous native execution and independent full-state replay'],current_run_terminal=True,current_run_resume_by_reviewer=0,completed_resume_owner='main executor, once only',no_automatic_new_experiment=True,new_experiment_suggestion_is_authorization=False,long_term_goal_complete=False,scientific_advantage_required_for_engineering=False,paper_writing='provisional narrow claims only',forbidden_shortcuts=['remove 12 blocked opportunities','substitute three-obstacle combo','count software interface tests as physical continuity','reuse old failed five-obstacle worker','unbounded local retry or training','TEST or expansion'])
routing['C1_full_claim_supported']='no';routing['C1_component_coverage_supported']='partial';routing['completed_resume_record']='PASS / NO_WORK_ALREADY_COMPLETE / 133 files unchanged; read-only receipt inspection';routing['repeat_resume_allowed']=False
put('routing.json',routing)
put('routing.md','''# Routing

partial / high；C1完整主张未通过，C2稳定正向收益未建立。先完成当前固定套件的报告/可视化/交付及本地提交。主执行者唯一一次完成resume已保存PASS/NO_WORK_ALREADY_COMPLETE、133文件不变receipt；reviewer只读receipt，未resume、未commit。

30分母保留：18有证据（6新静态+12历史动态逻辑，后者8独特）、12 BLOCKED；3/5场景。组合实际任务未冻结，软件接口检查不能替代连续物理证据。

长期工程目标仍未完成。缺口链是五障碍合格公开路径→实际合格终态→准确组合合同→同live状态连续执行/完整重放。当前新步预算和原同瓶颈机会已用完；只整理缺口/不同机制方案，不自动新增run、局部扫参、训练、扩库或TEST。任何未来实验建议不构成授权。论文写作仅可使用provisional限定主张。
''')
put('findings.md','''# Research Findings

- fullchain_acceptance_001：overall partial，高置信；完整C1 no、局部组件覆盖支持；C2稳定正向收益no，高置信。完整工程未通过。
- 2119保存证据检查0不一致。6新三障碍全部通过；836维保存独立重放差0；82500原生+82500重放=165000。
- 12动态逻辑来自8独特历史执行，4共享。全30机会中18合格证据、12阻塞；每配置9/15；覆盖3/5场景。
- 五障碍合格公开路径缺失，6连续组合任务尚未冻结。既有PhaseSession软件检查不提供物理连续组合证明。
- 125源码/589输入匹配。Git producer包含1项非换行旧用户依赖差异；复现用archive+manifest。
- 当前固定run收尾，不自动新增物理/训练/TEST；不缩小长期任务范围。缺PAPER_CLAIM_AUDIT，论文结论provisional。
''')
put('handoff.md','''# Reviewer handoff

已按全新独立paper architect/reviewer角色完成新run的RESULT-TO-CLAIM。见raw_response.md、verdict.json、routing.json与raw_checks.py/json。inputs.json保存942项输入hash/时间戳；pid_check.json证实记录PID均退出；source_provenance_note.json区分36换行差异与1旧用户依赖差异。

只写本trace目录；没有运行模拟器、碰撞查询、训练、optimizer、旧run审计、resume、目标修改或commit。动态仅核准确身份/原报告和既有完整状态review。6条新静态保存数组全部读取，2119项核对通过。

raw_checks.py第一次因连续关节URDF不含limit而退出，在审查脚本中修正后通过；Git producer比较的错误前缀已修正，归档本身始终匹配。没有修改实验或生产源码。几何extent和交换动量尺度无法从独立保存的每geom/raw mass数据重新生成，审查边界已明确，未伪称新物理复审。

主执行者已完成一次终态resume安全核验，保存receipt为PASS/NO_WORK_ALREADY_COMPLETE、133文件hash不变；reviewer只读receipt，未执行resume。报告/曲线/版本记录和本地提交由主执行者负责；后续不可重复resume。完整工程仍有五障碍与连续组合缺口；本阶段终态不等于长期目标complete，未来实验建议不授权新run。
''')
print(json.dumps(dict(trace=str(H),verdict='partial/high',C1='no for complete claim, partial component coverage',C2='no/high',files=len(list(H.iterdir()))),ensure_ascii=False))
