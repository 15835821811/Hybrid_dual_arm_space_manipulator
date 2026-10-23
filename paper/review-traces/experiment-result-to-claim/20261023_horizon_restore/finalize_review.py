"""Resolve the one investigated provenance difference and write the independent verdict.
This script reads evidence and source only. It never launches a model, worker or replay.
"""
from pathlib import Path
import ast, difflib, hashlib, json, subprocess, zipfile
from datetime import datetime, timezone

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3];RUN=ROOT/'artifacts/v7_1/horizon_restore_001'
def js(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
    return h.hexdigest()
def put(name,v):(HERE/name).write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
run=js(RUN/'run.json');tests=js(ROOT/'ff_prior_horizon_restore/evidence/software_tests.json');status=js(RUN/'status.json')
name='v6_lite/hierarchical_qp.py';recorded=run['identity']['source_hashes'][name]
with zipfile.ZipFile(RUN/'executed_source.zip') as z:archived=z.read(name)
head=subprocess.run(['git','show',run['producer_head']+':'+name],cwd=ROOT,capture_output=True,check=True).stdout
a=ast.parse(head.decode('utf-8-sig'));b=ast.parse(archived.decode('utf-8-sig'))
def strip(tree):
    cls=next(x for x in tree.body if isinstance(x,ast.ClassDef) and x.name=='HierarchicalVelocityQP')
    pos=next(i for i,x in enumerate(cls.body) if isinstance(x,ast.FunctionDef) and x.name=='solve')
    changed=cls.body[pos];cls.body[pos]=ast.Pass();return ast.dump(tree,include_attributes=False),changed
ta,ma=strip(a);tb,mb=strip(b)
refs=[]
for path in run['identity']['source_hashes']:
    if not path.endswith('.py'):continue
    for i,line in enumerate((ROOT/path).read_text(encoding='utf-8-sig').splitlines(),1):
        if 'HierarchicalVelocityQP' in line or 'posture_reference_' in line:refs.append({'path':path,'line':i,'text':line.strip()})
safe=(ta==tb and tests['source_hashes'][name]==recorded==hashlib.sha256(archived).hexdigest()==sha(ROOT/name) and tests['timestamp']<status['started_at'] and all(x['path'] in ('v6_lite/__init__.py',name) for x in refs))
if not safe:raise RuntimeError('Provenance resolution conditions changed; do not approve')
patch=''.join(difflib.unified_diff(head.decode('utf-8-sig').splitlines(keepends=True),archived.decode('utf-8-sig').splitlines(keepends=True),fromfile='producer_HEAD/'+name,tofile='executed_source.zip/'+name))
(HERE/'production_head_difference.patch').write_text(patch,encoding='utf-8')
resolution={'id':'D1','classification':'RESOLVED_PROVENANCE_DESCRIPTION_DIFFERENCE','path':name,'producer_head':run['producer_head'],'execution_sha256':recorded,'git_blob_sha256':hashlib.sha256(head).hexdigest(),'git_head_equals_archive':False,'normalized_AST_equal_outside_method':True,'changed_method':'HierarchicalVelocityQP.solve','changed_method_calls_in_frozen_source':'none; class appears only in its definition and v6_lite package exports','references':refs,'software_tests_timestamp':tests['timestamp'],'run_started_at':status['started_at'],'pre_run_tests_hash_equals_run_hash_equals_archive_equals_current':True,'resolution':'The frozen protocol identifies execution source jointly by source_hashes and executed_source.zip. The producer HEAD alone is insufficient: one pre-existing working-tree change was included in the pre-run tested immutable hash set. Preserve this distinction in final reporting. The changed optional posture-control method is not invoked by this run, which uses ff_prior_static_goal_seed.tracker.command/track. No frozen-source mutation or unexplained physical evidence inconsistency was found.','blocks_dynamic_handoff':False}
put('source_provenance.json',resolution)
d=js(HERE/'saved_checks.json');target='source/production_commit_normalized/'+name
original=[x for x in d['all_checks'] if x['name']==target]
assert len(original)==1 and original[0]['passed'] is False
original[0]['resolved']=True;original[0]['resolution_id']='D1';original[0]['resolution']='See source_provenance.json and production_head_difference.patch; producer HEAD differs, frozen executed identity is intact.'
d['resolved_differences']=[resolution];d['unresolved_failures']=[x for x in d['all_checks'] if x['passed'] is False and not x.get('resolved')]
d['checks_failed_but_explained']=1;d['status']='PASS_WITH_EXPLICIT_EVIDENCE_LIMITS_AND_RESOLVED_PROVENANCE_DIFFERENCE' if not d['unresolved_failures'] else 'FAIL_REQUIRES_RESOLUTION'
d['generated_utc']=datetime.now(timezone.utc).isoformat();d['review_tool_corrections']=['Reviewer inventory initially assumed every read NPZ was under run root; corrected path labels for the newly frozen prefix.','Reviewer URDF parser initially assumed every joint had a limit element; corrected continuous-joint metadata handling. No production data or code changed; final full checks completed.']
put('saved_checks.json',d)
assert not d['unresolved_failures']
shared={'confidence':'high','paper_claim_audit':'unavailable','paper_writing_status':'provisional — no paper claim audit available','pending_external_review':False}
verdict={'reviewer_role':'fresh independent paper architect/reviewer','run':'horizon_restore_001','scope':'one APP_DEV candidate; original capture_5_obstacles; no old raw reanalysis; no new physics','task_hash':run['identity']['protocol']['task_hash'],'claims':{
 'C1':dict(shared,claim_supported='yes',what_results_support='This single frozen full-horizon 323-variable restoration system produced one MODEL_VALID candidate, executed 13750 native steps of 67-channel torque control from the original complete initial state through 27 s motion plus 0.5 s hold, and independently replayed all 13751 × 836 saved integration states with exact zero differences under the original gates.',what_results_dont_support='No bare-network obstacle avoidance, relative advantage, generalization, continuous-time collision certificate, dynamic success, moving physical satellite or hardware success. Historical neural proposal 20264627 is an unqualified obstacle-unconditioned initialization, not the qualified planner.',missing_evidence='For a stronger independent reconstruction claim, raw mass blocks/exchange arrays, compiled geometry radii and all sparse numerical solve histories are absent. These source-verification boundaries are explicit in saved_checks.json; they do not falsify or block this bounded observed-execution claim.',suggested_claim_revision='限于原五障碍单个 APP_DEV 工程单元：完整323恢复系统经原模型认证、原67力矩执行门和独立836全状态重放通过；不能归因为网络单独通过。',next_experiments_needed=[],next_action='confirm this bounded static engineering claim, then freeze the separately authorized dynamic contract from this run’s exact saved full terminal state and remaining original budget; do not add static experiments'),
 'Caux':dict(shared,claim_supported='yes',what_results_support='The frozen source-hash/archive identity, all 233 bound-input hashes, 323-variable linear motion constraints, all 8 finite-difference matrices and 32 diagnostic observations, 12 backtracking trials, acceptance/radius/stop rules, accounting, original execution gates and media provenance agree with the frozen protocol within the documented saved-evidence boundary.',what_results_dont_support='Cannot call all SLSQP solves converged (first six hit 180 iterations), cannot call diagnostic half-step/sinusoidal observations a passed gradient safety gate, cannot call HEAD bfc4ea9 alone an exact clean-tree source snapshot, and cannot claim every physical quantity was independently reconstructed from saved arrays.',missing_evidence='Full SLSQP callback histories, earlier dense-grid trajectory arrays, per-column numerical residual arrays and exact per-pair/pruning geometry logs are not retained. Immutable executed source enforces the relevant checks. One HEAD/archive difference is fully explained in source_provenance.json and remains visibly recorded.',suggested_claim_revision='本轮遵循运行前冻结的142文件源码hash与executed_source.zip及233绑定输入；producer_head=bfc4ea9是Git基点，包含一个已归档冻结的原有hierarchical_qp工作树差异。全部可保存重算域完成全量核对，源码核验边界单列。',next_experiments_needed=[],next_action='retain source provenance correction and evidence limitations in the analysis/receipt; hand off actual terminal state without changing this frozen run')},
 'routing':{'action':'CONFIRM_BOUNDED_STATIC_AND_FREEZE_AUTHORIZED_DYNAMIC_CONTRACT','dynamic_handoff_blocked':False,'unresolved_evidence_inconsistencies':[],'dynamic_has_passed':False,'original_seven_waypoints_preserved':True,'must_use_saved_terminal_integration_state':True,'terminal_state_digest':d['domains']['execution']['final_state_digest'],'new_static_experiments_required':False},'checks':{'total':d['checks_total'],'passed':d['checks_passed'],'explained_source_HEAD_difference':1,'unresolved':0},'assurance_limits':d['limitations']}
put('verdict.json',verdict)
text='''# 独立 result-to-claim 审查：horizon_restore_001

C1 = **yes / high confidence**；Caux = **yes / high confidence**。本裁决仅覆盖原 capture_5_obstacles 的一个 APP_DEV 工程单元。未发现阻止从保存真实末态冻结后续动态合同的未决证据不一致。动态本身尚未通过，本文不预判其结果。没有 paper/PAPER_CLAIM_AUDIT.json，因此论文写作层面的裁决标为 **provisional — no paper claim audit available**；本次独立委派审查已实际完成，并非 pending external review。

审查全程没有新耦合、原生步、物理重放、优化求解、神经采样、训练、TEST 或 resume，也没有修改生产代码。新 run 的36个 NPZ全部完整读取，另读取本轮冻结的prefix.npz用于初猜来源与原初态代数核对；旧输入仅做233项hash验证和必要metadata检查，没有旧raw重分析。

## 证据结论

- 142个执行归档文件与当前文件、run.source_hashes、运行前software_tests.source_hashes完全一致；233个绑定输入hash全部一致。全部127个核心文件与唯一完成resume前后清单相同，458,379,716 bytes。唯一完成resume返回NO_WORK_ALREADY_COMPLETE，未增加工作。
- 545×17初猜拟合采样、1923×323线性矩阵及上下界由原URDF/17→67映射/样条导数控制多边形重建一致；35个返回或尝试参考的运动约束检查通过。拟合是未合格初猜，未把旧模型前缀末态注入真实执行。
- 13次稠密评估、53,261个保存状态、21,253条4mm缓冲记录、3,120个selected和18,133个omitted索引全部核对；选取规则、相位身份、逐相位最小值、完整遗漏集合及缓存key一致。
- 8轮×323列=2,584次差分，651,168个Jacobian元素全部由原始扰动输出重算一致。50,442,264个保存扰动状态元素全部读取、有限性检查，所有活跃/终态的67关节分量与相应扰动样条代数一致。未声称重新传播了这些物理状态。
- 8次稀疏/稠密匹配均在第一档通过；实际非均匀区间依次704、719、716、718、719、717、719、718。32个半步/正弦观察身份及输出一致。它们没有冻结的通过阈值，不能写成“梯度全部通过”或安全证书。
- 共12个回溯试验：前两轮各拒绝1、1/2，接受1/4；其余六轮接受1。8轮实际/预测merit、比值、accepted/next_Z、信赖域更新和停止全部重算一致。前六轮SLSQP达到180迭代上限，后两轮23/19次成功结束；协议允许返回已找到的最佳可行步，再由独立稠密实际下降决定接受。不能把八轮都写成SLSQP收敛。
- 原认证保存256/512/1024/2048/4096全部网格，共7,941状态和7,684几何扫描。最终预测与最后恢复评估的qpos/EE/base/phase逐元素一致。最终模型最小间隙0.00021727520651815269m，报告全机几何步0.001413403096111895m，原5mm/1°双世界目标门通过，MODEL_VALID=1/1。
- 原生执行13750步，13751状态；独立重放相同13750控制。全13751×836=11,495,836个integration元素差为0，另外qpos/qvel/act/time/qacc_warmstart/ctrl全域差亦为0。native第一完整状态与原initial_state及冻结prefix根完整836状态精确相同，末态与重放精确一致。没有用模型前缀状态替换真实起点。
- 27,501全机几何观察全部核对保存qpos、原生/中点索引和最小值；最小间隙0.00023385706019798946m。最大几何步报告值0.0002143638062307468m，低于原2mm门。这里只主张冻结离散几何门，不能上升到连续时间证明。
- 全251个hold状态最大位置误差分别3.5849871589425445e-05 / 2.6383432027438348e-05m；最大姿态误差2.075538873632144e-05 / 2.0966763691776774e-05rad。全部67通道raw torque限额比例最大0.017267387511025134，无饱和。实际planner速度/加速度限额比最大0.15875026720988486 / 0.08271430244751604。
- 计数独立重建为52次稠密+2624次稀疏+5次认证=2681次耦合；99840+1879440+7936=1,987,216个预留模型区间。执行13750+重放13750=27500原生步。worker总时468.375s；ledger单元442.3247557999566s是内部阶段计时，不是总worker时间，二者不矛盾。未把模型区间当原生步。
- 24项媒体清单hash一致；两视频各420帧、960×720、15fps全部解码；交互420选定状态的时间、双臂SE(3)、目标SE(3)、轨迹、五障碍及全部13751行CSV逐元素/数值一致。播放28s代表保存物理27.5s，不是新增物理。末端轴按姿态矩阵列、RGB=XYZ；目标与末端标签/线型及末端轨迹均有对应代码。视频像素的UI观感由父执行者另行检查，此处未把代码/解码检查夸称逐像素物理重建。

## 必须保留的来源说明

Git producer_head=bfc4ea9c436a32bb5fc9aa49db4e66857e11283f不是一个完全clean-tree执行快照。v6_lite/hierarchical_qp.py的归档与该Git blob不同，即使统一换行仍有差异。实际执行SHA256为479da0e42cf9c81496760e28325a3f9d12545b00256c3f64c8a069100b2d2003；它已经被运行前software_tests、run.source_hashes和executed_source.zip共同固定，当前字节也一致。

独立AST检查显示差异仅在HierarchicalVelocityQP.solve：增加可选posture_reference_q/dq及对应分支；模块其余AST相同。冻结源码集合中该类仅在定义和v6_lite包导出出现，没有本轮调用；本轮采用ff_prior_static_goal_seed.tracker的67力矩控制。故这是一个已解释的Git身份描述差异，并非运行期间源码漂移。完整差异保存在production_head_difference.patch，解释保存在source_provenance.json。最终报告应写“Git基点+142实际执行源码hash及归档”，不能只写“执行源码等于bfc4ea9”。原检查的false结果保留，并明确resolved，未伪改为true。

## 15门的可重算范围

| 执行门 | 独立核对依据与边界 |
|---|---|
| identity | 冻结task/初态/场景/pair-policy/reference/control、142源码及233输入hash；原初态836全值一致 |
| finite | 所有保存NPZ完整数组有限性 |
| dual_goal_hold | 全251 hold状态双臂保存SE(3)的位置/SO(3)误差重算 |
| whole_body_clearance | 全27501观察的保存minima/pair/qpos核对；几何距离求值本身依冻结源码，未新调用碰撞 |
| joint_limits | 全13751×17和×67状态、参考样条及原URDF范围 |
| derivative_limits | 全实际速度、13750差分加速度、17/67映射及参考导数 |
| raw_torque_limits | 全13750×67 raw/ctrl及原限幅门 |
| tracking | 全13751×17实际与解析参考差 |
| manifold | 全13751×67映射残差 |
| momentum | 全13751×6保存动量与峰值、门公式；未保存mass块/逐时刻exchange数组，分母来源依源码 |
| independent_replay | 836全状态及全部额外保存字段零差；独立初态+ctrl重放流程依归档源码 |
| complete_native_time | 每个0.002s间隔和完整27.5s、13750控制/13751状态 |
| no_external_wrench | 全状态中79广义力+75×6实体wrench字段为0，源码每步亦检查 |
| no_hidden_reset | 完整状态/时间/控制对齐、独立重放、归档tracker循环无qpos/qvel覆写；不声称有外部反篡改进程监控 |
| geometry_step | 报告0.0002143638062307468m及归档公式；所有中点qpos由纯四元数代数重建；未保存geom轨迹/rbound，无法独立重算该物理数值 |

更强的“所有物理量都从原始数组独立重建”主张不成立：早期稠密网格轨迹、逐扰动数值残差/质量矩阵、精确逐对距离/剪枝数组、momentum exchange全序列、完整SLSQP callback历史并未保存。当前主张只要求这一个冻结系统按原门得到保存的模型/执行/重放证据；以上缺口已完整单列为核验边界，不构成发现门失败的证据。本审查没有补跑物理去填缺口。

## 主张范围与路由

历史网络提案种子20264627（历史组20264623）、16步采样、102+零221、65维无障碍条件仅是准确未合格初猜来源。本轮0新采样/训练/扩库/TEST。系统通过不能归给裸网络；没有优势、泛化、动态通过、实体卫星运动或硬件成功证据。保存原生物理目标卫星pose全程恒定、velocity为0；后续不能暗注速度或把虚拟目标移动写成实体卫星运动。

下一步是用户已授权范围内从本轮保存真实完整末态与剩余账本冻结动态合同，保留原七个不规则航点。完整末态time=27.499999999995637s，836向量digest=ebfd85858591be85420250b54fbeaf9350cfac15979d712dc60dd4fb69dd618b；动态阶段必须绑定native_trace.npz最后一行及其完整模型/物理身份。原预留动态额度为2000耦合、1200s、27000原生含重放、2GiB；静态未用额度不自动授权额外候选。无需追加静态试验、其他种子、训练或优势对照。新动态场景/目标解释须由其独立合同明确，不能反向改写本轮原五障碍结论。

最终2249项检查：2248通过，1项Git HEAD差异已完整解释，0项未决失败。完整域尺寸、差值、逐轮记录、八类证据限制及15门映射见saved_checks.json；输入paths/hashes/timestamps见inputs.json。
'''
(HERE/'response.md').write_text(text,encoding='utf-8')
(HERE/'handoff.md').write_text('''# 独立审查交接

已完成本次唯一新run的全量保存证据检查，C1=yes、Caux=yes，confidence=high；论文级provisional（无paper claim audit）。0新物理/耦合/重放/优化/resume；未修改生产源码。

2249检查中2248通过、1个producer HEAD与实际归档源码差异已解释（source_provenance.json），0未决失败。实际冻结身份以142实际源码hash+executed_source.zip及233输入绑定为准。请在findings.md、中文分析和统一收据中同步此精确表述，勿写成全部执行源码等于纯Git提交。

可以进入已授权的动态合同冻结：必须绑定本轮native_trace.npz完整836真实末态及其time/模型/初态/物理身份，保留原七个不规则航点，按剩余账本；不要追加静态实验。动态尚未执行通过。实体卫星当前零速度，任何实体运动必须有单独物理一致起源，虚拟目标应如实标注。

未保存的物理中间量与源码核验边界详见response.md及saved_checks.json。没有把软件测试、半步/正弦观察或SLSQP迭代上限当成额外物理通过证书。父执行者负责将独立裁决记入项目findings/tracker并纳入最终交付。
''',encoding='utf-8')
inp=js(HERE/'inputs.json');table={v['path']:v for v in inp['files']}
extras=[HERE/'finalize_review.py',HERE/'source_provenance.json',HERE/'production_head_difference.patch',Path('C:/Users/admin/.codex/skills/experiment-result-to-claim/SKILL.md'),Path('C:/Users/admin/.codex/skills/shared-references/review-tracing.md')]
manual=['ff_prior_static_goal_seed/validation.py','ff_prior_static_goal_seed/physical_audit.py','ff_prior_static_goal_seed/tracker.py','ff_prior/model_gates.py','ff_prior_feasible/validation.py','ff_prior_fixed/correction.py','ff_prior/spline.py','ff_prior/proposal_generator.py','ff_prior/initial_state.py','ff_prior/storage.py','ff_prior/contracts.py','ff_prior/geometry.py','ff_prior/robot_adapter.py','model_test/robot_model_spec_v5.py','model_test/angle_convention.py','ff_prior_contact_steer/tree.py','v6_lite/__init__.py','v6_lite/hierarchical_qp.py','v6_lite/continuum_model_spec.py','ff_prior_horizon_restore/job.py','ff_prior_horizon_restore/job.ps1']
for p in extras+[ROOT/x for x in manual]:
    p=p.resolve();st=p.stat();roles=set(table.get(str(p),{}).get('roles',[]));roles.add('reviewed_source_or_resolution')
    table[str(p)]={'path':str(p),'sha256':sha(p),'bytes':st.st_size,'mtime_utc':datetime.fromtimestamp(st.st_mtime,timezone.utc).isoformat(),'roles':sorted(roles)}
inp['files']=list(table.values());inp['generated_utc']=datetime.now(timezone.utc).isoformat();put('inputs.json',inp)
print(json.dumps({'C1':'yes','Caux':'yes','checks':d['checks_total'],'passed':d['checks_passed'],'resolved':1,'unresolved':len(d['unresolved_failures']),'dynamic_handoff_blocked':False,'input_manifest_files':len(inp['files'])},ensure_ascii=False))
