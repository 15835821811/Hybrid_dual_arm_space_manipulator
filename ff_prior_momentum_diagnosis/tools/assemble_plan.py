"""Assemble read-only diagnosis, repair specification and exact budget."""
from pathlib import Path
from datetime import datetime
from decimal import Decimal
import hashlib, json, shutil, subprocess
ROOT=Path(__file__).resolve().parents[2]
PKG=ROOT/'ff_prior_momentum_diagnosis'
STAMP=datetime.now().strftime('%Y%m%d_%H%M%S')
def read(p):return json.loads((ROOT/p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def put(p,v):
    q=PKG/p;q.parent.mkdir(parents=True,exist_ok=True)
    q.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def doc(name,text):
    q=PKG/'docs';q.mkdir(exist_ok=True)
    p=q/f'{name}_{STAMP}.md';p.write_text(text,encoding='utf-8')
    shutil.copyfile(p,q/f'{name}.md')
def main():
    d=read('ff_prior_momentum_diagnosis/evidence/diagnosis.json')
    scope=read('ff_prior_momentum_diagnosis/evidence/scope.json')
    prior=read('ff_prior_satellite_origin/planning/remaining_budget.json')['remaining_after_reservation']
    balance=dict(prior);balance['wall_s']=float(Decimal(str(prior['wall_s']))-Decimal(1800));balance['bytes']-=64*1024**2
    phases=[dict(id='A',start_s=0,end_s=27.5,execution=13750,replay=13750),
            dict(id='B',start_s=27.5,end_s=54.5,execution=13500,replay=13500),
            dict(id='C',start_s=54.5,end_s=83.5,execution=14500,replay=14500)]
    wall=dict(implementation_software_freeze=1800,model_qualification=600,native_execution_replay=3600,
              body_geometry_diagnostics=900,new_unit_staged_independent_review=1800,media=1500,completed_resume_closeout=600)
    storage=[
      ('native_replay_and_full_frequency_diagnostics',41753,32000),
      ('all_control_sample_pair_lower_exact_status',4176*3406,18),
      ('active_constraint_rows_and_selection_reasons',600000,512),
      ('20ms_body_obstacle_aggregate_display_diagnostics',4176*7*71,128),
      ('native_geometry_refinement_records',100000,800),
      ('complete_model_requalification',512,1024**2),
      ('two_videos_3d_curves_exports',256,1024**2),
      ('source_contract_review_metadata',64,1024**2)]
    storage=[dict(name=n,units=u,bytes_per_unit=b,bytes=u*b) for n,u,b in storage]
    caps=dict(coupled=40,reserved_model_intervals=200000,native_including_replay=83500,wall_s=sum(wall.values()),bytes=3*1024**3,candidates=1,retries=0)
    short={k:max(0,caps[k]-balance[k]) for k in ('coupled','native_including_replay','bytes')}
    short['wall_s']=float(Decimal(caps['wall_s'])-Decimal(str(balance['wall_s'])))
    request=dict(coupled=0,native_including_replay=50000,wall_s=4500,bytes=int(2.5*1024**3))
    left={k:float(Decimal(str(balance[k]))+Decimal(str(request[k]))-Decimal(str(caps[k]))) if k=='wall_s'
          else balance[k]+request[k]-caps[k] for k in request}
    budget=dict(status='PROPOSED_NOT_APPROVED_NOT_FROZEN_NO_PHYSICS',
      basis='ff_prior_satellite_origin/planning/remaining_budget.json',conservative_before_diagnosis=prior,
      diagnosis_reservation=scope['read_only_cap'],conservative_after_diagnosis=balance,
      proposed_unit_caps=caps,exact_additional_required=short,rounded_additional_requested=request,
      remaining_if_extension_approved_and_full_caps_reserved=left,phases=phases,wall_caps_s=wall,
      storage_envelope=storage,storage_envelope_bytes=sum(x['bytes'] for x in storage),
      storage_headroom_bytes=caps['bytes']-sum(x['bytes'] for x in storage),
      rk4_internal_force_stages=334000,extra_observation_forward_calls_cap=1000000,
      native_geometry_states_cap=100000,geometry_distance_query_loose_cap=1100000000,
      query_bound='3406 legal pairs per scan times 200000 model intervals +100000 native/refinement states + twice 4176 diagnostic ticks, below 1.1e9; time/bytes guards stop first. Not expected runtime.',
      source_archive='Source and contract text archived; shared existing mesh/model inputs bound by hash without large duplicate files.',
      recording_guard='Count uncompressed envelopes and actual variable-length JSON bytes before physics; stop rather than discard rows or reduce checks.',
      original_global_caps=dict(coupled=80000,native=312500,wall_s=43200,bytes=40*1024**3),
      proposed_global_caps_if_approved=dict(coupled=80000,native=362500,wall_s=47700,bytes=int(42.5*1024**3)),
      original_D3_caps=dict(coupled=60000,native=165000,wall_s=21540,bytes=20*1024**3),
      proposed_D3_caps_if_approved=dict(coupled=60000,native=215000,wall_s=26040,bytes=int(22.5*1024**3)),
      old_reservations_reclaimed=False,ceilings_changed_now=False,worker_started=False)
    assert caps['wall_s']==10800 and short['native_including_replay']==49160
    assert short['wall_s']==4463.359 and short['bytes']==2600156150
    assert budget['storage_headroom_bytes']>0 and min(left.values())>=0
    put('evidence/budget_derivation.json',budget)
    contract=dict(status='ORIGINAL_GATE_SPECIFIED_NOT_CHANGED',scope='robot only, satellite excluded',
      base_dofs=d['base_dofs'],group_slices=[[0,3],[3,6]],
      momentum='p_b=M(q)[base_dofs,:] @ qvel',
      exchange='e_b=M(q)[base_dofs,actuated_joint_dofs] @ qvel[actuated_joint_dofs]',
      linear_frame='world free-joint translation',angular_frame='local base free-joint rotation',
      units=['kg*m/s','kg*m^2/s'],metric='Euclidean norm; maximum over EVERY phase native frame including initial',
      threshold='atol + relative_tolerance * maximum exchange norm in SAME phase',
      dynamic_atol=1e-5,dynamic_relative_tolerance=.002,
      static_threshold='unchanged original bound physical_contract acceptance values',
      windows=[dict(phase=p['id'],start_s=p['start_s'],end_s=p['end_s'],frames=p['execution']+1) for p in phases],
      no_initial_subtraction=True,no_prior_or_whole_chain_peak_reuse=True,no_artificial_exchange_increase=True,
      world_diagnostic='P=p_b[:3]; H_world_origin=R_base@p_b[3:]+x_base cross P',
      world_diagnostic_replaces_gate=False,full_integration_dimension=836,replay_atol=1e-10,
      original_failure=dict(initial_norm=d['phases'][2]['initial_norm'],threshold=d['phases'][2]['original_dynamic_style_threshold'],failed_frames=14501),
      sources=['ff_prior_balanced_restore/physical_audit.py','ff_prior_balanced_brake/evaluation.py','ff_prior_satellite_origin/evaluation.py'])
    put('evidence/momentum_contract.json',contract)
    inputs=['ff_prior_balanced_restore/evidence/initial_state.npz',
      'artifacts/v7_1/balanced_restore_001/units/u00/validated/validation.json',
      'ff_prior_balanced_restore/evidence/selection.json','ff_prior_balanced_scene/evidence/layout.json',
      'ff_prior_balanced_brake/evidence/contract.json','ff_prior_satellite_origin/evidence/contract.json']
    sources=['ff_prior/robot_adapter.py','ff_prior/initial_state.py','ff_prior_balanced_restore/tracker.py',
      'ff_prior_balanced_restore/physical_audit.py','ff_prior_static_goal_seed/validation.py',
      'ff_prior_balanced_brake/environment.py','ff_prior_balanced_brake/evaluation.py',
      'ff_prior_satellite_origin/environment.py','ff_prior_satellite_origin/evaluation.py',
      'ff_prior_satellite_origin/force.py','ff_prior_satellite_origin/ledger.py']
    val=read(inputs[1])
    spec=dict(status='SPEC_READY_BUDGET_EXTENSION_PENDING',unit='momentum_rebuild_001',module='ff_prior_momentum_rebuild.job',
      production_implemented=False,source_base_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
      method='Native MuJoCo3.3.2 RK4 at unchanged2ms, from original full numeric initial state',
      old_integrator='mjINT_IMPLICITFAST',new_integrator='mjINT_RK4',allowed_model_difference=['opt.integrator'],
      physical_parameter_or_gate_changes=[],proposal_source_input_hashes={p:sha(ROOT/p) for p in inputs+sources},
      static_reference=dict(source=inputs[1],field='Z',shape=[19,17],reference_hash=val['reference_hash'],optimization=False,new_model_qualification_required=True),
      new_network_samples=0,new_seeds=0,candidate_count=1,
      initial='original836 numeric vector, explicitly bound to new identity after equal physical arrays proof',
      forbidden_initial='old54.5s terminal must not be corrected/injected/relabeled as a new chain',
      actual_initializations=1,actual_phase_resets=0,
      handoff='One live actual MjData; phase controller constructors borrow data. No reset/setState/qpos/qvel/time writes after initial load.',
      control_memory='reference_q/reference_qd and filter memory inherited from new actual predecessor, not old archived terminal',
      references='Original public world goals, seven exact waypoints/orientations/durations, C2 bypass and final holds unchanged; no error-based goal reanchoring.',
      controls='Same feedforward/67torque controller and discrete braking filter; native2ms zero-order held input, no planner callback inside RK4 stages',
      force='same world+Z COM0.02*(1-cos(pi*t))N at2ms midpoint for2s, thenzero27s; robot external forcezero',
      phases=phases,replay='One independent replay MjData, initialized once at0; saved controls/wrenches only, no actual intermediate state copied',
      stage_release='A gates+replay+necessary independent review beforeB; B beforeC; both actual and replay objects wait without simulation advancement. One new staged review, no old review repeated.',
      recording='all native states and original geometry refinements; all active/omitted control pair IDs/distances/reasons; 20ms body distance aggregates for display never replace full physics gates',
      stop_on=['any physics gate','any budget cap','unknown/interrupted worker','identity mismatch','model rejection'],
      failure='preserve real prefix; block downstream; no retry/parameter tuning/second candidate',
      freeze_preconditions=['budget extension explicitly resolved','production implementation complete','zero-physics software checks',
        'physical arrays equal except integrator, initial836 equal','source zip and full input hashes frozen','single worker lock and counters'],
      budget='ff_prior_momentum_diagnosis/evidence/budget_derivation.json',
      success_unknown=True,training=False,expansion=False,TEST=False,advantage_comparison=False,worker_started=False)
    put('evidence/repair_spec.json',spec)
    doc('ROOT_CAUSE',f"""# 连续交接动量失败的根因
版本：{STAMP}。用户新授权的定向只读诊断；不重开旧认证、旧审查或旧resume，不改变旧裁决。

## 已定位的来源与触发条件

残差来源是2ms implicitfast的离散位姿—速度推进误差，主要在最初0–27.5秒静态运动中积累。之后两次交接的完整836值、保存动量均逐项相同，排除了交接漏状态或跳变。后29秒低运动段仍使用原公式，但段内交换量峰值更小，阈值比前驱约小22倍，因此继承初始帧已经失败。

|时刻或量|线性组 kg·m/s|转动组 kg·m²/s|
|---|---:|---:|
|原0秒|0|0|
|27.5秒末态|3.987307085e-5|5.570630306e-5|
|54.5秒末态/后段初态|4.314516949e-5|5.966906505e-5|
|后29秒原阈值|4.044799486e-5|4.472932761e-5|
|后29秒实际峰值|4.315719971e-5|5.966906505e-5|

转动组为基座自由关节局部广义共轭动量，不直接等于世界原点角动量。后段14501帧全部未过原门。最初0.012秒首次超过诊断显示用1e-12；此数不属于验收阈值。卫星公开外力作用于独立卫星，不能用于抵销机器人残差。

## 全41750个区间的证据

令p_b=M_b(q)v，P=p_b[:3]，H=R_b p_b[3:]+x_b×P。用A(q)表示到世界原点线/角动量的变换，G=A M_b，c_w=A c_b。每对相邻保存状态0、1及实际间隔h满足：

    e_v = G_0(v_1-v_0)+h c_w0
    e_q = (G_1-G_0)v_1-h c_w0
    P_world,1-P_world,0 = e_v+e_q

恒等式本身不证明根因；关键附加证据是每个区间e_v都在舍入误差量级，保存位姿差/h与下一速度一致。累计变化因此归于有限步长位姿相关动量输运项，而不是持续机器人外力、坐标解释错误或交接跳变。

|原段|最大速度平衡缺陷范数|世界线/角动量净变化范数|分解求和与实测净变化最大差|
|---|---:|---:|---:|
|0–27.5s|1.833e-16|3.9873e-5 / 5.5811e-5|1.617e-17|
|27.5–54.5s|2.473e-16|3.3114e-6 / 5.9356e-6|3.196e-17|
|54.5–83.5s|2.770e-18|5.4612e-8 / 1.5265e-8|2.546e-18|

41753保存状态含两个重复边界，质量矩阵/偏置静态重建的广义动量最大差0；两个动态段交换量重建最大差0。位姿差/h与下一速度最大差4.020e-13；逐区间恒等式最大分量差1.027e-15。基座阻尼、armature、frictionloss、重力/流体项均0；67直接关节执行器不施加基座广义输入；无tendon/equality。完整数字见evidence/diagnosis.json和discrete_defects_0/1/2.npz。

生产MuJoCo3.3.2采用implicitfast；官方文档描述其导数近似、半隐式位姿更新和RK4的精度/稳定性取舍：[MuJoCo3.3.2计算文档](https://mujoco.readthedocs.io/en/3.3.2/computation/index.html)。本结论是保存链的离散误差定位，不声称MuJoCo有软件bug。

## 修复依据和未知项

选择一个候选：原0秒完整数值初态起步，同2ms原生RK4重建整条83.5秒链。旧54.5秒状态已经携带残差，换积分器只能改变后续新增误差，不能合法删除继承值。

只读模型参数给出所有构型下的被动阻尼界：M(q)≥diag(armature)，所有非零阻尼DOF具有正armature，D≤5 diag(armature)，所以lambda_max(M^-1D)≤5/s，h·lambda≤0.01。RK4负实轴稳定区终点约2.785，由R(z)=1+z+z²/2+z³/6+z⁴/24求得。这只排除明显的被动阻尼步长障碍，不证明非线性闭环、几何或力矩门通过。另一次初态质量矩阵观察为4.999414/s，单列保留。

没有运行RK4轨迹，没有做步长收敛、优势对照或额外pilot。RK4是否能同时通过全部原门仍未知，必须由一次新单元实际执行裁决；失败不自动换步长、改控制器或调参。

tools/diagnose.py禁止step/积分/forward，以隔离MjData做静态运算，8.502秒；41753次全链质量/偏置查询加此前1次初态观察，共41754。tools/passive_bound.py仅读模型参数。源文件hash前后一致；0新原生、0重放、0耦合、0新几何查询。本诊断不是独立paper claim review，旧审查与唯一完成resume不重开。
""")
    doc('EXPERIMENT_PLAN',f"""# 原物理门下的完整链数值修复与预算
版本：{STAMP}。APP_DEV唯一拟议单元momentum_rebuild_001。根因及规格完成；预算增量待确认，生产实现、冻结和物理运行尚未开始。既有2秒推力/27秒追踪授权保持，只需解决超出原周期预算的增量。

## 唯一机制与不变条件

安装的MuJoCo3.3.2原生RK4，步长仍2ms。新模型差异白名单仅opt.integrator；保留2R5C、质量惯量/阻尼/armature、17→67、原七航点坐标姿态时长、C2参考、67力矩控制及离散制动过滤器、全部原门。旧冻结源码不改，新包实例级设置积分器。

沿用固定19×17参考Z，来自artifacts/v7_1/balanced_restore_001/units/u00/validated/validation.json；reference_hash={val['reference_hash']}。新身份下必须重新MODEL_VALID；不再次优化323，不增种子/候选/神经采样。历史Diffusion提案经过修正后的系统参考来源明确，不归功于裸网络。

旧54.5秒末态已携带误差，不能靠只修后段消除。新链从原0秒完整836数值初态开始，产生自己的新54.5秒末态；不能替换或冒称旧末态。

## 连续执行顺序

|阶段|时间|任务|执行/重放步|
|---|---|---|---:|
|A|0–27.5s|27秒接近＋0.5秒保持|13750/13750|
|B|27.5–54.5s|原七不规则航点及最终保持|13500/13500|
|C|54.5–83.5s|2秒公开推力＋27秒关机追踪|14500/14500|
|总计|83.5s|一条新实际连续链|41750/41750|

实际对象仅在0秒加载一次原836数值，A/B/C共用live MjData。27.5、54.5、56.5秒仅保存边界快照；不能reset/setState或写qpos/qvel/time。阶段控制器借用data，继承新实际前驱的reference_q/reference_qd与过滤记忆，公共目标不能按测量误差重新锚定。

独立重放对象也仅在0秒初始化一次，随后只应用保存控制/外力，跨阶段连续推进，不复制实际中间状态。每段原门、完整重放及必要审查通过后放行下一段；实际/重放对象暂停等待时不推进模拟时钟。一项新审查按阶段核对其新证据，最后汇总，避免反复重审。

每2ms冻结一次67力矩/外力输入；RK4内部4级不重新调用规划器。卫星保持原世界+Z COM推力0.02(1−cos(pi*t))N，中点采样1000步后关机13500步，峰值0.04N、冲量0.04Ns；保存全频推力/作用点/冲量/角冲量/功/动量。机器人外力0；仍为20kg固定质量公开外力模型，不是燃料/喷流硬件仿真。

## 验收与记录

evidence/momentum_contract.json明确原动量对象/坐标系/单位/两组索引及窗口。每阶段全部原生帧含初值，按原公式取各自交换量峰值。世界动量仅诊断，不替代原门、不扣初值、不借用全历史峰值、不添加无任务运动抬高分母、不投影或注入实际速度。

保持不穿透、5mm/1度、2mm全机几何步长、运动/力矩/关节/跟踪/流形、自由漂浮、完整836重放1e-10，以及原卫星冲量/角冲量/功/解析起源/关机惯性门。任一段失败或预算耗尽，保留实际前缀，下游BLOCKED；无第二候选、种子、调参或备用步长。

原生状态、力矩、基座、动量和重放差全量保存；碰撞门仍覆盖全部原生状态及原细分。另每20ms保存每障碍—臂段距离/几何ID/最近点，供展示；它不替代全频几何门。控制点3406合法对保存lower/exact/status；剪枝项明确标下界及原因，不冒充exact。所有活跃/遗漏约束保留身份、距离、选择原因、过滤前后命令。颜色不自动构成因果避障证据。

## 实现和冻结工作

- 新adapter：实例级积分器选择；物理数组逐项一致，原836数值不变，明确新model hash，不能绕过旧身份检查。
- 新inputs/protocol：固定Z与原场景/七点/时序/推力合同；旧终态字段仅历史来源，不参与新链初始化。
- continuous：同一实际data分阶段控制；独立连续replay；边界控制记忆及状态账本。
- recording/budget：原生步、内部级、耦合/模型区间、查询、时间和未压缩容量在推进前预留；PID创建时间和唯一worker锁。
- evaluation：原门逐项映射及原窗口不变，附加诊断不代替通过条件。
- 纯软件检查不推进物理：接口/初态/状态写入守卫/预算及记录形状/控制记忆检查；之后保存源码zip与所有源/输入hash再冻结运行。

RK4的完整闭环效果未知，是本单元要验证的风险。不得用额外短试跑绕过单元预算或失败边界。

## 完整预算与追加额

以前冻结预留全额占用为主账，不自动释放旧预留。本轮先占1800秒/64MiB，0耦合/原生。

|资源|本轮后保守余额|新单元硬上限|精确缺口|
|---|---:|---:|---:|
|耦合|10873|40|0|
|原生步含重放|34340|83500|49160|
|墙钟秒|6336.641|10800|4463.359|
|bytes|621069322|3221225472（3GiB）|2600156150（约2.422GiB）|

建议一次追加50000原生步、4500秒（75分钟）、2.5GiB，不增耦合。全周期312500步/12h/40GiB变为362500步/13h15m/42.5GiB；D3的165000步/21540秒/20GiB变为215000步/26040秒/22.5GiB。尚未应用任何上限变更。

若追加并按满额冻结，余10833耦合、840原生步、36.641秒、84198410bytes；余量不许可另一轮修复。

|时间子项|上限秒|
|---|---:|
|实现、软件核对、冻结|1800|
|固定323全程重新认证|600|
|实际执行及连续完整重放|3600|
|臂身/几何诊断|900|
|仅新单元分阶段独立审查|1800|
|全部媒体|1500|
|一次完成resume核对、报告、提交|600|
|合计|10800|

硬停止预算不是未测量的速度预测。生产2线程BLAS，一名worker。83500原生步对应334000个RK4内部动力学级，另有控制、观察forward及几何；不能当作单级积分成本。额外显式观察/forward上限100万；模型认证40耦合/200000预留模型积分区间（非原生步）；原生/细分几何上限100000状态，成对查询宽上界11亿。时间或容量先耗尽即停止，查询上界不准延长3小时。

未压缩记录上界：41753状态×32000bytes，4176全局20ms点×3406对×18bytes，600000活跃行×512bytes，4176×7障碍×71臂段组×128bytes，100000几何细分记录×800bytes，模型认证512MiB、媒体256MiB、源码/元数据64MiB。合计{budget['storage_envelope_bytes']}bytes，3GiB内余{budget['storage_headroom_bytes']}bytes。所有变长JSON按实际bytes计数；存储不足必须停止，不能丢记录或减少验收点。旧mesh/模型大文件只绑定hash，不重复复制。

旧预留若正式核销可能释放时间/存储，但34340原生余额不变，49160步缺口仍在。当前使用保守主账，不借用未核销额度。机器可读完整算式见evidence/budget_derivation.json。

## 执行目标

|里程碑|放行要求|当前状态|
|---|---|---|
|M1 根因与原合同|全区间分解/交接/源hash明确|完成，只读|
|M2 唯一机制与完整预算|固定参考/原门/停止规则，明确超额|规格完成，增量待确认|
|M3 实现与冻结|增量明确、零物理软件检查、完整源输入身份|未开始|
|M4A 新静态|MODEL_VALID、执行、重放及必要审查全过|依赖M3|
|M4B 原七点|真实连续、7/7及保持、全部门/重放/审查|依赖A|
|M4C 实体起源接续|原机器人和卫星全部门/重放/审查|依赖B|
|M5 全部交付|仅一次完成resume、原始输出bytes及前后清单、报告|依赖新结果|

成功媒体：两视角0–83.5秒实际视频、三维、EE/Goal RGB轴、累计双臂末端轨迹、W1–W7、臂段颜色/距离线/热图、形状曲率、误差/速度/加速度/力矩、基座/动量/推力/冲量/功、完整836重放差和数值索引。目标轴同步保存目标。显示抽帧不用于验收；失败仅交付明确失败的实际前缀，不补成功动画。

旧结果、旧审查、旧完成resume不重开；旧定时任务保持删除，长期goal不complete/paused。不训练、不扩库、不优势对照、不TEST，无硬件/接触捕获。
""")
    doc('EXPERIMENT_TRACKER',f"""# 执行跟踪 {STAMP}

|ID|内容|状态|下一步|
|---|---|---|---|
|M1|全41750区间根因及原动量合同|DONE_READ_ONLY|保留原门|
|M2|原生2ms RK4完整链与成本|SPEC_READY_BUDGET_EXTENSION_PENDING|追加50000步/75分钟/2.5GiB|
|M3|实现和冻结|NOT_STARTED|预算确定后零物理检查|
|M4A/B/C|新实际连续链、重放和审查|NOT_STARTED|单worker分阶段原门放行|
|M5|新run收尾及全部可视化|DEPENDENCY_PENDING|只一次完成resume|

旧moving_satellite_001仍为原动量失败；重放一致不代表全部通过。0新原生/重放/耦合，未增加预算或创建worker。
""")
    put('evidence/execution_goal.json',dict(objective='同2R5C完整原初态→静态→原七点→2秒公开推力→27秒关机追踪，原门及完整重放通过后全部可视化',
      status='INCOMPLETE_ROOT_CAUSE_DIAGNOSED_REBUILD_BUDGET_EXTENSION_REQUIRED',
      plan='ff_prior_momentum_diagnosis/docs/EXPERIMENT_PLAN.md',budget='ff_prior_momentum_diagnosis/evidence/budget_derivation.json',
      root_cause='ff_prior_momentum_diagnosis/docs/ROOT_CAUSE.md',repair='ff_prior_momentum_diagnosis/evidence/repair_spec.json',
      production_implemented=False,frozen=False,worker_started=False,app_goal_not_mutated=True,old_verdicts_unchanged=True))
    note="""# 2026-10-23 动量根因定位完成：完整链修复预算待增补

最新用户要求已推进：全41750保存区间只读分解，两个完整836交接逐项相同；残差主要来自初始静态段2ms implicitfast离散位姿—速度推进，后段低交换量阈值使继承初值超限。0新原生/重放/耦合；旧失败裁决不变。证据ff_prior_momentum_diagnosis/docs/ROOT_CAUSE.md。

具体方案ff_prior_momentum_diagnosis/docs/EXPERIMENT_PLAN.md：原0秒完整数值初态，同2R5C/固定323/原七点/控制器/公开2秒推力，仅原生积分器改2ms RK4，重建0–83.5秒连续链及独立连续836重放。原门及窗口不变、不扣初值/注速/重置。RK4效果尚未实际验证。

本轮1800秒/64MiB只读预留后余额34340原生/6336.641秒/621069322bytes/10873耦合。新单元83500原生/10800秒/3GiB/40耦合，缺49160步/4463.359秒/2600156150bytes；建议追加50000步/75分钟/2.5GiB，不增耦合。未改原上限，未生产实现/冻结/运行；预算确定后按规格实施单worker。旧定时任务保持删除，长期目标未完成。下文为历史，不覆盖本节及control。

"""
    for p in ['DIFFUSION_GOAL_CURRENT.md','ff_prior_fullchain/docs/CONTINUOUS_DELIVERY_GOAL.md']:
        q=ROOT/p;old=q.read_text(encoding='utf-8-sig');q.write_text(note+old,encoding='utf-8')
    c=read('ff_prior_fullchain/control.json')
    c.update(current_user_steering='ff_prior_momentum_diagnosis/evidence/scope.json',
      decision_document='ff_prior_momentum_diagnosis/docs/EXPERIMENT_PLAN.md',
      next_phase='D3_MOMENTUM_ROOT_CAUSE_DIAGNOSED_REBUILD_BUDGET_REQUIRED',
      next_phase_action='Concrete original-state2ms RK4 rebuild; additional50000native/4500s/2.5GiB must be resolved before new implementation freeze and one worker. No physics now.',
      implementation_handoff='ff_prior_momentum_diagnosis/evidence/repair_spec.json',
      latest_planning_goal='ff_prior_momentum_diagnosis/evidence/execution_goal.json',
      latest_planning_budget='ff_prior_momentum_diagnosis/evidence/budget_derivation.json',
      new_momentum_failure_impasse_turns=0,
      autonomous_scope='Latest root-cause and repair/budget request. Existing force/tracking scope persists; original budget shortfall unresolved.')
    (ROOT/'ff_prior_fullchain/control.json').write_text(json.dumps(c,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    outputs=sorted(p.relative_to(ROOT).as_posix() for p in PKG.rglob('*') if p.is_file())
    put('evidence/plan_assembly.json',dict(timestamp=STAMP,outputs=outputs,arithmetic='PASS',new_native_steps=0,new_coupled_calls=0,exact_shortfall=short))
    with (ROOT/'MANIFEST.md').open('a',encoding='utf-8') as f:
        for p in outputs+['ff_prior_momentum_diagnosis/evidence/plan_assembly.json']:
            f.write(f'\n| {STAMP} | /experiment-plan | {p} | engineering | 动量根因、原门修复规格及完整追加预算；0新物理 |\n')
    print(json.dumps(dict(timestamp=STAMP,balance=balance,shortfall=short,storage=budget['storage_envelope_bytes'],headroom=budget['storage_headroom_bytes'],remaining_if_approved=left),ensure_ascii=False))
if __name__=='__main__':main()
