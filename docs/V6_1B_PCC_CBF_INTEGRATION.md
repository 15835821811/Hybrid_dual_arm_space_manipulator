# V6.1-B：PCC/胶囊臂形感知 CBF 控制集成

## 1. 结论

V6.1-B 已把 V6.1-A 的连续体臂形几何从只读影子审计接入 V6-lite 的真实控制链，同时保留原 MuJoCo 几何 CBF：

```text
10 维连续体形状 + 浮动基座位姿 ─→ 任意弧长 PCC 曲线
                                  ├→ PCC 管体—移动卫星 OBB 净空/梯度 ─┐
60 维实际离散链 ──────────────────┴→ 61 胶囊—移动卫星 OBB 净空/梯度 ─┤
原 2,927 对 MuJoCo 几何距离 CBF ──────────────────────────────────────┤
目标卫星实测 6D 速度 ───────────────────────────────→ 外生距离漂移 ─┤
                                                                    ↓
                         单一 17 维约束速度 QP，50 Hz
                                      ↓
                         67 路逆动力学力矩伺服，500 Hz
```

新约束默认关闭；`enable_pcc_cbf=False`、`enable_capsule_cbf=False` 时仍是 V6-lite 基线。启用时没有增加第二个在线优化器，也没有删除或放宽任何原 MuJoCo 距离约束。

## 2. 任意臂形点与位置 Jacobian

连续体由 5 段等曲率段组成。给定形状变量 `q_c ∈ R^10`、当前浮动基座变换 `T_b` 和总弧长 `s ∈ [0,L]`，`ContinuumShapeModel.evaluate()` 返回：

```text
p(s), R(s), t(s), Jp(s)=∂p(s)/∂q_c, JR(s)
```

零曲率附近使用矩阵指数积分的级数形式，避免 `κ→0` 时除零；`batch_query()` 可一次查询多个弧长点。`continuum_jacobian.compute_position_jacobian()` 的固定接口输出 `(3,10)`。

正式审计在 `q_c∈[-1,1]^10` 内随机取 1,000 个构型和弧长，逐坐标中心差分 10,000 次；位置 Jacobian 最大相对误差为 `2.902799e-5`，低于 5% 门槛。

## 3. PCC 管体与移动卫星距离

目标卫星碰撞盒由 MuJoCo 当前 `geom_xpos/geom_xmat` 构造随时间平移和旋转的 OBB。第 `i` 段使用 V6.1-A 标定半径 `r_i`，净空定义为：

```text
d_pcc(q_c,T_b,T_T) = min_{i,s∈segment_i} [sd_OBB(p_i(s),T_T) - r_i]
```

在线查询先在每段做固定粗采样，再只对候选段执行有界局部细化。返回值包含距离、最近段、总弧长、PCC 中心线点、管体表面点、OBB 点、法向、半径、粗/细查询次数和 `∂d/∂q_c`。

在最近特征稳定且位于盒外的可微区域，包络定理给出：

```text
∂d_pcc/∂q_c = nᵀ Jp(s*)
```

64 个稳定外部案例的最大相对有限差分误差为 `9.704045e-6`。穿透、OBB 中轴面和最近特征切换处的最小距离本来就不可微，审计会显式排除这些点，而不会把某个任意法向伪装成唯一导数。

## 4. 胶囊距离与精确宽相位

实际 60 关节离散链的 61 个远端 collision geom 分别由有限半径胶囊包络；安装座 `collision_0003` 继续由原 MuJoCo 几何策略负责。

为满足 50 Hz，胶囊查询先使用 OBB signed-distance 的 1-Lipschitz 下界：

```text
d(capsule,OBB) ≥ sd_OBB(axis_midpoint) - axis_half_length - radius
```

候选按该下界排序；当剩余下界不可能优于当前精确最小值时停止。这个宽相位不会近似或漏掉最小胶囊，100 个随机状态与遍历全部 61 个胶囊的结果逐项一致。

## 5. 自由漂浮广义距离 Jacobian

PCC 内部形变只直接作用于前 10 个规划变量，但任意一维机械臂运动都会通过零动量反作用使基座运动。因此 PCC CBF 的受控梯度为：

```text
A_pcc = nᵀ J_base-point G + [∂d_pcc/∂q_c, 0_1×7]
```

其中 `G∈R^(79×17)` 是原 V6-lite 的自由基座反作用映射，`J_base-point` 是把当前 PCC 最近中心线点视作刚性附着在基座上的点 Jacobian。非退化弯曲构型上的完整 17 维方向导数已用 `mj_integratePos` 中心差分验证。

胶囊约束直接使用其实际离散链最近点的 MuJoCo 点 Jacobian：

```text
A_capsule = nᵀ J_arm-point G
```

移动目标卫星不是 QP 决策变量。其平移和旋转通过 OBB 最近点的目标 body Jacobian形成外生漂移：

```text
d_dot_T = -nᵀ J_target-point qvel_T^exo
A qdot_plan ≥ -γ(d-d_safe) - d_dot_T
```

PCC 与胶囊均使用 `d_safe=5 mm`；PCC/胶囊行与原 MuJoCo CBF 行统一由 `_build_all_clearance_constraints()` 装配。

## 6. QP 如何改变控制动作

控制器先计算不含安全行的名义任务速度，再检查 PCC 行的速度缺口；`pcc_avoidance_intervention` 报告归一化的最大正缺口。安全行与任务目标、速度盒约束、关节 barrier 一起进入同一个 17 维 ADMM QP。

启用版五场景中：

| 指标 | 结果 |
| --- | ---: |
| PCC 激活次数 | 3,718 |
| PCC 绑定次数 | 202 |
| 胶囊激活次数 | 579 |
| 最大 PCC 干预量 | 0.049459 |
| QP 失败次数 | 0 |
| 五场景最差 50 Hz 全链 p95 | 18.089305 ms |

求解器仍为单一 17 维 ADMM QP。为满足 20 ms 门限，V6.1-B 使用约束来源对偶热启动、松弛系数 1.8、PCC 候选段局部细化上限 5 次，以及上述有严格下界的胶囊宽相位。

## 7. 真实 MuJoCo 是否安全

两组五场景都经过相同的 26 项独立验证与 500 Hz 力矩重放：

| 模式 | 26 项检查 | QP 失败 | 连续体—卫星 500 Hz 最小间隙 |
| --- | ---: | ---: | ---: |
| 基线（PCC/胶囊关闭） | 26/26 | 0 | 24.960121 mm |
| V6.1-B（PCC/胶囊开启） | 26/26 | 0 | 78.529716 mm |

启用版的控制动作确实被 PCC 行改变，同时所有保存的 500 Hz 状态均由原 MuJoCo collision geom 独立重算，最低值仍远高于 5 mm。因此“安全”不是由代理距离自我证明的。

另有 10,000 个机械臂构型/卫星相对位姿案例覆盖盒面、盒棱、盒角、旋转、中段擦碰和主动穿入；假安全定义为代理 `>=5 mm` 而负责包络的真实 MuJoCo 几何 `<5 mm`。PCC 和胶囊的假安全计数均为 `0/10,000`。

这些结果是固定模型、有限种子和离散时刻上的回归证据，不是全状态解析证书、连续时间 CCD 证书或硬件安全认证。

## 8. 输入、输出与复现

新增在线输入是当前低层连续体关节位置、浮动基座位姿、目标卫星 OBB 位姿/6D 速度，以及两个默认关闭的功能开关。新增 50 Hz 输出包括三种净空、距离/梯度误差、最近 PCC 段/弧长、PCC/胶囊激活与绑定计数、干预量和几何计算时延。

正式产物：

```text
v6_lite/output/v6_1_b/pcc_audit.json
v6_lite/output/v6_1_b/clearance_compare.json
v6_lite/output/v6_1_b/regression_report.json
v6_lite/output/v6_1_b/plots/distance_comparison.png
v6_lite/output/v6_1_b/plots/gradient_comparison.png
v6_lite/output/v6_1_b/plots/minimum_clearance_comparison.png
```

复现命令：

```powershell
python -m v6_lite.audit_v6_1_b
python -m v6_lite.run_v6_lite --output-dir v6_lite/output/v6_1_b/baseline_root/output
python -m v6_lite.run_v6_lite --output-dir v6_lite/output/v6_1_b/enabled_root/output --enable-pcc-cbf --enable-capsule-cbf
python -m v6_lite.finalize_v6_1_b
python -m unittest v6_lite.test_v6_1_b -v
```

原始 A/B trace 约 275 MB，默认不纳入 Git；上述审计 JSON、聚合 50 Hz 监控和三张图会纳入版本控制。

`test_v6_1a_artifacts.py` 中“V6 控制源码哈希不得变化”的检查属于冻结的 `v6.1-a` 分支；它按设计会拒绝 V6.1-B 对 `hierarchical_qp.py` 和 `run_v6_lite.py` 的正式集成修改。V6.1-A 分支/标签仍保留该哈希门禁，V6.1-B 使用 `test_v6_1_b.py`、基线 26/26 和启用版 26/26 作为对应门禁，旧测试文件本身未删除或改写。
