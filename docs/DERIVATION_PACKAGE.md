# V6-lite 连续体机械臂—目标卫星防穿模：几何距离与安全约束推导包

## 0. Executive verdict

可以基于分段等曲率（PCC）建立整条臂形的解析曲线，并把“有限半径臂体到目标卫星”的最小距离作为 V6-lite 速度 QP 的硬约束。对本项目，推荐采用：

1. 5 段 PCC 中心线的矩阵指数递推；
2. 中心线与实际半径的 Minkowski 膨胀（管体/圆环体），而不是只检查中心点；
3. 目标卫星按随时间运动、旋转的 OBB（当前模型为边长 0.4 m 的立方体）表示；
4. 连续弧长最小距离或带误差证书的胶囊/球包络；
5. 将相对距离导数写成时变 CBF，加入当前 17 维、反作用感知的速度 QP；
6. 仅对有意抓取的刚性末端—卫星接触作白名单，连续体臂—卫星不得整体排除。

## Status

当前可下的最强结论是：

> **Status: COHERENT AFTER REFRAMING / EXTRA ASSUMPTION**

“PCC 中心线 + OBB 点有符号距离 + 一维弧长最小化”是完整、可实现的几何方法；但不能把它表述为“存在一个对所有盒体面/棱/角情况都处处光滑的全局闭式最小距离公式”。严格安全还依赖以下额外条件：PCC—MuJoCo 形状误差有上界、臂体半径包络真实网格、控制离散化与跟踪误差被安全裕度覆盖，且 QP 不允许用松弛变量破坏安全约束。

工程状态需与理论状态分开理解：`v6_lite_6` 已完成 P0——直接以 MuJoCo 精确碰撞几何加入目标 pair、移动目标外生漂移 CBF 和原生 500 Hz 离散重放门禁；P1 的解析 PCC 代理及其全状态域误差证书仍未实现，因此本文后半的 PCC 推导仍是下一阶段方法设计，而不是当前运行代码的几何后端。

## 1. 当前工程事实与修复前根因

### 1.1 `v6_lite_6` 已实现合同

- 规划变量是 17 维，其中前 10 维为 5 段连续体臂、每段 2 个弯曲变量；它们通过 12×2 映射重复到 60 个低层连续体转轴。任务级速度 QP 周期为 0.02 s，物理/力矩伺服周期为 0.002 s。
- QP 以 `mj_geomDistance` 的 signed distance、witness points 和双方点 Jacobian 构造一阶 barrier，并通过零基座动量映射考虑自由漂浮基座反作用。目标卫星碰撞体是通过 floating joint 运动的 0.4 m 立方体。
- shared verifier 的目标 pair 是 opt-in；V6 显式启用后，在带两个工作区障碍的场景中共有 2,927 个 pair，其中目标相关 75 个：`continuum_target=62`、`rigid_target=7`、`base_target=6`。
- 接触白名单严格限定为 `collision_0072/end_link` 与 `collision_0073/end_effector_r` 两个终端几何；`collision_0071/wrist` 仍受 `rigid_target` 约束，连续体和基座没有目标接触豁免。
- 在线缓冲按 pair 分类：`continuum_target`、`base_target` 以及普通 pair 为 25 mm，`rigid_target` 为 5 mm；统一正式离散验收门槛为 5 mm。
- 目标自由关节实测 6D `qvel` 作为外生漂移进入 CBF，包含 witness point 的平移与旋转速度。独立验证除 4×细分整机审计外，还对包括初始状态的全部原生 500 Hz 重放状态逐一扫描 62 个 `continuum_target` pair，并硬门禁低于 5 mm 与穿透状态数均为零。

### 1.2 `v6_lite_5` 修复前穿模基线

以下只描述旧 `v6_lite_5`，不能用于判断当前 `v6_lite_6`。当时根因不是“系统完全没有碰撞约束”，而是**目标卫星被整体排除在 clearance gate 之外**：

- 旧策略没有构造连续体臂—目标卫星 pair；
- 运行时接触响应关闭，物理引擎不会通过接触冲量把机械臂推出卫星；
- 因而旧 QP 不会主动绕开目标卫星，且没有独立的原生 500 Hz 目标间隙硬门禁。

对旧五场景 trace 的 500 Hz 复查均检测到连续体臂进入目标卫星；五场景最坏 signed clearance 依次约为 −44.132、−35.681、−53.262、−64.246、−96.146 mm。它们是**修复前失败基线**，不是当前合同结果，也不能替代 `v6_lite_6` 的正式五场景重跑与独立验收。历史诊断图已与当前正式输出隔离，保存在 `docs/historical/v6_lite_5/continuum_target_penetration_scenario04.png`。

## 2. Target

为 V6-lite 构造一个可进入实时 QP 的全臂安全量：

\[
h(q,t)=d_{\mathrm{arm,sat}}(q,t)-d_{\mathrm{safe}},
\]

并通过

\[
\dot h(q,t)+\gamma h(q,t)\ge 0,\qquad \gamma>0,
\]

使安全集 \(\mathcal C(t)=\{q:h(q,t)\ge0\}\) 在模型假设成立时保持前向不变。

## 3. Invariant Object

核心不变量不是末端位置，而是：

\[
\boxed{\min_{x\in\mathcal B(q),\;y\in\mathcal S(t)}\|x-y\|\ge d_{\mathrm{safe}}}
\]

其中 \(\mathcal B(q)\) 是整个连续体臂的有限体积占据集，\(\mathcal S(t)\) 是卫星占据集。只约束末端、每段端点或稀疏中心线采样点，均不等价于这个不变量。

## 4. Assumptions

| 编号 | 假设 | 性质与处理 |
|---|---|---|
| A1 | 每段在一个控制周期内可用恒定曲率/恒定应变表示 | PCC 模型假设；需用 MuJoCo FK 残差验证 |
| A2 | 当前 10 个变量可经固定的每段标定矩阵映射为二维总弯曲角或曲率 | 当前 12×2 分配支持此结构，但轴序、符号、尺度必须标定 |
| A3 | 每段实际碰撞几何被半径 \(r_i(s)\) 的管体包住 | 必须从碰撞 STL/geom 计算，不得直接使用绘图直径 |
| A4 | 目标卫星为已知位姿和 twist 的 OBB，半边长 \(a=(0.2,0.2,0.2)\) m | 与当前 URDF 一致；若改为网格，应替换 SDF 后端 |
| A5 | QP 到实际 500 Hz 伺服的误差可由 \(\varepsilon_{\rm track}\) 上界覆盖 | 需从 replay 统计或离线辨识得到 |
| A6 | 所有离散采样/数值最小化误差有保守上界 | 用 Lipschitz 或弦弧 sagitta 证书覆盖 |
| A7 | 安全约束是硬约束；不可行时降低/松弛跟踪任务，而非距离约束 | 控制策略要求 |

## 5. Notation

- \(i\in\{1,\ldots,5\}\)：PCC 段编号；\(s\in[0,L_i]\)：该段弧长。
- \(q_c=[\beta_1^\top,\ldots,\beta_5^\top]^\top\in\mathbb R^{10}\)：连续体规划变量，\(\beta_i\in\mathbb R^2\)。
- \((R_{i-1},p_{i-1})\)：第 \(i\) 段基座在世界系的姿态与位置。
- \(e_x=(1,0,0)^\top\)：本项目离散臂零位的局部主轴方向。
- \(c_T(t),R_T(t),v_T(t),\omega_T(t)\)：卫星中心、旋转、线速度、角速度。
- \([u]_\times\)：向量 \(u\) 的反对称叉乘矩阵。
- \(G(q)\in\mathbb R^{n_v\times17}\)：V6-lite 从 17 维规划速度到全系统广义速度的反作用感知映射。

## 6. Derivation Strategy

推导路线为：

\[
q_c
\longrightarrow
\{u_i,L_i\}_{i=1}^{5}
\longrightarrow
p_i(s;q_c)
\longrightarrow
\mathrm{sd}_{\rm box}(p_i(s),t)-r_i(s)
\longrightarrow
\min_{i,s}
\longrightarrow
h(q,t)
\longrightarrow
\nabla_q h\,\dot q+\partial_t h\ge-\gamma h.
\]

其中前半段是几何/运动学，后半段是安全控制。这样可以清楚区分“距离定义正确”与“控制器能否保证不进入不安全集”。

## 7. Derivation Map

| 步骤 | 对象 | 输出 | 需要验证 |
|---|---|---|---|
| D1 | 10 维弯曲变量 | 每段曲率向量 \(u_i\) | 有限差分标定轴序、符号和尺度 |
| D2 | PCC 指数映射 | 任意弧长点 \(p_i(s)\)、姿态 \(R_i(s)\) | 与 MuJoCo 每个 link/joint 中心对齐 |
| D3 | OBB SDF | 点到卫星的 signed distance 和法向 | 面、棱、角及盒内情况单元测试 |
| D4 | 管体膨胀 | 整段臂体 clearance | 半径必须包络实际碰撞网格 |
| D5 | 一维最小化/包络 | 连续全臂最小距离 | 不漏掉采样点间最小值 |
| D6 | Jacobian/相对速度 | \(\dot h\) | 有限差分距离梯度测试 |
| D7 | 时变 CBF-QP | 线性不等式行 | 动态卫星、基座反作用、不可行处理 |

## 8. Main Derivation

### 8.1 从当前 2 维段变量得到空间曲率

为避免 \((\kappa,\phi)\) 在 \(\kappa=0\) 处的坐标奇异，控制内部使用二维正交弯曲分量。若 \(\beta_i\) 表示该段两个方向的**总弯曲角**，定义

\[
u_i=\frac{1}{L_i}C_i\beta_i\in\mathbb R^3,
\qquad
u_i^\top e_x=0,
\]

其中 \(C_i\in\mathbb R^{3\times2}\) 是用 MuJoCo 有限差分标定的轴序/符号矩阵。若 \(\beta_i\) 本身已经是曲率，则不除以 \(L_i\)。

对当前仓库的零位有限差分与转轴顺序核对后，可采用

\[
C_i=
\begin{bmatrix}
0&0\\
0&1\\
1&0
\end{bmatrix},
\]

即第一分量 \(\alpha_i=\theta_{2i-1}\) 绕局部 \(+z\) 弯曲，使 \(+x\) 朝 \(+y\)；第二分量 \(\beta_i=\theta_{2i}\) 绕局部 \(+y\) 弯曲，使 \(+x\) 朝 \(-z\)。因此

\[
u_i=\frac1{L_i}[0,\beta_i,\alpha_i]^\top,
\qquad
\kappa_i=\frac{\sqrt{\alpha_i^2+\beta_i^2}}{L_i}.
\]

仓库几何给出的名义长度为

\[
L=[0.300,0.300,0.300,0.300,0.2975]\ \mathrm m,
\]

直态下基座体到第一弯曲站的固定偏移为约 \([0.4325,0.626,0]\) m。上述数值应固化为自动回归测试；若 URDF 或角度约定改变，必须重新标定，而不能依赖变量名称猜测。

### 8.2 单段 PCC 的精确中心线

令

\[
K_i=[u_i]_\times,qquad \kappa_i=\|u_i\|_2.
\]

对第 \(i\) 段任意弧长 \(s\in[0,L_i]\)，恒定应变的姿态和中心线为

\[
R_i(s)=R_{i-1}\exp(K_i s),
\]

\[
\boxed{
p_i(s)=p_{i-1}+R_{i-1}V(u_i,s)e_x
}
\]

其中

\[
V(u,s)=
sI+
\frac{1-\cos(\kappa s)}{\kappa^2}[u]_\times+
\frac{\kappa s-\sin(\kappa s)}{\kappa^3}[u]_\times^2.
\]

这是一个**恒等式（在恒定曲率/恒定应变假设内）**。当 \(\kappa\to0\) 时，不应直接除以小曲率，而应使用连续极限/级数：

\[
V(u,s)=sI+\frac{s^2}{2}[u]_\times+\frac{s^3}{6}[u]_\times^2+O(\kappa^3s^4),
\]

因此直臂极限为 \(p_i(s)=p_{i-1}+sR_{i-1}e_x\)。

多段递推：

\[
p_i^{\rm end}=p_i(L_i),\qquad
R_i^{\rm end}=R_i(L_i),
\]

并把它们作为下一段的 \((p_i,R_i)\)。该形式与 Webster–Jones 的圆弧公式等价，只是把文献常用的初始 \(+z\) 切向轴换成了本项目的初始 \(+x\) 轴。

对当前 \(u_i=[0,\beta_i/L_i,\alpha_i/L_i]^\top\)，单段在自身局部系还可显式写为

\[
p_i^{\rm local}(s)=
\begin{bmatrix}
\sin(\kappa_i s)/\kappa_i\\
\dfrac{\alpha_i}{\sqrt{\alpha_i^2+\beta_i^2}}
\dfrac{1-\cos(\kappa_i s)}{\kappa_i}\\
-\dfrac{\beta_i}{\sqrt{\alpha_i^2+\beta_i^2}}
\dfrac{1-\cos(\kappa_i s)}{\kappa_i}
\end{bmatrix},
\]

零弯曲时取连续极限 \([s,0,0]^\top\)。实现时仍推荐矩阵指数或稳定的 `sinc/cosc` 形式，以避免式中 \(0/0\)。

### 8.3 目标卫星 OBB 的点有符号距离

当前卫星盒体半边长为

\[
a=(0.2,0.2,0.2)^\top\ \mathrm m.
\]

对任意世界点 \(p\)，先变换到卫星局部系：

\[
z=R_T^\top(p-c_T),\qquad w=|z|-a.
\]

点到 OBB 的精确有符号距离（盒外为正、盒内为负）为

\[
\boxed{
\mathrm{sd}_{\rm box}(p,t)
=\|\max(w,0)\|_2+
\min\bigl(\max(w_x,w_y,w_z),0\bigr).
}
\]

在盒面/棱/角的特征切换处，它是连续且 Lipschitz 的，但不处处可微。盒外光滑区域的世界系单位法向记为 \(n_W(p,t)=\nabla_p\mathrm{sd}_{\rm box}\)。

实现法向时，若点在盒外，令 \(z_c=\operatorname{clip}(z,-a,a)\)，则

\[
n_W=R_T\frac{z-z_c}{\|z-z_c\|}.
\]

若点在盒内，则选择离最近面的轴

\[
j=\arg\min_j(a_j-|z_j|),\qquad
n_W=R_T\,\operatorname{sign}(z_j)e_j.
\]

等距面、棱、角会产生多值次梯度；控制中应保留并列活动特征或使用保守平滑下界。

### 8.4 从中心线扩张为真实臂体

只用 \(\mathrm{sd}_{\rm box}(p_i(s))\) 会把机械臂误当成零半径曲线。第 \(i\) 段的保守占据集应定义为

\[
\mathcal B_i(q)=
\bigcup_{s\in[0,L_i]}
\{p_i(s;q)+\delta:\|\delta\|_2\le r_i(s)\}.
\]

若横截面可用半径 \(r_i(s)\) 的圆包络，则这一理想管体到 OBB 的 signed clearance 为

\[
\boxed{
d(q,t)=
\min_{i=1,\ldots,5}\;
\min_{s\in[0,L_i]}
\left[
\mathrm{sd}_{\rm box}\bigl(p_i(s;q),t\bigr)-r_i(s)
\right].
}
\]

这对“PCC 圆管—盒体”的几何定义是精确的。对当前 MuJoCo STL，它只有在 \(r_i(s)\) 与额外误差裕度确实包络网格后才是保守代理。

特别注意：当前可视化助手的默认值写为 `continuum_diameter_mm=16.0`（`model_test/dual_arm_space_robot.py:465-488`），但它明确只是绘图参数。逐顶点检查当前 30 个 link collision mesh 后，网格到对应中心线段的最大径向距离约为 **16.000002 mm**；joint mesh 的横截面半尺寸约为 6.5 mm，末端盒半尺寸约为 5 mm。因此当前柔性链可由 30 个半径略大于 16 mm 的胶囊保守包络；安装基座等非该链几何仍应单独使用 exact geom。这里同时说明：把绘图参数的“16 mm 直径”直接解释为 8 mm 安全半径会低估真实碰撞几何约一倍。

还要区分两个模型：V6-lite 的 10 维变量在仿真中不是直接驱动解析圆弧，而是每段通过模式

\[
[\beta_i,\alpha_i,\alpha_i,\beta_i,\beta_i,\alpha_i,
\alpha_i,\beta_i,\beta_i,\alpha_i,\alpha_i,\beta_i]/6
\]

驱动 12 个离散转轴。对 5 条正式轨迹按 `task_qpos[::10]` 抽取约 680 个 50 Hz 状态，把上述解析 PCC 段端点与 MuJoCo 的 `joint_7/13/19/25/end_effector` 比对；在样本段合成弯曲角最大约 0.937 rad 的范围内，观察到的最大端点位置偏差为 **23.74 mm**（场景 03、第 4 段）。这只是“正式轨迹上的段端点诊断最大值”，不是全曲线 Hausdorff 上界；第一段在规划边界 \([ -\pi,\pi]^2\) 的 51×51 粗网格上，端点差异还可达到约 **50.19 mm**。所以“解析 PCC + 16 mm 管径”不能未经膨胀就作为最终硬安全几何，也不能直接把 23.74 mm 当作全域安全界；必须加入经所需状态域验证的 \(\varepsilon_{\rm PCC}(q)\)，或采用“PCC 做预测、MuJoCo/30 胶囊做硬约束”的双层方案。

### 8.5 连续最小值的实时计算

对每段定义一维函数

\[
f_i(s)=\mathrm{sd}_{\rm box}(p_i(s),t)-r_i(s).
\]

推荐顺序：

1. 用少量弧长节点做 broad phase；
2. 对接近激活阈值的区间做有界一维最小化；
3. 保留所有落入 activation band 的局部最小点，而不是只保留唯一全局最小点；
4. 在面/棱/角切换处同时保留竞争特征，或使用保守的 smooth-min 下界。

若采用离散弧长节点，必须给出不漏检证书。由于弧长参数满足 \(\|\partial p/\partial s\|=1\)，距离函数是 1-Lipschitz；若最大相邻采样间距为 \(\Delta s\)，且半径为常数，则

\[
\min_s f_i(s)
\ge
\min_k f_i(s_k)-\frac{\Delta s}{2}.
\]

因此保守的离散约束是

\[
\boxed{
\min_{i,k}\left[
\mathrm{sd}_{\rm box}(p_i(s_k),t)-r_i
\right]
-\frac{\Delta s}{2}
-\varepsilon_{\rm model}
\ge d_{\rm safe}.
}
\]

这是一个**命题（给定弧长参数、常半径及距离函数 Lipschitz 性）**。例如 \(\Delta s=50\) mm 会强制增加 25 mm 的采样裕度，过于保守；\(\Delta s=5\) mm 对应 2.5 mm 证书，更合理但约束更多。

更高效的替代是用相邻节点的弦线段生成胶囊，并把圆弧—弦的最大 sagitta 加到胶囊半径：

\[
\varepsilon_{\rm sag}
=\frac{1}{\kappa}
\left(1-\cos\frac{\kappa\Delta s}{2}\right)
\approx\frac{\kappa\Delta s^2}{8}.
\]

于是用“线段到 OBB 的距离 − \((r_i+\varepsilon_{\rm sag})\)”即可保守包住该小段圆弧。该方案通常比把 \(\Delta s/2\) 全部作为裕度更紧。

若用多个候选距离 \(d_j\)，保守 smooth-min 可定义为

\[
\underline d_\rho
=-\frac1\rho\log\sum_j e^{-\rho d_j}
\le \min_j d_j.
\]

所以约束 \(\underline d_\rho\ge d_{\rm safe}\) 是保守的；反方向的普通 softmax/平均不具有这个保证。

### 8.6 与当前离散 MuJoCo 臂完全对齐的胶囊模型

若目标是先解决当前仿真穿模，而不是先证明解析 PCC 代理，则可直接用 30 个离散 link frame 构造安全胶囊：

\[
c_k(\lambda)=x_k+\lambda R_k e_x,
\]

其中 \(x_k=\texttt{data.xpos[link\_k]}\)、\(R_k=\texttt{data.xmat[link\_k]}\)，对 \(k<30\) 取 \(\lambda\in[0,0.05]\) m，对末段取 \(\lambda\in[0,0.0475]\) m；网格审计值为 0.0160000016 m，工程实现可先取 **0.0161 m** 再叠加独立的状态/时间裕度。然后计算每个线段到卫星 OBB 的距离并减胶囊半径。

这是对当前离散链更贴近、也更容易审计的保守模型；它仍可通过线段最近点的 Jacobian 写成同样的 CBF 行。推荐的双层结构是：

- 解析 5 段 PCC：用于低维预测、规划和论文中的形状表达；
- 30 胶囊或 62 个 MuJoCo collision geoms：用于最终硬安全约束与独立 replay 验证。

仓库里已有可复用原型，而不必从零开始：

- `model_test/mujoco_track_circle_arc_gjm_plan_track.py:591-622` 已实现 OBB 点 SDF、盒内外逃逸方向；
- `model_test/mujoco_track_irregular_waypoints_diffusion_adapter.py:1081-1093` 已收集 30 个连续体 link body；
- 同文件 `1321-1403` 已有“点/臂半径—球障碍”的 Jacobian 与硬 barrier 行生成；
- 同文件 `4690-4697` 已把 0.016 m 写为连续体实际半径并提供 barrier 参数；
- 但同文件 `6630-6655` 的旧目标卫星避障只是软最小二乘速度，且 `signed_dist` 没有减去臂半径，不能作为防穿模硬保证。应把这里升级为 OBB/胶囊的硬 CBF，而不是继续调大软避障权重。

### 8.7 距离梯度与运动卫星的相对速度

若当前局部最小点 \((i^*,s^*)\) 唯一且处于光滑几何特征上，由 envelope theorem：

\[
\nabla_q d
=n_W^\top J_p(q,s^*)
-\nabla_q r_{i^*}(s^*),
\]

其中

\[
J_p(q,s)=\frac{\partial p_i(s;q)}{\partial q}.
\]

若半径与构型无关，第二项为零。设目标最近点为 \(y^*\)，其世界速度为

\[
v_{T,*}=v_T+\omega_T\times(y^*-c_T).
\]

若 \(G(q)\dot q_{17}\) 是包含自由漂浮基座反作用的全系统广义速度，则距离变化率为

\[
\boxed{
\dot d=
n_W^\top
\left[J_p^{\rm full}(q,s^*)G(q)\dot q_{17}-v_{T,*}\right].
}
\]

这一步非常重要：卫星在平移和自转，因此把它当静态盒体只会约束机械臂的绝对速度，而不是双方的相对闭合速度。

### 8.8 写成 V6-lite 的速度级时变 CBF 行

定义

\[
h=d-d_{\rm safe,total},
\]

并取线性 class-\(\mathcal K\) 函数 \(\alpha(h)=\gamma h\)。时变 CBF 为

\[
\dot h+\gamma h\ge0.
\]

代入上式得到可直接加入 QP 的线性不等式：

\[
\boxed{
n_W^\top J_p^{\rm full}G\,\dot q_{17}
\ge
n_W^\top v_{T,*}
-\gamma\left(d-d_{\rm safe,total}\right).
}
\]

`v6_lite_6` 的 `_clearance_constraints` 对每个 pair 构造

\[
A\dot q_{17}\ge-\gamma(d-d_{\rm safe,pair})-\dot d_T,
\]

其中

\[
A=n_W^\top(J_a-J_b)G,
\qquad
\dot d_T=n_W^\top(J_a-J_b)\dot q_T^{\rm exo}.
\]

`G` 只映射 17 维机器人决策速度，目标自由关节行保持零；`\dot q_T^{\rm exo}` 则只在目标自由关节 6 个速度分量上非零。因此

\[
\dot d=A\dot q_{17}+\dot d_T,
\]

并完整保留目标平移与自转在当前 witness point 上造成的闭合/分离速度。实现对 MuJoCo 混合几何的 `fromto` witness 顺序先按 geom 类型规范化，再映射回 pair 顺序；非零目标平移与旋转的有限差分测试分别锁定了 `\dot d_T` 的符号和数值。

目标 pair 策略也已经落地：V6 opt-in 75 个目标 pair，连续体 62 个、刚性臂 7 个、基座 6 个；仅 `collision_0072` 和 `collision_0073` 享有终端接触豁免，连续体目标 pair 全部保留为硬安全行。

### 8.9 总安全裕度

当前必须区分三个量：统一正式离散验收门槛为 5 mm；在线 `rigid_target` 缓冲为 5 mm；在线 `continuum_target`、`base_target` 与普通 pair 缓冲为 25 mm。若下一阶段用解析 PCC/胶囊代理替换或补充 MuJoCo exact geom，则不应只保留 5 mm 验收值，而应定义

\[
\boxed{
d_{\rm safe,total}
=d_{\rm nominal}
+\varepsilon_{\rm PCC}
+\varepsilon_{\rm radius}
+\varepsilon_{\rm sample}
+\varepsilon_{\rm servo}
+\varepsilon_{\rm time}
+\varepsilon_{\rm state}.
}
\]

- \(d_{\rm nominal}\)：pair 对应的在线名义缓冲（当前连续体/基座目标为 25 mm，刚性目标为 5 mm）；
- \(\varepsilon_{\rm PCC}\)：PCC 曲线与 MuJoCo 全臂几何之间的最大标定残差；
- \(\varepsilon_{\rm radius}\)：圆管包络与真实 STL 横截面的残差；
- \(\varepsilon_{\rm sample}\)：离散采样或弦弧近似误差；
- \(\varepsilon_{\rm servo}\)：50 Hz 命令到 500 Hz 力矩执行的跟踪误差；
- \(\varepsilon_{\rm time}\)：一个控制周期内的最坏相对闭合量；
- \(\varepsilon_{\rm state}\)：目标位姿/twist 与机器人状态估计误差。

这些项应从数据估计，而不是全部拍脑袋指定。可以先用验证集的 99.9% 分位数，再加独立的异常裕度；安全认证场景则应采用观测到的最大值或物理上界。

## 9. V6-lite implementation roadmap

### P0：已在 `v6_lite_6` 实现（复用精确 MuJoCo 距离）

1. V6 通过 opt-in 增加 62 个 `continuum_target`、7 个 `rigid_target` 和 6 个 `base_target` pair；
2. 即使物理 contact response 关闭，仍保留目标 collision geom 的 signed distance 查询；
3. QP 已加入目标已知 6D twist 的 witness-point 外生 drift；
4. 有意接触白名单仅为 `collision_0072/end_link` 与 `collision_0073/end_effector_r`，没有“整颗卫星”排除；
5. 独立 replay 已对包括初始状态的所有原生 500 Hz 状态扫描 62 个连续体—目标 pair，并以 5 mm、低于门槛状态数 0、穿透状态数 0 为硬门禁；整机另做 4×构型细分离散审计。

这个步骤直接约束当前仿真的碰撞网格，不引入尚未标定的 PCC 代理误差。正式五场景数值仍须由当前合同完整重跑并独立验收；本节不预填结果。

每个 50 Hz QP tick 的核心伪代码为：

```text
forward(q, target_pose, target_qvel)
G = reaction_velocity_map(q)
candidates = all_collision_pairs_within_activation_band()
for each candidate:
    d, x_a, x_b, n = signed_distance_and_witnesses(candidate)
    J_a, J_b = point_jacobians(x_a, x_b)
    A_row = n^T (J_a - J_b) G
    d_dot_target = n^T (J_a - J_b) qvel_target_exogenous
    d_safe_pair = 0.005 if class(candidate) == rigid_target else 0.025
    b_row = -gamma * (d - d_safe_pair) - d_dot_target
    append hard constraint A_row * qdot_17 >= b_row
solve QP; if infeasible, enter the implemented zero-velocity safety-stop branch;
never replace the safety rows with an oracle solution
```

这里 `qvel_target_exogenous` 只含目标自由关节的实测 6D 速度；对非目标 pair 该漂移自然为零。目标自由体不在 17 维决策映射中，因此该项必须显式保留在右端，不能以零决策列代替。

### P1：解析 PCC 预测约束（论文/实时优化友好）

1. 标定每段 \(C_i,L_i\)，并实现上面的稳定指数映射；
2. 用 OBB SDF + 有界一维最小化，输出每段若干活动最近点；
3. 用自动微分或解析 Jacobian构造 QP 行；
4. 用 MuJoCo exact geom distance 做离线 ground truth，得到 \(\varepsilon_{\rm PCC}+\varepsilon_{\rm radius}\)；
5. 在线保留 exact MuJoCo monitor 作为 backstop，至少在方法成熟前不要只信解析代理。

### P2：加速与鲁棒化

- broad phase 用段级 AABB/OBB；narrow phase 用弧—OBB 一维最小化；
- 对每段采用自适应 subdivision，曲率大或接近卫星时细分；
- 保留 activation band 内多个约束，避免最近特征切换引起梯度跳变；
- 加入 one-step/多步预测，覆盖 20 ms 任务周期和 2 ms 执行周期；
- QP 不可行时优先降低末端跟踪、姿态跟踪和速度目标，绝不松弛 `continuum_target` 安全行。

## 10. Acceptance tests

### 10.1 几何单元测试

- \(\kappa\to0\) 时 PCC 公式无 NaN/跳变，且回到直线；
- 单段平面弯曲与解析圆弧一致；
- 5 段端点、若干内部点与 MuJoCo FK 的位置/切向误差在阈值内；
- OBB SDF 对盒外面、棱、角和盒内点符号正确；
- 解析/自动微分距离梯度与中心差分一致。

### 10.2 不漏检测试

- 构造卫星只穿过两个采样点之间的 adversarial case，验证胶囊/sagitta 包络仍报警；
- 将解析代理与 62 个连续体 collision geoms 的 `mj_geomDistance` 做大规模随机构型对比；
- 要求代理下界不高估真实 clearance，或把所有高估量纳入 \(\varepsilon_{\rm model}\)。

### 10.3 控制闭环验收

- 已按 `v6_lite_6` 重跑 5 个正式 27 s 场景：67,500 个物理步、6,750 次 QP，QP 失败为 0；
- 独立力矩重放对包括初始状态在内的全部原生 500 Hz 状态扫描 62 个连续体—卫星 pair，并报告最小 signed clearance、低于 5 mm 状态数、穿透状态数、最小 pair 与时刻；
- 另对 50 Hz `task_qpos` 做 4×构型空间细分的 2,927-pair 整机离散审计，并核对 pair policy 数量与 SHA-256；
- 验收条件至少为：原生 500 Hz 连续体—目标最小离散间隙不低于 5 mm、低于门槛/穿透状态均为 0，且整机 4×细分最小离散间隙不低于 5 mm；
- 非零目标平移和绕 witness-point 有效力臂的旋转有限差分测试必须确认 drift 项的符号和数值；
- 单独验证刚性末端可执行预期抓取，而连续体臂仍受保护。

正式结果：整机 4×细分最小 signed clearance 为 14.995557 mm；`continuum_target` 4×细分最小值为 24.995132 mm；原生 500 Hz 专项回放最小值为 24.994694 mm（场景 04、10.200 s、`collision_0033/link_15` 对目标卫星），低于 5 mm 和穿透的状态数均为 0。独立验证 26/26 门禁通过。连续体活动路径 RMSE 最差为 0.159450 mm，任务层/力矩层 p95 分别为 7.809830/0.339205 ms。

## 11. Remarks / Interpretation

1. **能否用几何法？** 能。PCC 使任意弧长点具有解析位置，卫星又是 OBB，因此距离评估天然是低维问题。
2. **是否需要一个“完全闭式”的弧—盒最小距离？** 不需要。每段的一维有界最小化已经足以实时，并比硬写所有面/棱/角分支更可靠。
3. **几何距离是否自动等于安全保证？** 否。只有当距离导数进入时变 CBF/QP、误差有裕度、约束不被松弛时，才能谈模型内的前向不变性。
4. **为何保留 MuJoCo 精确距离？** PCC 对受载/摩擦/离散关节近似存在残差；MuJoCo collision mesh 是当前仿真的直接 ground truth。
5. **为何不能只避开卫星中心？** 卫星是 0.4 m 盒体，最近位置可能在任一面、棱或角；中心距离不能正确表达表面 clearance。

## 12. Boundaries / Non-Claims

- 不声称 PCC 在重力、摩擦、外载和抓取接触下严格等于真实臂形。
- 不声称全局最小距离处处可微；最近特征和最近段切换处只分段可微。
- 不声称有限离散采样天然安全；必须加 Lipschitz/sagitta 证书或执行连续最小化。
- 不声称 2012/2016 论文的运行时间可以直接证明当前 Python/MuJoCo 实现满足 20 ms；必须在本机 profiling。
- 不声称 2026 年 arXiv 预印本已经同行评审。
- 不把“关闭物理接触响应”等价为“没有碰撞”；安全结论来自显式 signed-distance replay。
- 不把 500 Hz 离散状态全部通过表述为连续时间证书：2 ms 采样之间仍未使用 continuous collision detection；4×整机细分同样只是更密的离散审计。

## 13. Open Risks

1. 当前 10 维变量的两个弯曲方向与世界/段局部轴的符号尚未形成机器可检验的标定契约。
2. 绘图直径与 collision STL 横截面疑似不一致；若半径取小会产生伪安全。
3. 动态目标 twist 外生漂移已实现，并由非零平移/旋转有限差分测试锁定；剩余风险是目标位姿/twist 的测量误差、时延以及仿真到硬件的模型不确定性尚未形成保守上界。
4. 初始状态若已在不安全集内，一阶 CBF 不能追溯地保证未曾碰撞；必须先生成安全初值或做恢复控制。
5. 当前允许接触集合已精确定义为 `collision_0072/end_link` 与 `collision_0073/end_effector_r`；URDF/geom 命名或抓取机构改变后必须重新审计白名单和 pair-policy hash，避免例外范围漂移。

## 14. Literature map and source verification

检索日期：2026-09-21。检索式包括：

- `constant curvature continuum robot kinematic modeling`
- `uniform-curvature continuum manipulator collision detection minimum distance`
- `continuum robot whole body obstacle distance Jacobian`
- `continuum robot control barrier function collision avoidance`
- `time-varying CBF dynamic obstacle signed distance`

纳入标准：原始论文、出版社/IEEE/arXiv/作者机构全文；直接支持 PCC 曲线、有限半径整臂距离、距离 Jacobian、CBF-QP 或运动障碍之一。排除：博客/二手摘要、只约束末端且无整臂几何定义、无法核对元数据的结果。同行评审论文与预印本分开标注。

| 文献 | 与本问题的直接贡献 | 不能据此声称的内容 | 状态 |
|---|---|---|---|
| Webster & Jones, 2010, *IJRR*, DOI 10.1177/0278364910368147 | PCC 单段中心线/齐次变换、多段连乘、微分运动学；明确 PCC 的载荷/摩擦等局限 | 未给臂—障碍距离或安全控制 | 同行评审，核心运动学依据 |
| Jones & Walker, 2006, *IEEE T-RO*, DOI 10.1109/TRO.2005.861458 | 多段运动学、执行器—形状—任务空间映射；二维正交曲率避免 \(\kappa=0\) 参数奇异 | 映射仍是机器人特定的 | 同行评审，核心参数化依据 |
| Li & Xiao, 2016, *Robotica*, DOI 10.1017/S0263574714002458 | 多段均匀曲率臂建模为截断圆环/圆柱，对多边形物体做精确碰撞和最小距离 | 无 CBF/QP、无运动障碍、未给处处光滑 signed penetration depth | 同行评审，与本问题几何最直接 |
| Ataka et al., 2016, *IROS*, DOI 10.1109/IROS.2016.7759438 | 多段连续体全臂形状估计、最近点 Jacobian、动态环境实时避障 | 离散点可能漏检；势场不提供前向不变性保证 | 同行评审，实时全臂基线 |
| Patterson et al., 2024, *RoboSoft*, DOI 10.1109/RoboSoft60065.2024.10522000 | PCC 状态经过链式 Jacobian 进入 CBF-QP，给出软—刚机器人自接触安全实验 | 对象不是卫星；是 HOCBF/力矩层而非当前速度层 | 同行评审，PCC-CBF 直接旁证 |
| Hachen et al., 2025, *IEEE RA-L*, DOI 10.1109/LRA.2025.3532159 | 用名义 PCC 输出全臂 spacer disk 位置，并把到 3D mesh 的 signed distance 作为 MPC 硬约束；讨论模型误差与裕度 | 仍是离散 disk、静态安全网格；不是运动卫星 CBF | 同行评审，全臂硬距离约束直接旁证 |
| Dai et al., 2025, *Robotics and Autonomous Systems*, DOI 10.1016/j.robot.2025.105182 | 动态凸障碍的可微时变 CBF，显式包含 \(\partial_t h\) | 非连续体机器人、距离度量不同 | 同行评审，运动卫星 drift 依据 |
| Ames et al., 2017, *IEEE TAC*, DOI 10.1109/TAC.2016.2638961 | CBF-QP 的安全集前向不变性与最小修改控制理论 | 不提供具体机器人几何距离 | 同行评审，控制理论依据 |
| Wong et al., 2026, arXiv:2603.19424 | 连续体全身密集球离散、球—障碍 clearance、保守 log-sum-exp、CLF-CBF | 静态球障碍、准静态模型；截至检索日是预印本 | 预印本，只作最新方法参考 |

## 15. References

1. R. J. Webster III and B. A. Jones, “Design and Kinematic Modeling of Constant Curvature Continuum Robots: A Review,” *IJRR*, 2010. [Publisher](https://journals.sagepub.com/doi/10.1177/0278364910368147), [author draft](https://static1.squarespace.com/static/661dc8f29cd91731ed8c0c8c/t/66abaaf668448d20f99795b0/1722526457028/WebsterDesignIJRR10_draft.pdf).
2. B. A. Jones and I. D. Walker, “Kinematics for Multisection Continuum Robots,” *IEEE T-RO*, 2006. [IEEE](https://ieeexplore.ieee.org/document/1588999/), DOI: 10.1109/TRO.2005.861458.
3. J. Li and J. Xiao, “An Efficient Algorithm for Real Time Collision Detection Involving a Continuum Manipulator with Multiple Uniform-Curvature Sections,” *Robotica*, 2016. [Cambridge Core](https://www.cambridge.org/core/journals/robotica/article/abs/an-efficient-algorithm-for-real-time-collision-detection-involving-a-continuum-manipulator-with-multiple-uniformcurvature-sections/4E9EDDAFD7A0D31479255D86E7BB8BFD), DOI: 10.1017/S0263574714002458.
4. A. Ataka et al., “Real-time Pose Estimation and Obstacle Avoidance for Multi-segment Continuum Manipulator in Dynamic Environments,” *IROS*, 2016. [UCL record/full text](https://discovery.ucl.ac.uk/id/eprint/1518015/), DOI: 10.1109/IROS.2016.7759438.
5. Z. J. Patterson et al., “Safe Control for Soft-Rigid Robots with Self-Contact Using Control Barrier Functions,” *RoboSoft*, 2024. [IEEE DOI](https://doi.org/10.1109/RoboSoft60065.2024.10522000), [full text](https://arxiv.org/html/2311.03189).
6. S. Hachen et al., “A Non-Linear Model Predictive Task-Space Controller Satisfying Shape Constraints for Tendon-Driven Continuum Robots,” *IEEE RA-L*, 2025. [IEEE](https://ieeexplore.ieee.org/document/10847881/), [arXiv](https://arxiv.org/abs/2409.09970), DOI: 10.1109/LRA.2025.3532159.
7. B. Dai et al., “Differentiable Optimization Based Time-Varying Control Barrier Functions for Dynamic Obstacle Avoidance,” *Robotics and Autonomous Systems*, 2025. [DOI](https://doi.org/10.1016/j.robot.2025.105182), [full text](https://arxiv.org/html/2309.17226).
8. A. D. Ames et al., “Control Barrier Function Based Quadratic Programs for Safety Critical Systems,” *IEEE TAC*, 2017. [Caltech repository](https://authors.library.caltech.edu/records/jnhr0-1ww05), DOI: 10.1109/TAC.2016.2638961.
9. K. Wong et al., “A Closed-Form CLF-CBF Controller for Whole-Body Continuum Soft Robot Collision Avoidance,” arXiv:2603.19424, 2026. [arXiv](https://arxiv.org/abs/2603.19424). **Preprint; not used as the sole basis of any safety claim.**
