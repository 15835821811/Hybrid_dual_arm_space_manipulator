# V6.2-B.2 固定区间安全函数与导数

本阶段只定义并验证区间几何接口。`pcc_interval_cbf.py` 不进入控制器导入链，在线动作仍由 A.1 控制器生成。

## 安全函数

五个 PCC 段各自使用材料弧长二叉分区。区间 ID 为 `pcc_interval:<segment_id>:<binary_path>`；根区间以 `root` 标识。叶区间不得缺失、重叠或跨段。`IntervalPartition.coverage()` 逐段检查完整覆盖，细分只替换一个叶区间为两个子区间；合并仅在两个子区间均为叶时有效。

对固定区间 $I_j=[a_j,b_j]$ 和其中点 $c_j$，定义

$$
\underline d_j=\operatorname{sd}_{\rm OBB}(p_i(c_j))-r_i-\frac{b_j-a_j}{2}-\varepsilon_{\rm num},
\qquad h_j=\underline d_j-d_{\rm safe}.
$$

这里 $r_i$ 是 V6.1-A 已冻结的第 $i$ 段 PCC 管体半径，$d_{\rm safe}=0.005\,\mathrm m$，$\varepsilon_{\rm num}=10^{-9}\,\mathrm m$。模型的段内材料弧长切向量为 $R(s)e_x$，精确模型下其范数为 1；点到 OBB 的有符号距离对点位置为 1-Lipschitz。因此每个区间的表达式在声明的精确模型中是该区间连续净空的下界。所有叶区间完整覆盖五段，且全部 $h_j\ge0$ 时，PCC 管体代理的全弧长净空达到门槛。

当前双精度计算及 1 nm 外扩没有经过形式化区间算术认证。`interval_well_formed`、`coverage_complete`、`analytic_bound_assumptions_satisfied` 和 `floating_point_certification` 分开报告。实际 60 维链是否位于声明的 10 维形状子空间，以及经验半径对真实几何的包络证据，也单独报告；代理下界不能被直接描述为真实机器人全域安全。

## 同一函数的导数

固定分区、半径、覆盖项和数值余量后，在 OBB 有符号距离可微处，使用该区间**自身中点**和 OBB witness point 得到

$$
\nabla_{q_c}h_j=n_j^{\mathsf T}J_p(c_j),
\qquad
A_j=n_j^{\mathsf T}J_{\rm base\ point}(c_j)G
+[n_j^{\mathsf T}J_p(c_j),0_{1\times7}],
$$

$$
b_j=-n_j^{\mathsf T}J_{\rm target\ point}(w_j)\dot q_T^{\rm exo},
\qquad \dot h_j=A_ju+b_j.
$$

$G$ 来自当前 MuJoCo 质量矩阵的零基座动量反作用映射。目标平移和旋转均通过目标 witness point 的 Jacobian 进入 $b_j$。OBB 内部最近面并列或法向不唯一时返回 `NONSMOOTH_OBB_INTERIOR_FEATURE`，不给出可用于控制的梯度。分支搜索、排序和 `nextafter` 过程不参与此固定函数的导数。

`test_b2_interval_cbf.py` 在冻结分区下检查 10 维形变梯度、完整 17 维方向导数和目标 6D 漂移；误差门槛为绝对 $2\times10^{-5}$ 加相对 $2\times10^{-3}$（内部形变坐标逐项同量级）。测试还覆盖区间细分与合并、缺失及重复区间、多区间和不可微内部特征。下一阶段才在保存的闭环状态上做影子评估与预算测量。
