# 有限结构实验 C：条件经验原型混合与解析 score diffusion

2026-10-02 预声明：代码与有限验证获授权；真实 fit、资格和实际执行须在源码就绪后由根代理启动。B 已按原停止规则结束：TRAIN raw 0/48，ddq 48/48，终端失败 86 项、航点失败 4 项，VAL/TEST/physics 0。C 是新方法，B 与所有历史结果保留。

## 模型与固定预算

方法名 `teacher_prior_empirical_score_diffusion`。复用原 13 条严格成功 TRAIN 标签与六个已保存 B 先验中心 P，不重做 TRAIN 中心。全部标签保留，包括对应几何失败中心的标签。只有 TRAIN 拟合原 517 条件正常化及 17 维残差 mean/scale（floor 1e-6）。推理条件为 normalized517 与 physical510 P；Task id/hash 仅作来源身份，不作推理查表键。

自由 CP 残差 R=C_free−P_free。在任何真实 fit 或资格开始前固定条件均值结构：六个唯一 TRAIN group 的条件 F=normalized517+physicalP510、组正常化残差均值 T，进行一次 float64 mean-centered 最小范数仿射拟合 A=pinv(F_center,rcond=1e-12)T_center。baseR(c)=Tmean+(F(c)−Fmean)A；13 个 route offset 为 R_m−T_g(m)，查询 component mean 为 mu_m(c)=baseR(c)+offset_m。因此推理支持为条件平移的有限 TRAIN route 模态，不是按推理 Task id 复制原参考。保存 A、均值、offset、rank 与 TRAIN 拟合残差；不以这些统计选择超参数。

真实训练六 TRAIN group 的线性 softmax 条件权重：CPU float64、seed71、Adam lr0.01、betas(0.9,0.999)、无 weight decay、固定 2000 steps、full batch13、无 early stop 或 VAL/TEST checkpoint 选择。组内路线权重固定均匀；只学习组权重，不声称学到了组内路线偏好。checkpoint 独立命名，保存条件权重、仿射均值、route offset、正常化器、来源和配置。训练成本是一次解析最小范数条件均值拟合及 2000 步 CE 条件权重训练，不伪称 DDPM 噪声训练。

对 alpha=abar_t，正向分布为 sum_m pi_m(c) Normal(sqrt(alpha) mu_m(c),(1-alpha)I)。posterior log weight 为 log(pi_m)−||x−sqrt(alpha)mu_m(c)||²/[2(1-alpha)]，用稳定 softmax；x0hat=sum_m posterior_m mu_m(c)，epsilon_hat=(x−sqrt(alpha)x0hat)/sqrt(1-alpha)。固定 cosine100、DDIM20、eta0、seed64..71；末步 next_alpha=1 仍用同一 posterior weighted mean，禁止 argmax、nearest-prototype 复制、输出吸附或退回 P。

生成结果唯一为 P+denormalize(Rhat)，再按原 codec 固定 C0/C1。无裁剪、投影、IK、repair、fallback 或学习力矩。保存逆扩散步骤与责任度证据。DDIM20 和标准高斯启动是有限步近似；末步可能仍在原型之间，不能保证离散模态或目标权重的精确采样。

## 资格、实际执行与停止

源码就绪后只 fit 一次。原六 TRAIN×8 候选，K1 固定 c0、K8 按原 static_score/index；资格为原 full raw gate 且独立原 ddq guard（2.5/4、dense1351）。TRAIN 0/48 就 STOP：不进入 VAL/TEST/physics，不追加 fit 或采样。否则固定原三 VAL×8 一轮，先验采用 B 同算法、同通用 SC04 来源、同有限预算；VAL 0/24 就 STOP。

有资格的真实候选按原 Task 顺序第一组、原 static_score/index 交接，源/Task/proposal/reference/checkpoint/gate/ddq 身份核对后，最多一次原 27 s QP/PCC/67 torque 执行。真实失败保存，不重选、不重复；TRAIN/VAL 实际成功不是后续阶段资格条件。进入正式 TEST 前冻结模型、来源、先验和规则；原三 TEST 全部保留 K1/K8 分母，所有独特所选候选至多一次 actual，相同 proposal 可带完整来源复用真实成功或失败。prior_only 是物理 R=0、精确 P，独立对照及成本；不称八条新路线。不得读取当前 VAL/TEST 未来 actual。

保持原 Task、工作域、安全距离、PCC、执行器、容差和 20 ms/2 ms 仿真合同；墙钟 20 ms 仍为非阻塞性能指标。若 C 仍不能入场，转向少量新增真实 TRAIN groups 与成功路线；不再在同结构上追加迭代或审计。

## 可声明范围

旧 Transformer 负结果不改写。C 最多检验 Teacher 先验支持下的有限路线复用、条件均值平移、条件加权与真实执行；13 个 route offset、条件平移、原型接近程度及先验成本均显式报告。不得宣称新拓扑、网络独立收益或硬实时/鲁棒性/连续安全。原最小交付仍须独立学习参考真正进入物理控制链；零物理拒绝不标 complete。

方法参考：[DDIM](https://arxiv.org/abs/2010.02502)；[条件混合解析 score 方法](https://arxiv.org/abs/2506.18227)。这些参考不证明本项目精度、采样质量或任务成功。
