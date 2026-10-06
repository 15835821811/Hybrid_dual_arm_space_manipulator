# V6.4-B.1 配对架构先导实验

两模型均已真实训练。M0 与 M1 的 raw 综合通过均为 0/24，冻结工程闭环完整成功均为 0/6。本轮未建立 M1 的主效果，默认保留 M0，结束本次架构假设，不扩大训练、候选、任务或修复预算。M0 在本测试集同样没有完整成功，保留基线不表示其已具备任务能力。

## 冻结身份

- 分支：`v6.4-b1-geometry-conditioned-diffusion`。
- 实验源码：`efc40f431ffb479268503e21cc9e4f2546f3eb2d`；派生 checkout：`d886480a689aa36dd5fda784624ae283489791bf`。
- A.1 执行算法来源：`9fe36eba65b383741d2a58dc94c72d7f82e7e230`。历史控制源码逐字节核对；最终配置 SHA256：`25a8f4aa060fcc3bc1f8aa6e5c791541d1f30a92b32565b8868f24e7da9f291e`。
- 工作目录：`E:/v64b1work_20261006_01`。
- 交付目录：`E:/v64b1work_20261006_01/v6_4/output/architecture_pilot_20261006_01`，下文称 OUT。

实验报告和权重在 OUT，最终 manifest 记录实际实验 producer 和随后仅添加交付文档的 Git HEAD。没有自动 push。旧 A.1 的 2/4 开发回归及旧失败结果保留。

## 共同路径与真实差异

`reference_training_data.py` 从已成功实际消费的原参考控制点建立可追溯标签，保持 source TaskSpec/split；B.1 experiment_split 独立按源任务分组。13 条参考来自 6 个源任务，3 TRAIN 任务/7 参考、3 VAL 任务/6 参考；另有 9 条失败或未执行的 proposal-only，全部排除。控制与条件 scaler 只拟合 TRAIN，没有将实际 q 自动作为理想参考，也没有使用 D0–D3 或未来 actual trace。

这里的“原参考”指旧运行实际消费的 `selected_reference.npz` 控制点，生成时已有 teacher 锚点优化。每条 `reference_provenance.json` 保留 teacher planning prior 可能具有 actual-q-fit 祖先的说明；本轮标签本身不是实际执行状态拟合，选中参考另有成功消费证据。不能据此将 teacher 构造过程称为未经优化的 raw 提案。

`typed_condition_encoder.py` 统一初始基座参考系中的全局状态、任务、障碍及初态解析臂形。M0 保留现有 `ConditionalDenoiser` 主体，展开同一信息为 2781 维，包含 mask/type。原 M0 已有控制点 self-attention；本轮不将它冒称无结构网络。M1 的 `conditioned_controlpoint_denoiser.py` 使用 32 控制点 token、4×128 维/4-head block、FFN4、带类型条件 cross-attention 和逐层全局/扩散步调制。固定 C0/C1 是边界 token，只有 30×17 自由控制点参与扩散。

整个条件化方案共同构成被比较的变化；本实验不能单独归因于 cross-attention、调制或边界 token。

`differentiable_spline_loss.py` 共享 v 预测及 `Lv+0.1 Lq+0.01 Ldq`，辅助损失仅在 alpha_bar≥0.1 的样本上启用。反归一化和 136 个物理时间网格的 B 样条 q/dq 均在 torch 内保留梯度；此损失在统一末段进度修复之前计算。原 codec、cosine100、DDIM20、内部 `epsilon_residual` 命名对应的 v 约定及未裁剪采样一致。

`architecture_pilot_training.py` 给两模型相同 AdamW、batch32、学习率1e-4、weight decay0.01、gradient norm clip1、6000 optimizer updates、每组192000参考样本曝光。两组每一步的数据/时间/噪声抽样协议相同，draw SHA256 为 `4a3803c8a135c96577e3bef22cde976a68a7a6bb22eefec813be0f3e09d871f9`。每250步在固定 VAL 上评估，分别选择 M0 update250 和 M1 update500；不按 TEST 选权重。

## 有限结果

训练/采样使用现有 RTX4080 与 torch2.6.0+cu126；raw、repair、actual 使用现有 CPU torch2.6 环境。两环境 MuJoCo3.3.2、numpy1.26.4、scipy1.11.2 一致。训练分别87.9845s/143.2299s，参数量1207441/1530513；训练拟合改善而 VAL 后期变差，不能将 loss 下降等同任务成功。

6 个新 TEST 在训练前冻结，3 类任务各2个。每任务配对 latent seeds `2026100611..2026100614`，共48计分 slots；每模型24个独特提案，K1永远是index0，K4包含同一4个。raw 无新增优化、裁剪、末段修复或 fallback。

| 计分结果 | M0 | M1 |
|---|---:|---:|
| raw 综合通过 /24 | 0 | 0 |
| raw 独立 Task 点通过 /24 | 0 | 0 |
| raw 速度通过 /24 | 0 | 24 |
| 主 raw 几何 NOT_RUN /24 | 24 | 24 |
| K1 或 any-of-K4 raw 通过 /6 | 0 | 0 |
| 工程 K1 repair/门禁拒绝 /6 | 6 | 3 |
| 进入 actual /6 | 0 | 3 |
| 完整27s成功 /6 | 0 | 0 |
| actual 力矩物理步 | 0 | 31180 |

M1 的位置误差描述统计下降、速度检查改善、更多工程提案获准执行，仅支持本轮组件变化。原 raw gate 在 Task/速度拒绝后不查询几何，NOT_RUN 不能写成通过或碰撞失败。额外2个不计分条件输出、3次1351状态的独立几何影子诊断另列；条件变化影响输出、障碍 token 排序测试通过，但诊断几何不通过，不能据此称学会绕障。

工程 K1 使用同一冻结 v2 repair、A.1 末段进度修复和控制器。M1 三条实际执行分别在23.340s/19.220s/19.800s，因 `RAMP_MICROSTATE_OUTSIDE_DECLARED_DOMAIN` 拒绝下一伺服步；保存11670/9610/9900步的 consumed prefix。该拒绝针对私有预演微状态，不等于实际发生域外执行。冻结 runner 未匹配 `_interval_partial_trace.npz` 文件名，其自动独立 prefix 评价为 NOT_RUN；本轮保留这个报告缺口及原始前缀，不补写通过，不计完整成功。pure-array prefix 描述统计（若字段可用）另列，不能替代独立任务/区间/几何验收。raw 和 K4 闭环均为 NOT_RUN。

## 命令与交付

OUT 的 `commands/*.json` 保存实际 executable、完整 argv、cwd、起止/耗时、退出码、producer 和前后源码 SHA；`commands/*.log` 保留输出。bootstrap 的早期失败也单列保留。核心执行顺序为：

```text
v6_4.reference_training_data             冻结数据与≤8条表示检查
v6_4.architecture_pilot_training prepare 冻结共享训练配置
v6_4.architecture_pilot_training smoke   各50步诊断，权重弃用
v6_4.architecture_pilot_evaluation freeze-candidates
v6_4.architecture_pilot_training train --model M0
v6_4.architecture_pilot_training train --model M1
v6_4.architecture_pilot_evaluation sample --device cuda
v6_4.architecture_pilot_evaluation raw
v6_4.architecture_pilot_evaluation actual --only MODEL:TASK （固定12项）
v6_4.architecture_pilot_evaluation diagnostics --device cuda
```

入口需使用各 receipt 中的完整冻结配置参数。保留的目录禁止覆盖或将未完成 actual 自动重跑；复现必须使用新的输出目录和同样的数据/配置身份。源码更改时 checkpoint/source guard 会拒绝继续冒用本轮身份。29项针对标签/条件/网络/可微损失/候选计分的测试已通过；未为了这次报告追加无关仓库测试。

OUT 含 `architecture_and_data_identity.json`、dataset/normalizers、训练和候选配置、两组 `checkpoint.pt`/`last_checkpoint.pt`/`optimizer_state.pt`、曲线、48 raw 提案、12工程尝试、拒绝前缀、条件诊断、两张对照表、独立结果判定和 `manifest.json`/`verification.json`。review trace 位于 `paper/review-traces/experiment-result-to-claim/2026-10-06_run01`。

## 20 ms 补充的边界

本目标明确采用 `research_simulation`，仿真控制周期20ms、物理步长2ms保持。墙钟20ms是观测，不是研究前置门槛；部署仍为 NOT_MET。补充附件针对历史 `0e76aba` 的 C.1 建议，当前继承执行层已有命令身份、有效期和拒绝机制，本轮不重复历史 C.1 接入或性能项目。

若最终目标仅是方法的仿真验证，已有功能基线可继续使用，墙钟超限不会自动否定其功能结果。若目标转为真实50Hz部署，就必须另行验证按时下发与迟到动作有效性；有限分位数通过仍不能推出硬实时。本轮没有硬件、备份恢复、鲁棒性或连续时间安全结论。

单训练seed、6 TEST 与极少源任务使泛化仍为 DATA_LIMITED。负结果只否定这次冻结试验中主效果的成立，不证明整个结构化条件或Diffusion研究方向永远无效，也不授权继续扩预算寻找正结果。
