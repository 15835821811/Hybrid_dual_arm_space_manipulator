# 参考源码补充核对：终点边界与实验范围

2026-10-20，运行期间只读核对，不改变已冻结的训练、场景、种子、预算或验收。

## 终点不是同一种约束

参考`src/htdcr_planning/interface/guided_sampler.py`的`apply_boundary_conditions`实际将索引0、1、2、3置为start_config，将-4、-3、-2、-1置为end_config。注释写“前三/后三”，但执行语句是**四个**，本分析以语句为准。

DDIM路径在初始化、每步前后及最终返回前调用边界约束（约413、429、453、471行）。因此给定起终关节构型被直接保持，不是全靠神经网络学习精准命中终点。关闭代价引导也仍保留这项边界操作。

本项目冻结条件是双末端世界位姿；自由漂浮基座的位姿随整条关节路径演化。固定关节终点一般不能独立确定最终世界末端位姿。本项目将前三控制点固定为初态、末三重复预测终态以保持零端点速度/加速度，但预测终态仍须通过耦合传播对齐世界目标。这是任务与表示的实质区别。

故本轮完整采样世界监督是针对本项目条件映射的独立改进，并非“原样复制参考训练损失”。同样，参考项目的终端关节容差不能直接解释为本项目无引导5mm/1°世界目标通过率。无引导网络指标与完整引导规划工程可用性继续分别报告。

## 修正敏感性实验描述

参考`configs/experiments/sensitivity.yaml`明确扫描：

- 候选数1、2、5、10；
- DDIM步数40、60、80、100；
- B样条控制点数10、15、20、25、30；
- 固定五障碍场景。

三/五障碍是主对照和消融使用的场景设置，**不是该敏感性配置的扫描变量**。本轮两障碍密度仅属于应用场景对照，不能称完整复现上游敏感性扫描。

参考`ablation.yaml`包括full、without_actuator、without_collision、diffusion_only四种。本轮只有同任务关闭全部物理引导这一消融，没有分别关闭碰撞与执行器引导，不应写成已复现全部消融矩阵。

因此本轮落实的是有界的数据复用→条件扩散训练→采样引导→相同场景基线比较→关闭引导诊断→独立物理执行重放闭环。完整上游超参数网格和不同连续体节数扩展性未运行，也不适合在保持本机17→67及同候选预算的本轮临时加入。

这一补充纠正初版REFERENCE_ALIGNMENT.md及EXPERIMENT_PLAN.md中可能将“三/五障碍对照”称为参考敏感性的措辞，保留初始版本供追溯，不改变任何实验内容或通过门。

源码：

- [采样边界](https://github.com/zhanglooong/htdcr-actuator-aware-diffusion-planning/blob/900d781ba63f0123dff95e49a76d51cb49a74803/src/htdcr_planning/interface/guided_sampler.py)
- [敏感性配置](https://github.com/zhanglooong/htdcr-actuator-aware-diffusion-planning/blob/900d781ba63f0123dff95e49a76d51cb49a74803/configs/experiments/sensitivity.yaml)
- [消融配置](https://github.com/zhanglooong/htdcr-actuator-aware-diffusion-planning/blob/900d781ba63f0123dff95e49a76d51cb49a74803/configs/experiments/ablation.yaml)
