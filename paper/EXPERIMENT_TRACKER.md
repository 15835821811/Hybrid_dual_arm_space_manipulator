# Experiment Tracker：有界主干对照v2

唯一有效run：artifacts/v7_1/backbone_pair_smoke_002，ff_prior_backbone_v2。原001启动文件写入失败保留，未进行正式更新或DEV。

|阶段|目的与范围|状态|证据|
|---|---|---|---|
|启动修复|001证据父目录缺失，0次正式更新|FIXED_IN_V2|原18文件保存hash，源码434f42ee不修改；独立v2父目录写入回归|
|M0|14222TRAIN/48VAL，28861绑定、168源码，数值检查|PASS：软件及实际临时数值检查|14/14测试通过；与原冻结输入/协议一致，未扩大实验|
|M1|从头MLP208998/Transformer211969参数各6000更新|RUNNING|观察到MLP 1800/6000；目标仍为相同codec/v+.05localG/随机序列、全部TRAIN消费|
|M2|各VAL48×2种子完整16步无引导，共192|PENDING|结果不作为DEV拦截门|
|M3|原DEV12×3种子×5候选×2新模型|PENDING|新增360/72，旧简单基线复用360/72；原物理门不变|
|M4|只读原始审计、新独立review、报告提交|PENDING|完成resume仅一次；无优势也如实收尾|

C1终点精度与C2完整收益未判定。仅一个训练种子，不宣称普遍稳定优势。不扩库、不启动pilot/study/strict_multimodal/TEST；保留全部旧结果和用户修改。

实际启动：源码4ccba221，PID 40916，状态RUNNING，阶段M1_training，心跳健康；接续mlp-transformer每15分钟只读状态，完成/失败再收尾。正式结果尚未产生，不预先通过。
