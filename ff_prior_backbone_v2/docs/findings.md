# 本轮结果与主张

backbone_pair_smoke_002完成，冻结源码4ccba221，耗时2741.157秒。
两个主干各6000正式更新、14222TRAIN全部参与、300随机序列摘要相同。
无引导各0/96，Transformer位置均值19.0167→21.0883mm、尾部恶化；C1=no。
原冻结DEV两组各180/180与36/36；Transformer相对新MLP执行位置均值
0.7872→0.6593mm，下降16.24%；成功率持平，条件均值平均精度更优；C2=partial。
新增360候选/72执行；旧简单基线360/72仅复用。工程PASS。
单训练种子、共同物理引导、已用开发任务，不证明普遍优势。
完整replay第二轨迹未保存；审查核残差/源码/native_trace，未追加物理仿真。
一次完成resume返回NO_WORK_ALREADY_COMPLETE，956文件清单/hash不变。
保留旧默认MLP与Transformer实验候选；无扩库或TEST。长期科学目标未完成。
新review见paper/review-traces/experiment-result-to-claim/20261020_backbone_run01。
paper claim audit不可用，下游论文写作为provisional。
