# Research Findings

## V6.4-B.1 — architecture_pilot_20261006_01

判定：主效果 `no`，confidence `high`（仅这次冻结结果）。有限实施及交付完成，默认M0，结束本轮架构假设。完整细节保存在EXPERIMENT_RESULT_TO_CLAIM.md及v6_4/ARCHITECTURE_PILOT.md。

两模型真实训练6000更新/192000参考曝光；同信息、共享损失和配对噪声/latent，没有TEST权重选择。raw综合通过0/24对0/24，完整任务0/6对0/6。M1速度项24/24对M0的0/24、位置误差下降及3个工程K1获准进入actual，仅支持组件改善。三个已执行prefix总31180个力矩步，均在私有预演工作域拒绝后终止；不计完整成功。

没有新完整轨迹供原独立全证据验收。原raw几何short-circuit与prefix文件名匹配缺口均记录为NOT_RUN；保存数组描述不替代独立Task、合同、区间或geometry检查。旧A.1的2/4保留。

训练loss下降、VAL后期变差表明本次拟合与验证表现不同。仅6源任务13成功参考、每模型1seed、6新TEST，数据覆盖可能限制迁移；这是解释假设，未经本轮分离验证。新TEST可行性未预先独立建立。不能将主负结果唯一归因于某个模块，也不能否定整个Diffusion方向。

后续约束：不按TEST调repair，不扩大层数/骨干/控制点维度，不增候选或训练预算寻找正结果，不绕过原门禁，不把两个0/6当总体非劣证明。M0保留作为公平基线，不是本轮任务能力认证。

墙钟20ms不作当前仿真研究前置门槛；20ms仿真控制周期与2ms物理保持。实际50Hz部署需另行运行时验证，当前NOT_MET。有限实验结果已足以结束本轮判定，不追加C.1或控制器修补。

Review route: end hypothesis / preserve baseline; next_experiments_needed=[]。无PAPER_CLAIM_AUDIT，后续论文措辞provisional。有限交叉复核角色及输入/输出哈希已记录于paper/review-traces/experiment-result-to-claim/2026-10-06_run01。
