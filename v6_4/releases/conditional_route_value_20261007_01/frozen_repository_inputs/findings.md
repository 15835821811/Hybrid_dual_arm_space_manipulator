# Research Findings

## V6.4-B.1 — architecture_pilot_20261006_01

判定：主效果 `no`，confidence `high`（仅这次冻结结果）。有限实施及交付完成，默认M0，结束本轮架构假设。完整细节保存在EXPERIMENT_RESULT_TO_CLAIM.md及v6_4/ARCHITECTURE_PILOT.md。

两模型真实训练6000更新/192000参考曝光；同信息、共享损失和配对噪声/latent，没有TEST权重选择。raw综合通过0/24对0/24，完整任务0/6对0/6。M1速度项24/24对M0的0/24、位置误差下降及3个工程K1获准进入actual，仅支持组件改善。三个已执行prefix总31180个力矩步，均在私有预演工作域拒绝后终止；不计完整成功。

没有新完整轨迹供原独立全证据验收。原raw几何short-circuit与prefix文件名匹配缺口均记录为NOT_RUN；保存数组描述不替代独立Task、合同、区间或geometry检查。旧A.1的2/4保留。

训练loss下降、VAL后期变差表明本次拟合与验证表现不同。仅6源任务13成功参考、每模型1seed、6新TEST，数据覆盖可能限制迁移；这是解释假设，未经本轮分离验证。新TEST可行性未预先独立建立。不能将主负结果唯一归因于某个模块，也不能否定整个Diffusion方向。

后续约束：不按TEST调repair，不扩大层数/骨干/控制点维度，不增候选或训练预算寻找正结果，不绕过原门禁，不把两个0/6当总体非劣证明。M0保留作为公平基线，不是本轮任务能力认证。

墙钟20ms不作当前仿真研究前置门槛；20ms仿真控制周期与2ms物理保持。实际50Hz部署需另行运行时验证，当前NOT_MET。有限实验结果已足以结束本轮判定，不追加C.1或控制器修补。

Review route: end hypothesis / preserve baseline; next_experiments_needed=[]。无PAPER_CLAIM_AUDIT，后续论文措辞provisional。有限交叉复核角色及输入/输出哈希已记录于paper/review-traces/experiment-result-to-claim/2026-10-06_run01。

## V6.4-B.2 — task_anchored_residual_20261007_01

独立result-to-claim判定partial、confidence high。有限研发交付完成，非零Cartesian残差表示能力在冻结补充TaskSpec及声明离散安全范围内建立；学习收益not_established。TEST E0=4/4、E1=4/4、E2固定K1=3/4，K4 actual NOT_RUN。保留基础Cartesian与TRAIN-only检索，E2仅研究产物，停止本轮，不扩网络、seed、候选或actual。

固定37次实际尝试488260步，35完整成功、30非零完整成功（非独立任务数）；teacher23/24，TRAIN17参考/6任务、VAL6参考/2任务。真实新训练4000更新/128000曝光，selected250/8000，VAL训练曝光0；raw16有限非零、合法14，2幅值拒绝保留。两个执行失败为teacher_17 14.900s及TEST_01_E2 16.620s的原工作域拒绝，无fallback/重试。

全部30条完整非零运行仍保留历史runtime continuum_irregular_waypoint_path_rmse失败；本轮完整成功仅指预声明允许中间绕行的end_effector_detour TaskSpec以及原合同/区间/native/binding，不改写旧严格全曲线路径协议或B1原0/24、0/6与A1修复2/4。锚点接受主要来自解析表示，实际路径改变不等于学习质量优势。

墙钟20ms不作本研究门禁；原20ms规划、2ms物理和27s时长保持，全部长尾记录。部署NOT_MET，真实计算延迟下状态演化的执行有效性未验证，非连续时间安全证明。旧B1封存4849产物/13external字节复核无差异。

路由narrow_claim_and_stop；next_experiments_needed=[]。无paper claim audit，后续论文措辞provisional。完整审阅位于paper/review-traces/experiment-result-to-claim/2026-10-07_run01。报告与机器表在v6_4/output/task_anchored_residual_20261007_01，保持全部原始失败及checkpoint。

## V6.4-B.3 — conditional_route_value_20261007_01

独立result-to-claim判定no、confidence high、blocking_issues=[]。一母场景两任务z0/±12mm固定六槽均完整27s并通过原五项独立门禁。较好非零方向随障碍换边反转（c+ z-，c- z+），相对零干预减少2.791%/3.813%，相对反向5.710%/5.721%；绝对差通过0.001rad/s但相对改善不足10%，不满足冻结A/B路线价值门槛。不能称完全无数值收益，也不能推广为整个20mm表示或Diffusion无效。

旧update250权重32次DDIM、16同噪声配对均观察条件数值响应；27/32幅值合法不等于参考或闭环合法。P2 teacher、P3训练、新模型采样和P4五组新TEST按P1停止规则均NOT_RUN_PILOT_STOP；不记为0/4。无检索胜出或条件学习优势结论。保留B.2基础参考与非学习基线，不扩大障碍搜索、幅值、网络、种子或预算。

当前已建立非学习可执行性见证，限制是冻结三候选的路线质量差额不足；后续若另行授权，应先研究任务区分度及安全执行后有效残差作用。原因未由本轮唯一分离。I_route复算保存范数RMS；原名义向量经过速度界裁剪，两个源向量未保存，范围已披露。

成本分开记actual81000、私有预演81000、独立保存力矩重放81000步，执行证据geometry201187130、路线质量5022372、输入预检41846、旧状态绑定预检186；旧权重DDIM32，新训练/新TEST为0。20ms墙钟非研究门禁；仿真20ms/2ms/27s及原安全检查保持，deployment NOT_MET。

路由STOP_CURRENT_FINITE_STUDY_WITH_NEGATIVE_ROUTE_VALUE_RESULT，next_experiments_needed=[]；科学研究结束，发布收尾由root完成。完整审阅paper/review-traces/experiment-result-to-claim/2026-10-07_run02；无paper claim audit，后续论文verdict provisional。
