# V6.4-C.1 冻结执行协议

从 `cd288d7c2db74dc938903332400c7e97dd56bca8` 开发。实现四个主要模块：协议、直接搜索、私有评价、最终执行与汇总。原控制器、安全配置、17D QP、67路伺服、20ms/2ms仿真、27s终态与23.98s保护截止均保持源码不变。

最多开放slot2及其最近合法前驱，每段独立20mm圆盘。四初值：v1零、v1远离12mm、v2远离12mm、v2远离20mm。后续至多8槽按A/B交替中心，共享顺序 `+prev_e1,-prev_e1,+prev_e2,-prev_e2,+key_e1,-key_e1,+key_e2,-key_e2,family切换`；2D任务删除不存在坐标。每个提案从当前已评估中心出发，逐段圆盘投影，精确内容去重。完整无改善轮询后才降低5/2.5/1.25mm半径；预算耗尽前未完成整轮时明确truncated。最多128次生成，失败槽不补额。

建议种子2026100701已出现在B.2发布的任务集中。冻结前依据旧B.2全部12个种子及B.3/B.3.1种子执行next-unused：在 `2026100701+104729*source_index` 序列取前两个未使用值，不读任何新结果。预计source_index=12/13，种子2027357449/2027462178。每母球心左右55mm、半径25mm，其他字段逐值相同；原初态/锚点预检失败不换任务。

A以固定前驱/关键区间并集I_support为新目标；strict最低与锚定+0.001rad/s并列后按路径/范数/family/ID选择分开。旧I_key与I_full另报。B严格要求同一并集相关原连续体—路线球碰撞对的原生离散净空≥30mm，再最小完整末端路径。29.999mm不通过；无解输出null，只保留诊断。30mm不改变安全配置。

私有候选调用原run_synchronous_scenario；每次新建MjData、控制器、provider、积分状态与QP/ADMM缓存。保留全部在线守卫和原runner自带检查，名义预测不再额外独立力矩重放。每个状态都mj_forward后查询冻结原生related pairs，保存window minimum的pair/fromto/time/witness及截断下界状态；完整路径使用fresh 2ms状态。保存裁剪名义与实际选中17D源向量、10D/7D分量并校验RMS恒等式。搜索独立认证明确NOT_RUN_SEARCH_SCREENING。

先封存全部四任务selection再启动最终actual，首槽为首任务Z0。不读取actual重新选点。同plan/config/identity可别名共享唯一执行，四方法各四逻辑分母。正式actual调用原execute_residual_attempt默认完整独立力矩重放、区间重算、500Hz robot-target与50Hz/subdivisions4 whole-body及参考绑定。validate命令核对已产生的原验收原件，不另加重放。

两偏好共享每任务12槽，总48；最终逻辑槽16。十步私有预演、独立replay、原生几何、QP与墙钟分别计账。缓存不积分，失败prefix不进入完整质量排序，工具错误与研究负结果区分。阶段完成仅验哈希后跳过；未终态已消费槽阻止自动重试，保留原件并要求显式恢复记录。

正式运行前提交代码、冻结producer与所有源码/模型资产/配置/任务哈希。新训练、神经采样、学习optimizer updates均为0；部署NOT_MET。负结果允许完成研究，不代表全局最优或物理不可行，不建立Diffusion、硬件或连续时间安全收益。
