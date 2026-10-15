# 针对世界目标精度的文献核查与工程选择

本次为针对性原始文献检索，不是系统综述；检索问题为“扩散先验产生可行轨迹但达不到指定目标时，怎样与约束求解结合”。不将其他机器人论文的结果视为本项目实测结果。

1. Carvalho等，Motion Planning Diffusion: Learning and Adapting Robot Motion Planning with Diffusion Models，arXiv:2412.19948v2，2025版本。
   https://arxiv.org/html/2412.19948v2
   III-E/III-G明确支持世界末端位姿条件、自由终端关节构型和目标代价；IV-D中单先验仍有厘米级误差，代价优化改善目标精度。其success率首先按碰撞和关节限位定义，末端误差另外报告，不能直接等价为本项目5mm/1度成功率。采用：样条先验与任务误差修正结合。该文不提供双臂自由漂浮执行验证。
2. Kurtz与Burdick，Equality Constrained Diffusion for Direct Trajectory Optimization，arXiv:2410.01939，2024。
   https://arxiv.org/html/2410.01939v1
   支持非线性等式约束的diffusion优化；文中也指出扩散优化相对于利用二阶信息的NLP求解器可能不高效。采用其“显式处理约束”的动机，本实现仍使用前向耦合传播的局部阻尼修正，不冒充复现其direct transcription算法或理论保证。
3. Constrained Synthesis with Projected Diffusion Models，arXiv:2402.03559 / NeurIPS 2024。
   https://arxiv.org/abs/2402.03559
   将采样与约束投影组合，涵盖非凸约束。采用投影机制设计思想；本项目有限次局部求解并非精确全局投影，不继承其条件下的保证。
4. DDAT: Diffusion Policies Enforcing Dynamically Admissible Robot Trajectories，arXiv:2502.15043，2025。
   https://arxiv.org/html/2502.15043v1
   V-B讨论高噪声投影可能无意义/不稳定，使用噪声相关投影调度；参考投影也强调减少偏离原提案。采用：仅在低噪声clean估计上修正，保持与DDIM当前状态分解一致。其动力学可达投影与本项目世界目标投影不同。
5. Wilde等，Equations of Motion of Free-Floating Spacecraft-Manipulator Systems: An Engineer's Tutorial，2018。
   https://www.frontiersin.org/journals/robotics-and-ai/articles/10.3389/frobt.2018.00041/full
   保持动量耦合、完整初态与自由基座传播，不能用固定基座末端FK替代目标误差。

工程决策：先冻结已训练模型，替换已知覆盖不足的八随机方向引导，使用17方向完整耦合目标雅可比和有界阻尼求解；加入保持关节端点的控制点曲率正则，随后重新修正世界目标。所有正式验收独立于投影代价。
反证安排：普通样条使用完全相同的修正器；Diffusion仅平滑作为消融。如果普通样条同样成功且更便宜，则只能证明约束修正有效，不能声称Diffusion有收益。
