# 参考源码与本项目任务逻辑对齐

2026-10-20只读核对上游HEAD为`900d781ba63f0123dff95e49a76d51cb49a74803`，与已保存源码相同。本分析依据源码/配置，不依据项目内其他对话。没有从源码中找出已发布论文的完整独立实验结果，不能把配置当作论文实测。

来源：[参考仓库](https://github.com/zhanglooong/htdcr-actuator-aware-diffusion-planning)。

|环节|参考源码|本轮适配|
|---|---|---|
|生成数据|generate_training_data.py：随机合法起终关节构型，线性＋六阶多项式多模态，三次B样条20控制点；入口默认100000对、每对5模式|复用已有14222 TRAIN；保持自由漂浮路径传播，不能拿固定基座FK直接当世界目标|
|数据修正|refine_training_data.py：中间控制点可行性投影，边界固定；obstacle_environment_path=None|本轮不扩库；使用已有合格教师，完整采样输出训练对齐。世界终点仍需耦合传播验收|
|条件扩散|proposed.yaml：DDIM60步、5候选、最后18步引导；控制点输出|本项目16步、5候选、late4，保持原对照预算；新增训练梯度穿过16步|
|引导|full.yaml：碰撞、关节限位、绳长行程/速度/加速度、平滑等多空间代价|本项目世界目标保持碰撞修正、运动/工作域与全机几何、力矩执行；不把绳索限值机械替换成刚性臂限值|
|主实验|main_comparison.yaml：固定三球/五球场景，多个基线，共同strict verifier|原DEV保持可比；新增三/五障碍抓捕接近，plain/aligned/强样条/条件均值同任务|
|消融|run_ablation.py和ablation.yaml|同抓捕场景aligned关闭物理引导，五候选同噪声|
|敏感性|run_sensitivity.py|先有界比较嵌套3/5障碍，其他超参数不扫描|
|可扩展性|section_scalability：4/5/6节连续体|本机固定双臂/17→67，不改机器人冒充可扩展性实验；动态航点在后续独立任务验证|
|验收语义|参考主配置clearance_mm=20，manifest末端关节容差1e-4rad，并有绳索运动硬门|本机继承仅不穿透、世界5mm/1°、全机几何、完整初态、自由漂浮、真实力矩仿真/重放；数字不可横向等价比较|

源码定位：

- [数据生成](https://github.com/zhanglooong/htdcr-actuator-aware-diffusion-planning/blob/900d781ba63f0123dff95e49a76d51cb49a74803/scripts/training/generate_training_data.py)
- [数据修正](https://github.com/zhanglooong/htdcr-actuator-aware-diffusion-planning/blob/900d781ba63f0123dff95e49a76d51cb49a74803/scripts/training/refine_training_data.py)
- [主实验配置](https://github.com/zhanglooong/htdcr-actuator-aware-diffusion-planning/blob/900d781ba63f0123dff95e49a76d51cb49a74803/configs/experiments/main_comparison.yaml)
- [采样配置](https://github.com/zhanglooong/htdcr-actuator-aware-diffusion-planning/blob/900d781ba63f0123dff95e49a76d51cb49a74803/configs/method/proposed.yaml)
- [多空间引导](https://github.com/zhanglooong/htdcr-actuator-aware-diffusion-planning/blob/900d781ba63f0123dff95e49a76d51cb49a74803/configs/guidance/full.yaml)

刚性臂“避障到达抓捕目标”适合上述固定任务＋障碍＋多候选引导＋严格验收逻辑。但本轮只是静止卫星非接触预抓捕，不能称已解决接触捕获。不规则航点规定整段时间上的参考，模型必须接收时序条件和相应表示；静态终点网络没有这些输入，不能借共同控制器的跟踪成功来冒充网络学会动态航点。

参考项目的主干、候选数、代价权重或毫米阈值不是必须照抄的任务合同。应复现共同任务、可比预算、明确失败分母和独立验收的实验原则；物理机器人与任务改变时必须重新建立相应证据。
