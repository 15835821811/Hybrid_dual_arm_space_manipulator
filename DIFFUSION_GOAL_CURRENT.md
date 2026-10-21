# 当前执行目标：07固定五障碍路径的有界表示保真验证

长期目标仍为静态避障抓捕完整闭环，再接动态不规则航点。旧目标已归档ff_prior_rep_fidelity/evidence/previous_goal.md。
当前用户授权以已执行合格07为基准，不要求先证明Diffusion优势才运行合格候选的全流程。

冻结协议：ff_prior_rep_fidelity/docs/EXPERIMENT_PLAN.md；独立run artifacts/v7_1/rep_fidelity_001。
先比较102/170/238/323维固定时间表示与raw、legacy_prepare、preserve_prepare的12个逻辑候选，检查原世界终点、避障及运动门；合格唯一reference继续原生67力矩执行与独立重放。323维精确身份为正对照，不算压缩或学习收益。

新增基只用旧TRAIN基及固定数学规则，不拟合DEV；本轮目标/障碍/完整初态均冻结，无训练数据或权重写入。全部投影/prepare中间控制、双重放轨迹及中点几何保存。准确复用旧模型验证与本轮重复项单列，实际执行另计。

M0软件和输入预检已PASS（10测试/127源码）；M1/M2后台已启动，M3审计报告待完成。仅不穿透、5mm/1度、2mm密度、完整初态/自由漂浮/17→67/全机几何/运动/力矩/重放所有门不变。
预算2数学库线程/2小时/8GiB输出/300耦合/300万积分/最多12执行。旧结果和用户修改保留。不训练、不扩库、不pilot、不TEST；不因效果不足追加候选，不放宽门。

结束后新独立审查、一次完成resume hash核验、中文报告与提交，并提出Diffusion输出及TRAIN训练的下一阶段建议，不自动启动下一轮。只可完成本轮有界目标，不能宣称长期科学优势完成。
终端 .\ff_prior_rep_fidelity\job.ps1 status|report|resume。

源码ec59aa1f；后台worker86908/launcher74700已确认存活、心跳正常；接续automation ACTIVE，完成后独立审查并报告。
