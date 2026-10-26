# 双臂轨迹生成实验台

本轮采用论文的流程：完整任务轨迹数据 → B 样条条件扩散训练 → 生成时约束引导 → 候选选择 → MuJoCo 力矩执行。条件分为训练时的起点/参考信息，以及去噪时的跟踪、关节和碰撞梯度；不是生成后才简单筛选。

## 第二轮：500 任务流程

当前执行协议见 `paper/EXPERIMENT_PLAN.md`，实际状态见 `paper/EXPERIMENT_TRACKER.md`。从项目根目录启动或恢复已登记流程：

```powershell
.\tracking_diffusion\execute_round2.ps1
.\tracking_diffusion\start_live.ps1 -Device cuda
```

入口先复用已有生产结果和仍在运行的阶段，依次推进教师数据、150任务中期训练、500任务T1/T2训练、验证评分、必要的T3、补充训练种子、60/40/30步选型、50×5完整验证、配置冻结、20×5新测试及教师/旧模型对照。不会逐项等待确认；运行异常和指标失败都保留。执行期间不要修改实验 Python 源码，评估快照会拒绝混合不同版本。

第二轮输出在 `tracking_output/paper500_v2/`：`teachers/summary.json` 是实际生产进度，各 `*_stage.json` 是当前阶段，`*.stderr.log` 是进程错误。最终配置冻结后生成 `deployment.json`；界面空闲时自动切换原子发布的模型。`RESULTS_ZH.md`、`results.json`、`all_test_runs.csv` 和 `failures.csv` 由全部新测试结束后生成，文件尚未生成时不能当作结果已完成。

界面“关闭梯度”仍保留最后的关节投影。候选阶段可查看开始引导时的干净估计、最后梯度引导后、投影后，粗线标识最终选择；开始引导时的估计不是独立运行的无引导结果。轻量模式每5步保存一个真实去噪状态，详细模式保存每一步；两者使用同一实际采样算法。误差仍保存50 Hz完整记录，实际关节/力矩与碰撞检查500 Hz。

下文保留首轮命令和结果用于复现；首轮路径不会自动生成500条数据。

从 Hybrid_dual_arm_space_manipulator 项目根目录运行：

```powershell
.\tracking_diffusion\start_live.ps1 -Device cuda
```

打开 http://127.0.0.1:8765，默认选择“条件扩散 + 生成时引导”，点击“生成并执行”。任务种子与模型采样种子分别设置；20261012 默认对应标称任务。可关闭生成时引导做对照，也可选择教师和仅反馈。27 秒为完整任务，3 秒仅用于接口试跑。

下方“从噪声到轨迹”显示五条候选在 60 步 DDIM 中的真实中间结果，后 18 步标出约束梯度注入。候选窗口通常为 0.5 秒，末尾缩短；可切换 17 个规划坐标、拖动滑块或播放去噪。展示使用真实 MuJoCo 骨架与实际末端轨迹，运动通过 67 路力矩推进。

启动脚本使用已验证的 E:\v7work_20261010_01\.venv-v7\Scripts\python.exe；可用 -Python 指定其它现有环境，-Port 改端口，-NoBrowser 只开服务。服务只监听本机，不依赖 CDN。不要关闭其它 V7 或用户进程；本服务 PID 在 tracking_output/live/server.pid。

## 可复现的当前命令

输出目录使用新名称；旧结果不覆盖。任务划分先于切片。以下命令适用于本轮的新模型；pipeline collect/train/demo 保留给旧原型，不是当前默认流程。

```powershell
$py = 'E:\v7work_20261010_01\.venv-v7\Scripts\python.exe'
# 冻结100任务并生成教师数据（已有结果可复用，代码版本不同时需新实验名）
& $py -m tracking_diffusion.paper_batch run --folder tracking_output/paper100_v1 --name teachers --mode teacher --workers 4
# 仅从合格训练/验证任务构建完整窗口，测试标签不读取
& $py -m tracking_diffusion.paper_batch build --folder tracking_output/paper100_v1 --name teachers --output tracking_output/paper100_v1/data
# 从头训练条件Transformer
& $py -m tracking_diffusion.paper_model --dataset tracking_output/paper100_v1/data/dataset.npz --output tracking_output/paper100_v1/train_a01 --steps 18000 --device cuda
# 固定10个测试任务，每个3个采样种子
& $py -m tracking_diffusion.paper_batch run --folder tracking_output/paper100_v1 --name test_a01 --mode diffusion --checkpoint tracking_output/paper100_v1/train_a01/model.pt --splits test --workers 4
# 同任务仅反馈对照
& $py -m tracking_diffusion.paper_batch run --folder tracking_output/paper100_v1 --name feedback_a01 --mode feedback-only --splits test --workers 4
```

单任务直接运行，示例使用首轮先导权重：

```powershell
& $py -m tracking_diffusion.evaluation --paper --checkpoint tracking_output/local10mm_v1/paper_train_a01/model.pt --device cuda --sampling-steps 60 --candidates 5 --sampling-seeds 73 --episodes pilot_00 --output tracking_output/my_guided_run
```

加 --no-guidance 可关闭梯度和最终关节投影。该对照同时改变两项，不能仅凭它证明梯度的独立贡献。扩散方法还需与使用解析参考速度的教师控制器比较；优于仅误差反馈不等于优于所有无学习控制器。

## 文件和结果

- 配置与阶段安排：EXECUTION_GOAL_ZH.md、EXECUTION_TRACKER_ZH.md。
- 20 任务先导实验：tracking_output/local10mm_v1。
- 100 任务扩展：tracking_output/paper100_v1。
- 每条执行：result.json 保存完整任务设置、误差及约束诊断；trace.npz 保存实际状态和条件。
- 实时运行：tracking_output/live，额外保存 events.jsonl，包括所有去噪中间状态。长任务文件可能达到数百 MB。
- 历史回放为控制传输量，较早窗口只展示真实首末状态，最后窗口保留全部 61 个状态。没有补造动画。

当前短时引导使用自由基座广义雅可比和碰撞距离的局部线性近似；最终关节边界投影单独列明。原文的绳长驱动模型没有直接套用到本项目的力矩执行模型。实时显示不代表已经达到硬实时计算期限。

数据检查可运行：

```powershell
& $py -m tracking_diffusion.verify_paper_workflow
```

检查包括任务分离、完整窗口覆盖、梯度数值核对、61 个实际采样状态、18 个引导步骤和“生成时不改写实际机器人状态”。

