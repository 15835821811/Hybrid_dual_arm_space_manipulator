# FFGD工程复现入口

当前状态：工程未完成。下面是现有真实CLI与修订任务的执行顺序，不是已经通过的最终链报告。MASTER首节的用户修订优先；基线比较与学习优势不阻塞工程复现。

## 环境与输入

已验证环境为Windows、Python3.10.19、MuJoCo3.3.2、PyTorch2.6.0。精确依赖见仓库根`requirements-ffgd.lock`。工作目录为本仓库根，现有GPU解释器为`.venv-gpu/Scripts/python.exe`。独立目录安装示例：

```powershell
uv venv --python 3.10.19 .venv
uv pip install --python .venv/Scripts/python.exe -r requirements-ffgd.lock
```

需要保留`ffgd/`、`model_test/robot_model_spec_v5.py`、`model_test/angle_convention.py`、真实`dual_arm_space_robot_2026/`模型及全部网格、配置、MASTER和依赖锁。不得用简化替代模型冒充复现。模型目录的实际相对路径由`robot_model_spec_v5.py`指定。

当前有效free357权重：`artifacts/ffgd/runs/20261028T051522_training_5113b0/best.pt`，同目录`best.identity.json`必需；SHA256为`b0adbbacb143b3e30f707dd63e153dafa42b0ab0af1754ad10d89e537e281350`。该模型来自真实2512条物理先验数据/3495次尝试，3000次训练更新。模型与scaler身份由加载器核验。数据满足需求时复用，不要求重新训练来证明流程完整。

迁移到另一目录时，只在**新的复现配置副本**内将checkpoint路径改为实际文件路径，记录副本和hash；不得修改旧run的配置或报告。现有规划配置含工作区绝对checkpoint路径，尚不能宣称开箱即用的最终便携交付。

全新交付目录可以运行下列初始化命令；它不继承旧成功状态，并拒绝覆盖已有state。已有工程应使用`resume`，不要重新`init`。

```powershell
.venv/Scripts/python.exe -m ffgd.cli init
.venv/Scripts/python.exe -m ffgd.cli doctor
.venv/Scripts/python.exe -m ffgd.cli resume
```

`doctor`本身不等于全部里程碑通过。`resume`核验MASTER、checkpoint与真实PID，返回接续动作，不会自动启动编码代理。现有独立环境证据`artifacts/ffgd/reproduction/clean_20261028T064122/smoke_result.json`仅证明依赖/模型/权重/新噪声推理smoke，不证明引导、执行或最终M8通过。

## 修订最终任务的实际命令

首轮开发时序为8秒接近、24秒七航点与刚性持续卫星追踪；空间航点、七个球障碍及物理门不变。根初态全零速度；卫星在第二段接受配置中明确的2秒力脉冲再自由漂移，禁止注入目标速度。时序可依据实测另建版本，不能事后改判旧结果。

```powershell
.venv-gpu/Scripts/python.exe -m ffgd.cli plan --config configs/ffgd/plan_final_v2_approach.json --task configs/ffgd/final_v2/approach.json
```

仅当该run的`planning_result.json`为`PLAN_VALID`且存在`selected_index`时，按原选中候选执行。以下变量须填写该次命令实际返回目录，不得指向旧演示：

```powershell
$approachPlan = '<本次planning目录>'
.venv-gpu/Scripts/python.exe -m ffgd.cli execute --run $approachPlan --config configs/ffgd/execution_substeps_v2.json
$approachExecution = '<本次selected_execution目录>'
.venv-gpu/Scripts/python.exe -m ffgd.cli replay --run $approachExecution
.venv-gpu/Scripts/python.exe -m ffgd.cli continue-state --run $approachExecution --task configs/ffgd/final_v2/waypoints_tracking_template.json
```

`execute`再次只读验证原候选；实际物理步长0.5ms、控制周期2ms，安全逐步检查。`continue-state`要求执行通过，导出完整实际积分状态和原参考的局部q/v/a，场景不变；只导出局部边界，不把旧未来路径作为下一规划输入。

第二段只能使用上述`continue-state`返回目录中的`problem.json`，不能直接将模板根初态当作接续。其正式规划预算须根据第一段实测在启动前保存；目前第二段规划配置和实际运行均为`NOT_RUN`。之后沿用同一`plan → execute → replay`入口，保留所有失败。

两段均完成后，在链检查配置中列明本次两个`execution_runs`和相应两个`replay_runs`，保持`require_uniform_planner_version: true`，调用：

```powershell
.venv-gpu/Scripts/python.exe -m ffgd.cli validate-chain --config '<本次链检查配置.json>'
```

链检查覆盖实际状态字节、C2参考边界、绝对时钟、累计动量和重放绑定；最终任务覆盖还须核对本次任务文件，不能用两条无关短轨迹代替。架构审计需绑定这版实际采样及其原生执行，不能继承V16旧后端的架构通过。

## 当前限制与证据边界

- 旧V16预选执行/重放通过仍有效，但不是新free357/完整状态最终链的验收。
- C2短时保持和首次5秒运动目标失败保留；未选中的候选不得执行。
- 当前multirate的旧`mass_evaluations`计数按微步计入，未折算矩阵刷新间隔，会高估刷新次数。原日志不改写；不能用该计数宣称匹配物理调用预算优势。墙钟、网络调用与梯度次数单列。
- 最终两段全链、其必要复验、完整干净环境执行与独立重放仍为`NOT_RUN`。完成后才能据真实证据替换本页的待运行状态。
- Diffusion相对优势、广泛泛化与论文级统计：`NOT_RUN／未评估`。
