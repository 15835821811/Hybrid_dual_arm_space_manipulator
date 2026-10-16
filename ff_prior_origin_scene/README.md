# 在实际任务完整初态审计新增场景

`ff_prior_origin_scene.runtime.at_scene_origin`和`TaskOriginSceneCache`为后续版本提供场景运行时。当前冻结的`independent_new_validation_001`不导入本模块，源码、协议、任务和结果保持原样。

实际诊断复现了该作业slot10的两次生成错误：新增障碍在未使用的默认home构型下分别穿透约12.120mm、34.321mm，初始化forward产生不同warmstart，旧home状态转移检查因此拒绝。在实际任务初态下，两例完整integration state逐字段完全一致，187项机器人/几何转移检查通过，初始全机最小间距约8.519mm。证据见`../ff_prior_pilot_tools/evidence/scene_state_diagnostic_002/`。

本模块先在父模型和新场景模型中构造**同一个实际任务初态**，再调用原`transfer_audit`。没有删除warmstart、映射、机器人参数、静态世界几何或其他完整状态检查；另外在实际初态执行全机仅不穿透检查。若真实任务采用发生接触的home构型，仍须拒绝。所有初态均为已有合同支持的静止零动量状态。

这只证明初始化兼容性，不证明轨迹MODEL_VALID或力矩执行通过，不将此前失败重新准入，也不能解释Diffusion的目标残差或证明收益。

```python
from ff_prior_origin_scene import at_scene_origin, TaskOriginSceneCache
robot, initial, collision, proof = at_scene_origin(canonical, canonical_initial, scene, q0)
cache = TaskOriginSceneCache(canonical, canonical_initial, acceptance_profile)
robot, initial, collision, proof = cache.get(frozen_task_row)
```

调用者仍须独立完成全轨迹几何、自由漂浮传播、世界目标、运动和执行器准入，并按预选协议执行真实67力矩及独立重放。模型、完整初态和目标的身份必须与相应任务完全匹配。

软件验证命令：

```powershell
.\.venv\Scripts\python.exe -m pytest ff_prior_origin_scene/tests -q
```

测试包括两例保留错误场景在实际初态的精确状态/187项转移检查、同两场景以碰撞home作为真实初态仍拒绝，以及全部12个新冻结任务的完整状态恢复和合同篡改拒绝。所有物理状态来自实际MuJoCo模型，没有mock物理结果。它们是软件和初始化验证，不计入正式新验证对照。

本轮实际5/5测试通过，61.876秒，见`evidence/software_tests.json`及`tests.xml`。已核对正在运行作业的全部149个源码哈希未变；本模块和独立诊断工具均不属于其冻结源码。本修复供后续明确冻结的新版本使用，未追溯改变本轮两次失败。
