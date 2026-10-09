# C.3 封存后的初值身份维护

正式实验 producer 是 `9f39b42775283432eb933f63a9047c488ba22070`。发布代码在全部实验、原独立验收、报告及便携源码封存后，仅修复 [独立实现审查 A1](C3_INDEPENDENT_IMPLEMENTATION_AUDIT.md) 记录的空身份匹配边界。

`summarize_teacher_pair` 仅在 proposal 的 plan SHA 或 content key 非空时尝试对应匹配，避免两个缺失值相等而错误绑定到先前 raw 拒绝的候选。原 raw 合法性、B30、直接种子、选中谱系、共同规则和 NO_PLAN 资格规则不变。

新增六项当前入口案例覆盖 content-key-only、plan-SHA-only、缺少全部身份、rule-only、NO_PLAN 和 raw 拒绝；连同原教师测试共 19 项通过。本轮 12 个冻结教师组合的保存输入在修复前后纯函数重算摘要完全一致，且与原封存摘要一致。没有新增教师搜索、actual、训练更新或模型样本，原报告/权重/数据均未改写。

证据：[19 项测试回执](audit_receipts/c3_delivery/seed_binding_maintenance_01/receipt.json)、[12 个组合摘要一致性](audit_receipts/c3_delivery/teacher_maintenance_parity.json)、[430 个 producer 文件与当前代码差异](audit_receipts/c3_delivery/publication_source_difference.json)、[冻结 producer 迁移加载](audit_receipts/c3_delivery/frozen_producer_load_01/load_receipt.json)。

原 producer 字节保留于发布包 `frozen_source/`，维护后的源码不能冒充它。现有 `PortableResolver.verify_sources()` 会正确拒绝使用维护后的 teacher 文件加载旧运行。便携字节验证仍可直接运行；需要验证旧权重加载时，用独立临时目录中的 frozen producer：

```powershell
python -B -X utf8 docs/audit_scripts/c3_frozen_loader_smoke.py --release v6_4/releases/search_aware_warmstart_20261008_01 --stage C:/temp/c3_frozen_load_unique
```

stage 必须不存在；脚本只复制封存源码和便携解析器，加载已有数据、scaler 与 D/S4000 权重，明确禁止模型 forward 和回读原仓库。它不运行新的实验。依赖仍使用本项目记录的 Python/NumPy/PyTorch/MuJoCo 环境。
