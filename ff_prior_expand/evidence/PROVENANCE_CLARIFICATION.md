# 本轮来源标记澄清

独立复核发现：继承索引中的 12 条旧 complex_train_val_001 记录（6 TRAIN、6 VAL）带有 new_in_this_run=true，该字段相对于其原生成批次，并不表示 qualified_expansion_002 的新增样本。本批 expand_train_* 条目没有该字段。

本轮新增数必须依据源目录、生成账本与唯一 sample_id 共同确定：744 条新增 TRAIN；继承 256 TRAIN、48 VAL；累计 1000 TRAIN、48 VAL。实际训练消费全部新增 744 条。保留原索引与运行输入字节不变，不能直接累加历史 new_in_this_run 字段来计算本轮新增数量。后续新索引应使用明确的 producer_run_id 和 source_run_id，不复用含糊的相对标记。

本说明不改变任何样本的物理验证等级，不代表未完成的 DEV 或边界审计已经通过。
