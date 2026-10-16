# 实际任务初态的场景积分状态错误诊断

状态DIAGNOSED；6例；0耦合/物理步/网络。

- origin_new_val_007_a0：不同字段['mjSTATE_WARMSTART']，初始最小间隙-30.972346mm；原ERROR保留，未准入或升级。
- origin_train_011_a0：不同字段['mjSTATE_WARMSTART']，初始最小间隙-13.233247mm；原ERROR保留，未准入或升级。
- origin_train_011_a1：不同字段['mjSTATE_WARMSTART']，初始最小间隙-34.290734mm；原ERROR保留，未准入或升级。
- origin_train_043_a0：不同字段['mjSTATE_WARMSTART']，初始最小间隙-38.242418mm；原ERROR保留，未准入或升级。
- origin_train_047_a0：不同字段['mjSTATE_WARMSTART']，初始最小间隙-21.760433mm；原ERROR保留，未准入或升级。
- origin_train_067_a0：不同字段['mjSTATE_WARMSTART']，初始最小间隙-50.402519mm；原ERROR保留，未准入或升级。

若DIAGNOSED，当前障碍在实际初态穿透，forward产生不同warmstart，转移检查先于几何拒绝触发。后续版本可先检查实际初态几何，使预期物理拒绝有明确原因；仍须保持完整状态和全部转移检查。
本次不修订正在运行的源码、冻结门槛或原始错误；若原数据门要求无ERROR，最终仍按原协议判定。
