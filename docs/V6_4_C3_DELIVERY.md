# C.3 最终交付记录

研究、实现、最终审计、当前可视化与独立分支上传均已完成。Diffusion 总体学习收益未建立，默认保留 C.1；没有根据 TEST 重训、重选 checkpoint 或扩充实验预算。

- [GitHub 独立分支](https://github.com/15835821811/Hybrid_dual_arm_space_manipulator/tree/v6.4-c3-search-aware-closed-loop-val)
- [冻结研究报告](../v6_4/releases/search_aware_warmstart_20261008_01/snapshot/REPORT.md)
- [106 项任务义务与最终状态核对](C3_FINAL_COMPLETION_AUDIT.md)
- [当前查看器](../v6_4/visualization/search_aware_warmstart_20261008_01/index.html)及[播放、图表与来源说明](V6_4_C3_VISUALIZATION.md)
- [远端完整文件树及原始字节核验](audit_receipts/c3_github_publication_02.json)

实验 producer：`9f39b42775283432eb933f63a9047c488ba22070`。完整科学产物与可视化的交付提交：`7cd661ace9e8eb7aa9058b1fc09467acdd6a522e`。本记录及最终核验清单由后续文档提交追加，不冒充早期实验 producer；该提交后再核对分支 HEAD。

2026-10-09 的远端核验对上述产物提交的全部 **15,777** 个 Git 文件对象逐项比较 SHA 与 mode，包含 release **4,717** 文件和当前媒体 **399** 文件，全部一致。另从 GitHub 下载 README、报告、两份 manifest、D/S4000 权重、一份 38MB actual NPZ、查看器 HTML、连续体侧视 MP4 与原生 tracking CSV，共十份文件，原始字节均与提交一致。GitHub HTML 源码页不直接运行播放器；clone 后可按可视化说明启动本地服务。

五次顺序普通 push 均成功，保持独立分支，没有合并 main、强推或改写历史。[提交/推送原回执](audit_receipts/c3_publication_commands/copy_manifest.json)保留完整命令和退出码。首次补充二进制下载检查遇到本机 `gh` 输出转换错误；那次已经通过完整文件树核验，失败属于读取工具，未改动远端字节。改为 GitHub JSON/base64 或 API 返回的原始下载 URL 后十份文件全部核验成功；[失败与成功的工具回执](audit_receipts/c3_github_verification_commands/copy_manifest.json)均保留。

release manifest SHA-256：`64c55f17530ff359cff1754a69f693eb87b968c9299411a9e902561e7d79fd9a`。媒体 manifest SHA-256：`10f9174dc8419ea8561e547e67bb02de45dd1120598730ae076de5ff5bf2bbae`。权重、统一 TRAIN 池和全部 VAL/TEST 必要实际数组随分支保存；大型私有预测/计时/几何档案仍保留本地，遗漏清单逐文件声明 SHA、大小和可用性。

保持 DATA_LIMITED、deployment NOT_MET、continuous_time_safety/hardware_safety NOT_ESTABLISHED。媒体仅显示保存状态，不是额外物理验收。
